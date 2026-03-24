"""
Shared pytest fixtures for anomaly-service tests.

Key design decisions:
  - Tests NEVER hit MinIO or require a real CSV from storage.
  - Models are loaded for real (lightweight: IF=970KB, LSTM=276KB).
  - Sensor data is generated in-memory using numpy + pandas.
  - Normal and anomalous sequences are clearly distinct by design.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from fastapi.testclient import TestClient

# Number of sensor columns the model was trained on
N_SENSORS   = 50
SENSOR_COLS = [f"sensor_{i:02d}" for i in range(N_SENSORS)]

# Realistic sensor value ranges (post MinMaxScaler: 0.0 - 1.0)
NORMAL_LOW  = 0.30
NORMAL_HIGH = 0.60


# ── data generators ───────────────────────────────────────────────────────────

def make_normal_csv(n_rows: int = 2000, seed: int = 0) -> tuple[pd.DataFrame, pd.Series]:
    """
    Generate a DataFrame of stable NORMAL sensor readings.

    - Values stay within [NORMAL_LOW, NORMAL_HIGH] with low variance.
    - No spikes, no drift.
    - Returns (sensor_df, timestamps) matching the scorer's expected inputs.
    """
    rng = np.random.default_rng(seed)
    data = rng.uniform(NORMAL_LOW, NORMAL_HIGH, (n_rows, N_SENSORS))
    # add small gaussian noise to simulate realistic readings
    noise = rng.normal(0, 0.02, (n_rows, N_SENSORS))
    data  = np.clip(data + noise, 0.0, 1.0)

    sensor_df  = pd.DataFrame(data, columns=SENSOR_COLS)
    timestamps = pd.Series(
        pd.date_range("2026-01-01", periods=n_rows, freq="1min")
    )
    return sensor_df, timestamps


def make_anomalous_csv(n_rows: int = 2000, seed: int = 0) -> tuple[pd.DataFrame, pd.Series]:
    """
    Generate a DataFrame that contains clear anomalies in the last 30% of rows.

    - First 70%: stable normal readings.
    - Last 30%: 5 sensors drift toward 0.95+ with large spikes (degradation pattern).
    - This simulates a pump showing signs of failure before breakdown.
    """
    rng       = np.random.default_rng(seed)
    data      = rng.uniform(NORMAL_LOW, NORMAL_HIGH, (n_rows, N_SENSORS))
    split     = int(n_rows * 0.70)

    # inject degradation into sensors 0-4 in the last 30%
    for col_idx in range(5):
        drift  = np.linspace(0, 0.4, n_rows - split)
        spikes = rng.uniform(0, 0.3, n_rows - split)
        data[split:, col_idx] = np.clip(
            data[split:, col_idx] + drift + spikes, 0.0, 1.0
        )

    sensor_df  = pd.DataFrame(data, columns=SENSOR_COLS)
    timestamps = pd.Series(
        pd.date_range("2026-01-01", periods=n_rows, freq="1min")
    )
    return sensor_df, timestamps


def make_short_csv(n_rows: int = 100, seed: int = 0) -> tuple[pd.DataFrame, pd.Series]:
    """
    Generate a CSV with fewer rows than WINDOW_LSTM=240.
    Used to test edge cases where windows cannot be formed.
    """
    rng       = np.random.default_rng(seed)
    data      = rng.uniform(NORMAL_LOW, NORMAL_HIGH, (n_rows, N_SENSORS))
    sensor_df = pd.DataFrame(data, columns=SENSOR_COLS)
    timestamps = pd.Series(
        pd.date_range("2026-02-01", periods=n_rows, freq="1min")
    )
    return sensor_df, timestamps


def make_pre_incident_csv(
    n_rows: int = 2000,
    spike_in_last_days: int = 7,
    seed: int = 0,
) -> tuple[pd.DataFrame, pd.Series, str]:
    """
    Generate a CSV where anomalies appear specifically in the last N days
    before a fixed claim date of 2026-02-10.

    Returns (sensor_df, timestamps, claim_date_str).
    The caller controls whether spikes fall inside or outside the 7-day window
    by passing spike_in_last_days=7 (inside) or spike_in_last_days=30 (outside).
    """
    rng        = np.random.default_rng(seed)
    claim_date = pd.Timestamp("2026-02-10")
    # data covers 40 days ending exactly on claim_date
    start      = claim_date - pd.Timedelta(days=40)
    timestamps = pd.Series(pd.date_range(start, periods=n_rows, freq="1min"))

    data = rng.uniform(NORMAL_LOW, NORMAL_HIGH, (n_rows, N_SENSORS))

    # inject spikes in the window [claim_date - spike_in_last_days, claim_date]
    cutoff = claim_date - pd.Timedelta(days=spike_in_last_days)
    mask   = (timestamps >= cutoff) & (timestamps <= claim_date)
    spike_indices = np.where(mask.values)[0]

    if len(spike_indices) > 0:
        for col_idx in range(5):
            data[spike_indices, col_idx] = np.clip(
                data[spike_indices, col_idx] + rng.uniform(0.35, 0.45, len(spike_indices)),
                0.0, 1.0,
            )

    sensor_df = pd.DataFrame(data, columns=SENSOR_COLS)
    return sensor_df, timestamps, "2026-02-10"


# ── FastAPI test client fixture ───────────────────────────────────────────────

@pytest.fixture(scope="session")
def client():
    """
    Session-scoped TestClient — models load once for the entire test session.
    Uses real models from app/models/ (no mocking needed, they are lightweight).
    """
    from app.main import app
    with TestClient(app) as c:
        yield c


@pytest.fixture(scope="session")
def loaded_models(client):
    """
    Return the models dict from app.state after the client has started.
    Allows scorer tests to call compute_anomaly_score() directly.
    """
    from app.main import app
    return app.state.models