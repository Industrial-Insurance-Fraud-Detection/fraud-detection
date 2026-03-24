"""
tests/test_pre_incident.py

Dedicated tests for the pre_incident_anomaly flag —
the most important fraud detection signal in the anomaly service.

Core fraud logic:
  pre_incident_anomaly = False → no degradation before claim → FRAUD signal
  pre_incident_anomaly = True  → machine was degrading before claim → REAL failure

These tests mirror the depth that test_ela.py gives to ELA in vision-service.
All tests call compute_anomaly_score() directly — no HTTP needed.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from app.scorer import compute_anomaly_score, WINDOW_LSTM
from tests.conftest import (
    SENSOR_COLS,
    make_normal_csv,
    make_pre_incident_csv,
)


# ── clean data — flag must be False ──────────────────────────────────────────

class TestPreIncidentClean:
    def test_clean_data_flag_false(self, loaded_models):
        """
        Normal data all the way through — no anomalies anywhere —
        flag must be False.
        """
        sensor_df, timestamps = make_normal_csv(n_rows=2000, seed=0)
        result = compute_anomaly_score(
            sensor_df, timestamps, "2026-02-10", loaded_models
        )
        assert result["pre_incident_anomaly"] is False

    def test_false_flag_consistent_across_seeds(self, loaded_models):
        """False flag must be stable across different normal datasets."""
        for seed in range(4):
            sensor_df, timestamps = make_normal_csv(n_rows=2000, seed=seed)
            result = compute_anomaly_score(
                sensor_df, timestamps, "2026-02-10", loaded_models
            )
            assert result["pre_incident_anomaly"] is False, (
                f"Seed {seed}: expected False, got True on clean data"
            )


# ── anomalous data inside 7-day window — flag must be True ───────────────────

class TestPreIncidentAnomalous:
    def test_spikes_in_last_7_days_flag_true(self, loaded_models):
        """
        Spikes injected in the last 7 days before claim_date must
        trigger pre_incident_anomaly=True.
        """
        sensor_df, timestamps, claim_date = make_pre_incident_csv(
            n_rows=3000,
            spike_in_last_days=7,
            seed=0,
        )
        result = compute_anomaly_score(
            sensor_df, timestamps, claim_date, loaded_models
        )
        assert result["pre_incident_anomaly"] is True, (
            "Spikes in last 7 days must produce pre_incident_anomaly=True"
        )

    def test_spikes_in_last_3_days_flag_true(self, loaded_models):
        """Even a 3-day spike window must be detected."""
        sensor_df, timestamps, claim_date = make_pre_incident_csv(
            n_rows=3000,
            spike_in_last_days=3,
            seed=1,
        )
        result = compute_anomaly_score(
            sensor_df, timestamps, claim_date, loaded_models
        )
        assert result["pre_incident_anomaly"] is True


# ── anomalies OUTSIDE the 7-day window — flag must be False ──────────────────

class TestPreIncidentOutsideWindow:
    def test_spikes_30_days_before_flag_false(self, loaded_models):
        """
        Spikes injected 30 days before claim_date fall outside the 7-day window.
        The last 7 days are clean, so flag must be False.
        """
        sensor_df, timestamps, claim_date = make_pre_incident_csv(
            n_rows=3000,
            spike_in_last_days=30,  # spikes at day 30, window checks day 7
            seed=0,
        )
        # Override: inject spikes only between day 30 and day 20, not in last 7
        claim_dt = pd.Timestamp(claim_date)
        spike_start = claim_dt - pd.Timedelta(days=30)
        spike_end   = claim_dt - pd.Timedelta(days=20)
        mask = (timestamps >= spike_start) & (timestamps <= spike_end)

        rng  = np.random.default_rng(99)
        data = sensor_df.values.copy()
        spike_indices = np.where(mask.values)[0]
        for col_idx in range(5):
            data[spike_indices, col_idx] = np.clip(
                data[spike_indices, col_idx] + 0.4, 0.0, 1.0
            )

        sensor_df_modified = pd.DataFrame(data, columns=SENSOR_COLS)
        result = compute_anomaly_score(
            sensor_df_modified, timestamps, claim_date, loaded_models
        )
        # Last 7 days are clean — flag must be False
        assert result["pre_incident_anomaly"] is False, (
            "Spikes outside 7-day window must not trigger pre_incident_anomaly"
        )


# ── edge cases ────────────────────────────────────────────────────────────────

class TestPreIncidentEdgeCases:
    def test_short_pre_window_returns_false(self, loaded_models):
        """
        If fewer than WINDOW_LSTM rows exist before the claim date,
        the scorer cannot form sequences — must return False without crashing.
        """
        # Only 50 rows total, all before claim_date
        n_rows    = 50
        rng       = np.random.default_rng(0)
        data      = rng.uniform(0.3, 0.6, (n_rows, len(SENSOR_COLS)))
        sensor_df = pd.DataFrame(data, columns=SENSOR_COLS)
        timestamps = pd.Series(
            pd.date_range("2026-02-09", periods=n_rows, freq="1min")
        )
        result = compute_anomaly_score(
            sensor_df, timestamps, "2026-02-10", loaded_models
        )
        # Not enough rows for LSTM window — must default to False
        assert result["pre_incident_anomaly"] is False
        assert "score" in result  # must not crash

    def test_invalid_claim_date_does_not_crash(self, loaded_models):
        """
        A malformed claim_date string must not crash the scorer.
        The except block in scorer.py catches this and defaults to False.
        """
        sensor_df, timestamps = make_normal_csv(n_rows=2000)
        result = compute_anomaly_score(
            sensor_df, timestamps, "not-a-date", loaded_models
        )
        assert result["pre_incident_anomaly"] is False
        assert "score" in result

    def test_claim_date_before_all_data_returns_false(self, loaded_models):
        """
        claim_date earlier than all timestamps → empty pre-window slice
        → must return False without crashing.
        """
        sensor_df, timestamps = make_normal_csv(n_rows=2000)
        # All data starts 2026-01-01; claim_date is before that
        result = compute_anomaly_score(
            sensor_df, timestamps, "2025-12-01", loaded_models
        )
        assert result["pre_incident_anomaly"] is False

    def test_claim_date_after_all_data_returns_false(self, loaded_models):
        """
        claim_date far after all data — the 7-day window has no rows.
        Must return False without crashing.
        """
        sensor_df, timestamps = make_normal_csv(n_rows=2000)
        result = compute_anomaly_score(
            sensor_df, timestamps, "2030-01-01", loaded_models
        )
        assert result["pre_incident_anomaly"] is False

    def test_result_always_has_all_keys_on_edge_cases(self, loaded_models):
        """
        Edge case inputs must still return a complete result dict
        with all 4 required keys.
        """
        sensor_df, timestamps = make_normal_csv(n_rows=50)
        result = compute_anomaly_score(
            sensor_df, timestamps, "not-a-date", loaded_models
        )
        for key in ("score", "anomalies", "pre_incident_anomaly", "fraud_indicator"):
            assert key in result, f"Edge case: missing key '{key}'"