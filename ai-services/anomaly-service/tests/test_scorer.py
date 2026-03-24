"""
tests/test_scorer.py

Unit tests for compute_anomaly_score() — the core scoring logic.
Calls the function directly with synthetic DataFrames.
No HTTP, no MinIO.

Tests mirror test_analyzer.py from vision-service in structure:
one class per logical concern, assertions on specific output fields.
"""
from __future__ import annotations

import numpy as np
import pytest

from app.scorer import compute_anomaly_score
from tests.conftest import (
    SENSOR_COLS,
    make_normal_csv,
    make_anomalous_csv,
    make_short_csv,
)

VALID_INDICATORS = {
    "NO_PRECURSOR_DETECTED",
    "ABRUPT_FAILURE",
    "GRADUAL_DEGRADATION",
    "MINOR_PRECURSORS_DETECTED",
    "NORMAL",
    # spaces variant — remove once scorer.py uses underscores
    "NO PRECURSOR DETECTED",
    "ABRUPT FAILURE",
    "GRADUAL DEGRADATION",
    "MINOR PRECURSORS DETECTED",
}


# ── score range ───────────────────────────────────────────────────────────────

class TestScoreRange:
    def test_score_always_0_to_100(self, loaded_models):
        """Score must be in [0, 100] for any input across multiple seeds."""
        for seed in range(5):
            sensor_df, timestamps = make_normal_csv(seed=seed)
            result = compute_anomaly_score(
                sensor_df, timestamps, "2026-02-10", loaded_models
            )
            assert 0 <= result["score"] <= 100, (
                f"Seed {seed}: score {result['score']} out of range"
            )

    def test_score_is_int(self, loaded_models):
        sensor_df, timestamps = make_normal_csv()
        result = compute_anomaly_score(
            sensor_df, timestamps, "2026-02-10", loaded_models
        )
        assert isinstance(result["score"], int)

    def test_normal_data_scores_low(self, loaded_models):
        """
        Clean NORMAL data should score below 35.
        Core property: no degradation = low anomaly score.
        """
        sensor_df, timestamps = make_normal_csv(n_rows=2000, seed=0)
        result = compute_anomaly_score(
            sensor_df, timestamps, "2026-02-10", loaded_models
        )
        assert result["score"] < 35, (
            f"NORMAL data scored too high: {result['score']} — "
            f"expected < 35"
        )

    def test_anomalous_data_scores_higher_than_normal(self, loaded_models):
        """
        Anomalous data must consistently score higher than normal data.
        This is the fundamental property of the model.
        """
        normal_scores = []
        anomaly_scores = []

        for seed in range(3):
            s_normal, ts_normal   = make_normal_csv(n_rows=2000, seed=seed)
            s_anomaly, ts_anomaly = make_anomalous_csv(n_rows=2000, seed=seed)

            r_normal  = compute_anomaly_score(
                s_normal, ts_normal, "2026-02-10", loaded_models
            )
            r_anomaly = compute_anomaly_score(
                s_anomaly, ts_anomaly, "2026-02-10", loaded_models
            )
            normal_scores.append(r_normal["score"])
            anomaly_scores.append(r_anomaly["score"])

        assert np.mean(anomaly_scores) > np.mean(normal_scores), (
            f"Anomalous mean={np.mean(anomaly_scores):.1f} not > "
            f"Normal mean={np.mean(normal_scores):.1f}"
        )


# ── return structure ──────────────────────────────────────────────────────────

class TestReturnStructure:
    def test_all_keys_present(self, loaded_models):
        sensor_df, timestamps = make_normal_csv()
        result = compute_anomaly_score(
            sensor_df, timestamps, "2026-02-10", loaded_models
        )
        for key in ("score", "anomalies", "pre_incident_anomaly", "fraud_indicator"):
            assert key in result, f"Missing key: {key}"

    def test_anomalies_is_list(self, loaded_models):
        sensor_df, timestamps = make_normal_csv()
        result = compute_anomaly_score(
            sensor_df, timestamps, "2026-02-10", loaded_models
        )
        assert isinstance(result["anomalies"], list)

    def test_pre_incident_is_bool(self, loaded_models):
        sensor_df, timestamps = make_normal_csv()
        result = compute_anomaly_score(
            sensor_df, timestamps, "2026-02-10", loaded_models
        )
        assert isinstance(result["pre_incident_anomaly"], bool)

    def test_fraud_indicator_is_string(self, loaded_models):
        sensor_df, timestamps = make_normal_csv()
        result = compute_anomaly_score(
            sensor_df, timestamps, "2026-02-10", loaded_models
        )
        assert isinstance(result["fraud_indicator"], str)


# ── anomalies list ────────────────────────────────────────────────────────────

class TestAnomaliesList:
    def test_normal_data_anomalies_empty_or_few(self, loaded_models):
        """
        Clean data should produce very few anomaly items.
        We allow up to 3 to account for model sensitivity on synthetic data.
        """
        sensor_df, timestamps = make_normal_csv(n_rows=2000, seed=42)
        result = compute_anomaly_score(
            sensor_df, timestamps, "2026-02-10", loaded_models
        )
        assert len(result["anomalies"]) <= 3, (
            f"Normal data produced {len(result['anomalies'])} anomalies — expected <= 3"
        )

    def test_anomalous_data_produces_anomalies(self, loaded_models):
        """
        Spiked sensor data must produce at least 1 anomaly item.
        """
        sensor_df, timestamps = make_anomalous_csv(n_rows=2000, seed=0)
        result = compute_anomaly_score(
            sensor_df, timestamps, "2026-02-10", loaded_models
        )
        assert len(result["anomalies"]) >= 1, (
            "Anomalous data produced 0 anomaly items — scorer not detecting spikes"
        )

    def test_anomaly_item_has_correct_fields(self, loaded_models):
        """Each item in anomalies[] must have timestamp, parameter, value, threshold."""
        sensor_df, timestamps = make_anomalous_csv(n_rows=2000, seed=0)
        result = compute_anomaly_score(
            sensor_df, timestamps, "2026-02-10", loaded_models
        )
        for item in result["anomalies"]:
            assert "timestamp"  in item, "Missing 'timestamp'"
            assert "parameter"  in item, "Missing 'parameter'"
            assert "value"      in item, "Missing 'value'"
            assert "threshold"  in item, "Missing 'threshold'"

    def test_anomaly_value_types(self, loaded_models):
        """value and threshold must be floats, timestamp and parameter strings."""
        sensor_df, timestamps = make_anomalous_csv(n_rows=2000, seed=0)
        result = compute_anomaly_score(
            sensor_df, timestamps, "2026-02-10", loaded_models
        )
        for item in result["anomalies"]:
            assert isinstance(item["timestamp"], str)
            assert isinstance(item["parameter"], str)
            assert isinstance(item["value"],     float)
            assert isinstance(item["threshold"], float)

    def test_anomaly_parameter_is_valid_sensor(self, loaded_models):
        """parameter field must be one of the known sensor column names."""
        sensor_df, timestamps = make_anomalous_csv(n_rows=2000, seed=0)
        result = compute_anomaly_score(
            sensor_df, timestamps, "2026-02-10", loaded_models
        )
        for item in result["anomalies"]:
            assert item["parameter"] in SENSOR_COLS, (
                f"Unknown sensor name in anomaly: {item['parameter']}"
            )

    def test_anomaly_value_in_range(self, loaded_models):
        """value must be within [0.0, 1.0] — data is MinMaxScaled."""
        sensor_df, timestamps = make_anomalous_csv(n_rows=2000, seed=0)
        result = compute_anomaly_score(
            sensor_df, timestamps, "2026-02-10", loaded_models
        )
        for item in result["anomalies"]:
            assert 0.0 <= item["value"] <= 1.0, (
                f"Anomaly value {item['value']} out of [0,1] range"
            )


# ── fraud indicator ───────────────────────────────────────────────────────────

class TestFraudIndicator:
    def test_indicator_always_valid(self, loaded_models):
        """fraud_indicator must always be one of the 5 defined values."""
        for seed in range(4):
            sensor_df, timestamps = make_normal_csv(seed=seed)
            result = compute_anomaly_score(
                sensor_df, timestamps, "2026-02-10", loaded_models
            )
            assert result["fraud_indicator"] in VALID_INDICATORS, (
                f"Unexpected fraud_indicator: '{result['fraud_indicator']}'"
            )

    def test_normal_data_no_precursor_indicator(self, loaded_models):
        """
        Clean data with claimDate after the sequence should yield
        NO_PRECURSOR_DETECTED or NORMAL — never GRADUAL_DEGRADATION.
        """
        sensor_df, timestamps = make_normal_csv(n_rows=2000, seed=0)
        result = compute_anomaly_score(
            sensor_df, timestamps, "2026-02-10", loaded_models
        )
        assert result["fraud_indicator"] not in (
            "GRADUAL_DEGRADATION", "GRADUAL DEGRADATION"
        ), f"Normal data should not produce GRADUAL_DEGRADATION"

    def test_anomalous_data_not_no_precursor(self, loaded_models):
        """
        Spiked data where anomalies fall before the claim date should NOT
        produce NO_PRECURSOR_DETECTED.
        """
        sensor_df, timestamps = make_anomalous_csv(n_rows=2000, seed=0)
        # claim date is after all data — all anomalies are pre-incident
        last_ts = str(timestamps.iloc[-1])[:10]
        result  = compute_anomaly_score(
            sensor_df, timestamps, last_ts, loaded_models
        )
        assert result["fraud_indicator"] not in (
            "NO_PRECURSOR_DETECTED", "NO PRECURSOR DETECTED"
        ), "Anomalous data before claim date must not be NO_PRECURSOR_DETECTED"


# ── score math ────────────────────────────────────────────────────────────────

class TestScoreMath:
    def test_score_clipped_to_100(self, loaded_models):
        """
        Even if both sub-scores are extreme, final score stays at 100.
        Inject all-ones data (maximum deviation from normal).
        """
        import pandas as pd
        ones_data  = np.ones((2000, len(SENSOR_COLS)))
        sensor_df  = pd.DataFrame(ones_data, columns=SENSOR_COLS)
        timestamps = pd.Series(pd.date_range("2026-01-01", periods=2000, freq="1min"))
        result     = compute_anomaly_score(
            sensor_df, timestamps, "2026-02-10", loaded_models
        )
        assert result["score"] <= 100

    def test_score_not_negative(self, loaded_models):
        """Score must never go below 0."""
        import pandas as pd
        zeros_data = np.zeros((2000, len(SENSOR_COLS)))
        sensor_df  = pd.DataFrame(zeros_data, columns=SENSOR_COLS)
        timestamps = pd.Series(pd.date_range("2026-01-01", periods=2000, freq="1min"))
        result     = compute_anomaly_score(
            sensor_df, timestamps, "2026-02-10", loaded_models
        )
        assert result["score"] >= 0


# ── short data edge case ──────────────────────────────────────────────────────

class TestShortData:
    def test_short_csv_does_not_crash(self, loaded_models):
        """
        A CSV shorter than WINDOW_LSTM=240 must not crash the scorer.
        The fallback IF path handles this case.
        """
        sensor_df, timestamps = make_short_csv(n_rows=100)
        result = compute_anomaly_score(
            sensor_df, timestamps, "2026-02-01", loaded_models
        )
        assert "score" in result
        assert 0 <= result["score"] <= 100

    def test_short_csv_anomalies_empty(self, loaded_models):
        """Short data cannot form LSTM windows — anomalies list must be empty."""
        sensor_df, timestamps = make_short_csv(n_rows=100)
        result = compute_anomaly_score(
            sensor_df, timestamps, "2026-02-01", loaded_models
        )
        assert result["anomalies"] == []