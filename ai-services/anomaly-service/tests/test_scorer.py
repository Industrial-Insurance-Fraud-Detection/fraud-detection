"""
tests/test_scorer.py

Unit tests for compute_anomaly_score() — the core scoring logic.
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
}


# ── score range ───────────────────────────────────────────────────────────────

class TestScoreRange:
    def test_score_always_0_to_100(self, loaded_models):
        for seed in range(5):
            sensor_df, timestamps = make_normal_csv(seed=seed)
            result = compute_anomaly_score(
                sensor_df, timestamps, "2026-02-10", loaded_models
            )
            assert 0 <= result["score"] <= 100

    def test_score_is_int(self, loaded_models):
        sensor_df, timestamps = make_normal_csv()
        result = compute_anomaly_score(
            sensor_df, timestamps, "2026-02-10", loaded_models
        )
        assert isinstance(result["score"], int)

    def test_normal_data_scores_low(self, loaded_models):
        sensor_df, timestamps = make_normal_csv(n_rows=2000, seed=0)
        result = compute_anomaly_score(
            sensor_df, timestamps, "2026-02-10", loaded_models
        )
        assert result["score"] < 35

    def test_anomalous_data_scores_higher_than_normal(self, loaded_models):
        normal_scores, anomaly_scores = [], []
        for seed in range(3):
            s_normal,  ts_normal  = make_normal_csv(n_rows=2000, seed=seed)
            s_anomaly, ts_anomaly = make_anomalous_csv(n_rows=2000, seed=seed)
            normal_scores.append(
                compute_anomaly_score(s_normal, ts_normal, "2026-02-10", loaded_models)["score"]
            )
            anomaly_scores.append(
                compute_anomaly_score(s_anomaly, ts_anomaly, "2026-02-10", loaded_models)["score"]
            )
        assert np.mean(anomaly_scores) > np.mean(normal_scores)


# ── return structure ──────────────────────────────────────────────────────────

class TestReturnStructure:
    def test_all_keys_present(self, loaded_models):
        sensor_df, timestamps = make_normal_csv()
        result = compute_anomaly_score(sensor_df, timestamps, "2026-02-10", loaded_models)
        for key in ("score", "anomalies", "pre_incident_anomaly", "fraud_indicator"):
            assert key in result

    def test_anomalies_is_list(self, loaded_models):
        sensor_df, timestamps = make_normal_csv()
        result = compute_anomaly_score(sensor_df, timestamps, "2026-02-10", loaded_models)
        assert isinstance(result["anomalies"], list)

    def test_pre_incident_is_bool(self, loaded_models):
        sensor_df, timestamps = make_normal_csv()
        result = compute_anomaly_score(sensor_df, timestamps, "2026-02-10", loaded_models)
        assert isinstance(result["pre_incident_anomaly"], bool)

    def test_fraud_indicator_is_string(self, loaded_models):
        sensor_df, timestamps = make_normal_csv()
        result = compute_anomaly_score(sensor_df, timestamps, "2026-02-10", loaded_models)
        assert isinstance(result["fraud_indicator"], str)


# ── anomalies list ────────────────────────────────────────────────────────────

class TestAnomaliesList:
    def test_normal_data_anomalies_empty_or_few(self, loaded_models):
        sensor_df, timestamps = make_normal_csv(n_rows=2000, seed=42)
        result = compute_anomaly_score(sensor_df, timestamps, "2026-02-10", loaded_models)
        assert len(result["anomalies"]) <= 3

    def test_anomalous_data_produces_anomalies(self, loaded_models):
        sensor_df, timestamps = make_anomalous_csv(n_rows=2000, seed=0)
        result = compute_anomaly_score(sensor_df, timestamps, "2026-02-10", loaded_models)
        assert len(result["anomalies"]) >= 1

    def test_anomaly_item_has_correct_fields(self, loaded_models):
        sensor_df, timestamps = make_anomalous_csv(n_rows=2000, seed=0)
        result = compute_anomaly_score(sensor_df, timestamps, "2026-02-10", loaded_models)
        for item in result["anomalies"]:
            assert "timestamp" in item
            assert "parameter" in item
            assert "value"     in item
            assert "threshold" in item

    def test_anomaly_value_types(self, loaded_models):
        sensor_df, timestamps = make_anomalous_csv(n_rows=2000, seed=0)
        result = compute_anomaly_score(sensor_df, timestamps, "2026-02-10", loaded_models)
        for item in result["anomalies"]:
            assert isinstance(item["timestamp"], str)
            assert isinstance(item["parameter"], str)
            assert isinstance(item["value"],     float)
            assert isinstance(item["threshold"], float)

    def test_anomaly_parameter_is_valid_sensor(self, loaded_models):
        sensor_df, timestamps = make_anomalous_csv(n_rows=2000, seed=0)
        result = compute_anomaly_score(sensor_df, timestamps, "2026-02-10", loaded_models)
        for item in result["anomalies"]:
            assert item["parameter"] in SENSOR_COLS

    def test_anomaly_value_in_range(self, loaded_models):
        sensor_df, timestamps = make_anomalous_csv(n_rows=2000, seed=0)
        result = compute_anomaly_score(sensor_df, timestamps, "2026-02-10", loaded_models)
        for item in result["anomalies"]:
            assert 0.0 <= item["value"] <= 1.0


# ── fraud indicator ───────────────────────────────────────────────────────────

class TestFraudIndicator:
    def test_indicator_always_valid(self, loaded_models):
        for seed in range(4):
            sensor_df, timestamps = make_normal_csv(seed=seed)
            result = compute_anomaly_score(sensor_df, timestamps, "2026-02-10", loaded_models)
            assert result["fraud_indicator"] in VALID_INDICATORS

    def test_indicator_uses_underscores(self, loaded_models):
        """Ensure no space-separated variants leak through — NestJS requires underscores."""
        for seed in range(4):
            sensor_df, timestamps = make_normal_csv(seed=seed)
            result = compute_anomaly_score(sensor_df, timestamps, "2026-02-10", loaded_models)
            assert " " not in result["fraud_indicator"], (
                f"fraud_indicator contains spaces: '{result['fraud_indicator']}'"
            )

    def test_normal_data_no_precursor_indicator(self, loaded_models):
        sensor_df, timestamps = make_normal_csv(n_rows=2000, seed=0)
        result = compute_anomaly_score(sensor_df, timestamps, "2026-02-10", loaded_models)
        assert result["fraud_indicator"] != "GRADUAL_DEGRADATION"

    def test_anomalous_data_not_no_precursor(self, loaded_models):
        sensor_df, timestamps = make_anomalous_csv(n_rows=2000, seed=0)
        last_ts = str(timestamps.iloc[-1])[:10]
        result  = compute_anomaly_score(sensor_df, timestamps, last_ts, loaded_models)
        assert result["fraud_indicator"] != "NO_PRECURSOR_DETECTED"


# ── score math ────────────────────────────────────────────────────────────────

class TestScoreMath:
    def test_score_clipped_to_100(self, loaded_models):
        import pandas as pd
        ones_data  = np.ones((2000, len(SENSOR_COLS)))
        sensor_df  = pd.DataFrame(ones_data, columns=SENSOR_COLS)
        timestamps = pd.Series(pd.date_range("2026-01-01", periods=2000, freq="1min"))
        result     = compute_anomaly_score(sensor_df, timestamps, "2026-02-10", loaded_models)
        assert result["score"] <= 100

    def test_score_not_negative(self, loaded_models):
        import pandas as pd
        zeros_data = np.zeros((2000, len(SENSOR_COLS)))
        sensor_df  = pd.DataFrame(zeros_data, columns=SENSOR_COLS)
        timestamps = pd.Series(pd.date_range("2026-01-01", periods=2000, freq="1min"))
        result     = compute_anomaly_score(sensor_df, timestamps, "2026-02-10", loaded_models)
        assert result["score"] >= 0


# ── short data edge case ──────────────────────────────────────────────────────

class TestShortData:
    def test_short_csv_does_not_crash(self, loaded_models):
        sensor_df, timestamps = make_short_csv(n_rows=100)
        result = compute_anomaly_score(sensor_df, timestamps, "2026-02-01", loaded_models)
        assert "score" in result
        assert 0 <= result["score"] <= 100

    def test_short_csv_anomalies_empty(self, loaded_models):
        sensor_df, timestamps = make_short_csv(n_rows=100)
        result = compute_anomaly_score(sensor_df, timestamps, "2026-02-01", loaded_models)
        assert result["anomalies"] == []