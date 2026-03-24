"""
tests/test_classification_service.py
──────────────────────────────────────
Tests automatiques pour le classification-service.

Lancer avec :
    pytest tests/ -v
    pytest tests/ -v --cov=models
"""

import pytest
import pandas as pd
import numpy as np
import sys
import os

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from app.models.preprocessor import MaintenancePreprocessor
from app.models.model import MaintenanceClassifier


# ── Fixtures ────────────────────────────────────────────────

@pytest.fixture
def preprocessor():
    return MaintenancePreprocessor()


@pytest.fixture
def sample_raw_df():
    """Sample raw DataFrame mimicking Kaggle dataset."""
    return pd.DataFrame({
        'UDI': [1, 2, 3, 4, 5],
        'Product ID': ['M14860', 'L47181', 'L47182', 'L47183', 'M14861'],
        'Type': ['M', 'L', 'L', 'L', 'M'],
        'Air temperature [K]': [298.1, 298.2, 298.1, 298.2, 298.1],
        'Process temperature [K]': [308.6, 308.7, 308.5, 308.6, 308.7],
        'Rotational speed [rpm]': [1551.0, 1408.0, 1498.0, 1433.0, 1551.0],
        'Torque [Nm]': [42.8, 46.3, 49.4, 39.5, 42.8],
        'Tool wear [min]': [0, 3, 5, 7, 0],
        'Target': [0, 0, 0, 0, 0],
        'Failure Type': ['No Failure', 'No Failure', 'TWF', 'HDF', 'No Failure']
    })


@pytest.fixture
def trained_classifier():
    """Load trained classifier if available."""
    clf = MaintenanceClassifier(
        model_path='app/models/classifier.pkl'
    )
    if os.path.exists(clf.model_path):
        clf.load()
        return clf
    pytest.skip("Model not trained yet.")


# ── Preprocessor Tests ───────────────────────────────────────

class TestPreprocessor:

    def test_drops_udi_and_product_id(self, preprocessor, sample_raw_df):
        """UDI and Product ID should be dropped."""
        result = preprocessor.preprocess(sample_raw_df, is_training=False)
        assert 'UDI' not in result.columns
        assert 'Product ID' not in result.columns

    def test_encodes_type_column(self, preprocessor, sample_raw_df):
        """Type column should be encoded as 0/1/2."""
        result = preprocessor.preprocess(sample_raw_df, is_training=False)
        assert result['Type'].dtype in [np.int64, np.float64]
        assert set(result['Type'].unique()).issubset({0, 1, 2})

    def test_engineers_power_feature(self, preprocessor, sample_raw_df):
        """Power = Torque × Rotational_speed."""
        result = preprocessor.preprocess(sample_raw_df, is_training=False)
        assert 'Power' in result.columns
        expected = result['Torque'] * result['Rotational_speed']
        pd.testing.assert_series_equal(result['Power'], expected, check_names=False)

    def test_engineers_temp_diff_feature(self, preprocessor, sample_raw_df):
        """Temp_diff = Process_temperature - Air_temperature."""
        result = preprocessor.preprocess(sample_raw_df, is_training=False)
        assert 'Temp_diff' in result.columns
        expected = result['Process_temperature'] - result['Air_temperature']
        pd.testing.assert_series_equal(result['Temp_diff'], expected, check_names=False)

    def test_engineers_wear_rate_feature(self, preprocessor, sample_raw_df):
        """Wear_rate = Tool_wear / Rotational_speed."""
        result = preprocessor.preprocess(sample_raw_df, is_training=False)
        assert 'Wear_rate' in result.columns

    def test_maps_failure_types_training(self, preprocessor, sample_raw_df):
        """Failure types should be mapped to 4 classes when is_training=True."""
        result = preprocessor.preprocess(sample_raw_df, is_training=True)
        assert 'Target_Class' in result.columns
        valid_classes = {'NORMAL_WEAR', 'REAL_FAILURE', 'FAKE', 'SABOTAGE'}
        assert set(result['Target_Class'].unique()).issubset(valid_classes)

    def test_no_failure_maps_to_normal_wear(self, preprocessor, sample_raw_df):
        """No Failure → NORMAL_WEAR."""
        result = preprocessor.preprocess(sample_raw_df, is_training=True)
        no_failure_rows = sample_raw_df[sample_raw_df['Failure Type'] == 'No Failure'].index
        assert all(result.loc[no_failure_rows, 'Target_Class'] == 'NORMAL_WEAR')

    def test_twf_maps_to_real_failure(self, preprocessor, sample_raw_df):
        """TWF → REAL_FAILURE."""
        result = preprocessor.preprocess(sample_raw_df, is_training=True)
        twf_rows = sample_raw_df[sample_raw_df['Failure Type'] == 'TWF'].index
        assert all(result.loc[twf_rows, 'Target_Class'] == 'REAL_FAILURE')

    def test_no_target_class_when_not_training(self, preprocessor, sample_raw_df):
        """Target_Class should NOT be created when is_training=False."""
        result = preprocessor.preprocess(sample_raw_df, is_training=False)
        assert 'Target_Class' not in result.columns

    def test_no_nan_values(self, preprocessor, sample_raw_df):
        """No NaN values in output."""
        result = preprocessor.preprocess(sample_raw_df, is_training=False)
        feature_cols = ['Type', 'Air_temperature', 'Process_temperature',
                       'Rotational_speed', 'Torque', 'Tool_wear',
                       'Power', 'Temp_diff', 'Wear_rate']
        assert not result[feature_cols].isnull().any().any()


# ── Model Tests ──────────────────────────────────────────────

class TestModel:

    def test_model_has_correct_classes(self, trained_classifier):
        """Model should have exactly 4 classes."""
        classes = list(trained_classifier.label_encoder.classes_)
        assert set(classes) == {'FAKE', 'NORMAL_WEAR', 'REAL_FAILURE', 'SABOTAGE'}

    def test_predict_fake(self, trained_classifier):
        """RPM=0, Torque=0 should predict FAKE."""
        preprocessor = MaintenancePreprocessor()
        test = pd.DataFrame([{
            'Type': 'M',
            'Air temperature [K]': 298.1,
            'Process temperature [K]': 308.6,
            'Rotational speed [rpm]': 0,
            'Torque [Nm]': 0,
            'Tool wear [min]': 0
        }])
        processed = preprocessor.preprocess(test, is_training=False)
        result = trained_classifier.predict(processed)
        assert result[0] == 'FAKE'

    def test_predict_sabotage(self, trained_classifier):
        """Extreme temperature and torque should predict SABOTAGE."""
        preprocessor = MaintenancePreprocessor()
        test = pd.DataFrame([{
            'Type': 'H',
            'Air temperature [K]': 892.0,
            'Process temperature [K]': 1200.0,
            'Rotational speed [rpm]': 1400,
            'Torque [Nm]': 180.0,
            'Tool wear [min]': 50
        }])
        processed = preprocessor.preprocess(test, is_training=False)
        result = trained_classifier.predict(processed)
        assert result[0] == 'SABOTAGE'

    def test_predict_normal_wear(self, trained_classifier):
        """Normal readings should predict NORMAL_WEAR."""
        preprocessor = MaintenancePreprocessor()
        test = pd.DataFrame([{
            'Type': 'M',
            'Air temperature [K]': 298.1,
            'Process temperature [K]': 308.6,
            'Rotational speed [rpm]': 1551,
            'Torque [Nm]': 42.8,
            'Tool wear [min]': 0
        }])
        processed = preprocessor.preprocess(test, is_training=False)
        result = trained_classifier.predict(processed)
        assert result[0] == 'NORMAL_WEAR'

    def test_predict_returns_valid_class(self, trained_classifier):
        """Prediction should always return a valid class."""
        preprocessor = MaintenancePreprocessor()
        test = pd.DataFrame([{
            'Type': 'L',
            'Air temperature [K]': 298.0,
            'Process temperature [K]': 318.0,
            'Rotational speed [rpm]': 1408,
            'Torque [Nm]': 68.0,
            'Tool wear [min]': 210
        }])
        processed = preprocessor.preprocess(test, is_training=False)
        result = trained_classifier.predict(processed)
        valid_classes = {'FAKE', 'NORMAL_WEAR', 'REAL_FAILURE', 'SABOTAGE'}
        assert result[0] in valid_classes

    def test_feature_importance_available(self, trained_classifier):
        """Feature importance should be available after training."""
        assert trained_classifier.model is not None
        importances = trained_classifier.model.feature_importances_
        assert len(importances) == len(trained_classifier.feature_cols)
        assert all(imp >= 0 for imp in importances)


# ── Synthetic Data Tests ─────────────────────────────────────

class TestSyntheticData:

    def test_generates_fake_class(self):
        """generate_synthetic_data should create FAKE rows."""
        preprocessor = MaintenancePreprocessor()
        clf = MaintenanceClassifier()

        df = pd.DataFrame({
            'Type': [1, 0],
            'Air_temperature': [298.1, 298.2],
            'Process_temperature': [308.6, 308.7],
            'Rotational_speed': [1551.0, 1408.0],
            'Torque': [42.8, 46.3],
            'Tool_wear': [0, 3],
            'Power': [1551*42.8, 1408*46.3],
            'Temp_diff': [10.5, 10.5],
            'Wear_rate': [0, 0.002],
            'Target_Class': ['NORMAL_WEAR', 'REAL_FAILURE']
        })

        result = clf.generate_synthetic_data(df)
        assert 'FAKE' in result['Target_Class'].values
        assert 'SABOTAGE' in result['Target_Class'].values

    def test_fake_rows_have_zero_power(self):
        """FAKE rows should have Power=0."""
        clf = MaintenanceClassifier()

        df = pd.DataFrame({
            'Type': [1],
            'Air_temperature': [298.1],
            'Process_temperature': [308.6],
            'Rotational_speed': [1551.0],
            'Torque': [42.8],
            'Tool_wear': [0],
            'Power': [1551*42.8],
            'Temp_diff': [10.5],
            'Wear_rate': [0.001],
            'Target_Class': ['REAL_FAILURE']
        })

        result = clf.generate_synthetic_data(df)
        fake_rows = result[result['Target_Class'] == 'FAKE']
        assert len(fake_rows) > 0
        assert all(fake_rows['Power'] == 0)
        assert all(fake_rows['Torque'] == 0)

    def test_sabotage_rows_have_high_temperature(self):
        """SABOTAGE rows should have increased temperature."""
        clf = MaintenanceClassifier()

        original_temp = 298.1
        df = pd.DataFrame({
            'Type': [1],
            'Air_temperature': [original_temp],
            'Process_temperature': [308.6],
            'Rotational_speed': [1551.0],
            'Torque': [42.8],
            'Tool_wear': [0],
            'Power': [1551*42.8],
            'Temp_diff': [10.5],
            'Wear_rate': [0.001],
            'Target_Class': ['NORMAL_WEAR']
        })

        result = clf.generate_synthetic_data(df)
        sabotage_rows = result[result['Target_Class'] == 'SABOTAGE']
        assert len(sabotage_rows) > 0
        assert all(sabotage_rows['Air_temperature'] > original_temp)
