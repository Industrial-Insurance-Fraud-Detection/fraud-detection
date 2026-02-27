import os
import joblib
import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, classification_report
from sklearn.preprocessing import LabelEncoder

class MaintenanceClassifier:
    def __init__(self, model_path="artifacts/classifier.pkl"):
        self.model_path = model_path
        self.model = None
        self.label_encoder = LabelEncoder()
        self.feature_cols = [
            'Type', 'Air_temperature', 'Process_temperature', 
            'Rotational_speed', 'Torque', 'Tool_wear', 
            'Power', 'Temp_diff', 'Wear_rate'
        ]

    def generate_synthetic_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Create synthetic data for FAKE and SABOTAGE classes.
        - FAKE: copy REAL_FAILURE rows, set anomaly features to zero.
        - SABOTAGE: copy NORMAL rows, increase temperature and torque by 300%.
        """
        # Separate Real Failure and Normal Wear
        real_failures = df[df['Target_Class'] == 'REAL_FAILURE'].copy()
        normal_wear = df[df['Target_Class'] == 'NORMAL_WEAR'].copy()

        # Generate FAKE
        fake_data = real_failures.copy()
        # "set anomaly features to zero" - assuming this means the engineered anomaly indicators
        # or just zero out the main features that might cause failure.
        # Given the instruction is brief, I'll zero out rotational speed and torque as a proxy.
        fake_data['Rotational_speed'] = 0
        fake_data['Torque'] = 0
        fake_data['Power'] = 0
        fake_data['Target_Class'] = 'FAKE'

        # Generate SABOTAGE
        sabotage_data = normal_wear.copy()
        # "increase temperature and torque by 300%"
        sabotage_data['Air_temperature'] *= 4.0 # increase BY 300% means 4x
        # For torque, it's a critical factor in failure
        sabotage_data['Torque'] *= 4.0
        # Re-engineer features affected by these changes
        sabotage_data['Power'] = sabotage_data['Torque'] * sabotage_data['Rotational_speed']
        sabotage_data['Target_Class'] = 'SABOTAGE'

        # Combine all
        combined_df = pd.concat([df, fake_data, sabotage_data], ignore_index=True)
        return combined_df

    def train(self, df: pd.DataFrame):
        """
        Train XGBoost classifier for 4-class classification.
        Split: 70% train / 10% validation / 20% test.
        """
        X = df[self.feature_cols]
        y = self.label_encoder.fit_transform(df['Target_Class'])

        # Split: 70/30 first
        X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=0.3, random_state=42, stratify=y)
        
        # Split 30% into 1/3 (10%) and 2/3 (20%)
        X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.66, random_state=42, stratify=y_temp)
        
        num_class = len(np.unique(y))
        
        self.model = xgb.XGBClassifier(
            objective='multi:softprob',
            eval_metric='mlogloss',
            num_class=num_class,
            random_state=42,
            n_estimators=300,
            max_depth=6,
            learning_rate=0.05
        )
        self.model.fit(
            X_train, y_train,
            eval_set=[(X_val, y_val)],
            verbose=False
        )

        return X_test, y_test

    def evaluate(self, X_test, y_test):
        """
        Evaluate model performance.
        Requirements: Accuracy > 80%, Precision > 75%, Recall > 80%.
        """
        y_pred = self.model.predict(X_test)
        
        metrics = {
            'accuracy': accuracy_score(y_test, y_pred),
            'precision': precision_score(y_test, y_pred, average='weighted'),
            'recall': recall_score(y_test, y_pred, average='weighted')
        }
        
        print("\nModel Evaluation Results:")
        for k, v in metrics.items():
            print(f"{k.capitalize()}: {v:.4f}")
            
        print("\nClassification Report:")
        print(classification_report(y_test, y_pred, target_names=self.label_encoder.classes_))
        
        return metrics

    def save(self):
        """Save model and label encoder."""
        os.makedirs(os.path.dirname(self.model_path), exist_ok=True)
        joblib.dump({
            'model': self.model,
            'label_encoder': self.label_encoder,
            'feature_cols': self.feature_cols
        }, self.model_path)
        print(f"Model saved to {self.model_path}")

    def load(self):
        """Load model and label encoder."""
        if os.path.exists(self.model_path):
            data = joblib.load(self.model_path)
            self.model = data['model']
            self.label_encoder = data['label_encoder']
            self.feature_cols = data['feature_cols']
        else:
            raise FileNotFoundError(f"Model file not found at {self.model_path}")
            
    def predict(self, input_features: pd.DataFrame):
        """Predict class for input features."""
        if self.model is None:
            self.load()
        
        # Ensure correct column order
        X = input_features[self.feature_cols]
        preds = self.model.predict(X)
        return self.label_encoder.inverse_transform(preds)
