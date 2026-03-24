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
        - FAKE: copy REAL_FAILURE rows, zero out the anomaly features.
        - SABOTAGE: copy NORMAL rows, spike temperature and torque by 300%.
        """
        real_failures = df[df['Target_Class'] == 'REAL_FAILURE'].copy()
        normal_wear   = df[df['Target_Class'] == 'NORMAL_WEAR'].copy()

        # Generate FAKE: copy REAL_FAILURE rows, zero out features as per Kaggle
        fake_data = real_failures.copy()
        fake_data['Rotational_speed'] = 0
        fake_data['Torque']           = 0
        fake_data['Power']            = 0
        fake_data['Target_Class']     = 'FAKE'

        # Generate SABOTAGE: copy NORMAL rows, spike temperature and torque (400% total)
        sabotage_data = normal_wear.copy()
        sabotage_data['Air_temperature'] *= 4.0
        sabotage_data['Torque']          *= 4.0
        sabotage_data['Power']            = sabotage_data['Torque'] * sabotage_data['Rotational_speed']
        sabotage_data['Target_Class']     = 'SABOTAGE'

        # Combine all
        combined_df = pd.concat([df, fake_data, sabotage_data], ignore_index=True)
        # Filter to only the 4 required classes
        combined_df = combined_df[combined_df['Target_Class'].isin(['NORMAL_WEAR', 'REAL_FAILURE', 'FAKE', 'SABOTAGE'])]
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
        
        num_class = len(self.label_encoder.classes_)
        
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

        # Calculate training accuracy for overfitting check
        y_train_pred = self.model.predict(X_train)
        train_accuracy = accuracy_score(y_train, y_train_pred)

        return X_test, y_test, train_accuracy

    def evaluate(self, X_test, y_test):
        """
        Evaluate model performance.
        Requirements: Accuracy > 80%, Precision > 75%, Recall > 80%.
        """
        y_pred = self.model.predict(X_test)
        
        metrics = {
            'accuracy': accuracy_score(y_test, y_pred),
            'precision': precision_score(y_test, y_pred, average='weighted', zero_division=0),
            'recall': recall_score(y_test, y_pred, average='weighted', zero_division=0)
        }
        
        print("\nModel Evaluation Results:")
        for k, v in metrics.items():
            print(f"{k.capitalize()}: {v:.4f}")
            
        print("\nClassification Report:")
        print(classification_report(y_test, y_pred, target_names=self.label_encoder.classes_, zero_division=0))
        
        return metrics

    def class_to_fraud_score(self, predicted_class: str, confidence: float) -> int:
        """
        Convert a class label + confidence into a 0-100 fraud score.
        Formula: score = base * confidence + 50 * (1 - confidence)
        """
        # Base fraud score per class (how suspicious each class is)
        CLASS_TO_BASE_SCORE = {
            "FAKE":         90,   # direct fraud signal — no physical signature
            "SABOTAGE":     85,   # deliberate damage — strong fraud signal
            "REAL_FAILURE": 20,   # genuine failure — low fraud
            "NORMAL_WEAR":  10,   # normal usage — very low fraud
        }
        base = CLASS_TO_BASE_SCORE.get(predicted_class, 50)
        score = base * confidence + 50 * (1 - confidence)
        return int(round(score))

    def build_classification_response(self, row: pd.Series) -> dict:
        """
        Given one row of features, run the model and return structured JSON response.
        Matches the spec: { score, class, confidence, featureImportance }
        """
        if self.model is None:
            self.load()

        # Prepare input as a single-row DataFrame
        X_input = pd.DataFrame([row])[self.feature_cols]

        # Get softmax probabilities
        proba = self.model.predict_proba(X_input)[0]

        # Predicted class index and name
        class_idx = int(np.argmax(proba))
        predicted_class = self.label_encoder.classes_[class_idx]
        confidence = float(round(proba[class_idx], 4))

        # Convert to 0-100 fraud score
        score = self.class_to_fraud_score(predicted_class, confidence)

        # Feature importance (model-level)
        feature_importance = {
            col: round(float(imp), 4)
            for col, imp in zip(self.feature_cols, self.model.feature_importances_)
        }

        return {
            "score": score,
            "class": predicted_class,
            "confidence": confidence,
            "featureImportance": feature_importance
        }

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
