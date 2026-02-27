import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import kagglehub
from models.classification.preprocessor import MaintenancePreprocessor
from models.classification.model import MaintenanceClassifier

def generate_mock_data(n_samples=1000):
    """Generates a mock dataset matching the Kaggle predictive maintenance schema."""
    np.random.seed(42)
    data = {
        'UDI': range(1, n_samples + 1),
        'Product ID': [f"{np.random.choice(['L', 'M', 'H'])}{10000+i}" for i in range(n_samples)],
        'Type': np.random.choice(['L', 'M', 'H'], n_samples),
        'Air temperature [K]': np.random.normal(300, 2, n_samples),
        'Process temperature [K]': np.random.normal(310, 2, n_samples),
        'Rotational speed [rpm]': np.random.normal(1500, 200, n_samples),
        'Torque [Nm]': np.random.normal(40, 10, n_samples),
        'Tool wear [min]': np.random.randint(0, 250, n_samples),
        'Failure Type': np.random.choice(['No Failure', 'TWF', 'HDF', 'PWF', 'OSF', 'RNF'], n_samples, p=[0.95, 0.01, 0.01, 0.01, 0.01, 0.01])
    }
    return pd.DataFrame(data)

def main():
    # 1. Load Data from Kaggle
    print("Downloading dataset from Kaggle...")
    try:
        # Using kagglehub to download the specific dataset
        path = kagglehub.dataset_download("shivamb/machine-predictive-maintenance-classification")
        print(f"Dataset downloaded to: {path}")
        
        # Locate the CSV file in the downloaded path
        csv_files = [f for f in os.listdir(path) if f.endswith('.csv')]
        if not csv_files:
            raise FileNotFoundError("No CSV file found in the downloaded Kaggle dataset.")
            
        csv_path = os.path.join(path, csv_files[0])
        print(f"Loading dataset from {csv_path}...")
        df = pd.read_csv(csv_path)
    except Exception as e:
        print(f"Error downloading from Kaggle: {e}")
        print("Falling back to local data if available or generation...")
        data_path = "predictive_maintenance.csv"
        if os.path.exists(data_path):
            df = pd.read_csv(data_path)
        else:
            print("Generating mock data for fallback...")
            df = generate_mock_data(2000)

    # 2. Preprocess
    print("Preprocessing data...")
    preprocessor = MaintenancePreprocessor()
    df_processed = preprocessor.preprocess(df, is_training=True)

    # 3. Model Orchestration
    print("Initializing classifier and generating synthetic data...")
    classifier = MaintenanceClassifier(model_path="models/classification/artifacts/classifier.pkl")
    df_processed = df_processed[df_processed['Target_Class'].isin(['NORMAL_WEAR', 'REAL_FAILURE'])].copy()
    df_synthetic = classifier.generate_synthetic_data(df_processed)

    # 4. Train
    print("Training XGBoost model...")
    X_test, y_test = classifier.train(df_synthetic)

    # 5. Evaluate
    print("Evaluating model...")
    metrics = classifier.evaluate(X_test, y_test)

    # 6. Save Model
    classifier.save()

    # 7. Feature Importance Chart
    print("Generating feature importance chart...")
    importance = classifier.model.get_booster().get_score(importance_type='weight')
    # Map back encoded column names if necessary (they should match feature_cols)
    feat_importances = pd.Series(importance)
    feat_importances.nlargest(10).plot(kind='barh', title='Feature Importance')
    
    chart_path = "models/classification/artifacts/feature_importance.png"
    plt.tight_layout()
    plt.savefig(chart_path)
    print(f"Feature importance chart saved to {chart_path}")

    if metrics['accuracy'] > 0.8:
        print("\nSUCCESS: Model meets accuracy requirements (> 80%)")
    else:
        print("\nWARNING: Model does not meet accuracy requirements.")

if __name__ == "__main__":
    main()
