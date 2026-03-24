import os
import sys
import kagglehub
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# Adjust sys.path to import from app
sys.path.append(os.path.join(os.path.dirname(__file__), 'app'))

from app.models.preprocessor import MaintenancePreprocessor
from app.models.model import MaintenanceClassifier
from app.config import MODEL_PATH, FEATURE_IMPORTANCE_PATH

def train():
    """
    Automated training pipeline:
    1. Download dataset from Kaggle
    2. Preprocess and Engineer features
    3. Generate synthetic data (FAKE/SABOTAGE)
    4. Train XGBoost model
    5. Evaluate performance
    6. Save artifacts
    """
    print("\n🚀 Starting Classification Model Training...")
    
    # 1. Download dataset from Kaggle
    print("📥 Downloading dataset from Kaggle (shivamb/machine-predictive-maintenance-classification)...")
    try:
        path = kagglehub.dataset_download("shivamb/machine-predictive-maintenance-classification")
    except Exception as e:
        print(f"❌ Error downloading dataset: {e}")
        return

    csv_path = os.path.join(path, "predictive_maintenance.csv")

    if not os.path.exists(csv_path):
        # Fallback if filename is different
        files = os.listdir(path)
        csv_files = [f for f in files if f.endswith('.csv')]
        if csv_files:
            csv_path = os.path.join(path, csv_files[0])
        else:
            print(f"❌ No CSV file found in dataset at {path}")
            return

    print(f"📂 Loading data from {csv_path}...")
    df = pd.read_csv(csv_path)

    # 2. Preprocess
    print("⚙️ Preprocessing data...")
    preprocessor = MaintenancePreprocessor()
    # Path is relative to the directory where we want artifacts to go
    # Classifier saves to artifacts/classifier.pkl by default, let's use the one from config
    classifier = MaintenanceClassifier(model_path=os.path.join('app', MODEL_PATH))
    
    processed_df = preprocessor.preprocess(df, is_training=True)
    
    # 3. Augment data (FAKE and SABOTAGE classes)
    print("🧪 Generating synthetic fraud data (FAKE/SABOTAGE)...")
    augmented_df = classifier.generate_synthetic_data(processed_df)

    # 4. Train
    print("🧠 Training XGBoost model...")
    X_test, y_test, train_acc = classifier.train(augmented_df)
    
    # 5. Evaluate
    print(f"📈 Training Accuracy: {train_acc:.4f}")
    metrics = classifier.evaluate(X_test, y_test)
    
    # Acceptance criteria (matching main.py logic)
    print("\n🔍 Checking performance against requirements:")
    req_met = True
    if metrics['accuracy'] < 0.80:
        print(f"❌ Accuracy {metrics['accuracy']:.2%} is below 80%")
        req_met = False
    else:
        print(f"✅ Accuracy {metrics['accuracy']:.2%} >= 80%")

    if metrics['precision'] < 0.75:
        print(f"❌ Precision {metrics['precision']:.2%} is below 75%")
        req_met = False
    else:
        print(f"✅ Precision {metrics['precision']:.2%} >= 75%")

    if metrics['recall'] < 0.80:
        print(f"❌ Recall {metrics['recall']:.2%} is below 80%")
        req_met = False
    else:
        print(f"✅ Recall {metrics['recall']:.2%} >= 80%")
    
    if not req_met:
        print("⚠️ Warning: Model does not meet all acceptance criteria, but saving anyway.")
    
    # 6. Save model
    print("💾 Saving model artifacts...")
    classifier.save()
    
    # 7. Generate feature importance chart
    print("📊 Generating feature importance chart...")
    save_feature_importance_chart(classifier, os.path.join('app', FEATURE_IMPORTANCE_PATH))
    
    print("\n✨ Training complete! Services can now use the new model.")

def save_feature_importance_chart(clf, path):
    """
    Reproduce the chart generation from main.py for consistency.
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    
    if clf.model is None:
        return

    importances = clf.model.feature_importances_
    features    = clf.feature_cols
    indices     = np.argsort(importances)

    fig, ax = plt.subplots(figsize=(10, 6))
    bars = ax.barh(
        [features[i] for i in indices],
        importances[indices],
        color="#4F81BD",
    )
    ax.set_xlabel("Importance (gain)", fontsize=12)
    ax.set_title(
        "XGBoost Feature Importance — Failure Classification",
        fontsize=14, fontweight="bold",
    )
    ax.bar_label(bars, fmt="%.3f", padding=4, fontsize=9)
    plt.tight_layout()

    os.makedirs(os.path.dirname(path), exist_ok=True)
    plt.savefig(path, dpi=150)
    plt.close(fig)

if __name__ == "__main__":
    train()
