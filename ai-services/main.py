import os
import io
import base64
import tempfile
import uvicorn
import pandas as pd
from fastapi import FastAPI, HTTPException, File, UploadFile
from fastapi.responses import RedirectResponse, FileResponse, JSONResponse
from pydantic import BaseModel
from models.classification.preprocessor import MaintenancePreprocessor
from models.classification.model import MaintenanceClassifier

app = FastAPI(
    title="classification-service",
    description="XGBoost 4-class failure classification — Fraud Detection Pipeline",
    version="1.0.0",
)

# ── Initialize ──────────────────────────────────────────────
preprocessor = MaintenancePreprocessor()
classifier   = MaintenanceClassifier(
    model_path="models/classification/artifacts/classifier.pkl"
)

FEATURE_IMPORTANCE_PATH = "models/classification/artifacts/feature_importance.png"

# ── Schemas ─────────────────────────────────────────────────
class MaintenanceInput(BaseModel):
    Type: str          # L, M, H
    Air_temperature: float
    Process_temperature: float
    Rotational_speed: float
    Torque: float
    Tool_wear: float


# ── Routes ──────────────────────────────────────────────────

@app.get("/", include_in_schema=False)
async def root():
    return RedirectResponse(url="/docs")


@app.get("/health")
def health_check():
    import os
    model_ready = os.path.exists(classifier.model_path)
    return {
        "status":       "healthy",
        "service":      "classification-service",
        "model_loaded": model_ready,
    }


# ── Original endpoint (kept for compatibility) ───────────────
@app.post("/predict/maintenance")
def predict_maintenance(data: MaintenanceInput):
    """Single-row prediction from JSON — original endpoint."""
    try:
        input_df = pd.DataFrame([{
            'Type':                    data.Type,
            'Air temperature [K]':     data.Air_temperature,
            'Process temperature [K]': data.Process_temperature,
            'Rotational speed [rpm]':  data.Rotational_speed,
            'Torque [Nm]':             data.Torque,
            'Tool wear [min]':         data.Tool_wear,
        }])

        processed_df = preprocessor.preprocess(input_df, is_training=False)
        prediction   = classifier.predict(processed_df)

        # Fraud score mapping
        fraud_map = {"FAKE": 90, "SABOTAGE": 95, "REAL_FAILURE": 30, "NORMAL_WEAR": 5}
        fraud_score = fraud_map.get(prediction[0], 50)

        return {
            "prediction":    prediction[0],
            "fraud_score":   fraud_score,
            "input_received": data.dict(),
            "service":       "classification-service",
        }
    except FileNotFoundError:
        raise HTTPException(status_code=503, detail="Model not trained yet. POST /train first.")
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# ── New endpoint: classify from CSV file ─────────────────────
@app.post("/classify-failure")
async def classify_failure(file: UploadFile = File(...)):
    """
    Predict failure class from a full sensor CSV file.
    Returns fraud_score (0-100), majority class, and feature importance.
    """
    if not file.filename.endswith(".csv"):
        raise HTTPException(status_code=400, detail="Only CSV files are accepted.")

    if not os.path.exists(classifier.model_path):
        raise HTTPException(status_code=503, detail="Model not trained yet. POST /train first.")

    # Save upload to temp file
    with tempfile.NamedTemporaryFile(suffix=".csv", delete=False) as tmp:
        tmp.write(await file.read())
        tmp_path = tmp.name

    try:
        df = pd.read_csv(tmp_path)
        processed_df = preprocessor.preprocess(df, is_training=False)
        predictions  = classifier.predict(processed_df)

        # Majority vote
        from collections import Counter
        majority = Counter(predictions).most_common(1)[0][0]

        # Fraud score
        fraud_map   = {"FAKE": 90, "SABOTAGE": 95, "REAL_FAILURE": 30, "NORMAL_WEAR": 5}
        fraud_score = fraud_map.get(majority, 50)

        # Class distribution
        distribution = dict(Counter(predictions))

        # Feature importance
        fi = {}
        if classifier.model is not None:
            fi = dict(zip(
                classifier.feature_cols,
                [round(float(v), 4) for v in classifier.model.feature_importances_]
            ))

        # Feature importance chart as base64
        chart_b64 = None
        if os.path.exists(FEATURE_IMPORTANCE_PATH):
            with open(FEATURE_IMPORTANCE_PATH, "rb") as f:
                chart_b64 = base64.b64encode(f.read()).decode("utf-8")

        return {
            "predicted_class":              majority,
            "fraud_score":                  fraud_score,
            "row_count":                    len(predictions),
            "class_distribution":           distribution,
            "feature_importance":           fi,
            "feature_importance_chart_b64": chart_b64,
            "model":                        "XGBoost",
            "service":                      "classification-service",
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
    finally:
        os.unlink(tmp_path)


# ── New endpoint: classify from JSON features ────────────────
@app.post("/classify-features")
def classify_features(data: MaintenanceInput):
    """Single-row prediction — same as /predict/maintenance but with richer response."""
    return predict_maintenance(data)


# ── New endpoint: train from uploaded CSV ────────────────────
@app.post("/train")
async def train_model(file: UploadFile = File(...)):
    """
    Retrain XGBoost from an uploaded CSV (Kaggle Predictive Maintenance dataset).
    Saves classifier.pkl and feature_importance.png.
    """
    if not file.filename.endswith(".csv"):
        raise HTTPException(status_code=400, detail="Only CSV files are accepted.")

    with tempfile.NamedTemporaryFile(suffix=".csv", delete=False) as tmp:
        tmp.write(await file.read())
        tmp_path = tmp.name

    try:
        df = pd.read_csv(tmp_path)

        # Preprocess
        processed_df = preprocessor.preprocess(df, is_training=True)

        # Generate synthetic FAKE + SABOTAGE data
        augmented_df = classifier.generate_synthetic_data(processed_df)

        # Train
        X_test, y_test = classifier.train(augmented_df)

        # Evaluate
        metrics = classifier.evaluate(X_test, y_test)

        # Check acceptance criteria
        assert metrics["accuracy"]  >= 0.80, f"Accuracy {metrics['accuracy']:.2%} < 80%"
        assert metrics["precision"] >= 0.75, f"Precision {metrics['precision']:.2%} < 75%"
        assert metrics["recall"]    >= 0.80, f"Recall {metrics['recall']:.2%} < 80%"

        # Save model
        classifier.save()

        # Save feature importance chart
        _save_feature_importance_chart(classifier)

        return {
            "status":    "trained",
            "accuracy":  round(metrics["accuracy"],  4),
            "precision": round(metrics["precision"], 4),
            "recall":    round(metrics["recall"],    4),
        }

    except AssertionError as e:
        raise HTTPException(status_code=422, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
    finally:
        os.unlink(tmp_path)


# ── New endpoint: feature importance chart ───────────────────
@app.get("/feature-importance")
def get_feature_importance():
    """Return the feature importance chart as PNG."""
    if not os.path.exists(FEATURE_IMPORTANCE_PATH):
        raise HTTPException(
            status_code=404,
            detail="Chart not found. Train the model first via POST /train.",
        )
    return FileResponse(FEATURE_IMPORTANCE_PATH, media_type="image/png")


# ── Helper ───────────────────────────────────────────────────
def _save_feature_importance_chart(clf: MaintenanceClassifier):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np

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

    os.makedirs(os.path.dirname(FEATURE_IMPORTANCE_PATH), exist_ok=True)
    plt.savefig(FEATURE_IMPORTANCE_PATH, dpi=150)
    plt.close(fig)


# ── Entry point ──────────────────────────────────────────────
if __name__ == "__main__":
    uvicorn.run(app, host="127.0.0.1", port=8002)