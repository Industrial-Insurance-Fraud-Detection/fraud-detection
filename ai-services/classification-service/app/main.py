import os
import io
import base64
import tempfile
import uvicorn
import pandas as pd
from fastapi import FastAPI, HTTPException, File, UploadFile
from fastapi.responses import RedirectResponse, FileResponse
from minio import Minio

# Internal imports
from config import (
    SERVICE_NAME, SERVICE_VERSION, SERVICE_PORT,
    MODEL_PATH, FEATURE_IMPORTANCE_PATH,
    MINIO_ENDPOINT, MINIO_ACCESS_KEY, MINIO_SECRET_KEY, MINIO_BUCKET,
)
from schemas import (
    MaintenanceInput, ClassificationResponse,
    ClassifyFailureResponse, ClassifyJsonRequest,
    TrainResponse, HealthResponse
)
from models.preprocessor import MaintenancePreprocessor
from models.model import MaintenanceClassifier

app = FastAPI(
    title=SERVICE_NAME,
    description="XGBoost 4-class failure classification — Fraud Detection Pipeline",
    version=SERVICE_VERSION,
)

# ── Initialize ──────────────────────────────────────────────
preprocessor = MaintenancePreprocessor()
classifier   = MaintenanceClassifier(model_path=MODEL_PATH)


# ── MinIO client ─────────────────────────────────────────────
def get_minio() -> Minio:
    return Minio(
        MINIO_ENDPOINT,
        access_key=MINIO_ACCESS_KEY,
        secret_key=MINIO_SECRET_KEY,
        secure=False,
    )


# ── Routes ──────────────────────────────────────────────────

@app.get("/", include_in_schema=False)
async def root():
    return RedirectResponse(url="/docs")


@app.get("/health", response_model=HealthResponse)
def health_check():
    model_ready = os.path.exists(classifier.model_path)
    return {
        "status":       "healthy" if model_ready else "degraded",
        "service":      SERVICE_NAME,
        "model_loaded": model_ready,
    }


# ── Main endpoint called by NestJS worker (JSON) ─────────────
@app.post("/classify-json", response_model=ClassifyFailureResponse)
def classify_json(req: ClassifyJsonRequest):
    """
    Called by the NestJS queue worker.
    Accepts JSON with csvPath (MinIO object key), downloads the CSV,
    runs XGBoost classification, returns fraud_score 0-100.

    This is the primary integration endpoint — consistent with how
    anomaly-service works (JSON in, score out, MinIO fetch internal).
    """
    if not os.path.exists(classifier.model_path):
        raise HTTPException(status_code=503, detail="Model not trained yet.")

    if not req.csvPath:
        raise HTTPException(status_code=400, detail="csvPath is required")

    try:
        # download CSV from MinIO
        minio_client = get_minio()
        response     = minio_client.get_object(MINIO_BUCKET, req.csvPath)
        csv_bytes    = response.read()
        df           = pd.read_csv(io.BytesIO(csv_bytes))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to fetch CSV from MinIO: {str(e)}")

    try:
        return _run_classification(df)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# ── Original endpoint: classify from CSV file upload ─────────
@app.post("/classify-failure", response_model=ClassifyFailureResponse)
async def classify_failure(file: UploadFile = File(...)):
    """
    Predict failure class from a full sensor CSV file upload.
    Used for direct testing via Swagger or curl.
    """
    if not file.filename.endswith(".csv"):
        raise HTTPException(status_code=400, detail="Only CSV files are accepted.")

    if not os.path.exists(classifier.model_path):
        raise HTTPException(status_code=503, detail="Model not trained yet.")

    with tempfile.NamedTemporaryFile(suffix=".csv", delete=False) as tmp:
        tmp.write(await file.read())
        tmp_path = tmp.name

    try:
        df = pd.read_csv(tmp_path)
        return _run_classification(df)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
    finally:
        os.unlink(tmp_path)


# ── Shared classification logic ───────────────────────────────
def _run_classification(df: pd.DataFrame) -> dict:
    """
    Runs XGBoost classification on a DataFrame.
    Shared by both /classify-json and /classify-failure.
    """
    processed_df = preprocessor.preprocess(df, is_training=False)

    results  = [classifier.build_classification_response(row) for _, row in processed_df.iterrows()]

    from collections import Counter
    classes  = [r["class"] for r in results]
    majority = Counter(classes).most_common(1)[0][0]

    majority_confidences = [r["confidence"] for r in results if r["class"] == majority]
    avg_confidence       = sum(majority_confidences) / len(majority_confidences)
    fraud_score          = classifier.class_to_fraud_score(majority, avg_confidence)

    distribution = dict(Counter(classes))
    fi           = results[0]["featureImportance"]

    chart_b64 = None
    if os.path.exists(FEATURE_IMPORTANCE_PATH):
        with open(FEATURE_IMPORTANCE_PATH, "rb") as f:
            chart_b64 = base64.b64encode(f.read()).decode("utf-8")

    return {
        "predicted_class":              majority,
        "fraud_score":                  fraud_score,
        "score":                        fraud_score,  # alias for worker extractScore
        "row_count":                    len(results),
        "class_distribution":           distribution,
        "feature_importance":           fi,
        "feature_importance_chart_b64": chart_b64,
        "model":                        "XGBoost",
        "service":                      SERVICE_NAME,
    }


# ── classify from JSON features (single row) ────────────────
@app.post("/classify-features", response_model=ClassificationResponse)
def classify_features(data: MaintenanceInput):
    """Single-row prediction from JSON features."""
    return predict_maintenance(data)


@app.post("/predict/maintenance", response_model=ClassificationResponse)
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

        processed_df      = preprocessor.preprocess(input_df, is_training=False)
        prediction_result = classifier.build_classification_response(processed_df.iloc[0])

        return {
            "prediction":     prediction_result["class"],
            "fraud_score":    prediction_result["score"],
            "confidence":     prediction_result["confidence"],
            "input_received": data.dict(),
            "service":        SERVICE_NAME,
        }
    except FileNotFoundError:
        raise HTTPException(status_code=503, detail="Model not trained yet. POST /train first.")
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# ── train from uploaded CSV ──────────────────────────────────
@app.post("/train", response_model=TrainResponse)
async def train_model(file: UploadFile = File(...)):
    """Retrain XGBoost from an uploaded CSV."""
    if not file.filename.endswith(".csv"):
        raise HTTPException(status_code=400, detail="Only CSV files are accepted.")

    with tempfile.NamedTemporaryFile(suffix=".csv", delete=False) as tmp:
        tmp.write(await file.read())
        tmp_path = tmp.name

    try:
        df           = pd.read_csv(tmp_path)
        processed_df = preprocessor.preprocess(df, is_training=True)
        augmented_df = classifier.generate_synthetic_data(processed_df)

        X_test, y_test, _ = classifier.train(augmented_df)
        metrics           = classifier.evaluate(X_test, y_test)

        assert metrics["accuracy"]  >= 0.80
        assert metrics["precision"] >= 0.75
        assert metrics["recall"]    >= 0.80

        classifier.save()
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


# ── feature importance chart ─────────────────────────────────
@app.get("/feature-importance")
def get_feature_importance():
    if not os.path.exists(FEATURE_IMPORTANCE_PATH):
        raise HTTPException(status_code=404, detail="Chart not found. Train the model first.")
    return FileResponse(FEATURE_IMPORTANCE_PATH, media_type="image/png")


# ── helpers ───────────────────────────────────────────────────
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
    bars    = ax.barh(
        [features[i] for i in indices],
        importances[indices],
        color="#4F81BD",
    )
    ax.set_xlabel("Importance (gain)", fontsize=12)
    ax.set_title("XGBoost Feature Importance — Failure Classification",
                 fontsize=14, fontweight="bold")
    ax.bar_label(bars, fmt="%.3f", padding=4, fontsize=9)
    plt.tight_layout()

    os.makedirs(os.path.dirname(FEATURE_IMPORTANCE_PATH), exist_ok=True)
    plt.savefig(FEATURE_IMPORTANCE_PATH, dpi=150)
    plt.close(fig)


# ── Entry point ──────────────────────────────────────────────
if __name__ == "__main__":
    host = os.getenv("SERVICE_HOST", "127.0.0.1")
    uvicorn.run(app, host=host, port=SERVICE_PORT)