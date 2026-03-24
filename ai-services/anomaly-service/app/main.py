from contextlib import asynccontextmanager
from fastapi import FastAPI, HTTPException
import pandas as pd
import numpy as np
import joblib
import io
import tensorflow as tf
from minio import Minio

from app.config import MINIO_ENDPOINT, MINIO_ACCESS_KEY, MINIO_SECRET_KEY, MINIO_BUCKET, MODEL_DIR
from app.schemas import AnalyzeRequest, AnalyzeResponse, AnomalyItem, HealthResponse
from app.scorer import compute_anomaly_score

# Charger les modeles au demarrage
@asynccontextmanager
async def lifespan(app: FastAPI):
    try:
        app.state.models = {
            "iso_forest":  joblib.load(f"{MODEL_DIR}/isolation_forest.pkl"),
            "scaler":      joblib.load(f"{MODEL_DIR}/scaler.pkl"),
            "sensor_cols": joblib.load(f"{MODEL_DIR}/sensor_cols.pkl"),
            "threshold":   joblib.load(f"{MODEL_DIR}/lstm_threshold.pkl"),
            "if_bounds":   joblib.load(f"{MODEL_DIR}/if_bounds.pkl"),
            "autoencoder": tf.keras.models.load_model(f"{MODEL_DIR}/lstm_autoencoder.keras"),
        }
        app.state.models_loaded = True
        print("Modeles charges avec succes")
    except Exception as e:
        print(f"Erreur chargement modeles: {e}")
        app.state.models_loaded = False
    yield

app = FastAPI(title="Anomaly Detection Service", version="1.0", lifespan=lifespan)

# MinIO client
def get_minio():
    return Minio(
        MINIO_ENDPOINT,
        access_key=MINIO_ACCESS_KEY,
        secret_key=MINIO_SECRET_KEY,
        secure=False
    )

@app.get("/health", response_model=HealthResponse)
def health():
    return HealthResponse(
        status="ok",
        models_loaded=app.state.models_loaded
    )

@app.post("/detect-anomalies", response_model=AnalyzeResponse)
def detect_anomalies(req: AnalyzeRequest):
    if not app.state.models_loaded:
        raise HTTPException(500, "Modeles non charges")

    try:
        # 1. Telecharger CSV depuis MinIO
        minio_client = get_minio()
        response     = minio_client.get_object(MINIO_BUCKET, req.csvPath)
        csv_bytes    = response.read()

        # 2. Parser le CSV
        df = pd.read_csv(io.BytesIO(csv_bytes))
        sensor_cols = app.state.models["sensor_cols"]
        available   = [c for c in sensor_cols if c in df.columns]

        if len(available) < 5:
            raise HTTPException(400, "CSV : pas assez de colonnes capteurs")

        if "timestamp" not in df.columns:
            raise HTTPException(400, "CSV : colonne timestamp manquante")

        timestamps = df["timestamp"].reset_index(drop=True)
        df[available] = df[available].ffill().bfill()

        scaler      = app.state.models["scaler"]
        df[available] = scaler.transform(df[available])
        sensor_data   = df[available].reset_index(drop=True)

        # 3. Calcul du score
        result = compute_anomaly_score(
            sensor_data, timestamps, req.claimDate, app.state.models
        )

        return AnalyzeResponse(
            score=result["score"],
            anomalies=[AnomalyItem(**a) for a in result["anomalies"]],
            pre_incident_anomaly=result["pre_incident_anomaly"],
            fraud_indicator=result["fraud_indicator"]
        )

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(500, f"Erreur analyse: {str(e)}")
@app.post("/test-local")
def test_local():
    """Test avec donnees synthetiques sans MinIO"""
    import numpy as np
    import pandas as pd

    sensor_cols_list = app.state.models["sensor_cols"]
    n = 500

    # Sequence NORMALE (fraude)
    normal_data = np.random.uniform(0.3, 0.6, (n, len(sensor_cols_list)))
    df_test = pd.DataFrame(normal_data, columns=sensor_cols_list)
    timestamps = pd.Series(pd.date_range("2018-04-01", periods=n, freq="1min"))

    from app.scorer import compute_anomaly_score
    result = compute_anomaly_score(df_test, timestamps, "2018-04-20", app.state.models)

    return {
        "test": "NORMAL_sequence",
        "result": result
    }
