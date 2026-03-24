import numpy as np
import pandas as pd
from datetime import datetime, timedelta

WINDOW      = 1440
WINDOW_LSTM = 240
STRIDE      = 60

def compute_anomaly_score(sensor_data, timestamps, claim_date_str, models):
    iso_forest   = models["iso_forest"]
    autoencoder  = models["autoencoder"]
    threshold    = models["threshold"]
    if_bounds    = models["if_bounds"]
    sensor_cols  = models["sensor_cols"]

    data = sensor_data.values

    # Isolation Forest
    features = []
    for i in range(0, len(data) - WINDOW, WINDOW):
        w = data[i:i+WINDOW]
        feat = np.concatenate([
            w.mean(axis=0), w.std(axis=0),
            w.min(axis=0),  w.max(axis=0),
            (w[-1] - w[0])
        ])
        features.append(feat)

    if len(features) == 0:
        feat = np.concatenate([
            data.mean(axis=0), data.std(axis=0),
            data.min(axis=0),  data.max(axis=0),
            (data[-1] - data[0])
        ])
        features = [feat]

    features      = np.array(features)
    if_scores_raw = iso_forest.score_samples(features)
    if_min        = if_bounds["if_min"]
    if_max        = if_bounds["if_max"]
    if_normalized = 100 * (1 - (if_scores_raw - if_min) / (if_max - if_min + 1e-9))
    if_score      = float(np.clip(if_normalized.mean(), 0, 100))

    # LSTM + anomalies
    seqs, seq_indices = [], []
    for i in range(0, len(data) - WINDOW_LSTM, STRIDE):
        seqs.append(data[i:i+WINDOW_LSTM])
        seq_indices.append(i)

    anomalies        = []
    lstm_score_global = 0.0

    if len(seqs) > 0:
        seqs_arr = np.array(seqs)
        recs     = autoencoder.predict(seqs_arr, verbose=0)
        mse_per_seq = np.mean(np.power(seqs_arr - recs, 2), axis=(1, 2))
        lstm_norm   = np.clip(mse_per_seq / threshold, 0, 3) / 3 * 100
        lstm_score_global = float(lstm_norm.mean())

        for idx, mse_val in enumerate(mse_per_seq):
            if mse_val > threshold:
                start        = seq_indices[idx]
                rec_errors   = np.mean(np.power(seqs_arr[idx] - recs[idx], 2), axis=0)
                worst_idx    = int(np.argmax(rec_errors))
                worst_name   = sensor_cols[worst_idx]
                worst_mean   = float(seqs_arr[idx][:, worst_idx].mean())
                ts           = timestamps.iloc[start] if start < len(timestamps) else str(start)
                anomalies.append({
                    "timestamp": str(ts),
                    "parameter": worst_name,
                    "value":     round(worst_mean, 4),
                    "threshold": round(float(threshold), 6)
                })

    # pre_incident_anomaly
    pre_incident_anomaly = False
    try:
        claim_date = datetime.strptime(claim_date_str, "%Y-%m-%d")
        cutoff     = claim_date - timedelta(days=7)
        ts_series  = pd.to_datetime(timestamps.values)
        mask       = (ts_series >= cutoff) & (ts_series <= claim_date)
        pre_data   = data[mask]

        if len(pre_data) >= WINDOW_LSTM:
            pre_seqs = np.array([
                pre_data[i:i+WINDOW_LSTM]
                for i in range(0, len(pre_data) - WINDOW_LSTM, STRIDE)
            ])
            pre_recs = autoencoder.predict(pre_seqs, verbose=0)
            pre_mse  = np.mean(np.power(pre_seqs - pre_recs, 2), axis=(1, 2))
            pre_incident_anomaly = bool((pre_mse > threshold).any())
    except Exception as e:
        print(f"Erreur pre_incident: {e}")

    final_score = int(np.clip(0.5 * if_score + 0.5 * lstm_score_global, 0, 100))

    # fraud_indicator
    if not pre_incident_anomaly and final_score < 30:
        fraud_indicator = "NO PRECURSOR DETECTED"
    elif not pre_incident_anomaly and final_score >= 30:
        fraud_indicator = "ABRUPT FAILURE"
    elif pre_incident_anomaly and len(anomalies) >= 3:
        fraud_indicator = "GRADUAL DEGRADATION"
    elif pre_incident_anomaly and len(anomalies) > 0:
        fraud_indicator = "MINOR PRECURSORS DETECTED"
    else:
        fraud_indicator = "NORMAL"

    return {
        "score":                final_score,
        "anomalies":            anomalies,
        "pre_incident_anomaly": pre_incident_anomaly,
        "fraud_indicator":      fraud_indicator
    }