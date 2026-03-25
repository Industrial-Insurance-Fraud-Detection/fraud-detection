import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification

from config import MODEL_PATH, MAX_LENGTH

# ── module-level references (populated by load_model) ────────
_tokenizer = None
_model     = None


def load_model():
    """
    Load tokenizer and model from MODEL_PATH.
    Called once at startup by the lifespan handler in main.py.
    Raises FileNotFoundError if the model folder does not exist.
    """
    global _tokenizer, _model

    import os
    if not os.path.exists(MODEL_PATH):
        raise FileNotFoundError(
            f"Model folder not found at '{MODEL_PATH}'. "
            "Make sure the BERT model files are present before starting the service."
        )

    print(f"Loading NLP model from {MODEL_PATH}...")
    _tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
    _model     = AutoModelForSequenceClassification.from_pretrained(MODEL_PATH)
    _model.eval()
    print("NLP model loaded successfully.")


def predict_fraud(text: str) -> dict:
    """
    Analyze text and return a fraud score 0-100.

    Returns:
        fraud_score:      float  0-100
        label:            str    FRAUD | LEGITIMATE
        confidence:       float  0-100
        fraud_prob:       float  0-100
        legitimate_prob:  float  0-100
    """
    if _tokenizer is None or _model is None:
        raise RuntimeError("Model not loaded. Call load_model() first.")

    inputs = _tokenizer(
        text,
        return_tensors="pt",
        truncation=True,
        max_length=MAX_LENGTH,
        padding=True,
    )

    with torch.no_grad():
        outputs = _model(**inputs)

    probs           = torch.softmax(outputs.logits, dim=1)[0].tolist()
    legitimate_prob = probs[0]
    fraud_prob      = probs[1]

    fraud_score = round(fraud_prob * 100, 1)
    confidence  = round(max(fraud_prob, legitimate_prob) * 100, 1)
    label       = "FRAUD" if fraud_score >= 50 else "LEGITIMATE"

    return {
        "fraud_score":     fraud_score,
        "label":           label,
        "confidence":      confidence,
        "fraud_prob":      round(fraud_prob * 100, 1),
        "legitimate_prob": round(legitimate_prob * 100, 1),
    }