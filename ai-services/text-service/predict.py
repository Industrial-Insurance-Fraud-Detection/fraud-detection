"""
predict.py
----------
This file does ONE job:
  - Load your trained BERT model from disk
  - Take some text as input
  - Return a fraud score between 0 and 100

Think of this as the BRAIN of your microservice.
"""

import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import os

# ─────────────────────────────────────────────
# 1. WHERE IS YOUR MODEL?
#    Put the path to your 4 files here.
#    (The folder that contains model.safetensors)
# ─────────────────────────────────────────────
MODEL_PATH = "./model"   # <-- change this if your folder has a different name

# ─────────────────────────────────────────────
# 2. LOAD THE MODEL ONCE WHEN THE APP STARTS
#    (Loading is slow, we don't want to do it
#     every time someone sends a request)
# ─────────────────────────────────────────────
print("⏳ Loading NLP model... please wait...")

tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
model     = AutoModelForSequenceClassification.from_pretrained(MODEL_PATH)
model.eval()  # put model in "read-only" mode (not training anymore)

print("✅ Model loaded successfully!")

# ─────────────────────────────────────────────
# 3. THE PREDICTION FUNCTION
#    Input:  text (a string)
#    Output: a score from 0 to 100
#            + a label "FRAUD" or "LEGITIMATE"
#            + a confidence percentage
# ─────────────────────────────────────────────
def predict_fraud(text: str) -> dict:
    """
    Takes a piece of text (maintenance report or claim description)
    and returns how fraudulent it looks, on a scale of 0 to 100.
    
    0   = definitely NOT fraud
    100 = definitely IS fraud
    """

    # Step A: Convert text into numbers the model understands
    # (this is called "tokenization" — like turning words into codes)
    inputs = tokenizer(
        text,
        return_tensors="pt",      # "pt" means PyTorch format
        truncation=True,          # cut off if text is too long
        max_length=512,           # BERT can only read 512 words at a time
        padding=True
    )

    # Step B: Ask the model to analyze the text
    with torch.no_grad():         # don't waste memory calculating gradients
        outputs = model(**inputs)

    # Step C: Convert the model's raw numbers into probabilities
    # softmax turns any numbers into percentages that add up to 100%
    probabilities = torch.softmax(outputs.logits, dim=1)
    probabilities = probabilities[0].tolist()  # convert to a simple Python list

    # Step D: Figure out which class is which
    # Your model has 2 classes: 0 = LEGITIMATE, 1 = FRAUD
    # (check your model's config.json to confirm the order)
    legitimate_prob = probabilities[0]   # probability it's a real claim
    fraud_prob      = probabilities[1]   # probability it's fraud

    # Step E: Convert fraud probability to a 0-100 score
    fraud_score = round(fraud_prob * 100, 1)

    # Step F: Create a human-readable label
    label = "FRAUD" if fraud_score >= 50 else "LEGITIMATE"
    confidence = round(max(fraud_prob, legitimate_prob) * 100, 1)

    return {
        "fraud_score":      fraud_score,     # e.g. 89.3
        "label":            label,            # "FRAUD" or "LEGITIMATE"
        "confidence":       confidence,       # e.g. 89.3%
        "fraud_prob":       round(fraud_prob * 100, 1),
        "legitimate_prob":  round(legitimate_prob * 100, 1),
    }


# ─────────────────────────────────────────────
# QUICK TEST — run this file directly to test
# python predict.py
# ─────────────────────────────────────────────
if __name__ == "__main__":
    test_text_1 = """
    The compressor experienced a sudden failure with no prior warning.
    All maintenance was performed regularly and on schedule.
    The machine was in perfect condition the day before the incident.
    """

    test_text_2 = """
    Gradual increase in vibration observed over 3 weeks before failure.
    Temperature readings showed steady rise from 65°C to 89°C.
    Maintenance team was alerted but repair was delayed due to budget.
    Final breakdown occurred after 23 days of documented deterioration.
    """

    print("=== TEST 1 (suspicious text) ===")
    result = predict_fraud(test_text_1)
    print(f"Score:  {result['fraud_score']} / 100")
    print(f"Label:  {result['label']}")
    print(f"Confidence: {result['confidence']}%")

    print("\n=== TEST 2 (genuine text) ===")
    result = predict_fraud(test_text_2)
    print(f"Score:  {result['fraud_score']} / 100")
    print(f"Label:  {result['label']}")
    print(f"Confidence: {result['confidence']}%")
