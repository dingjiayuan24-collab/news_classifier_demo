"""
compare.py — Evaluate both saved models and print a side-by-side comparison.

Usage:
    python compare.py

Requires both models to be trained first:
    python train.py
"""

import os
import pickle
import textwrap

import numpy as np
import torch
from sklearn.metrics import accuracy_score, classification_report

from transformers import AutoTokenizer, AutoModelForSequenceClassification

from utils import (
    load_ag_news,
    preprocess,
    LABEL_NAMES,
    MODEL_DIR,
    LR_MODEL_PATH,
    BERT_MODEL_PATH,
)


# ─────────────────────────────────────────────────────────────
# Model loaders
# ─────────────────────────────────────────────────────────────

def load_lr_model():
    if not os.path.exists(LR_MODEL_PATH):
        raise FileNotFoundError(
            f"Model A not found at '{LR_MODEL_PATH}'. "
            "Run 'python train.py' first."
        )
    with open(LR_MODEL_PATH, "rb") as fh:
        return pickle.load(fh)


def load_bert_model():
    if not os.path.exists(BERT_MODEL_PATH):
        raise FileNotFoundError(
            f"Model B not found at '{BERT_MODEL_PATH}'. "
            "Run 'python train.py' first."
        )
    tokenizer = AutoTokenizer.from_pretrained(BERT_MODEL_PATH)
    model = AutoModelForSequenceClassification.from_pretrained(BERT_MODEL_PATH)
    model.eval()
    return model, tokenizer


# ─────────────────────────────────────────────────────────────
# Batch prediction for DistilBERT
# ─────────────────────────────────────────────────────────────

def predict_bert_batch(
    model,
    tokenizer,
    texts: list[str],
    batch_size: int = 64,
) -> list[int]:
    """Run inference on a list of texts and return predicted label indices."""
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = model.to(device)

    all_preds = []
    for i in range(0, len(texts), batch_size):
        batch = texts[i : i + batch_size]
        inputs = tokenizer(
            batch,
            return_tensors="pt",
            truncation=True,
            max_length=128,
            padding=True,
        )
        inputs = {k: v.to(device) for k, v in inputs.items()}

        with torch.no_grad():
            logits = model(**inputs).logits

        preds = logits.argmax(dim=-1).cpu().numpy().tolist()
        all_preds.extend(preds)

    return all_preds


# ─────────────────────────────────────────────────────────────
# Pretty-print comparison
# ─────────────────────────────────────────────────────────────

def print_comparison(
    lr_acc: float,
    bert_acc: float,
    lr_report: str,
    bert_report: str,
) -> None:
    sep = "=" * 64

    print(f"\n{sep}")
    print("              MODEL COMPARISON RESULTS")
    print(sep)

    print(f"\n{'Model':<38} {'Accuracy':>8}")
    print("-" * 48)
    print(f"{'Model A  (TF-IDF + Logistic Regression)':<38} {lr_acc:>7.4f}")
    print(f"{'Model B  (DistilBERT fine-tuned)':<38} {bert_acc:>7.4f}")
    print("-" * 48)

    diff = bert_acc - lr_acc
    if abs(diff) < 0.001:
        verdict = "Both models perform about the same."
    elif diff > 0:
        verdict = f"Model B (DistilBERT) wins by {diff:+.4f} accuracy."
    else:
        verdict = f"Model A (Logistic Regression) wins by {-diff:+.4f} accuracy."

    print(f"\nVerdict: {verdict}")

    print(f"\n{'─'*64}")
    print("Model A — per-class report")
    print(f"{'─'*64}")
    print(lr_report)

    print(f"{'─'*64}")
    print("Model B — per-class report")
    print(f"{'─'*64}")
    print(bert_report)
    print(sep)


# ─────────────────────────────────────────────────────────────
# Entry point
# ─────────────────────────────────────────────────────────────

def main() -> None:
    # Load test data (use 1 000 samples for a stable evaluation)
    _, _, test_texts, test_labels = load_ag_news(train_size=100, test_size=1000)
    test_texts = preprocess(test_texts)

    print("\nLoading saved models ...")
    lr_model = load_lr_model()
    bert_model, bert_tokenizer = load_bert_model()

    # ── Model A predictions ───────────────────────────────
    print("Evaluating Model A ...")
    lr_preds = lr_model.predict(test_texts)
    lr_acc = accuracy_score(test_labels, lr_preds)
    lr_report = classification_report(
        test_labels, lr_preds, target_names=LABEL_NAMES
    )

    # ── Model B predictions ───────────────────────────────
    print("Evaluating Model B ...")
    bert_preds = predict_bert_batch(bert_model, bert_tokenizer, test_texts)
    bert_acc = accuracy_score(test_labels, bert_preds)
    bert_report = classification_report(
        test_labels, bert_preds, target_names=LABEL_NAMES
    )

    print_comparison(lr_acc, bert_acc, lr_report, bert_report)


if __name__ == "__main__":
    main()
