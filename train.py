"""
train.py — Train both models on AG News and save artifacts.

Usage:
    python train.py                     # full training (3000 samples, 2 epochs)
    python train.py --quick             # quick training (500 samples, 1 epoch)

Model A : TF-IDF + Logistic Regression  →  models/lr_model.pkl
Model B : Fine-tuned DistilBERT         →  models/distilbert/
"""

import os
import pickle
import argparse

import numpy as np
import torch
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    TrainingArguments,
    Trainer,
    DataCollatorWithPadding,
)
from datasets import Dataset

from utils import (
    load_ag_news,
    preprocess,
    NUM_LABELS,
    MODEL_DIR,
    LR_MODEL_PATH,
    BERT_MODEL_PATH,
)

BERT_BASE = "distilbert-base-uncased"


# ─────────────────────────────────────────────────────────────
# Model A: TF-IDF + Logistic Regression
# ─────────────────────────────────────────────────────────────

def train_logistic_regression(train_texts: list, train_labels: list) -> Pipeline:
    """
    Build a TF-IDF feature extractor + Logistic Regression classifier.
    The sklearn Pipeline makes it easy to save and load as a single object.
    """
    print("\n[Model A] Training TF-IDF + Logistic Regression ...")

    pipeline = Pipeline([
        (
            "tfidf",
            TfidfVectorizer(
                max_features=30_000,
                ngram_range=(1, 2),   # unigrams + bigrams
                sublinear_tf=True,    # dampen high-frequency terms
            ),
        ),
        (
            "clf",
            LogisticRegression(
                max_iter=1000,
                C=5.0,
                solver="lbfgs",
                multi_class="multinomial",
                n_jobs=-1,
            ),
        ),
    ])

    pipeline.fit(train_texts, train_labels)
    print("[Model A] Training complete.")
    return pipeline


# ─────────────────────────────────────────────────────────────
# Model B: DistilBERT fine-tuning
# ─────────────────────────────────────────────────────────────

def train_distilbert(
    train_texts: list,
    train_labels: list,
    epochs: int = 2,
    batch_size: int = 16,
) -> None:
    """
    Fine-tune distilbert-base-uncased for sequence classification.
    The tokenizer + model weights are saved to BERT_MODEL_PATH.
    """
    print(f"\n[Model B] Fine-tuning DistilBERT for {epochs} epoch(s) ...")

    tokenizer = AutoTokenizer.from_pretrained(BERT_BASE)
    model = AutoModelForSequenceClassification.from_pretrained(
        BERT_BASE, num_labels=NUM_LABELS
    )

    # Build a Hugging Face Dataset and tokenise it
    def tokenize_batch(batch):
        return tokenizer(batch["text"], truncation=True, max_length=128)

    hf_train = Dataset.from_dict({"text": train_texts, "label": train_labels})
    hf_train = hf_train.map(tokenize_batch, batched=True, remove_columns=["text"])
    hf_train.set_format("torch")

    training_args = TrainingArguments(
        output_dir=BERT_MODEL_PATH,
        num_train_epochs=epochs,
        per_device_train_batch_size=batch_size,
        save_strategy="no",          # we save manually after training
        logging_steps=50,
        learning_rate=2e-5,
        weight_decay=0.01,
        fp16=torch.cuda.is_available(),
        report_to="none",            # disable wandb / tensorboard
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=hf_train,
        data_collator=DataCollatorWithPadding(tokenizer),
    )

    trainer.train()

    # Persist model and tokenizer so app.py can load them
    model.save_pretrained(BERT_MODEL_PATH)
    tokenizer.save_pretrained(BERT_MODEL_PATH)
    print(f"[Model B] Saved to {BERT_MODEL_PATH}/")


# ─────────────────────────────────────────────────────────────
# Entry point
# ─────────────────────────────────────────────────────────────

def main(quick: bool = False) -> None:
    os.makedirs(MODEL_DIR, exist_ok=True)

    train_size = 500 if quick else 3000
    epochs = 1 if quick else 2

    print(f"Mode: {'quick' if quick else 'full'} "
          f"| train_size={train_size} | distilbert_epochs={epochs}")

    # ── Load data ──────────────────────────────────────────
    train_texts, train_labels, _, _ = load_ag_news(
        train_size=train_size, test_size=100
    )
    train_texts = preprocess(train_texts)

    # ── Model A ────────────────────────────────────────────
    lr_pipeline = train_logistic_regression(train_texts, train_labels)
    with open(LR_MODEL_PATH, "wb") as fh:
        pickle.dump(lr_pipeline, fh)
    print(f"[Model A] Saved to {LR_MODEL_PATH}")

    # ── Model B ────────────────────────────────────────────
    train_distilbert(train_texts, train_labels, epochs=epochs)

    print("\nAll models trained and saved. Run compare.py to see evaluation results.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--quick",
        action="store_true",
        help="Use 500 training samples and 1 epoch (faster, for testing).",
    )
    args = parser.parse_args()
    main(quick=args.quick)
