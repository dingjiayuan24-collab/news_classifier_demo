"""
utils.py — Shared utilities: data loading, text cleaning, label mapping.
"""

import re
import os
from datasets import load_dataset

# AG News has 4 news categories
LABEL_NAMES = ["World", "Sports", "Business", "Sci/Tech"]
NUM_LABELS = len(LABEL_NAMES)

# Directory for saved model artifacts
MODEL_DIR = "models"
LR_MODEL_PATH = os.path.join(MODEL_DIR, "lr_model.pkl")
BERT_MODEL_PATH = os.path.join(MODEL_DIR, "distilbert")


def load_ag_news(train_size: int = 3000, test_size: int = 1000):
    """
    Load a small subset of the AG News dataset from Hugging Face.

    AG News is a 4-class news topic classification dataset.
    We use a small subset to keep training fast.

    Returns:
        (train_texts, train_labels, test_texts, test_labels)
    """
    print("Loading AG News dataset from Hugging Face...")
    dataset = load_dataset("ag_news", trust_remote_code=True)

    train = dataset["train"].shuffle(seed=42).select(range(train_size))
    test = dataset["test"].shuffle(seed=42).select(range(test_size))

    train_texts = list(train["text"])
    train_labels = list(train["label"])
    test_texts = list(test["text"])
    test_labels = list(test["label"])

    print(f"  Train samples : {len(train_texts)}")
    print(f"  Test  samples : {len(test_texts)}")
    return train_texts, train_labels, test_texts, test_labels


def clean_text(text: str) -> str:
    """Normalise whitespace and strip leading/trailing spaces."""
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def preprocess(texts: list[str]) -> list[str]:
    """Apply clean_text to every item in a list."""
    return [clean_text(t) for t in texts]


def label_name(idx: int) -> str:
    """Convert a numeric label to its human-readable category name."""
    return LABEL_NAMES[idx] if 0 <= idx < NUM_LABELS else "Unknown"
