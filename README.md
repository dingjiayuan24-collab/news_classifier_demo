---
title: News Classifier — LogReg vs DistilBERT
emoji: 📰
colorFrom: blue
colorTo: green
sdk: gradio
sdk_version: "4.7.0"
app_file: app.py
pinned: false
---

# 📰 News Category Classifier

A side-by-side comparison of two text classifiers trained on the
[AG News](https://huggingface.co/datasets/ag_news) dataset.

| | Model A | Model B |
|---|---|---|
| **Type** | TF-IDF + Logistic Regression | DistilBERT (fine-tuned) |
| **Library** | scikit-learn | Hugging Face Transformers |
| **Categories** | World · Sports · Business · Sci/Tech | same |

---

## Project Structure

```
hugging_face_spaces_demo/
├── app.py           # Gradio demo (entry point for HF Spaces)
├── train.py         # Train both models and save artifacts
├── compare.py       # Evaluate saved models, print comparison table
├── utils.py         # Shared helpers: data loading, text cleaning
├── requirements.txt
├── README.md
└── models/          # Created by train.py (not committed by default)
    ├── lr_model.pkl         # Model A weights
    └── distilbert/          # Model B weights + tokenizer
```

---

## Running Locally

### 1 — Install dependencies

```bash
pip install -r requirements.txt
```

### 2 — Train both models

**Full training** (3 000 samples, 2 epochs — recommended for good accuracy):

```bash
python train.py
```

**Quick training** (500 samples, 1 epoch — for testing the pipeline fast):

```bash
python train.py --quick
```

Training artifacts are saved to `models/`.

### 3 — Compare model performance

```bash
python compare.py
```

This evaluates both models on 1 000 held-out test samples and prints
accuracy + per-class precision/recall/F1.

Example output:

```
================================================================
              MODEL COMPARISON RESULTS
================================================================

Model                                  Accuracy
------------------------------------------------
Model A  (TF-IDF + Logistic Regression)   0.9060
Model B  (DistilBERT fine-tuned)          0.9350

Verdict: Model B (DistilBERT) wins by +0.0290 accuracy.
```

### 4 — Launch the Gradio app

```bash
python app.py
```

Then open `http://localhost:7860` in your browser.

> **First-run note:** If you skipped step 2, the app will automatically
> run a quick training pass before launching (takes a few minutes).

---

## Deploying to Hugging Face Spaces

### Option A — Push pre-trained models (recommended)

1. [Create a new Space](https://huggingface.co/new-space) and choose **Gradio** as the SDK.
2. Clone your Space repository:
   ```bash
   git clone https://huggingface.co/spaces/<your-username>/<your-space-name>
   ```
3. Copy all project files into the cloned repo.
4. Train locally and copy the `models/` directory into the repo as well.
5. Commit and push:
   ```bash
   git add .
   git commit -m "Add classifier demo with pre-trained models"
   git push
   ```
   Large files (> 10 MB) require [Git LFS](https://git-lfs.com/):
   ```bash
   git lfs install
   git lfs track "models/distilbert/*"
   git add .gitattributes
   ```

### Option B — Let the Space auto-train on first start

Skip copying `models/`. The app detects missing models and runs
`train.py --quick` on startup. This works on the free CPU tier but
adds ~3–5 minutes to the cold-start time.

---

## Dataset

**AG News** — 120 000 English news articles in four categories.  
Loaded directly from Hugging Face via `datasets.load_dataset("ag_news")`.  
Only a small subset is used to keep training fast.

## Model Details

| Detail | Model A | Model B |
|---|---|---|
| Features | TF-IDF (1–2 grams, 30 k vocab) | Contextual embeddings |
| Classifier | Multinomial Logistic Regression | Linear head over `[CLS]` token |
| Train time (3 k samples) | ~5 s | ~5–10 min (CPU) / ~1 min (GPU) |
| Typical accuracy | ~90 % | ~93 % |

## License

MIT

[##View on Kaggle
]([https://www.kaggle.com/code/%3C%E4%BD%A0%E7%9A%84%E7%94%A8%E6%88%B7%E5%90%8D%3E/%3Cnotebook%E5%90%8D%3E](https://www.kaggle.com/code/jiayuan24ding/notebooka4117ea469))
https://www.kaggle.com/code/jiayuan24ding/notebooka4117ea469
