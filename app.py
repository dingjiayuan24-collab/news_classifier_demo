"""
app.py — Gradio demo: compare TF-IDF + LogReg vs DistilBERT on AG News.

Run locally:
    python app.py

Deploy to Hugging Face Spaces:
    Commit this file (and the models/ directory) — Spaces will run it automatically.

If no trained models are found, the app will auto-train a quick version
(~500 samples, 1 epoch) before launching. This takes a few minutes on first run.
"""

import os
import pickle

import numpy as np
import torch
import gradio as gr
from transformers import AutoTokenizer, AutoModelForSequenceClassification

from utils import (
    clean_text,
    label_name,
    LABEL_NAMES,
    NUM_LABELS,
    LR_MODEL_PATH,
    BERT_MODEL_PATH,
)

# ─────────────────────────────────────────────────────────────
# Global model handles (loaded once at startup)
# ─────────────────────────────────────────────────────────────
_lr_model = None
_bert_model = None
_bert_tokenizer = None


def _ensure_models_exist() -> None:
    """If models haven't been trained yet, run a quick training pass."""
    lr_missing = not os.path.exists(LR_MODEL_PATH)
    bert_missing = not os.path.exists(BERT_MODEL_PATH)

    if lr_missing or bert_missing:
        print(
            "Trained models not found — running quick training "
            "(500 samples, 1 epoch). This may take a few minutes …"
        )
        # Import here to avoid circular dependency at module level
        from train import main as run_training
        run_training(quick=True)


def load_models() -> None:
    """Load both models into global variables."""
    global _lr_model, _bert_model, _bert_tokenizer

    _ensure_models_exist()

    with open(LR_MODEL_PATH, "rb") as fh:
        _lr_model = pickle.load(fh)

    _bert_tokenizer = AutoTokenizer.from_pretrained(BERT_MODEL_PATH)
    _bert_model = AutoModelForSequenceClassification.from_pretrained(BERT_MODEL_PATH)
    _bert_model.eval()

    print("Both models loaded successfully.")


# ─────────────────────────────────────────────────────────────
# Prediction helpers
# ─────────────────────────────────────────────────────────────

def _predict_lr(text: str):
    """Return (predicted_label_index, probability_array) for Model A."""
    probs = _lr_model.predict_proba([text])[0]
    return int(np.argmax(probs)), probs


def _predict_bert(text: str):
    """Return (predicted_label_index, probability_array) for Model B."""
    device = "cuda" if torch.cuda.is_available() else "cpu"
    _bert_model.to(device)

    inputs = _bert_tokenizer(
        text, return_tensors="pt", truncation=True, max_length=128, padding=True
    )
    inputs = {k: v.to(device) for k, v in inputs.items()}

    with torch.no_grad():
        logits = _bert_model(**inputs).logits

    probs = torch.softmax(logits, dim=-1)[0].cpu().numpy()
    return int(np.argmax(probs)), probs


def _format_probs(probs: np.ndarray) -> str:
    """Render a simple ASCII bar chart of class probabilities."""
    lines = []
    for name, p in zip(LABEL_NAMES, probs):
        bar = "█" * int(round(p * 24))
        lines.append(f"{name:<12} {bar:<24} {p:5.1%}")
    return "\n".join(lines)


# ─────────────────────────────────────────────────────────────
# Main prediction function (called by Gradio)
# ─────────────────────────────────────────────────────────────

def classify(text: str):
    """
    Classify text with both models and return three outputs:
        - Model A result (string)
        - Model B result (string)
        - Comparison summary (string)
    """
    text = clean_text(text)

    if not text:
        empty = "⚠️  请在上方输入文本。"
        return empty, empty, ""

    lr_idx, lr_probs = _predict_lr(text)
    bert_idx, bert_probs = _predict_bert(text)

    lr_label = LABEL_NAMES[lr_idx]
    bert_label = LABEL_NAMES[bert_idx]

    lr_out = (
        f"预测类别：{lr_label}\n"
        f"置信度  ：{lr_probs[lr_idx]:.1%}\n\n"
        f"{_format_probs(lr_probs)}"
    )

    bert_out = (
        f"预测类别：{bert_label}\n"
        f"置信度  ：{bert_probs[bert_idx]:.1%}\n\n"
        f"{_format_probs(bert_probs)}"
    )

    if lr_idx == bert_idx:
        comparison = (
            f"✅  两个模型预测一致：**{lr_label}**\n\n"
            f"模型 A 置信度：{lr_probs[lr_idx]:.1%}　｜　"
            f"模型 B 置信度：{bert_probs[bert_idx]:.1%}"
        )
    else:
        more_confident = (
            "模型 B（DistilBERT）"
            if bert_probs[bert_idx] > lr_probs[lr_idx]
            else "模型 A（逻辑回归）"
        )
        comparison = (
            f"⚡ 两个模型预测不一致\n\n"
            f"模型 A → **{lr_label}**（{lr_probs[lr_idx]:.1%}）\n"
            f"模型 B → **{bert_label}**（{bert_probs[bert_idx]:.1%}）\n\n"
            f"置信度更高的模型：{more_confident}"
        )

    return lr_out, bert_out, comparison


# ─────────────────────────────────────────────────────────────
# Build Gradio UI
# ─────────────────────────────────────────────────────────────

EXAMPLES = [
    ["Tesla reports record quarterly profits, beating Wall Street estimates."],
    ["The Lakers defeated the Warriors 112-98 in Game 5 of the playoff series."],
    ["Scientists discover a new exoplanet that may support liquid water."],
    ["World leaders gather in Geneva for emergency climate summit."],
    ["Amazon acquires robotics startup for $1.2 billion to boost warehouse automation."],
]

with gr.Blocks(
    title="新闻分类器：逻辑回归 vs DistilBERT",
    theme=gr.themes.Soft(),
) as demo:

    gr.Markdown(
        """
        # 📰 新闻类别分类器
        ### 模型 A（TF-IDF + 逻辑回归）vs 模型 B（DistilBERT 微调）

        输入一条新闻标题或短文，两个模型将同时给出预测结果。  
        分类范围（AG News 数据集，共四类）：**国际 · 体育 · 商业 · 科技**
        """
    )

    with gr.Row():
        text_input = gr.Textbox(
            label="输入文本",
            placeholder="在此粘贴新闻标题或句子……",
            lines=3,
            scale=4,
        )

    classify_btn = gr.Button("🔍  开始分类", variant="primary", size="lg")

    with gr.Row():
        with gr.Column():
            gr.Markdown("### 🅰 模型 A — TF-IDF + 逻辑回归")
            lr_output = gr.Textbox(
                label="预测结果与各类概率",
                lines=8,
                interactive=False,
            )

        with gr.Column():
            gr.Markdown("### 🅱 模型 B — DistilBERT（微调）")
            bert_output = gr.Textbox(
                label="预测结果与各类概率",
                lines=8,
                interactive=False,
            )

    comparison_output = gr.Markdown("*分类结果将在点击按钮后显示于此处。*")

    gr.Examples(examples=EXAMPLES, inputs=text_input, label="示例输入（点击即可填入）")

    # 按钮点击与回车键均触发分类
    classify_btn.click(
        classify,
        inputs=text_input,
        outputs=[lr_output, bert_output, comparison_output],
    )
    text_input.submit(
        classify,
        inputs=text_input,
        outputs=[lr_output, bert_output, comparison_output],
    )

    gr.Markdown(
        """
        ---
        **数据集：** [AG News](https://huggingface.co/datasets/ag_news)（四分类英文新闻语料）  
        **模型 A：** scikit-learn TF-IDF 特征提取 + 多项式逻辑回归  
        **模型 B：** `distilbert-base-uncased` 微调 2 个 epoch  
        """
    )


# ─────────────────────────────────────────────────────────────
# Startup & launch
# ─────────────────────────────────────────────────────────────

load_models()

if __name__ == "__main__":
    demo.launch()
