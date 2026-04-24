"""
gradio_demo.py — Interactive Fake News Detector UI for FakeGuard.
XAI Edition: LIME word-level explanations + Explanation Summary Sentence added.

Source of truth:
    - instructions/09_features_and_routes.md
    - instructions/04_model_training_strategy.md
    - config.py

What this does:
    Launches a Gradio web interface where anyone can type a news article
    (title + body) and instantly get a REAL / FAKE / UNCERTAIN verdict
    with a confidence score AND a LIME explanation showing which words
    triggered the prediction, plus a plain-English summary sentence.

Features:
    - Single article prediction with confidence bar
    - Colour-coded verdict: 🟢 REAL | 🔴 FAKE | 🟡 UNCERTAIN
    - XAI: LIME word-level highlights (red = pushed toward FAKE, green = pushed toward REAL)
    - XAI: Plain-English explanation summary sentence (top 3 trigger words)
    - Batch prediction from CSV upload
    - Example articles pre-loaded (one real, one fake)
    - Public shareable URL via Gradio tunnel (share=True)
    - Graceful fallback if model not yet trained
    - Graceful fallback if LIME not installed (explanation skipped, prediction still works)

Usage:
    # Run locally or in Kaggle:
    python app/gradio_demo.py

    # Or from a notebook cell:
    from app.gradio_demo import launch_demo
    launch_demo(share=True)

IMPORTANT:
    Run src/train.py FIRST to get competition-grade accuracy.
    Without the trained model, this falls back to roberta-base
    base weights which will give random predictions.

    LIME must be installed for XAI:
        pip install lime
    If LIME is not installed, predictions still work — XAI section is skipped gracefully.
"""

import sys
import os
from pathlib import Path
from typing import Optional

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from config import (
    MODEL_DIR,
    TRANSFORMER,
    MAX_LEN,
    CONFIDENCE_THRESHOLD,
    ID2LABEL,
    SEED,
)
from src.utils import get_logger, set_seed

logger = get_logger("gradio_demo")


# ─────────────────────────────────────────────────────────────────────────────
# Lazy-loaded globals — model loads once on first prediction, not at import
_model     = None
_tokenizer = None
_device    = None


def _load_model_once():
    global _model, _tokenizer, _device
    if _model is not None:
        return

    import torch
    from transformers import RobertaForSequenceClassification, AutoTokenizer

    _device    = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    local_path = Path(MODEL_DIR) / "roberta_fakenews"

    if local_path.exists() and any(local_path.iterdir()):
        load_from = str(local_path)
        logger.info("✅ Loading fine-tuned model from: %s", load_from)
    else:
        load_from = TRANSFORMER
        logger.warning(
            "⚠️  Fine-tuned model not found at '%s'.\n"
            "   Loading base '%s' weights — predictions will NOT be accurate.\n"
            "   Run  python src/train.py  first for competition-grade accuracy.",
            local_path, TRANSFORMER,
        )

    _model     = RobertaForSequenceClassification.from_pretrained(load_from)
    _tokenizer = AutoTokenizer.from_pretrained(load_from)
    _model.to(_device)
    _model.eval()
    logger.info("Model loaded on: %s", _device)


# ─────────────────────────────────────────────────────────────────────────────
# XAI — LIME explainability
# ─────────────────────────────────────────────────────────────────────────────

def _lime_predict_proba(texts: list) -> "np.ndarray":
    import torch
    import torch.nn.functional as F
    import numpy as np

    _load_model_once()
    results = []

    batch_size = 8
    for i in range(0, len(texts), batch_size):
        batch = texts[i : i + batch_size]
        encoded = _tokenizer(
            batch,
            truncation=True,
            padding=True,
            max_length=MAX_LEN,
            return_tensors="pt",
        )
        with torch.no_grad():
            input_ids      = encoded["input_ids"].to(_device)
            attention_mask = encoded["attention_mask"].to(_device)
            outputs        = _model(input_ids=input_ids, attention_mask=attention_mask)
            probs          = F.softmax(outputs.logits, dim=-1).cpu().numpy()
        results.append(probs)

    return __import__("numpy").vstack(results)


def _generate_explanation_summary(word_weights: dict, pred_label: str) -> str:
    """
    Generate a plain-English summary sentence from LIME word weights.

    Example output:
        "This article was flagged as FAKE because it contained strong signals:
         'SHOCKING' (+0.312), 'leaked document' (+0.289), 'secret meeting' (+0.201)."

    Args:
        word_weights : dict of {word: weight} from LIME
        pred_label   : "FAKE" or "REAL"

    Returns:
        A plain-English string summarising the top 3 trigger words.
    """
    # Sort by absolute weight descending, take top 3
    top_words = sorted(word_weights.items(), key=lambda x: abs(x[1]), reverse=True)[:3]

    if not top_words:
        return f"The model predicted {pred_label} based on overall language patterns."

    word_list = ", ".join(
        f"'{w}' ({'+' if s > 0 else ''}{s:.3f})"
        for w, s in top_words
    )

    if pred_label == "FAKE":
        return (
            f"⚠️ This article was flagged as FAKE because it contained "
            f"high-weight signals: {word_list}. "
            f"These words are strongly associated with fake news in the training data."
        )
    else:
        return (
            f"✅ This article was classified as REAL because it contained "
            f"credible language signals: {word_list}. "
            f"These words are strongly associated with real, verified news."
        )


def _generate_lime_explanation(combined_text: str, pred_label: str, n_samples: int = 200):
    """
    Generate LIME explanation — returns (html_string, summary_sentence).

    Returns:
        tuple: (lime_html: str, summary_sentence: str)
    """
    try:
        from lime.lime_text import LimeTextExplainer
        import numpy as np
    except ImportError:
        fallback = (
            "<p style='color:#888; font-style:italic;'>"
            "⚠️ LIME not installed — XAI unavailable. "
            "Run: <code>pip install lime</code> then restart."
            "</p>"
        )
        return fallback, "XAI summary unavailable — LIME not installed."

    explain_class = 0 if pred_label == "FAKE" else 1

    try:
        explainer = LimeTextExplainer(
            class_names=["FAKE", "REAL"],
            random_state=42,
        )
        explanation = explainer.explain_instance(
            combined_text,
            _lime_predict_proba,
            num_features=12,
            num_samples=n_samples,
            labels=[explain_class],
        )

        word_weights = dict(explanation.as_list(label=explain_class))

        # ── Plain-English Summary Sentence ──────────────────────────────────
        summary_sentence = _generate_explanation_summary(word_weights, pred_label)

        # ── HTML Word Highlights ─────────────────────────────────────────────
        words = combined_text.split()

        html_parts = [
            "<div style='line-height:2.2; font-size:0.95em; font-family:sans-serif;'>"
            "<p style='color:#555; font-size:0.85em; margin-bottom:8px;'>"
            "🔴 <b>Red</b> words pushed toward <b>FAKE</b> &nbsp;|&nbsp; "
            "🟢 <b>Green</b> words pushed toward <b>REAL</b>"
            "</p>"
        ]

        for word in words:
            clean  = word.strip(".,!?\"'()[]{}:;")
            weight = word_weights.get(clean, 0.0)

            if abs(weight) < 0.01:
                html_parts.append(f"<span>{word} </span>")
            elif weight > 0:
                intensity = min(int(abs(weight) * 800), 200)
                if pred_label == "FAKE":
                    color = f"rgba(220,50,50,0.{intensity:03d})"
                    html_parts.append(
                        f"<span style='background:{color}; padding:2px 4px; "
                        f"border-radius:3px; font-weight:600;' "
                        f"title='FAKE signal: +{weight:.3f}'>{word} </span>"
                    )
                else:
                    color = f"rgba(40,180,80,0.{intensity:03d})"
                    html_parts.append(
                        f"<span style='background:{color}; padding:2px 4px; "
                        f"border-radius:3px; font-weight:600;' "
                        f"title='REAL signal: +{weight:.3f}'>{word} </span>"
                    )
            else:
                intensity = min(int(abs(weight) * 800), 200)
                if pred_label == "FAKE":
                    color = f"rgba(40,180,80,0.{intensity:03d})"
                    html_parts.append(
                        f"<span style='background:{color}; padding:2px 4px; "
                        f"border-radius:3px;' "
                        f"title='REAL signal: {weight:.3f}'>{word} </span>"
                    )
                else:
                    color = f"rgba(220,50,50,0.{intensity:03d})"
                    html_parts.append(
                        f"<span style='background:{color}; padding:2px 4px; "
                        f"border-radius:3px;' "
                        f"title='FAKE signal: {weight:.3f}'>{word} </span>"
                    )

        html_parts.append("</div>")
        return "".join(html_parts), summary_sentence

    except Exception as e:
        logger.warning("LIME explanation failed: %s", e)
        err_html = f"<p style='color:#888;'>XAI explanation unavailable: {e}</p>"
        return err_html, f"XAI summary unavailable: {e}"


# ─────────────────────────────────────────────────────────────────────────────
def predict_single(title: str, body: str, run_xai: bool = True) -> dict:
    import torch
    import torch.nn.functional as F

    title = (title or "").strip()
    body  = (body  or "").strip()

    if not title and not body:
        return {
            "verdict":          "🟡 PLEASE ENTER TEXT",
            "confidence":       0.0,
            "label":            "N/A",
            "uncertain":        True,
            "explanation":      "Please enter a headline and/or article body above.",
            "lime_html":        "",
            "summary_sentence": "",
        }

    if title and body:
        combined = f"{title} [SEP] {body}"
    elif title:
        combined = title
    else:
        combined = body

    display_text = combined[:200] + "..." if len(combined) > 200 else combined

    _load_model_once()

    encoded = _tokenizer(
        combined,
        truncation=True,
        padding=True,
        max_length=MAX_LEN,
        return_tensors="pt",
    )

    with torch.no_grad():
        input_ids      = encoded["input_ids"].to(_device)
        attention_mask = encoded["attention_mask"].to(_device)
        outputs        = _model(input_ids=input_ids, attention_mask=attention_mask)
        probs          = __import__("torch").nn.functional.softmax(
                             outputs.logits, dim=-1
                         ).cpu().numpy()[0]

    pred_id    = int(probs.argmax())
    confidence = float(probs.max())
    label      = ID2LABEL.get(pred_id, str(pred_id))
    uncertain  = confidence < CONFIDENCE_THRESHOLD

    fake_prob = float(probs[0])
    real_prob = float(probs[1])

    if uncertain:
        verdict = f"🟡 UNCERTAIN ({confidence:.1%} confidence — below {CONFIDENCE_THRESHOLD:.0%} threshold)"
        explanation = (
            f"The model is not confident enough to make a reliable prediction.\n"
            f"FAKE probability: {fake_prob:.1%} | REAL probability: {real_prob:.1%}\n"
            f"Threshold: {CONFIDENCE_THRESHOLD:.0%} — human review recommended."
        )
    elif label == "FAKE":
        verdict = f"🔴 FAKE NEWS ({confidence:.1%} confidence)"
        explanation = (
            f"This article is likely FAKE news.\n"
            f"FAKE probability: {fake_prob:.1%} | REAL probability: {real_prob:.1%}\n"
            f"Analysed text: \"{display_text}\""
        )
    else:
        verdict = f"🟢 REAL NEWS ({confidence:.1%} confidence)"
        explanation = (
            f"This article appears to be REAL news.\n"
            f"FAKE probability: {fake_prob:.1%} | REAL probability: {real_prob:.1%}\n"
            f"Analysed text: \"{display_text}\""
        )

    lime_html        = ""
    summary_sentence = ""

    if run_xai and not uncertain:
        lime_html, summary_sentence = _generate_lime_explanation(
            combined_text=combined,
            pred_label=label,
            n_samples=200,
        )
    elif uncertain:
        lime_html        = (
            "<p style='color:#888; font-style:italic;'>"
            "XAI unavailable for UNCERTAIN predictions — confidence too low."
            "</p>"
        )
        summary_sentence = "Confidence too low for XAI summary."

    return {
        "verdict":          verdict,
        "confidence":       round(confidence, 4),
        "label":            label,
        "uncertain":        uncertain,
        "fake_prob":        round(fake_prob, 4),
        "real_prob":        round(real_prob, 4),
        "explanation":      explanation,
        "lime_html":        lime_html,
        "summary_sentence": summary_sentence,
    }


def _gradio_predict(title: str, body: str):
    """
    Gradio interface function — returns 5 values:
    (verdict, explanation, confidence, summary_sentence, lime_html)
    """
    result = predict_single(title, body, run_xai=True)
    return (
        result["verdict"],
        result["explanation"],
        result["confidence"],
        result["summary_sentence"],
        result["lime_html"],
    )


# ─────────────────────────────────────────────────────────────────────────────
def launch_demo(share: bool = True, server_port: int = 7860) -> None:
    try:
        import gradio as gr
    except ImportError:
        logger.error("Gradio not installed. Run: pip install gradio>=4.0.0")
        raise

    examples = [
        [
            "Scientists discover water on the Moon's surface",
            "NASA researchers have confirmed the presence of water ice in permanently shadowed craters near the lunar south pole, opening possibilities for future human missions.",
        ],
        [
            "BREAKING: Government puts 5G chips in COVID vaccines to control minds",
            "A leaked document from a secret globalist meeting reveals that every COVID-19 vaccine contains a microscopic 5G receiver that allows remote control of human behavior.",
        ],
        [
            "Stock market reaches record high amid strong earnings reports",
            "The S&P 500 climbed to a new record high on Thursday following better-than-expected earnings from major technology companies and positive economic data.",
        ],
        [
            "SHOCKING: Celebrity fakes own death to avoid paying taxes",
            "Sources close to the situation claim the A-list celebrity staged their death last month in an elaborate scheme to escape millions in back taxes owed to the government.",
        ],
    ]

    with gr.Blocks(
        title="FakeGuard — AI Fake News Detector",
        theme=gr.themes.Soft(),
        css="""
            .verdict-box { font-size: 1.4em; font-weight: bold; padding: 12px; border-radius: 8px; }
            .footer-text { color: #888; font-size: 0.85em; text-align: center; margin-top: 8px; }
            .summary-box { background: #f0f7ff; border-left: 4px solid #2196F3; padding: 12px 16px;
                           border-radius: 6px; font-size: 0.95em; margin-top: 8px; }
        """,
    ) as demo:

        gr.Markdown(
            """
            # 🛡️ FakeGuard — AI Fake News Detector
            ### NeuroLogic '26 Datathon | RoBERTa + XAI Explainability
            Enter a news article headline and body to get an instant **REAL / FAKE** prediction
            — plus **word-level highlights** and a **plain-English explanation** of WHY the model decided.
            """
        )

        with gr.Row():
            with gr.Column(scale=3):
                title_input = gr.Textbox(
                    label="📰 Article Headline",
                    placeholder="Paste the news headline here...",
                    lines=2,
                )
                body_input = gr.Textbox(
                    label="📄 Article Body (optional but recommended)",
                    placeholder="Paste the article body text here...",
                    lines=8,
                )
                predict_btn = gr.Button("🔍 Analyse Article", variant="primary", size="lg")
                gr.Markdown(
                    "<p style='color:#888; font-size:0.85em;'>"
                    "⏱️ First prediction loads the model (~10s). "
                    "XAI word highlights add ~5-10s per prediction."
                    "</p>"
                )

            with gr.Column(scale=2):
                verdict_output = gr.Textbox(
                    label="🎯 Verdict",
                    lines=2,
                    elem_classes=["verdict-box"],
                    interactive=False,
                )
                confidence_output = gr.Slider(
                    label="Confidence Score",
                    minimum=0.0,
                    maximum=1.0,
                    value=0.0,
                    interactive=False,
                    info=f"Below {CONFIDENCE_THRESHOLD:.0%} = UNCERTAIN (needs human review)",
                )
                explanation_output = gr.Textbox(
                    label="📊 Detailed Breakdown",
                    lines=4,
                    interactive=False,
                )
                summary_output = gr.Textbox(
                    label="💡 XAI Summary — Why did the model decide this?",
                    lines=3,
                    interactive=False,
                    elem_classes=["summary-box"],
                )

        # XAI section — full width below
        gr.Markdown("---")
        gr.Markdown(
            "## 🔬 XAI Word Highlights\n"
            "The highlighted words below show the model's reasoning word-by-word. "
            "**Red** = pushed toward FAKE &nbsp;|&nbsp; **Green** = pushed toward REAL. "
            "Powered by LIME (Local Interpretable Model-agnostic Explanations)."
        )
        lime_output = gr.HTML(
            value="<p style='color:#aaa; font-style:italic;'>Run a prediction above to see word-level highlights here.</p>",
            label="XAI Word Highlights",
        )

        # Wire up button — 5 outputs now
        predict_btn.click(
            fn=_gradio_predict,
            inputs=[title_input, body_input],
            outputs=[verdict_output, explanation_output, confidence_output, summary_output, lime_output],
        )

        title_input.submit(
            fn=_gradio_predict,
            inputs=[title_input, body_input],
            outputs=[verdict_output, explanation_output, confidence_output, summary_output, lime_output],
        )

        gr.Examples(
            examples=examples,
            inputs=[title_input, body_input],
            label="📌 Click an example to try it",
            examples_per_page=4,
        )

        gr.Markdown(
            f"""
            ---
            **Model:** `roberta-base` fine-tuned on WELFake (~72k articles)
            **XAI methods:** LIME word highlights + Plain-English explanation summary
            **Uncertainty threshold:** {CONFIDENCE_THRESHOLD:.0%} confidence
            **Labels:** FAKE (0) | REAL (1)
            **Baseline (TF-IDF + LR):** 94.66% accuracy | **FakeGuard target:** ~97–98%
            <div class='footer-text'>NeuroLogic '26 Datathon — FakeGuard by Ayush Tiwari</div>
            """
        )

    logger.info("Launching Gradio demo (share=%s, port=%d)", share, server_port)
    demo.launch(
        share=share,
        server_port=server_port,
        show_error=True,
    )


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Launch FakeGuard Gradio demo")
    parser.add_argument("--no-share", action="store_true", help="Disable public sharing link")
    parser.add_argument("--port", type=int, default=7860, help="Local server port")
    args = parser.parse_args()

    set_seed(SEED)
    launch_demo(share=not args.no_share, server_port=args.port)
