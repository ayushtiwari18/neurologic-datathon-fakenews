"""
gradio_demo.py — Interactive Fake News Detector UI for FakeGuard.
XAI Edition: LIME word-level explanations added for judge Innovation + Impact scores.

Source of truth:
    - instructions/09_features_and_routes.md
    - instructions/04_model_training_strategy.md
    - config.py

What this does:
    Launches a Gradio web interface where anyone can type a news article
    (title + body) and instantly get a REAL / FAKE / UNCERTAIN verdict
    with a confidence score AND a LIME explanation showing which words
    triggered the prediction.

Features:
    - Single article prediction with confidence bar
    - Colour-coded verdict: 🟢 REAL | 🔴 FAKE | 🟡 UNCERTAIN
    - XAI: LIME word-level highlights (red = pushed toward FAKE, green = pushed toward REAL)
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
# This prevents slow startup and avoids OOM if model isn't needed yet.
_model     = None
_tokenizer = None
_device    = None


def _load_model_once():
    """
    Load model and tokenizer into global cache on first call.
    Subsequent calls return immediately (no re-loading).
    """
    global _model, _tokenizer, _device
    if _model is not None:
        return  # already loaded

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
    """
    Probability predictor for LIME.

    LIME works by masking random words in the input and asking:
    'how much did the prediction change?'
    Words that cause big changes = most important words.

    Args:
        texts: list of strings (LIME passes many masked variants here)

    Returns:
        np.ndarray of shape (n_samples, 2) — [prob_FAKE, prob_REAL]
    """
    import torch
    import torch.nn.functional as F
    import numpy as np

    _load_model_once()
    results = []

    # Process in small batches to avoid OOM
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


def _generate_lime_explanation(combined_text: str, pred_label: str, n_samples: int = 200) -> str:
    """
    Generate a LIME explanation as an HTML string showing highlighted words.

    How LIME works (beginner explanation):
    - LIME creates ~200 slightly modified versions of your article
      (randomly hiding some words each time)
    - It asks the model to predict each version
    - Words whose removal changes the prediction most = most important words
    - Red words pushed toward FAKE, green words pushed toward REAL

    Args:
        combined_text : The full article text (title [SEP] body)
        pred_label    : "FAKE" or "REAL" — the model's prediction
        n_samples     : How many masked variants LIME generates (more = slower but better)

    Returns:
        HTML string with colour-highlighted words, or a fallback message string.
    """
    try:
        from lime.lime_text import LimeTextExplainer
        import numpy as np
    except ImportError:
        return (
            "<p style='color:#888; font-style:italic;'>"
            "⚠️ LIME not installed — XAI unavailable. "
            "Run: <code>pip install lime</code> then restart."
            "</p>"
        )

    # Which class index are we explaining?
    # FAKE = 0, REAL = 1
    explain_class = 0 if pred_label == "FAKE" else 1

    try:
        explainer = LimeTextExplainer(
            class_names=["FAKE", "REAL"],
            random_state=42,
        )
        explanation = explainer.explain_instance(
            combined_text,
            _lime_predict_proba,
            num_features=12,          # show top 12 most important words
            num_samples=n_samples,    # 200 masked variants
            labels=[explain_class],
        )

        # Build colour-highlighted HTML
        # LIME returns (word, weight) pairs
        # positive weight = word supports this class
        # negative weight = word opposes this class
        word_weights = dict(explanation.as_list(label=explain_class))

        # Tokenise text by whitespace for highlighting
        words = combined_text.split()

        html_parts = [
            "<div style='line-height:2.2; font-size:0.95em; font-family:sans-serif;'>"
            "<p style='color:#555; font-size:0.85em; margin-bottom:8px;'>"
            "🔴 <b>Red</b> words pushed toward <b>FAKE</b> &nbsp;|&nbsp; "
            "🟢 <b>Green</b> words pushed toward <b>REAL</b>"
            "</p>"
        ]

        for word in words:
            # Strip punctuation for lookup but keep original for display
            clean = word.strip(".,!?\"'()[]{}:;")
            weight = word_weights.get(clean, 0.0)

            if abs(weight) < 0.01:
                # Unimportant word — plain text
                html_parts.append(f"<span>{word} </span>")
            elif weight > 0:
                # Supports the predicted class
                intensity = min(int(abs(weight) * 800), 200)
                if pred_label == "FAKE":
                    # Supporting FAKE → red
                    color = f"rgba(220,50,50,0.{intensity:03d})"
                    html_parts.append(
                        f"<span style='background:{color}; padding:2px 4px; "
                        f"border-radius:3px; font-weight:600;' "
                        f"title='FAKE signal: +{weight:.3f}'>{word} </span>"
                    )
                else:
                    # Supporting REAL → green
                    color = f"rgba(40,180,80,0.{intensity:03d})"
                    html_parts.append(
                        f"<span style='background:{color}; padding:2px 4px; "
                        f"border-radius:3px; font-weight:600;' "
                        f"title='REAL signal: +{weight:.3f}'>{word} </span>"
                    )
            else:
                # Opposes the predicted class
                intensity = min(int(abs(weight) * 800), 200)
                if pred_label == "FAKE":
                    # Opposing FAKE → green (pushes toward REAL)
                    color = f"rgba(40,180,80,0.{intensity:03d})"
                    html_parts.append(
                        f"<span style='background:{color}; padding:2px 4px; "
                        f"border-radius:3px;' "
                        f"title='REAL signal: {weight:.3f}'>{word} </span>"
                    )
                else:
                    # Opposing REAL → red (pushes toward FAKE)
                    color = f"rgba(220,50,50,0.{intensity:03d})"
                    html_parts.append(
                        f"<span style='background:{color}; padding:2px 4px; "
                        f"border-radius:3px;' "
                        f"title='FAKE signal: {weight:.3f}'>{word} </span>"
                    )

        html_parts.append("</div>")
        return "".join(html_parts)

    except Exception as e:
        logger.warning("LIME explanation failed: %s", e)
        return f"<p style='color:#888;'>XAI explanation unavailable: {e}</p>"


# ─────────────────────────────────────────────────────────────────────────────
def predict_single(title: str, body: str, run_xai: bool = True) -> dict:
    """
    Run inference on a single article (title + body).

    Combines title and body using the same [SEP] format as preprocessing.
    Returns verdict, confidence, detailed breakdown, and LIME HTML explanation.

    Args:
        title   : Article headline.
        body    : Article body text.
        run_xai : If True, generate LIME explanation (adds ~5-10 seconds).

    Returns:
        dict with keys: verdict, confidence, label, uncertain, explanation, lime_html
    """
    import torch
    import torch.nn.functional as F

    title = (title or "").strip()
    body  = (body  or "").strip()

    if not title and not body:
        return {
            "verdict":     "🟡 PLEASE ENTER TEXT",
            "confidence":  0.0,
            "label":       "N/A",
            "uncertain":   True,
            "explanation": "Please enter a headline and/or article body above.",
            "lime_html":   "",
        }

    # Combine exactly as preprocessing does: title [SEP] body
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

    # Generate LIME explanation
    if run_xai and not uncertain:
        lime_html = _generate_lime_explanation(
            combined_text=combined,
            pred_label=label,
            n_samples=200,
        )
    elif uncertain:
        lime_html = (
            "<p style='color:#888; font-style:italic;'>"
            "XAI unavailable for UNCERTAIN predictions — confidence too low."
            "</p>"
        )
    else:
        lime_html = ""

    return {
        "verdict":     verdict,
        "confidence":  round(confidence, 4),
        "label":       label,
        "uncertain":   uncertain,
        "fake_prob":   round(fake_prob, 4),
        "real_prob":   round(real_prob, 4),
        "explanation": explanation,
        "lime_html":   lime_html,
    }


def _gradio_predict(title: str, body: str):
    """
    Gradio interface function — returns (verdict, explanation, confidence, lime_html).
    This is the function Gradio calls directly on every button press.
    """
    result = predict_single(title, body, run_xai=True)
    return (
        result["verdict"],
        result["explanation"],
        result["confidence"],
        result["lime_html"],
    )


# ─────────────────────────────────────────────────────────────────────────────
def launch_demo(share: bool = True, server_port: int = 7860) -> None:
    """
    Build and launch the Gradio interface with XAI word highlights.

    Args:
        share       : If True, creates a public shareable link via Gradio tunnel.
                      Share=True is REQUIRED for Kaggle (no localhost access).
        server_port : Local port to serve on (default: 7860).
    """
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
            .xai-header  { font-weight: 600; color: #333; margin-bottom: 4px; }
        """,
    ) as demo:

        gr.Markdown(
            """
            # 🛡️ FakeGuard — AI Fake News Detector
            ### NeuroLogic '26 Datathon | RoBERTa + XAI Explainability
            Enter a news article headline and body to get an instant **REAL / FAKE** prediction
            — plus **word-level highlights** showing exactly WHY the model decided.
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
                    lines=5,
                    interactive=False,
                )

        # XAI section — full width below
        gr.Markdown("---")
        gr.Markdown(
            "## 🔬 XAI — Why did the model decide this?\n"
            "The highlighted words below show the model's reasoning. "
            "**Red** = pushed toward FAKE &nbsp;|&nbsp; **Green** = pushed toward REAL. "
            "This uses LIME (Local Interpretable Model-agnostic Explanations)."
        )
        lime_output = gr.HTML(
            value="<p style='color:#aaa; font-style:italic;'>Run a prediction above to see word-level highlights here.</p>",
            label="XAI Word Highlights",
        )

        # Wire up button — now returns 4 values including lime_html
        predict_btn.click(
            fn=_gradio_predict,
            inputs=[title_input, body_input],
            outputs=[verdict_output, explanation_output, confidence_output, lime_output],
        )

        title_input.submit(
            fn=_gradio_predict,
            inputs=[title_input, body_input],
            outputs=[verdict_output, explanation_output, confidence_output, lime_output],
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
            **XAI method:** LIME (Local Interpretable Model-agnostic Explanations)
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
    parser.add_argument(
        "--no-share",
        action="store_true",
        help="Disable public sharing link (local only)",
    )
    parser.add_argument(
        "--port",
        type=int,
        default=7860,
        help="Local server port (default: 7860)",
    )
    args = parser.parse_args()

    set_seed(SEED)
    launch_demo(share=not args.no_share, server_port=args.port)
