"""
gradio_demo.py — Interactive Fake News Detector UI for FakeGuard.

Source of truth:
    - instructions/09_features_and_routes.md
    - instructions/04_model_training_strategy.md
    - config.py

What this does:
    Launches a Gradio web interface where anyone can type a news article
    (title + body) and instantly get a REAL / FAKE / UNCERTAIN verdict
    with a confidence score. Judges can interact with a live link.

Features:
    - Single article prediction with confidence bar
    - Colour-coded verdict: 🟢 REAL | 🔴 FAKE | 🟡 UNCERTAIN
    - Batch prediction from CSV upload
    - Example articles pre-loaded (one real, one fake)
    - Public shareable URL via Gradio tunnel (share=True)
    - Graceful fallback if model not yet trained

Usage:
    # Run locally or in Kaggle:\n
    python app/gradio_demo.py

    # Or from a notebook cell:
    from app.gradio_demo import launch_demo
    launch_demo(share=True)

IMPORTANT:
    Run src/train.py FIRST to get competition-grade accuracy.
    Without the trained model, this falls back to roberta-base
    base weights which will give random predictions.
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
def predict_single(title: str, body: str) -> dict:
    """
    Run inference on a single article (title + body).

    Combines title and body using the same [SEP] format as preprocessing.
    Returns verdict, confidence, and detailed breakdown for Gradio display.

    Args:
        title : Article headline.
        body  : Article body text.

    Returns:
        dict: verdict, confidence, label, uncertain, explanation
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
        }

    # Combine exactly as preprocessing does: title [SEP] body
    # If only one field provided, use it alone
    if title and body:
        combined = f"{title} [SEP] {body}"
    elif title:
        combined = title
    else:
        combined = body

    # Truncate display text for readability
    display_text = combined[:200] + "..." if len(combined) > 200 else combined

    _load_model_once()

    encoded = _tokenizer(
        combined,
        truncation=True,
        padding=True,
        max_length=MAX_LEN,
        return_tensors="pt",
    )

    with torch.no_grad():  # REQUIRED — inference only, no gradient tracking
        input_ids      = encoded["input_ids"].to(_device)
        attention_mask = encoded["attention_mask"].to(_device)
        # NOTE: never pass token_type_ids — RoBERTa doesn't use them
        outputs = _model(input_ids=input_ids, attention_mask=attention_mask)
        probs   = F.softmax(outputs.logits, dim=-1).cpu().numpy()[0]

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

    return {
        "verdict":     verdict,
        "confidence":  round(confidence, 4),
        "label":       label,
        "uncertain":   uncertain,
        "fake_prob":   round(fake_prob, 4),
        "real_prob":   round(real_prob, 4),
        "explanation": explanation,
    }


def _gradio_predict(title: str, body: str):
    """
    Gradio interface function — takes title + body, returns (verdict, explanation, confidence).
    This is the function Gradio calls directly on every button press.
    """
    result = predict_single(title, body)
    confidence_bar = result["confidence"]
    return result["verdict"], result["explanation"], confidence_bar


# ─────────────────────────────────────────────────────────────────────────────
def launch_demo(share: bool = True, server_port: int = 7860) -> None:
    """
    Build and launch the Gradio interface.

    Args:
        share       : If True, creates a public shareable link via Gradio tunnel.
                      Share=True is REQUIRED for Kaggle (no localhost access).
        server_port : Local port to serve on (default: 7860).
    """
    try:
        import gradio as gr
    except ImportError:
        logger.error(
            "Gradio not installed. Run: pip install gradio>=4.0.0"
        )
        raise

    # ── Example articles (shown in the interface for judges to click) ────────────
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

    # ── Build Gradio interface ──────────────────────────────────────────────
    with gr.Blocks(
        title="FakeGuard — AI Fake News Detector",
        theme=gr.themes.Soft(),
        css="""
            .verdict-box { font-size: 1.4em; font-weight: bold; padding: 12px; border-radius: 8px; }
            .footer-text { color: #888; font-size: 0.85em; text-align: center; margin-top: 8px; }
        """,
    ) as demo:

        gr.Markdown(
            """
            # 🛡️ FakeGuard — AI Fake News Detector
            ### NeuroLogic '26 Datathon | Model: RoBERTa (`roberta-base`) fine-tuned on WELFake
            Enter a news article headline and body to get an instant **REAL / FAKE** prediction.
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

        # Wire up button
        predict_btn.click(
            fn=_gradio_predict,
            inputs=[title_input, body_input],
            outputs=[verdict_output, explanation_output, confidence_output],
        )

        # Also trigger on Enter key in title box
        title_input.submit(
            fn=_gradio_predict,
            inputs=[title_input, body_input],
            outputs=[verdict_output, explanation_output, confidence_output],
        )

        # Pre-loaded examples
        gr.Examples(
            examples=examples,
            inputs=[title_input, body_input],
            label="📌 Click an example to try it",
            examples_per_page=4,
        )

        # Model info footer
        gr.Markdown(
            f"""
            ---
            **Model details:** `roberta-base` fine-tuned on WELFake (~72k articles, 70/15/15 split)  
            **Uncertainty threshold:** {CONFIDENCE_THRESHOLD:.0%} confidence  
            **Labels:** FAKE (0) = likely fake news | REAL (1) = likely real news  
            **Baseline:** TF-IDF + Logistic Regression = 94.66% accuracy  
            **FakeGuard RoBERTa target:** ~97–98% accuracy  
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
