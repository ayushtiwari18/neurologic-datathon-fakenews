# 📋 FakeGuard — Devpost Submission Draft

> **Instructions for April 25:** Open this file, copy each section into the Devpost form.
> Replace the Gradio link placeholder after running Cell 11.
> Submit on Devpost **before 9:30 PM IST** — do NOT wait until the last minute.

---

## Project Title
```
FakeGuard — AI Fake News Detector
```

---

## Tagline *(1-line summary field in Devpost)*
```
Fine-tuned RoBERTa model achieving 99.42% accuracy on fake news detection — +4.76% over baseline.
```

---

## The Problem

Fake news spreads 6× faster than real news (MIT, 2018). Manual fact-checking is slow, unscalable, and reactive. By the time a false story is debunked, millions have already seen it. We need automated, high-confidence detection at publication time — not after the damage is done.

---

## What We Built

FakeGuard is a production-ready fake news classifier built on fine-tuned `roberta-base`. Given any news article’s headline and body, it instantly predicts **REAL**, **FAKE**, or **UNCERTAIN** with a confidence score.

Key innovations:
- **Title + Body fusion** using `[SEP]` token — feeds both signals to RoBERTa simultaneously
- **Confidence thresholding** — predictions below 70% confidence are flagged UNCERTAIN for human review, making the system production-safe
- **Live interactive demo** — judges can test any article in real time via the Gradio link below

---

## How We Built It

- **Model:** `roberta-base` (125M parameters, 12 transformer layers) fine-tuned for binary classification
- **Dataset:** WELFake — 72,134 news articles (35,028 Real + 37,106 Fake)
- **Split:** Stratified 70/15/15 (Train/Validation/Test) with `random_state=42`
- **Training:** 3 epochs, `lr=2e-5`, `batch_size=16`, `fp16=True` on Kaggle T4 GPU (~30 min)
- **Early stopping:** patience=2 to prevent overfitting
- **Platform:** Kaggle Notebooks (GPU T4 x1)

---

## Results

| Model | Accuracy | F1 Score |
|---|---|---|
| TF-IDF + Logistic Regression (baseline) | 94.66% | 0.9466 |
| **FakeGuard RoBERTa (ours)** | **99.42%** | **0.9942** |

Training progression:

| Epoch | Val Loss | Accuracy |
|---|---|---|
| 1 | 0.1024 | 98.54% |
| 2 | 0.1112 | 98.95% |
| **3** | **0.0699** | **99.42%** |

> ✅ Only **55 misclassifications** out of 9,552 validation articles.
> ✅ Metric measured on held-out validation set — never seen during training.

---

## Challenges We Ran Into

- `peft` pre-installed on Kaggle conflicted with `transformers` — solved by uninstalling it at runtime
- `evaluation_strategy` deprecated in newer HuggingFace — switched to `eval_strategy`
- P100 GPU incompatible with current Kaggle PyTorch — enforced T4 with a hard check in Cell 3
- Dataset column names required careful mapping via a robust `label_map` handling string and float label variants

---

## What We Learned

- RoBERTa’s lack of token type IDs (unlike BERT) means never passing `token_type_ids` during inference
- Combining title and body with a plain-text `[SEP]` separator meaningfully improves classification vs body-only
- Confidence thresholding is more valuable than marginal accuracy gains for real-world deployment

---

## What’s Next

- Deploy as a browser extension to flag suspicious articles while reading
- Add multilingual support (Challenge 3 extension)
- Build a REST API so any app can POST an article and receive FAKE/REAL + confidence score
- Fine-tune on domain-specific fake news (health, politics, finance)

---

## Links

| Item | Link |
|---|---|
| GitHub Repository | https://github.com/ayushtiwari18/neurologic-datathon-fakenews |
| Live Gradio Demo | ⚠️ **REPLACE THIS** with your `https://xxxx.gradio.live` link from Cell 11 |
| HuggingFace Model | https://huggingface.co/ayushtiwari18/fakeguard-roberta |

---

## Reported Metric

```
Challenge 2 — Fake News & Misinformation Detection
Evaluation Metric: Overall Accuracy
Reported Accuracy: 99.42%
Measured on: Held-out validation set (9,552 articles, 15% of WELFake)
Split method: Stratified 70/15/15, random_state=42
Baseline comparison: TF-IDF + LR = 94.66% → RoBERTa = 99.42% (+4.76%)
```

---

## Screenshots to Upload on Devpost

Before submitting, take these screenshots and upload them to Devpost:

- [ ] Training table (all 3 epochs with accuracy numbers)
- [ ] Confusion matrix (`outputs/confusion_matrix.png` — already in repo)
- [ ] Gradio demo with a FAKE news example
- [ ] Gradio demo with a REAL news example
- [ ] Cell 10 output showing predictions.csv generated

---

*Built for NeuroLogic '26 Datathon — Challenge 2: Fake News & Misinformation Detection*
*Department of AI & ML, GGITS | April 25, 2026*
