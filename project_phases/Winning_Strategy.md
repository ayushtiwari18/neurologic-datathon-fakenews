# Winning Strategy Document
**Project:** NeuroLogic '26 Datathon — Fake News Detection  
**Objective:** Maximize probability of 1st place across all 5 judging criteria  
**Version:** 1.0 | **Updated:** 2026-04-17

---

## Judging Criteria & Our Scores

| Criteria | Weight | Our Approach | Target Score |
|----------|--------|-------------|-------------|
| Model Accuracy | 40% | Fine-tuned RoBERTa ≥ 98% | 40/40 |
| Methodology | 20% | Baseline → RoBERTa progression + error analysis | 18/20 |
| Innovation | 15% | Title+Body fusion + confidence thresholding + Gradio demo | 14/15 |
| Real-World Impact | 15% | Browser extension concept + newsroom use case in README | 13/15 |
| Documentation | 10% | Full README with images, metrics, reproducibility guide | 10/10 |
| **Total** | **100%** | | **95/100** |

---

## 1. Accuracy Strategy (40% of score)

### Primary Lever: RoBERTa > BERT > TF-IDF
RoBERTa was trained on 160GB of data vs BERT's 16GB, with dynamic masking and no Next Sentence Prediction — giving it ~98%+ accuracy on NLP classification tasks vs ~95% for BERT and ~92% for TF-IDF.

### Execution Steps for Maximum Accuracy
1. **Text Fusion:** Combine `title + " [SEP] " + text` — gives model 2x the signal vs body-only
2. **Full 512 Tokens:** Use `MAX_LEN=512` — never truncate to 128 or 256 (common shortcut that loses context)
3. **3+ Epochs:** Do not stop at epoch 1 or 2. Accuracy typically jumps +1–1.5% from epoch 2 to 3
4. **Best Checkpoint:** `load_best_model_at_end=True` ensures we always use peak accuracy epoch
5. **LR Sweep:** Try [1e-5, 2e-5, 3e-5] in Phase 6. Default 2e-5 is good but not always optimal
6. **FP16:** Enables larger effective batch without OOM, stabilizes training

### Accuracy Target vs Competition
| Model | Expected Accuracy | Competition Rank |
|-------|------------------|------------------|
| TF-IDF + LogReg (theirs) | ~91–93% | Bottom tier |
| BERT fine-tuned | ~95–96% | Mid tier |
| **RoBERTa fine-tuned (ours)** | **~97–99%** | **Top tier** |
| RoBERTa + ensemble | ~98–99%+ | Top tier |

---

## 2. Methodology Strategy (20% of score)

### What Judges Want to See
Judges score methodology on: clear pipeline, baseline comparison, train/val split, error analysis, reproducibility.

### Our Methodology Differentiators
1. **Baseline First:** TF-IDF + LogReg run before transformer — establishes comparison point judges require
2. **Stratified Split:** 80/20 with `stratify=labels` — prevents accidental class skew
3. **Error Analysis Table:** Top 10 misclassified samples with confidence scores — shows we understand failure modes
4. **`model_comparison.json`:** Machine-readable proof of improvement over baseline
5. **Confidence Thresholding:** CONFIDENCE_THRESHOLD=0.70 — shows awareness of prediction uncertainty
6. **Per-class Metrics:** `classification_report` with REAL/FAKE breakdown, not just overall accuracy

### Evidence Package for Judges
- `outputs/baseline_metrics.json` — baseline score
- `outputs/roberta_eval_metrics.json` — RoBERTa score
- `outputs/model_comparison.json` — delta (improvement)
- `outputs/roberta_confusion_matrix.png` — visual proof
- `outputs/error_analysis.csv` — failure mode analysis

---

## 3. Innovation Strategy (15% of score)

### Innovation #1: Title + Body Text Fusion
- Most teams use body text only
- We combine `title + " [SEP] " + text` — headlines are often the most fake-signal-rich part
- Expected accuracy gain: +0.5–1%

### Innovation #2: Confidence Thresholding
- Raw classifiers output a label with no uncertainty signal
- We apply `CONFIDENCE_THRESHOLD=0.70` — predictions below this are flagged for human review
- Real-world applicable: a newsroom editor only sees low-confidence articles

### Innovation #3: Live Gradio Demo
- Interactive UI judges can use in 30 seconds
- Shows the model works on real inputs, not just a test CSV
- Differentiates from static notebook submissions

### Innovation #4: FakeGuard Branding
- Name the system "FakeGuard" throughout README and demo
- Branding makes the project memorable to judges
- Add a one-line tagline: *"AI-powered first-pass filter for newsrooms and social platforms"

---

## 4. Real-World Impact Strategy (15% of score)

### Impact Narrative (Include in README)
> Misinformation spreads 6x faster than real news (MIT, 2018). Manual fact-checking cannot scale. FakeGuard acts as a first-pass filter — automatically flagging suspicious articles before they reach human reviewers.

### Use Cases to Document
1. **Newsroom Tool:** Auto-flag incoming wire stories before editorial review
2. **Browser Extension Concept:** Highlight fake news probability on any webpage (link to concept, no need to build it)
3. **Social Media Moderation:** API endpoint for platforms to classify posts at scale
4. **Educational Tool:** Show readers why an article was flagged (confidence score + top TF-IDF features)

### Technical Proof of Impact
- FastAPI endpoint spec in `instructions/09_features_and_routes.md` — shows deployment readiness
- Gradio demo — live proof the model works on unseen text
- `CONFIDENCE_THRESHOLD` — production-safe design (uncertain → human review, not auto-reject)

---

## 5. Documentation Strategy (10% of score)

### README Must Include
1. 🧠 Project title + tagline
2. 🏆 Datathon context + judging criteria table
3. 📊 Accuracy results: Baseline vs RoBERTa (with `model_comparison.json` values)
4. 🖼️ Confusion matrix image embedded
5. 📋 Error analysis table (10 rows inline)
6. 🔧 Architecture diagram (text-based from `instructions/03_pipeline_architecture.md`)
7. ⚡ Quick start: `git clone` + `pip install` + run instructions
8. 🌍 Real-world impact section
9. ⚠️ Limitations + future work
10. 📄 License

### `REPRODUCIBILITY.md` Must Include
```
1. git clone https://github.com/ayushtiwari18/neurologic-datathon-fakenews
2. cd neurologic-datathon-fakenews
3. pip install -r requirements.txt
4. Place train.csv and test.csv in data/
5. python src/preprocess.py
6. python src/baseline.py
7. python src/train.py
8. python src/evaluate.py
9. python src/predict.py
10. outputs/submission.csv is now ready
```

---

## 6. Speed Strategy (Execution Velocity)

### Why Speed Matters in a Hackathon
- Finishing early = time to fix bugs + polish documentation
- Most teams are still debugging at T-30 min; we should be done at T-2h 42min
- Buffer of 45+ minutes for unexpected issues

### Speed Levers
1. **Pre-built phases** — no decision fatigue, follow the plan
2. **Colab notebook** — cell-by-cell execution, no local setup
3. **config.py** — all hyperparameters in one place, no hardcoded values to hunt
4. **`save_strategy='epoch'`** — never lose more than 1 epoch to a disconnect
5. **Baseline in 3 minutes** — validates data pipeline before committing 40 min to training

---

## 7. Submission Strategy

### Submission File Checklist
- [ ] `outputs/submission.csv` — row count matches `test.csv`
- [ ] Columns: `id`, `label`, `confidence`
- [ ] `label` values: only `REAL` or `FAKE` (no integers)
- [ ] No null values
- [ ] Submitted to judge portal by T-5 min

### Repository Checklist
- [ ] All `src/*.py` files committed
- [ ] `outputs/*.json` and `outputs/*.png` committed
- [ ] `README.md` fully expanded
- [ ] `REPRODUCIBILITY.md` committed
- [ ] `requirements.txt` complete
- [ ] Repo is public

### Competitive Intelligence
- **Most teams will:** Use BERT or DistilBERT, body-only text, no demo, minimal README
- **We will:** Use RoBERTa, title+body fusion, confidence thresholding, live Gradio demo, full README
- **Our moat:** The combination of technical depth (98%+ accuracy) + presentation quality (demo + docs) targets all 5 criteria simultaneously

---

## One-Line Winning Formula

> **Train RoBERTa correctly. Document thoroughly. Demo visibly. Submit early.**
