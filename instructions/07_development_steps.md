# 🚀 Development Steps — In Order

## Timeline (12-Hour Hackathon)

```
Hour 0–1:   Setup + Data Exploration
Hour 1–2:   Preprocessing + Baseline Model
Hour 2–5:   RoBERTa Fine-tuning (runs in background)
Hour 5–6:   Evaluation + Error Analysis
Hour 6–8:   Optional Deployment (Gradio/FastAPI)
Hour 8–10:  README + Documentation
Hour 10–12: Final review, screenshots, submission
```

---

## Step 1: Repository & Environment Setup
- [ ] Clone this repo
- [ ] Open Colab, set GPU (T4), run install cell
- [ ] Mount Google Drive
- [ ] Upload dataset to `data/raw/`

## Step 2: Exploratory Data Analysis (notebook 01)
- [ ] Load dataset, check shape and columns
- [ ] Check for nulls, duplicates
- [ ] Plot class distribution → save PNG
- [ ] Plot text length distribution → save PNG
- [ ] Print 3 sample REAL articles and 3 FAKE articles
- [ ] **Document findings in a markdown cell** (judges read notebooks too)

## Step 3: Preprocessing (notebook 02 or src/preprocess.py)
- [ ] Run all cleaning steps from 06_data_preprocessing.md
- [ ] Combine title + text with [SEP]
- [ ] Encode labels
- [ ] Stratified 80/20 split
- [ ] Save processed CSVs

## Step 4: Baseline Model (notebook 02)
- [ ] TF-IDF + Logistic Regression
- [ ] Print accuracy, F1, confusion matrix
- [ ] Save confusion matrix PNG
- [ ] **Write down baseline number** — you will compare to this

## Step 5: RoBERTa Fine-Tuning (notebook 03) ← MAIN WORK
- [ ] Tokenize with RobertaTokenizer
- [ ] Create PyTorch Dataset
- [ ] Load RobertaForSequenceClassification
- [ ] Configure TrainingArguments (use settings from 04_model_training_strategy.md)
- [ ] Run Trainer.train() — this will take 25–40 min, let it run
- [ ] Evaluate on validation set
- [ ] Save model to Drive

## Step 6: Evaluation & Reporting
- [ ] Print: Accuracy, Precision, Recall, F1
- [ ] Generate confusion matrix → save PNG
- [ ] Run confidence thresholding
- [ ] Extract top 10 wrong predictions (error analysis)
- [ ] Create comparison table: Baseline vs RoBERTa

## Step 7: Optional Deployment (Do If Time Allows)
- [ ] Build Gradio interface (10 lines of code)
- [ ] Test with 3 real-world news headlines
- [ ] Take screenshot of working demo

```python
# Minimal Gradio demo — 10 lines
import gradio as gr

def predict(text):
    inputs = tokenizer(text, return_tensors='pt', truncation=True, max_length=512)
    outputs = model(**inputs)
    pred = outputs.logits.argmax().item()
    return "✅ REAL News" if pred == 1 else "🚫 FAKE News"

interface = gr.Interface(fn=predict, inputs='text', outputs='text',
                         title='FakeGuard — Fake News Detector')
interface.launch(share=True)  # share=True gives public URL
```

## Step 8: README + Submission
- [ ] Follow 11_readme_structure.md exactly
- [ ] Add all screenshots to README
- [ ] Verify notebook runs top-to-bottom without errors
- [ ] Push everything to GitHub
- [ ] Submit GitHub link + Devpost form
