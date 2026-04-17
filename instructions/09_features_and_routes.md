# ✨ Features & Optional API Routes

## Core Features (Must Have)
| Feature | Description | File |
|---|---|---|
| Data Preprocessing | Clean, combine, encode, split | `src/preprocess.py` |
| Baseline Model | TF-IDF + Logistic Regression | `notebooks/02_baseline_model.ipynb` |
| RoBERTa Fine-tuning | Main classifier | `notebooks/03_roberta_finetune.ipynb` |
| Evaluation Metrics | Accuracy, F1, Confusion Matrix | `src/evaluate.py` |
| Error Analysis | Top 10 misclassified examples | Inside notebook 03 |
| Model Saving | Save/load trained weights | `src/train.py` |

## Innovation Features (High Judge Impact)
| Feature | Description | Difficulty |
|---|---|---|
| Confidence Thresholding | Flag uncertain predictions for human review | Easy |
| Title+Text Fusion | Combine both inputs using [SEP] token | Easy |
| Baseline Comparison | Explicit table showing improvement | Easy |
| Error Analysis Section | Why model fails on specific examples | Medium |
| Gradio Demo | Interactive web UI in 10 lines | Easy |

## Optional Deployment Routes (FastAPI)

If you have time to deploy a FastAPI app:

```python
# app/app.py
from fastapi import FastAPI
from pydantic import BaseModel
from transformers import RobertaTokenizer, RobertaForSequenceClassification
import torch

app = FastAPI(title="FakeGuard API")

# Detect device — use GPU if available, else CPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

tokenizer = RobertaTokenizer.from_pretrained('./models/roberta_fakenews')
model = RobertaForSequenceClassification.from_pretrained('./models/roberta_fakenews')
model.to(device)  # Move model to correct device
model.eval()      # Set to inference mode

class NewsInput(BaseModel):
    title: str
    text: str

class PredictionOutput(BaseModel):
    label: str
    confidence: float

@app.get("/")
def root():
    return {"message": "FakeGuard Fake News Detection API", "version": "1.0"}

@app.post("/predict", response_model=PredictionOutput)
def predict(news: NewsInput):
    combined = f"{news.title} [SEP] {news.text}"[:2000]
    inputs = tokenizer(
        combined, return_tensors='pt', truncation=True, max_length=512
    )
    # Move inputs to same device as model
    inputs = {k: v.to(device) for k, v in inputs.items()}
    with torch.no_grad():  # REQUIRED — disables gradient tracking for inference
        outputs = model(**inputs)
    probs = torch.softmax(outputs.logits, dim=-1)
    pred = probs.argmax().item()
    confidence = probs.max().item()
    return PredictionOutput(
        label="REAL" if pred == 1 else "FAKE",
        confidence=round(confidence, 4)
    )

@app.get("/health")
def health():
    return {"status": "ok"}
```

### Running the API
```bash
cd app
pip install fastapi uvicorn
uvicorn app:app --reload --port 8000
# Visit: http://localhost:8000/docs  (Auto Swagger UI)
```

## Gradio (Easiest Demo — Recommended)
```python
import gradio as gr
import torch

# Detect device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)
model.eval()

def predict_news(title, text):
    """Returns a string prediction label with confidence."""
    combined = f"{title} [SEP] {text}"[:2000]
    inputs = tokenizer(
        combined, return_tensors='pt', truncation=True, max_length=512
    )
    inputs = {k: v.to(device) for k, v in inputs.items()}  # Move to device
    with torch.no_grad():  # REQUIRED — prevents silent CPU fallback and memory leak
        outputs = model(**inputs)
    probs = torch.softmax(outputs.logits, dim=-1)
    pred = probs.argmax().item()
    conf = probs.max().item()
    label = "✅ REAL News" if pred == 1 else "🚫 FAKE News"
    return f"{label} (Confidence: {conf:.1%})"  # Always returns str

gr.Interface(
    fn=predict_news,
    inputs=[gr.Textbox(label="News Title"), gr.Textbox(label="News Text", lines=5)],
    outputs=gr.Textbox(label="Prediction"),
    title="FakeGuard — AI Fake News Detector",
    description="Enter a news article title and body. Our RoBERTa model will classify it."
).launch(share=True)
```
