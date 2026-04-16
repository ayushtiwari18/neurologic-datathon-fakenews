# 🎯 Ultimate Goal of This Project

## What Are We Building?
A **Fake News Detection System** that classifies news articles as **Real** or **Fake** using a fine-tuned transformer model (RoBERTa) from HuggingFace.

## Why This Matters
- Misinformation spreads 6x faster than real news (MIT Study, 2018)
- Manual fact-checking is slow and unscalable
- An automated classifier can act as a first-pass filter for newsrooms, social platforms, and browser tools

## Hackathon Goal
- **Primary:** Achieve highest possible accuracy on Challenge 2 dataset
- **Secondary:** Beat baseline (TF-IDF + Logistic Regression ~92%) using RoBERTa (~98%+)
- **Bonus:** Deploy a minimal working demo (FastAPI or Gradio) to score Innovation + Real-World Impact points

## Judging Criteria Breakdown
| Criteria | Weight | How We Win It |
|---|---|---|
| Model Accuracy | 40% | Fine-tuned RoBERTa, clean preprocessing |
| Methodology | 20% | Baseline → improved model, clear train/val split, error analysis |
| Innovation | 15% | Title+Text fusion, confidence thresholding, optional deployment |
| Real-World Impact | 15% | README section on applications, browser extension idea |
| Documentation | 10% | Clean README, reproducible code, screenshots |

## One-Line Mission Statement
> **Build a transformer-powered fake news classifier that is accurate, reproducible, deployable, and well-documented — winning on all 5 judge criteria.**
