# 📌 Points to Remember — Quick Reference Checklist

## Before You Start
- [ ] Read 01_ultimate_goal.md — know WHY we are building this
- [ ] Set up Colab with GPU (see 05_colab_setup.md)
- [ ] Set random seed = 42 in all notebooks
- [ ] Check dataset column names before writing any code
- [ ] Add `.gitignore` — never commit model weights or raw data

## During Development
- [ ] Always save checkpoints to Google Drive every epoch
- [ ] Only fit preprocessing on training data — not full dataset
- [ ] Use stratified split — not random split
- [ ] Run baseline FIRST before fine-tuning
- [ ] Write accuracy numbers down as you get them
- [ ] Take screenshots of every important output
- [ ] Run `Runtime → Restart and Run All` before pushing notebook

## Model Training
- [ ] Use `fp16=True` for 2x speed on GPU
- [ ] Use `evaluation_strategy='epoch'` to track progress
- [ ] Use `load_best_model_at_end=True` to auto-keep best model
- [ ] Watch for overfitting: val loss should decrease, not increase after epoch 2
- [ ] If CUDA OOM: reduce `per_device_train_batch_size` from 16 to 8

## Evaluation
- [ ] Report metrics on VALIDATION set only — never training set
- [ ] Generate and save confusion matrix as PNG
- [ ] Run error analysis — find 5–10 wrong predictions and explain why
- [ ] Create comparison table: Baseline vs RoBERTa
- [ ] Add confidence scores to predictions

## Documentation & Submission
- [ ] README follows 11_readme_structure.md exactly
- [ ] All screenshots are in `outputs/` folder and linked in README
- [ ] `requirements.txt` lists all dependencies with versions
- [ ] GitHub repo is public (or link shared with judges)
- [ ] Devpost submission form is completed with GitHub link
- [ ] Submission is done BEFORE deadline (set alarm 30 min before)

## Golden Rules
1. **Done is better than perfect** — a working 95% model beats a broken 99% model
2. **Document everything** — judges read your README as much as your code
3. **Reproducibility wins** — if it only works on your machine, it doesn't count
4. **Show your work** — plots, tables, error analysis all add points
5. **Deploy if you can** — a live demo is worth more than 2% accuracy gain

## Emergency Fallbacks
| Problem | Fallback |
|---|---|
| RoBERTa OOM | Use distilbert-base-uncased (smaller, still ~96%) |
| Colab quota gone | Switch to Kaggle Notebooks |
| Training too slow | Reduce dataset to 20k samples for speed |
| Can't deploy | Screenshot Gradio in Colab — still counts as demo |
| Forgot to save model | Retrain from checkpoint in Drive |
