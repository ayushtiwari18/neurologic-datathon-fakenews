# Phase 6 — Optimization & Hyperparameter Tuning

## GOAL
Push model accuracy from ~97% toward 98–99% through targeted optimizations: learning rate adjustment, epoch extension, class weight balancing, and optional ensemble with baseline — without exceeding Colab T4 runtime limits.

## WHY THIS PHASE EXISTS
- **Risk reduced:** Default hyperparameters rarely produce peak performance; systematic tuning recovers 1–2% accuracy
- **Capability created:** An optimized model with measurably better metrics than Phase 4 defaults
- **Competitive advantage:** The difference between 97% and 99% accuracy is the difference between 1st and 3rd place on the leaderboard

## PREREQUISITES
- Phase 4 complete (`models/roberta_fakenews/` exists)
- Phase 5 complete (`outputs/roberta_eval_metrics.json` accuracy ≥ 0.97)
- Remaining Colab session time ≥ 45 minutes
- Baseline metrics confirmed in `outputs/baseline_metrics.json`

## INPUTS
- `data/processed/train_clean.csv`
- `data/processed/val_clean.csv`
- `outputs/roberta_eval_metrics.json` (current best accuracy)
- `config.py`

## TASKS
1. **Learning Rate Sweep (fast):** Try LR in [1e-5, 2e-5, 3e-5] — train 1 epoch each, compare val accuracy. Select best LR.
2. **Epoch Extension:** If best val accuracy still improving at epoch 3, train 1 more epoch (epoch 4) with best LR.
3. **Class Weight Check:** Compute class distribution. If imbalanced (>60/40 split), add `class_weight='balanced'` equivalent via weighted loss.
4. **Warmup Tuning:** Try warmup_steps=100 vs 500 — pick the one with lower epoch-1 val loss.
5. **Gradient Accumulation:** If batch_size was reduced to 8 (OOM recovery), set `gradient_accumulation_steps=2` to simulate batch_size=16.
6. **Soft Ensemble (optional):** Average softmax probabilities from baseline LogReg and RoBERTa. Test if ensemble accuracy > RoBERTa alone.
7. Save best model as `models/roberta_fakenews_best/`
8. Save optimized metrics to `outputs/optimized_metrics.json`
9. Document all tried configurations in `outputs/hyperparam_log.csv`

## AI EXECUTION PROMPTS
- "Run a 1-epoch training sweep with learning rates [1e-5, 2e-5, 3e-5]. For each, print val accuracy after epoch 1. Select the LR with highest val accuracy. Do not use early stopping — evaluate manually."
- "Check label distribution in train_clean.csv. If any class has < 40% of samples, compute class weights using sklearn compute_class_weight and pass to a custom weighted CrossEntropyLoss."
- "If val accuracy has not plateaued after epoch 3 (loss still decreasing), run one additional epoch with the best LR. Compare epoch 3 vs epoch 4 accuracy. Keep whichever is higher."
- "Optional: Load baseline TF-IDF LogReg probabilities and RoBERTa softmax probabilities on val set. Average them. Compute ensemble accuracy. If ensemble > RoBERTa alone by > 0.3%, adopt ensemble for prediction."

## ALGORITHMS
- **LR Sweep:** Manual grid search over 3 values (not Optuna — too slow for hackathon)
- **Class Weighting:** `sklearn.utils.class_weight.compute_class_weight('balanced', classes, labels)`
- **Ensemble:** Soft voting — average softmax probabilities from two models
- **Gradient Accumulation:** Effective batch = `batch_size × gradient_accumulation_steps`

## CODE SNIPPETS
```python
# LR sweep — 1 epoch each
for lr in [1e-5, 2e-5, 3e-5]:
    args = TrainingArguments(
        output_dir=f'./models/lr_sweep_{lr}',
        num_train_epochs=1,
        per_device_train_batch_size=16,
        learning_rate=lr,
        fp16=True, report_to='none',
        evaluation_strategy='epoch', seed=42
    )
    trainer = Trainer(model=fresh_model, args=args,
                      train_dataset=train_dataset,
                      eval_dataset=val_dataset,
                      data_collator=data_collator,
                      compute_metrics=compute_metrics)
    result = trainer.train()
    print(f"LR {lr}: val acc = {trainer.evaluate()['eval_accuracy']:.4f}")
```
```python
# Class weight computation
from sklearn.utils.class_weight import compute_class_weight
import numpy as np
weights = compute_class_weight('balanced',
    classes=np.unique(train_df['label']),
    y=train_df['label'].values)
class_weights = torch.tensor(weights, dtype=torch.float).to(device)
loss_fn = torch.nn.CrossEntropyLoss(weight=class_weights)
```
```python
# Soft ensemble
roberta_probs  = get_roberta_softmax(val_texts)   # shape (N, 2)
baseline_probs = get_logreg_proba(val_texts)       # shape (N, 2)
ensemble_preds = np.argmax((roberta_probs + baseline_probs) / 2, axis=1)
print(f"Ensemble acc: {accuracy_score(val_labels, ensemble_preds):.4f}")
```

## OUTPUTS
- `models/roberta_fakenews_best/` — best optimized model
- `outputs/optimized_metrics.json` — best accuracy + F1 after tuning
- `outputs/hyperparam_log.csv` — all LR/epoch combinations tried
- `src/train.py` — updated with best hyperparameters

## EXPECTED RESULTS
- Post-optimization accuracy: ≥ 98%
- Post-optimization F1 (weighted): ≥ 0.98
- Improvement over Phase 4 baseline: ≥ +0.5%
- `hyperparam_log.csv` shows ≥ 3 configurations tested

## VALIDATION CHECKS
- [ ] `outputs/optimized_metrics.json` accuracy > Phase 4 accuracy
- [ ] `models/roberta_fakenews_best/config.json` exists
- [ ] `outputs/hyperparam_log.csv` has ≥ 3 rows
- [ ] Best LR documented in hyperparam_log
- [ ] No training on val set at any point

## FAILURE CONDITIONS
- LR sweep causes OOM → reduce per_device_train_batch_size to 8
- Epoch 4 accuracy lower than epoch 3 → model overfitting, keep epoch 3 weights
- Ensemble hurts accuracy → discard ensemble, use RoBERTa alone
- Colab session expires mid-sweep → resume from last saved checkpoint

## RECOVERY ACTIONS
- OOM during sweep: reduce batch_size, add `gradient_accumulation_steps=2`
- Overfitting at epoch 4: restore `models/roberta_fakenews/` checkpoint from epoch 3
- Session expires: reload model from last `save_strategy='epoch'` checkpoint
- Ensemble worse: log result and proceed with RoBERTa-only prediction

## PERFORMANCE TARGETS
- Total optimization time: < 60 minutes on Colab T4
- Final val accuracy: ≥ 98%
- Memory stable throughout: < 14 GB GPU

## RISKS
- Over-tuning on val set (validation leakage) → run sweep on a held-out test portion, not the main val set
- Colab time limit hit during sweep → run only LR sweep, skip epoch extension if low on time
- Ensemble adds latency to inference → only adopt if accuracy gain ≥ 0.3%

## DELIVERABLES
- ✅ `models/roberta_fakenews_best/` (best checkpoint)
- ✅ `outputs/optimized_metrics.json` (accuracy ≥ 0.98)
- ✅ `outputs/hyperparam_log.csv`
- ✅ `src/train.py` updated with optimal hyperparameters
- ✅ Decision documented: ensemble vs single model
