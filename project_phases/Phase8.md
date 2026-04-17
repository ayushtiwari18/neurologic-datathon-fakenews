# Phase 8 — Deployment, Demo & Submission Preparation

## GOAL
Build a minimal Gradio demo, finalize the README with all metrics and visuals, and package the complete submission with reproducible instructions — targeting the Innovation (15%), Real-World Impact (15%), and Documentation (10%) judging criteria.

## WHY THIS PHASE EXISTS
- **Risk reduced:** A 40% accuracy score alone cannot win — the remaining 60% of judging weight (Methodology + Innovation + Impact + Documentation) requires visible, polished deliverables
- **Capability created:** A live interactive demo that judges can use in < 30 seconds, and a README that tells the full story
- **Competitive advantage:** Most teams skip the demo and write minimal READMEs. A working Gradio app + detailed README = automatic top-3 position on non-accuracy criteria

## PREREQUISITES
- Phase 5 complete (all metrics + visuals in `outputs/`)
- Phase 7 complete (`outputs/submission.csv` exists)
- `models/roberta_fakenews_best/` exists
- All `outputs/*.png` and `outputs/*.json` files exist

## INPUTS
- `models/roberta_fakenews_best/`
- `outputs/roberta_eval_metrics.json`
- `outputs/roberta_confusion_matrix.png`
- `outputs/error_analysis.csv`
- `outputs/model_comparison.json`
- `outputs/submission.csv`
- `config.py`

## TASKS
### Demo (Gradio)
1. Load saved model + tokenizer
2. Build `predict_with_confidence(text)` function — returns `(label, confidence_pct)`
3. Apply confidence threshold: < 0.70 → append "(Low Confidence — Human Review Recommended)"
4. Create Gradio Interface: `gr.Interface(fn=predict, inputs=gr.Textbox(...), outputs=[gr.Label(), gr.Textbox()])`
5. Add title: "FakeGuard — Fake News Detector (NeuroLogic '26)"
6. Add description with model accuracy and link to repo
7. Launch with `share=True` for public link
8. Screenshot the demo → save as `outputs/demo_screenshot.png`

### README Finalization
9. Expand `README.md` using structure from `instructions/11_readme_structure.md`
10. Include: badges, model accuracy, confusion matrix image, error analysis table, judging criteria table
11. Add reproducibility section: step-by-step run instructions
12. Add real-world impact section: browser extension concept, newsroom use case
13. Add limitations and future work section

### Final Packaging
14. Ensure all source files committed: `src/preprocess.py`, `src/dataset.py`, `src/train.py`, `src/evaluate.py`, `src/predict.py`, `src/baseline.py`
15. Ensure all output artifacts committed: `outputs/*.json`, `outputs/*.png`, `outputs/submission.csv`
16. Verify `requirements.txt` is complete
17. Create `REPRODUCIBILITY.md` — exact steps to reproduce training from scratch

## AI EXECUTION PROMPTS
- "Build a Gradio interface that takes a news article text input and returns the predicted label (REAL/FAKE) and confidence percentage. Load the model from models/roberta_fakenews_best/. Apply confidence threshold 0.70. Launch with share=True."
- "Expand README.md to include: project description, judging criteria table with our scores, model accuracy badge, confusion matrix image, error analysis table (10 rows), training instructions, and real-world impact section."
- "Create REPRODUCIBILITY.md with exact commands: git clone, pip install, data placement, preprocess, baseline, train, evaluate, predict. Each step on a separate line with expected output."

## ALGORITHMS
- **Demo inference:** Same `predict_with_confidence` from Phase 4 training strategy
- **README:** Markdown with embedded images from `outputs/`
- **Reproducibility:** Shell command sequence

## CODE SNIPPETS
```python
import gradio as gr
import torch, torch.nn.functional as F
from transformers import RobertaForSequenceClassification, RobertaTokenizer
from config import ID2LABEL, CONFIDENCE_THRESHOLD

model = RobertaForSequenceClassification.from_pretrained('./models/roberta_fakenews_best')
tokenizer = RobertaTokenizer.from_pretrained('./models/roberta_fakenews_best')
model.eval()

def predict(text):
    enc = tokenizer(text, return_tensors='pt', truncation=True, max_length=512)
    with torch.no_grad():
        probs = F.softmax(model(**enc).logits, dim=-1)
    conf  = probs.max().item()
    label = ID2LABEL[probs.argmax().item()]
    note  = " (Low Confidence — Human Review Recommended)" if conf < CONFIDENCE_THRESHOLD else ""
    return label + note, f"{conf:.1%}"

gr.Interface(
    fn=predict,
    inputs=gr.Textbox(lines=6, placeholder="Paste a news article here...", label="News Article"),
    outputs=[gr.Textbox(label="Verdict"), gr.Textbox(label="Confidence")],
    title="FakeGuard — Fake News Detector (NeuroLogic '26)",
    description=f"Fine-tuned RoBERTa model. Val Accuracy: ≥97%. Confidence threshold: {CONFIDENCE_THRESHOLD:.0%}"
).launch(share=True)
```

## OUTPUTS
- `src/app.py` — Gradio demo script
- `REPRODUCIBILITY.md` — step-by-step reproduction guide
- `README.md` — fully expanded (replaces stub)
- `outputs/demo_screenshot.png` — screenshot of live demo
- All `src/*.py` files verified committed
- All `outputs/*.json` and `outputs/*.png` files committed

## EXPECTED RESULTS
- Gradio app launches with public URL (`share=True`)
- README renders correctly on GitHub with images
- `REPRODUCIBILITY.md` has ≥ 8 steps, all commands executable
- All committed files verified in repo

## VALIDATION CHECKS
- [ ] Gradio launches without error
- [ ] `predict('Breaking: Scientists discover...') ` returns (label, confidence) tuple
- [ ] README.md has > 200 lines (stub was 2 lines)
- [ ] `REPRODUCIBILITY.md` exists with step-by-step commands
- [ ] `outputs/submission.csv` committed to repo
- [ ] All 6 `src/*.py` files exist in repo

## FAILURE CONDITIONS
- Gradio fails to load model → check model path, ensure `config.json` exists in model dir
- README images not rendering → use relative paths `./outputs/img.png` not absolute
- Missing src files → run Cycles 5–7 source code commits first

## RECOVERY ACTIONS
- Gradio model load error: use `from_pretrained` with `local_files_only=True`
- README images broken: move images to `assets/` folder, update paths
- Missing src files: commit them before Phase 8 — do not skip source code phases

## PERFORMANCE TARGETS
- Gradio demo responds in < 3 seconds per prediction
- README loads in < 2 seconds on GitHub
- Total Phase 8 completion: < 30 minutes

## RISKS
- Model too large for Colab to keep loaded + run Gradio simultaneously → use `device='cpu'` for demo if GPU OOM
- `share=True` Gradio link expires after 72 hours → screenshot before expiry
- README images not committed → outputs/ must be in repo (not .gitignored)

## DELIVERABLES
- ✅ `src/app.py` (Gradio demo, working)
- ✅ `README.md` (fully expanded, images embedded)
- ✅ `REPRODUCIBILITY.md`
- ✅ All `src/*.py` files in repo
- ✅ All `outputs/*.json`, `outputs/*.png`, `outputs/submission.csv` committed
- ✅ Project is judge-ready on all 5 scoring criteria
