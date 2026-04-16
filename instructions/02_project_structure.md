# 📁 Project Structure

```
neurologic-datathon-fakenews/
│
├── instructions/                  ← You are here (team guides)
│   ├── 01_ultimate_goal.md
│   ├── 02_project_structure.md
│   ├── 03_pipeline_architecture.md
│   ├── 04_model_training_strategy.md
│   ├── 05_colab_setup.md
│   ├── 06_data_preprocessing.md
│   ├── 07_development_steps.md
│   ├── 08_key_failure_points.md
│   ├── 09_features_and_routes.md
│   ├── 10_resources.md
│   ├── 11_readme_structure.md
│   └── 12_points_to_remember.md
│
├── notebooks/
│   ├── 01_EDA.ipynb                ← Exploratory Data Analysis
│   ├── 02_baseline_model.ipynb     ← TF-IDF + Logistic Regression
│   └── 03_roberta_finetune.ipynb   ← Main RoBERTa fine-tuning notebook
│
├── data/
│   ├── raw/                        ← Original dataset (DO NOT MODIFY)
│   │   └── dataset.csv
│   ├── processed/                  ← Cleaned data
│   │   ├── train.csv
│   │   └── val.csv
│   └── README.md                   ← Data source description
│
├── src/
│   ├── preprocess.py               ← All cleaning functions
│   ├── train.py                    ← Training loop
│   ├── evaluate.py                 ← Metrics + confusion matrix
│   └── predict.py                  ← Inference on new text
│
├── models/
│   └── roberta_fakenews/           ← Saved model weights (after training)
│       ├── config.json
│       ├── tokenizer_config.json
│       └── pytorch_model.bin
│
├── outputs/
│   ├── confusion_matrix.png
│   ├── accuracy_plot.png
│   └── predictions.csv
│
├── app/                            ← Optional deployment
│   ├── app.py                      ← FastAPI or Gradio app
│   └── requirements.txt
│
├── requirements.txt                ← All Python dependencies
├── .gitignore
└── README.md                       ← Final polished README
```

## Key Rules
- **Never commit raw model weights** to GitHub (too large). Use `models/` only locally or HuggingFace Hub.
- **Never modify `data/raw/`** — always work on copies in `data/processed/`
- **All notebooks must be runnable top-to-bottom** with no manual steps
- **Use relative paths** everywhere — no hardcoded `/Users/ayush/...` paths
