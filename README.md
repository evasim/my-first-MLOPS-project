---
title: Simple Text Classifier
emoji: 📰
colorFrom: blue
colorTo: purple
sdk: gradio
sdk_version: 6.24.0
app_file: app.py
pinned: false
license: mit
short_description: Classifies news into World, Sports, Business, or Sci/Tech.
---

# My first MLOps Project

[![CI](https://github.com/evasim/my-first-MLOPS-project/actions/workflows/ci.yml/badge.svg)](https://github.com/evasim/my-first-MLOPS-project/actions/workflows/ci.yml)

🚧 Currently just started to learn MLOps (how to build and deploy models) — following the [Made With ML](https://madewithml.com/) course.

📊 Using one of the public datasets on Hugging Face, [fancyzhx/ag_news](https://huggingface.co/datasets/fancyzhx/ag_news), to classify news headlines into World, Sports, Business, or Sci/Tech.

🚀 **Live demo:** [huggingface.co/spaces/Evasim/Simple-Text-Classifier](https://huggingface.co/spaces/Evasim/Simple-Text-Classifier)

🤗 **Trained model:** [huggingface.co/Evasim/First_Project](https://huggingface.co/Evasim/First_Project) (fine-tuned SciBERT)

## Reproducing locally

```bash
git clone https://github.com/evasim/my-first-MLOPS-project.git
cd my-first-MLOPS-project
pip install -r requirements.txt

# regenerate the dataset (not committed — it's public and regenerable)
python src/data_ingestion.py

# run the test suite
pytest src/
```

To run predictions against the trained model, add to a `.env` file:
```
HF_REPO_ID=Evasim/First_Project
```
Then: `python -m src.predict "your headline here"`

Training happens in [notebooks/MLOPS_project.ipynb](notebooks/MLOPS_project.ipynb).
