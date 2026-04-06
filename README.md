# Amazon Fake Review Detector

Chrome Extension + ML Backend for detecting fake Amazon product reviews.

## Architecture

- **Chrome Extension** (Manifest V3) — scrapes Amazon review DOM, sends to API, injects scores
- **FastAPI Backend** — hosted on AWS EC2, serves ML predictions
- **Hybrid ML Model** — rule-based scorer + fine-tuned BERT, fused with weighted average + consistency check

## Project Structure

```
amazon-fake-review-detector/
├── data/               # datasets, preprocessing, EDA
│   ├── raw/            # downloaded datasets
│   ├── processed/      # cleaned & tokenized
│   └── notebooks/      # EDA & feature engineering
├── model/              # ML model code
│   ├── rule_engine/    # manual + learned rules
│   ├── bert/           # BERT fine-tuning
│   ├── baseline/       # TF-IDF + XGBoost
│   ├── fusion/         # hybrid score fusion
│   ├── configs/        # YAML configs
│   └── checkpoints/    # saved model weights
├── backend/            # FastAPI server
│   └── app/
│       ├── routes/     # API endpoints
│       ├── services/   # inference logic
│       └── schemas/    # Pydantic models
├── extension/          # Chrome extension
│   ├── scripts/        # JS (content, background, popup)
│   └── styles/         # CSS
└── notebooks/          # Colab training notebooks
```

## Quick Start

```bash
# Setup
python -m venv venv && source venv/bin/activate
pip install -r requirements.txt
cp .env.example .env

# Download data
python data/download.py

# Run backend locally
cd backend && uvicorn app.main:app --reload
```

## Build Phases

1. **Data** — download datasets, EDA, feature engineering
2. **Model** — rule engine, BERT fine-tuning, hybrid fusion
3. **Backend** — FastAPI deployment on AWS
4. **Extension** — Chrome extension with inline score injection
