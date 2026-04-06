# Genzon — Amazon Fake Review Detector

Chrome Extension + ML Backend for detecting fake Amazon product reviews using a hybrid architecture combining rule-based heuristics and fine-tuned BERT.

## Architecture

- **Chrome Extension** (Manifest V3) — Scrapes Amazon review DOM, sends to API, injects genuineness scores (0-10) inline
- **FastAPI Backend** — Serves ML predictions locally or on AWS EC2
- **Hybrid ML Model** — Rule-based scorer + fine-tuned BERT, fused with weighted average + consistency check

## Project Structure

```
genzon/
├── data/                          # Phase 1 — Data pipeline
│   ├── raw/                       # Downloaded datasets (git-ignored)
│   ├── processed/                 # Cleaned & split CSVs (git-ignored)
│   │   ├── full_processed.csv
│   │   ├── train.csv
│   │   ├── val.csv
│   │   └── test.csv
│   ├── notebooks/
│   │   └── 01_eda.py             # Exploratory data analysis
│   ├── config.py                  # Paths, constants, feature columns
│   ├── download.py                # Dataset download script
│   ├── preprocess.py              # Cleaning, feature engineering, splits
│   └── eda.py                     # EDA helper functions
│
├── model/                         # Phase 2 — ML Models
│   ├── rule_engine/
│   │   ├── manual_rules.py        # Hand-crafted heuristic rules
│   │   ├── learned_rules.py       # Decision tree learned rules
│   │   └── scorer.py              # Combined rule-based scorer
│   ├── bert/
│   │   ├── dataset.py             # PyTorch Dataset for BERT
│   │   ├── train.py               # BERT fine-tuning script
│   │   ├── evaluate.py            # Full evaluation on test set
│   │   ├── predict.py             # Single-review inference
│   │   ├── quick_eval.py          # Fast CPU-friendly evaluation
│   │   └── diagnose.py            # Debug label/prediction issues
│   ├── baseline/
│   │   └── tfidf_xgb.py           # TF-IDF + XGBoost baseline
│   ├── fusion/
│   │   ├── fusion.py              # Hybrid fusion layer
│   │   └── calibration.py         # Weight tuning + probability calibration
│   ├── utils/
│   │   └── metrics.py             # Shared metric functions
│   ├── configs/
│   │   ├── training_config.yaml   # Hyperparameters
│   │   └── bert_config.yaml       # Model variants for ablation
│   └── checkpoints/               # Saved models (git-ignored)
│       ├── bert_best/             # ⬅ BERT model from Google Colab
│       ├── learned_rules.pkl
│       ├── tfidf_xgb.pkl
│       └── calibration.pkl
│
├── backend/                       # Phase 3 — FastAPI Server
│   ├── app/
│   │   ├── main.py                # App entry point
│   │   ├── config.py              # Environment settings
│   │   ├── routes/
│   │   │   ├── predict.py         # POST /api/v1/predict
│   │   │   └── health.py          # GET /health
│   │   ├── services/
│   │   │   ├── inference.py       # Model loading + hybrid prediction
│   │   │   └── preprocessing.py   # Clean incoming review data
│   │   └── schemas/
│   │       ├── request.py         # Pydantic request models
│   │       └── response.py        # Pydantic response models
│   ├── requirements.txt
│   └── Dockerfile
│
├── extension/                     # Phase 4 — Chrome Extension
│   ├── manifest.json              # Manifest V3 config
│   ├── popup.html                 # Extension popup UI
│   ├── icons/                     # Extension icons
│   ├── scripts/
│   │   ├── content.js             # DOM scraper + score injector
│   │   ├── background.js          # Service worker (routes API calls)
│   │   ├── popup.js               # Popup UI logic
│   │   └── api.js                 # API client (via background worker)
│   └── styles/
│       ├── content.css            # Injected badge styles on Amazon
│       └── popup.css              # Popup styles
│
├── notebooks/                     # Google Colab notebooks
│   └── train_bert_colab.ipynb     # BERT training notebook for Colab
│
├── requirements.txt               # Full project dependencies
├── .env.example                   # Environment variable template
├── .gitignore
└── README.md
```

## Quick Start

### 1. Setup

```bash
cd genzon
python -m venv venv
source venv/bin/activate        # Mac/Linux
# venv\Scripts\activate         # Windows
pip install -r requirements.txt
pip install pydantic-settings
cp .env.example .env
```

### 2. Download & Preprocess Data

```bash
python -m data.download --skip-amazon    # downloads Ott corpus, guides for Kaggle
python -m data.preprocess --no-smote     # clean, engineer features, split
```

**Note:** The Kaggle dataset requires manual download from [kaggle.com/datasets/mexwell/fake-reviews-dataset](https://www.kaggle.com/datasets/mexwell/fake-reviews-dataset). Download the CSV and place it in `data/raw/`.

### 3. Train Models Locally

```bash
# Rule-based scorer
python -m model.rule_engine.scorer

# TF-IDF + XGBoost baseline
python -m model.baseline.tfidf_xgb
```

### 4. Train BERT on Google Colab (GPU required)

BERT training requires a GPU. Use Google Colab (free T4 GPU):

1. Open [Google Colab](https://colab.research.google.com)
2. Go to `Runtime → Change runtime type → GPU (T4)`
3. Upload `notebooks/train_bert_colab.ipynb` or copy the code from it
4. When prompted, upload `data/processed/train.csv` and `data/processed/val.csv`
5. Run all cells — training takes ~20-30 minutes
6. At the end, it will download `bert_best.zip`

**⚠️ IMPORTANT — After Colab training:**

```bash
# Unzip the downloaded model into the checkpoints folder
unzip bert_best.zip -d model/checkpoints/
```

Your folder should look like:
```
model/checkpoints/bert_best/
├── config.json
├── model.safetensors
├── tokenizer.json
├── tokenizer_config.json
├── special_tokens_map.json
└── vocab.txt
```

Without these files, the backend cannot run BERT inference.

### 5. Verify BERT Works

```bash
# Quick single-review test (2-5 seconds)
python -m model.bert.predict "This product is amazing I love it"

# Fast evaluation on 200 random test samples (~2 min)
python -m model.bert.quick_eval --samples 200

# Full test set evaluation (~15-30 min on CPU)
python -m model.bert.quick_eval --full
```

### 6. Run Calibration (Optional but Recommended)

Finds optimal fusion weights and threshold:

```bash
python -m model.fusion.calibration
```

After running, update the printed weights in `backend/app/config.py`.

### 7. Start the Backend Server

```bash
uvicorn backend.app.main:app --reload --port 8000
```

The server loads all models on startup. Verify at:
- Health check: [http://localhost:8000/health](http://localhost:8000/health)
- API docs: [http://localhost:8000/docs](http://localhost:8000/docs)

Test with curl:
```bash
curl -X POST http://localhost:8000/api/v1/predict \
  -H "Content-Type: application/json" \
  -d '{
    "reviews": [
      {"review_text": "Great product, works perfectly!", "star_rating": 5},
      {"review_text": "AMAZING BUY NOW BEST EVER!!!", "star_rating": 5, "verified_purchase": false}
    ]
  }'
```

### 8. Load Chrome Extension

1. Open `chrome://extensions/` in Chrome
2. Enable **Developer mode** (toggle top right)
3. Click **Load unpacked** → select the `genzon/extension/` folder
4. Pin the extension to your toolbar (puzzle icon → pin Genzon)
5. Visit any Amazon product page — scores appear next to each review

**Note:** The backend server must be running for the extension to work.

## How It Works

### Scoring Pipeline

```
Amazon Review → Chrome Extension scrapes text + metadata
                        ↓
               Background worker sends to API
                        ↓
               FastAPI preprocesses (clean text, extract features)
                        ↓
            ┌───────────┴───────────┐
            ↓                       ↓
    Rule-Based Scorer          BERT Model
    (manual + learned)      (fine-tuned on 28K reviews)
            ↓                       ↓
            └───────────┬───────────┘
                        ↓
               Hybrid Fusion Layer
          (weighted avg + consistency check)
                        ↓
              Genuineness Score (0-10)
              0-4: Likely Fake (red)
              5-7: Uncertain (yellow)
              8-10: Likely Genuine (green)
```

### Rule-Based Scorer (Component 1)

**Manual rules** — hand-crafted heuristics:
- Unverified purchase → suspicious
- Star rating vs sentiment mismatch → suspicious
- Very short review → suspicious
- ALL CAPS / excessive punctuation → suspicious
- Low vocabulary diversity → suspicious

**Learned rules** — decision tree that discovers thresholds:
- Trained on 14 numeric features extracted from review text
- Learns patterns like "if caps_ratio > 0.15 AND word_count < 20 → likely fake"

### BERT Model (Component 2)

- Fine-tuned `bert-base-uncased` (110M parameters) on 28K labeled reviews
- Trained on Google Colab with T4 GPU, 5 epochs, FP16 mixed precision
- Catches subtle linguistic patterns that rules miss

### Hybrid Fusion (Component 3)

- Weighted average: `score = 0.35 × rule_score + 0.65 × BERT_score`
- If rule and BERT scores diverge by > 3 points, flags as "Uncertain"
- Weights can be optimized using `calibration.py`

## Datasets

| Dataset | Size | Labels | Use |
|---------|------|--------|-----|
| [Fake Reviews Dataset](https://www.kaggle.com/datasets/mexwell/fake-reviews-dataset) (Kaggle) | ~40K reviews | CG (fake) / OR (genuine) | Primary training + evaluation |
| [Deceptive Opinion Spam Corpus](https://myleott.com/op-spam.html) (Ott et al.) | ~1,600 reviews | Fake / genuine | Cross-domain benchmark |
| [Amazon Review Data](https://cseweb.ucsd.edu/~jmcauley/datasets/amazon_v2/) (McAuley Labs) | Millions | Unlabeled | BERT domain adaptation (optional) |

## Academic Extensions

1. **Imbalanced data handling** — SMOTE, class weights, threshold tuning
2. **Manual + learned rules** — hand-crafted heuristics + decision tree
3. **Model comparison** — TF-IDF+XGBoost vs BERT vs DistilBERT vs RoBERTa
4. **Hybrid integration** — weighted fusion, consistency checking, probability calibration

## Tech Stack

| Layer | Technology |
|-------|-----------|
| Chrome Extension | HTML + CSS + Vanilla JS (Manifest V3) |
| Backend API | Python + FastAPI |
| ML Training | scikit-learn + HuggingFace Transformers + XGBoost |
| Deep Learning | PyTorch + BERT |
| Cloud Training | Google Colab (T4 GPU) |
| Deployment | AWS EC2 + S3 + API Gateway (planned) |

## Troubleshooting

| Problem | Solution |
|---------|----------|
| `zsh: command not found: uvicorn` | Run `source venv/bin/activate` first |
| Extension shows "API offline" | Start the backend: `uvicorn backend.app.main:app --reload --port 8000` |
| BERT model not found | Download from Colab and unzip to `model/checkpoints/bert_best/` |
| Slow BERT inference on CPU | Use `quick_eval.py --samples 200` or run evaluation on Colab |
| Extension not appearing in Chrome | Make sure Developer mode is ON and you selected the `extension/` folder |
| `pydantic ValidationError: extra inputs` | Update `backend/app/config.py` — add `extra = "ignore"` in Config class |

## License

Academic project — for educational purposes.