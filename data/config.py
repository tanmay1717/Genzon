"""
Genzon — Data configuration and path constants.
"""

from pathlib import Path

# ─── Paths ──────────────────────────────────────────────────

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "data"
RAW_DIR = DATA_DIR / "raw"
PROCESSED_DIR = DATA_DIR / "processed"
MODEL_DIR = PROJECT_ROOT / "model"
CHECKPOINTS_DIR = MODEL_DIR / "checkpoints"

# ─── Dataset identifiers ────────────────────────────────────

KAGGLE_DATASET = "mexwell/fake-reviews-dataset"  # primary labeled dataset (~40K reviews)
OTT_CORPUS_URL = "https://myleott.com/op_spam_v1.4.zip"  # cross-domain benchmark
MCAULEY_BASE_URL = "https://datarepo.eng.ucsd.edu/mcauley_group/data/amazon_2023/"

# ─── Processing constants ───────────────────────────────────

MAX_REVIEW_LENGTH = 512   # BERT max token length
TEST_SPLIT = 0.15
VAL_SPLIT = 0.15
RANDOM_SEED = 42

# ─── Label encoding ─────────────────────────────────────────

LABEL_GENUINE = 0
LABEL_FAKE = 1
LABEL_NAMES = {0: "genuine", 1: "fake"}

# ─── Feature columns (used across preprocessing & training) ─

NUMERIC_FEATURES = [
    "char_count",
    "word_count",
    "avg_word_length",
    "caps_ratio",
    "exclamation_count",
    "question_count",
    "punct_ratio",
    "unique_word_ratio",
    "sentence_count",
    "avg_sentence_length",
    "first_person_ratio",
    "sentiment_polarity",
    "sentiment_subjectivity",
    "star_sentiment_gap"
]

TEXT_COL = "text_clean"         # cleaned text for BERT
TEXT_COL_TFIDF = "text_tfidf"   # aggressively cleaned text for TF-IDF baseline
LABEL_COL = "label_encoded"     # binary: 0=genuine, 1=fake