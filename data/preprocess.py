"""
Genzon — Data Preprocessing & Feature Engineering

Processes the raw Kaggle Fake Reviews dataset into train/val/test splits
with engineered features for both the rule-based scorer and BERT.

Usage:
    python data/preprocess.py                  # full pipeline
    python data/preprocess.py --no-smote       # skip SMOTE (for initial EDA)
"""

import argparse
import re
import string

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from textblob import TextBlob

from data.config import (
    MAX_REVIEW_LENGTH,
    PROCESSED_DIR,
    RANDOM_SEED,
    RAW_DIR,
    TEST_SPLIT,
    VAL_SPLIT,
)


# ─── Load raw data ──────────────────────────────────────────


def load_kaggle_dataset() -> pd.DataFrame:
    """Load the Kaggle Fake Reviews CSV."""
    possible = [
        RAW_DIR / "fake reviews dataset.csv",
        RAW_DIR / "fake_reviews_dataset.csv",
        RAW_DIR / "Fake Reviews Dataset.csv",
    ]

    for path in possible:
        if path.exists():
            print(f"  ✓ Loading Kaggle dataset from {path}")
            df = pd.read_csv(path)
            print(f"    Shape: {df.shape}")
            print(f"    Columns: {list(df.columns)}")
            return df

    raise FileNotFoundError(
        f"Kaggle dataset not found in {RAW_DIR}/. Run: python data/download.py"
    )


# ─── Text cleaning ──────────────────────────────────────────


def clean_text(text: str) -> str:
    """Clean review text for model input."""
    if not isinstance(text, str):
        return ""

    # Normalize whitespace
    text = re.sub(r"\s+", " ", text).strip()

    # Remove URLs
    text = re.sub(r"http\S+|www\.\S+", "", text)

    # Remove HTML tags
    text = re.sub(r"<[^>]+>", "", text)

    return text


def clean_text_aggressive(text: str) -> str:
    """More aggressive cleaning for TF-IDF baseline (not for BERT)."""
    text = clean_text(text)
    text = text.lower()
    text = re.sub(r"[^a-z0-9\s]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


# ─── Feature engineering ────────────────────────────────────


def extract_text_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Extract rule-based features from review text.
    These feed into the rule-based scorer (Component 1 of the hybrid model).
    """
    print("  ⏳ Extracting text features...")

    # Basic length features
    df["char_count"] = df["text_clean"].str.len()
    df["word_count"] = df["text_clean"].str.split().str.len()
    df["avg_word_length"] = df["char_count"] / df["word_count"].replace(0, 1)

    # Caps and punctuation abuse (spam signals)
    df["caps_ratio"] = df["text"].apply(
        lambda x: sum(1 for c in str(x) if c.isupper()) / max(len(str(x)), 1)
    )
    df["exclamation_count"] = df["text"].str.count("!")
    df["question_count"] = df["text"].str.count(r"\?")

    # Punctuation density
    df["punct_ratio"] = df["text"].apply(
        lambda x: sum(1 for c in str(x) if c in string.punctuation) / max(len(str(x)), 1)
    )

    # Unique word ratio (lexical diversity — fakes often repeat phrases)
    df["unique_word_ratio"] = df["text_clean"].apply(
        lambda x: len(set(str(x).lower().split())) / max(len(str(x).split()), 1)
    )

    # Sentence count (very short single-sentence reviews are suspicious)
    df["sentence_count"] = df["text"].apply(
        lambda x: max(len(re.split(r"[.!?]+", str(x))), 1)
    )

    # Average sentence length
    df["avg_sentence_length"] = df["word_count"] / df["sentence_count"]

    # First person pronoun ratio (fakes tend to use more "I", "my", "me")
    first_person = {"i", "me", "my", "mine", "myself", "we", "our", "ours"}
    df["first_person_ratio"] = df["text_clean"].apply(
        lambda x: sum(1 for w in str(x).lower().split() if w in first_person)
        / max(len(str(x).split()), 1)
    )

    print(f"    ✓ Extracted {9} text features")
    return df


def extract_sentiment_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Extract sentiment polarity and subjectivity using TextBlob.
    The gap between star rating and sentiment is a strong fake signal.
    """
    print("  ⏳ Extracting sentiment features (this may take a minute)...")

    sentiments = df["text_clean"].apply(
        lambda x: TextBlob(str(x)).sentiment
    )

    df["sentiment_polarity"] = sentiments.apply(lambda s: s.polarity)    # -1 to 1
    df["sentiment_subjectivity"] = sentiments.apply(lambda s: s.subjectivity)  # 0 to 1

    # Star-sentiment gap (if rating column exists)
    if "rating" in df.columns:
        # Normalize rating to -1..1 scale: rating 1→-1, rating 3→0, rating 5→1
        df["rating_normalized"] = (df["rating"] - 3) / 2
        df["star_sentiment_gap"] = abs(df["rating_normalized"] - df["sentiment_polarity"])
    else:
        df["star_sentiment_gap"] = 0.0

    print("    ✓ Extracted sentiment features")
    return df


# ─── Label encoding ─────────────────────────────────────────


def encode_labels(df: pd.DataFrame) -> pd.DataFrame:
    """
    Standardize labels to binary: 0 = genuine, 1 = fake.
    Kaggle dataset uses 'CG' (computer generated = fake) and 'OR' (original = genuine).
    """
    label_col = None
    for col in ["label", "Label", "class", "Class"]:
        if col in df.columns:
            label_col = col
            break

    if label_col is None:
        raise ValueError(f"No label column found. Columns: {list(df.columns)}")

    print(f"  Label column: '{label_col}'")
    print(f"  Unique values: {df[label_col].unique()}")

    # Map to standard labels
    label_map = {
        "CG": 1,  # computer generated = fake
        "OR": 0,  # original = genuine
        "fake": 1,
        "genuine": 0,
        "FAKE": 1,
        "REAL": 0,
        1: 1,
        0: 0,
    }

    df["label_encoded"] = df[label_col].map(label_map)

    unmapped = df["label_encoded"].isna().sum()
    if unmapped > 0:
        print(f"  ⚠ {unmapped} rows with unmapped labels — dropping them")
        df = df.dropna(subset=["label_encoded"])

    df["label_encoded"] = df["label_encoded"].astype(int)

    n_fake = (df["label_encoded"] == 1).sum()
    n_genuine = (df["label_encoded"] == 0).sum()
    ratio = n_genuine / max(n_fake, 1)
    print(f"  Class distribution: Genuine={n_genuine}, Fake={n_fake} (ratio {ratio:.2f}:1)")

    return df


# ─── Train / Val / Test split ───────────────────────────────


def split_data(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Stratified split into train/val/test."""
    print(f"  Splitting: test={TEST_SPLIT}, val={VAL_SPLIT}, seed={RANDOM_SEED}")

    train_val, test = train_test_split(
        df,
        test_size=TEST_SPLIT,
        random_state=RANDOM_SEED,
        stratify=df["label_encoded"],
    )

    val_ratio = VAL_SPLIT / (1 - TEST_SPLIT)
    train, val = train_test_split(
        train_val,
        test_size=val_ratio,
        random_state=RANDOM_SEED,
        stratify=train_val["label_encoded"],
    )

    print(f"    Train: {len(train)} | Val: {len(val)} | Test: {len(test)}")
    return train, val, test


# ─── SMOTE oversampling ─────────────────────────────────────


def apply_smote(train: pd.DataFrame, feature_cols: list[str]) -> pd.DataFrame:
    """
    Apply SMOTE to the training set if classes are imbalanced.
    Only applied to numeric feature columns (not raw text).
    """
    from imblearn.over_sampling import SMOTE

    counts = train["label_encoded"].value_counts()
    ratio = counts.min() / counts.max()

    if ratio > 0.8:
        print("  ✓ Classes roughly balanced — skipping SMOTE")
        return train

    print(f"  ⏳ Applying SMOTE (minority ratio: {ratio:.2f})...")

    available_features = [c for c in feature_cols if c in train.columns]

    X = train[available_features].fillna(0)
    y = train["label_encoded"]

    smote = SMOTE(random_state=RANDOM_SEED, k_neighbors=5)
    X_res, y_res = smote.fit_resample(X, y)

    train_resampled = pd.DataFrame(X_res, columns=available_features)
    train_resampled["label_encoded"] = y_res

    print(f"    Before SMOTE: {len(train)} | After: {len(train_resampled)}")
    print(f"    New distribution: {train_resampled['label_encoded'].value_counts().to_dict()}")

    return train_resampled


# ─── Main pipeline ───────────────────────────────────────────


NUMERIC_FEATURE_COLS = [
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
    "star_sentiment_gap",
]


def main():
    parser = argparse.ArgumentParser(description="Preprocess Genzon datasets")
    parser.add_argument("--no-smote", action="store_true", help="Skip SMOTE resampling")
    args = parser.parse_args()

    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)

    print("=" * 60)
    print("  Genzon — Data Preprocessing Pipeline")
    print("=" * 60)

    # 1. Load
    print("\n[1/6] Loading raw data...")
    df = load_kaggle_dataset()

    # Identify the text column
    text_col = None
    for col in ["text_", "text", "review_text", "Review", "review"]:
        if col in df.columns:
            text_col = col
            break
    if text_col is None:
        raise ValueError(f"No text column found. Columns: {list(df.columns)}")

    df = df.rename(columns={text_col: "text"})

    # Identify rating column if present
    for col in ["rating", "Rating", "stars", "Stars", "star_rating"]:
        if col in df.columns and col != "rating":
            df = df.rename(columns={col: "rating"})
            break

    # 2. Clean
    print("\n[2/6] Cleaning text...")
    df = df.dropna(subset=["text"])
    df["text_clean"] = df["text"].apply(clean_text)
    df["text_tfidf"] = df["text"].apply(clean_text_aggressive)

    # Drop empty reviews
    empty_mask = df["text_clean"].str.strip() == ""
    if empty_mask.any():
        print(f"  Dropping {empty_mask.sum()} empty reviews")
        df = df[~empty_mask]

    print(f"  ✓ {len(df)} reviews after cleaning")

    # 3. Encode labels
    print("\n[3/6] Encoding labels...")
    df = encode_labels(df)

    # 4. Feature engineering
    print("\n[4/6] Feature engineering...")
    df = extract_text_features(df)
    df = extract_sentiment_features(df)

    # 5. Split
    print("\n[5/6] Splitting data...")
    train, val, test = split_data(df)

    # 6. SMOTE (optional, training set only)
    if not args.no_smote:
        print("\n[6/6] Imbalance handling...")
        train_smote = apply_smote(train, NUMERIC_FEATURE_COLS)
        train_smote.to_csv(PROCESSED_DIR / "train_smote.csv", index=False)
        print(f"  ✓ Saved SMOTE training set")
    else:
        print("\n[6/6] Skipping SMOTE (--no-smote)")

    # Save all splits
    train.to_csv(PROCESSED_DIR / "train.csv", index=False)
    val.to_csv(PROCESSED_DIR / "val.csv", index=False)
    test.to_csv(PROCESSED_DIR / "test.csv", index=False)

    # Save full processed dataset (for EDA notebook)
    df.to_csv(PROCESSED_DIR / "full_processed.csv", index=False)

    # Summary
    print("\n" + "=" * 60)
    print("  ✓ Preprocessing complete!")
    print("=" * 60)
    print(f"  Saved to: {PROCESSED_DIR}/")
    print(f"    full_processed.csv  — {len(df)} reviews (all features)")
    print(f"    train.csv           — {len(train)} reviews")
    print(f"    val.csv             — {len(val)} reviews")
    print(f"    test.csv            — {len(test)} reviews")
    if not args.no_smote:
        print(f"    train_smote.csv     — SMOTE-balanced training set")
    print(f"\n  Feature columns: {NUMERIC_FEATURE_COLS}")
    print(f"\n  Next step: Open data/notebooks/01_eda.ipynb for EDA")


if __name__ == "__main__":
    main()