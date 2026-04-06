"""
Genzon — TF-IDF + XGBoost Baseline Model
Fast, interpretable baseline before moving to BERT (Section 4.2).

Usage:
    python -m model.baseline.tfidf_xgb
"""

import pickle
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import (
    accuracy_score, f1_score, precision_score, recall_score,
    classification_report, roc_auc_score,
)
from xgboost import XGBClassifier


class TfidfXgbModel:
    """TF-IDF vectorizer + XGBoost classifier pipeline."""

    def __init__(
        self,
        max_features: int = 10000,
        ngram_range: tuple = (1, 2),
        n_estimators: int = 300,
        max_depth: int = 6,
        learning_rate: float = 0.1,
    ):
        self.vectorizer = TfidfVectorizer(
            max_features=max_features,
            ngram_range=ngram_range,
            strip_accents="unicode",
            sublinear_tf=True,
        )

        self.model = XGBClassifier(
            n_estimators=n_estimators,
            max_depth=max_depth,
            learning_rate=learning_rate,
            eval_metric="logloss",
            random_state=42,
            n_jobs=-1,
        )

        self.is_fitted = False

    def fit(self, train_df: pd.DataFrame, text_col: str = "text_tfidf", label_col: str = "label_encoded"):
        """
        Train the TF-IDF + XGBoost pipeline.
        
        Args:
            train_df: training DataFrame
            text_col: column with cleaned text for TF-IDF
            label_col: binary label column
        """
        print("\n🔧 Training TF-IDF + XGBoost baseline...")

        texts = train_df[text_col].fillna("").values
        labels = train_df[label_col].values

        # Fit TF-IDF
        print(f"  ⏳ Fitting TF-IDF on {len(texts)} reviews...")
        X = self.vectorizer.fit_transform(texts)
        print(f"    ✓ Vocabulary size: {len(self.vectorizer.vocabulary_)}")
        print(f"    ✓ Feature matrix: {X.shape}")

        # Train XGBoost
        print(f"  ⏳ Training XGBoost...")
        self.model.fit(X, labels)
        self.is_fitted = True

        train_acc = self.model.score(X, labels)
        print(f"    ✓ Training accuracy: {train_acc:.4f}")

        return self

    def predict(self, df: pd.DataFrame, text_col: str = "text_tfidf") -> np.ndarray:
        """Predict binary labels (0=genuine, 1=fake)."""
        if not self.is_fitted:
            raise RuntimeError("Model not fitted. Call .fit() first.")

        texts = df[text_col].fillna("").values
        X = self.vectorizer.transform(texts)
        return self.model.predict(X)

    def predict_proba(self, df: pd.DataFrame, text_col: str = "text_tfidf") -> np.ndarray:
        """Predict fake probability (0.0 to 1.0)."""
        if not self.is_fitted:
            raise RuntimeError("Model not fitted. Call .fit() first.")

        texts = df[text_col].fillna("").values
        X = self.vectorizer.transform(texts)
        return self.model.predict_proba(X)[:, 1]

    def evaluate(self, df: pd.DataFrame, text_col: str = "text_tfidf", label_col: str = "label_encoded") -> dict:
        """Full evaluation with all metrics."""
        y_true = df[label_col].values
        y_pred = self.predict(df, text_col)
        y_proba = self.predict_proba(df, text_col)

        metrics = {
            "accuracy": accuracy_score(y_true, y_pred),
            "f1": f1_score(y_true, y_pred),
            "precision": precision_score(y_true, y_pred),
            "recall": recall_score(y_true, y_pred),
            "auc_roc": roc_auc_score(y_true, y_proba),
        }

        print("\n  TF-IDF + XGBoost Evaluation:")
        print("  " + "-" * 40)
        for k, v in metrics.items():
            bar = "█" * int(v * 30)
            print(f"    {k:<12} {v:.4f}  {bar}")

        print("\n  Classification Report:")
        print(classification_report(y_true, y_pred, target_names=["Genuine", "Fake"]))

        return metrics

    def get_top_features(self, n: int = 20) -> dict:
        """Get most important TF-IDF features from XGBoost."""
        if not self.is_fitted:
            return {}

        feature_names = self.vectorizer.get_feature_names_out()
        importances = self.model.feature_importances_

        top_idx = np.argsort(importances)[-n:][::-1]

        top_features = {}
        for idx in top_idx:
            top_features[feature_names[idx]] = float(importances[idx])

        return top_features

    def save(self, path: str | Path):
        """Save model + vectorizer to disk."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)

        data = {
            "vectorizer": self.vectorizer,
            "model": self.model,
            "is_fitted": self.is_fitted,
        }
        with open(path, "wb") as f:
            pickle.dump(data, f)
        print(f"  ✓ Saved TF-IDF+XGBoost to {path}")

    def load(self, path: str | Path):
        """Load model + vectorizer from disk."""
        with open(path, "rb") as f:
            data = pickle.load(f)

        self.vectorizer = data["vectorizer"]
        self.model = data["model"]
        self.is_fitted = data["is_fitted"]
        print(f"  ✓ Loaded TF-IDF+XGBoost from {path}")
        return self


# ─── Main ────────────────────────────────────────────────────

if __name__ == "__main__":
    import sys
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

    from data.config import PROCESSED_DIR

    # Load data
    train = pd.read_csv(PROCESSED_DIR / "train.csv")
    val = pd.read_csv(PROCESSED_DIR / "val.csv")
    test = pd.read_csv(PROCESSED_DIR / "test.csv")

    print(f"Train: {len(train)} | Val: {len(val)} | Test: {len(test)}")

    # Train
    model = TfidfXgbModel()
    model.fit(train)

    # Evaluate
    print("\n── Validation set ──")
    model.evaluate(val)

    print("\n── Test set ──")
    model.evaluate(test)

    # Top features
    print("\n  Top 20 TF-IDF features:")
    for word, importance in model.get_top_features(20).items():
        bar = "█" * int(importance * 500)
        print(f"    {word:<20} {importance:.4f}  {bar}")

    # Save
    model.save(Path("model/checkpoints/tfidf_xgb.pkl"))