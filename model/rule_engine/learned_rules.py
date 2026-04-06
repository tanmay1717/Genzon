"""
Genzon — Learned Rule-Based Scorer
Rules that are learned from data patterns (Section 4.1 — learned rules).

These rules look at reviewer profile signals and behavioral patterns
that can't be hard-coded — they need thresholds learned from data.
"""

import numpy as np
import pandas as pd
from sklearn.tree import DecisionTreeClassifier
from pathlib import Path
import pickle


class LearnedRuleScorer:
    """
    A lightweight decision tree that learns thresholds for reviewer behavior rules.
    
    Covers design doc signals:
        - Total reviews ever written (1-2 = suspicious)
        - % verified purchase reviews (low = red flag)
        - % 5-star ratings given (always 5 = suspicious)
        - Account age vs review count (new + many = fake)
        - Category diversity (only one brand = suspicious)
        - Review burst detection (many reviews in one day = fake campaign)
    
    Since the Kaggle dataset doesn't have all profile fields,
    we learn from the text-based features that ARE available.
    """

    def __init__(self):
        self.model = DecisionTreeClassifier(
            max_depth=5,          # keep it interpretable
            min_samples_leaf=50,  # prevent overfitting
            random_state=42,
        )
        self.feature_cols = []
        self.is_fitted = False
        self.thresholds = {}

    def fit(self, df: pd.DataFrame, label_col: str = "label_encoded"):
        """
        Learn rule thresholds from training data.
        
        Args:
            df: training DataFrame with numeric features and labels.
            label_col: name of the binary label column (0=genuine, 1=fake).
        """
        # Features we can learn rules from
        candidate_features = [
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
            "char_count",
        ]

        # Only use features that exist in the dataframe
        self.feature_cols = [f for f in candidate_features if f in df.columns]

        X = df[self.feature_cols].fillna(0).values
        y = df[label_col].values

        self.model.fit(X, y)
        self.is_fitted = True

        # Extract learned thresholds for interpretability
        self._extract_thresholds()

        # Print learned rules
        print(f"  ✓ Learned rules from {len(df)} samples using {len(self.feature_cols)} features")
        print(f"    Tree depth: {self.model.get_depth()}")
        print(f"    Training accuracy: {self.model.score(X, y):.3f}")

        return self

    def _extract_thresholds(self):
        """Extract the decision thresholds from the tree for interpretability."""
        tree = self.model.tree_
        self.thresholds = {}

        for node_id in range(tree.node_count):
            if tree.feature[node_id] >= 0:  # not a leaf
                feat_name = self.feature_cols[tree.feature[node_id]]
                threshold = tree.threshold[node_id]

                if feat_name not in self.thresholds:
                    self.thresholds[feat_name] = []
                self.thresholds[feat_name].append(round(threshold, 4))

    def predict_proba(self, df: pd.DataFrame) -> np.ndarray:
        """
        Predict fake probability for each review.
        
        Returns:
            numpy array of fake probabilities (0.0 to 1.0)
        """
        if not self.is_fitted:
            raise RuntimeError("Model not fitted. Call .fit() first.")

        X = df[self.feature_cols].fillna(0).values
        proba = self.model.predict_proba(X)

        # Return probability of class 1 (fake)
        return proba[:, 1]

    def score_single(self, features: dict) -> float:
        """
        Score a single review.
        
        Args:
            features: dict of feature_name → value
        Returns:
            float: fake probability (0.0 to 1.0)
        """
        if not self.is_fitted:
            raise RuntimeError("Model not fitted. Call .fit() first.")

        row = pd.DataFrame([features])
        # Fill missing features with 0
        for col in self.feature_cols:
            if col not in row.columns:
                row[col] = 0

        X = row[self.feature_cols].fillna(0).values
        proba = self.model.predict_proba(X)
        return float(proba[0, 1])

    def get_learned_rules(self) -> dict:
        """Return the learned thresholds for inspection."""
        return self.thresholds

    def print_rules(self):
        """Pretty-print the learned decision thresholds."""
        if not self.thresholds:
            print("  No rules learned yet. Call .fit() first.")
            return

        print("\n  Learned decision thresholds:")
        print("  " + "-" * 50)
        for feat, thresholds in sorted(self.thresholds.items()):
            vals = ", ".join(str(t) for t in thresholds)
            print(f"    {feat:<25} splits at: {vals}")

    def save(self, path: str | Path):
        """Save the trained model to disk."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)

        data = {
            "model": self.model,
            "feature_cols": self.feature_cols,
            "thresholds": self.thresholds,
            "is_fitted": self.is_fitted,
        }
        with open(path, "wb") as f:
            pickle.dump(data, f)
        print(f"  ✓ Saved learned rules to {path}")

    def load(self, path: str | Path):
        """Load a trained model from disk."""
        path = Path(path)
        with open(path, "rb") as f:
            data = pickle.load(f)

        self.model = data["model"]
        self.feature_cols = data["feature_cols"]
        self.thresholds = data["thresholds"]
        self.is_fitted = data["is_fitted"]
        print(f"  ✓ Loaded learned rules from {path}")
        return self


# ─── Quick test ──────────────────────────────────────────────

if __name__ == "__main__":
    import sys
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

    from data.config import PROCESSED_DIR

    # Load training data
    train_path = PROCESSED_DIR / "train.csv"
    if not train_path.exists():
        print("Run 'python -m data.preprocess' first!")
        exit(1)

    train = pd.read_csv(train_path)
    print(f"Loaded {len(train)} training samples\n")

    # Fit
    scorer = LearnedRuleScorer()
    scorer.fit(train)
    scorer.print_rules()

    # Test on a few samples
    print("\n  Sample predictions:")
    sample = train.head(5)
    probas = scorer.predict_proba(sample)
    for i, (_, row) in enumerate(sample.iterrows()):
        actual = "FAKE" if row["label_encoded"] == 1 else "GENUINE"
        pred = probas[i]
        print(f"    Review {i+1}: predicted={pred:.3f}, actual={actual}")

    # Save
    scorer.save(PROCESSED_DIR.parent.parent / "model" / "checkpoints" / "learned_rules.pkl")