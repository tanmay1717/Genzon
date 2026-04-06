"""
Genzon — Combined Rule-Based Scorer
Merges manual rules + learned rules into a single rule-based score.
This is Component 1 of the hybrid architecture (Section 4.1).
"""

import pandas as pd
import numpy as np
from pathlib import Path

from model.rule_engine.manual_rules import compute_manual_rule_score
from model.rule_engine.learned_rules import LearnedRuleScorer


class RuleBasedScorer:
    """
    Combines manual heuristic rules and learned decision-tree rules
    into a single rule-based suspicion score (0 = genuine, 1 = fake).
    """

    def __init__(self, manual_weight: float = 0.4, learned_weight: float = 0.6):
        """
        Args:
            manual_weight: weight for hand-crafted rules (default 0.4)
            learned_weight: weight for learned rules (default 0.6)
        """
        self.manual_weight = manual_weight
        self.learned_weight = learned_weight
        self.learned_scorer = LearnedRuleScorer()
        self.is_fitted = False

    def fit(self, train_df: pd.DataFrame, label_col: str = "label_encoded"):
        """Train the learned rules component on training data."""
        print("\n🔧 Training Rule-Based Scorer...")
        print(f"  Weights: manual={self.manual_weight}, learned={self.learned_weight}")

        self.learned_scorer.fit(train_df, label_col)
        self.is_fitted = True

        print("  ✓ Rule-based scorer ready")
        return self

    def score_dataframe(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Score an entire DataFrame of reviews.
        
        Adds columns:
            - manual_rule_score: score from hand-crafted rules
            - learned_rule_score: score from decision tree
            - combined_rule_score: weighted combination
        
        Returns:
            DataFrame with score columns added.
        """
        if not self.is_fitted:
            raise RuntimeError("Scorer not fitted. Call .fit() first.")

        result = df.copy()

        # ── Manual rules ──
        # We use text-based features since the Kaggle dataset
        # doesn't have verified_purchase, helpful_votes, etc.
        manual_scores = []
        for _, row in df.iterrows():
            text = str(row.get("text_clean", row.get("text", "")))
            star = int(row.get("rating", 3))
            sentiment = float(row.get("sentiment_polarity", 0.0))
            word_count = int(row.get("word_count", len(text.split())))

            score = compute_manual_rule_score(
                text=text,
                star_rating=star,
                sentiment_polarity=sentiment,
                verified_purchase=True,   # not available in dataset
                has_media=False,           # not available in dataset
                helpful_votes=0,           # not available in dataset
                word_count=word_count,
            )
            manual_scores.append(score["combined_manual_score"])

        result["manual_rule_score"] = manual_scores

        # ── Learned rules ──
        result["learned_rule_score"] = self.learned_scorer.predict_proba(df)

        # ── Combined ──
        result["combined_rule_score"] = (
            self.manual_weight * result["manual_rule_score"]
            + self.learned_weight * result["learned_rule_score"]
        )

        return result

    def score_single(self, review: dict) -> dict:
        """
        Score a single review.
        
        Args:
            review: dict with keys like text, rating, sentiment_polarity,
                    word_count, caps_ratio, etc.
        Returns:
            dict with manual_score, learned_score, combined_score
        """
        # Manual rules
        manual = compute_manual_rule_score(
            text=review.get("text", ""),
            star_rating=review.get("rating", 3),
            sentiment_polarity=review.get("sentiment_polarity", 0.0),
            verified_purchase=review.get("verified_purchase", True),
            has_media=review.get("has_media", False),
            helpful_votes=review.get("helpful_votes", 0),
            word_count=review.get("word_count"),
        )

        # Learned rules
        learned = self.learned_scorer.score_single(review)

        # Combine
        combined = (
            self.manual_weight * manual["combined_manual_score"]
            + self.learned_weight * learned
        )

        return {
            "manual_score": manual["combined_manual_score"],
            "manual_details": manual,
            "learned_score": learned,
            "combined_rule_score": round(combined, 4),
        }

    def evaluate(self, df: pd.DataFrame, label_col: str = "label_encoded") -> dict:
        """
        Evaluate rule-based scorer on a dataset.
        
        Returns:
            dict with accuracy, f1, precision, recall
        """
        from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score

        scored = self.score_dataframe(df)
        y_true = df[label_col].values

        # Convert scores to binary predictions using 0.5 threshold
        y_pred = (scored["combined_rule_score"] >= 0.5).astype(int).values

        metrics = {
            "accuracy": accuracy_score(y_true, y_pred),
            "f1": f1_score(y_true, y_pred),
            "precision": precision_score(y_true, y_pred),
            "recall": recall_score(y_true, y_pred),
        }

        print("\n  Rule-Based Scorer Evaluation:")
        print("  " + "-" * 40)
        for k, v in metrics.items():
            bar = "█" * int(v * 30)
            print(f"    {k:<12} {v:.4f}  {bar}")

        return metrics

    def save(self, path: str | Path):
        """Save the learned component."""
        self.learned_scorer.save(path)

    def load(self, path: str | Path):
        """Load the learned component."""
        self.learned_scorer.load(path)
        self.is_fitted = True
        return self


# ─── Quick test ──────────────────────────────────────────────

if __name__ == "__main__":
    import sys
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

    from data.config import PROCESSED_DIR

    train = pd.read_csv(PROCESSED_DIR / "train.csv")
    val = pd.read_csv(PROCESSED_DIR / "val.csv")

    print(f"Train: {len(train)} | Val: {len(val)}")

    # Fit and evaluate
    scorer = RuleBasedScorer()
    scorer.fit(train)

    print("\n── Training set ──")
    scorer.evaluate(train)

    print("\n── Validation set ──")
    scorer.evaluate(val)

    # Save
    scorer.save(Path("model/checkpoints/learned_rules.pkl"))