"""
Genzon — Hybrid Fusion Layer
Combines Rule-Based scores + BERT scores into a final genuineness score.
Covers academic extension #4: hybrid integration strategies.

Strategies:
    1. Weighted average
    2. Consistency check (flag divergence)
    3. Confidence-weighted fusion
"""

from pathlib import Path

import numpy as np
import pandas as pd

from model.rule_engine.scorer import RuleBasedScorer
from model.bert.predict import BertPredictor
from model.utils.metrics import compute_metrics, print_metrics


SCORE_LABELS = {
    (8, 10): "Likely Genuine",
    (5, 7):  "Uncertain",
    (0, 4):  "Likely Fake",
}

BADGE_COLORS = {
    "Likely Genuine": "green",
    "Uncertain": "amber",
    "Likely Fake": "red",
    "Uncertain (Divergence)": "gray",
}


def score_to_label(score: float) -> str:
    """Convert 0-10 score to a human-readable label."""
    if score >= 8:
        return "Likely Genuine"
    elif score >= 5:
        return "Uncertain"
    else:
        return "Likely Fake"


class HybridFusionModel:
    """
    Fuses rule-based and BERT scores using configurable strategies.
    
    Final score is 0-10 (genuineness):
        10 = definitely genuine
         0 = definitely fake
    """

    def __init__(
        self,
        rule_weight: float = 0.35,
        bert_weight: float = 0.65,
        divergence_threshold: float = 3.0,
        bert_model_dir: str | Path = "model/checkpoints/bert_best",
    ):
        """
        Args:
            rule_weight: weight for rule-based score (default 0.35)
            bert_weight: weight for BERT score (default 0.65)
            divergence_threshold: if rule and BERT scores differ by more
                                  than this (on 0-10 scale), flag as uncertain
            bert_model_dir: path to saved BERT model
        """
        self.rule_weight = rule_weight
        self.bert_weight = bert_weight
        self.divergence_threshold = divergence_threshold
        self.bert_model_dir = bert_model_dir

        self.rule_scorer = RuleBasedScorer()
        self.bert_predictor = None  # lazy load

    def fit_rules(self, train_df: pd.DataFrame, label_col: str = "label_encoded"):
        """Train the rule-based component."""
        self.rule_scorer.fit(train_df, label_col)
        return self

    def _load_bert(self):
        """Lazy-load BERT model (heavy, only when needed)."""
        if self.bert_predictor is None:
            print("  Loading BERT model...")
            self.bert_predictor = BertPredictor(self.bert_model_dir)
            print("  ✓ BERT loaded")

    def score_review(self, review: dict) -> dict:
        """
        Score a single review using both components.
        
        Args:
            review: dict with at least 'text' key. Optional: 'rating',
                    'sentiment_polarity', 'verified_purchase', etc.
        
        Returns:
            dict with final score (0-10), label, component scores, flags
        """
        self._load_bert()

        text = review.get("text", "")

        # ── Rule-based score (0=genuine, 1=fake) ──
        rule_result = self.rule_scorer.score_single(review)
        rule_fake_prob = rule_result["combined_rule_score"]

        # ── BERT score (0=genuine, 1=fake) ──
        bert_fake_prob = self.bert_predictor.get_bert_score(text)

        # ── Fusion ──
        return self._fuse_scores(rule_fake_prob, bert_fake_prob)

    def _fuse_scores(self, rule_fake_prob: float, bert_fake_prob: float) -> dict:
        """
        Combine rule and BERT fake-probabilities into a final genuineness score.
        
        Both inputs are fake probabilities: 0.0 = genuine, 1.0 = fake.
        Output genuineness score: 0 = fake, 10 = genuine.
        """
        # Convert to 0-10 genuineness scores (invert: high = genuine)
        rule_score_10 = round((1 - rule_fake_prob) * 10, 1)
        bert_score_10 = round((1 - bert_fake_prob) * 10, 1)

        # Weighted average
        fused_fake_prob = (
            self.rule_weight * rule_fake_prob
            + self.bert_weight * bert_fake_prob
        )
        final_score = round((1 - fused_fake_prob) * 10, 1)
        final_score = max(0, min(10, final_score))  # clamp to 0-10

        # Divergence check
        divergence = abs(rule_score_10 - bert_score_10)
        flags = []

        if divergence > self.divergence_threshold:
            flags.append("rule_bert_divergence")
            label = "Uncertain (Divergence)"
            badge_color = "gray"
        else:
            label = score_to_label(final_score)
            badge_color = BADGE_COLORS.get(label, "gray")

        return {
            "score": final_score,
            "label": label,
            "badge_color": badge_color,
            "rule_score": rule_score_10,
            "bert_score": bert_score_10,
            "confidence": round(1 - (divergence / 10), 2),
            "divergence": round(divergence, 1),
            "flags": flags,
        }

    def score_reviews(self, reviews: list[dict]) -> list[dict]:
        """Score multiple reviews."""
        return [self.score_review(r) for r in reviews]

    def aggregate_product_score(self, review_scores: list[dict]) -> dict:
        """
        Compute an aggregate product score from all review scores.
        This is shown at the top of the review section in the extension.
        """
        if not review_scores:
            return {"score": 5.0, "label": "Uncertain", "total_reviews": 0}

        scores = [r["score"] for r in review_scores]
        avg_score = round(np.mean(scores), 1)

        # Count by category
        genuine_count = sum(1 for s in scores if s >= 8)
        uncertain_count = sum(1 for s in scores if 5 <= s < 8)
        fake_count = sum(1 for s in scores if s < 5)
        flagged_count = sum(1 for r in review_scores if r.get("flags"))

        return {
            "score": avg_score,
            "label": score_to_label(avg_score),
            "badge_color": BADGE_COLORS.get(score_to_label(avg_score), "gray"),
            "total_reviews": len(review_scores),
            "genuine_count": genuine_count,
            "uncertain_count": uncertain_count,
            "fake_count": fake_count,
            "flagged_count": flagged_count,
        }

    def evaluate_on_dataset(
        self, df: pd.DataFrame, label_col: str = "label_encoded",
    ) -> dict:
        """
        Evaluate the full hybrid model on a dataset.
        Uses precomputed rule scores + BERT batch predictions.
        """
        self._load_bert()

        print("\n  Scoring with rule engine...")
        scored_df = self.rule_scorer.score_dataframe(df)
        rule_fake_probs = scored_df["combined_rule_score"].values

        print("  Scoring with BERT (this may take a while)...")
        texts = df["text_clean"].fillna("").tolist()
        bert_fake_probs = np.array([
            self.bert_predictor.get_bert_score(t) for t in texts
        ])

        # Fuse
        fused_results = []
        for r, b in zip(rule_fake_probs, bert_fake_probs):
            fused_results.append(self._fuse_scores(r, b))

        # Extract predictions
        y_true = df[label_col].values
        genuineness_scores = np.array([r["score"] for r in fused_results])
        # Convert genuineness (0-10) back to binary fake prediction
        y_pred = (genuineness_scores < 5).astype(int)
        y_proba = 1 - (genuineness_scores / 10)  # convert to fake probability

        metrics = compute_metrics(y_true, y_pred, y_proba)
        print_metrics(metrics, title="Hybrid Fusion Model")

        # Divergence stats
        n_flagged = sum(1 for r in fused_results if "rule_bert_divergence" in r.get("flags", []))
        print(f"\n    Flagged (divergence > {self.divergence_threshold}): {n_flagged}/{len(df)}")

        return metrics


# ─── Main ────────────────────────────────────────────────────

if __name__ == "__main__":
    import sys
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

    from data.config import PROCESSED_DIR

    train = pd.read_csv(PROCESSED_DIR / "train.csv")
    test = pd.read_csv(PROCESSED_DIR / "test.csv")

    print(f"Train: {len(train)} | Test: {len(test)}")

    # Build hybrid model
    hybrid = HybridFusionModel()
    hybrid.fit_rules(train)

    # Quick single-review demo
    demo = {
        "text": "Great product, works as advertised. Shipped fast.",
        "rating": 5,
        "sentiment_polarity": 0.5,
        "word_count": 8,
    }
    print("\n  Demo review:", demo["text"])
    result = hybrid.score_review(demo)
    print(f"    Score: {result['score']}/10 ({result['label']})")
    print(f"    Rule: {result['rule_score']}/10 | BERT: {result['bert_score']}/10")
    print(f"    Divergence: {result['divergence']} | Flags: {result['flags']}")