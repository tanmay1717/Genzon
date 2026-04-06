"""
Genzon — Manual Rule-Based Scorer
Hand-crafted heuristic rules from the design doc (Section 4.1).

Each rule returns a suspicion score from 0.0 (genuine) to 1.0 (fake).
These are combined into an overall rule-based score.
"""

import re
import string


def rule_verified_purchase(verified: bool) -> float:
    """Unverified purchase = suspicious."""
    return 0.0 if verified else 0.7


def rule_star_sentiment_gap(star_rating: int, sentiment_polarity: float) -> float:
    """
    5 stars but negative text = suspicious.
    1 star but positive text = suspicious.
    
    star_rating: 1-5
    sentiment_polarity: -1.0 to 1.0 (from TextBlob)
    """
    # Normalize star to -1..1 scale: 1→-1, 3→0, 5→1
    star_normalized = (star_rating - 3) / 2

    gap = abs(star_normalized - sentiment_polarity)

    # gap ranges from 0 to ~2. Normalize to 0..1
    return min(gap / 2.0, 1.0)


def rule_review_length(word_count: int) -> float:
    """
    Extremely short reviews are low-confidence / suspicious.
    Very long reviews are slightly more trustworthy.
    """
    if word_count < 10:
        return 0.8  # very short = suspicious
    elif word_count < 25:
        return 0.5  # short
    elif word_count < 50:
        return 0.3  # moderate
    else:
        return 0.1  # long = likely genuine


def rule_has_media(has_photo_or_video: bool) -> float:
    """Reviews with photos/videos are more likely genuine."""
    return 0.1 if has_photo_or_video else 0.4


def rule_helpful_votes(helpful_count: int) -> float:
    """Community validation — more helpful votes = more genuine."""
    if helpful_count >= 10:
        return 0.05
    elif helpful_count >= 3:
        return 0.15
    elif helpful_count >= 1:
        return 0.3
    else:
        return 0.5  # no votes = neutral-suspicious


def rule_caps_abuse(text: str) -> float:
    """
    Excessive caps or punctuation = spam signal.
    Normal writing has ~2-5% caps ratio.
    """
    if not text or len(text) == 0:
        return 0.5

    caps_ratio = sum(1 for c in text if c.isupper()) / len(text)
    exclamation_count = text.count("!")

    score = 0.0
    if caps_ratio > 0.3:
        score += 0.5  # lots of CAPS
    elif caps_ratio > 0.15:
        score += 0.2

    if exclamation_count > 5:
        score += 0.3  # too many !!!
    elif exclamation_count > 2:
        score += 0.1

    return min(score, 1.0)


def rule_lexical_diversity(text: str) -> float:
    """
    Low unique word ratio = repetitive/templated text = suspicious.
    Genuine reviews tend to have more diverse vocabulary.
    """
    if not text:
        return 0.5

    words = text.lower().split()
    if len(words) == 0:
        return 0.5

    unique_ratio = len(set(words)) / len(words)

    if unique_ratio < 0.4:
        return 0.8  # very repetitive
    elif unique_ratio < 0.6:
        return 0.5
    elif unique_ratio < 0.8:
        return 0.3
    else:
        return 0.1  # diverse vocabulary


def compute_manual_rule_score(
    text: str,
    star_rating: int = 3,
    sentiment_polarity: float = 0.0,
    verified_purchase: bool = True,
    has_media: bool = False,
    helpful_votes: int = 0,
    word_count: int | None = None,
) -> dict:
    """
    Compute all manual rule scores and return a weighted average.

    Returns:
        dict with individual rule scores and final combined score.
        Score 0.0 = likely genuine, 1.0 = likely fake.
    """
    if word_count is None:
        word_count = len(text.split()) if text else 0

    # Individual rule scores
    scores = {
        "verified_purchase": rule_verified_purchase(verified_purchase),
        "star_sentiment_gap": rule_star_sentiment_gap(star_rating, sentiment_polarity),
        "review_length": rule_review_length(word_count),
        "has_media": rule_has_media(has_media),
        "helpful_votes": rule_helpful_votes(helpful_votes),
        "caps_abuse": rule_caps_abuse(text),
        "lexical_diversity": rule_lexical_diversity(text),
    }

    # Weights — how much each rule matters
    weights = {
        "verified_purchase": 0.20,
        "star_sentiment_gap": 0.20,
        "review_length": 0.10,
        "has_media": 0.10,
        "helpful_votes": 0.10,
        "caps_abuse": 0.15,
        "lexical_diversity": 0.15,
    }

    # Weighted average
    combined = sum(scores[k] * weights[k] for k in scores)
    scores["combined_manual_score"] = round(combined, 4)

    return scores


# ─── Quick test ──────────────────────────────────────────────

if __name__ == "__main__":
    # Test with a likely genuine review
    genuine = compute_manual_rule_score(
        text="I bought this blender 3 months ago and it still works perfectly. "
             "The motor is powerful and it blends smoothly. Great value for money.",
        star_rating=5,
        sentiment_polarity=0.65,
        verified_purchase=True,
        has_media=True,
        helpful_votes=12,
    )
    print("Genuine review scores:")
    for k, v in genuine.items():
        print(f"  {k:<25} {v:.3f}")

    print()

    # Test with a likely fake review
    fake = compute_manual_rule_score(
        text="AMAZING PRODUCT!!! BEST EVER!!! BUY NOW!!!",
        star_rating=5,
        sentiment_polarity=-0.1,
        verified_purchase=False,
        has_media=False,
        helpful_votes=0,
    )
    print("Fake review scores:")
    for k, v in fake.items():
        print(f"  {k:<25} {v:.3f}")