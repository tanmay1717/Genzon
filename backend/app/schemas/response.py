from pydantic import BaseModel


class ReviewScore(BaseModel):
    score: float            # 0-10 genuineness score
    label: str              # "Likely Genuine" | "Uncertain" | "Likely Fake"
    rule_score: float       # rule-based sub-score
    bert_score: float       # BERT sub-score
    confidence: float       # model confidence
    flags: list[str] = []   # e.g., ["rule_ml_divergence", "unverified"]


class PredictResponse(BaseModel):
    review_scores: list[ReviewScore]
    aggregate_score: float
    aggregate_label: str
    total_reviews_analyzed: int
