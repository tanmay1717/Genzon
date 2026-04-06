"""
Genzon — Backend Configuration
Loads settings from environment variables or .env file.
"""

from pathlib import Path
from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    # API
    api_host: str = "0.0.0.0"
    api_port: int = 8000
    debug: bool = False

    # Model paths (relative to project root)
    project_root: Path = Path(__file__).resolve().parent.parent.parent
    bert_model_dir: str = "model/checkpoints/bert_best"
    rules_model_path: str = "model/checkpoints/learned_rules.pkl"
    tfidf_model_path: str = "model/checkpoints/tfidf_xgb.pkl"

    # Inference
    device: str = "auto"
    max_length: int = 512
    batch_size: int = 16

    # Fusion weights
    rule_weight: float = 0.30
    bert_weight: float = 0.70
    divergence_threshold: float = 3.0

    @property
    def bert_model_full_path(self) -> Path:
        return self.project_root / self.bert_model_dir

    @property
    def rules_model_full_path(self) -> Path:
        return self.project_root / self.rules_model_path

    @property
    def tfidf_model_full_path(self) -> Path:
        return self.project_root / self.tfidf_model_path

    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"
        extra = "ignore"  # ignore extra vars like AWS_*, WANDB_* in .env


settings = Settings()