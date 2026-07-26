from pathlib import Path

import os
import yaml
from dotenv import load_dotenv
from pydantic import BaseModel


class DataConfig(BaseModel):
    ticker: str
    start_date: str
    end_date: str | None = None
    raw_path: str
    processed_dir: str


class SplitConfig(BaseModel):
    test_size_ratio: float
    validation_size_ratio: float


class FeaturesConfig(BaseModel):
    lookback_window: int
    target_column: str
    use_columns: list[str]


class LstmConfig(BaseModel):
    units: list[int]
    dropout: float
    epochs: int
    batch_size: int
    early_stopping_patience: int
    learning_rate: float


class ArimaConfig(BaseModel):
    order: tuple[int, int, int]


class ModelConfig(BaseModel):
    type: str
    lstm: LstmConfig
    arima: ArimaConfig


class AcceptanceConfig(BaseModel):
    max_mape_pct: float
    must_beat_naive_baseline: bool


class EvaluationConfig(BaseModel):
    metrics: list[str]
    acceptance: AcceptanceConfig


class AppConfig(BaseModel):
    data: DataConfig
    split: SplitConfig
    features: FeaturesConfig
    model: ModelConfig
    evaluation: EvaluationConfig
    random_seed: int


def load_config(path: str = "config.yaml") -> AppConfig:
    """Carrega config.yaml e aplica overrides de variáveis de ambiente (.env via python-dotenv).
    TICKER em .env, se presente, sobrescreve data.ticker."""
    load_dotenv()
    raw = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
    if ticker_override := os.getenv("TICKER"):
        raw["data"]["ticker"] = ticker_override
    return AppConfig(**raw)
