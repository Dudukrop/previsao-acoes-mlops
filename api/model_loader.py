import json
from dataclasses import dataclass
from pathlib import Path

import joblib
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras import Model
from tensorflow.keras.models import load_model


@dataclass
class LoadedModel:
    model: Model
    scaler: MinMaxScaler
    version: int
    ticker: str
    lookback_window: int
    created_at: str
    evaluation_metrics: dict


def load_active_model(metadata_dir: str) -> LoadedModel:
    """Lê current_version.json para descobrir a versão ativa, carrega o .keras e o .joblib
    correspondentes descritos em model_metadata_v{N}.json.

    Levanta FileNotFoundError com mensagem clara se current_version.json não existir — isso
    significa que nenhum modelo foi promovido ainda (rode o pipeline de treino primeiro)."""
    current_path = Path(metadata_dir) / "current_version.json"
    if not current_path.exists():
        raise FileNotFoundError(
            f"{current_path} não encontrado. Rode o pipeline de treino "
            "(python -m src.training.run_training) antes de iniciar a API."
        )
    version = json.loads(current_path.read_text(encoding="utf-8"))["active_version"]

    metadata_path = Path(metadata_dir) / f"model_metadata_v{version}.json"
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))

    model = load_model(metadata["model_path"])
    scaler = joblib.load(metadata["scaler_path"])

    return LoadedModel(
        model=model,
        scaler=scaler,
        version=metadata["version"],
        ticker=metadata["ticker"],
        lookback_window=metadata["lookback_window"],
        created_at=metadata["created_at"],
        evaluation_metrics=metadata["evaluation_metrics"],
    )
