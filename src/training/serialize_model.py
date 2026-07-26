import json
from datetime import datetime, timezone
from pathlib import Path

import joblib
import sklearn
import tensorflow as tf
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras import Model


def next_version(metadata_dir: str) -> int:
    """Lê os arquivos model_metadata_v*.json existentes e retorna max(N) + 1. Retorna 1 se
    nenhum artefato existir ainda."""
    existing = list(Path(metadata_dir).glob("model_metadata_v*.json"))
    if not existing:
        return 1
    versions = [int(p.stem.split("_v")[-1]) for p in existing]
    return max(versions) + 1


def serialize_artifacts(
    model: Model, scaler: MinMaxScaler,
    artifact_dir: str, metadata_dir: str,
    ticker: str, lookback_window: int,
    evaluation_metrics: dict, config_snapshot: dict,
    train_close_stats: dict,
) -> int:
    """Serializa modelo + scaler + metadados com versão incremental. Retorna o número da versão
    criada. NÃO promove automaticamente a versão a 'current' — isso é uma etapa manual via
    `promote_version()`."""
    version = next_version(metadata_dir)
    Path(artifact_dir).mkdir(parents=True, exist_ok=True)
    Path(metadata_dir).mkdir(parents=True, exist_ok=True)

    model_path = f"{artifact_dir}/lstm_model_v{version}.keras"
    scaler_path = f"{artifact_dir}/scaler_v{version}.joblib"
    model.save(model_path)
    joblib.dump(scaler, scaler_path)

    metadata = {
        "version": version,
        "ticker": ticker,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "lookback_window": lookback_window,
        "target_transform": "log_return",
        "model_path": model_path,
        "scaler_path": scaler_path,
        "evaluation_metrics": evaluation_metrics,
        "train_close_stats": train_close_stats,
        "config_snapshot": config_snapshot,
        "framework_versions": {
            "tensorflow": tf.__version__,
            "scikit-learn": sklearn.__version__,
        },
    }
    Path(f"{metadata_dir}/model_metadata_v{version}.json").write_text(
        json.dumps(metadata, indent=2), encoding="utf-8"
    )
    return version


def promote_version(metadata_dir: str, version: int) -> None:
    """Grava current_version.json apontando para `version` — é este arquivo que a API lê no
    startup, permitindo rollback manual para uma versão anterior."""
    Path(f"{metadata_dir}/current_version.json").write_text(
        json.dumps({"active_version": version}), encoding="utf-8"
    )
