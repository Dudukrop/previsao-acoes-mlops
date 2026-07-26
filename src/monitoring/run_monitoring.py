"""
Uso:
    python -m src.monitoring.run_monitoring

Lê monitoring/logs/predictions.jsonl, busca o fechamento real via yfinance para as predições
cuja data-alvo já passou, calcula erro de produção e drift de entrada, e grava
monitoring/reports/monitoring_report_{data}.json.
"""

import json
import os
from datetime import date, datetime, timedelta, timezone
from pathlib import Path

from dotenv import load_dotenv

from src.data_collection.fetch_data import fetch_historical_data
from src.evaluation.metrics import evaluate_all

load_dotenv()

PREDICTION_LOG_PATH = os.getenv("PREDICTION_LOG_PATH", "monitoring/logs/predictions.jsonl")
MODEL_METADATA_DIR = os.getenv("MODEL_METADATA_DIR", "models/metadata")
WINDOW_DAYS = 30
Z_THRESHOLD = 3.0


def load_predictions(log_path: str) -> list[dict]:
    path = Path(log_path)
    if not path.exists():
        return []
    lines = path.read_text(encoding="utf-8").strip().splitlines()
    return [json.loads(line) for line in lines if line.strip()]


def load_active_metadata(metadata_dir: str) -> dict:
    current_path = Path(metadata_dir) / "current_version.json"
    version = json.loads(current_path.read_text(encoding="utf-8"))["active_version"]
    metadata_path = Path(metadata_dir) / f"model_metadata_v{version}.json"
    return json.loads(metadata_path.read_text(encoding="utf-8"))


def fetch_actual_close(ticker: str, target_date: date) -> float | None:
    """Busca o Close real de `target_date`. Retorna None se não houver pregão nessa data
    (feriado) — o chamador deve descartar a predição correspondente da amostra avaliada."""
    try:
        df = fetch_historical_data(
            ticker, target_date.isoformat(), (target_date + timedelta(days=1)).isoformat()
        )
    except ValueError:
        return None
    if df.empty:
        return None
    return float(df["Close"].iloc[0])


def check_drift(
    recent_mean: float, train_mean: float, train_std: float, z_threshold: float = Z_THRESHOLD
) -> bool:
    """Retorna True se a média recente estiver a mais de `z_threshold` desvios-padrão de treino
    da média de treino — heurística simples de drift."""
    if train_std == 0:
        return False
    z = abs(recent_mean - train_mean) / train_std
    return z > z_threshold


def main() -> None:
    predictions = load_predictions(PREDICTION_LOG_PATH)
    metadata = load_active_metadata(MODEL_METADATA_DIR)

    today = date.today()
    window_start = today - timedelta(days=WINDOW_DAYS)

    evaluated_true, evaluated_pred, recent_input_means = [], [], []
    for entry in predictions:
        ts = datetime.fromisoformat(entry["timestamp"]).date()
        if ts < window_start:
            continue
        recent_input_means.append(entry["input_closes_mean"])

        target_date = date.fromisoformat(entry["prediction_for_date"])
        if target_date >= today:
            continue  # ainda não temos o fechamento real dessa data
        actual = fetch_actual_close(entry["ticker"], target_date)
        if actual is None:
            continue
        evaluated_true.append(actual)
        evaluated_pred.append(entry["predicted_close"])

    production_metrics = (
        evaluate_all(_to_array(evaluated_true), _to_array(evaluated_pred))
        if evaluated_true
        else {"mae": None, "rmse": None, "mape": None}
    )

    training_metrics_reference = metadata["evaluation_metrics"]
    degradation_detected = (
        production_metrics["mape"] is not None
        and training_metrics_reference["mape"] is not None
        and production_metrics["mape"] > 1.5 * training_metrics_reference["mape"]
    )

    train_stats = metadata.get("train_close_stats", {})
    input_drift_detected = (
        bool(recent_input_means)
        and train_stats
        and check_drift(
            sum(recent_input_means) / len(recent_input_means),
            train_stats["mean"], train_stats["std"],
        )
    )

    report = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "window_days": WINDOW_DAYS,
        "n_predictions_evaluated": len(evaluated_true),
        "production_metrics": production_metrics,
        "training_metrics_reference": training_metrics_reference,
        "degradation_detected": degradation_detected,
        "input_drift_detected": input_drift_detected,
    }

    reports_dir = Path("monitoring/reports")
    reports_dir.mkdir(parents=True, exist_ok=True)
    report_path = reports_dir / f"monitoring_report_{today.isoformat()}.json"
    report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"[monitoramento] relatorio salvo em {report_path}")
    print(json.dumps(report, indent=2))


def _to_array(values: list[float]):
    import numpy as np

    return np.array(values)


if __name__ == "__main__":
    main()
