import json
from datetime import datetime, timezone
from pathlib import Path


def log_prediction(
    log_path: str, ticker: str, model_version: int,
    closes: list[float], predicted_close: float,
    prediction_for_date: str, latency_ms: float, request_id: str,
) -> None:
    """Append de uma linha JSONL. Cria o diretório pai se necessário. Nunca lança exceção que
    interrompa a resposta da API — falha de logging é registrada em stderr, mas não derruba a
    requisição do usuário."""
    entry = {
        "request_id": request_id,
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "ticker": ticker,
        "model_version": model_version,
        "input_closes_last_value": closes[-1],
        "input_closes_mean": sum(closes) / len(closes),
        "predicted_close": predicted_close,
        "prediction_for_date": prediction_for_date,
        "latency_ms": latency_ms,
    }
    try:
        Path(log_path).parent.mkdir(parents=True, exist_ok=True)
        with open(log_path, "a", encoding="utf-8") as f:
            f.write(json.dumps(entry) + "\n")
    except OSError as e:
        print(f"[WARN] falha ao gravar log de predição: {e}")
