# 10 — Monitoramento e Observabilidade em Produção

## 1. O que monitorar

| Categoria | Métrica | Por quê |
|---|---|---|
| Operacional | Latência por requisição, taxa de erro (4xx/5xx) | Saúde do serviço |
| Qualidade de entrada (data drift) | Distribuição de `closes` recebidos vs. distribuição do treino | Detectar se o modelo está recebendo entradas muito diferentes do que aprendeu |
| Qualidade de predição (model drift) | Erro real (MAE/MAPE) comparando predições passadas com o fechamento real, assim que disponível | Detectar degradação do modelo ao longo do tempo |

## 2. Logging de cada requisição

Toda chamada a `POST /predict/by-ticker` ou `POST /predict` (doc 08) grava uma linha em
`monitoring/logs/predictions.jsonl` (formato JSON Lines — um objeto JSON por linha, apend-only,
fácil de processar depois). O schema de log é idêntico para os dois endpoints, pois ambos
convergem para a mesma chamada de `predict_next_close` antes de logar.

**Schema de cada linha:**

```json
{
  "request_id": "b3f1c2e0-....",
  "timestamp": "2026-07-16T14:32:01Z",
  "ticker": "PETR4.SA",
  "model_version": 1,
  "input_closes_last_value": 32.45,
  "input_closes_mean": 31.02,
  "predicted_close": 33.12,
  "prediction_for_date": "2026-07-17",
  "latency_ms": 84.2
}
```

Implementado como middleware FastAPI (`api/logging_middleware.py`), para não misturar
responsabilidade de logging dentro da lógica de inferência.

```python
# api/logging_middleware.py
import json, time, uuid
from datetime import datetime, timezone
from pathlib import Path

def log_prediction(log_path: str, ticker: str, model_version: int,
                    closes: list[float], predicted_close: float,
                    prediction_for_date: str, latency_ms: float, request_id: str) -> None:
    """Append de uma linha JSONL. Cria o diretório pai se necessário. Nunca lança exceção que
    interrompa a resposta da API — falha de logging é registrada em stderr, mas não derruba a
    requisição do usuário (logging é observabilidade, não deve virar ponto único de falha)."""
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
```

## 3. Job de monitoramento (offline, executado sob demanda)

Local: `src/monitoring/run_monitoring.py` — não roda dentro da API; é um script separado, pensado
para ser executado periodicamente (manualmente para o escopo do desafio; um cron/GitHub Actions
agendado é a evolução natural).

**Passos:**

1. Ler `monitoring/logs/predictions.jsonl`.
2. Para cada predição cuja `prediction_for_date` já passou, buscar o `Close` real daquele dia via
   `yfinance` (reutiliza `fetch_historical_data`, doc 02) e calcular o erro real
   (`|predicted_close - close_real|`, `%`).
3. Calcular MAE/MAPE de produção sobre a janela recente (ex.: últimos 30 dias) e comparar com o
   `evaluation_metrics` do `model_metadata_v{N}.json` (doc 06) — se o erro em produção estiver
   **significativamente pior** (ex. MAPE de produção > 1.5x o MAPE de teste registrado), emitir um
   alerta.
4. **Data drift de entrada:** calcular a média/desvio-padrão dos `input_closes_mean` recebidos na
   janela recente e comparar com a média/desvio-padrão do `Close` de treino (salvos em
   `config_snapshot` do metadata). Um desvio grande indica que o mercado mudou de regime desde o
   treino (ex.: evento macroeconômico) e o modelo pode precisar de re-treino.

```python
def check_drift(recent_mean: float, recent_std: float,
                 train_mean: float, train_std: float, z_threshold: float = 3.0) -> bool:
    """Retorna True se a média recente estiver a mais de `z_threshold` desvios-padrão de treino
    da média de treino — heurística simples de drift (equivalente a um z-score check)."""
    z = abs(recent_mean - train_mean) / train_std
    return z > z_threshold
```

## 4. Saída do job de monitoramento

Local: `monitoring/reports/monitoring_report_{data}.json`

```json
{
  "generated_at": "2026-07-16T18:00:00Z",
  "window_days": 30,
  "n_predictions_evaluated": 22,
  "production_metrics": {"mae": 0.0, "mape": 0.0},
  "training_metrics_reference": {"mae": 0.0, "mape": 0.0},
  "degradation_detected": false,
  "input_drift_detected": false
}
```

Este relatório é a evidência objetiva do requisito "monitore o desempenho do modelo em produção"
— referenciar no README e mostrar no vídeo de apresentação (rodar o script ao vivo, com dados
reais acumulados de algumas execuções da API).

## 5. Endpoint opcional (recomendado, mas não obrigatório): `GET /monitoring/summary`

Se houver tempo, expor um endpoint que lê o último `monitoring_report_*.json` e retorna via API —
transforma o monitoramento em algo demonstrável diretamente pela URL pública, sem precisar rodar
script localmente durante o vídeo.

## 6. Trilha de evolução (documentar como "próximos passos", não implementar agora)

- Agendar `run_monitoring.py` via GitHub Actions `schedule` (cron) para rodar diariamente.
- Alertas via webhook (Slack/e-mail) quando `degradation_detected == true`.
- Dashboard com Grafana/Streamlit lendo o histórico de `monitoring_report_*.json`.
- Gatilho de re-treino automático quando drift for detectado (fora de escopo deste desafio).
