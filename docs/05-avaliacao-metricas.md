# 05 — Avaliação e Métricas

## 1. Métricas

| Métrica | Fórmula | Por que usar |
|---|---|---|
| MAE (Mean Absolute Error) | `mean(|y_true - y_pred|)` | Erro médio na mesma unidade do preço (R$/US$) — interpretável diretamente |
| RMSE (Root Mean Squared Error) | `sqrt(mean((y_true - y_pred)^2))` | Penaliza mais erros grandes; sensível a outliers de previsão |
| MAPE (Mean Absolute Percentage Error) | `mean(|y_true - y_pred| / |y_true|) * 100` | Métrica percentual, comparável entre ações de preços diferentes — é a métrica de **critério de aceite** |

## 2. Contrato do módulo

Local: `src/evaluation/metrics.py`

```python
import numpy as np

def mae(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float(np.mean(np.abs(y_true - y_pred)))

def rmse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float(np.sqrt(np.mean((y_true - y_pred) ** 2)))

def mape(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Retorna em percentual (0-100). Assume y_true sem zeros (preço de fechamento nunca é 0)."""
    return float(np.mean(np.abs((y_true - y_pred) / y_true)) * 100)

def evaluate_all(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float]:
    return {"mae": mae(y_true, y_pred), "rmse": rmse(y_true, y_pred), "mape": mape(y_true, y_pred)}
```

## 3. Protocolo de avaliação (idêntico para os 3 modelos, para comparação justa)

1. Gerar previsões **one-step-ahead** sobre `data/processed/test.parquet` — ou seja, para cada dia
   `t` do teste, prever `Close(t)` usando apenas dados disponíveis até `t-1`. Isso vale para os 3
   modelos (Naive, ARIMA, LSTM) e simula fielmente o uso em produção (a API só prevê o próximo
   dia, nunca múltiplos dias à frente sem dados novos).
2. Desnormalizar as previsões do LSTM (`scaler.inverse_transform`) antes de calcular métricas —
   **métricas são sempre calculadas em preço real (R$/US$), nunca na escala normalizada [0,1]**,
   senão o MAPE fica sem significado de negócio.
3. Rodar `evaluate_all(y_true, y_pred)` para os 3 modelos sobre o mesmo `test.parquet`.
4. Montar tabela comparativa (formato exato usado no relatório e no README):

   | Modelo | MAE | RMSE | MAPE (%) |
   |---|---|---|---|
   | Naive (baseline) | ... | ... | ... |
   | ARIMA | ... | ... | ... |
   | LSTM | ... | ... | ... |

## 4. Critério de aceite (gate de produção)

Há uma distinção deliberada entre **hard gate** (bloqueia serialização) e **soft gate** (registra
aviso, não bloqueia):

- **Hard gate:** `MAPE_lstm <= config.evaluation.acceptance.max_mape_pct` (default: 5.0%). Um
  modelo com erro percentual acima disso não é seguro para expor em produção.
- **Soft gate (aviso, não bloqueio):** `MAPE_lstm < MAPE_naive`. Idealmente o modelo deveria
  superar "prever que amanhã é igual a hoje" — mas é um fato bem documentado na literatura de
  séries financeiras que superar o baseline naive em previsão de ponto de um passo à frente é
  muito difícil para preço de fechamento diário de ações líquidas (a variação dia-a-dia é próxima
  de ruído, mercado fracamente eficiente). **Se este projeto tratasse isso como hard gate, o
  pipeline poderia nunca produzir um modelo aprovado para serialização — inviabilizando a entrega
  de uma API com modelo deployado.** Por isso `must_beat_naive_baseline` é `false` por padrão
  (doc 12): o resultado é sempre registrado no `evaluation_report.json` e discutido no README,
  mas não impede o deploy.

Se o hard gate falhar, o pipeline **não deve serializar o modelo automaticamente** — o script de
avaliação levanta uma exceção com uma mensagem explícita, e a decisão de re-treinar/re-tunar fica
com quem está executando (não há re-treino automático nesta versão do projeto, por escopo).

```python
# src/evaluation/gate.py
class ModelNotAcceptedError(Exception):
    pass

def check_acceptance(
    mape_model: float, mape_naive: float, max_mape_pct: float, must_beat_naive: bool
) -> None:
    """Levanta ModelNotAcceptedError somente se o hard gate (max_mape_pct) falhar — essa é a
    única condição que impede a serialização.

    Se `must_beat_naive=True` e o modelo não superar o naive, NÃO levanta exceção: apenas emite
    um aviso (print/log). Isso é intencional (soft gate, ver texto acima) — o valor default de
    `must_beat_naive` em config.yaml é `false`, então esse ramo normalmente nem é avaliado.
    Não retorna nada em caso de sucesso (fail-fast por exceção apenas para o hard gate)."""
    if mape_model > max_mape_pct:
        raise ModelNotAcceptedError(
            f"MAPE do modelo ({mape_model:.2f}%) excede o limite aceito ({max_mape_pct}%)."
        )
    if must_beat_naive and mape_model >= mape_naive:
        print(
            f"[AVISO] MAPE do modelo ({mape_model:.2f}%) não supera o baseline naive "
            f"({mape_naive:.2f}%). Serialização prossegue; documentar esta constatação no README."
        )
```

## 5. Artefato de saída desta etapa

Local: `models/metadata/evaluation_report.json`

```json
{
  "ticker": "PETR4.SA",
  "evaluated_at": "2026-07-16T00:00:00Z",
  "test_period": {"start": "2025-10-01", "end": "2026-07-15"},
  "results": {
    "naive": {"mae": 0.0, "rmse": 0.0, "mape": 0.0},
    "arima": {"mae": 0.0, "rmse": 0.0, "mape": 0.0},
    "lstm":  {"mae": 0.0, "rmse": 0.0, "mape": 0.0}
  },
  "accepted_model": "lstm",
  "acceptance_criteria": {"max_mape_pct": 5.0, "must_beat_naive_baseline": false},
  "beats_naive_baseline": false,
  "passed": true
}
```

> O campo `evaluated_at` é preenchido pelo processo em tempo de execução (não é um valor fixo de
> template) — implementar com `datetime.now(timezone.utc).isoformat()`.

Este JSON é referenciado diretamente no README (seção "Resultados") e no vídeo de apresentação —
é a evidência objetiva de que o modelo cumpre o requisito "avalie o desempenho ... utilizando
métricas relevantes".

## 6. Gráficos obrigatórios (notebook `notebooks/02_avaliacao.ipynb`)

- Série real vs. previsto (LSTM) no período de teste, sobrepostos.
- Série real vs. previsto (Naive) no mesmo período, para comparação visual direta.
- Histograma dos erros residuais (`y_true - y_pred`) do LSTM — verificar se estão centrados em 0
  e sem viés sistemático.
