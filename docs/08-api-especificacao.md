# 08 — Especificação da API

## 1. Stack

**FastAPI** + **Uvicorn**. Escolhido por gerar documentação OpenAPI/Swagger automaticamente
(`/docs`), validação de schema nativa via Pydantic (equivalente aos DTOs/`ModelState.IsValid` de
um controller ASP.NET Core), e ser o padrão de mercado para servir modelos Python.

## 2. Endpoints

### `GET /health`

Health check obrigatório para a plataforma de deploy verificar se o container está pronto.

**Resposta 200:**
```json
{ "status": "ok", "model_version": 1, "ticker": "PETR4.SA" }
```

### `GET /model/info`

Metadados do modelo atualmente carregado — usado para auditoria/monitoramento sem precisar abrir
o container.

**Resposta 200:**
```json
{
  "version": 1,
  "ticker": "PETR4.SA",
  "created_at": "2026-07-16T00:00:00Z",
  "lookback_window": 60,
  "evaluation_metrics": { "mae": 0.0, "rmse": 0.0, "mape": 0.0 }
}
```

### `POST /predict/by-ticker` — fluxo principal

Este é o endpoint que efetivamente cumpre "receber a requisição, **extrair os dados de entrada**
e enviar ao modelo": o cliente manda apenas o ticker, e é a própria API que busca e extrai o
histórico recente (reaproveitando `fetch_historical_data` do doc 02) antes de montar a janela e
chamar o modelo. Não exige que quem consome a API já saiba o que é `lookback_window`.

**Request body:**
```json
{ "ticker": "PETR4.SA" }
```

Se `ticker` for omitido, a API usa o ticker do modelo carregado (`loaded.ticker`) — útil porque,
neste projeto, um único modelo/ticker está em produção por vez (ver ADR-01, doc 01). Enviar um
ticker diferente do que o modelo foi treinado é um erro do cliente (ver tabela de erros, seção 6).

**Contrato Pydantic (`api/schemas.py`):**

```python
from pydantic import BaseModel

class PredictByTickerRequest(BaseModel):
    ticker: str | None = None   # se None, usa o ticker do modelo carregado

class PredictResponse(BaseModel):
    ticker: str
    predicted_close: float
    prediction_for_date: str          # ISO date do próximo pregão previsto
    model_version: int
    request_id: str                   # UUID gerado por requisição, usado para correlacionar com o log
```

**Resposta 200:**
```json
{
  "ticker": "PETR4.SA",
  "predicted_close": 33.12,
  "prediction_for_date": "2026-07-17",
  "model_version": 1,
  "request_id": "b3f1c2e0-....."
}
```

**Contrato do handler (`api/routers/predict.py`):**

```python
from datetime import date, timedelta
from src.data_collection.fetch_data import fetch_historical_data

FETCH_MARGIN_CALENDAR_DAYS = 15  # cobre feriados + fins de semana sem 2ª tentativa na prática

def handle_predict_by_ticker(loaded: LoadedModel, ticker: str | None) -> tuple[str, float, str]:
    """Resolve o ticker (usa loaded.ticker se None), rejeita se um ticker diferente do treinado
    for pedido (ver seção 6). Busca histórico via yfinance com `start = hoje - (lookback_window
    dias úteis equivalentes + FETCH_MARGIN_CALENDAR_DAYS) dias corridos` — a margem fixa de 15
    dias corridos garante linhas suficientes mesmo com feriados/emendas de fim de semana no meio
    da janela, sem precisar de retry. Extrai a coluna Close, pega exatamente os
    `lookback_window` valores mais recentes (`.iloc[-lookback_window:]`) — se ainda assim vierem
    menos linhas que `lookback_window` (feriado atípico prolongado), relança como erro 502 (ver
    seção 6), não tenta compensar com dado sintético. Delega a `predict_next_close` (seção 5).
    Retorna (ticker_resolvido, predicted_close, prediction_for_date).
    """
```

### `POST /predict` — modo avançado/manual

Variante de baixo nível para quem já tem os preços em mãos (testes automatizados, backtesting,
ou reprodução de um cenário específico sem depender de rede) — recebe os últimos
`lookback_window + 1` preços de fechamento diretamente.

**Correção pós-implementação:** o modelo prevê sobre retorno logarítmico, não preço absoluto (ver
doc 03 seção 4 — necessário para o `MinMaxScaler` generalizar em ações com forte
valorização/desvalorização ao longo dos anos). Por isso são exigidos `lookback_window + 1` preços
em nível: os `lookback_window` retornos diários que alimentam o LSTM só existem entre pares
consecutivos de preço, e a reconstrução do preço previsto usa o último preço fornecido:
`Close_previsto = closes[-1] * exp(retorno_previsto)`.

**Request body:**
```json
{
  "closes": [32.10, 32.45, 31.98, "... 61 valores no total (lookback_window + 1), do mais antigo para o mais recente"]
}
```

**Contrato Pydantic (`api/schemas.py`):**

```python
from pydantic import BaseModel, field_validator

class PredictRequest(BaseModel):
    closes: list[float]

    @field_validator("closes")
    @classmethod
    def validate_length(cls, v: list[float]) -> list[float]:
        """O tamanho exato é validado no endpoint contra o lookback_window do modelo carregado
        (não é um valor fixo aqui, pois depende do metadata do modelo em produção — ver seção 4).
        Aqui validamos apenas invariantes que nunca mudam: lista não vazia e todos os valores > 0
        (preço de fechamento nunca é negativo ou zero)."""
        if not v:
            raise ValueError("closes não pode ser vazio")
        if any(p <= 0 for p in v):
            raise ValueError("todos os valores de closes devem ser > 0")
        return v
```

Resposta: mesmo formato `PredictResponse` do endpoint acima.

**Resposta 422 (validação falhou — tratado automaticamente pelo FastAPI/Pydantic):**
```json
{ "detail": [ { "loc": ["body", "closes"], "msg": "closes não pode ser vazio", "type": "value_error" } ] }
```

**Resposta 400 (tamanho de `closes` incompatível com o `lookback_window` do modelo):**
```json
{ "detail": "Esperado exatamente 60 valores em 'closes', recebido 45." }
```

## 2.1 Cálculo de `prediction_for_date`

Local: `api/date_utils.py` (compartilhado pelos dois endpoints acima).

```python
from datetime import date, timedelta

def next_trading_day(reference_date: date) -> date:
    """Retorna o próximo dia útil (segunda a sexta) após `reference_date`, pulando fim de semana.

    Limitação conhecida e aceita para o escopo do desafio: NÃO considera feriados da bolsa
    (B3/NYSE) — apenas fins de semana. Documentar essa limitação no README. Se um feriado cair
    no dia retornado, o valor de `prediction_for_date` estará um dia adiantado em relação ao
    pregão real seguinte; isso não afeta `predicted_close` (o valor previsto), só o rótulo de
    data exibido na resposta.
    """
    next_day = reference_date + timedelta(days=1)
    while next_day.weekday() >= 5:  # 5=sábado, 6=domingo
        next_day += timedelta(days=1)
    return next_day
```

## 3. Estrutura de arquivos da API

```
api/
  main.py              # cria o app, registra rotas, evento de startup carrega modelo
  schemas.py           # PredictRequest, PredictByTickerRequest, PredictResponse
  model_loader.py       # lê current_version.json + model_metadata_v{N}.json, carrega .keras e .joblib
  inference.py          # lógica pura de predição (recebe closes, scaler, model -> float)
  date_utils.py          # next_trading_day (seção 2.1)
  logging_middleware.py # grava monitoring/logs/predictions.jsonl a cada request
  routers/
    predict.py          # POST /predict/by-ticker (principal), POST /predict (manual)
    health.py            # GET /health, GET /model/info
```

## 4. Contrato de `model_loader.py`

```python
from dataclasses import dataclass
from tensorflow.keras import Model
from sklearn.preprocessing import MinMaxScaler
import json

@dataclass
class LoadedModel:
    model: Model
    scaler: MinMaxScaler
    version: int
    ticker: str
    lookback_window: int
    evaluation_metrics: dict

def load_active_model(metadata_dir: str) -> LoadedModel:
    """Lê current_version.json para descobrir a versão ativa, carrega o .keras e o .joblib
    correspondentes descritos em model_metadata_v{N}.json.
    Levanta FileNotFoundError com mensagem clara se current_version.json não existir — isso
    significa que nenhum modelo foi promovido ainda (falha de deploy, não deveria acontecer em
    produção; a mensagem deve orientar rodar o pipeline de treino)."""
    ...
```

Chamado uma única vez, no evento `@app.on_event("startup")` de `api/main.py`, e guardado em
`app.state.loaded_model` — nunca recarregado durante o ciclo de vida do processo (recarregar
exigiria reiniciar o container, o que é aceitável para este escopo: promoção de nova versão =
novo deploy).

## 5. Contrato de `inference.py`

```python
import numpy as np
from api.model_loader import LoadedModel

def predict_next_close(loaded: LoadedModel, closes: list[float]) -> float:
    """Recebe exatamente `loaded.lookback_window` preços em nível (não normalizados), na ordem
    do mais antigo para o mais recente. Normaliza com o scaler carregado, roda a inferência do
    LSTM, desnormaliza e retorna um float em preço real.

    Levanta ValueError se len(closes) != loaded.lookback_window (validação de contrato de negócio,
    verificada no router antes de chamar esta função, para devolver HTTP 400 em vez de 500)."""
    arr = np.array(closes).reshape(-1, 1)
    scaled = loaded.scaler.transform(arr).reshape(1, loaded.lookback_window, 1)
    pred_scaled = loaded.model.predict(scaled, verbose=0)
    pred = loaded.scaler.inverse_transform(pred_scaled)
    return float(pred[0][0])
```

## 6. Tratamento de erros (mapa de exceção → HTTP status)

| Situação | Status | Corpo |
|---|---|---|
| `closes` vazio ou com valor <= 0 | 422 | Gerado automaticamente pelo Pydantic |
| `len(closes)` diferente do `lookback_window` do modelo | 400 | `{"detail": "Esperado exatamente N valores..."}` |
| `ticker` informado difere do ticker do modelo carregado | 400 | `{"detail": "Este serviço só prevê PETR4.SA; ticker 'VALE3.SA' não corresponde ao modelo carregado."}` |
| Falha ao buscar histórico via yfinance (`/predict/by-ticker`) | 502 | `{"detail": "Não foi possível obter dados de mercado no momento. Tente novamente."}` |
| Modelo não carregado (falha de startup) | 503 | `{"detail": "Modelo indisponível. Tente novamente mais tarde."}` |
| Erro inesperado durante inferência | 500 | `{"detail": "Erro interno ao gerar predição."}` + log completo no servidor (nunca expor stacktrace ao cliente) |

## 7. Exemplo de uso (curl)

```bash
# Fluxo principal — API extrai os dados sozinha a partir do ticker
curl -X POST http://localhost:8000/predict/by-ticker \
  -H "Content-Type: application/json" \
  -d '{"ticker": "PETR4.SA"}'

# Modo avançado/manual — cliente já fornece a janela de preços
curl -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d '{"closes": [32.10, 32.45, "...58 valores omitidos..."]}'
```

## 8. CORS

Liberar CORS amplo (`allow_origins=["*"]`) é aceitável para este desafio (API pública de
demonstração, sem autenticação/dados sensíveis). Documentar explicitamente essa decisão no README
como simplificação consciente para o escopo acadêmico, não recomendada para um sistema com dados
sensíveis em produção real.
