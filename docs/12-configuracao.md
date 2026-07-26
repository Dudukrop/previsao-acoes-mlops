# 12 — Configuração Central

Todo módulo do sistema (coleta, preprocessamento, treino, API) lê a mesma fonte de configuração.
Nenhum módulo hardcoda ticker, janelas de tempo ou paths — isso é o equivalente a injetar
`IOptions<T>` em vez de espalhar `appsettings` por toda parte em um projeto .NET.

## 1. Arquivo `.env` (segredos e overrides de ambiente)

Local: raiz do projeto. **Nunca commitado** (adicionar ao `.gitignore`). Um `.env.example` deve
ser commitado com valores placeholder.

```dotenv
# .env.example
TICKER=PETR4.SA
API_LOG_LEVEL=INFO
MODEL_ARTIFACT_DIR=models/artifacts
MODEL_METADATA_DIR=models/metadata
PREDICTION_LOG_PATH=monitoring/logs/predictions.jsonl
```

## 2. Arquivo `config.yaml` (parâmetros de negócio/experimento, versionado no git)

Local: raiz do projeto, path `config.yaml`.

```yaml
# config.yaml
data:
  ticker: "PETR4.SA"          # sobrescrito por env var TICKER se presente
  start_date: "2015-01-01"
  end_date: null               # null = até a data de execução
  raw_path: "data/raw/{ticker}.csv"
  processed_dir: "data/processed"

split:
  test_size_ratio: 0.15        # últimos 15% dos pregões viram teste (ordem cronológica)
  validation_size_ratio: 0.15  # 15% imediatamente anteriores ao teste viram validação

features:
  lookback_window: 60          # dias de histórico usados como input do modelo
  target_column: "Close"
  use_columns: ["Close"]       # MVP é univariado (ADR-02, doc 01) — somente Close é usado no
                                # window/scaler/LSTM. Adicionar "Volume" aqui hoje é um no-op:
                                # nenhum código consome outra coluna além de target_column. Só
                                # inclua outra coluna depois de estender make_windows (doc 03)
                                # e a arquitetura do LSTM (doc 04) para entrada multivariada.

model:
  type: "lstm"                 # "lstm" | "arima" — controla qual pipeline de treino roda
  lstm:
    units: [64, 32]
    dropout: 0.2
    epochs: 100
    batch_size: 32
    early_stopping_patience: 10
    learning_rate: 0.001
  arima:
    order: [5, 1, 0]           # (p, d, q) inicial; ajustado por auto_arima no treino

evaluation:
  metrics: ["mae", "rmse", "mape"]
  acceptance:
    max_mape_pct: 5.0          # bloqueia serialização (hard gate) se MAPE de teste > 5%
    must_beat_naive_baseline: false   # aviso (soft gate), não bloqueia — ver nota abaixo

random_seed: 42
```

> **Nota sobre `must_beat_naive_baseline`:** é conhecido na literatura de séries financeiras que
> superar o baseline naive (`Close(t+1) = Close(t)`) em previsão de ponto de um passo à frente é
> muito difícil para preço de fechamento diário de ações líquidas — a variação dia-a-dia é
> próxima de ruído (mercado fracamente eficiente). Por isso este campo é `false` por padrão: é
> tratado como **aviso registrado no relatório de avaliação**, não como bloqueio de serialização.
> Só `max_mape_pct` é hard gate. Ver [05-avaliacao-metricas.md](05-avaliacao-metricas.md) seção 4
> para o racional completo. Mude para `true` apenas se, na prática, o LSTM treinado realmente
> superar o naive — nesse caso é um resultado a destacar no README, não o critério mínimo de
> aceite.

## 3. Contrato de acesso (Python)

```python
# src/config.py
from pydantic import BaseModel, Field
from pathlib import Path
import yaml, os

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
    TICKER em .env, se presente, sobrescreve data.ticker.
    Levanta pydantic.ValidationError se o YAML estiver com schema inválido — falha rápido e explícito."""
    raw = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
    if ticker_override := os.getenv("TICKER"):
        raw["data"]["ticker"] = ticker_override
    return AppConfig(**raw)
```

## 4. Regras de uso

- Todo script de pipeline (`src/data_collection/*.py`, `src/preprocessing/*.py`,
  `src/training/*.py`) começa chamando `cfg = load_config()`. É proibido ler `config.yaml` mais de
  uma vez por processo (carregar no `main()` e passar `cfg` como parâmetro, nunca reabrir o arquivo
  dentro de funções internas).
- A API (`api/main.py`) também usa `load_config()` no evento de startup, mas **apenas** para
  resolver paths de artefato — ela não deve precisar de `data`, `split` nem `model.lstm.epochs`
  em tempo de inferência.
- `random_seed` é propagado explicitamente para: `numpy.random.seed`, `tensorflow.random.set_seed`
  e `train_test_split`-equivalentes. Nenhuma chamada estocástica sem seed.
