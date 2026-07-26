# 02 — Coleta de Dados

## 1. Fonte de dados

**Biblioteca:** `yfinance` (dados do Yahoo Finance, gratuitos, sem necessidade de API key).

**Justificativa:** cobre qualquer ticker listado (B3 com sufixo `.SA`, NYSE/NASDAQ sem sufixo),
histórico diário desde IPO, sem custo, sem cadastro. Suficiente para o escopo do desafio.

## 2. Contrato do módulo

Local: `src/data_collection/fetch_data.py`

```python
import pandas as pd
import yfinance as yf
from pathlib import Path
from src.config import AppConfig

REQUIRED_COLUMNS = ["Open", "High", "Low", "Close", "Volume"]

def fetch_historical_data(ticker: str, start_date: str, end_date: str | None) -> pd.DataFrame:
    """Baixa OHLCV diário do Yahoo Finance para `ticker` entre start_date e end_date.

    Contrato de saída:
    - Índice: DatetimeIndex, nome "Date", ordenado ascendente, sem duplicatas.
    - Colunas obrigatórias: Open, High, Low, Close, Volume (todas float64, exceto Volume int64).
    - Nenhuma linha com Close nulo (dias sem pregão já vêm ausentes do yfinance, não geram NaN).

    Levanta:
    - ValueError se o DataFrame retornado vier vazio (ticker inválido ou sem dados no período).
    """
    df = yf.download(ticker, start=start_date, end=end_date, progress=False, auto_adjust=True)
    if df.empty:
        raise ValueError(f"Nenhum dado retornado para ticker='{ticker}'. Verifique o símbolo.")
    if isinstance(df.columns, pd.MultiIndex):
        # yfinance >= 0.2.31 retorna colunas MultiIndex (ticker, campo) mesmo para um único
        # ticker, dependendo da versão instalada. Achata para o formato de coluna simples que
        # o resto do pipeline espera.
        df.columns = df.columns.get_level_values(0)
    df = df[REQUIRED_COLUMNS].copy()
    df.index.name = "Date"
    return df

def save_raw_data(df: pd.DataFrame, ticker: str, raw_path_template: str) -> Path:
    """Persiste o DataFrame bruto em CSV, path resolvido por raw_path_template.format(ticker=...).
    Sobrescreve o arquivo existente (raw é sempre um snapshot completo, não incremental)."""
    path = Path(raw_path_template.format(ticker=ticker))
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, encoding="utf-8")
    return path
```

## 3. Script executável

Local: `src/data_collection/run_collection.py` — ponto de entrada de linha de comando.

```python
"""
Uso:
    python -m src.data_collection.run_collection

Efeito colateral: grava data/raw/{TICKER}.csv (definido em config.yaml / .env).
Idempotente: pode ser executado quantas vezes forem necessárias; sempre sobrescreve com o
histórico mais atualizado disponível no Yahoo Finance.
"""
from src.config import load_config
from src.data_collection.fetch_data import fetch_historical_data, save_raw_data

def main() -> None:
    cfg = load_config()
    df = fetch_historical_data(cfg.data.ticker, cfg.data.start_date, cfg.data.end_date)
    path = save_raw_data(df, cfg.data.ticker, cfg.data.raw_path)
    print(f"[coleta] {len(df)} linhas salvas em {path}")

if __name__ == "__main__":
    main()
```

## 4. Schema de dados brutos (contrato de saída desta etapa)

| Coluna | Tipo | Descrição | Regra de qualidade |
|---|---|---|---|
| Date (índice) | datetime64 | Data do pregão | Único, ordenado ascendente |
| Open | float64 | Preço de abertura ajustado | > 0 |
| High | float64 | Máxima do dia | >= Open, >= Close, >= Low |
| Low | float64 | Mínima do dia | <= Open, <= Close, <= High |
| Close | float64 | Fechamento ajustado (dividendos/splits já incorporados via `auto_adjust=True`) | > 0, é a coluna alvo |
| Volume | int64 | Volume negociado | >= 0 |

## 5. Validação pós-coleta (checklist manual/notebook, não bloqueia o pipeline automatizado)

- Plotar `Close` ao longo do tempo — verificar ausência de saltos artificiais (indicaria split/
  dividendo não ajustado corretamente).
- Checar `df.isna().sum()` — deve ser zero em todas as colunas.
- Checar `df.index.is_monotonic_increasing` — deve ser `True`.
- Checar quantidade de linhas plausível: `dias_uteis_aproximados = (end - start).days * 5/7`.

## 6. Limitações conhecidas (documentar no README final)

- `yfinance` depende de disponibilidade do Yahoo Finance; não há SLA garantido. Para o escopo do
  desafio isso é aceitável — o dado bruto baixado é versionado em `data/raw/` justamente para não
  depender da API estar no ar durante o treino/avaliação.
- Preços são ajustados (`auto_adjust=True`), portanto refletem o valor "justo" histórico, não o
  preço nominal exato negociado naquele dia — decisão deliberada para evitar quebras artificiais
  de série por split/dividendo.
