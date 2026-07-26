from pathlib import Path

import pandas as pd
import yfinance as yf

REQUIRED_COLUMNS = ["Open", "High", "Low", "Close", "Volume"]


def fetch_historical_data(ticker: str, start_date: str, end_date: str | None) -> pd.DataFrame:
    """Baixa OHLCV diário do Yahoo Finance para `ticker` entre start_date e end_date.

    Levanta ValueError se o DataFrame retornado vier vazio (ticker inválido ou sem dados
    no período)."""
    df = yf.download(ticker, start=start_date, end=end_date, progress=False, auto_adjust=True)
    if df.empty:
        raise ValueError(f"Nenhum dado retornado para ticker='{ticker}'. Verifique o símbolo.")
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)
    df = df[REQUIRED_COLUMNS].copy()
    df.index.name = "Date"
    return df


def save_raw_data(df: pd.DataFrame, ticker: str, raw_path_template: str) -> Path:
    """Persiste o DataFrame bruto em CSV. Sobrescreve o arquivo existente (raw é sempre um
    snapshot completo, não incremental)."""
    path = Path(raw_path_template.format(ticker=ticker))
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, encoding="utf-8")
    return path
