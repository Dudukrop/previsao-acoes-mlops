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
