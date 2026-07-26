"""
Uso:
    python -m src.preprocessing.run_preprocessing

Lê data/raw/{ticker}.csv, grava em data/processed/:
    train.parquet, validation.parquet, test.parquet   (dados em nível, para o ARIMA)
    scaler.joblib                                       (ajustado apenas no train)
"""

from pathlib import Path

import joblib
from sklearn.preprocessing import MinMaxScaler

from src.config import load_config
from src.preprocessing.prepare_dataset import (
    chronological_split,
    clean,
    compute_log_returns,
    load_raw,
)


def main() -> None:
    cfg = load_config()
    raw_path = cfg.data.raw_path.format(ticker=cfg.data.ticker)

    df = load_raw(raw_path)
    df = clean(df)

    # Retorno logarítmico calculado sobre a série completa (antes do split) para preservar
    # continuidade: o primeiro retorno de validation/test usa o último Close do split anterior.
    df["log_return"] = compute_log_returns(df[cfg.features.target_column])
    df = df.dropna(subset=["log_return"])  # descarta só a primeira linha da série inteira

    train, validation, test = chronological_split(
        df, cfg.split.test_size_ratio, cfg.split.validation_size_ratio
    )

    assert train.index.max() < validation.index.min() < test.index.min(), (
        "Sobreposição temporal detectada entre train/validation/test."
    )

    processed_dir = Path(cfg.data.processed_dir)
    processed_dir.mkdir(parents=True, exist_ok=True)

    train.to_parquet(processed_dir / "train.parquet")
    validation.to_parquet(processed_dir / "validation.parquet")
    test.to_parquet(processed_dir / "test.parquet")

    scaler = MinMaxScaler()
    scaler.fit(train[["log_return"]])
    joblib.dump(scaler, processed_dir / "scaler.joblib")

    print(
        f"[preprocessamento] train={len(train)} validation={len(validation)} test={len(test)} "
        f"linhas salvas em {processed_dir}"
    )


if __name__ == "__main__":
    main()
