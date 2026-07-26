"""
Uso:
    python -m src.training.run_training

Lê data/processed/{train,validation,test}.parquet e scaler.joblib. Treina Naive, ARIMA e LSTM,
avalia os 3 com o mesmo protocolo one-step-ahead, grava models/metadata/evaluation_report.json e,
se o LSTM passar no hard gate (doc 05), serializa e promove o modelo.
"""

import json
from datetime import datetime, timezone
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from src.config import load_config
from src.evaluation.gate import ModelNotAcceptedError, check_acceptance
from src.evaluation.metrics import evaluate_all
from src.preprocessing.prepare_dataset import make_windows
from src.training.baseline_naive import predict_naive
from src.training.serialize_model import promote_version, serialize_artifacts
from src.training.train_arima import append_actuals, evaluate_arima_one_step, fit_arima
from src.training.train_lstm import build_lstm_model, set_global_seed, train_lstm


def main() -> None:
    cfg = load_config()
    set_global_seed(cfg.random_seed)

    processed_dir = Path(cfg.data.processed_dir)
    train = pd.read_parquet(processed_dir / "train.parquet")
    validation = pd.read_parquet(processed_dir / "validation.parquet")
    test = pd.read_parquet(processed_dir / "test.parquet")
    scaler = joblib.load(processed_dir / "scaler.joblib")

    target_col = cfg.features.target_column
    lookback = cfg.features.lookback_window
    test_true = test[target_col].values

    # ---- Naive baseline ----
    naive_history = pd.concat([validation[target_col].tail(1), test[target_col]])
    naive_pred = predict_naive(naive_history.values)
    naive_metrics = evaluate_all(test_true, naive_pred)

    # ---- ARIMA baseline ----
    arima_model = fit_arima(train[target_col], fallback_order=cfg.model.arima.order)
    if len(validation) > 0:
        arima_model = append_actuals(arima_model, validation[target_col])
    arima_pred = evaluate_arima_one_step(arima_model, test[target_col])
    arima_metrics = evaluate_all(test_true, arima_pred)

    # ---- LSTM ----
    # Alvo é o retorno logarítmico (log_return), não o preço absoluto — ver
    # src/preprocessing/prepare_dataset.py::compute_log_returns para o motivo (MinMaxScaler
    # ajustado só no treino não generaliza para o nível de preço do teste em ações que
    # valorizaram/desvalorizaram muito ao longo dos anos).
    train_scaled = scaler.transform(train[["log_return"]]).flatten()
    val_scaled = scaler.transform(validation[["log_return"]]).flatten()
    X_train, y_train = make_windows(train_scaled, lookback)
    X_val, y_val = make_windows(val_scaled, lookback)

    lstm_cfg = cfg.model.lstm
    model = build_lstm_model(lookback, lstm_cfg.units, lstm_cfg.dropout, lstm_cfg.learning_rate)
    train_lstm(
        model, X_train, y_train, X_val, y_val,
        epochs=lstm_cfg.epochs, batch_size=lstm_cfg.batch_size,
        patience=lstm_cfg.early_stopping_patience,
    )

    history_for_test = pd.concat([validation["log_return"].tail(lookback), test["log_return"]])
    history_scaled = scaler.transform(history_for_test.to_frame()).flatten()
    X_test, _ = make_windows(history_scaled, lookback)
    lstm_pred_return_scaled = model.predict(X_test, verbose=0).flatten()
    lstm_pred_return = scaler.inverse_transform(lstm_pred_return_scaled.reshape(-1, 1)).flatten()
    # naive_pred[i] já é exatamente o Close do dia anterior ao dia de teste i — reconstrói o
    # preço previsto pelo LSTM a partir dele: Close_pred = Close_anterior * exp(retorno_previsto).
    lstm_pred = naive_pred * np.exp(lstm_pred_return)
    lstm_metrics = evaluate_all(test_true, lstm_pred)

    print("Naive:", naive_metrics)
    print("ARIMA:", arima_metrics)
    print("LSTM:", lstm_metrics)

    # ---- Gate + evaluation report ----
    acceptance = cfg.evaluation.acceptance
    passed = True
    try:
        check_acceptance(
            lstm_metrics["mape"], naive_metrics["mape"],
            acceptance.max_mape_pct, acceptance.must_beat_naive_baseline,
        )
    except ModelNotAcceptedError as e:
        passed = False
        print(f"[GATE] Modelo REPROVADO: {e}")

    report = {
        "ticker": cfg.data.ticker,
        "evaluated_at": datetime.now(timezone.utc).isoformat(),
        "test_period": {
            "start": str(test.index.min().date()),
            "end": str(test.index.max().date()),
        },
        "results": {"naive": naive_metrics, "arima": arima_metrics, "lstm": lstm_metrics},
        "accepted_model": "lstm" if passed else None,
        "acceptance_criteria": {
            "max_mape_pct": acceptance.max_mape_pct,
            "must_beat_naive_baseline": acceptance.must_beat_naive_baseline,
        },
        "beats_naive_baseline": bool(lstm_metrics["mape"] < naive_metrics["mape"]),
        "passed": passed,
    }
    metadata_dir = Path("models/metadata")
    metadata_dir.mkdir(parents=True, exist_ok=True)
    (metadata_dir / "evaluation_report.json").write_text(
        json.dumps(report, indent=2), encoding="utf-8"
    )
    print(f"[avaliacao] relatorio salvo em {metadata_dir / 'evaluation_report.json'}")

    if not passed:
        print("[serializacao] pulada — modelo nao passou no hard gate de MAPE.")
        return

    train_close_stats = {
        "mean": float(train[target_col].mean()),
        "std": float(train[target_col].std()),
    }
    version = serialize_artifacts(
        model, scaler,
        artifact_dir="models/artifacts", metadata_dir=str(metadata_dir),
        ticker=cfg.data.ticker, lookback_window=lookback,
        evaluation_metrics=lstm_metrics, config_snapshot=cfg.model_dump(),
        train_close_stats=train_close_stats,
    )
    promote_version(str(metadata_dir), version)
    print(f"[serializacao] modelo v{version} serializado e promovido em models/artifacts/.")


if __name__ == "__main__":
    main()
