import numpy as np

from api.model_loader import LoadedModel

REQUIRED_CLOSES_COUNT_OFFSET = 1  # o modelo prevê sobre retornos: N retornos exigem N+1 preços


def predict_next_close(loaded: LoadedModel, closes: list[float]) -> float:
    """Recebe exatamente `loaded.lookback_window + 1` preços em nível (não normalizados), na
    ordem do mais antigo para o mais recente. O modelo foi treinado sobre retorno logarítmico,
    não preço absoluto (ver src/preprocessing/prepare_dataset.py::compute_log_returns) — por
    isso são necessários N+1 preços para montar as N variações diárias que formam a janela de
    entrada. Normaliza os retornos com o scaler carregado, roda a inferência do LSTM, desnormaliza
    o retorno previsto e reconstrói o preço: `Close_previsto = último_close * exp(retorno_previsto)`.

    Levanta ValueError se len(closes) != loaded.lookback_window + 1."""
    expected = loaded.lookback_window + REQUIRED_CLOSES_COUNT_OFFSET
    if len(closes) != expected:
        raise ValueError(f"Esperado exatamente {expected} valores em 'closes', recebido {len(closes)}.")

    closes_arr = np.array(closes)
    log_returns = np.log(closes_arr[1:] / closes_arr[:-1])
    scaled = loaded.scaler.transform(log_returns.reshape(-1, 1)).reshape(1, loaded.lookback_window, 1)
    pred_return_scaled = loaded.model.predict(scaled, verbose=0)
    pred_return = loaded.scaler.inverse_transform(pred_return_scaled)[0][0]
    return float(closes_arr[-1] * np.exp(pred_return))
