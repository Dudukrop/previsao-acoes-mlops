import numpy as np


def predict_naive(series: np.ndarray) -> np.ndarray:
    """Previsão ingênua: valor previsto para t+1 é o valor observado em t. Recebe a série de
    teste em nível (não normalizada) e retorna as previsões deslocadas em 1, já alinhadas em
    tamanho com `series[1:]` (o primeiro valor da série não tem previsão)."""
    return series[:-1]
