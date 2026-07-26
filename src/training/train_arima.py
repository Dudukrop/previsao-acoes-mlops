import numpy as np
import pandas as pd
import pmdarima as pm
from statsmodels.tsa.arima.model import ARIMA, ARIMAResults


def fit_arima(
    train_close: pd.Series | np.ndarray, fallback_order: tuple[int, int, int],
    seasonal: bool = False,
) -> ARIMAResults:
    """Ajusta automaticamente (p,d,q) via auto_arima (grid guiado por AIC) usando o Close de
    treino, depois refita com statsmodels.ARIMA para obter o objeto de resultado completo
    (necessário para `.append()`/`.forecast()` e para serialização).

    Recebe o Close como array puro (`np.asarray`), descartando qualquer DatetimeIndex: séries de
    pregão têm frequência irregular (feriados/fins de semana) e o `.append()` do statsmodels
    exige um índice de datas com frequência inferível para validar continuidade — sem isso,
    `.append()` levanta ValueError. Como o modelo não precisa de features de calendário, é mais
    simples e robusto trabalhar só com a sequência de valores.

    Se `auto_arima` levantar exceção, usa `fallback_order` (= config.model.arima.order)
    diretamente com `ARIMA(...).fit()`.
    """
    endog = np.asarray(train_close)
    try:
        auto_model = pm.auto_arima(
            endog, seasonal=seasonal, suppress_warnings=True, error_action="ignore",
            stepwise=True,
        )
        order = auto_model.order
    except Exception as e:
        print(f"[AVISO] auto_arima falhou ({e}); usando fallback_order={fallback_order} do config.")
        order = fallback_order
    return ARIMA(endog, order=order).fit()


def append_actuals(model: ARIMAResults, values: pd.Series | np.ndarray) -> ARIMAResults:
    """Incorpora valores reais (ex.: o período de validação) ao modelo já ajustado, sem
    reajustar (p,d,q) — usado para trazer o filtro de estado até o início do período de teste
    antes da avaliação one-step-ahead."""
    return model.append(np.asarray(values), refit=False)


def evaluate_arima_one_step(model: ARIMAResults, test_close: pd.Series | np.ndarray) -> np.ndarray:
    """Gera previsões one-step-ahead sobre `test_close` com refit incremental: para cada dia do
    teste, prevê usando somente dados conhecidos até o dia anterior, depois incorpora o valor
    REAL observado (não o previsto) antes de prever o próximo — simula produção real, onde cada
    nova predição usa o fechamento real do dia anterior.

    Implementado via `model.append(novo_valor_real, refit=False)`: atualiza o filtro de estado
    sem reajustar (p,d,q) a cada passo.
    """
    predictions = []
    current_model = model
    for value in np.asarray(test_close):
        pred = current_model.forecast(steps=1)
        predictions.append(float(np.asarray(pred)[0]))
        current_model = current_model.append([value], refit=False)
    return np.array(predictions)
