# 04 — Modelagem e Treinamento

## 1. Modelos a implementar

| Modelo | Papel | Biblioteca |
|---|---|---|
| Naive (baseline ingênuo) | Piso de comparação: `previsão(t+1) = Close(t)` | Sem biblioteca, 3 linhas de código |
| ARIMA/SARIMAX | Baseline estatístico | `pmdarima` (auto_arima) + `statsmodels` |
| LSTM | Modelo principal de produção | `tensorflow.keras` |

O baseline Naive é obrigatório: sem ele não é possível provar que o modelo agrega valor (ver doc 05).

## 2. Contrato — Baseline Naive

Local: `src/training/baseline_naive.py`

```python
import numpy as np

def predict_naive(series: np.ndarray) -> np.ndarray:
    """Previsão ingênua: valor previsto para t+1 é o valor observado em t.
    Recebe a série de teste em nível (não normalizada) e retorna as previsões deslocadas em 1,
    já alinhadas em tamanho com `series[1:]` (o primeiro valor da série não tem previsão)."""
    return series[:-1]
```

## 3. Contrato — ARIMA

Local: `src/training/train_arima.py`

```python
import pandas as pd
import numpy as np
import pmdarima as pm
from statsmodels.tsa.arima.model import ARIMA, ARIMAResults

def fit_arima(
    train_close: pd.Series, fallback_order: tuple[int, int, int], seasonal: bool = False
) -> ARIMAResults:
    """Ajusta automaticamente (p,d,q) via auto_arima (grid guiado por AIC) usando o Close de
    treino, depois refita com statsmodels.ARIMA para obter o objeto de resultado completo
    (necessário para `.append()`/`.forecast()` e para serialização via pickle — o objeto de
    pmdarima não expõe a mesma API de append incremental usada na avaliação).

    Se `auto_arima` levantar exceção (não converge / série malformada), o fallback REAL é usar
    `fallback_order` (= config.yaml → model.arima.order) diretamente com `ARIMA(...).fit()` — não
    é só uma menção em docstring, o try/except abaixo implementa isso de fato.
    """
    try:
        auto_model = pm.auto_arima(
            train_close, seasonal=seasonal, suppress_warnings=True, error_action="ignore",
            stepwise=True,
        )
        order = auto_model.order
    except Exception as e:
        print(f"[AVISO] auto_arima falhou ({e}); usando fallback_order={fallback_order} do config.")
        order = fallback_order
    return ARIMA(train_close, order=order).fit()

def evaluate_arima_one_step(model: ARIMAResults, test_close: pd.Series) -> np.ndarray:
    """Gera previsões one-step-ahead sobre `test_close` com refit incremental: para cada dia do
    teste, prevê usando somente dados conhecidos até o dia anterior, depois incorpora o valor
    REAL observado (não o previsto) antes de prever o próximo — exatamente como o sistema se
    comportaria em produção, onde cada nova predição usa o fechamento real do dia anterior.

    Implementação via `model.append(novo_valor_real, refit=False)`, que atualiza o filtro de
    estado sem reajustar (p,d,q) a cada passo (reajustar 'refit=True' a cada dia seria custoso e
    não é o que este projeto faz; os coeficientes ficam fixos, só o histórico do filtro avança).

    Retorna um array com uma previsão por valor de `test_close` (mesmo tamanho de test_close).
    """
    predictions = []
    current_model = model
    for value in test_close:
        pred = current_model.forecast(steps=1)
        predictions.append(float(pred.iloc[0]))
        current_model = current_model.append([value], refit=False)
    return np.array(predictions)
```

> A avaliação em doc 05 usa `evaluate_arima_one_step`, nunca `forecast(steps=len(test))` de uma
> vez só — gerar todas as previsões de teste de uma tacada deixaria o erro acumular sem correção,
> o que não reflete como o sistema operaria em produção (uma predição por dia, sempre com o dado
> real mais recente disponível).

## 4. Contrato — LSTM

Local: `src/training/train_lstm.py`

> **Alvo do modelo (correção pós-implementação):** o LSTM prevê **retorno logarítmico**
> (`log_return`), não o `Close` absoluto — ver [03-preprocessamento-eda.md](03-preprocessamento-eda.md)
> seção 4 para o motivo (o `MinMaxScaler` ajustado só no treino não generaliza para o nível de
> preço do teste em ações com forte valorização ao longo dos anos; isso foi descoberto rodando o
> pipeline de ponta a ponta — o LSTM sobre preço absoluto teve MAPE de teste de ~5-6%, pior que o
> baseline naive de ~1,1%; sobre retorno, ficou em ~1,18%, no mesmo patamar do naive/ARIMA).
> `X_train`/`y_train` abaixo são janelas de `log_return` escalado, não de `Close` escalado. Para
> obter o preço previsto: `Close_previsto(t) = Close_real(t-1) * exp(retorno_previsto(t))`.

```python
import numpy as np
import tensorflow as tf
from tensorflow.keras import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping

def build_lstm_model(
    lookback_window: int, units: list[int], dropout: float, learning_rate: float
) -> Sequential:
    """Constrói uma rede LSTM empilhada (stacked LSTM).

    Arquitetura:
        Input (lookback_window, 1)
        -> LSTM(units[0], return_sequences=True) -> Dropout(dropout)
        -> LSTM(units[1], return_sequences=False) -> Dropout(dropout)
        -> Dense(1)   # saída: Close(t+1) normalizado, escalar único

    `learning_rate` é sempre config.model.lstm.learning_rate — nunca a taxa default do Keras.
    Passar o otimizador já instanciado com a taxa explícita (não a string "adam") é o que garante
    que mudar `learning_rate` no config.yaml realmente tenha efeito no treino.
    """
    model = Sequential([
        LSTM(units[0], return_sequences=True, input_shape=(lookback_window, 1)),
        Dropout(dropout),
        LSTM(units[1], return_sequences=False),
        Dropout(dropout),
        Dense(1),
    ])
    model.compile(optimizer=Adam(learning_rate=learning_rate), loss="mean_squared_error")
    return model

def train_lstm(
    model: Sequential,
    X_train: np.ndarray, y_train: np.ndarray,
    X_val: np.ndarray, y_val: np.ndarray,
    epochs: int, batch_size: int, patience: int,
) -> tf.keras.callbacks.History:
    """Treina com early stopping monitorando val_loss. Restaura os melhores pesos ao final
    (`restore_best_weights=True`) — o modelo devolvido é sempre o de menor val_loss, não o da
    última época."""
    early_stop = EarlyStopping(monitor="val_loss", patience=patience, restore_best_weights=True)
    return model.fit(
        X_train, y_train,
        validation_data=(X_val, y_val),
        epochs=epochs, batch_size=batch_size,
        callbacks=[early_stop], verbose=2,
    )
```

## 5. Reprodutibilidade

No topo de todo script de treino:

```python
import os, random, numpy as np, tensorflow as tf

def set_global_seed(seed: int) -> None:
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    tf.random.set_seed(seed)
```

Chamado com `cfg.random_seed` antes de qualquer construção de modelo.

## 6. Script executável

Local: `src/training/run_training.py`

```python
"""
Uso:
    python -m src.training.run_training

Lê data/processed/{train,validation}.parquet e scaler.joblib.
Treina Naive, ARIMA e LSTM.
NÃO decide sozinho qual modelo "vence" nem serializa nada — apenas produz os 3 modelos treinados
em memória e delega a decisão de aceite para a etapa de Avaliação (doc 05), que é quem decide se
o LSTM está bom o suficiente para ser serializado.
Persiste temporariamente os modelos treinados (não serializados oficialmente) em
`models/artifacts/_tmp/` para a etapa de avaliação consumir, evitando re-treinar.
"""
```

## 7. Grade de hiperparâmetros (se houver tempo para tuning, opcional)

| Hiperparâmetro | Valores testados | Critério de escolha |
|---|---|---|
| `lookback_window` | 30, 60, 90 | Menor `val_loss` |
| `units` | [32,16], [64,32], [128,64] | Menor `val_loss` sem overfitting (gap treino/val) |
| `dropout` | 0.1, 0.2, 0.3 | Menor `val_loss` |
| `batch_size` | 16, 32, 64 | Menor `val_loss`, tempo de treino aceitável |

Documentar no README final apenas a combinação vencedora e o racional, não a busca inteira.
