import os
import random

import numpy as np
import tensorflow as tf
from tensorflow.keras import Sequential
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.optimizers import Adam


def set_global_seed(seed: int) -> None:
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    tf.random.set_seed(seed)


def build_lstm_model(
    lookback_window: int, units: list[int], dropout: float, learning_rate: float
) -> Sequential:
    """Constrói uma rede LSTM empilhada (stacked LSTM).

    Input (lookback_window, 1)
    -> LSTM(units[0], return_sequences=True) -> Dropout(dropout)
    -> LSTM(units[1], return_sequences=False) -> Dropout(dropout)
    -> Dense(1)   # saída: Close(t+1) normalizado, escalar único
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
    """Treina com early stopping monitorando val_loss. Restaura os melhores pesos ao final —
    o modelo devolvido é sempre o de menor val_loss, não o da última época."""
    early_stop = EarlyStopping(monitor="val_loss", patience=patience, restore_best_weights=True)
    return model.fit(
        X_train, y_train,
        validation_data=(X_val, y_val),
        epochs=epochs, batch_size=batch_size,
        callbacks=[early_stop], verbose=2,
    )
