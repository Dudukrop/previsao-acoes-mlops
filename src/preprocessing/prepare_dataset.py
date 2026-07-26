import numpy as np
import pandas as pd


def load_raw(raw_path: str) -> pd.DataFrame:
    """Lê o CSV salvo pela etapa de coleta, reconstruindo o DatetimeIndex."""
    df = pd.read_csv(raw_path, index_col="Date", parse_dates=True)
    return df.sort_index()


def clean(df: pd.DataFrame) -> pd.DataFrame:
    """Remove duplicatas de índice (mantém a última ocorrência) e linhas com Close nulo. Não
    interpola dias faltantes: dias sem pregão são omitidos por design, não preenchidos
    artificialmente."""
    df = df[~df.index.duplicated(keep="last")]
    return df.dropna(subset=["Close"])


def chronological_split(
    df: pd.DataFrame, test_size_ratio: float, validation_size_ratio: float
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Divide em train/validation/test respeitando ORDEM CRONOLÓGICA (jamais shuffle) — embaralhar
    série temporal vaza informação futura para o treino."""
    n = len(df)
    test_n = int(n * test_size_ratio)
    val_n = int(n * validation_size_ratio)
    train_end = n - test_n - val_n
    val_end = n - test_n
    return df.iloc[:train_end], df.iloc[train_end:val_end], df.iloc[val_end:]


def compute_log_returns(close: pd.Series) -> pd.Series:
    """Retorno logarítmico diário: log(Close_t / Close_{t-1}). A primeira posição é NaN (não há
    Close_{t-1} antes do início da série) — quem chamar deve descartar essa linha.

    Motivo de usar retorno em vez de preço absoluto como alvo do LSTM: ações com forte
    valorização/desvalorização ao longo dos anos (ex.: PETR4 saiu de ~R$2 em 2015 para ~R$40 em
    2026) fazem o `MinMaxScaler` ajustado só no treino ficar com um range completamente
    incompatível com o nível de preço do período de teste — o modelo passa a extrapolar fora da
    faixa em que foi treinado, degradando a previsão. Retorno logarítmico é aproximadamente
    estacionário (sua distribuição não muda sistematicamente com o nível de preço), então o
    scaler ajustado no treino continua válido para validação/teste."""
    return np.log(close / close.shift(1))


def make_windows(series: np.ndarray, lookback_window: int) -> tuple[np.ndarray, np.ndarray]:
    """Gera pares (X, y) por janela deslizante para o LSTM.

    X.shape == (n_amostras, lookback_window, 1)
    y.shape == (n_amostras,)
    """
    X, y = [], []
    for i in range(len(series) - lookback_window):
        X.append(series[i : i + lookback_window])
        y.append(series[i + lookback_window])
    X_arr = np.array(X).reshape(-1, lookback_window, 1)
    y_arr = np.array(y)
    return X_arr, y_arr
