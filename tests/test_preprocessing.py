import numpy as np
import pandas as pd

from src.preprocessing.prepare_dataset import (
    chronological_split,
    clean,
    compute_log_returns,
    make_windows,
)


def _synthetic_df(n: int = 100) -> pd.DataFrame:
    dates = pd.date_range("2024-01-01", periods=n, freq="D")
    close = np.linspace(10, 20, n)
    return pd.DataFrame({"Close": close, "Volume": np.arange(n)}, index=dates)


def test_chronological_split_has_no_overlap():
    df = _synthetic_df(100)
    train, validation, test = chronological_split(df, test_size_ratio=0.15, validation_size_ratio=0.15)
    assert len(train) + len(validation) + len(test) == len(df)
    assert train.index.max() < validation.index.min()
    assert validation.index.max() < test.index.min()


def test_make_windows_shapes():
    series = np.arange(10, dtype=float)
    X, y = make_windows(series, lookback_window=3)
    assert X.shape == (7, 3, 1)
    assert y.shape == (7,)
    assert list(X[0].flatten()) == [0.0, 1.0, 2.0]
    assert y[0] == 3.0


def test_clean_drops_duplicate_index_and_null_close():
    dates = pd.to_datetime(["2024-01-01", "2024-01-01", "2024-01-02"])
    df = pd.DataFrame({"Close": [10.0, 11.0, None]}, index=dates)
    result = clean(df)
    assert len(result) == 1
    assert result["Close"].iloc[0] == 11.0


def test_compute_log_returns_first_value_is_nan():
    close = pd.Series([10.0, 11.0, 9.9])
    returns = compute_log_returns(close)
    assert np.isnan(returns.iloc[0])
    assert abs(returns.iloc[1] - np.log(11.0 / 10.0)) < 1e-9
