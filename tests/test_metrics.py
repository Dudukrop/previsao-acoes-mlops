import numpy as np

from src.evaluation.gate import ModelNotAcceptedError, check_acceptance
from src.evaluation.metrics import evaluate_all, mae, mape, rmse


def test_mae_simple_case():
    y_true = np.array([10.0, 20.0, 30.0])
    y_pred = np.array([12.0, 18.0, 33.0])
    assert abs(mae(y_true, y_pred) - 7 / 3) < 1e-9


def test_rmse_zero_when_predictions_match():
    y_true = np.array([1.0, 2.0, 3.0])
    assert rmse(y_true, y_true) == 0.0


def test_mape_percentage_scale():
    y_true = np.array([100.0, 200.0])
    y_pred = np.array([110.0, 190.0])
    result = mape(y_true, y_pred)
    assert abs(result - 7.5) < 1e-6


def test_evaluate_all_returns_all_keys():
    y_true = np.array([1.0, 2.0])
    y_pred = np.array([1.0, 2.0])
    result = evaluate_all(y_true, y_pred)
    assert set(result.keys()) == {"mae", "rmse", "mape"}
    assert result["mae"] == 0.0


def test_check_acceptance_raises_when_mape_exceeds_limit():
    try:
        check_acceptance(mape_model=10.0, mape_naive=2.0, max_mape_pct=5.0, must_beat_naive=False)
        raised = False
    except ModelNotAcceptedError:
        raised = True
    assert raised


def test_check_acceptance_passes_when_mape_within_limit():
    check_acceptance(mape_model=3.0, mape_naive=1.0, max_mape_pct=5.0, must_beat_naive=False)


def test_check_acceptance_soft_gate_does_not_raise():
    # must_beat_naive=True mas o modelo não supera o naive: deve apenas avisar, não bloquear.
    check_acceptance(mape_model=3.0, mape_naive=1.0, max_mape_pct=5.0, must_beat_naive=True)
