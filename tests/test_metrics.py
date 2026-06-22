import numpy as np

from evaluation.metrics import price_metrics, regression_metrics


def test_regression_metrics_perfect_prediction():
    y = [1.0, 2.0, 3.0]
    result = regression_metrics(y, y)
    assert result["mae"] == 0.0
    assert result["rmse"] == 0.0
    assert result["r2"] == 1.0


def test_price_metrics_keys_and_pln_scale():
    y_log = np.log([10000.0, 20000.0, 30000.0])
    result = price_metrics(y_log, y_log)
    assert set(result) == {"MAE_log", "RMSE_log", "R2_log", "MAE_PLN", "RMSE_PLN"}
    assert result["MAE_PLN"] == 0.0
    assert result["R2_log"] == 1.0
