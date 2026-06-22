import numpy as np
from sklearn.metrics import mean_absolute_error, r2_score, root_mean_squared_error


def regression_metrics(y_true, y_pred) -> dict[str, float]:
    return {
        "mae": float(mean_absolute_error(y_true, y_pred)),
        "rmse": float(root_mean_squared_error(y_true, y_pred)),
        "r2": float(r2_score(y_true, y_pred)),
    }


def price_metrics(y_true_log, y_pred_log) -> dict[str, float]:
    """Metryki w skali logarytmicznej oraz po powrocie do skali PLN (`np.exp`)."""
    y_price = np.exp(np.asarray(y_true_log))
    pred_price = np.exp(np.asarray(y_pred_log))
    return {
        "MAE_log": float(mean_absolute_error(y_true_log, y_pred_log)),
        "RMSE_log": float(root_mean_squared_error(y_true_log, y_pred_log)),
        "R2_log": float(r2_score(y_true_log, y_pred_log)),
        "MAE_PLN": float(mean_absolute_error(y_price, pred_price)),
        "RMSE_PLN": float(root_mean_squared_error(y_price, pred_price)),
    }
