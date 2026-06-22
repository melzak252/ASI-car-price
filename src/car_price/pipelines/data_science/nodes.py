"""Węzły pipeline'u treningu i ewaluacji."""

import logging

import pandas as pd

from evaluation.metrics import price_metrics
from models.predict_model import predict_log_price
from models.train_model import train_lgbm

logger = logging.getLogger(__name__)


def train(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    model_params: dict,
    random_state: int,
):
    """Trenuje model LightGBM."""
    return train_lgbm(X_train, y_train, model_params=model_params, random_state=random_state)


def evaluate(model, X_test: pd.DataFrame, y_test: pd.Series) -> dict[str, float]:
    """Ewaluuje model na zbiorze testowym (metryki log + PLN)."""
    y_pred = predict_log_price(model, X_test)
    metrics = price_metrics(y_test, y_pred)
    logger.info(
        "Ewaluacja modelu: R2_log=%.4f, MAE_PLN=%.0f, RMSE_PLN=%.0f",
        metrics["R2_log"],
        metrics["MAE_PLN"],
        metrics["RMSE_PLN"],
    )
    return metrics
