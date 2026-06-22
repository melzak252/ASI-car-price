import numpy as np
import pandas as pd


def predict_log_price(model, X: pd.DataFrame) -> np.ndarray:
    """Zwraca predykcję logarytmu ceny (`Log_Price`)."""
    return model.predict(X)


def predict_price(model, X: pd.DataFrame) -> np.ndarray:
    """Zwraca predykcję ceny w PLN (odwrócenie logarytmu)."""
    return np.exp(model.predict(X))
