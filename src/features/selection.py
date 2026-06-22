"""Selekcja cech na podstawie ważności z LightGBM."""

import pandas as pd

from models.train_model import train_lgbm

# Lekki model do oszacowania ważności cech.
_IMPORTANCE_PARAMS = {
    "n_estimators": 200,
    "learning_rate": 0.05,
    "num_leaves": 64,
    "objective": "regression",
    "n_jobs": -1,
    "verbosity": -1,
}


def select_features(
    X_train: pd.DataFrame, y_train: pd.Series, top_k: int, random_state: int = 42
) -> list[str]:
    """Zwraca `top_k` najważniejszych cech wg ważności z LightGBM."""
    model = train_lgbm(X_train, y_train, model_params=_IMPORTANCE_PARAMS, random_state=random_state)
    importances = pd.Series(model.feature_importances_, index=X_train.columns)
    top_k = min(top_k, len(importances))
    return importances.sort_values(ascending=False).head(top_k).index.tolist()


def apply_selection(X: pd.DataFrame, features: list[str]) -> pd.DataFrame:
    """Zawęża macierz cech do wybranych kolumn."""
    return X[features].copy()
