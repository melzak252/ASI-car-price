import pandas as pd
from lightgbm import LGBMRegressor


def train_lgbm(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    model_params: dict,
    random_state: int = 42,
) -> LGBMRegressor:
    """Trenuje model LightGBM na przygotowanej macierzy cech."""
    model = LGBMRegressor(random_state=random_state, **model_params)
    model.fit(X_train, y_train)
    return model
