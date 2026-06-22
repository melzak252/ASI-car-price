"""AutoML na danych ogłoszeń przy użyciu AutoGluon."""

import pandas as pd
from autogluon.tabular import TabularPredictor

from evaluation.metrics import price_metrics


def _prepare(df: pd.DataFrame, label: str, exclude_columns: list[str]) -> pd.DataFrame:
    """Zostawia target i cechy surowe, usuwając identyfikatory, leakage i kolumny-listy.

    AutoGluon sam zajmuje się kodowaniem i imputacją.
    Kolumny typu lista (np. `feat_list`) nie są odrzucane.
    """
    keep = [c for c in df.columns if c not in exclude_columns]
    data = df[keep].copy()
    list_columns = [c for c in data.columns if data[c].apply(lambda v: isinstance(v, list)).any()]
    if list_columns:
        data = data.drop(columns=list_columns)
    if label not in data.columns:
        data[label] = df[label]
    return data


def run_autogluon(
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    label: str = "Log_Price",
    time_limit: int = 300,
    presets: str = "medium_quality",
    eval_metric: str = "root_mean_squared_error",
    exclude_columns: list[str] | None = None,
) -> tuple[pd.DataFrame, dict]:
    """Trenuje AutoGluon i zwraca leaderboard oraz metryki najlepszego modelu na teście."""
    exclude_columns = exclude_columns or []
    train_data = _prepare(train_df, label, exclude_columns)
    test_data = _prepare(test_df, label, exclude_columns)

    predictor = TabularPredictor(
        label=label, problem_type="regression", eval_metric=eval_metric
    ).fit(train_data, time_limit=time_limit, presets=presets)

    leaderboard = predictor.leaderboard(test_data).reset_index(drop=True)

    y_pred_log = predictor.predict(test_data.drop(columns=[label]))
    metrics = price_metrics(test_data[label].to_numpy(), y_pred_log.to_numpy())
    return leaderboard, metrics
