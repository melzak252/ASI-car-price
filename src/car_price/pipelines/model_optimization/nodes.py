"""Węzły potoku udoskonalania modelu.

Każdy etap, który ocenia model, loguje przebieg do lokalnego MLflow.
Funkcje merytoryczne pochodzą z modułów pomocniczych w `src/`.
"""

import pandas as pd
from lightgbm import LGBMRegressor
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.linear_model import Ridge

from evaluation.metrics import price_metrics
from features.selection import apply_selection, select_features
from models.automl import run_autogluon
from models.train_model import train_lgbm
from models.tuning import tune_lgbm
from tracking.mlflow_utils import log_run, setup_mlflow


def _as_tuned(best_params: dict) -> dict:
    """Uzupełnia hiperparametry z Optuny o stałe ustawienia LightGBM."""
    return {**best_params, "objective": "regression", "n_jobs": -1, "verbosity": -1}


def _build_model(name: str, lgbm_default_params: dict, best_params: dict, random_state: int):
    if name == "ridge":
        return Ridge(alpha=1.0, random_state=random_state)
    if name == "hist_gbr":
        return HistGradientBoostingRegressor(random_state=random_state)
    if name == "lgbm_default":
        return LGBMRegressor(random_state=random_state, **lgbm_default_params)
    if name == "lgbm_tuned":
        return LGBMRegressor(random_state=random_state, **_as_tuned(best_params))
    raise ValueError(f"Nieznany model: {name}")


def feature_selection(
    X_train: pd.DataFrame,
    X_test: pd.DataFrame,
    y_train: pd.Series,
    feature_selection_params: dict,
    random_state: int,
) -> tuple[pd.DataFrame, pd.DataFrame, list[str]]:
    """Wybiera najważniejsze cechy i zawęża do nich obie macierze."""
    features = select_features(
        X_train, y_train, top_k=feature_selection_params["top_k"], random_state=random_state
    )
    return apply_selection(X_train, features), apply_selection(X_test, features), features


def tune_hyperparameters(
    X_train_sel: pd.DataFrame,
    y_train: pd.Series,
    optuna_params: dict,
    random_state: int,
    mlflow_params: dict,
) -> dict:
    """Stroi LightGBM (Optuna/TPE) i zapisuje najlepsze hiperparametry do MLflow."""
    setup_mlflow(**mlflow_params)
    best_params = tune_lgbm(
        X_train_sel,
        y_train,
        n_trials=optuna_params["n_trials"],
        cv_folds=optuna_params["cv_folds"],
        random_state=random_state,
    )
    log_run("optuna_best_params", params=best_params, metrics={})
    return best_params


def compare_models(
    X_train_sel: pd.DataFrame,
    X_test_sel: pd.DataFrame,
    y_train: pd.Series,
    y_test: pd.Series,
    best_params: dict,
    lgbm_default_params: dict,
    candidates: list[str],
    random_state: int,
    mlflow_params: dict,
) -> pd.DataFrame:
    """Trenuje i ocenia kilka modeli, każdy jako osobny przebieg MLflow."""
    setup_mlflow(**mlflow_params)
    results = []
    for name in candidates:
        model = _build_model(name, lgbm_default_params, best_params, random_state)
        model.fit(X_train_sel, y_train)
        metrics = price_metrics(y_test, model.predict(X_test_sel))
        log_run(name, params=model.get_params(), metrics=metrics, model=model)
        results.append({"model": name, **metrics})
    return pd.DataFrame(results)


def automl(
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    automl_params: dict,
    mlflow_params: dict,
) -> tuple[pd.DataFrame, dict]:
    """Uruchamia AutoGluon i loguje metryki najlepszego modelu do MLflow."""
    setup_mlflow(**mlflow_params)
    leaderboard, metrics = run_autogluon(train_df, test_df, **automl_params)
    log_run(
        "autogluon",
        params={"presets": automl_params["presets"], "time_limit": automl_params["time_limit"]},
        metrics=metrics,
    )
    return leaderboard, metrics


def select_best(
    candidate_comparison: pd.DataFrame,
    automl_metrics: dict,
    best_params: dict,
    X_train_sel: pd.DataFrame,
    y_train: pd.Series,
    random_state: int,
    mlflow_params: dict,
) -> tuple[LGBMRegressor, pd.DataFrame]:
    """Zestawia wyniki wszystkich modeli i zapisuje wdrażalny artefakt."""
    setup_mlflow(**mlflow_params)
    rows = candidate_comparison.to_dict("records")
    rows.append({"model": "autogluon", **automl_metrics})
    comparison = (
        pd.DataFrame(rows).sort_values("R2_log", ascending=False).reset_index(drop=True)
    )

    best_model = train_lgbm(
        X_train_sel, y_train, model_params=_as_tuned(best_params), random_state=random_state
    )
    tuned_metrics = next(
        (row for row in rows if row["model"] == "lgbm_tuned"), {}
    )
    tuned_metrics = {k: v for k, v in tuned_metrics.items() if k != "model"}
    log_run("best_model_lgbm_tuned", params=best_params, metrics=tuned_metrics, model=best_model)
    return best_model, comparison
