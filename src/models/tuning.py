"""Strojenie hiperparametrów LightGBM optymalizacją bayesowską (Optuna TPE)."""

import optuna
from lightgbm import LGBMRegressor
from sklearn.model_selection import cross_val_score


def tune_lgbm(
    X_train,
    y_train,
    n_trials: int = 30,
    cv_folds: int = 3,
    random_state: int = 42,
) -> dict:
    """Szuka najlepszych hiperparametrów minimalizując RMSE w walidacji krzyżowej.

    Optuna używa samplera TPE. Zwraca słownik najlepszych hiperparametrów.
    """
    optuna.logging.set_verbosity(optuna.logging.WARNING)

    def objective(trial: optuna.Trial) -> float:
        params = {
            "n_estimators": trial.suggest_int("n_estimators", 200, 800),
            "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.2, log=True),
            "num_leaves": trial.suggest_int("num_leaves", 16, 128),
            "subsample": trial.suggest_float("subsample", 0.6, 1.0),
            "colsample_bytree": trial.suggest_float("colsample_bytree", 0.6, 1.0),
            "min_child_samples": trial.suggest_int("min_child_samples", 5, 60),
            "reg_alpha": trial.suggest_float("reg_alpha", 1e-3, 10.0, log=True),
            "reg_lambda": trial.suggest_float("reg_lambda", 1e-3, 10.0, log=True),
        }
        model = LGBMRegressor(
            objective="regression",
            random_state=random_state,
            n_jobs=-1,
            verbosity=-1,
            **params,
        )
        scores = cross_val_score(
            model, X_train, y_train, cv=cv_folds, scoring="neg_root_mean_squared_error"
        )
        return -scores.mean()

    study = optuna.create_study(
        direction="minimize",
        sampler=optuna.samplers.TPESampler(seed=random_state),
    )
    study.optimize(objective, n_trials=n_trials)
    return study.best_params
