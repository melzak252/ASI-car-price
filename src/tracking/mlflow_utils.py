"""Pomocniki do logowania eksperymentów w MLflow."""

import logging

import mlflow
import mlflow.sklearn

logger = logging.getLogger(__name__)


def setup_mlflow(experiment_name: str, tracking_uri: str) -> None:
    """Ustawia lokalny tracking i wybiera (lub tworzy) eksperyment."""
    mlflow.set_tracking_uri(tracking_uri)
    mlflow.set_experiment(experiment_name)


def log_run(run_name: str, params: dict, metrics: dict, model=None) -> None:
    """Zapisuje jeden przebieg: parametry, metryki i opcjonalnie model."""
    with mlflow.start_run(run_name=run_name):
        if params:
            mlflow.log_params(params)
        if metrics:
            mlflow.log_metrics(metrics)
        if model is not None:
            try:
                # cloudpickle zamiast domyślnego skops, który odrzuca typy LightGBM jako "untrusted".
                mlflow.sklearn.log_model(model, name=run_name, serialization_format="cloudpickle")
            except Exception as exc:
                logger.warning("Nie udało się zalogować modelu '%s': %s", run_name, exc)
