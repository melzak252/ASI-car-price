"""Rejestr pipeline'ów projektu."""

from kedro.pipeline import Pipeline

from car_price.pipelines import data_processing as dp
from car_price.pipelines import data_science as ds
from car_price.pipelines import model_optimization as mo


def register_pipelines() -> dict[str, Pipeline]:
    """Rejestruje potoki; `__default__` to data_processing + data_science.

    `model_optimization` (Optuna + AutoGluon) jest osobno - trwa minuty,
    więc uruchamiamy go na żądanie: `kedro run --pipeline model_optimization`.
    """
    data_processing_pipeline = dp.create_pipeline()
    data_science_pipeline = ds.create_pipeline()

    return {
        "data_processing": data_processing_pipeline,
        "data_science": data_science_pipeline,
        "model_optimization": mo.create_pipeline(),
        "__default__": data_processing_pipeline + data_science_pipeline,
    }
