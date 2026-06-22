"""Rejestr pipeline'ów projektu."""

from kedro.pipeline import Pipeline

from car_price.pipelines import data_processing as dp
from car_price.pipelines import data_science as ds


def register_pipelines() -> dict[str, Pipeline]:
    """Rejestruje potoki; `__default__` uruchamia oba po kolei."""
    data_processing_pipeline = dp.create_pipeline()
    data_science_pipeline = ds.create_pipeline()

    return {
        "data_processing": data_processing_pipeline,
        "data_science": data_science_pipeline,
        "__default__": data_processing_pipeline + data_science_pipeline,
    }
