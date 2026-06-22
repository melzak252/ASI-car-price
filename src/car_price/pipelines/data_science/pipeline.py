"""Definicja pipeline'u treningu i ewaluacji."""

from kedro.pipeline import Pipeline, node, pipeline

from .nodes import evaluate, train


def create_pipeline(**kwargs) -> Pipeline:
    return pipeline(
        [
            node(
                func=train,
                inputs=["X_train", "y_train", "params:model_params", "params:random_state"],
                outputs="model",
                name="train_node",
            ),
            node(
                func=evaluate,
                inputs=["model", "X_test", "y_test"],
                outputs="metrics",
                name="evaluate_node",
            ),
        ]
    )
