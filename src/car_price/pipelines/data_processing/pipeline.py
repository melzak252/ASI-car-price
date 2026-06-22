"""Definicja pipeline'u preprocessingu i inżynierii cech."""

from kedro.pipeline import Pipeline, node, pipeline

from .nodes import build_features, preprocess, split


def create_pipeline(**kwargs) -> Pipeline:
    return pipeline(
        [
            node(
                func=preprocess,
                inputs="raw_car_ads",
                outputs="preprocessed_car_ads",
                name="preprocess_node",
            ),
            node(
                func=split,
                inputs=["preprocessed_car_ads", "params:test_size", "params:random_state"],
                outputs=["train_df", "test_df"],
                name="split_node",
            ),
            node(
                func=build_features,
                inputs=[
                    "train_df",
                    "test_df",
                    "params:categorical_columns",
                    "params:numeric_columns",
                    "params:drop_columns",
                    "params:target",
                ],
                outputs=["X_train", "X_test", "y_train", "y_test"],
                name="build_features_node",
            ),
        ]
    )
