"""Definicja potoku udoskonalania modelu."""

from kedro.pipeline import Pipeline, node, pipeline

from .nodes import automl, compare_models, feature_selection, select_best, tune_hyperparameters


def create_pipeline(**kwargs) -> Pipeline:
    return pipeline(
        [
            node(
                func=feature_selection,
                inputs=[
                    "X_train",
                    "X_test",
                    "y_train",
                    "params:feature_selection",
                    "params:random_state",
                ],
                outputs=["X_train_sel", "X_test_sel", "selected_features"],
                name="feature_selection_node",
            ),
            node(
                func=tune_hyperparameters,
                inputs=["X_train_sel", "y_train", "params:optuna", "params:random_state", "params:mlflow"],
                outputs="best_params",
                name="tune_node",
            ),
            node(
                func=compare_models,
                inputs=[
                    "X_train_sel",
                    "X_test_sel",
                    "y_train",
                    "y_test",
                    "best_params",
                    "params:model_params",
                    "params:candidates",
                    "params:random_state",
                    "params:mlflow",
                ],
                outputs="candidate_comparison",
                name="compare_node",
            ),
            node(
                func=automl,
                inputs=["train_df", "test_df", "params:automl", "params:mlflow"],
                outputs=["automl_leaderboard", "automl_metrics"],
                name="automl_node",
            ),
            node(
                func=select_best,
                inputs=[
                    "candidate_comparison",
                    "automl_metrics",
                    "best_params",
                    "X_train_sel",
                    "y_train",
                    "params:random_state",
                    "params:mlflow",
                ],
                outputs=["best_model", "model_comparison"],
                name="select_best_node",
            ),
        ]
    )
