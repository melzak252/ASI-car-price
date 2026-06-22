import mlflow

from tracking.mlflow_utils import log_run, setup_mlflow


def test_setup_and_log_run_creates_run(tmp_path):
    db_path = str(tmp_path / "mlflow.db").replace("\\", "/")
    tracking_uri = f"sqlite:///{db_path}"
    setup_mlflow("test_experiment", tracking_uri)

    log_run("dummy_run", params={"alpha": 1.0}, metrics={"R2_log": 0.9})

    experiment = mlflow.get_experiment_by_name("test_experiment")
    assert experiment is not None
    runs = mlflow.search_runs(experiment_ids=[experiment.experiment_id])
    assert len(runs) == 1
    assert runs.iloc[0]["params.alpha"] == "1.0"
    assert runs.iloc[0]["metrics.R2_log"] == 0.9
