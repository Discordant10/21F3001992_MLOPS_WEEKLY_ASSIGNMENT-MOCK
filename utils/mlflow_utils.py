import mlflow


def initialize_mlflow(config: dict) -> None:
    """
    Configure MLflow.
    """

    mlflow.set_tracking_uri(config["mlflow"]["tracking_uri"])

    mlflow.set_experiment(config["mlflow"]["experiment_name"])


def log_best_run(best_params: dict, metrics: dict, model_path: str) -> None:
    """
    Log final results.
    """

    mlflow.log_params(best_params)

    mlflow.log_metrics(
        {
            "accuracy": metrics["accuracy"],
            "precision": metrics["precision"],
            "recall": metrics["recall"],
            "f1_score": metrics["f1_score"],
        }
    )

    mlflow.log_artifact(model_path)
