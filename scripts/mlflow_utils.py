import mlflow
import mlflow.sklearn


def setup_mlflow(config):

    mlflow.set_tracking_uri(config["mlflow"]["tracking_uri"])

    mlflow.set_experiment(config["mlflow"]["experiment_name"])


def log_run(
    model,
    params,
    metrics,
    iteration,
):
    """
    Log one training run.
    """

    with mlflow.start_run(nested=True, run_name=f"Iteration_{iteration}"):

        mlflow.log_params(params)

        for k, v in metrics.items():

            if isinstance(v, (int, float)):

                mlflow.log_metric(k, v)

        mlflow.sklearn.log_model(
            model,
            artifact_path="model",
            registered_model_name="IrisClassifier",
        )
