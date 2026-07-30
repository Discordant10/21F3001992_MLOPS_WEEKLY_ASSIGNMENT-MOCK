import yaml
import mlflow.pyfunc

# MLflow settings are stored in params.yaml

PARAMS_PATH = "params.yaml"

def load_params():
    with open(PARAMS_PATH, "r") as f:
        return yaml.safe_load(f)

def load_registered_model():
    params = load_params()

    tracking_uri = params["mlflow"]["tracking_uri"]
    model_name = params["mlflow"]["registered_model_name"]

    mlflow.set_tracking_uri(tracking_uri)

    # Load latest registered model
    model_uri = f"models:/{model_name}/latest"
    print(
        f"Loading model from MLflow: "
        f"{model_uri}"
    )
    return mlflow.pyfunc.load_model(model_uri)

