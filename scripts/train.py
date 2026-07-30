import json
import os
import tempfile

import mlflow
import mlflow.sklearn
import pandas as pd
import yaml
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)
from sklearn.model_selection import cross_val_score, train_test_split

# Dataset and output locations

DATA_PATH = "data/iris_data_adapted_for_feast.csv"

METRICS_DIR = "metrics"
METRICS_PATH = os.path.join(
    METRICS_DIR,
    "train_metrics.json",
)

PARAMS_PATH = "params.yaml"


# Helper functions
def load_dataset(path):
    if not os.path.exists(path):
        raise FileNotFoundError(f"Dataset not found: {path}")
    return pd.read_csv(path)


def load_params():
    if not os.path.exists(PARAMS_PATH):
        raise FileNotFoundError(f"Parameters file not found: {PARAMS_PATH}")
    with open(PARAMS_PATH, "r") as f:
        return yaml.safe_load(f)


def preprocess(df):
    feature_columns = [
        "sepal_length",
        "sepal_width",
        "petal_length",
        "petal_width",
    ]
    target_column = "species"
    X = df[feature_columns]
    y = df[target_column]
    return X, y


def build_model(n_estimators, max_depth):
    return RandomForestClassifier(
        n_estimators=n_estimators,
        max_depth=max_depth,
        random_state=42,
    )


def train_model(model, X_train, y_train):
    model.fit(
        X_train,
        y_train,
    )
    return model


def calculate_metrics(y_true, predictions):
    accuracy = accuracy_score(
        y_true,
        predictions,
    )
    precision = precision_score(
        y_true,
        predictions,
        average="weighted",
        zero_division=0,
    )
    recall = recall_score(
        y_true,
        predictions,
        average="weighted",
        zero_division=0,
    )
    f1 = f1_score(
        y_true,
        predictions,
        average="weighted",
        zero_division=0,
    )
    return {
        "accuracy": float(accuracy),
        "precision": float(precision),
        "recall": float(recall),
        "f1_score": float(f1),
    }


def evaluate_model(
    model,
    X_train,
    y_train,
    X_test,
    y_test,
):
    train_predictions = model.predict(X_train)
    test_predictions = model.predict(X_test)
    train_metrics = calculate_metrics(
        y_train,
        train_predictions,
    )
    test_metrics = calculate_metrics(
        y_test,
        test_predictions,
    )
    cv_scores = cross_val_score(
        model,
        pd.concat([X_train, X_test]),
        pd.concat([y_train, y_test]),
        cv=5,
        scoring="accuracy",
    )
    return {
        "train_accuracy": train_metrics["accuracy"],
        "train_precision": train_metrics["precision"],
        "train_recall": train_metrics["recall"],
        "train_f1_score": train_metrics["f1_score"],
        "test_accuracy": test_metrics["accuracy"],
        "test_precision": test_metrics["precision"],
        "test_recall": test_metrics["recall"],
        "test_f1_score": test_metrics["f1_score"],
        "cv_mean_accuracy": float(cv_scores.mean()),
        "cv_std_accuracy": float(cv_scores.std()),
        "confusion_matrix": confusion_matrix(
            y_test,
            test_predictions,
        ).tolist(),
    }


def save_metrics(metrics):
    os.makedirs(
        METRICS_DIR,
        exist_ok=True,
    )
    with open(
        METRICS_PATH,
        "w",
    ) as f:
        json.dump(
            metrics,
            f,
            indent=4,
        )
    print(f"Metrics saved to " f"{METRICS_PATH}")


def log_to_mlflow(
    params,
    model,
    metrics,
):
    tracking_uri = params["mlflow"]["tracking_uri"]
    experiment_name = params["mlflow"]["experiment_name"]
    registered_model_name = params["mlflow"]["registered_model_name"]
    mlflow.set_tracking_uri(tracking_uri)
    mlflow.set_experiment(experiment_name)
    with mlflow.start_run():
        mlflow.log_param(
            "n_estimators",
            params["model"]["n_estimators"],
        )
        mlflow.log_param(
            "max_depth",
            params["model"]["max_depth"],
        )
        for key, value in metrics.items():
            if key != "confusion_matrix":
                mlflow.log_metric(
                    key,
                    value,
                )
        with tempfile.NamedTemporaryFile(
            mode="w",
            suffix=".json",
            delete=False,
        ) as tmp:
            json.dump(
                metrics["confusion_matrix"],
                tmp,
                indent=4,
            )
            confusion_matrix_file = tmp.name
        mlflow.log_artifact(
            confusion_matrix_file,
            artifact_path="confusion_matrix",
        )
        os.remove(confusion_matrix_file)
        mlflow.sklearn.log_model(
            sk_model=model,
            artifact_path="model",
            registered_model_name=registered_model_name,
        )
        print(f"Model registered as " f"{registered_model_name}")


# Training starts here


def main():
    params = load_params()
    model_params = params["model"]
    n_estimators = model_params["n_estimators"]
    max_depth = model_params["max_depth"]
    print("Loading dataset...")
    df = load_dataset(DATA_PATH)
    X, y = preprocess(df)
    (
        X_train,
        X_test,
        y_train,
        y_test,
    ) = train_test_split(
        X,
        y,
        test_size=0.20,
        random_state=42,
        stratify=y,
    )
    print(
        f"Training model "
        f"(n_estimators="
        f"{n_estimators}, "
        f"max_depth="
        f"{max_depth})"
    )
    model = build_model(
        n_estimators,
        max_depth,
    )
    model = train_model(
        model,
        X_train,
        y_train,
    )
    metrics = evaluate_model(
        model,
        X_train,
        y_train,
        X_test,
        y_test,
    )
    print("\nEvaluation Metrics")
    for key, value in metrics.items():
        print(f"{key}: {value}")
    save_metrics(metrics)
    log_to_mlflow(
        params=params,
        model=model,
        metrics=metrics,
    )
    print("\nTraining completed " "successfully.")


if __name__ == "__main__":
    main()
