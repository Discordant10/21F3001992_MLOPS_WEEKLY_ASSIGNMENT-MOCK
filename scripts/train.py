import argparse
import json
from itertools import product
from pathlib import Path

import joblib
import mlflow
import mlflow.sklearn
import pandas as pd
import yaml
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)
from sklearn.model_selection import train_test_split

# --------------------------------------------------------
# Configuration
# --------------------------------------------------------

CONFIG = yaml.safe_load(open("params.yaml"))

DATA_DIR = Path("data/processed")

MODEL_DIR = Path("models")
MODEL_DIR.mkdir(exist_ok=True)

METRIC_DIR = Path("metrics")
METRIC_DIR.mkdir(exist_ok=True)


# --------------------------------------------------------
# MLflow
# --------------------------------------------------------

mlflow.set_tracking_uri(CONFIG["mlflow"]["tracking_uri"])

mlflow.set_experiment(CONFIG["mlflow"]["experiment_name"])


# --------------------------------------------------------
# Dataset helpers
# --------------------------------------------------------

FEATURE_COLUMNS = [
    "sepal_length",
    "sepal_width",
    "petal_length",
    "petal_width",
]


TARGET_COLUMN = "target"


def load_iteration_dataset(iteration):

    if iteration == 1:
        return DATA_DIR / "iris_v0_processed.csv"

    if iteration == 2:
        return DATA_DIR / "iris_merged.csv"

    raise ValueError("Iteration must be 1 or 2")


def load_dataset(path):

    if not path.exists():
        raise FileNotFoundError(path)

    return pd.read_csv(path)


def split_features_target(df):

    X = df[FEATURE_COLUMNS]

    y = df[TARGET_COLUMN]

    return X, y


# --------------------------------------------------------
# Metrics
# --------------------------------------------------------


def calculate_metrics(
    y_true,
    y_pred,
):

    return {
        "accuracy": accuracy_score(
            y_true,
            y_pred,
        ),
        "precision": precision_score(
            y_true,
            y_pred,
            average="weighted",
            zero_division=0,
        ),
        "recall": recall_score(
            y_true,
            y_pred,
            average="weighted",
            zero_division=0,
        ),
        "f1_score": f1_score(
            y_true,
            y_pred,
            average="weighted",
            zero_division=0,
        ),
        "classification_report": classification_report(
            y_true,
            y_pred,
            output_dict=True,
            zero_division=0,
        ),
        "confusion_matrix": confusion_matrix(
            y_true,
            y_pred,
        ).tolist(),
    }


# --------------------------------------------------------
# Save helpers
# --------------------------------------------------------


def save_model(
    model,
    path,
):

    joblib.dump(
        model,
        path,
    )


def save_metrics(
    metrics,
    path,
):

    with open(path, "w") as f:

        json.dump(
            metrics,
            f,
            indent=4,
        )


# --------------------------------------------------------
# Hyperparameter Search
# --------------------------------------------------------


def parameter_grid():

    hp = CONFIG["training"]["hyperparameter_search"]

    return product(
        hp["n_estimators"],
        hp["max_depth"],
        hp["min_samples_split"],
        hp["criterion"],
    )


# --------------------------------------------------------
# Model Training
# --------------------------------------------------------


def train_model(
    dataset_path,
    iteration,
):

    df = load_dataset(
        dataset_path,
    )

    X, y = split_features_target(df)

    X_train, X_val, y_train, y_val = train_test_split(
        X,
        y,
        test_size=CONFIG["training"]["test_size"],
        random_state=CONFIG["training"]["random_state"],
        stratify=y,
    )

    best_model = None

    best_metrics = None

    best_params = None

    best_accuracy = -1

    # ----------------------------------------------------
    # Hyperparameter Search
    # ----------------------------------------------------

    for (
        n_estimators,
        max_depth,
        min_samples_split,
        criterion,
    ) in parameter_grid():

        params = {
            "iteration": iteration,
            "n_estimators": n_estimators,
            "max_depth": max_depth,
            "min_samples_split": min_samples_split,
            "criterion": criterion,
        }

        with mlflow.start_run(
            nested=True,
            run_name=f"iter_{iteration}_{n_estimators}_{max_depth}",
        ):

            model = RandomForestClassifier(
                n_estimators=n_estimators,
                max_depth=max_depth,
                min_samples_split=min_samples_split,
                criterion=criterion,
                random_state=CONFIG["training"]["random_state"],
            )

            model.fit(
                X_train,
                y_train,
            )

            predictions = model.predict(
                X_val,
            )

            metrics = calculate_metrics(
                y_val,
                predictions,
            )

            mlflow.log_params(params)

            mlflow.log_metrics(
                {
                    "accuracy": metrics["accuracy"],
                    "precision": metrics["precision"],
                    "recall": metrics["recall"],
                    "f1_score": metrics["f1_score"],
                }
            )

            mlflow.sklearn.log_model(
                sk_model=model,
                artifact_path="model",
            )

            if metrics["accuracy"] > best_accuracy:

                best_accuracy = metrics["accuracy"]

                best_model = model

                best_metrics = metrics

                best_params = params

    return (
        best_model,
        best_metrics,
        best_params,
    )


# --------------------------------------------------------
# Main
# --------------------------------------------------------


def main():

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--iteration",
        required=True,
        type=int,
    )

    args = parser.parse_args()

    dataset = load_iteration_dataset(
        args.iteration,
    )

    print()

    print(f"Training using {dataset}")

    print()

    with mlflow.start_run(
        run_name=f"Training_Iteration_{args.iteration}",
    ):

        model, metrics, params = train_model(
            dataset,
            args.iteration,
        )

        mlflow.log_params(
            {
                "best_n_estimators": params["n_estimators"],
                "best_max_depth": params["max_depth"],
                "best_min_samples_split": params["min_samples_split"],
                "best_criterion": params["criterion"],
            }
        )

        mlflow.log_metrics(
            {
                "best_accuracy": metrics["accuracy"],
                "best_precision": metrics["precision"],
                "best_recall": metrics["recall"],
                "best_f1": metrics["f1_score"],
            }
        )

        model_path = MODEL_DIR / f"model_iteration_{args.iteration}.pkl"

        metric_path = METRIC_DIR / f"iteration{args.iteration}_metrics.json"

        save_model(
            model,
            model_path,
        )

        save_metrics(
            metrics,
            metric_path,
        )

        mlflow.log_artifact(
            model_path,
        )

        mlflow.log_artifact(
            metric_path,
        )

        print()

        print("=" * 60)

        print("TRAINING COMPLETE")

        print("=" * 60)

        print()

        print(f"Iteration : {args.iteration}")

        print()

        print("Best Parameters")

        print("----------------")

        for k, v in params.items():

            if k != "iteration":

                print(f"{k:20}: {v}")

        print()

        print("Metrics")

        print("-------")

        print(
            json.dumps(
                metrics,
                indent=4,
            )
        )

        print()

        print(f"Model saved to : {model_path}")

        print(f"Metrics saved : {metric_path}")

        print()

        print("=" * 60)


if __name__ == "__main__":

    main()
