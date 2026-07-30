"""
Train RandomForest model for IITM OPPE MLOps project.

Features
--------
1. Reads configuration from params.yaml
2. Loads processed dataset
3. Splits train/test
4. Performs GridSearchCV
5. Evaluates best model
6. Saves model
7. Logs to MLflow
8. Saves metrics.json
"""

import argparse
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
from sklearn.model_selection import GridSearchCV, train_test_split

################################################################################
# Configuration
################################################################################


def load_config(config_path: str = "params.yaml") -> dict:
    """
    Load YAML configuration.
    """
    with open(config_path, "r") as f:
        return yaml.safe_load(f)


CONFIG = load_config()

MODEL_DIR = Path(CONFIG["paths"]["model_dir"])
METRICS_DIR = Path(CONFIG["paths"]["metrics_dir"])

MODEL_DIR.mkdir(parents=True, exist_ok=True)
METRICS_DIR.mkdir(parents=True, exist_ok=True)

################################################################################
# MLflow
################################################################################

mlflow.set_tracking_uri(CONFIG["mlflow"]["tracking_uri"])

mlflow.set_experiment(CONFIG["mlflow"]["experiment_name"])

################################################################################
# Dataset
################################################################################

FEATURE_COLUMNS = [
    "sepal_length",
    "sepal_width",
    "petal_length",
    "petal_width",
]

TARGET_COLUMN = "target"


def get_dataset(iteration: int) -> Path:
    """
    Return processed dataset path.
    """

    if iteration == 1:
        return Path(CONFIG["dataset"]["processed"]["iteration1"])

    if iteration == 2:
        return Path(CONFIG["dataset"]["processed"]["iteration2"])

    raise ValueError("Iteration must be 1 or 2.")


################################################################################
# Metrics
################################################################################


def compute_metrics(y_true, y_pred) -> dict:
    """
    Compute evaluation metrics.
    """

    return {
        "accuracy": float(
            accuracy_score(
                y_true,
                y_pred,
            )
        ),
        "precision": float(
            precision_score(
                y_true,
                y_pred,
                average="weighted",
                zero_division=0,
            )
        ),
        "recall": float(
            recall_score(
                y_true,
                y_pred,
                average="weighted",
                zero_division=0,
            )
        ),
        "f1_score": float(
            f1_score(
                y_true,
                y_pred,
                average="weighted",
                zero_division=0,
            )
        ),
        "confusion_matrix": confusion_matrix(
            y_true,
            y_pred,
        ).tolist(),
        "classification_report": classification_report(
            y_true,
            y_pred,
            output_dict=True,
            zero_division=0,
        ),
    }


################################################################################
# Training
################################################################################


def train(iteration: int):
    """
    Train model for one iteration.
    """

    dataset = get_dataset(iteration)

    print(f"\nLoading dataset: {dataset}")

    df = pd.read_csv(dataset)

    X = df[FEATURE_COLUMNS]
    y = df[TARGET_COLUMN]

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=CONFIG["model"]["test_size"],
        random_state=CONFIG["model"]["random_state"],
        stratify=y,
    )

    param_grid = CONFIG["model"]["parameters"]

    estimator = RandomForestClassifier(random_state=CONFIG["model"]["random_state"])

    grid = GridSearchCV(
        estimator=estimator,
        param_grid=param_grid,
        cv=CONFIG["model"]["cv"],
        scoring="accuracy",
        n_jobs=-1,
        refit=True,
        verbose=1,
    )

    print("\nRunning GridSearchCV...")

    grid.fit(
        X_train,
        y_train,
    )

    best_model = grid.best_estimator_

    predictions = best_model.predict(X_test)

    metrics = compute_metrics(
        y_test,
        predictions,
    )

    model_path = MODEL_DIR / f"model_iteration_{iteration}.pkl"

    metrics_path = METRICS_DIR / f"metrics_iteration_{iteration}.json"

    return (
        best_model,
        model_path,
        metrics,
        metrics_path,
        grid.best_params_,
        grid.best_score_,
    )


################################################################################
# Main
################################################################################


def main():

    parser = argparse.ArgumentParser(description="Train RandomForest model.")

    parser.add_argument(
        "--iteration",
        type=int,
        required=True,
        choices=[1, 2],
        help="Training iteration number.",
    )

    args = parser.parse_args()

    iteration = args.iteration

    print("=" * 70)
    print(f"TRAINING ITERATION {iteration}")
    print("=" * 70)

    (
        model,
        model_path,
        metrics,
        metrics_path,
        best_params,
        cv_score,
    ) = train(iteration)

    #
    # Save model
    #
    joblib.dump(model, model_path)

    #
    # Save metrics
    #
    import json

    with open(metrics_path, "w") as f:
        json.dump(metrics, f, indent=4)

    #
    # MLflow logging
    #
    with mlflow.start_run(run_name=f"training_iteration_{iteration}"):

        #
        # Best hyperparameters
        #
        mlflow.log_params(best_params)

        #
        # Cross validation score
        #
        mlflow.log_metric(
            "cv_accuracy",
            float(cv_score),
        )

        #
        # Test metrics
        #
        mlflow.log_metric(
            "accuracy",
            metrics["accuracy"],
        )

        mlflow.log_metric(
            "precision",
            metrics["precision"],
        )

        mlflow.log_metric(
            "recall",
            metrics["recall"],
        )

        mlflow.log_metric(
            "f1_score",
            metrics["f1_score"],
        )

        #
        # Artifacts
        #
        mlflow.log_artifact(str(model_path))
        mlflow.log_artifact(str(metrics_path))

    #
    # Console summary
    #
    print("\n")
    print("=" * 70)
    print("Training Complete")
    print("=" * 70)

    print("\nBest Parameters")

    for key, value in best_params.items():
        print(f"{key:20} : {value}")

    print(f"\nCross Validation Accuracy : {cv_score:.4f}")

    print("\nTest Metrics")

    print(f"Accuracy  : {metrics['accuracy']:.4f}")
    print(f"Precision : {metrics['precision']:.4f}")
    print(f"Recall    : {metrics['recall']:.4f}")
    print(f"F1 Score  : {metrics['f1_score']:.4f}")

    print("\nArtifacts")

    print(f"Model   : {model_path}")
    print(f"Metrics : {metrics_path}")

    print("=" * 70)


if __name__ == "__main__":
    main()
