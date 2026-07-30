import argparse
from pathlib import Path

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
from utils import feature_target_split, load_dataset, save_metrics, save_model

CONFIG = yaml.safe_load(open("params.yaml"))

DATA_DIR = Path("data/processed")

MODEL_DIR = Path("models")

METRIC_DIR = Path("metrics")


def load_iteration_dataset(iteration):

    if iteration == 1:

        return DATA_DIR / "iris_v0_processed.csv"

    elif iteration == 2:

        return DATA_DIR / "iris_merged.csv"

    raise ValueError("Invalid iteration")


def train_model(dataset_path):

    df = load_dataset(dataset_path)

    X, y = feature_target_split(df)

    X_train, X_val, y_train, y_val = train_test_split(
        X,
        y,
        test_size=CONFIG["training"]["test_size"],
        random_state=CONFIG["training"]["random_state"],
        stratify=y,
    )

    model = RandomForestClassifier(
        n_estimators=100,
        max_depth=5,
        random_state=42,
    )

    model.fit(X_train, y_train)

    predictions = model.predict(X_val)

    metrics = {
        "accuracy": accuracy_score(y_val, predictions),
        "precision": precision_score(
            y_val,
            predictions,
            average="weighted",
        ),
        "recall": recall_score(
            y_val,
            predictions,
            average="weighted",
        ),
        "f1_score": f1_score(
            y_val,
            predictions,
            average="weighted",
        ),
        "classification_report": classification_report(
            y_val,
            predictions,
            output_dict=True,
        ),
        "confusion_matrix": confusion_matrix(
            y_val,
            predictions,
        ).tolist(),
    }

    return model, metrics


def main():

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--iteration",
        type=int,
        required=True,
    )

    args = parser.parse_args()

    dataset = load_iteration_dataset(args.iteration)

    print(f"Training using {dataset}")

    model, metrics = train_model(dataset)

    model_path = MODEL_DIR / f"model_iteration_{args.iteration}.pkl"

    metric_path = METRIC_DIR / f"iteration{args.iteration}_metrics.json"

    save_model(model, model_path)

    save_metrics(metrics, metric_path)

    print()

    print("Training Complete")

    print()

    print(metrics)


if __name__ == "__main__":

    main()
