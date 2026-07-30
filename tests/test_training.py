import json
import os
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.append(str(PROJECT_ROOT))

import pandas as pd
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from sklearn.model_selection import train_test_split

from mlflow_utils import load_registered_model

DATA_PATH = "data/iris_data_adapted_for_feast.csv"
METRICS_PATH = "metrics/train_metrics.json"

FEATURE_COLUMNS = [
    "sepal_length",
    "sepal_width",
    "petal_length",
    "petal_width",
]

TARGET_COLUMN = "species"

MIN_ACCURACY = 0.90
MIN_PRECISION = 0.90
MIN_RECALL = 0.90
MIN_F1 = 0.90


def load_dataset():
    assert os.path.exists(DATA_PATH), (
        "Dataset not found. " "Run 'dvc pull' before executing tests."
    )
    return pd.read_csv(DATA_PATH)


# Load model from MLflow Registry
def load_model():
    model = load_registered_model()
    assert model is not None
    return model


def prepare_test_data(df):
    X = df[FEATURE_COLUMNS]
    y = df[TARGET_COLUMN]
    _, X_test, _, y_test = train_test_split(
        X,
        y,
        test_size=0.20,
        random_state=42,
        stratify=y,
    )
    return X_test, y_test


def test_model_can_be_loaded():
    model = load_model()
    assert model is not None


def test_model_prediction_shape():
    model = load_model()
    df = load_dataset()
    X_test, y_test = prepare_test_data(df)
    predictions = model.predict(X_test)
    assert len(predictions) == len(y_test)


def test_prediction_labels():
    model = load_model()
    df = load_dataset()
    X_test, _ = prepare_test_data(df)
    predictions = model.predict(X_test)
    valid = {
        "setosa",
        "versicolor",
        "virginica",
    }
    assert set(predictions).issubset(valid)


def test_model_accuracy():
    model = load_model()
    df = load_dataset()
    X_test, y_test = prepare_test_data(df)
    predictions = model.predict(X_test)
    accuracy = accuracy_score(
        y_test,
        predictions,
    )
    assert accuracy >= MIN_ACCURACY


def test_model_precision():
    model = load_model()
    df = load_dataset()
    X_test, y_test = prepare_test_data(df)
    predictions = model.predict(X_test)
    precision = precision_score(
        y_test,
        predictions,
        average="weighted",
    )
    assert precision >= MIN_PRECISION


def test_model_recall():
    model = load_model()
    df = load_dataset()
    X_test, y_test = prepare_test_data(df)
    predictions = model.predict(X_test)
    recall = recall_score(
        y_test,
        predictions,
        average="weighted",
    )
    assert recall >= MIN_RECALL


def test_model_f1_score():
    model = load_model()
    df = load_dataset()
    X_test, y_test = prepare_test_data(df)
    predictions = model.predict(X_test)
    f1 = f1_score(
        y_test,
        predictions,
        average="weighted",
    )
    assert f1 >= MIN_F1


def test_metrics_json_exists():
    assert os.path.exists(METRICS_PATH)


def test_metrics_json_contents():
    with open(METRICS_PATH, "r") as f:
        metrics = json.load(f)
    required = [
        "train_accuracy",
        "train_precision",
        "train_recall",
        "train_f1_score",
        "test_accuracy",
        "test_precision",
        "test_recall",
        "test_f1_score",
        "cv_mean_accuracy",
        "cv_std_accuracy",
        "confusion_matrix",
    ]
    for key in required:
        assert key in metrics


def test_saved_metrics_threshold():
    with open(METRICS_PATH, "r") as f:
        metrics = json.load(f)
    assert metrics["test_accuracy"] >= MIN_ACCURACY
    assert metrics["test_precision"] >= MIN_PRECISION
    assert metrics["test_recall"] >= MIN_RECALL
    assert metrics["test_f1_score"] >= MIN_F1
