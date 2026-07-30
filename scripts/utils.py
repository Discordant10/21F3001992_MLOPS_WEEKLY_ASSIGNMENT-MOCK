import json
from pathlib import Path

import joblib
import pandas as pd


def load_dataset(path):
    """
    Load processed dataset.
    """
    return pd.read_csv(path)


def save_model(model, output_path):
    """
    Save trained model.
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, output_path)


def save_metrics(metrics, output_file):
    """
    Save metrics as JSON.
    """
    output_file = Path(output_file)
    output_file.parent.mkdir(parents=True, exist_ok=True)

    with open(output_file, "w") as f:
        json.dump(metrics, f, indent=4)


def feature_target_split(df):

    X = df[
        [
            "sepal_length",
            "sepal_width",
            "petal_length",
            "petal_width",
        ]
    ]

    y = df["target"]

    return X, y
