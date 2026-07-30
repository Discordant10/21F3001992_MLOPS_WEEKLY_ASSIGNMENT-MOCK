from pathlib import Path

import pandas as pd
import yaml


def load_config(config_path: str = "params.yaml") -> dict:
    with open(config_path) as f:
        return yaml.safe_load(f)


def read_csv(path: str | Path) -> pd.DataFrame:
    return pd.read_csv(path)


def save_csv(df: pd.DataFrame, path: str | Path) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False)


def get_iteration_dataset(iteration: int, processed: bool = True) -> str:
    config = load_config()

    section = "processed" if processed else "raw"

    return config["dataset"][section][f"iteration{iteration}"]


FEATURE_COLUMNS = [
    "sepal_length",
    "sepal_width",
    "petal_length",
    "petal_width",
]

TARGET_COLUMN = "target"
