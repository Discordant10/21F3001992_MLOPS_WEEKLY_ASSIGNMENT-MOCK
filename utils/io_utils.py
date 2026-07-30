import json
from pathlib import Path

import joblib


def ensure_directory(path: str | Path) -> None:
    """
    Create directory if it doesn't exist.
    """
    Path(path).mkdir(parents=True, exist_ok=True)


def save_model(model, output_path: str | Path) -> None:
    """
    Save trained model.
    """
    output_path = Path(output_path)
    ensure_directory(output_path.parent)
    joblib.dump(model, output_path)


def load_model(model_path: str | Path):
    """
    Load trained model.
    """
    return joblib.load(model_path)


def save_json(data: dict, output_path: str | Path) -> None:
    """
    Save dictionary as JSON.
    """
    output_path = Path(output_path)
    ensure_directory(output_path.parent)

    with open(output_path, "w") as f:
        json.dump(data, f, indent=4)


def load_json(path: str | Path):
    with open(path) as f:
        return json.load(f)
