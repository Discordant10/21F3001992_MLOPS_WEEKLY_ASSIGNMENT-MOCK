from pathlib import Path

import joblib


def test_model_exists():
    """
    Verify that the trained model file exists.
    """
    model_path = Path("models/model_iteration_1.pkl")

    assert model_path.exists(), "Model file was not created."


def test_model_loads():
    """
    Verify that the model can be loaded.
    """
    model = joblib.load("models/model_iteration_1.pkl")

    assert model is not None
