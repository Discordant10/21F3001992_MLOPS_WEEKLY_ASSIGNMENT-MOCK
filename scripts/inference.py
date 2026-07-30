"""
Inference script for IITM OPPE MLOps Project.

Usage:

Single sample:
python scripts/inference.py --model models/model_iteration_2.pkl \
    --sample 5.1 3.5 1.4 0.2

Batch prediction:
python scripts/inference.py --model models/model_iteration_2.pkl \
    --input data/processed/iris_v1_processed.csv
"""

import argparse
from pathlib import Path

import joblib
import pandas as pd

FEATURE_COLUMNS = [
    "sepal_length",
    "sepal_width",
    "petal_length",
    "petal_width",
]

CLASS_MAPPING = {
    0: "setosa",
    1: "versicolor",
    2: "virginica",
}


def load_model(model_path: str):
    """
    Load trained model.
    """
    return joblib.load(model_path)


def predict_single(model, values):
    """
    Predict one flower.
    """
    df = pd.DataFrame([values], columns=FEATURE_COLUMNS)

    prediction = model.predict(df)[0]

    print("\nPrediction")
    print("-------------------------")
    print(f"Class ID : {prediction}")
    print(f"Species  : {CLASS_MAPPING[prediction]}")


def predict_batch(model, input_csv):
    """
    Predict an entire CSV.
    """
    df = pd.read_csv(input_csv)

    predictions = model.predict(df[FEATURE_COLUMNS])

    output = df.copy()

    output["prediction"] = predictions

    output["predicted_species"] = [CLASS_MAPPING[p] for p in predictions]

    output_path = Path("reports")

    output_path.mkdir(exist_ok=True)

    output_file = output_path / "predictions.csv"

    output.to_csv(
        output_file,
        index=False,
    )

    print(f"\nPredictions saved to\n{output_file}")


def main():

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--model",
        required=True,
    )

    parser.add_argument(
        "--input",
        default=None,
    )

    parser.add_argument(
        "--sample",
        nargs=4,
        type=float,
        default=None,
    )

    args = parser.parse_args()

    model = load_model(args.model)

    if args.sample is not None:

        predict_single(
            model,
            args.sample,
        )

    elif args.input is not None:

        predict_batch(
            model,
            args.input,
        )

    else:

        raise ValueError("Provide either --sample or --input")


if __name__ == "__main__":
    main()
