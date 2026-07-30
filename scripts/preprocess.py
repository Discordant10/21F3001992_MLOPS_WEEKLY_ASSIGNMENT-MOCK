"""
Preprocess Iris dataset.

Features:
1. Loads raw dataset.
2. Encodes species labels.
3. Handles missing values.
4. Saves processed dataset.
"""

import argparse
from pathlib import Path

import pandas as pd

TARGET_MAPPING = {
    "setosa": 0,
    "versicolor": 1,
    "virginica": 2,
}


FEATURES = [
    "sepal_length",
    "sepal_width",
    "petal_length",
    "petal_width",
]


def clean_species(series: pd.Series) -> pd.Series:
    """
    Normalise species names.

    Accepts:
    Iris-setosa
    setosa
    SETOSA
    """

    return (
        series.astype(str).str.lower().str.replace("iris-", "", regex=False).str.strip()
    )


def fill_missing_values(df: pd.DataFrame) -> pd.DataFrame:
    """
    Fill missing feature values using the
    previous 10 samples of the same class.

    If fewer than 10 exist,
    use all previous samples.
    """

    df = df.copy()

    for idx in range(len(df)):

        species = df.loc[idx, "species"]

        previous = df.iloc[:idx]

        previous = previous[previous["species"] == species]

        previous = previous.tail(10)

        for feature in FEATURES:

            if pd.isna(df.loc[idx, feature]):

                if len(previous):

                    value = previous[feature].mean()

                else:

                    value = df[feature].mean()

                df.loc[idx, feature] = value

    return df


def preprocess(dataset_name: str):

    raw_path = Path("data/raw") / f"{dataset_name}.csv"

    output_path = Path("data/processed") / f"{dataset_name}_processed.csv"

    print(f"Loading {raw_path}")

    df = pd.read_csv(raw_path)

    df["species"] = clean_species(df["species"])

    unknown = set(df["species"]) - set(TARGET_MAPPING.keys())

    if unknown:
        raise ValueError(f"Unknown species detected: {unknown}")

    df = fill_missing_values(df)

    df["target"] = df["species"].map(TARGET_MAPPING)

    output_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    df.to_csv(output_path, index=False)

    print(f"Saved {output_path}")


def main():

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--dataset",
        required=True,
        choices=["iris_v0", "iris_v1"],
    )

    args = parser.parse_args()

    preprocess(args.dataset)


if __name__ == "__main__":
    main()
