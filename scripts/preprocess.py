import argparse
from pathlib import Path

import numpy as np
import pandas as pd

RAW_DIR = Path("data/raw")
PROCESSED_DIR = Path("data/processed")

PROCESSED_DIR.mkdir(parents=True, exist_ok=True)

FEATURE_COLUMNS = [
    "sepal_length",
    "sepal_width",
    "petal_length",
    "petal_width",
]

TARGET_MAPPING = {
    "setosa": 0,
    "versicolor": 1,
    "virginica": 2,
    "Iris-setosa": 0,
    "Iris-versicolor": 1,
    "Iris-virginica": 2,
}


def validate_columns(df: pd.DataFrame):
    """
    Ensure all required columns are present.
    """

    required = FEATURE_COLUMNS + ["species"]

    missing = set(required) - set(df.columns)

    if missing:
        raise ValueError(f"Missing columns: {missing}")


def encode_target(df: pd.DataFrame):
    """
    Encode species labels.
    """

    df["target"] = df["species"].map(TARGET_MAPPING)

    if df["target"].isna().any():
        raise ValueError("Unknown species found.")

    return df


def impute_last10_species_mean(df: pd.DataFrame):
    """
    Impute missing values using the mean of
    the last 10 available samples belonging
    to the same species.
    """

    df = df.copy()

    for species in df["species"].unique():

        species_idx = df[df["species"] == species].index

        for feature in FEATURE_COLUMNS:

            for idx in species_idx:

                if pd.isna(df.loc[idx, feature]):

                    previous = df.loc[
                        species_idx[species_idx < idx],
                        feature,
                    ].dropna()

                    if len(previous) == 0:

                        fallback = (
                            df.loc[
                                species_idx,
                                feature,
                            ]
                            .dropna()
                            .mean()
                        )

                        df.loc[idx, feature] = fallback

                    else:

                        last10 = previous.tail(10)

                        df.loc[idx, feature] = last10.mean()

    return df


def save_processed(df, name):

    output = PROCESSED_DIR / f"{name}_processed.csv"

    df.to_csv(output, index=False)

    print(f"Saved {output}")


def preprocess_dataset(dataset_name):

    input_file = RAW_DIR / f"{dataset_name}.csv"

    print(f"Loading {input_file}")

    df = pd.read_csv(input_file)

    validate_columns(df)

    df = encode_target(df)

    df = impute_last10_species_mean(df)

    save_processed(df, dataset_name)


def merge_datasets():

    v0 = pd.read_csv(PROCESSED_DIR / "iris_v0_processed.csv")

    v1 = pd.read_csv(PROCESSED_DIR / "iris_v1_processed.csv")

    merged = pd.concat(
        [v0, v1],
        ignore_index=True,
    )

    merged.to_csv(
        PROCESSED_DIR / "iris_merged.csv",
        index=False,
    )

    print("Merged dataset saved.")


def main():

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--dataset",
        type=str,
        help="iris_v0 or iris_v1",
    )

    parser.add_argument(
        "--merge",
        action="store_true",
    )

    args = parser.parse_args()

    if args.dataset:

        preprocess_dataset(args.dataset)

    elif args.merge:

        merge_datasets()

    else:

        raise ValueError("Specify --dataset or --merge")


if __name__ == "__main__":

    main()
