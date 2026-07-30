import os
import numpy as np
import pandas as pd
import pytest

DATA_PATH = "data/iris_data_adapted_for_feast.csv"
EXPECTED_COLUMNS = [
    "iris_id",
    "event_timestamp",
    "sepal_length",
    "sepal_width",
    "petal_length",
    "petal_width",
    "species",
    "created_timestamp",
]
FEATURE_COLUMNS = [
    "sepal_length",
    "sepal_width",
    "petal_length",
    "petal_width",
]
VALID_SPECIES = {
    "setosa",
    "versicolor",
    "virginica",
}


@pytest.fixture(scope="module")
def iris_dataframe():
    """Load the Feast-adapted Iris dataset."""

    assert os.path.exists(DATA_PATH), (
        f"Dataset not found at {DATA_PATH}. "
        "Did you forget to run 'dvc pull'?"
    )
    return pd.read_csv(DATA_PATH)


def test_dataset_exists():
    """Dataset file should exist."""
    assert os.path.exists(DATA_PATH)


def test_dataset_not_empty(iris_dataframe):
    """Dataset should not be empty."""
    assert len(iris_dataframe) > 0


def test_expected_columns(iris_dataframe):
    """Dataset schema should match the expected schema."""
    assert set(iris_dataframe.columns) == set(EXPECTED_COLUMNS)


def test_no_missing_values(iris_dataframe):
    """Dataset should not contain missing values."""
    assert iris_dataframe.isnull().sum().sum() == 0


def test_numeric_feature_types(iris_dataframe):
    """Feature columns must be numeric."""
    for column in FEATURE_COLUMNS:
        assert pd.api.types.is_numeric_dtype(
            iris_dataframe[column]
        ), f"{column} is not numeric"


def test_species_column_exists(iris_dataframe):
    """Species column should exist."""
    assert "species" in iris_dataframe.columns


def test_valid_species_labels(iris_dataframe):
    """Species labels should belong to the valid Iris classes."""
    labels = set(iris_dataframe["species"].unique())
    assert labels.issubset(VALID_SPECIES)


def test_species_present(iris_dataframe):
    """
    Dataset should contain at least two valid classes.
    The adapted Feast dataset supplied by the institute
    may not contain all three classes.
    """
    labels = set(iris_dataframe["species"].unique())
    assert len(labels) >= 2


def test_feature_values_are_finite(iris_dataframe):
    """
    Feature values should be finite numbers.
    """
    for column in FEATURE_COLUMNS:
        assert np.isfinite(iris_dataframe[column]).all(), (
            f"{column} contains NaN or infinite values."
        )


def test_reasonable_feature_ranges(iris_dataframe):
    """
    Feature values should lie within broad
    sanity limits.
    """
    for column in FEATURE_COLUMNS:
        invalid = iris_dataframe[
            ~iris_dataframe[column].between(-10, 20)
        ]
        assert invalid.empty, (
            f"{column} contains values outside "
            "the expected range."
        )


def test_timestamp_columns_exist(iris_dataframe):
    """Timestamp columns should be present."""
    assert "event_timestamp" in iris_dataframe.columns
    assert "created_timestamp" in iris_dataframe.columns


def test_iris_id_exists(iris_dataframe):
    """iris_id column should exist."""
    assert "iris_id" in iris_dataframe.columns