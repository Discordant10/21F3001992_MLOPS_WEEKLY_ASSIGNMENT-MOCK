import pandas as pd


def test_processed_columns():

    df = pd.read_csv("data/processed/iris_v0_processed.csv")

    expected = {
        "sepal_length",
        "sepal_width",
        "petal_length",
        "petal_width",
        "species",
        "target",
    }

    assert expected.issubset(df.columns)
