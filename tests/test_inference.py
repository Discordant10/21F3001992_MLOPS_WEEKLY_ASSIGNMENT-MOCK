import joblib
import pandas as pd

FEATURE_COLUMNS = [
    "sepal_length",
    "sepal_width",
    "petal_length",
    "petal_width",
]


def test_prediction():

    model = joblib.load("models/model_iteration_1.pkl")

    sample = pd.DataFrame(
        [[5.1, 3.5, 1.4, 0.2]],
        columns=FEATURE_COLUMNS,
    )

    prediction = model.predict(sample)[0]

    assert prediction in [0, 1, 2]


def test_batch_prediction():

    model = joblib.load("models/model_iteration_1.pkl")

    df = pd.read_csv("data/processed/iris_v0_processed.csv")

    predictions = model.predict(df[FEATURE_COLUMNS])

    assert len(predictions) == len(df)
