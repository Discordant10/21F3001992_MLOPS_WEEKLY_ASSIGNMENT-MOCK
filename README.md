# IITM BS MLOps OPPE-1 Mock

## Objective

Develop an end-to-end MLOps pipeline for classifying Iris flower species using:

- DVC
- MLflow
- GitHub Actions

---

## Pipeline

```
Raw Data

iris_v0
iris_v1

↓

DVC

↓

Preprocessing

↓

Training

↓

Hyperparameter Tuning

↓

MLflow Tracking

↓

Best Model

↓

Inference

↓

GitHub Actions
```

---

## Project Structure

```
data/
metrics/
models/
scripts/
tests/
```

---

## Training Workflow

Iteration 1

Train on

```
iris_v0
```

Iteration 2

Train on

```
iris_v0 + iris_v1
```

Missing values are imputed using

> Mean of the last 10 available samples belonging to the same species.

---

## Running

Preprocess

```bash
python scripts/preprocess.py
```

Train

```bash
python scripts/train.py
```

Inference

```bash
python scripts/inference.py
```

Testing

```bash
pytest
```

---

## MLOps Components

- DVC Data Versioning
- MLflow Experiment Tracking
- Hyperparameter Tuning
- GitHub Actions CI
- Automated Testing
