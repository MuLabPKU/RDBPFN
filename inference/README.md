# Inference

This directory provides a lightweight standalone inference path for RDB_PFN.
It is intentionally isolated from the data generation, preprocessing, and model
pretraining pipeline elsewhere in the repository.

## Scope

The v1 inference package supports:

- numeric-only RDBPFN backbone
- local checkpoint loading from `inference/checkpoints/`
- flat `numpy` / `pandas` inputs
- binary classification
- multiclass classification via one-vs-rest
- ensembles to increase context size

## Install

From the `inference/` directory:

```bash
pip install numpy pandas scikit-learn torch
```

## Run

Run the scripts directly from the `inference/` directory.

## Python Usage

### Numpy arrays

```python
from sklearn.datasets import load_breast_cancer
from sklearn.metrics import accuracy_score, roc_auc_score
from sklearn.model_selection import train_test_split

import sys
from pathlib import Path

sys.path.insert(0, str(Path("src").resolve()))

from src.predictor import RDBPFNClassifier

X, y = load_breast_cancer(return_X_y=True)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.5)

clf = RDBPFNClassifier.from_pretrained("RDBPFN")
clf.fit(X_train, y_train)

prob = clf.predict_proba(X_test)
pred = clf.predict(X_test)

print("ROC AUC", roc_auc_score(y_test, prob[:, 1]))
print("Accuracy", accuracy_score(y_test, pred))
```

### Dataframes

```python
import pandas as pd
import sys
from pathlib import Path

sys.path.insert(0, str(Path("src").resolve()))

from src.predictor import RDBPFNClassifier

train_df = pd.read_csv("train.csv")
test_df = pd.read_csv("test.csv")

clf = RDBPFNClassifier.from_pretrained("RDBPFN")
clf.fit(train_df, target="label")

prob = clf.predict_proba(test_df.drop(columns=["label"]))
pred = clf.predict(test_df.drop(columns=["label"]))
```

## Script Usage

```bash
python predict_csv.py \
  --train train.csv \
  --test test.csv \
  --target label \
  --output predictions.csv
```

Example with the included tiny dataset:

```bash
python predict_csv.py \
  --train examples/demo_train.csv \
  --test examples/demo_test.csv \
  --target label \
  --output examples/demo_predictions.csv
```

## Demo

Run the built-in sklearn demo from the `inference/` directory:

```bash
python demo.py
```

Run the multiclass demo from the `inference/` directory:

```bash
python demo_multiclass.py
```

## Notes

- The estimator is sklearn-like: `fit`, `predict_proba`, `predict`.
- Object or categorical dataframe columns are ordinal-encoded from training
  data so users can try simple tables without extra preprocessing.
- `fit()` keeps at most 1024 training rows as context.
- If `1024 < n_train < 10000`, `fit()` builds a simple ensemble of random
  1024-row contexts and averages probabilities across them.
- The ensemble number is computed automatically as `ceil(n_train / 1024)`.
- For multiclass tasks, `predict_proba()` uses one-vs-rest binary heads and
  normalizes the per-class positive probabilities to sum to 1.
