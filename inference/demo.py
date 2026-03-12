from __future__ import annotations

import sys
from pathlib import Path

from sklearn.datasets import load_breast_cancer
from sklearn.metrics import accuracy_score, roc_auc_score
from sklearn.model_selection import train_test_split

from src.predictor import RDBPFNClassifier


def main() -> None:
    X, y = load_breast_cancer(return_X_y=True)
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.5,
        random_state=0,
        stratify=y,
    )

    clf = RDBPFNClassifier.from_pretrained("RDBPFN")
    clf.fit(X_train, y_train)

    prob = clf.predict_proba(X_test)
    pred = clf.predict(X_test)

    print("ROC AUC", roc_auc_score(y_test, prob[:, 1]))
    print("Accuracy", accuracy_score(y_test, pred))


if __name__ == "__main__":
    main()
