from __future__ import annotations

from sklearn.datasets import load_wine
from sklearn.metrics import accuracy_score, roc_auc_score
from sklearn.model_selection import train_test_split

from src.predictor import RDBPFNClassifier


def main() -> None:
    X, y = load_wine(return_X_y=True)
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

    print("Multiclass ROC AUC", roc_auc_score(y_test, prob, multi_class="ovr"))
    print("Accuracy", accuracy_score(y_test, pred))


if __name__ == "__main__":
    main()
