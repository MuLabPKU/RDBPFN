from __future__ import annotations

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    balanced_accuracy_score,
    f1_score,
    log_loss,
    recall_score,
    roc_auc_score,
)


def compute_classification_metric(
    metric: str,
    y_true: np.ndarray,
    prob: np.ndarray,
    pred: np.ndarray,
) -> float:
    metric_name = metric.lower()
    if metric_name == "accuracy":
        return float(accuracy_score(y_true, pred))
    if metric_name == "balanced_acc":
        return float(balanced_accuracy_score(y_true, pred))
    if metric_name == "f1":
        return float(f1_score(y_true, pred, average="weighted"))
    if metric_name == "recall":
        return float(recall_score(y_true, pred, average="weighted"))
    if metric_name == "ap":
        if prob.shape[1] == 2:
            return float(average_precision_score(y_true, prob[:, 1]))
        return float(average_precision_score(y_true, prob, average="macro"))
    if metric_name == "auroc":
        if prob.shape[1] == 2:
            return float(roc_auc_score(y_true, prob[:, 1]))
        return float(roc_auc_score(y_true, prob, multi_class="ovr"))
    if metric_name == "logloss":
        return float(log_loss(y_true, prob))
    raise ValueError(f"Unsupported metric '{metric}'.")


def summarize_classification(
    y_true: np.ndarray,
    prob: np.ndarray,
    pred: np.ndarray,
    primary_metric: str = "auroc",
) -> dict[str, float]:
    return {
        "metric": primary_metric,
        "metric_value": compute_classification_metric(
            primary_metric, y_true, prob, pred
        ),
        "accuracy": float(accuracy_score(y_true, pred)),
        "balanced_acc": float(balanced_accuracy_score(y_true, pred)),
    }
