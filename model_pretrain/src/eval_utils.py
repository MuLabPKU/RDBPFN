from __future__ import annotations

import hashlib
import logging
import os
from pathlib import Path
from typing import Iterable, Tuple

import numpy as np
import pandas as pd

try:
    import openml
except ImportError:
    openml = None

from sklearn.datasets import load_breast_cancer
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    balanced_accuracy_score,
    f1_score,
    log_loss,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split

from .dbinfer_bench_simplified.rdb_dataset import DBBRDBTask
from .dbinfer_bench_simplified.dataset_meta import DBBColumnDType, DBBTaskEvalMetric


logger = logging.getLogger(__name__)

DEFAULT_EVAL_DIRS = [Path("datasets/clf_cat")]
SUBSAMPLE_SUFFIX = "_subsamples"


def _stable_random_state(name: str) -> int:
    digest = hashlib.sha256(name.encode("utf-8")).digest()
    return int.from_bytes(digest[:4], "big")


TARGET_COLUMN_CANDIDATES = [
    "target",
    "class",
    "label",
    "labels",
    "y",
    "response",
    "outcome",
]


def _infer_target_column(df: pd.DataFrame) -> str:
    lookup = {col.lower(): col for col in df.columns}
    for candidate in TARGET_COLUMN_CANDIDATES:
        lowered = candidate.lower()
        if lowered in lookup:
            return lookup[lowered]
    return df.columns[-1]


def _prepare_features(df: pd.DataFrame) -> np.ndarray:
    data = df.copy()
    for column in data.columns:
        series = data[column]
        if not pd.api.types.is_numeric_dtype(series):
            codes, _ = pd.factorize(series.astype(str), sort=True)
            series = pd.Series(codes, index=series.index)
        else:
            series = pd.to_numeric(series, errors="coerce")
        if series.isna().any():
            fill_value = float(series.median()) if not series.dropna().empty else 0.0
            series = series.fillna(fill_value)
        data[column] = series.astype(np.float32)
    return data.to_numpy(dtype=np.float32)


def _prepare_target(series: pd.Series) -> np.ndarray:
    codes, _ = pd.factorize(series)
    if (codes < 0).any():
        raise ValueError("Target column contains invalid values after encoding.")
    return codes.astype(np.int64)


def _load_csv_dataset(csv_path: Path) -> tuple[np.ndarray, np.ndarray]:
    df = pd.read_csv(csv_path)
    df = df.replace("?", np.nan)
    target_column = _infer_target_column(df)
    df = df.dropna(subset=[target_column])
    y = _prepare_target(df[target_column])
    X = _prepare_features(df.drop(columns=[target_column]))
    return X, y


def _subsample_dir_for(csv_dir: Path) -> Path:
    return csv_dir.parent / f"{csv_dir.name}{SUBSAMPLE_SUFFIX}"


def _load_local_csv_datasets(data_dirs: list[Path]):
    dataset_splits = []
    dataset_names = []
    for directory in data_dirs:
        if not directory.exists():
            continue
        subsample_dir = _subsample_dir_for(directory)
        subsample_dir.mkdir(parents=True, exist_ok=True)
        for csv_path in sorted(directory.glob("*.csv")):
            npz_path = subsample_dir / f"{csv_path.stem}_split.npz"
            if npz_path.exists():
                data = np.load(npz_path, allow_pickle=True)
                X_train, X_test, y_train, y_test = (
                    data["X_train"],
                    data["X_test"],
                    data["y_train"],
                    data["y_test"],
                )
                dataset_splits.append((X_train, X_test, y_train, y_test))
                dataset_names.append(csv_path.stem)
                continue
            try:
                X, y = _load_csv_dataset(csv_path)
            except Exception as exc:
                print(f"Skipping {csv_path}: {exc}")
                continue
            if len(np.unique(y)) < 2:
                print(f"Skipping {csv_path}: target has only one class.")
                continue
            random_state = _stable_random_state(csv_path.name)
            X_train, X_test, y_train, y_test = train_test_split(
                X,
                y,
                test_size=0.3,
                random_state=random_state,
                stratify=y,
            )
            X_train, y_train = downsample_split(
                X_train, y_train, max_samples=1000, seed=random_state
            )
            X_test, y_test = downsample_split(
                X_test, y_test, max_samples=1000, seed=random_state
            )
            np.savez_compressed(
                npz_path,
                X_train=X_train,
                X_test=X_test,
                y_train=y_train,
                y_test=y_test,
            )
            dataset_splits.append((X_train, X_test, y_train, y_test))
            dataset_names.append(csv_path.stem)
    return dataset_splits, dataset_names


def prepare_eval_splits(
    data_dirs: list[Path] | None = None,
):
    dirs = data_dirs if data_dirs else DEFAULT_EVAL_DIRS
    splits, names = _load_local_csv_datasets(dirs)
    return splits, names


def evaluate_classifier(classifier, splits):
    scores = {"roc_auc": 0, "acc": 0, "balanced_acc": 0}
    for X_train, X_test, y_train, y_test in splits:
        classifier.fit(X_train, y_train)
        prob = classifier.predict_proba(X_test)
        pred = prob.argmax(axis=1)
        if prob.shape[1] == 2:
            roc = roc_auc_score(y_test, prob[:, 1])
        else:
            roc = roc_auc_score(y_test, prob, multi_class="ovr")
        scores["roc_auc"] += float(roc)
        scores["acc"] += float(accuracy_score(y_test, pred))
        scores["balanced_acc"] += float(balanced_accuracy_score(y_test, pred))
    scores = {k: v / len(splits) for k, v in scores.items()}
    return scores


def load_task_split(task: DBBRDBTask, split: str) -> Tuple[np.ndarray, np.ndarray]:
    if split == "train":
        source = task.train_set
    elif split in {"val", "validation"}:
        source = task.validation_set
    elif split == "test":
        source = task.test_set
    else:
        raise ValueError(f"Unknown split {split}")
    if source is None:
        raise ValueError(f"Task {task.metadata.name} has no {split} data")
    target_col = task.metadata.target_column
    feature_cols = [
        col.name
        for col in task.metadata.columns
        if (
            col.dtype == DBBColumnDType.float_t
            or col.dtype == DBBColumnDType.category_t
        )
        and col.name != target_col
        # # If a column is feature includes max, min, sum, remove it
        # and not any(sub in col.name for sub in ("MAX", "MIN", "SUM"))
    ]
    # print(feature_cols)
    if not feature_cols:
        raise ValueError(f"Task {task.metadata.name} has no feature columns")
    feature_cols.sort()
    X = np.column_stack([source[col] for col in feature_cols]).astype(np.float32)
    y = np.asarray(source[target_col])
    return X, y


def downsample_split(
    X: np.ndarray, y: np.ndarray, max_samples: int | None, seed: int
) -> Tuple[np.ndarray, np.ndarray]:
    if max_samples is None or len(X) <= max_samples:
        return X, y
    rng = np.random.default_rng(seed)
    indices = rng.choice(len(X), size=max_samples, replace=False)
    return X[indices], y[indices]


def predict_proba_in_chunks(
    classifier, X: np.ndarray, chunk_size: int | None
) -> np.ndarray:
    if chunk_size is None or len(X) <= chunk_size:
        return classifier.predict_proba(X)
    probs = []
    for start in range(0, len(X), chunk_size):
        end = start + chunk_size
        probs.append(classifier.predict_proba(X[start:end]))
    return np.concatenate(probs, axis=0)


def fill_nans(
    X_train: np.ndarray, X_test: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    X_train = X_train.copy()
    X_test = X_test.copy()

    for col_idx in range(X_train.shape[1]):
        train_col = X_train[:, col_idx]
        test_col = X_test[:, col_idx]

        train_has_nan = np.isnan(train_col).any()
        test_has_nan = np.isnan(test_col).any()

        if train_has_nan or test_has_nan:
            # Compute fill value from training data only
            median_val = np.nanmedian(train_col)
            fill_value = median_val if not np.isnan(median_val) else 0.0

            if train_has_nan:
                X_train[np.isnan(X_train[:, col_idx]), col_idx] = fill_value
            if test_has_nan:
                X_test[np.isnan(X_test[:, col_idx]), col_idx] = fill_value

    return X_train, X_test


# =============================================================================
# Metric Computation
# =============================================================================


def compute_metric(
    metric: DBBTaskEvalMetric,
    y_true: np.ndarray,
    prob: np.ndarray,
    pred: np.ndarray,
) -> float:
    """Compute a classification metric.

    Args:
        metric: The metric type to compute
        y_true: Ground truth labels
        prob: Predicted probabilities (shape: [n_samples, n_classes])
        pred: Predicted class labels

    Returns:
        The computed metric value, or NaN if computation fails.
    """
    try:
        if metric == DBBTaskEvalMetric.accuracy:
            return float(accuracy_score(y_true, pred))
        if metric == DBBTaskEvalMetric.f1:
            return float(f1_score(y_true, pred, average="weighted"))
        if metric == DBBTaskEvalMetric.recall:
            return float(recall_score(y_true, pred, average="weighted"))
        if metric == DBBTaskEvalMetric.ap:
            if prob.shape[1] == 2:
                return float(average_precision_score(y_true, prob[:, 1]))
            return float(average_precision_score(y_true, prob, average="macro"))
        if metric == DBBTaskEvalMetric.auroc:
            if prob.shape[1] == 2:
                return float(roc_auc_score(y_true, prob[:, 1]))
            return float(roc_auc_score(y_true, prob, multi_class="ovr"))
        if metric == DBBTaskEvalMetric.balanced_acc:
            return float(balanced_accuracy_score(y_true, pred))
        if metric == DBBTaskEvalMetric.logloss:
            return float(log_loss(y_true, prob))
    except ValueError as exc:
        logger.warning("Failed to compute metric %s: %s", metric, exc)
        return float("nan")
    logger.warning("Metric %s not implemented; returning NaN", metric)
    return float("nan")


# =============================================================================
# Path Utilities
# =============================================================================


def derive_output_name(checkpoint_path_str: str | None) -> str:
    """Derive a CSV output name from the checkpoint path.

    Formats:
        checkpoints/xxx.pt            -> xxx
        checkpoints/xxx/              -> xxx
        checkpoints/xxx/model.pt      -> xxx
        checkpoints/xxx/model_evalyyy.pt -> xxx_yyy
    """
    if not checkpoint_path_str:
        return "results"
    if "checkpoints" not in checkpoint_path_str:
        return checkpoint_path_str
    path = Path(checkpoint_path_str)

    # If it's a directory (or ends with /), use directory name
    if checkpoint_path_str.endswith("/") or path.suffix == "":
        return path.name

    # It's a .pt file
    filename = path.stem  # filename without .pt

    if filename == "model":
        # checkpoints/xxx/model.pt -> xxx
        return path.parent.name
    elif filename.startswith("model_eval"):
        # checkpoints/xxx/model_evalyyy.pt -> xxx_yyy
        suffix = filename[len("model_eval") :]  # extract yyy
        return f"{path.parent.name}_{suffix}"
    else:
        # checkpoints/xxx.pt -> xxx
        return filename


def simplify_model_label(label: str) -> str:
    """Extract relative path under checkpoints/ from full path."""
    # Find "checkpoints/" in the path and keep everything after it
    marker = "checkpoints/"
    idx = label.find(marker)
    if idx != -1:
        return label[idx + len(marker) :]
    # Fallback: just return the basename
    return Path(label).name


# =============================================================================
# Results I/O
# =============================================================================


def save_results_to_csv(
    all_model_results: dict[str, list[dict]],
    output_path: Path,
    metric_key: str = "metric_value",
) -> None:
    """Save results to CSV with tasks as columns and models as rows.

    Args:
        all_model_results: Dict mapping model_label -> list of aggregated task results
        output_path: Path to save the CSV file
        metric_key: Which metric to use as the cell value (default: "metric_value")
    """
    df = build_results_dataframe(all_model_results, metric_key=metric_key)
    df.to_csv(output_path, index=False)
    logger.info("Results saved to %s", output_path)


def build_results_dataframe(
    all_model_results: dict[str, list[dict]],
    metric_key: str = "metric_value",
) -> pd.DataFrame:
    """Build a dataframe with tasks as columns and models as rows."""
    # Collect all unique task names (dataset/task) across all models
    all_tasks: set[str] = set()
    for results in all_model_results.values():
        for r in results:
            task_col = f"{r['dataset']}/{r['task']}"
            all_tasks.add(task_col)

    # Sort tasks for consistent column order
    task_columns = sorted(all_tasks)

    # Build rows: each row is a model, columns are tasks
    rows = []
    for model_label, results in all_model_results.items():
        # Simplify model label to relative path under checkpoints/
        simplified_label = simplify_model_label(model_label)
        row = {"model": simplified_label}
        task_to_metric = {f"{r['dataset']}/{r['task']}": r[metric_key] for r in results}
        for task_col in task_columns:
            value = task_to_metric.get(task_col, float("nan"))
            # Round to 4 decimal places
            row[task_col] = round(value, 4) if not np.isnan(value) else value
        # Add average across all tasks
        values = [row[t] for t in task_columns if not np.isnan(row[t])]
        row["average"] = round(float(np.mean(values)), 4) if values else float("nan")
        rows.append(row)

    df = pd.DataFrame(rows)
    # Reorder columns: model first, then tasks, then average
    column_order = ["model"] + task_columns + ["average"]
    df = df[column_order]
    return df


def append_results_to_csv(
    all_model_results: dict[str, list[dict]],
    output_path: Path,
    metric_key: str = "metric_value",
) -> None:
    """Append new model rows into an existing results CSV, unioning columns."""
    new_df = build_results_dataframe(all_model_results, metric_key=metric_key)

    if output_path.exists():
        existing_df = pd.read_csv(output_path)
        # Union columns, keeping order: model, sorted tasks, average (if present)
        all_cols = list(existing_df.columns)
        for col in new_df.columns:
            if col not in all_cols:
                all_cols.insert(-1 if "average" in all_cols else len(all_cols), col)
        for col in all_cols:
            if col not in existing_df.columns:
                existing_df[col] = np.nan
            if col not in new_df.columns:
                new_df[col] = np.nan
        combined = pd.concat([existing_df[all_cols], new_df[all_cols]], ignore_index=True)
        # Drop duplicate models keeping the last occurrence
        combined = combined.drop_duplicates(subset=["model"], keep="last")
        combined.to_csv(output_path, index=False)
        logger.info("Appended results to %s", output_path)
    else:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        new_df.to_csv(output_path, index=False)
        logger.info("Results saved to %s", output_path)
