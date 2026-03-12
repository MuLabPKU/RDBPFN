from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, List

from tqdm import tqdm
import h5py
import numpy as np

from dbinfer_bench_simplified.dataset_meta import (
    DBBColumnDType,
    DBBTaskType,
)
from dbinfer_bench_simplified.rdb_dataset import DBBRDBDataset, DBBRDBTask


NUMERIC_DTYPES = {
    DBBColumnDType.float_t,
    DBBColumnDType.datetime_t,
    DBBColumnDType.timestamp_t,
}

CATEGORICAL_DTYPES = {
    DBBColumnDType.category_t,
    DBBColumnDType.foreign_key,
    DBBColumnDType.primary_key,
    DBBColumnDType.text_t,
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Merge DBInfer datasets into a NanoTabPFN-compatible HDF5 prior."
    )
    parser.add_argument(
        "--dataset-root",
        type=Path,
        action="append",
        required=True,
        help="Root directory containing DBInfer datasets (each with metadata.yaml). Can be specified multiple times.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        required=True,
        help="Output HDF5 file path.",
    )
    parser.add_argument(
        "--total-rows",
        type=int,
        default=256,
        help="Number of rows per synthesized dataset (train+test).",
    )
    parser.add_argument(
        "--min-train-ratio",
        type=float,
        default=0.5,
        help="Lower bound for random train split ratio.",
    )
    parser.add_argument(
        "--max-train-ratio",
        type=float,
        default=0.8,
        help="Upper bound for random train split ratio.",
    )
    parser.add_argument(
        "--max-columns",
        type=int,
        default=128,
        help="Maximum number of feature columns to keep per task.",
    )
    parser.add_argument(
        "--importance-json",
        type=Path,
        action="append",
        default=None,
        help="Optional feature importance JSON (can be repeated).",
    )
    parser.add_argument(
        "--importance-top-k",
        type=int,
        default=None,
        help="Number of top features (per task) to keep when importance JSON is provided.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=0,
        help="Random seed.",
    )
    return parser.parse_args()


def _ensure_ratio_bounds(min_ratio: float, max_ratio: float):
    if not (0.0 < min_ratio < max_ratio < 1.0):
        raise ValueError("Require 0 < min_train_ratio < max_train_ratio < 1.")


def discover_datasets(roots: List[Path]) -> List[Path]:
    dataset_dirs: List[Path] = []
    for root in roots:
        if not root.exists():
            raise FileNotFoundError(f"{root} does not exist.")
        for entry in sorted(root.iterdir()):
            if entry.is_dir() and (entry / "metadata.yaml").exists():
                dataset_dirs.append(entry)
    return dataset_dirs


def _is_numeric_column(column_meta, column_name: str) -> bool:
    dtype = column_meta[column_name].dtype
    if dtype in NUMERIC_DTYPES:
        return True
    if dtype in CATEGORICAL_DTYPES:
        return False
    return True


def _select_feature_names(
    task: DBBRDBTask,
    max_columns: int,
    importance_lookup: Dict[tuple[str, str], List[str]] | None,
    top_k: int | None,
    dataset_name: str,
) -> tuple[list[str], int]:
    target = task.metadata.target_column
    base_features = [
        col.name
        for col in task.metadata.columns
        if col.name != target
        and (
            col.dtype == DBBColumnDType.float_t
            or col.dtype == DBBColumnDType.category_t
        )
    ]
    key = (dataset_name, task.metadata.name)
    if importance_lookup and key in importance_lookup:
        ordered = [f for f in importance_lookup[key] if f in base_features]
        if top_k:
            ordered = ordered[:top_k]
        feature_names = ordered
    else:
        feature_names = base_features
        if len(feature_names) > max_columns:
            rng = np.random.default_rng(abs(hash(task.metadata.name)) % (2**32))
            indices = rng.choice(len(feature_names), size=max_columns, replace=False)
            indices.sort()
            feature_names = [feature_names[i] for i in indices]
    return feature_names[:max_columns], len(base_features)


def _build_encoders(task: DBBRDBTask, feature_names: list[str]) -> Dict[str, tuple[str, object]]:
    encoders = {}
    column_meta = task.metadata.column_dict
    for name in feature_names:
        values = task.train_set[name]
        if values.size == 0:
            raise ValueError(f"Column {name} in task {task.metadata.name} has no rows.")
        if _is_numeric_column(column_meta, name):
            arr = values.astype(np.float32, copy=False)
            finite = arr[np.isfinite(arr)]
            mean = float(finite.mean()) if finite.size else 0.0
            encoders[name] = ("numeric", mean)
        else:
            arr_str = values.astype(str)
            uniques = np.unique(arr_str)
            mapping = {val: idx for idx, val in enumerate(uniques)}
            encoders[name] = ("categorical", mapping)
    return encoders


def _transform_split(
    split_data: dict[str, np.ndarray],
    feature_names: list[str],
    encoders: dict[str, tuple[str, object]],
) -> np.ndarray:
    rows = len(next(iter(split_data.values())))
    columns = []
    for name in feature_names:
        encoder_type, info = encoders[name]
        if encoder_type == "numeric":
            arr = split_data[name].astype(np.float32, copy=False)
            if np.isnan(arr).any():
                arr = arr.copy()
                arr[np.isnan(arr)] = info
            columns.append(arr.reshape(rows, 1))
        else:
            arr = split_data[name].astype(str)
            mapping = info
            encoded = np.array(
                [mapping.get(val, len(mapping)) for val in arr], dtype=np.float32
            )
            columns.append(encoded.reshape(rows, 1))
    return np.concatenate(columns, axis=1)


def _binarize_labels(
    y_train: np.ndarray,
    y_test: np.ndarray,
    rng: np.random.Generator,
) -> tuple[np.ndarray, np.ndarray] | tuple[None, None]:
    combined = np.concatenate([y_train, y_test])
    combined_str = combined.astype(str)
    unique = np.unique(combined_str)
    if unique.size < 2:
        return None, None
    if unique.size == 2:
        positive = {unique[1]}
    else:
        shuffled = unique.copy()
        rng.shuffle(shuffled)
        split_idx = max(1, len(shuffled) // 2)
        positive = set(shuffled[:split_idx])

    def encode(values: np.ndarray) -> np.ndarray:
        arr = values.astype(str)
        encoded = np.array([1 if val in positive else 0 for val in arr], dtype=np.int64)
        if encoded.max() == 0:
            encoded[rng.integers(0, encoded.size)] = 1
        elif encoded.min() == 1:
            encoded[rng.integers(0, encoded.size)] = 0
        return encoded

    return encode(y_train), encode(y_test)


def _prepare_task_sample(
    dataset_name: str,
    task: DBBRDBTask,
    total_rows: int,
    min_ratio: float,
    max_ratio: float,
    max_columns: int,
    importance_lookup: Dict[tuple[str, str], List[str]] | None,
    top_k: int | None,
    rng: np.random.Generator,
) -> dict | None:
    if task.metadata.task_type != DBBTaskType.classification:
        return None
    if total_rows < 2:
        raise ValueError("total_rows must be at least 2.")
    feature_names, total_available_features = _select_feature_names(
        task, max_columns, importance_lookup, top_k, dataset_name
    )
    if not feature_names:
        return None
    encoders = _build_encoders(task, feature_names)

    X_train = _transform_split(task.train_set, feature_names, encoders)
    X_test = _transform_split(task.test_set, feature_names, encoders)
    y_train = task.train_set[task.metadata.target_column]
    y_test = task.test_set[task.metadata.target_column]
    if y_train.size == 0 or y_test.size == 0:
        return None
    y_train_bin, y_test_bin = _binarize_labels(y_train, y_test, rng)
    if y_train_bin is None:
        return None

    X_combined = np.concatenate([X_train, X_test], axis=0)
    y_combined = np.concatenate([y_train_bin, y_test_bin], axis=0)
    replace = X_combined.shape[0] < total_rows
    indices = rng.choice(X_combined.shape[0], size=total_rows, replace=replace)
    X_sampled = X_combined[indices]
    y_sampled = y_combined[indices]

    if y_sampled.max() == 0:
        y_sampled[rng.integers(0, total_rows)] = 1
    elif y_sampled.min() == 1:
        y_sampled[rng.integers(0, total_rows)] = 0

    train_ratio = float(rng.uniform(min_ratio, max_ratio))
    train_rows = int(round(total_rows * train_ratio))
    train_rows = max(1, min(train_rows, total_rows - 1))

    category_mask = np.array(
        [1 if encoders[name][0] == "categorical" else 0 for name in feature_names],
        dtype=np.uint8,
    )

    return {
        "dataset": dataset_name,
        "task": task.metadata.name,
        "X": X_sampled.astype(np.float32),
        "y": y_sampled.astype(np.int32),
        "num_features": X_sampled.shape[1],
        "num_available_features": total_available_features,
        "split_idx": train_rows,
        "category_mask": category_mask,
    }


def _load_all_samples(
    dataset_dirs: list[Path],
    total_rows: int,
    min_ratio: float,
    max_ratio: float,
    max_columns: int,
    importance_lookup: Dict[tuple[str, str], List[str]] | None,
    top_k: int | None,
    seed: int,
) -> list[dict]:
    rng = np.random.default_rng(seed)
    samples: list[dict] = []
    for dataset_path in tqdm(dataset_dirs):
        try:
            dataset = DBBRDBDataset(dataset_path)
        except Exception as exc:
            print(f"Skipping {dataset_path}: {exc}")
            continue
        for task in dataset.tasks:
            sample = _prepare_task_sample(
                dataset.dataset_name,
                task,
                total_rows,
                min_ratio,
                max_ratio,
                max_columns,
                importance_lookup,
                top_k,
                rng,
            )
            if sample is not None:
                samples.append(sample)
    return samples


def _write_hdf5(samples: list[dict], output: Path, total_rows: int, max_columns: int):
    if not samples:
        raise RuntimeError("No samples to write.")
    output.parent.mkdir(parents=True, exist_ok=True)
    total = len(samples)
    chunk_rows = min(4, total)
    with h5py.File(output, "w") as h5:
        dset_X = h5.create_dataset(
            "X",
            shape=(total, total_rows, max_columns),
            dtype="float32",
            compression="lzf",
            chunks=(chunk_rows, total_rows, max_columns),
        )
        dset_y = h5.create_dataset(
            "y",
            shape=(total, total_rows),
            dtype="int32",
            compression="lzf",
            chunks=(chunk_rows, total_rows),
        )
        dset_num_features = h5.create_dataset("num_features", shape=(total,), dtype="int32")
        dset_num_available_features = h5.create_dataset(
            "num_available_features", data=np.zeros(total, dtype="int32")
        )
        dset_num_datapoints = h5.create_dataset(
            "num_datapoints", data=np.zeros(total, dtype="int32")
        )
        dset_single_eval_pos = h5.create_dataset(
            "single_eval_pos", data=np.zeros(total, dtype="int32")
        )
        dset_category_mask = h5.create_dataset(
            "feature_is_categorical",
            shape=(total, max_columns),
            dtype="uint8",
            compression="lzf",
        )
        h5.create_dataset("max_num_classes", data=np.array([1], dtype="int32"))

        num_datapoints_buffer = np.full(total, total_rows, dtype=np.int32)
        num_features_buffer = np.zeros(total, dtype=np.int32)
        num_available_buffer = np.zeros(total, dtype=np.int32)
        split_idx_buffer = np.zeros(total, dtype=np.int32)
        for idx, sample in enumerate(samples):
            cols = sample["num_features"]
            dset_X[idx, :, :cols] = sample["X"]
            dset_y[idx, :] = sample["y"]
            num_features_buffer[idx] = cols
            num_available_buffer[idx] = sample["num_available_features"]
            split_idx_buffer[idx] = sample["split_idx"]
            dset_category_mask[idx, :cols] = sample["category_mask"]
            if (idx + 1) % 50 == 0 or idx + 1 == total:
                print(f"Written {idx + 1}/{total} tasks", end="\r", flush=True)
        dset_num_features[...] = num_features_buffer
        dset_num_available_features[...] = num_available_buffer
        dset_num_datapoints[...] = num_datapoints_buffer
        dset_single_eval_pos[...] = split_idx_buffer
    print(f"\nSuccessfully wrote {total} tasks to {output}")


def main():
    args = parse_args()
    _ensure_ratio_bounds(args.min_train_ratio, args.max_train_ratio)
    if args.max_columns <= 0:
        raise ValueError("max-columns must be positive.")
    importance_lookup = None
    if args.importance_json:
        importance_lookup = {}
        for json_path in args.importance_json:
            with json_path.open("r", encoding="utf-8") as fh:
                importance_data = json.load(fh)
            for entry in importance_data:
                key = (entry["dataset"], entry["task"])
                ordered_features = [item["feature"] for item in entry["importances"]]
                importance_lookup[key] = ordered_features
        if args.importance_top_k is None:
            raise ValueError("importance-top-k must be specified when importance JSON is used.")
    dataset_dirs = discover_datasets(args.dataset_root)
    if not dataset_dirs:
        raise RuntimeError("No datasets found under the specified root.")

    samples = _load_all_samples(
        dataset_dirs,
        args.total_rows,
        args.min_train_ratio,
        args.max_train_ratio,
        args.max_columns,
        importance_lookup,
        args.importance_top_k,
        args.seed,
    )
    _write_hdf5(samples, args.output, args.total_rows, args.max_columns)


if __name__ == "__main__":
    main()
