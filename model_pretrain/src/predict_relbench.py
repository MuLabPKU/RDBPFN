"""Predict RelBench test rows from existing DBInfer/DFS caches, without uploading.

Training labels establish class semantics; test labels are never loaded here.
The official key set selects and orders cached rows, including across versions
that removed rows. Unknown/missing keys and ambiguous label mappings fail closed.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import time

import numpy as np
import pandas as pd
import yaml
from huggingface_hub import HfApi, hf_hub_download
from relbench.hf import RELBENCH_HF
from relbench.submit import LEADERBOARD_TASKS

BASE = Path(__file__).resolve().parents[1]


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def write_json(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(".tmp")
    temporary.write_text(json.dumps(value, indent=2) + "\n")
    temporary.replace(path)


def key_frame(arrays, metadata: dict, official: dict) -> pd.DataFrame:
    key_columns = [
        c["name"] for c in metadata["columns"]
        if c["dtype"] in {"primary_key", "foreign_key"}
    ]
    entity = official["entity_col"]
    if entity in key_columns:
        local_entity = entity
    elif len(key_columns) == 1:
        local_entity = key_columns[0]
    else:
        raise ValueError(f"Cannot resolve entity column {entity!r}: {key_columns}")
    local_time = metadata.get("time_column")
    if not local_time:
        raise ValueError("The cache has no time column")
    # DBInfer's dt2ts stores nanoseconds since the Unix epoch.
    times = np.asarray(arrays[local_time])
    times = pd.to_datetime(times, unit="ns") if times.dtype.kind in "iu" else pd.to_datetime(times)
    result = pd.DataFrame({entity: arrays[local_entity], official["time_col"]: times})
    if result.isna().any().any() or result.duplicated().any():
        raise ValueError("Null or duplicate entity/time keys in cached data")
    return result


def align_rows(local_keys: pd.DataFrame, official_keys: pd.DataFrame) -> tuple[np.ndarray, dict]:
    if official_keys.isna().any().any() or official_keys.duplicated().any():
        raise ValueError("Null or duplicate official keys")
    if local_keys.isna().any().any() or local_keys.duplicated().any():
        raise ValueError("Null or duplicate local keys")
    local = pd.MultiIndex.from_frame(local_keys[official_keys.columns])
    official = pd.MultiIndex.from_frame(official_keys)
    positions = local.get_indexer(official)
    missing = int((positions < 0).sum())
    summary = {
        "cached_rows": len(local), "official_rows": len(official),
        "missing_official_rows": missing,
        "excluded_cached_rows": int((~local.isin(official)).sum()),
        "same_row_order": local.equals(official),
    }
    return positions, summary


def positive_class(local_labels: np.ndarray, official_labels: np.ndarray) -> int:
    """Resolve a bijective binary label encoding using matched TRAIN rows only."""
    if set(np.unique(local_labels)) != {0, 1} or set(np.unique(official_labels)) != {0, 1}:
        raise ValueError("Both binary classes must be present in matched training labels")
    mapping = {}
    for label in (0, 1):
        values = np.unique(official_labels[local_labels == label])
        if len(values) != 1:
            raise ValueError("Cached and official training targets disagree beyond a class permutation")
        mapping[label] = int(values[0])
    if set(mapping.values()) != {0, 1}:
        raise ValueError("Non-bijective training label mapping")
    return next(label for label, official in mapping.items() if official == 1)


def get_official(dataset: str, task: str, revision: str, split: str = "test"):
    if split not in {"test", "val"}:
        raise ValueError("Expected test or val split")
    def fetch(name):
        return Path(hf_hub_download(
            RELBENCH_HF, f"{dataset}/tasks/{task}/{name}",
            repo_type="dataset", revision=revision,
        ))
    metadata = yaml.safe_load(fetch("manifest.yaml").read_text())
    if metadata["task_type"] != "binary_classification":
        raise ValueError("Expected a binary classification task")
    keys = [metadata["entity_col"], metadata["time_col"]]
    # Project only keys from the test parquet. Never use test targets for
    # alignment, class encoding, checkpoint selection, or inference.
    test = pd.read_parquet(fetch(f"{split}.parquet"), columns=keys)
    train = pd.read_parquet(fetch("train.parquet"), columns=keys + [metadata["target_col"]])
    return metadata, train, test


def inspect_task(data_root: Path, full_name: str, revision: str, split: str = "test"):
    dataset, name = full_name.split("/")
    folder = data_root / f"{dataset}-dfs-2"
    metadata_path = folder / "metadata.yaml"
    metadata = yaml.safe_load(metadata_path.read_text())
    task = next(t for t in metadata["tasks"] if t["name"] == name)
    if task["task_type"] != "classification" or task["format"] != "numpy":
        raise ValueError("Expected a classification NPZ cache")
    official, train, test_keys = get_official(dataset, name, revision, split)
    keys = [official["entity_col"], official["time_col"]]
    train_path = folder / task["source"].format(split="train")
    test_path = folder / task["source"].format(split="validation" if split == "val" else split)
    with np.load(train_path, allow_pickle=True) as arrays:
        train_keys = key_frame(arrays, task, official)
        train_labels = np.asarray(arrays[task["target_column"]])
    with np.load(test_path, allow_pickle=True) as arrays:
        local_test_keys = key_frame(arrays, task, official)
    test_positions, test_summary = align_rows(local_test_keys, test_keys)
    train_positions, train_summary = align_rows(train_keys, train[keys])
    if test_summary["missing_official_rows"]:
        raise ValueError(f"Missing official {split} rows: {test_summary}")
    matched = train_positions >= 0
    official_train_labels = train.loc[matched, official["target_col"]].to_numpy()
    positive = positive_class(train_labels[train_positions[matched]], official_train_labels)
    # Preserve the original cached training order for the existing seeded
    # context sampler. Rows absent from the official train split are excluded.
    eligible = np.sort(train_positions[matched])
    if len(eligible) == 0:
        raise ValueError("No usable training context")
    if train_keys.iloc[eligible][keys[1]].max() >= test_keys[keys[1]].min():
        raise ValueError(f"Training context timestamps overlap the {split} timestamps")
    feature_names = sorted(
        c["name"] for c in task["columns"]
        if c["dtype"] in {"float", "category"} and c["name"] != task["target_column"]
    )
    if not feature_names:
        raise ValueError("No model features")
    summary = {
        "task": full_name, "train": train_summary, split: test_summary,
        "local_positive_class": positive, "features": feature_names,
        "cached_target": task["target_column"], "official_target": official["target_col"],
        "metadata_sha256": sha256(metadata_path),
        "train_sha256": sha256(train_path), f"{split}_sha256": sha256(test_path),
    }
    return summary, (metadata, task, official, eligible, test_positions, test_keys, train_path, test_path)


def feature_matrix(path: Path, names: list[str], positions: np.ndarray) -> np.ndarray:
    with np.load(path, allow_pickle=True) as arrays:
        return np.column_stack([arrays[name][positions] for name in names]).astype(np.float32)


def predict_task(args, summary, prepared, model, model_config, device):
    import torch
    from .eval_utils import _stable_random_state, fill_nans
    from .models import build_classifier

    started = time.monotonic()
    metadata, task, official, eligible, test_positions, test_keys, train_path, test_path = prepared
    random_state = _stable_random_state(f"{metadata['dataset_name']}:{task['name']}:{args.seed}")
    if len(eligible) > args.context_size:
        selection = np.random.default_rng(random_state).choice(len(eligible), args.context_size, replace=False)
        context_positions = eligible[selection]
    else:
        context_positions = eligible
    names = summary["features"]
    X_train = feature_matrix(train_path, names, context_positions)
    X_test = feature_matrix(test_path, names, test_positions)
    with np.load(train_path, allow_pickle=True) as arrays:
        y_train = np.asarray(arrays[task["target_column"]][context_positions])
    if set(np.unique(y_train)) != {0, 1}:
        raise ValueError("Sampled context does not contain both binary classes")
    X_train, X_test = fill_nans(X_train, X_test)
    if not np.isfinite(X_train).all() or not np.isfinite(X_test).all():
        raise ValueError("Non-finite features remain after the existing NaN treatment")
    classifier = build_classifier(model, device, model_config)
    classifier.fit(X_train, y_train)
    predictions = np.empty(len(X_test), dtype=np.float32)
    chunk_size = args.chunk_size
    start = 0
    last_log = time.monotonic()
    while start < len(X_test):
        end = min(start + chunk_size, len(X_test))
        try:
            probability = classifier.predict_proba(X_test[start:end])
        except torch.cuda.OutOfMemoryError:
            if chunk_size == 1:
                raise
            chunk_size = max(1, chunk_size // 2)
            torch.cuda.empty_cache()
            print(f"{summary['task']}: reducing chunk size to {chunk_size}", flush=True)
            continue
        predictions[start:end] = probability[:, summary["local_positive_class"]]
        start = end
        if time.monotonic() - last_log >= 20 or start == len(X_test):
            print(f"{summary['task']}: {start}/{len(X_test)}", flush=True)
            last_log = time.monotonic()
    if not np.isfinite(predictions).all() or ((predictions < 0) | (predictions > 1)).any():
        raise ValueError("Invalid output probabilities")
    result = test_keys.copy()
    result[official["target_col"]] = predictions
    output = args.output_root / "preds" / (summary["task"].replace("/", "__") + ".csv")
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_suffix(".tmp")
    result.to_csv(temporary, index=False)
    temporary.replace(output)
    summary.update({
        "prediction_file": str(output), "prediction_sha256": sha256(output),
        "context_rows": len(context_positions), "context_indices": context_positions.tolist(),
        "actual_chunk_size": chunk_size,
        "prediction_seconds": time.monotonic() - started,
    })


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, default=BASE / "rdb_datasets")
    parser.add_argument("--output-root", type=Path, default=BASE / "outputs/relbench_submission_seed0")
    parser.add_argument("--checkpoint", type=Path, default=BASE / "checkpoints/RDBPFN/model_eval00528.pt")
    parser.add_argument("--model-config", type=Path, default=BASE / "conf_eval/model/RDBPFN.yaml")
    parser.add_argument("--tasks", nargs="+", choices=LEADERBOARD_TASKS["classification"], default=LEADERBOARD_TASKS["classification"])
    parser.add_argument("--revision", help="Official HF dataset commit; default resolves current main once")
    parser.add_argument("--context-size", type=int, default=1024)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--chunk-size", type=int, default=2000)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--audit-only", action="store_true")
    args = parser.parse_args()
    if args.context_size < 2 or args.chunk_size < 1:
        parser.error("context-size must be >= 2 and chunk-size must be >= 1")
    args.output_root = args.output_root.resolve()
    if (args.output_root / "run.json").exists():
        parser.error("Output already has a run.json; choose a new output-root")
    if list((args.output_root / "preds").glob("*")):
        parser.error("Prediction directory is not empty; choose a new output-root")
    revision = HfApi().dataset_info(RELBENCH_HF, revision=args.revision or "main").sha
    report = {
        "official_repo": RELBENCH_HF, "official_revision": revision,
        "settings": {k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()},
        "checkpoint_sha256": sha256(args.checkpoint),
        "model_config_sha256": sha256(args.model_config),
        "prediction_script_sha256": sha256(Path(__file__)),
        "status": "auditing", "tasks": [],
        "submission_readiness": "Not established: cached feature provenance and checkpoint selection require review. No uploads are performed.",
    }
    report_path = args.output_root / "run.json"
    write_json(report_path, report)
    prepared_tasks = []
    for full_name in args.tasks:
        print(f"Checking {full_name}", flush=True)
        try:
            summary, prepared = inspect_task(args.data_root, full_name, revision)
            report["tasks"].append(summary)
            prepared_tasks.append(prepared)
            print(json.dumps({k: summary[k] for k in ("task", "train", "test", "local_positive_class")}), flush=True)
        except Exception as error:
            report["tasks"].append({"task": full_name, "error": str(error)})
            print(f"FAILED {full_name}: {error}", flush=True)
        write_json(report_path, report)
    if any("error" in task for task in report["tasks"]):
        report["status"] = "audit_failed"
        write_json(report_path, report)
        return 1
    report["status"] = "audit_passed"
    write_json(report_path, report)
    if args.audit_only:
        return 0

    import torch
    from .models import ModelConfig, build_model, load_checkpoint

    np.random.seed(42)
    torch.manual_seed(42)
    config = ModelConfig(**yaml.safe_load(args.model_config.read_text())["nanopfn"])
    model = build_model(config)
    load_checkpoint(model, args.checkpoint, "cpu")
    device = torch.device(args.device)
    model.to(device).eval()
    report["status"] = "predicting"
    report["torch_version"] = torch.__version__
    write_json(report_path, report)
    try:
        for summary, prepared in zip(report["tasks"], prepared_tasks):
            predict_task(args, summary, prepared, model, config, device)
            write_json(report_path, report)
    except Exception as error:
        report["status"] = "prediction_failed"
        report["error"] = str(error)
        write_json(report_path, report)
        raise
    report["status"] = "predictions_complete"
    write_json(report_path, report)
    print(f"Predictions complete: {args.output_root / 'preds'}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
