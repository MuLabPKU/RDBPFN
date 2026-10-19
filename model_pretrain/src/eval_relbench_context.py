"""Evaluate fixed context ensembles on official validation or test rows.

Defaults to validation; test evaluation requires --split test. Labels are used
only for scoring, never for sampling or inference. Configurations share a training
pool; its first 1024 rows reproduce the seed-0 submission sampler. Within an
ensemble, windows are disjoint and the last window may be shorter. When all
training rows are exhausted, no duplicate windows are added.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import yaml
from huggingface_hub import HfApi, hf_hub_download
from relbench.hf import RELBENCH_HF
from relbench.submit import LEADERBOARD_TASKS
from sklearn.metrics import roc_auc_score

from .eval_utils import _stable_random_state, fill_nans
from .models import ModelConfig, build_classifier, build_model, load_checkpoint
from .predict_relbench import BASE, feature_matrix, inspect_task, sha256, write_json

DEFAULT_CONFIGS = ["1024x1", "2048x1", "4096x1", "1024x4", "4096x4"]


def parse_config(value: str) -> tuple[int, int]:
    try:
        context, members = map(int, value.split("x"))
    except ValueError as error:
        raise ValueError(f"Expected CONTEXTxMEMBERS, got {value!r}") from error
    if context < 2 or members < 1:
        raise ValueError("Context must be >= 2 and members >= 1")
    return context, members


def context_pool(eligible: np.ndarray, size: int, seed: int) -> np.ndarray:
    """Preserve the original 16384-row experiment pool, then extend it.

    A fixed-size anchor is necessary: numpy choice with a different requested
    size does not preserve the old sample prefix. Later blocks are fixed-size
    too, so changing the configuration grid never changes earlier windows.
    """
    if len(eligible) <= 1024:
        return eligible[:size].copy()
    rng = np.random.default_rng(seed)
    first = rng.choice(len(eligible), 1024, replace=False)
    if size <= 1024:
        return eligible[first[:size]]
    remaining = np.ones(len(eligible), dtype=bool)
    remaining[first] = False
    candidates = np.flatnonzero(remaining)
    extra = rng.choice(candidates, min(16384 - 1024, len(candidates)), replace=False)
    selected = [first, extra]
    remaining[extra] = False
    count = len(first) + len(extra)
    while count < min(size, len(eligible)):
        candidates = np.flatnonzero(remaining)
        extra = rng.choice(candidates, min(16384, len(candidates)), replace=False)
        selected.append(extra)
        remaining[extra] = False
        count += len(extra)
    return eligible[np.concatenate(selected)[:size]]


def context_windows(pool_size: int, context: int, members: int) -> list[np.ndarray]:
    return [np.arange(start, min(start + context, pool_size))
            for start in range(0, min(pool_size, context * members), context)]


def predict_ensemble(model, config, device, X_pool, y_pool, X_eval,
                     windows, positive, chunk_size, log_prefix):
    probabilities = np.zeros(len(X_eval), dtype=np.float64)
    member_probabilities = []
    actual_chunks = []
    last_log = time.monotonic()
    for member, indices in enumerate(windows):
        labels = y_pool[indices]
        if set(np.unique(labels)) != {0, 1}:
            raise ValueError(f"{log_prefix}: window {member} lacks one binary class")
        context, evaluation = fill_nans(X_pool[indices], X_eval)
        if not np.isfinite(context).all() or not np.isfinite(evaluation).all():
            raise ValueError("Non-finite features after training-context NaN imputation")
        classifier = build_classifier(model, device, config)
        classifier.fit(context, labels)
        current = np.empty(len(X_eval), dtype=np.float32)
        start = 0
        batch = chunk_size
        while start < len(evaluation):
            end = min(start + batch, len(evaluation))
            try:
                pred = classifier.predict_proba(evaluation[start:end])
            except torch.cuda.OutOfMemoryError:
                if batch == 1:
                    raise
                batch = max(1, batch // 2)
                torch.cuda.empty_cache()
                print(f"{log_prefix}: reducing evaluation batch to {batch}", flush=True)
                continue
            current[start:end] = pred[:, positive]
            start = end
            if time.monotonic() - last_log >= 20:
                print(f"{log_prefix}: member {member+1}/{len(windows)}, {start}/{len(evaluation)}", flush=True)
                last_log = time.monotonic()
        if not np.isfinite(current).all() or ((current < 0) | (current > 1)).any():
            raise ValueError("Invalid probabilities")
        probabilities += current
        member_probabilities.append(current)
        actual_chunks.append(batch)
    if not windows:
        raise ValueError("No context windows")
    return probabilities / len(windows), np.stack(member_probabilities), actual_chunks


def run_task(args, full_name, revision, model, config, device):
    summary, prepared = inspect_task(args.data_root, full_name, revision, split=args.split)
    metadata, task, official, eligible, val_positions, val_keys, train_path, val_path = prepared
    dataset, task_name = full_name.split("/")
    path = hf_hub_download(RELBENCH_HF, f"{dataset}/tasks/{task_name}/{args.split}.parquet",
                           repo_type="dataset", revision=revision)
    official_val = pd.read_parquet(path, columns=list(val_keys.columns) + [official["target_col"]])
    pd.testing.assert_frame_equal(official_val[list(val_keys.columns)], val_keys)
    y_val = official_val[official["target_col"]].to_numpy()
    if set(np.unique(y_val)) != {0, 1}:
        raise ValueError("Evaluation split must contain both classes for AUROC")
    pool_size = max(c * m for c, m in map(parse_config, args.configs))
    sampling_seed = _stable_random_state(f"{metadata['dataset_name']}:{task_name}:{args.seed}")
    pool = context_pool(eligible, pool_size, sampling_seed)
    X_pool = feature_matrix(train_path, summary["features"], pool)
    X_val = feature_matrix(val_path, summary["features"], val_positions)
    with np.load(train_path, allow_pickle=True) as arrays:
        y_pool = arrays[task["target_column"]][pool]
    task_dir = args.output_root / full_name.replace("/", "__")
    task_dir.mkdir(parents=True, exist_ok=True)
    summary["context_pool_indices"] = pool.tolist()
    summary["split"] = args.split
    summary[f"official_{args.split}_sha256"] = sha256(Path(path))
    row_key = "validation_rows" if args.split == "val" else "test_rows"
    summary[row_key] = len(val_keys)
    summary["results"] = []
    write_json(task_dir / "audit.json", summary)
    # Warm up the model before timing each task. No validation labels enter it.
    warmup_indices = context_windows(len(pool), 1024, 1)
    predict_ensemble(model, config, device, X_pool, y_pool, X_val[:1],
                     warmup_indices, summary["local_positive_class"], 1, "warmup")
    results = []
    for setting in args.configs:
        context, members = parse_config(setting)
        windows = context_windows(len(pool), context, members)
        if device.type == "cuda":
            torch.cuda.synchronize(device)
            torch.cuda.empty_cache()
            torch.cuda.reset_peak_memory_stats(device)
        started = time.perf_counter()
        probabilities, individual, chunks = predict_ensemble(
            model, config, device, X_pool, y_pool, X_val, windows,
            summary["local_positive_class"], args.chunk_size, f"{full_name} {setting}",
        )
        if device.type == "cuda":
            torch.cuda.synchronize(device)
        elapsed = time.perf_counter() - started
        record = {
            "task": full_name, "configuration": setting, "split": args.split,
            "context_size": context, "requested_members": members,
            "effective_members": len(windows), "context_rows": [len(w) for w in windows],
            "unique_context_rows": sum(len(w) for w in windows),
            row_key: len(val_keys), "auroc": float(roc_auc_score(y_val, probabilities)),
            "inference_seconds": elapsed,
            "peak_allocated_gib": torch.cuda.max_memory_allocated(device) / 2**30 if device.type == "cuda" else None,
            "peak_reserved_gib": torch.cuda.max_memory_reserved(device) / 2**30 if device.type == "cuda" else None,
            "actual_chunk_sizes": chunks,
        }
        # Keep validation and test artifacts explicitly named and separated.
        frame = val_keys.copy()
        frame[official["target_col"]] = probabilities
        suffix = "validation" if args.split == "val" else "test"
        prediction_path = task_dir / f"{setting}.{suffix}.csv"
        frame.to_csv(prediction_path, index=False)
        np.savez_compressed(task_dir / f"{setting}.members.npz", probabilities=individual)
        record["prediction_sha256"] = sha256(prediction_path)
        summary["results"].append(record)
        write_json(task_dir / "audit.json", summary)
        results.append(record)
        print(f"{full_name} {setting}: AUROC={record['auroc']:.6f}, "
              f"{elapsed:.2f}s, peak={record['peak_allocated_gib']}, windows={record['context_rows']}", flush=True)
    return results


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, default=BASE / "rdb_datasets")
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, default=BASE / "checkpoints/RDBPFN/model_eval00528.pt")
    parser.add_argument("--model-config", type=Path, default=BASE / "conf_eval/model/RDBPFN.yaml")
    parser.add_argument("--tasks", nargs="+", choices=LEADERBOARD_TASKS["classification"], default=LEADERBOARD_TASKS["classification"])
    parser.add_argument("--configs", nargs="+", default=DEFAULT_CONFIGS)
    parser.add_argument("--revision", default="21d7284df2a9621c71f21bc4a1a30b7e8effb8dd")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--split", choices=["val", "test"], default="val")
    parser.add_argument("--chunk-size", type=int, default=2000)
    parser.add_argument("--device", default="cuda:0")
    args = parser.parse_args()
    if args.chunk_size < 1 or len(set(args.configs)) != len(args.configs) or len(set(args.tasks)) != len(args.tasks):
        parser.error("Positive chunk size and unique tasks/configurations are required")
    for setting in args.configs:
        parse_config(setting)
    args.output_root = args.output_root.resolve()
    if args.output_root.exists() and any(args.output_root.iterdir()):
        parser.error("Choose a new, empty output directory")
    revision = HfApi().dataset_info(RELBENCH_HF, revision=args.revision).sha
    torch.manual_seed(42)
    np.random.seed(42)
    # Avoid the native MHA fastpath, which can materialize quadratic attention
    # scores even when weights are not returned. This process uses SDPA instead.
    torch.backends.mha.set_fastpath_enabled(False)
    config = ModelConfig(**yaml.safe_load(args.model_config.read_text())["nanopfn"])
    model = build_model(config)
    load_checkpoint(model, args.checkpoint, "cpu")
    for block in model.transformer_blocks:
        block.memory_efficient_attention = True
    device = torch.device(args.device)
    if device.type == "cuda":
        torch.cuda.set_device(device)
    model.to(device).eval()
    report = {
        "status": "running", "split": args.split, "official_repo": RELBENCH_HF,
        "official_revision": revision, "checkpoint_sha256": sha256(args.checkpoint),
        "model_config_sha256": sha256(args.model_config),
        "code_sha256": {p.name: sha256(p) for p in [Path(__file__), BASE / "src/models.py", BASE / "src/predict_relbench.py"]},
        "settings": {k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()},
        "sampling": "nested uniform disjoint windows; fixed 16384-row anchor preserves prior experiment; equal-weight probability mean; no repeated windows",
        "torch_version": torch.__version__, "dtype": "float32", "results": [],
    }
    report_path = args.output_root / "run.json"
    write_json(report_path, report)
    try:
        for task in args.tasks:
            print(f"Preparing {args.split} data: {task}", flush=True)
            report["results"].extend(run_task(args, task, revision, model, config, device))
            write_json(report_path, report)
            pd.DataFrame(report["results"]).to_csv(args.output_root / "results.csv", index=False)
    except Exception as error:
        report["status"] = "failed"
        report["error"] = str(error)
        write_json(report_path, report)
        raise
    report["status"] = "complete"
    write_json(report_path, report)
    print(pd.DataFrame(report["results"]).groupby("configuration")["auroc"].mean().to_string(), flush=True)


if __name__ == "__main__":
    main()
