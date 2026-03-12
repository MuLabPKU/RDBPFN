from __future__ import annotations

import logging
import time
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import List
from contextlib import contextmanager
import fcntl

import hydra
import numpy as np
import torch
from hydra.utils import to_absolute_path
from omegaconf import OmegaConf
from sklearn.metrics import accuracy_score, balanced_accuracy_score

from .dbinfer_bench_simplified.dataset_meta import DBBTaskType
from .dbinfer_bench_simplified.rdb_dataset import DBBRDBDataset
from .models import ModelConfig
from .utils import get_default_device, set_randomness_seed
from .eval_utils import (
    load_task_split,
    downsample_split,
    predict_proba_in_chunks,
    _stable_random_state,
    compute_metric,
    derive_output_name,
    save_results_to_csv,
    append_results_to_csv,
    fill_nans,
)
from .eval_classifiers import AutoGluonConfig, LimiXConfig, build_classifier_factory


logger = logging.getLogger(__name__)


DEFAULT_MAX_TEST_SAMPLES = 50000


@dataclass
class DatasetConfig:
    paths: List[str] = field(default_factory=list)
    max_train_samples: int | None = None
    max_test_samples: int | None = DEFAULT_MAX_TEST_SAMPLES
    eval_chunk_size: int | None = None
    seeds: List[int] = field(default_factory=lambda: [0])
    enable_eval_timing: bool = False


@dataclass
class ModelSelectionConfig:
    name: str = "nanopfn"
    checkpoint_path: str | None = None
    nanopfn: ModelConfig = field(default_factory=ModelConfig)
    autogluon: AutoGluonConfig = field(default_factory=AutoGluonConfig)
    limix: LimiXConfig = field(default_factory=LimiXConfig)
    eval_chunk_size_override: int | None = None


@dataclass
class EvaluationConfig:
    model: ModelSelectionConfig = field(default_factory=ModelSelectionConfig)
    dataset: DatasetConfig = field(default_factory=DatasetConfig)
    output_path: str | None = None
    append_output_path: str | None = None


@contextmanager
def _file_lock(lock_path: Path):
    lock_path.parent.mkdir(parents=True, exist_ok=True)
    with open(lock_path, "a+") as lock_file:
        fcntl.flock(lock_file, fcntl.LOCK_EX)
        try:
            yield
        finally:
            fcntl.flock(lock_file, fcntl.LOCK_UN)


def _resolve(path_str: str | None) -> Path | None:
    if not path_str:
        return None
    return Path(to_absolute_path(path_str))


def _collect_checkpoint_paths(path_str: str | None) -> list[Path]:
    checkpoint_path = _resolve(path_str)
    if checkpoint_path is None:
        return []
    if checkpoint_path.is_file():
        return [checkpoint_path]
    if checkpoint_path.is_dir():
        checkpoint_files = sorted(
            p for p in checkpoint_path.glob("*.pt") if p.is_file()
        )
        if not checkpoint_files:
            raise ValueError(f"No .pt checkpoints found under {checkpoint_path}")
        return checkpoint_files
    raise ValueError(
        f"checkpoint_path must be a file or directory, got {checkpoint_path}"
    )


def _evaluate_task(
    dataset_name: str,
    task,
    classifier,
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_test: np.ndarray,
    y_test: np.ndarray,
    chunk_size: int | None,
):
    classifier.fit(X_train, y_train)
    prob = predict_proba_in_chunks(classifier, X_test, chunk_size)
    pred = prob.argmax(axis=1)
    metric = task.metadata.evaluation_metric
    metric_value = compute_metric(metric, y_test, prob, pred)
    return {
        "dataset": dataset_name,
        "task": task.metadata.name,
        "metric": metric,
        "metric_value": metric_value,
        "accuracy": float(accuracy_score(y_test, pred)),
        "balanced_acc": float(balanced_accuracy_score(y_test, pred)),
    }


def _evaluate_datasets(
    cfg: EvaluationConfig,
    classifier_factory,
    eval_chunk_size_override: int | None,
    model_label: str,
) -> tuple[list[dict], list[dict]]:
    per_seed_results: list[dict] = []
    aggregate: dict[tuple[str, str], list[dict]] = defaultdict(list)
    chunk_override = eval_chunk_size_override
    dataset_eval_times: dict[int, float] = defaultdict(float)

    for seed in cfg.dataset.seeds:
        logger.info("Evaluating with seed %s", seed)
        dataset_start_time = time.perf_counter()
        for dataset_path_str in cfg.dataset.paths:
            dataset_path = _resolve(dataset_path_str)
            if dataset_path is None or not dataset_path.exists():
                logger.warning("Skipping dataset %s (path missing)", dataset_path_str)
                continue
            dataset = DBBRDBDataset(dataset_path)
            logger.info("Dataset %s", dataset.dataset_name)
            for task in dataset.tasks:
                if task.metadata.task_type != DBBTaskType.classification:
                    logger.info(
                        "Skipping task %s (%s)",
                        task.metadata.name,
                        task.metadata.task_type,
                    )
                    continue
                try:
                    X_train, y_train = load_task_split(task, "train")
                    X_test, y_test = load_task_split(task, "test")
                except ValueError as exc:
                    logger.warning(
                        "Failed to load splits for %s/%s: %s",
                        dataset.dataset_name,
                        task.metadata.name,
                        exc,
                    )
                    continue

                # Log original dataset sizes
                n_train_orig, n_cols = X_train.shape
                n_test_orig = X_test.shape[0]
                logger.info(
                    "Task %s/%s: #train=%d, #test=%d, #cols=%d",
                    dataset.dataset_name,
                    task.metadata.name,
                    n_train_orig,
                    n_test_orig,
                    n_cols,
                )

                # Downsample training data
                seed_key = f"{dataset.dataset_name}:{task.metadata.name}:{seed}"
                seed_offset = _stable_random_state(seed_key)
                X_train_ds, y_train_ds = downsample_split(
                    X_train, y_train, cfg.dataset.max_train_samples, seed_offset
                )

                # Downsample test data if it exceeds the threshold
                test_seed_key = (
                    f"{dataset.dataset_name}:{task.metadata.name}:{seed}:test"
                )
                test_seed_offset = _stable_random_state(test_seed_key)
                max_test_samples = OmegaConf.select(
                    cfg, "dataset.max_test_samples", default=DEFAULT_MAX_TEST_SAMPLES
                )
                X_test_ds, y_test_ds = downsample_split(
                    X_test, y_test, max_test_samples, test_seed_offset
                )

                # Fill NaNs using train data statistics
                X_train_ds, X_test_ds = fill_nans(X_train_ds, X_test_ds)

                unique_labels = np.unique(y_train_ds)
                if unique_labels.size < 2:
                    logger.warning(
                        "Skipping task %s/%s: train split has a single class after downsampling",
                        dataset.dataset_name,
                        task.metadata.name,
                    )
                    continue

                classifier = classifier_factory()
                task_result = _evaluate_task(
                    dataset.dataset_name,
                    task,
                    classifier,
                    X_train_ds,
                    y_train_ds,
                    X_test_ds,
                    y_test_ds,
                    (
                        None
                        if (chunk_override is not None and chunk_override <= 0)
                        else (
                            chunk_override
                            if chunk_override is not None
                            else cfg.dataset.eval_chunk_size
                        )
                    ),
                )
                task_result["seed"] = seed
                per_seed_results.append(task_result)
                aggregate[(task_result["dataset"], task_result["task"])].append(
                    task_result
                )
                logger.info(
                    "seed %s | %s / %s -> %s %.4f (acc %.4f)",
                    seed,
                    dataset.dataset_name,
                    task.metadata.name,
                    task_result["metric"],
                    task_result["metric_value"],
                    task_result["accuracy"],
                )
        dataset_elapsed = time.perf_counter() - dataset_start_time
        dataset_eval_times[seed] = dataset_elapsed

    aggregated_results: list[dict] = []
    for (dataset_name, task_name), entries in aggregate.items():
        metric_name = entries[0]["metric"] if entries else "unknown"
        metric_value = float(np.nanmean([e["metric_value"] for e in entries]))
        acc = float(np.nanmean([e["accuracy"] for e in entries]))
        bal_acc = float(np.nanmean([e["balanced_acc"] for e in entries]))
        aggregated_results.append(
            {
                "dataset": dataset_name,
                "task": task_name,
                "metric": metric_name,
                "metric_value": metric_value,
                "accuracy": acc,
                "balanced_acc": bal_acc,
                "num_runs": len(entries),
            }
        )
        logger.info(
            "AVG | %s / %s -> %s %.4f (acc %.4f over %d runs)",
            dataset_name,
            task_name,
            metric_name,
            metric_value,
            acc,
            len(entries),
        )

    enable_eval_timing = OmegaConf.select(
        cfg, "dataset.enable_eval_timing", default=False
    )
    if enable_eval_timing:
        _save_dataset_eval_times(dataset_eval_times, model_label)
    return per_seed_results, aggregated_results


def _save_dataset_eval_times(
    dataset_eval_times: dict[str, float], model_label: str
) -> None:
    if not dataset_eval_times:
        return
    timing_dir = Path(to_absolute_path("results/timing"))
    timing_dir.mkdir(parents=True, exist_ok=True)
    sanitized_label = model_label.replace("/", "_").replace("\\", "_").replace(" ", "_")
    filename = f"average__{sanitized_label}.time.txt"
    output_path = timing_dir / filename
    with open(output_path, "w", encoding="utf-8") as handle:
        handle.write(f"{np.mean(list(dataset_eval_times.values())):.6f}\n")


@hydra.main(config_path="../conf_eval", config_name="eval", version_base=None)
def main(cfg: EvaluationConfig):
    if not cfg.dataset.paths:
        raise ValueError("No dataset paths provided for evaluation.")

    set_randomness_seed(42)
    device = torch.device(get_default_device())
    model_name = cfg.model.name.lower()
    checkpoint_targets: list[Path | None]
    if model_name == "nanopfn":
        checkpoint_files = _collect_checkpoint_paths(cfg.model.checkpoint_path)
        if not checkpoint_files:
            raise ValueError("checkpoint_path required for NanoTabPFN evaluation")
        checkpoint_targets = checkpoint_files
    else:
        checkpoint_targets = [None]

    multiple_checkpoints = len(checkpoint_targets) > 1
    best_checkpoint_label: str | None = None
    best_checkpoint_metric = float("-inf")
    best_results: dict | None = None

    # Collect results from all models/checkpoints for CSV output
    all_model_results: dict[str, list[dict]] = {}

    for checkpoint_path in checkpoint_targets:
        label = str(checkpoint_path) if checkpoint_path else cfg.model.name
        if checkpoint_path:
            logger.info("Evaluating checkpoint %s", checkpoint_path)
        if cfg.model.name == "nanopfn":
            classifier_factory = build_classifier_factory(
                cfg.model.name,
                model_config=cfg.model.nanopfn,
                device=device,
                checkpoint_path=checkpoint_path,
            )
        else:
            classifier_factory = build_classifier_factory(
                cfg.model.name,
                autogluon_config=cfg.model.autogluon,
                tabpfn_config=cfg.model.tabpfn,
                tabicl_config=cfg.model.tabicl,
                limix_config=cfg.model.limix,
                device=device,
            )
        eval_chunk_override = OmegaConf.select(
            cfg, "model.eval_chunk_size_override", default=None
        )
        per_seed_results, aggregated_results = _evaluate_datasets(
            cfg, classifier_factory, eval_chunk_override, label
        )

        # Store results for CSV output
        all_model_results[label] = aggregated_results

        if aggregated_results:
            mean_metric = float(
                np.nanmean([result["metric_value"] for result in aggregated_results])
            )
            logger.info(
                "Average primary metric across tasks [%s]: %.4f",
                label,
                mean_metric,
            )
            if multiple_checkpoints and mean_metric > best_checkpoint_metric:
                best_checkpoint_metric = mean_metric
                best_checkpoint_label = label
                best_results = aggregated_results

    if multiple_checkpoints and best_checkpoint_label is not None:
        if best_results is None:
            raise ValueError("best_results is None")
        logger.info(
            "Best checkpoint overall: %s (average metric %.4f)",
            best_checkpoint_label,
            best_checkpoint_metric,
        )
        for result in best_results:
            logger.info(
                "AVG | %s / %s -> %s %.4f (acc %.4f over %d runs)",
                result["dataset"],
                result["task"],
                result["metric"],
                result["metric_value"],
                result["accuracy"],
                result["num_runs"],
            )

    # Save or append results to CSV
    ckpt_path_str = OmegaConf.select(cfg, "model.checkpoint_path", default=None)
    if not ckpt_path_str:
        ckpt_path_str = cfg.model.name
    default_output = f"results/{derive_output_name(ckpt_path_str)}.csv"
    append_output = OmegaConf.select(cfg, "append_output_path", default=None)
    output_path = Path(
        append_output
        if append_output
        else OmegaConf.select(cfg, "output_path", default=default_output)
    )
    print(f"Saving results to {output_path}")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    if append_output:
        lock_path = output_path.with_suffix(output_path.suffix + ".lock")
        with _file_lock(lock_path):
            append_results_to_csv(all_model_results, output_path)
    else:
        save_results_to_csv(all_model_results, output_path)


if __name__ == "__main__":
    main()
