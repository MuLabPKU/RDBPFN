from __future__ import annotations

import csv
import logging
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Literal

import hydra
import numpy as np
import torch
from hydra.utils import to_absolute_path
from sklearn.metrics import accuracy_score, balanced_accuracy_score, roc_auc_score
from sklearn.model_selection import train_test_split

from .models import ModelConfig
from .utils import get_default_device, set_randomness_seed
from .eval_utils import (
    downsample_split,
    predict_proba_in_chunks,
    _stable_random_state,
    fill_nans,
    _load_csv_dataset,
    derive_output_name,
    save_results_to_csv,
)
from .eval_classifiers import AutoGluonConfig, LimiXConfig, build_classifier_factory
from omegaconf import OmegaConf


logger = logging.getLogger(__name__)


@dataclass
class CsvDatasetConfig:
    dirs: List[str] = field(default_factory=list)
    mode: Literal["npz", "resample", "summary"] = "npz"
    npz_pattern: str = "*.npz"
    csv_pattern: str = "*.csv"
    seeds: List[int] = field(default_factory=lambda: [0])
    max_train_samples: int | None = None
    max_test_samples: int | None = None
    eval_chunk_size: int | None = None
    test_size: float = 0.3
    output_path: str | None = None


@dataclass
class ModelSelectionConfig:
    name: str = "nanopfn"
    checkpoint_path: str | None = None
    nanopfn: ModelConfig = field(default_factory=ModelConfig)
    autogluon: AutoGluonConfig = field(default_factory=AutoGluonConfig)
    limix: LimiXConfig = field(default_factory=LimiXConfig)
    eval_chunk_size_override: int | None = None


@dataclass
class CsvEvaluationConfig:
    model: ModelSelectionConfig = field(default_factory=ModelSelectionConfig)
    dataset: CsvDatasetConfig = field(default_factory=CsvDatasetConfig)
    output_path: str | None = None


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


def _load_npz_splits(npz_path: Path):
    try:
        data = np.load(npz_path, allow_pickle=True)
    except Exception as exc:  # pylint: disable=broad-except
        logger.warning("Failed to load %s: %s", npz_path, exc)
        return None
    required = {"X_train", "X_test", "y_train", "y_test"}
    if not required.issubset(data.keys()):
        logger.warning("NPZ %s missing required arrays %s", npz_path, required)
        return None
    X_train = data["X_train"].astype(np.float32, copy=False)
    X_test = data["X_test"].astype(np.float32, copy=False)
    y_train = data["y_train"].astype(np.int64, copy=False)
    y_test = data["y_test"].astype(np.int64, copy=False)
    return X_train, y_train, X_test, y_test


def _record_result(
    dataset_name: str,
    task_name: str,
    seed: int,
    classifier_factory,
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_test: np.ndarray,
    y_test: np.ndarray,
    chunk_size: int | None,
    per_seed_results: list[dict],
    aggregate: dict[tuple[str, str], list[dict]],
):
    if np.unique(y_train).size < 2 or np.unique(y_test).size < 2:
        logger.info(
            "Skipping %s/%s seed %s due to insufficient class diversity.",
            dataset_name,
            task_name,
            seed,
        )
        return
    X_train_filled, X_test_filled = fill_nans(X_train, X_test)
    classifier = classifier_factory()
    classifier.fit(X_train_filled, y_train)
    prob = predict_proba_in_chunks(classifier, X_test_filled, chunk_size)
    pred = prob.argmax(axis=1)
    if prob.shape[1] == 2:
        metric_value = float(roc_auc_score(y_test, prob[:, 1]))
    else:
        metric_value = float(roc_auc_score(y_test, prob, multi_class="ovr"))
    result = {
        "dataset": dataset_name,
        "task": task_name,
        "metric": "roc_auc",
        "metric_value": metric_value,
        "accuracy": float(accuracy_score(y_test, pred)),
        "balanced_acc": float(balanced_accuracy_score(y_test, pred)),
        "seed": seed,
    }
    per_seed_results.append(result)
    aggregate[(dataset_name, task_name)].append(result)
    logger.info(
        "seed %s | %s / %s -> roc_auc %.4f (acc %.4f)",
        seed,
        dataset_name,
        task_name,
        metric_value,
        result["accuracy"],
    )


def _evaluate_csv_datasets(
    cfg: CsvDatasetConfig,
    classifier_factory,
    eval_chunk_size_override: int | None,
) -> tuple[list[dict], list[dict]]:
    resolved_dirs = []
    for path_str in cfg.dirs:
        resolved = _resolve(path_str)
        if resolved is None or not resolved.exists():
            logger.warning("CSV directory %s missing; skipping", path_str)
            continue
        resolved_dirs.append(resolved)
    if not resolved_dirs:
        raise ValueError("No valid CSV directories provided.")

    per_seed_results: list[dict] = []
    aggregate: dict[tuple[str, str], list[dict]] = defaultdict(list)

    mode = cfg.mode.lower()
    chunk_override = eval_chunk_size_override
    chunk_size = (
        None
        if (chunk_override is not None and chunk_override <= 0)
        else (
            chunk_override
            if chunk_override is not None
            else getattr(cfg, "eval_chunk_size", None)
        )
    )
    max_train_samples = getattr(cfg, "max_train_samples", None)
    max_test_samples = getattr(cfg, "max_test_samples", None)
    seeds = getattr(cfg, "seeds", [0]) or [0]
    test_size = getattr(cfg, "test_size", 0.3)
    npz_pattern = getattr(cfg, "npz_pattern", "*.npz")
    csv_pattern = getattr(cfg, "csv_pattern", "*.csv")

    if mode == "npz":
        for directory in resolved_dirs:
            for npz_path in sorted(directory.glob(npz_pattern)):
                splits = _load_npz_splits(npz_path)
                if splits is None:
                    continue
                X_train, y_train, X_test, y_test = splits
                _record_result(
                    dataset_name=directory.name,
                    task_name=npz_path.stem,
                    seed=0,
                    classifier_factory=classifier_factory,
                    X_train=X_train,
                    y_train=y_train,
                    X_test=X_test,
                    y_test=y_test,
                    chunk_size=chunk_size,
                    per_seed_results=per_seed_results,
                    aggregate=aggregate,
                )
    elif mode == "resample":
        for seed in seeds:
            logger.info("Evaluating CSV datasets with seed %s", seed)
            for directory in resolved_dirs:
                for csv_path in sorted(directory.glob(csv_pattern)):
                    try:
                        X, y = _load_csv_dataset(csv_path)
                    except Exception as exc:  # pylint: disable=broad-except
                        logger.warning("Failed to load %s: %s", csv_path.stem, exc)
                        continue
                    if np.unique(y).size < 2:
                        logger.info("Skipping %s due to a single class.", csv_path.stem)
                        continue
                    split_seed = _stable_random_state(f"{csv_path.stem}:{seed}:split")
                    stratify = y if np.unique(y).size > 1 else None
                    X_train, X_test, y_train, y_test = train_test_split(
                        X,
                        y,
                        test_size=test_size,
                        random_state=split_seed,
                        stratify=stratify,
                    )
                    train_seed = _stable_random_state(f"{csv_path.stem}:{seed}:train")
                    test_seed = _stable_random_state(f"{csv_path.stem}:{seed}:test")
                    X_train_ds, y_train_ds = downsample_split(
                        X_train, y_train, max_train_samples, train_seed
                    )
                    X_test_ds, y_test_ds = downsample_split(
                        X_test, y_test, max_test_samples, test_seed
                    )
                    _record_result(
                        dataset_name=directory.name,
                        task_name=csv_path.stem,
                        seed=seed,
                        classifier_factory=classifier_factory,
                        X_train=X_train_ds,
                        y_train=y_train_ds,
                        X_test=X_test_ds,
                        y_test=y_test_ds,
                        chunk_size=chunk_size,
                        per_seed_results=per_seed_results,
                        aggregate=aggregate,
                    )
    else:
        raise ValueError(f"Unknown CSV evaluation mode '{cfg.mode}'.")

    aggregated_results: list[dict] = []
    for (dataset_name, task_name), entries in aggregate.items():
        metric_value = float(np.nanmean([e["metric_value"] for e in entries]))
        acc = float(np.nanmean([e["accuracy"] for e in entries]))
        bal_acc = float(np.nanmean([e["balanced_acc"] for e in entries]))
        aggregated_results.append(
            {
                "dataset": dataset_name,
                "task": task_name,
                "metric": "roc_auc",
                "metric_value": metric_value,
                "accuracy": acc,
                "balanced_acc": bal_acc,
                "num_runs": len(entries),
            }
        )
        logger.info(
            "AVG | %s / %s -> roc_auc %.4f (acc %.4f over %d runs)",
            dataset_name,
            task_name,
            metric_value,
            acc,
            len(entries),
        )

    return per_seed_results, aggregated_results


def _summarize_csv_datasets(cfg: CsvDatasetConfig) -> list[dict]:
    resolved_dirs = []
    for path_str in cfg.dirs:
        resolved = _resolve(path_str)
        if resolved is None or not resolved.exists():
            logger.warning("CSV directory %s missing; skipping", path_str)
            continue
        resolved_dirs.append(resolved)
    if not resolved_dirs:
        raise ValueError("No valid CSV directories provided.")

    csv_pattern = getattr(cfg, "csv_pattern", "*.csv")
    summaries: list[dict] = []
    for directory in resolved_dirs:
        for csv_path in sorted(directory.glob(csv_pattern)):
            try:
                X, y = _load_csv_dataset(csv_path)
            except Exception as exc:  # pylint: disable=broad-except
                logger.warning("Failed to load %s: %s", csv_path.stem, exc)
                continue
            summaries.append(
                {
                    "dataset": directory.name,
                    "task": csv_path.stem,
                    "num_samples": int(len(y)),
                    "num_features": int(X.shape[1]),
                }
            )
    if not summaries:
        raise ValueError("No CSV datasets could be summarized.")
    return summaries


def _save_csv_summary(rows: list[dict], output_path="summary.csv") -> None:
    fieldnames = ["dataset", "task", "num_samples", "num_features"]
    with output_path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    logger.info("CSV dataset summary saved to %s", output_path)


@hydra.main(config_path="../conf_eval", config_name="eval_csv", version_base=None)
def main(cfg: CsvEvaluationConfig):
    if not cfg.dataset.dirs:
        raise ValueError("dataset.dirs must list CSV directories to evaluate.")

    if cfg.dataset.mode.lower() == "summary":
        summaries = _summarize_csv_datasets(cfg.dataset)
        output_path_value = getattr(cfg, "output_path", None)
        output_path = (
            Path(output_path_value)
            if output_path_value
            else Path("results/csv_dataset_summary.csv")
        )
        output_path.parent.mkdir(parents=True, exist_ok=True)
        print(f"Saving CSV dataset summary to {output_path}")
        _save_csv_summary(summaries, output_path)
        return

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
    best_results: list[dict] | None = None

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
        _, aggregated_results = _evaluate_csv_datasets(
            cfg.dataset, classifier_factory, eval_chunk_override
        )
        all_model_results[label] = aggregated_results

        if aggregated_results:
            mean_metric = float(
                np.nanmean([result["metric_value"] for result in aggregated_results])
            )
            logger.info(
                "Average roc_auc across CSV tasks [%s]: %.4f",
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
            "Best checkpoint overall: %s (average roc_auc %.4f)",
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

    ckpt_path_str = cfg.model.checkpoint_path or cfg.model.name
    default_output = (
        f"results/{derive_output_name(ckpt_path_str)}_csv_eval.csv"
        if ckpt_path_str
        else "results/csv_eval.csv"
    )
    output_path_value = getattr(cfg, "output_path", None)
    output_path = Path(output_path_value) if output_path_value else Path(default_output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    print(f"Saving CSV evaluation results to {output_path}")
    save_results_to_csv(all_model_results, output_path)


if __name__ == "__main__":
    main()
