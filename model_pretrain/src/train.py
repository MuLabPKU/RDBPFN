from __future__ import annotations
import torch

import logging
import os
from dataclasses import asdict, dataclass, field, is_dataclass
from pathlib import Path
from typing import Literal, Sequence

import hydra
import wandb
import schedulefree
from hydra.utils import to_absolute_path
from omegaconf import OmegaConf
from torch.utils.data import DataLoader

from .eval_utils import evaluate_classifier, prepare_eval_splits, DEFAULT_EVAL_DIRS
from .models import (
    ModelConfig,
    load_checkpoint,
    build_classifier,
    build_model,
)
from .training import ColumnModificationConfig, train
from accelerate import Accelerator
from accelerate.utils import set_seed

from .utils import set_randomness_seed


logger = logging.getLogger(__name__)


@dataclass
class PriorDatasetConfig:
    path: str = ""
    weight: float | None = None
    sampled_columns: int | Sequence[int] | None = None
    targets_per_subset: int | None = None
    group_size_for_augment_data: int | None = None
    column_modification_config: ColumnModificationConfig | None = None


@dataclass
class TrainConfig:
    prior_path: str = "category_large_v2.h5"
    prior_paths: list[str] = field(default_factory=list)
    loader_weights: list[float] | None = None
    datasets: list[PriorDatasetConfig] = field(default_factory=list)
    num_steps: int = 10000
    num_epochs: int = 1
    batch_size: int = 32
    lr: float = 4e-3
    weight_decay: float = 0.0
    steps_per_eval: int = 100
    augment_times: int = 0
    augment_split_ratio_range: tuple[float, float] = (0.1, 0.9)
    load_model_path: str | None = None
    save_model_path: str = "checkpoints/nanoTabPFN.pt"
    save_every_evals: int | None = None
    dataset_start_index: int = 0  # Start index in dataset when resuming training
    load_optimizer_state: bool = (
        False  # Whether to load optimizer state from checkpoint
    )
    sampled_columns: int | Sequence[int] | None = None
    targets_per_subset: int = 0  # Extra target augmentations per subset
    group_size_for_augment_data: int = 1
    global_column_modify_config: ColumnModificationConfig = field(
        default_factory=ColumnModificationConfig
    )
    num_gpus: int | None = None
    gradient_accumulation_steps: int = 1


@dataclass
class EvalConfig:
    csv_dirs: list[str] = field(default_factory=list)


@dataclass
class WandbConfig:
    enabled: bool = False
    project: str = "nanoTabPFN"
    run_name: str | None = None


@dataclass
class Config:
    seed: int = 0
    train: TrainConfig = field(default_factory=TrainConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    eval: EvalConfig = field(default_factory=EvalConfig)
    wandb: WandbConfig = field(default_factory=WandbConfig)


def _resolve_path(path_str: str | None) -> Path | None:
    if path_str in (None, ""):
        return None
    return Path(to_absolute_path(path_str))


def _validate_model_config(model_cfg: ModelConfig):
    if model_cfg.type != "categorical":
        invalid = [
            model_cfg.per_column_embeddings,
            model_cfg.sort_category_embeddings,
            model_cfg.invariant_noise_encoder,
            model_cfg.dual_feature_attention,
            model_cfg.category_as_numeric,
        ]
        if any(invalid):
            raise ValueError(
                "Categorical-specific options require model.type to be 'categorical'."
            )
    if model_cfg.category_as_numeric and (
        model_cfg.per_column_embeddings
        or model_cfg.sort_category_embeddings
        or model_cfg.invariant_noise_encoder
    ):
        raise ValueError(
            "category_as_numeric cannot be combined with categorical embedding options."
        )


def _log_training_schedule(train_cfg: TrainConfig, world_size: int):
    per_device_batch = int(train_cfg.batch_size)
    world_size = max(1, int(world_size))
    grad_accum_steps = max(1, int(getattr(train_cfg, "gradient_accumulation_steps", 1)))
    global_batch = per_device_batch * world_size * grad_accum_steps
    steps_per_eval_cfg = train_cfg.steps_per_eval
    steps_per_eval = int(steps_per_eval_cfg) if steps_per_eval_cfg else 0
    logger.info(
        "Batch sizing -> per-device: %d | processes: %d | grad accum: %d | global: %d",
        per_device_batch,
        world_size,
        grad_accum_steps,
        global_batch,
    )
    if steps_per_eval > 0:
        single_device_equiv = steps_per_eval * world_size
        samples_per_eval = steps_per_eval * global_batch
        logger.info(
            "Evaluation cadence -> every %d optimizer steps (~%d single-device steps) covering ~%d samples",
            steps_per_eval,
            single_device_equiv,
            samples_per_eval,
        )
    else:
        logger.info(
            "Evaluation cadence -> disabled (steps_per_eval=%s)",
            train_cfg.steps_per_eval,
        )
    save_every_evals = train_cfg.save_every_evals
    if save_every_evals and steps_per_eval > 0:
        optimizer_steps_per_save = steps_per_eval * save_every_evals
        single_device_steps = optimizer_steps_per_save * world_size
        samples_per_save = optimizer_steps_per_save * global_batch
        logger.info(
            "Checkpoint cadence -> every %d evals (%d optimizer steps, ~%d single-device steps) covering ~%d samples",
            save_every_evals,
            optimizer_steps_per_save,
            single_device_steps,
            samples_per_save,
        )
    elif save_every_evals:
        logger.info(
            "Checkpoint cadence -> configured every %d evals but evaluations are disabled",
            save_every_evals,
        )


@hydra.main(config_path="../conf_train", config_name="config", version_base=None)
def main(cfg: Config):
    grad_accum_steps = max(1, int(getattr(cfg.train, "gradient_accumulation_steps", 1)))
    accelerator = Accelerator(gradient_accumulation_steps=grad_accum_steps)
    set_seed(cfg.seed)
    _validate_model_config(cfg.model)

    eval_dirs = cfg.eval.csv_dirs
    if eval_dirs:
        eval_dirs = [Path(to_absolute_path(str(path))) for path in eval_dirs]
    else:
        eval_dirs = [Path(to_absolute_path(str(path))) for path in DEFAULT_EVAL_DIRS]
    eval_splits, _ = prepare_eval_splits(data_dirs=eval_dirs)

    def eval_fn(classifier):
        return evaluate_classifier(classifier, eval_splits)

    if cfg.wandb.enabled and accelerator.is_main_process:
        # Force wandb relogin using WANDB_API_KEY from environment
        wandb_api_key = os.environ.get("WANDB_API_KEY")
        if not wandb_api_key:
            raise ValueError(
                "WANDB_API_KEY environment variable must be set when wandb is enabled"
            )
        logger.info("Found WANDB_API_KEY in environment, forcing wandb relogin")
        wandb.login(key=wandb_api_key, relogin=True)

        wandb_config = (
            asdict(cfg)
            if is_dataclass(cfg)
            else OmegaConf.to_container(cfg, resolve=True)
        )
        wandb.init(
            project=cfg.wandb.project,
            name=cfg.wandb.run_name,
            config=wandb_config,
        )
    accelerator.wait_for_everyone()

    device = accelerator.device
    model = build_model(cfg.model)
    optimizer = schedulefree.AdamWScheduleFree(
        model.parameters(), lr=cfg.train.lr, weight_decay=cfg.train.weight_decay
    )

    load_model_path = _resolve_path(cfg.train.load_model_path)
    optimizer_state = None

    if load_model_path:
        optimizer_state = load_checkpoint(
            model, load_model_path, device, output_log=accelerator.is_main_process
        )
        load_optimizer_state = getattr(cfg.train, "load_optimizer_state", False)
        if not load_optimizer_state:
            optimizer_state = None
            if accelerator.is_main_process:
                logger.info(
                    "Skipping optimizer state loading (load_optimizer_state=False)"
                )

    model, optimizer = accelerator.prepare(model, optimizer)

    if optimizer_state is not None:
        optimizer.load_state_dict(optimizer_state)
        if accelerator.is_main_process:
            logger.info("Loaded optimizer state from checkpoint")

    if accelerator.is_main_process:
        configured_gpus = getattr(cfg.train, "num_gpus", None)
        if configured_gpus is not None and configured_gpus != accelerator.num_processes:
            logger.warning(
                "Configured train.num_gpus=%s but accelerator reports %s processes. Check launch args.",
                configured_gpus,
                accelerator.num_processes,
            )
        _log_training_schedule(cfg.train, accelerator.num_processes)

    dataset_entries: list[dict] = []
    for ds_cfg in cfg.train.datasets:
        resolved = _resolve_path(ds_cfg.path)
        if resolved is None:
            raise ValueError("Each dataset entry must specify a valid path.")
        dataset_entries.append(
            {
                "path": resolved,
                "weight": ds_cfg.weight,
                "sampled_columns": ds_cfg.sampled_columns,
                "targets_per_subset": ds_cfg.targets_per_subset,
                "group_size": ds_cfg.group_size_for_augment_data,
                "column_modification_config": getattr(
                    ds_cfg, "column_modification_config", None
                ),
            }
        )

    if not dataset_entries:
        raise ValueError("No dataset entries available for training.")
    dataset_start_index = getattr(cfg.train, "dataset_start_index", 0)
    dataset_sampled_columns = [entry["sampled_columns"] for entry in dataset_entries]
    dataset_targets_per_subset = [
        entry["targets_per_subset"] for entry in dataset_entries
    ]
    dataset_group_sizes = [entry["group_size"] for entry in dataset_entries]
    dataset_column_modify_config = [
        entry["column_modification_config"] for entry in dataset_entries
    ]
    joint_seed = cfg.seed
    world_size = accelerator.num_processes
    shard_id = accelerator.process_index
    weights = [entry["weight"] for entry in dataset_entries]
    if any(weight is not None for weight in weights):
        if not all(weight is not None for weight in weights):
            raise ValueError(
                "When specifying dataset weights, every dataset must provide one."
            )
        loader_weights = [float(w) for w in weights]
    else:
        loader_weights = None

    if accelerator.num_processes > 1:
        # Using InmemoryDataset and JointDataset for multi-process training
        from .inmemory_dataloader import InMemoryDataset, JointDataset, collate_batch

        datasets = [
            InMemoryDataset(
                str(entry["path"]),
                start_index=dataset_start_index,
                output_log=accelerator.is_main_process,
            )
            for entry in dataset_entries
        ]
        joint_dataset = JointDataset(
            datasets=datasets,
            steps_per_epoch=cfg.train.num_steps,
            batch_size=cfg.train.batch_size,
            weights=loader_weights,
            seed=joint_seed,
            shard_id=shard_id,
            num_shards=world_size,
        )
        prior = DataLoader(
            joint_dataset,
            batch_size=cfg.train.batch_size,
            shuffle=False,
            num_workers=0,
            pin_memory=True,
            collate_fn=collate_batch,
        )
        if joint_dataset.steps_per_shard <= 0:
            raise ValueError(
                "JointDataset produced zero steps. Increase train.num_steps or adjust shard count."
            )
        prior.steps_per_epoch = joint_dataset.steps_per_shard
        prior.set_epoch = joint_dataset.set_epoch
    else:
        # Using Plain PriorDumpDataLoader for single-process training
        from .dataloaders import JointPriorLoader, PriorDumpDataLoader

        prior_loaders = [
            PriorDumpDataLoader(
                entry["path"],
                num_steps=None,
                batch_size=cfg.train.batch_size,
                device=device,
                start_index=dataset_start_index,
            )
            for entry in dataset_entries
        ]
        prior = JointPriorLoader(
            prior_loaders,
            steps_per_epoch=cfg.train.num_steps,
            weights=loader_weights,
            seed=cfg.seed,
        )

    checkpoint_path = _resolve_path(cfg.train.save_model_path)
    if checkpoint_path:
        checkpoint_path.parent.mkdir(parents=True, exist_ok=True)

    def log_callback(time_elapsed, losses, metrics):
        if cfg.wandb.enabled and accelerator.is_main_process:
            wandb.log({**losses, **metrics, "time": time_elapsed})

    def classifier_factory(m, d):
        return build_classifier(m, d, cfg.model)

    augment_times = getattr(cfg.train, "augment_times", 0)
    augment_split_range = getattr(cfg.train, "augment_split_ratio_range", (0.1, 0.9))

    model, _ = train(
        model,
        prior,
        optimizer,
        steps_per_epoch=prior.steps_per_epoch,
        steps_per_eval=cfg.train.steps_per_eval,
        augment_repeats=augment_times,
        augment_split_ratio_range=augment_split_range,
        eval_func=eval_fn,
        checkpoint_path=checkpoint_path,
        save_every_evals=cfg.train.save_every_evals,
        device=device,
        log_callback=log_callback,
        classifier_factory=classifier_factory,
        sampled_columns=cfg.train.sampled_columns,
        targets_per_subset=cfg.train.targets_per_subset,
        num_epochs=cfg.train.num_epochs,
        group_size_for_augment_data=cfg.train.group_size_for_augment_data,
        global_column_modify_config=cfg.train.global_column_modify_config,
        per_dataset_sampled_columns=dataset_sampled_columns,
        per_dataset_targets_per_subset=dataset_targets_per_subset,
        per_dataset_group_size=dataset_group_sizes,
        per_dataset_column_modify_config=dataset_column_modify_config,
        accelerator=accelerator,
    )
    if accelerator.is_main_process:
        final_metrics = eval_fn(build_classifier(model, device, cfg.model))
        logger.info("Final evaluation: %s", final_metrics)
        if cfg.wandb.enabled:
            wandb.log({f"final/{k}": v for k, v in final_metrics.items()})
            wandb.finish()


if __name__ == "__main__":
    main()
