from __future__ import annotations

import time
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Sequence

import torch
from accelerate import Accelerator
from torch import nn
from torch.utils.data import DataLoader

from .models import NanoTabPFNClassifier, NanoTabPFNModel


logger = logging.getLogger(__name__)


class ColumnSubsetCache:
    def __init__(self, reuse_limit: int = 100):
        self._permutations: dict[int, list[tuple[torch.Tensor, int]]] = {}
        self._reuse_limit = max(1, reuse_limit)

    def reset(self):
        self._permutations.clear()

    def get_indices(
        self,
        available_cols: int,
        device: torch.device,
        required_cols: int | None = None,
    ) -> torch.Tensor:
        available_cols = int(max(0, available_cols))
        if available_cols <= 0:
            return torch.empty(0, dtype=torch.long, device=device)
        required = (
            available_cols
            if required_cols is None
            else min(required_cols, available_cols)
        )
        entries = self._permutations.setdefault(available_cols, [])
        if entries:
            perm, count = entries[0]
            if count < self._reuse_limit and perm.numel() >= required:
                entries[0] = (perm, count + 1)
                return perm[:required].to(device)
        new_perm = torch.randperm(available_cols, device=torch.device("cpu"))
        entries.insert(0, (new_perm, 1))
        if len(entries) > self._reuse_limit:
            entries.pop()
        return new_perm[:required].to(device)


_SUPPORTED_TRANSFORM_KINDS = {
    "abs",
    "cos",
    "gaussian",
    "interaction",
    "linear",
    "log",
    "reciprocal",
    "sin",
    "sqrt",
    "square",
    "tanh",
}
_DEFAULT_TRANSFORM_KINDS = tuple(sorted(_SUPPORTED_TRANSFORM_KINDS))
_TRANSFORM_MAX_ABS = 1e6


@dataclass
class ColumnModificationConfig:
    noise_columns_range: tuple[int, int] = (0, 0)
    noise_high_cardinality_prob: float = 0.5
    noise_low_cardinality_max: int = 5
    noise_scale: float = 1.0
    transform_columns_range: tuple[int, int] = (0, 0)
    transform_scale_range: tuple[float, float] = (0.5, 2.0)
    transform_shift_range: tuple[float, float] = (-0.5, 0.5)
    transform_noise_scale: tuple[float, float] = (0.0, 0.0)


def _normalize_int_range(
    value: int | Sequence[int] | None,
    default: tuple[int, int] = (0, 0),
) -> tuple[int, int]:
    if value is None:
        low, high = default
    elif isinstance(value, int):
        low = high = value
    else:
        try:
            if len(value) == 0:
                low, high = default
            elif len(value) == 1:
                low = high = int(value[0])
            else:
                low, high = int(value[0]), int(value[1])
        except TypeError:
            low, high = default
    if high < low:
        low, high = high, low
    return max(0, int(low)), max(0, int(high))


def _sample_int_from_range(
    value: int | Sequence[int] | None, device: torch.device
) -> int:
    low, high = _normalize_int_range(value)
    if high <= low:
        return int(low)
    return int(torch.randint(low, high + 1, (1,), device=device).item())


def _sample_float_from_range(
    value: float | Sequence[float] | None,
    device: torch.device,
    dtype: torch.dtype,
) -> torch.Tensor:
    low, high = float(value[0]), float(value[1])
    if high <= low:
        return torch.tensor(low, device=device, dtype=dtype)
    return (high - low) * torch.rand((), device=device, dtype=dtype) + low


def _sample_feature_indices(
    usable_counts: torch.Tensor, num_columns: int, device: torch.device
) -> torch.Tensor:
    if num_columns <= 0:
        return torch.empty((usable_counts.shape[0], 0), device=device, dtype=torch.long)
    usable_counts = usable_counts.to(device=device, dtype=torch.long).clamp(min=1)
    rand = torch.rand(usable_counts.shape[0], num_columns, device=device)
    indices = torch.floor(rand * usable_counts.unsqueeze(1).to(torch.float32)).to(
        torch.long
    )
    indices = torch.minimum(indices, usable_counts.unsqueeze(1) - 1)
    return indices


def _generate_noise_columns(
    x: torch.Tensor,
    num_noise: int,
    high_cardinality_prob: float,
    low_cardinality_max: int,
    noise_scale: float,
) -> torch.Tensor | None:
    if num_noise <= 0:
        return None
    batch_size, num_rows, _ = x.shape
    device = x.device
    dtype = x.dtype
    high_prob = float(max(0.0, min(1.0, high_cardinality_prob)))
    low_cardinality_max = max(1, int(low_cardinality_max))
    noise_scale = float(abs(noise_scale))

    high_mask = torch.rand(num_noise, device=device) < high_prob
    num_high = int(high_mask.sum().item())
    num_low = int(num_noise - num_high)
    noise_chunks: list[torch.Tensor] = []

    if num_high > 0:
        high_noise = torch.randn(
            batch_size, num_rows, num_high, device=device, dtype=dtype
        )
        if noise_scale != 1.0:
            high_noise = high_noise * noise_scale
        noise_chunks.append(high_noise)

    if num_low > 0:
        low_noise = torch.empty(
            batch_size, num_rows, num_low, device=device, dtype=dtype
        )
        for idx in range(num_low):
            num_values = int(
                torch.randint(1, low_cardinality_max + 1, (1,), device=device).item()
            )
            values = torch.randn(num_values, device=device, dtype=dtype)
            if noise_scale != 1.0:
                values = values * noise_scale
            choices = torch.randint(
                0, num_values, (batch_size, num_rows), device=device
            )
            low_noise[:, :, idx] = values[choices]
        noise_chunks.append(low_noise)

    if not noise_chunks:
        return None
    noise = torch.cat(noise_chunks, dim=2)
    if num_noise > 1:
        perm = torch.randperm(num_noise, device=device)
        noise = noise[:, :, perm]
    return noise


def _generate_transformed_columns(
    source_x: torch.Tensor,
    valid_feature_counts: torch.Tensor | None,
    num_transforms: int,
    transform_scale_range: float | Sequence[float] | None,
    transform_shift_range: float | Sequence[float] | None,
    transform_noise_scale: float | Sequence[float] | None,
) -> torch.Tensor | None:
    transform_kinds = _DEFAULT_TRANSFORM_KINDS
    if num_transforms <= 0 or not transform_kinds:
        return None
    batch_size, num_rows, num_features = source_x.shape
    if num_features <= 0:
        return None
    device = source_x.device
    dtype = source_x.dtype
    if valid_feature_counts is None:
        usable_counts = torch.full(
            (batch_size,), num_features, dtype=torch.long, device=device
        )
    else:
        usable_counts = valid_feature_counts.to(device=device, dtype=torch.long)
        usable_counts = torch.minimum(
            usable_counts, torch.full_like(usable_counts, num_features)
        )
    if usable_counts.max().item() <= 0:
        return None

    base_indices = _sample_feature_indices(usable_counts, num_transforms, device)
    kind_indices = torch.randint(
        0, len(transform_kinds), (num_transforms,), device=device
    )
    transformed_cols: list[torch.Tensor] = []

    for col in range(num_transforms):
        idx = base_indices[:, col]
        base_vals = torch.gather(
            source_x,
            2,
            idx.view(batch_size, 1, 1).expand(-1, num_rows, 1),
        ).squeeze(2)
        base_vals = torch.nan_to_num(base_vals, nan=0.0)
        kind = transform_kinds[int(kind_indices[col].item())]

        if kind == "linear":
            scale = _sample_float_from_range(transform_scale_range, device, dtype)
            shift = _sample_float_from_range(transform_shift_range, device, dtype)
            out = base_vals * scale + shift
        elif kind == "abs":
            out = torch.abs(base_vals)
        elif kind == "square":
            out = base_vals * base_vals
        elif kind == "gaussian":
            out = torch.exp(-base_vals * base_vals)
        elif kind == "log":
            out = torch.sign(base_vals) * torch.log1p(torch.abs(base_vals))
        elif kind == "reciprocal":
            out = torch.sign(base_vals) / (torch.abs(base_vals) + 1e-6)
        elif kind == "sqrt":
            out = torch.sign(base_vals) * torch.sqrt(torch.abs(base_vals) + 1e-6)
        elif kind == "sin":
            out = torch.sin(base_vals)
        elif kind == "cos":
            out = torch.cos(base_vals)
        elif kind == "tanh":
            out = torch.tanh(base_vals)
        elif kind == "interaction":
            other_idx = _sample_feature_indices(usable_counts, 1, device).squeeze(1)
            other_vals = torch.gather(
                source_x,
                2,
                other_idx.view(batch_size, 1, 1).expand(-1, num_rows, 1),
            ).squeeze(2)
            other_vals = torch.nan_to_num(other_vals, nan=0.0)
            if torch.rand((), device=device) < 0.5:
                out = base_vals * other_vals
            else:
                out = base_vals + other_vals
        else:
            out = base_vals

        noise_scale = _sample_float_from_range(transform_noise_scale, device, dtype)
        if noise_scale.item() != 0.0:
            out = out + torch.randn_like(out) * noise_scale * out.abs()

        if not torch.isfinite(out).all() or out.abs().max() > _TRANSFORM_MAX_ABS:
            continue

        out = torch.nan_to_num(out, nan=0.0, posinf=0.0, neginf=0.0)
        transformed_cols.append(out.unsqueeze(2))

    if not transformed_cols:
        return None
    return torch.cat(transformed_cols, dim=2)


def _augment_feature_columns(
    x_batch: torch.Tensor,
    category_mask: torch.Tensor | None,
    valid_feature_counts: torch.Tensor,
    column_modify_config: ColumnModificationConfig,
) -> tuple[torch.Tensor, torch.Tensor | None]:
    device = x_batch.device
    num_noise = _sample_int_from_range(column_modify_config.noise_columns_range, device)
    num_transforms = _sample_int_from_range(
        column_modify_config.transform_columns_range, device
    )

    source_x = x_batch
    augmented_x = x_batch

    if num_transforms > 0:
        transformed = _generate_transformed_columns(
            source_x,
            valid_feature_counts,
            num_transforms,
            column_modify_config.transform_scale_range,
            column_modify_config.transform_shift_range,
            column_modify_config.transform_noise_scale,
        )
        if transformed is not None and transformed.numel() > 0:
            augmented_x = torch.cat([augmented_x, transformed], dim=2)
            if category_mask is not None:
                zeros = torch.zeros(
                    category_mask.shape[0],
                    transformed.shape[2],
                    device=category_mask.device,
                    dtype=category_mask.dtype,
                )
                category_mask = torch.cat([category_mask, zeros], dim=1)

    if num_noise > 0:
        noise = _generate_noise_columns(
            augmented_x,
            num_noise,
            column_modify_config.noise_high_cardinality_prob,
            column_modify_config.noise_low_cardinality_max,
            column_modify_config.noise_scale,
        )
        if noise is not None and noise.numel() > 0:
            augmented_x = torch.cat([augmented_x, noise], dim=2)
            if category_mask is not None:
                zeros = torch.zeros(
                    category_mask.shape[0],
                    noise.shape[2],
                    device=category_mask.device,
                    dtype=category_mask.dtype,
                )
                category_mask = torch.cat([category_mask, zeros], dim=1)

    return augmented_x, category_mask


def _augment_batch_with_random_targets(
    x: torch.Tensor,
    y: torch.Tensor,
    category_mask: torch.Tensor | None,
    split_ratio_range: Sequence[float],
    valid_feature_counts: torch.Tensor | None = None,
    split_index_override: int | None = None,
    num_targets: int = 1,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor | None, int]:
    device = x.device
    base_batch, num_rows, num_features = x.shape
    repeats = max(1, num_targets)

    def _expand(tensor, dims: int):
        if repeats == 1:
            return tensor
        shape = (repeats,) + (-1,) * dims
        expanded = tensor.unsqueeze(0).expand(*shape)
        flat_shape = (repeats * tensor.shape[0],) + tensor.shape[1:]
        return expanded.reshape(flat_shape)

    x = _expand(x, 3)
    y = _expand(y, 2)
    if category_mask is not None:
        category_mask = _expand(category_mask, 2)
    batch_size = x.shape[0]

    if valid_feature_counts is None:
        usable_counts = torch.full(
            (base_batch,), num_features, dtype=torch.long, device=device
        )
    else:
        usable_counts = valid_feature_counts.to(device=device, dtype=torch.long).clamp(
            min=0
        )
    usable_counts = _expand(usable_counts, 1).reshape(batch_size)

    safe_counts = torch.clamp(usable_counts, min=1)
    rand = torch.rand(batch_size, device=device)
    target_col_idx = torch.floor(rand * safe_counts.to(torch.float32)).to(torch.long)
    target_col_idx = torch.minimum(target_col_idx, safe_counts - 1)
    column_indices = target_col_idx.view(batch_size, 1, 1).expand(-1, num_rows, 1)
    column_values = torch.gather(x, 2, column_indices).squeeze(2)
    numeric_values = torch.nan_to_num(column_values, nan=0.0)
    col_min, col_max = torch.aminmax(numeric_values, dim=1)
    diff = col_max - col_min
    rand_offsets = torch.rand(batch_size, device=device)
    thresholds = torch.where(diff > 1e-6, col_min + rand_offsets * diff, col_min)
    binary_target = (numeric_values > thresholds.unsqueeze(1)).to(dtype=y.dtype)

    if num_features > 1:
        base_indices = torch.arange(num_features - 1, device=device)
        kept_indices = base_indices.unsqueeze(0).expand(batch_size, -1)
        offsets = target_col_idx.unsqueeze(1)
        kept_indices = kept_indices + (kept_indices >= offsets).to(kept_indices.dtype)
        kept_indices = kept_indices.to(torch.long)
        gather_indices = kept_indices.unsqueeze(1).expand(batch_size, num_rows, -1)
        features_without_target = torch.gather(x, 2, gather_indices)
    else:
        features_without_target = x.new_zeros((batch_size, num_rows, 0))

    y_as_feature = y.unsqueeze(-1).to(x.dtype)
    features_with_y = torch.cat([features_without_target, y_as_feature], dim=2)

    col_noise = torch.rand(batch_size, num_features, device=device)
    col_perm = col_noise.argsort(dim=1)
    col_perm_expanded = col_perm.unsqueeze(1).expand(batch_size, num_rows, num_features)
    shuffled_features = torch.gather(features_with_y, 2, col_perm_expanded)

    row_noise = torch.rand(batch_size, num_rows, device=device)
    row_perm = row_noise.argsort(dim=1)
    row_perm_expanded = row_perm.unsqueeze(-1).expand(
        batch_size, num_rows, num_features
    )
    augmented_x = torch.gather(shuffled_features, 1, row_perm_expanded)
    augmented_y = torch.gather(binary_target, 1, row_perm)

    augmented_mask = None
    if category_mask is not None:
        if num_features > 1:
            mask_without_target = torch.gather(category_mask, 1, kept_indices)
        else:
            mask_without_target = category_mask.new_zeros((batch_size, 0))
        zeros = torch.zeros(
            batch_size, 1, dtype=category_mask.dtype, device=category_mask.device
        )
        mask_with_y = torch.cat([mask_without_target, zeros], dim=1)
        augmented_mask = torch.gather(mask_with_y, 1, col_perm)

    if split_index_override is not None:
        split_index = split_index_override
    else:
        split_index = _sample_split_index(num_rows, split_ratio_range, device)

    return augmented_x, augmented_y, augmented_mask, int(split_index)


def _sample_column_subsets(
    x_batch: torch.Tensor,
    category_mask: torch.Tensor | None,
    num_features: torch.Tensor,
    num_available: torch.Tensor,
    sampled_columns: int | Sequence[int] | None,
    sampler_cache: ColumnSubsetCache | None,
) -> tuple[torch.Tensor, torch.Tensor | None, torch.Tensor]:
    device = x_batch.device
    num_features = num_features.to(device=device, dtype=torch.long)
    num_available = num_available.to(device=device, dtype=torch.long)
    effective_available = torch.minimum(num_features, num_available).clamp(min=0)

    if sampled_columns is None or sampled_columns <= 0:
        return x_batch, category_mask, effective_available

    batch_size, num_rows, total_features = x_batch.shape
    if total_features == 0:
        valid_counts = torch.zeros(batch_size, dtype=torch.long, device=device)
        return x_batch, category_mask, valid_counts

    candidate_limit = min(sampled_columns, total_features)
    candidate_limit = max(1, candidate_limit)
    if effective_available.max().item() == 0:
        return x_batch.new_zeros((batch_size, num_rows, 0)), None, effective_available

    valid_counts = torch.minimum(
        effective_available, torch.full_like(effective_available, candidate_limit)
    )
    target_cols = int(valid_counts.max().item())
    if target_cols <= 0:
        return x_batch.new_zeros((batch_size, num_rows, 0)), None, valid_counts

    sentinel_index = total_features
    x_ext = torch.cat(
        [
            x_batch,
            torch.zeros(batch_size, num_rows, 1, device=device, dtype=x_batch.dtype),
        ],
        dim=2,
    )
    if category_mask is not None:
        category_mask = torch.cat(
            [
                category_mask,
                torch.zeros(batch_size, 1, device=device, dtype=category_mask.dtype),
            ],
            dim=1,
        )

    if sampler_cache is not None:
        selected = torch.full(
            (batch_size, target_cols), sentinel_index, dtype=torch.long, device=device
        )
        for i in range(batch_size):
            target = int(valid_counts[i].item())
            if target <= 0:
                continue
            available_cols = int(effective_available[i].item())
            perm = sampler_cache.get_indices(available_cols, device, target)
            if perm.numel() < target:
                continue
            selected[i, :target] = perm
    else:
        rand_scores = torch.rand(batch_size, total_features, device=device)
        feature_range = torch.arange(total_features, device=device).unsqueeze(0)
        allowed = feature_range < effective_available.unsqueeze(1)
        rand_scores = rand_scores.masked_fill(~allowed, -float("inf"))
        selected_full = rand_scores.topk(k=candidate_limit, dim=1).indices
        truncated = selected_full[:, :target_cols]
        mask = torch.arange(target_cols, device=device).unsqueeze(
            0
        ) < valid_counts.unsqueeze(1)
        filler = torch.full_like(truncated, sentinel_index)
        selected = torch.where(mask, truncated, filler)

    gather_idx = selected.unsqueeze(1).expand(-1, num_rows, -1)
    subset_x = torch.gather(x_ext, 2, gather_idx)

    subset_mask = None
    if category_mask is not None:
        subset_mask = torch.gather(category_mask, 1, selected)

    return subset_x, subset_mask, valid_counts


def _prepare_batch(
    full_data: dict,
    device: torch.device,
    sampled_columns: int | Sequence[int] | None,
    sampler_cache: ColumnSubsetCache | None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor | None, int, torch.Tensor]:
    x_batch = full_data["x"].to(device)
    y_batch = full_data["y"].to(device)
    # category_mask = full_data.get("category_mask")
    # Currently, we don't use category mask for training since the model does not have additional processings
    category_mask = None
    if category_mask is not None:
        category_mask = category_mask.to(device)
    train_test_split_index = full_data["train_test_split_index"]

    num_features_tensor = full_data.get("num_features")
    if num_features_tensor is None:
        num_features_tensor = torch.full(
            (x_batch.shape[0],),
            x_batch.shape[2],
            dtype=torch.long,
            device=device,
        )
    else:
        num_features_tensor = num_features_tensor.to(device=device, dtype=torch.long)
    num_available_tensor = full_data.get("num_available_features")
    if num_available_tensor is None:
        num_available_tensor = num_features_tensor
    else:
        num_available_tensor = num_available_tensor.to(device=device, dtype=torch.long)

    sampled_columns_value = (
        _sample_int_from_range(sampled_columns, device)
        if sampled_columns is not None
        else None
    )
    (
        x_batch,
        category_mask,
        valid_feature_counts,
    ) = _sample_column_subsets(
        x_batch,
        category_mask,
        num_features_tensor,
        num_available_tensor,
        sampled_columns_value,
        sampler_cache,
    )
    return x_batch, y_batch, category_mask, train_test_split_index, valid_feature_counts


def _handle_evaluation(
    model: NanoTabPFNModel,
    optimizer: torch.optim.Optimizer,
    classifier_factory: (
        Callable[[NanoTabPFNModel, torch.device], NanoTabPFNClassifier] | None
    ),
    device: torch.device,
    eval_func,
    log_callback: Callable[[float, dict[str, float], dict], None] | None,
    eval_history: list[tuple[float, dict[str, float]]],
    total_start_time: float,
    train_time: float,
    eval_time: float,
    global_step: int,
    original_loss_sum: float,
    original_loss_count: int,
    total_loss_sum: float,
    total_loss_count: int,
    checkpoint_path: Path | None,
    best_score: float,
    eval_checkpoint_template: str | None,
    eval_checkpoint_parent: Path | None,
    eval_save_interval: int | None,
    eval_counter: int,
    accelerator: Accelerator | None = None,
) -> tuple[float, float, int, float, int, float, int]:
    if eval_func is None:
        return (
            eval_time,
            best_score,
            eval_counter,
            original_loss_sum,
            original_loss_count,
            total_loss_sum,
            total_loss_count,
        )

    eval_start_time = time.time()
    model.eval()
    optimizer.eval()

    model_to_eval = accelerator.unwrap_model(model) if accelerator else model
    factory = classifier_factory or (lambda m, d: NanoTabPFNClassifier(m, d))
    classifier = factory(model_to_eval, device)
    scores = eval_func(classifier)
    eval_duration = time.time() - eval_start_time
    eval_time += eval_duration
    total_time = time.time() - total_start_time
    avg_loss = original_loss_sum / max(original_loss_count, 1)
    avg_aug_loss = total_loss_sum / max(total_loss_count, 1)
    if accelerator is None or accelerator.is_main_process:
        eval_history.append((total_time, scores))
        if log_callback:
            log_callback(
                total_time,
                {"loss": avg_loss, "augmented_loss": avg_aug_loss},
                scores,
            )
        score_str = " | ".join([f"{k} {v:7.4f}" for k, v in scores.items()])
        logger.info(
            "step %7d | time %7.1fs (train %7.1fs eval %7.1fs) | loss %7.4f | augmented_loss %7.4f | %s",
            global_step,
            total_time,
            train_time,
            eval_time,
            avg_loss,
            avg_aug_loss,
            score_str,
        )
    original_loss_sum = 0.0
    original_loss_count = 0
    total_loss_sum = 0.0
    total_loss_count = 0

    eval_counter += 1
    checkpoint_payload = None
    current_score = scores.get("roc_auc")
    if (
        checkpoint_path
        and (accelerator is None or accelerator.is_main_process)
        and current_score is not None
        and current_score > best_score
    ):
        best_score = current_score
        if accelerator is not None:
            model_state_dict = accelerator.unwrap_model(model).state_dict()
            try:
                optimizer_state_dict = accelerator.get_state_dict(optimizer)
            except Exception:
                optimizer_state_dict = optimizer.state_dict()
        else:
            model_state_dict = model.state_dict()
            optimizer_state_dict = optimizer.state_dict()
        checkpoint_payload = {
            "model_state_dict": model_state_dict,
            "optimizer_state_dict": optimizer_state_dict,
            "step": global_step,
            "train_time": train_time,
            "eval_time": eval_time,
            "scores": scores,
        }
        torch.save(checkpoint_payload, checkpoint_path)
        logger.info(
            "New best ROC-AUC %.4f; saved checkpoint to %s",
            current_score,
            checkpoint_path,
        )
    if (
        eval_checkpoint_template
        and eval_checkpoint_parent
        and eval_save_interval
        and eval_counter % eval_save_interval == 0
        and (accelerator is None or accelerator.is_main_process)
    ):
        if checkpoint_payload is None:
            if accelerator is not None:
                model_state_dict = accelerator.unwrap_model(model).state_dict()
                try:
                    optimizer_state_dict = accelerator.get_state_dict(optimizer)
                except Exception:
                    optimizer_state_dict = optimizer.state_dict()
            else:
                model_state_dict = model.state_dict()
                optimizer_state_dict = optimizer.state_dict()
            checkpoint_payload = {
                "model_state_dict": model_state_dict,
                "optimizer_state_dict": optimizer_state_dict,
                "step": global_step,
                "train_time": train_time,
                "eval_time": eval_time,
                "scores": scores,
            }
        eval_checkpoint_path = eval_checkpoint_parent / eval_checkpoint_template.format(
            eval_counter
        )
        torch.save(checkpoint_payload, eval_checkpoint_path)
        logger.info(
            "Saved eval checkpoint %d to %s",
            eval_counter,
            eval_checkpoint_path,
        )

    model.train()
    optimizer.train()
    return (
        eval_time,
        best_score,
        eval_counter,
        original_loss_sum,
        original_loss_count,
        total_loss_sum,
        total_loss_count,
    )


def _compute_batch_loss(
    model: NanoTabPFNModel,
    criterion: nn.Module,
    x_batch: torch.Tensor,
    y_batch: torch.Tensor,
    category_mask: torch.Tensor | None,
    train_test_split_index: int,
) -> torch.Tensor:
    data = (
        x_batch,
        y_batch[:, :train_test_split_index],
    )
    if getattr(model, "use_category_mask", False):
        if category_mask is None:
            raise ValueError(
                "Model expects category masks but dataloader did not provide them."
            )
        data = data + (category_mask,)
    targets = y_batch

    output = model(data, train_test_split_index=train_test_split_index)
    targets = targets[:, train_test_split_index:]

    targets = targets.reshape((-1,)).to(torch.long)
    output = output.view(-1, output.shape[-1])

    loss = criterion(output, targets).mean()
    return loss


def _sample_split_index(
    num_rows: int, ratio_range: Sequence[float], device: torch.device
) -> int:
    if num_rows <= 1:
        return num_rows
    min_ratio = float(ratio_range[0])
    max_ratio = float(ratio_range[1])
    if max_ratio < min_ratio:
        min_ratio, max_ratio = max_ratio, min_ratio
    min_index = max(1, min(num_rows - 1, int(min_ratio * num_rows)))
    max_index = max(1, min(num_rows - 1, int(max_ratio * num_rows)))
    if max_index < min_index:
        max_index = min_index
    split = torch.randint(min_index, max_index + 1, (1,), device=device)
    return int(split.item())


def train(
    model: NanoTabPFNModel,
    prior,
    optimizer: torch.optim.Optimizer,
    steps_per_epoch: int,
    augment_repeats: int = 0,
    augment_split_ratio_range: Sequence[float] = (0.1, 0.9),
    device: torch.device | None = None,
    steps_per_eval: int = 10,
    eval_func=None,
    checkpoint_path: str | Path | None = None,
    save_every_evals: int | None = None,
    log_callback: Callable[[float, dict[str, float], dict], None] | None = None,
    classifier_factory: (
        Callable[[NanoTabPFNModel, torch.device], NanoTabPFNClassifier] | None
    ) = None,
    sampled_columns: int | Sequence[int] | None = None,
    targets_per_subset: int = 0,
    num_epochs: int = 1,
    group_size_for_augment_data: int = 1,
    global_column_modify_config: ColumnModificationConfig | None = None,
    per_dataset_sampled_columns: list[int | Sequence[int] | None] | None = None,
    per_dataset_targets_per_subset: list[int | None] | None = None,
    per_dataset_group_size: list[int | None] | None = None,
    per_dataset_column_modify_config: (
        list[ColumnModificationConfig | None] | None
    ) = None,
    accelerator: Accelerator | None = None,
):
    device = accelerator.device

    criterion = nn.CrossEntropyLoss()

    model.train()
    optimizer.train()

    train_time = 0.0
    eval_time = 0.0
    eval_history = []
    best_score = float("-inf")
    checkpoint_path = Path(checkpoint_path) if checkpoint_path else None
    eval_save_interval = (
        save_every_evals if save_every_evals and save_every_evals > 0 else None
    )
    eval_checkpoint_template = None
    eval_checkpoint_parent = None
    if checkpoint_path and eval_save_interval:
        eval_checkpoint_template = (
            f"{checkpoint_path.stem}_eval{{:05d}}{checkpoint_path.suffix}"
        )
        eval_checkpoint_parent = checkpoint_path.parent
    total_start_time = time.time()
    original_loss_sum = 0.0
    original_loss_count = 0
    total_loss_sum = 0.0
    total_loss_count = 0
    eval_counter = 0
    if global_column_modify_config is None:
        global_column_modify_config = ColumnModificationConfig()
    sampling_enabled = sampled_columns is not None or (
        per_dataset_sampled_columns is not None
        and any(col is not None for col in per_dataset_sampled_columns)
    )
    column_sampler_cache = ColumnSubsetCache() if sampling_enabled else None
    total_epochs = max(num_epochs, 1)
    steps_per_epoch = max(1, steps_per_epoch)
    global_step = 0
    try:
        for epoch in range(total_epochs):
            if accelerator.is_main_process:
                logger.info("Starting epoch %d / %d", epoch + 1, total_epochs)
            if hasattr(prior, "set_epoch"):
                try:
                    prior.set_epoch(epoch)
                except TypeError:
                    prior.set_epoch()
            epoch_iterator = iter(prior)
            if column_sampler_cache is not None:
                column_sampler_cache.reset()
            for _ in range(steps_per_epoch):
                try:
                    full_data = next(epoch_iterator)
                except StopIteration:
                    epoch_iterator = iter(prior)
                    full_data = next(epoch_iterator)
                step_start_time = time.time()
                source_id = int(full_data.get("source_id", 0))
                batch_sampled_columns = (
                    per_dataset_sampled_columns[source_id]
                    if (
                        per_dataset_sampled_columns is not None
                        and source_id < len(per_dataset_sampled_columns)
                    )
                    else sampled_columns
                )
                dataset_targets = (
                    per_dataset_targets_per_subset[source_id]
                    if (
                        per_dataset_targets_per_subset is not None
                        and source_id < len(per_dataset_targets_per_subset)
                    )
                    else targets_per_subset
                )
                dataset_group_size = (
                    per_dataset_group_size[source_id]
                    if (
                        per_dataset_group_size is not None
                        and source_id < len(per_dataset_group_size)
                    )
                    else group_size_for_augment_data
                )
                dataset_column_modify_config = (
                    per_dataset_column_modify_config[source_id]
                    if (
                        per_dataset_column_modify_config is not None
                        and source_id < len(per_dataset_column_modify_config)
                        and per_dataset_column_modify_config[source_id] is not None
                    )
                    else global_column_modify_config
                )
                dataset_target_repeats = max(dataset_targets or 0, 0)
                total_aug_repeats = max(augment_repeats, 0) + dataset_target_repeats
                (
                    x_batch,
                    y_batch,
                    category_mask,
                    train_test_split_index,
                    valid_feature_counts,
                ) = _prepare_batch(
                    full_data, device, batch_sampled_columns, column_sampler_cache
                )

                with accelerator.accumulate(model):
                    (
                        base_loss_value,
                        aug_loss_sum,
                        aug_loss_count,
                    ) = _compute_losses_with_augmentations(
                        model,
                        criterion,
                        x_batch,
                        y_batch,
                        category_mask,
                        train_test_split_index,
                        total_aug_repeats,
                        augment_split_ratio_range,
                        valid_feature_counts,
                        dataset_group_size,
                        dataset_column_modify_config,
                        accelerator,
                    )
                    original_loss_sum += base_loss_value
                    original_loss_count += 1
                    total_loss_sum += base_loss_value + aug_loss_sum
                    total_loss_count += 1 + aug_loss_count

                    if accelerator.sync_gradients:
                        accelerator.clip_grad_norm_(model.parameters(), 1.0)
                    optimizer.step()
                    optimizer.zero_grad()

                step_train_duration = time.time() - step_start_time
                train_time += step_train_duration

                if accelerator.sync_gradients:
                    global_step += 1

                    if (
                        steps_per_eval > 0
                        and global_step % steps_per_eval == 0
                        and eval_func is not None
                    ):
                        accelerator.wait_for_everyone()
                        if accelerator.is_main_process:
                            (
                                eval_time,
                                best_score,
                                eval_counter,
                                original_loss_sum,
                                original_loss_count,
                                total_loss_sum,
                                total_loss_count,
                            ) = _handle_evaluation(
                                model,
                                optimizer,
                                classifier_factory,
                                device,
                                eval_func,
                                log_callback,
                                eval_history,
                                total_start_time,
                                train_time,
                                eval_time,
                                global_step,
                                original_loss_sum,
                                original_loss_count,
                                total_loss_sum,
                                total_loss_count,
                                checkpoint_path,
                                best_score,
                                eval_checkpoint_template,
                                eval_checkpoint_parent,
                                eval_save_interval,
                                eval_counter,
                                accelerator=accelerator,
                            )
                        accelerator.wait_for_everyone()
                    elif (
                        steps_per_eval > 0
                        and global_step % steps_per_eval == 0
                        and eval_func is None
                    ):
                        total_time = train_time + eval_time
                        avg_loss = original_loss_sum / max(original_loss_count, 1)
                        avg_aug_loss = total_loss_sum / max(total_loss_count, 1)
                        if accelerator.is_main_process:
                            logger.info(
                                "step %7d | time %7.1fs (train %7.1fs) | loss %7.4f | augmented_loss %7.4f",
                                global_step,
                                total_time,
                                train_time,
                                avg_loss,
                                avg_aug_loss,
                            )
                        original_loss_sum = 0.0
                        original_loss_count = 0
                        total_loss_sum = 0.0
                        total_loss_count = 0
    except KeyboardInterrupt:
        pass

    if checkpoint_path and best_score == float("-inf") and accelerator.is_main_process:
        torch.save(
            {"model_state_dict": accelerator.unwrap_model(model).state_dict()},
            checkpoint_path,
        )
        logger.info("Saved final model checkpoint to %s", checkpoint_path)

    return accelerator.unwrap_model(model), eval_history


def _compute_losses_with_augmentations(
    model: NanoTabPFNModel,
    criterion: nn.Module,
    base_x: torch.Tensor,
    base_y: torch.Tensor,
    category_mask: torch.Tensor | None,
    train_test_split_index: int,
    augment_repeats: int,
    augment_split_ratio_range: Sequence[float],
    valid_feature_counts: torch.Tensor,
    group_size_for_augment_data: int,
    column_modify_config: ColumnModificationConfig,
    accelerator: Accelerator,
) -> tuple[float, float, int]:
    # Preserve original feature counts so target sampling ignores added columns.
    base_x, category_mask = _augment_feature_columns(
        base_x,
        category_mask,
        valid_feature_counts,
        column_modify_config,
    )
    original_loss_tensor = _compute_batch_loss(
        model,
        criterion,
        base_x,
        base_y,
        category_mask,
        train_test_split_index,
    )
    accelerator.backward(original_loss_tensor)
    original_loss = original_loss_tensor.detach().item()
    aug_loss_sum = 0.0
    aug_loss_count = 0

    group_size = max(group_size_for_augment_data, 1)
    if augment_repeats <= 0:
        return original_loss, aug_loss_sum, aug_loss_count

    if group_size == 1:
        for _ in range(augment_repeats):
            (
                aug_x,
                aug_y,
                aug_mask,
                aug_split_index,
            ) = _augment_batch_with_random_targets(
                base_x,
                base_y,
                category_mask,
                augment_split_ratio_range,
                valid_feature_counts,
            )
            aug_loss_tensor = _compute_batch_loss(
                model,
                criterion,
                aug_x,
                aug_y,
                aug_mask,
                aug_split_index,
            )
            accelerator.backward(aug_loss_tensor)
            aug_loss = aug_loss_tensor.detach().item()
            aug_loss_sum += aug_loss
            aug_loss_count += 1
        return original_loss, aug_loss_sum, aug_loss_count

    remaining = augment_repeats
    num_rows = base_x.shape[1]
    device = base_x.device
    while remaining > 0:
        current_group = min(group_size, remaining)
        split_override = _sample_split_index(
            num_rows, augment_split_ratio_range, device
        )
        concat_x, concat_y, concat_mask, _ = _augment_batch_with_random_targets(
            base_x,
            base_y,
            category_mask,
            augment_split_ratio_range,
            valid_feature_counts,
            split_index_override=split_override,
            num_targets=current_group,
        )
        group_loss = _compute_batch_loss(
            model,
            criterion,
            concat_x,
            concat_y,
            concat_mask,
            split_override,
        )
        scaled_loss = group_loss * current_group
        accelerator.backward(scaled_loss)
        group_loss_value = scaled_loss.detach().item()
        aug_loss_sum += group_loss_value
        aug_loss_count += current_group
        remaining -= current_group

    return original_loss, aug_loss_sum, aug_loss_count
