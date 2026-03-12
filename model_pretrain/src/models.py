from __future__ import annotations

import argparse
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

import numpy as np
import torch
import torch.nn.functional as F
import yaml
from torch import nn
from torch.nn.modules.transformer import LayerNorm, Linear, MultiheadAttention
from torch.nn.modules.utils import consume_prefix_in_state_dict_if_present

logger = logging.getLogger(__name__)


@dataclass
class ModelConfig:
    type: Literal["base", "categorical"] = "base"
    embedding_size: int = 96
    num_attention_heads: int = 4
    mlp_hidden_size: int = 192
    num_layers: int = 3
    num_outputs: int = 2
    num_category_buckets: int = 10
    per_column_embeddings: bool = False
    sort_category_embeddings: bool = False
    invariant_noise_encoder: bool = False
    dual_feature_attention: bool = False
    category_as_numeric: bool = False


class FeatureEncoder(nn.Module):
    def __init__(self, embedding_size: int):
        super().__init__()
        self.linear_layer = nn.Linear(1, embedding_size)

    def forward(self, x: torch.Tensor, train_test_split_index: int) -> torch.Tensor:
        x = x.unsqueeze(-1)
        train_slice = x[:, :train_test_split_index]
        valid_mask = ~torch.isnan(train_slice)
        valid_count = valid_mask.sum(dim=1, keepdims=True).clamp(min=1)
        train_filled = torch.where(
            valid_mask, train_slice, torch.zeros_like(train_slice)
        )
        mean = train_filled.sum(dim=1, keepdims=True) / valid_count
        diff = torch.where(
            valid_mask, train_slice - mean, torch.zeros_like(train_slice)
        )
        var = (diff**2).sum(dim=1, keepdims=True) / valid_count
        std = torch.sqrt(var + 1e-20)
        x = torch.where(torch.isnan(x), mean, x)
        x = (x - mean) / std
        x = torch.clip(x, min=-100, max=100)
        return self.linear_layer(x)


class TargetEncoder(nn.Module):
    def __init__(self, embedding_size: int):
        super().__init__()
        self.linear_layer = nn.Linear(1, embedding_size)

    def forward(self, y_train: torch.Tensor, num_rows: int) -> torch.Tensor:
        mean = torch.mean(y_train.to(torch.float), dim=1, keepdim=True)
        padding = mean.repeat(1, num_rows - y_train.shape[1], 1)
        y = torch.cat([y_train, padding], dim=1)
        y = y.unsqueeze(-1)
        return self.linear_layer(y)


class TransformerEncoderLayer(nn.Module):
    def __init__(
        self,
        embedding_size: int,
        nhead: int,
        mlp_hidden_size: int,
        layer_norm_eps: float = 1e-5,
        batch_first: bool = True,
        device=None,
        dtype=None,
        dual_feature_attention: bool = False,
    ):
        super().__init__()
        self.self_attention_between_datapoints = MultiheadAttention(
            embedding_size, nhead, batch_first=batch_first, device=device, dtype=dtype
        )
        self.use_dual_feature_attention = dual_feature_attention
        if self.use_dual_feature_attention:
            self.numeric_feature_attention = MultiheadAttention(
                embedding_size,
                nhead,
                batch_first=batch_first,
                device=device,
                dtype=dtype,
            )
            self.categorical_feature_attention = MultiheadAttention(
                embedding_size,
                nhead,
                batch_first=batch_first,
                device=device,
                dtype=dtype,
            )
        else:
            self.self_attention_between_features = MultiheadAttention(
                embedding_size,
                nhead,
                batch_first=batch_first,
                device=device,
                dtype=dtype,
            )
        self.linear1 = Linear(
            embedding_size, mlp_hidden_size, device=device, dtype=dtype
        )
        self.linear2 = Linear(
            mlp_hidden_size, embedding_size, device=device, dtype=dtype
        )
        self.norm1 = LayerNorm(
            embedding_size, eps=layer_norm_eps, device=device, dtype=dtype
        )
        self.norm2 = LayerNorm(
            embedding_size, eps=layer_norm_eps, device=device, dtype=dtype
        )
        self.norm3 = LayerNorm(
            embedding_size, eps=layer_norm_eps, device=device, dtype=dtype
        )

    def forward(
        self,
        src: torch.Tensor,
        train_test_split_index: int,
        category_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        batch_size, rows_size, col_size, embedding_size = src.shape
        src = src.reshape(batch_size * rows_size, col_size, embedding_size)
        if self.use_dual_feature_attention:
            if category_mask is None:
                raise ValueError(
                    "Dual feature attention requires a category mask to be provided."
                )
            mask = category_mask
            if mask.dim() == 2:
                mask = mask.unsqueeze(1).expand(batch_size, rows_size, col_size)
            mask = mask.reshape(batch_size * rows_size, col_size)
            mask = mask.unsqueeze(-1).to(dtype=torch.bool, device=src.device)
            numeric_out = self.numeric_feature_attention(src, src, src)[0]
            categorical_out = self.categorical_feature_attention(src, src, src)[0]
            src = torch.where(mask, categorical_out, numeric_out) + src
        else:
            src = self.self_attention_between_features(src, src, src)[0] + src
        src = src.reshape(batch_size, rows_size, col_size, embedding_size)
        src = self.norm1(src)
        src = src.transpose(1, 2)
        src = src.reshape(batch_size * col_size, rows_size, embedding_size)
        src_left = self.self_attention_between_datapoints(
            src[:, :train_test_split_index],
            src[:, :train_test_split_index],
            src[:, :train_test_split_index],
        )[0]
        src_right = self.self_attention_between_datapoints(
            src[:, train_test_split_index:],
            src[:, :train_test_split_index],
            src[:, :train_test_split_index],
        )[0]
        src = torch.cat([src_left, src_right], dim=1) + src
        src = src.reshape(batch_size, col_size, rows_size, embedding_size)
        src = src.transpose(2, 1)
        src = self.norm2(src)
        src = self.linear2(F.gelu(self.linear1(src))) + src
        src = self.norm3(src)
        return src


class Decoder(nn.Module):
    def __init__(self, embedding_size: int, mlp_hidden_size: int, num_outputs: int):
        super().__init__()
        self.linear1 = nn.Linear(embedding_size, mlp_hidden_size)
        self.linear2 = nn.Linear(mlp_hidden_size, num_outputs)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.linear2(F.gelu(self.linear1(x)))


class NanoTabPFNModel(nn.Module):
    def __init__(
        self,
        embedding_size: int,
        num_attention_heads: int,
        mlp_hidden_size: int,
        num_layers: int,
        num_outputs: int,
        dual_feature_attention: bool = False,
    ):
        super().__init__()
        self.dual_feature_attention = dual_feature_attention
        self.use_category_mask = False
        self.feature_encoder = FeatureEncoder(embedding_size)
        self.target_encoder = TargetEncoder(embedding_size)
        self.transformer_blocks = nn.ModuleList(
            [
                TransformerEncoderLayer(
                    embedding_size,
                    num_attention_heads,
                    mlp_hidden_size,
                    dual_feature_attention=dual_feature_attention,
                )
                for _ in range(num_layers)
            ]
        )
        self.decoder = Decoder(embedding_size, mlp_hidden_size, num_outputs)

    def forward(
        self, src: tuple[torch.Tensor, torch.Tensor], train_test_split_index: int
    ) -> torch.Tensor:
        x_src, y_src = src
        if len(y_src.shape) < len(x_src.shape):
            y_src = y_src.unsqueeze(-1)
        x_src = self.feature_encoder(x_src, train_test_split_index)
        num_rows = x_src.shape[1]
        y_src = self.target_encoder(y_src, num_rows)
        src = torch.cat([x_src, y_src], 2)
        category_mask = None
        for block in self.transformer_blocks:
            src = block(
                src,
                train_test_split_index=train_test_split_index,
                category_mask=category_mask,
            )
        output = src[:, train_test_split_index:, -1, :]
        output = self.decoder(output)
        return output


class FeatureEncoderWithCategories(nn.Module):
    def __init__(
        self,
        embedding_size: int,
        num_category_buckets: int = 10,
        per_column_embeddings: bool = False,
        sort_embeddings: bool = False,
        invariant_noise_encoder: bool = False,
    ):
        super().__init__()
        self.linear_layer = nn.Linear(1, embedding_size)
        self.num_category_buckets = num_category_buckets
        self.per_column_embeddings = per_column_embeddings
        self.invariant_noise_encoder = invariant_noise_encoder
        if self.invariant_noise_encoder and not self.per_column_embeddings:
            raise ValueError(
                "Invariant noise encoder requires per-column embeddings to be enabled."
            )
        self.sort_embeddings = sort_embeddings and not self.invariant_noise_encoder
        if not per_column_embeddings:
            self.category_embedding = nn.Embedding(
                num_embeddings=num_category_buckets, embedding_dim=embedding_size
            )
        else:
            self.category_embedding = None
        self.embedding_size = embedding_size
        self.column_embeddings: dict[int, nn.Embedding] = {}
        self.invariant_noise_hidden_dim = embedding_size // 2
        self.noise_size = 32
        if self.invariant_noise_encoder:
            self.noise_encoder_mlp1 = nn.Sequential(
                nn.Linear(1, self.invariant_noise_hidden_dim // 2),
                nn.GELU(),
                nn.Linear(
                    self.invariant_noise_hidden_dim // 2,
                    self.invariant_noise_hidden_dim,
                ),
            )
            self.noise_encoder_mlp2 = nn.Linear(
                self.invariant_noise_hidden_dim, self.embedding_size
            )
            # self.noise_size = 32
        else:
            self.category_encoder = nn.Sequential(
                nn.Linear(self.noise_size, self.embedding_size // 2),
                nn.GELU(),
                nn.Linear(self.embedding_size // 2, self.embedding_size),
            )

    def _apply_invariant_noise_encoder(self, noise: torch.Tensor) -> torch.Tensor:
        encoded = self.noise_encoder_mlp1(noise.unsqueeze(-1))
        encoded = encoded.sum(dim=-2)
        return self.noise_encoder_mlp2(encoded)

    def forward(
        self,
        x: torch.Tensor,
        train_test_split_index: int,
        category_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        x = x.unsqueeze(-1)
        mean = torch.mean(x[:, :train_test_split_index], dim=1, keepdims=True)
        std = torch.std(x[:, :train_test_split_index], dim=1, keepdims=True) + 1e-20
        normalized = torch.clip((x - mean) / std, min=-100, max=100)
        numeric_emb = self.linear_layer(normalized)

        if category_mask is None:
            return numeric_emb

        mask = (
            category_mask.unsqueeze(1).unsqueeze(-1).to(dtype=x.dtype, device=x.device)
        )
        if not torch.any(mask.bool()):
            return numeric_emb

        cat_values = torch.nan_to_num(x.squeeze(-1), nan=0.0)
        cat_values = cat_values.round().long()

        if self.per_column_embeddings:
            cat_emb_list = []
            for col_idx in range(cat_values.shape[-1]):
                emb = self.column_embeddings.get(col_idx)
                if emb is None:
                    emb = nn.Embedding(
                        num_embeddings=self.num_category_buckets,
                        embedding_dim=self.noise_size,
                    )
                    self.column_embeddings[col_idx] = emb.to(x.device)
                values = cat_values[..., col_idx]
                values = torch.clamp(values, min=0, max=self.num_category_buckets - 1)
                temp = emb(values)
                if self.invariant_noise_encoder:
                    temp = self._apply_invariant_noise_encoder(temp)
                else:
                    if self.sort_embeddings:
                        temp, _ = torch.sort(temp, dim=-1)
                    temp = self.category_encoder(temp)
                cat_emb_list.append(temp)
            cat_emb = torch.stack(cat_emb_list, dim=-2)
        else:
            cat_values = torch.clamp(
                cat_values, min=0, max=self.category_embedding.num_embeddings - 1
            )
            cat_emb = self.category_embedding(cat_values)
        return torch.where(mask.bool(), cat_emb, numeric_emb)


class NanoTabPFNModelCategorical(NanoTabPFNModel):
    def __init__(
        self,
        embedding_size: int,
        num_attention_heads: int,
        mlp_hidden_size: int,
        num_layers: int,
        num_outputs: int,
        num_category_buckets: int = 10,
        per_column_embeddings: bool = False,
        sort_embeddings: bool = False,
        invariant_noise_encoder: bool = False,
        dual_feature_attention: bool = False,
        category_as_numeric: bool = False,
    ):
        super().__init__(
            embedding_size,
            num_attention_heads,
            mlp_hidden_size,
            num_layers,
            num_outputs,
            dual_feature_attention=dual_feature_attention,
        )
        self.category_as_numeric = category_as_numeric
        if self.category_as_numeric:
            self.feature_encoder = FeatureEncoder(embedding_size)
            pass
        else:
            self.feature_encoder = FeatureEncoderWithCategories(
                embedding_size,
                num_category_buckets=num_category_buckets,
                per_column_embeddings=per_column_embeddings,
                sort_embeddings=sort_embeddings,
                invariant_noise_encoder=invariant_noise_encoder,
            )
        self.use_category_mask = (
            not self.category_as_numeric
        ) or dual_feature_attention

    def forward(
        self,
        src: tuple[torch.Tensor, torch.Tensor, torch.Tensor],
        train_test_split_index: int,
    ) -> torch.Tensor:
        if len(src) == 2:
            if not self.category_as_numeric or self.dual_feature_attention:
                raise ValueError(
                    "Category mask is required unless using category-as-numeric without dual attention."
                )
            x_src, y_src = src
            category_mask = None
        elif len(src) == 3:
            x_src, y_src, category_mask = src
        else:
            raise ValueError(
                "NanoTabPFNModelCategorical expects (x, y, [category_mask]) tuple."
            )
        if len(y_src.shape) < len(x_src.shape):
            y_src = y_src.unsqueeze(-1)
        if self.category_as_numeric:
            x_src = self.feature_encoder(x_src, train_test_split_index)
        else:
            x_src = self.feature_encoder(x_src, train_test_split_index, category_mask)
        num_rows = x_src.shape[1]
        y_src = self.target_encoder(y_src, num_rows)
        src = torch.cat([x_src, y_src], 2)
        feature_category_mask = None
        if category_mask is not None:
            mask = category_mask.to(dtype=torch.bool, device=src.device)
            mask = mask.unsqueeze(1).expand(-1, num_rows, -1)
            target_mask = torch.zeros(
                mask.shape[0], num_rows, 1, dtype=torch.bool, device=mask.device
            )
            feature_category_mask = torch.cat([mask, target_mask], dim=2)
        for block in self.transformer_blocks:
            src = block(
                src,
                train_test_split_index=train_test_split_index,
                category_mask=feature_category_mask,
            )
        output = src[:, train_test_split_index:, -1, :]
        output = self.decoder(output)
        return output


class NanoTabPFNClassifier:
    def __init__(self, model: NanoTabPFNModel, device: torch.device):
        self.model = model.to(device)
        self.device = device

    def fit(self, X_train: np.ndarray, y_train: np.ndarray):
        self.X_train = X_train
        self.y_train = y_train
        self.num_classes = max(set(y_train)) + 1

    def predict_proba(self, X_test: np.ndarray) -> np.ndarray:
        x = np.concatenate((self.X_train, X_test))
        y = self.y_train
        with torch.no_grad():
            x = torch.from_numpy(x).unsqueeze(0).to(torch.float).to(self.device)
            y = torch.from_numpy(y).unsqueeze(0).to(torch.float).to(self.device)
            out = self.model((x, y), train_test_split_index=len(self.X_train)).squeeze(
                0
            )
            out = out[:, : self.num_classes]
            probabilities = F.softmax(out, dim=1)
            return probabilities.to("cpu").numpy()

    def predict(self, X_test: np.ndarray) -> np.ndarray:
        return self.predict_proba(X_test).argmax(axis=1)


class NanoTabPFNClassifierCategorical(NanoTabPFNClassifier):
    def __init__(
        self,
        model: NanoTabPFNModel,
        device: torch.device,
        max_category_cardinality: int = 50,
    ):
        super().__init__(model, device)
        self.max_category_cardinality = max_category_cardinality
        self.category_mask: np.ndarray | None = None

    def fit(self, X_train: np.ndarray, y_train: np.ndarray):
        super().fit(X_train, y_train)
        num_features = X_train.shape[1]
        mask = np.zeros(num_features, dtype=np.uint8)
        for idx in range(num_features):
            vals = X_train[:, idx]
            if np.unique(vals).size <= self.max_category_cardinality:
                mask[idx] = 1
        self.category_mask = mask

    def predict_proba(self, X_test: np.ndarray) -> np.ndarray:
        if not getattr(self.model, "use_category_mask", False):
            return super().predict_proba(X_test)
        if self.category_mask is None:
            raise RuntimeError("Category mask not set; call fit first.")
        x = np.concatenate((self.X_train, X_test))
        y = self.y_train
        category_mask = (
            torch.from_numpy(self.category_mask).unsqueeze(0).to(torch.float32)
        )
        with torch.no_grad():
            x_tensor = (
                torch.from_numpy(x).unsqueeze(0).to(torch.float32).to(self.device)
            )
            y_tensor = (
                torch.from_numpy(y).unsqueeze(0).to(torch.float32).to(self.device)
            )
            out = self.model(
                (x_tensor, y_tensor, category_mask.to(self.device)),
                train_test_split_index=len(self.X_train),
            ).squeeze(0)
            out = out[:, : self.num_classes]
            probabilities = F.softmax(out, dim=1)
            return probabilities.to("cpu").numpy()


def load_checkpoint(model: torch.nn.Module, path: Path, device: str, output_log: bool = False):
    """Load checkpoint and return optimizer state dict if available."""
    checkpoint = torch.load(path, map_location=device)
    state_dict = checkpoint.get("model_state_dict", checkpoint)
    consume_prefix_in_state_dict_if_present(state_dict, "module.")
    model.load_state_dict(state_dict)
    optimizer_state = checkpoint.get("optimizer_state_dict")
    if output_log:
        if optimizer_state:
            logger.info("Loaded checkpoint from %s (with optimizer state)", path)
        else:
            logger.info("Loaded checkpoint from %s (model only)", path)
    return optimizer_state


def build_model(model_cfg: ModelConfig):
    if model_cfg.type == "categorical":
        return NanoTabPFNModelCategorical(
            embedding_size=model_cfg.embedding_size,
            num_attention_heads=model_cfg.num_attention_heads,
            mlp_hidden_size=model_cfg.mlp_hidden_size,
            num_layers=model_cfg.num_layers,
            num_outputs=model_cfg.num_outputs,
            num_category_buckets=model_cfg.num_category_buckets,
            per_column_embeddings=model_cfg.per_column_embeddings,
            sort_embeddings=model_cfg.sort_category_embeddings,
            invariant_noise_encoder=model_cfg.invariant_noise_encoder,
            dual_feature_attention=model_cfg.dual_feature_attention,
            category_as_numeric=model_cfg.category_as_numeric,
        )
    return NanoTabPFNModel(
        embedding_size=model_cfg.embedding_size,
        num_attention_heads=model_cfg.num_attention_heads,
        mlp_hidden_size=model_cfg.mlp_hidden_size,
        num_layers=model_cfg.num_layers,
        num_outputs=model_cfg.num_outputs,
    )


def build_classifier(model, device, model_cfg: ModelConfig):
    if model_cfg.type == "categorical":
        return NanoTabPFNClassifierCategorical(model, device)
    return NanoTabPFNClassifier(model, device)


def count_model_parameters(model: nn.Module, trainable_only: bool = True) -> int:
    """Return the number of model parameters."""
    if trainable_only:
        return sum(param.numel() for param in model.parameters() if param.requires_grad)
    return sum(param.numel() for param in model.parameters())


def model_size_mb(model: nn.Module, trainable_only: bool = True) -> float:
    """Return the parameter size in megabytes (MB, base-2)."""
    if trainable_only:
        params = (param for param in model.parameters() if param.requires_grad)
    else:
        params = model.parameters()
    total_bytes = sum(param.numel() * param.element_size() for param in params)
    return total_bytes / (1024**2)


def _load_model_config_from_yaml(config_path: Path) -> ModelConfig:
    with config_path.open("r") as handle:
        data = yaml.safe_load(handle)
    if not isinstance(data, dict) or "model" not in data:
        raise ValueError(f"Missing 'model' section in config: {config_path}")
    model_data = data["model"]
    if not isinstance(model_data, dict):
        raise ValueError(f"Invalid 'model' section in config: {config_path}")
    return ModelConfig(**model_data)


def _resolve_model_path(config_data: dict, override: str | None) -> Path | None:
    if override:
        return Path(override)
    train_cfg = config_data.get("train", {}) if isinstance(config_data, dict) else {}
    for key in ("load_model_path", "save_model_path"):
        candidate = train_cfg.get(key)
        if candidate:
            return Path(candidate)
    return None


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Compute NanoTabPFN model parameter counts from a YAML config."
    )
    parser.add_argument(
        "--config",
        type=str,
        required=True,
        help="Path to a YAML config containing a 'model' section.",
    )
    parser.add_argument(
        "--model-path",
        type=str,
        default=None,
        help="Optional checkpoint path; overrides train.load_model_path/save_model_path.",
    )
    parser.add_argument(
        "--trainable-only",
        action="store_true",
        help="Count only trainable parameters.",
    )
    return parser.parse_args()


def _main() -> None:
    args = _parse_args()
    config_path = Path(args.config)
    with config_path.open("r") as handle:
        config_data = yaml.safe_load(handle)
    model_cfg = ModelConfig(**config_data.get("model", {}))
    model = build_model(model_cfg)
    model_path = _resolve_model_path(config_data, args.model_path)
    if model_path and model_path.exists():
        load_checkpoint(model, model_path, device="cpu", output_log=True)
    size_mb = model_size_mb(model, trainable_only=args.trainable_only)
    print(f"parameters_mb: {size_mb:.2f}")


if __name__ == "__main__":
    _main()
