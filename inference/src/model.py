from __future__ import annotations

from dataclasses import asdict, dataclass

import torch
import torch.nn.functional as F
from torch import nn
from torch.nn.modules.transformer import LayerNorm, Linear, MultiheadAttention


@dataclass
class ModelConfig:
    embedding_size: int = 96
    num_attention_heads: int = 4
    mlp_hidden_size: int = 192
    num_layers: int = 6
    num_outputs: int = 2

    @classmethod
    def from_dict(cls, payload: dict) -> "ModelConfig":
        allowed = {key: value for key, value in payload.items() if key in asdict(cls())}
        return cls(**allowed)

    def to_dict(self) -> dict:
        return asdict(self)


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
        mean = torch.mean(y_train.to(torch.float32), dim=1, keepdim=True)
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
    ):
        super().__init__()
        self.self_attention_between_datapoints = MultiheadAttention(
            embedding_size,
            nhead,
            batch_first=batch_first,
            device=device,
            dtype=dtype,
        )
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

    def forward(self, src: torch.Tensor, train_test_split_index: int) -> torch.Tensor:
        batch_size, rows_size, col_size, embedding_size = src.shape
        src = src.reshape(batch_size * rows_size, col_size, embedding_size)
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
    ):
        super().__init__()
        self.feature_encoder = FeatureEncoder(embedding_size)
        self.target_encoder = TargetEncoder(embedding_size)
        self.transformer_blocks = nn.ModuleList(
            [
                TransformerEncoderLayer(
                    embedding_size,
                    num_attention_heads,
                    mlp_hidden_size,
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
        for block in self.transformer_blocks:
            src = block(src, train_test_split_index=train_test_split_index)
        output = src[:, train_test_split_index:, -1, :]
        output = self.decoder(output)
        return output


def build_model(model_config: ModelConfig) -> NanoTabPFNModel:
    return NanoTabPFNModel(
        embedding_size=model_config.embedding_size,
        num_attention_heads=model_config.num_attention_heads,
        mlp_hidden_size=model_config.mlp_hidden_size,
        num_layers=model_config.num_layers,
        num_outputs=model_config.num_outputs,
    )
