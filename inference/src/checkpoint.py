from __future__ import annotations

from pathlib import Path

import torch
from torch.nn.modules.utils import consume_prefix_in_state_dict_if_present

from .model import ModelConfig, build_model


DEFAULT_PRESET_NAME = "RDBPFN"
DEFAULT_MODEL_CONFIG = ModelConfig(
    embedding_size=96,
    num_attention_heads=4,
    mlp_hidden_size=192,
    num_layers=6,
    num_outputs=2,
)


def inference_root() -> Path:
    return Path(__file__).resolve().parents[1]


def checkpoints_root() -> Path:
    return inference_root() / "checkpoints"


def resolve_checkpoint_path(identifier: str | Path | None) -> Path:
    if identifier is None or str(identifier) == DEFAULT_PRESET_NAME:
        return checkpoints_root() / DEFAULT_PRESET_NAME / "model.pt"

    path = Path(identifier)
    if path.is_file():
        return path
    if path.is_dir():
        return path / "model.pt"
    raise FileNotFoundError(
        f"Unknown checkpoint '{identifier}'. Pass '{DEFAULT_PRESET_NAME}' or a local checkpoint path."
    )


def load_checkpoint_state(path: Path, device: torch.device) -> dict:
    if not path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {path}")
    checkpoint = torch.load(path, map_location=device)
    state_dict = checkpoint.get("model_state_dict", checkpoint)
    consume_prefix_in_state_dict_if_present(state_dict, "module.")
    return state_dict


def load_model(
    identifier: str | Path | None,
    *,
    device: torch.device,
):
    model = build_model(DEFAULT_MODEL_CONFIG)
    checkpoint_path = resolve_checkpoint_path(identifier)
    state_dict = load_checkpoint_state(checkpoint_path, device)
    model.load_state_dict(state_dict)
    model.to(device)
    model.eval()
    return model
