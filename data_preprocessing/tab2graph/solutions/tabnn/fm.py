from typing import Tuple, Dict, Optional, List, Any, Union
import pydantic
import torch
import torch.nn as nn
import torch.nn.functional as F

from .mlp import MLPTabNN, MLPTabNNConfig
from . import registry

class FMLayer(nn.Module):
    """Factorization Machine layer.

    Parameters
    ----------
    num_fields : int
        Number of fields of the input.

    Forward Parameters
    ------------------
    x : torch.Tensor
        Input of shape (batch_size, num_fields, embed_size)

    Forward Returns
    ---------------
    y : torch.Tensor
        Output of shape (batch_size, (num_fields * (num_fields - 1))/2)
    """
    def __init__(self, num_fields):
        super().__init__()
        self.num_fields = num_fields
        assert num_fields > 1, 'FM model requires number of fields > 1'

    @property
    def out_size(self):
        return (self.num_fields * (self.num_fields - 1)) // 2

    def forward(self, x):
        inter = torch.matmul(x, x.transpose(-2, -1))
        indices = torch.triu_indices(
            self.num_fields, self.num_fields, 1, device=x.device)
        return inter[:, indices[0], indices[1]].view(inter.size(0), -1)

    def __repr__(self):
        return f"""{self.__class__.__name__}(
    num_fields={self.num_fields},
    out_size={self.out_size}
)"""

DeepFMTabNNConfig = MLPTabNNConfig

@registry.tabnn
class DeepFMTabNN(nn.Module):
    """Deep Factorization Machine."""
    name = "deepfm"
    config_class = DeepFMTabNNConfig
    def __init__(
        self,
        config : DeepFMTabNNConfig,
        num_fields : int,
        field_size : int,
        out_size : int
    ):
        super().__init__()
        self.fm = FMLayer(num_fields)
        self.fm_out = nn.Linear(self.fm.out_size, out_size)
        self.mlp = MLPTabNN(config, num_fields, field_size, out_size)

    def forward(self, X : torch.Tensor) -> torch.Tensor:
        y_fm = self.fm_out(self.fm(X))
        y_mlp = self.mlp(X)
        y = y_fm + y_mlp
        return y
