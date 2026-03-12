from typing import List, Dict, Optional, Any
import torch
import torch.nn as nn
import torch.nn.functional as F
from dbinfer_bench import DBBColumnDType

from .base import BaseFeatEncoder, feat_encoder

@feat_encoder(DBBColumnDType.float_t)
class NumericEncoder(BaseFeatEncoder):

    def __init__(
        self,
        config : Dict[str, Any],
        out_size : Optional[int] = None
    ):
        in_size = config['in_size']
        if out_size is None:
            # Do not perform projection.
            out_size = in_size
            self.has_proj = False
        else:
            self.has_proj = True
        super().__init__(config, out_size)
        if self.has_proj:
            self.proj = nn.Linear(in_size, out_size, bias=False)

    def forward(self, input_feat : torch.Tensor) -> torch.Tensor:
        if input_feat.ndim == 1:
            input_feat = input_feat.view(-1, 1)
        if self.has_proj:
            return self.proj(input_feat)
        else:
            return input_feat
