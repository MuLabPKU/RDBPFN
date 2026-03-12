from typing import List, Dict, Optional, Any
import torch
import torch.nn as nn
import torch.nn.functional as F
from dbinfer_bench import DBBColumnDType

from .base import BaseFeatEncoder, feat_encoder

@feat_encoder(DBBColumnDType.category_t)
class CategoricalEncoder(BaseFeatEncoder):

    def __init__(
        self,
        config : Dict[str, Any],
        out_size : Optional[int] = None
    ):
        num_categories = config['num_categories']
        if out_size is None:
            # Decide the embedding size based on rules.
            # Borrowed from https://github.com/fastai/fastai/blob/master/fastai/tabular/model.py#L12
            out_size = int(min(128, 1.6 * num_categories ** 0.56))
        super().__init__(config, out_size)
        self.embed = nn.Embedding(num_categories, out_size)
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.xavier_uniform_(self.embed.weight)

    def forward(self, input_feat : torch.Tensor) -> torch.Tensor:
        return self.embed(input_feat.view(-1))
