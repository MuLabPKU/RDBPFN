"""Composite class to encode a dictionary of features based on their config."""
import logging
from typing import Dict, Optional, List, Any
import torch
import torch.nn as nn

from .base import get_encoder_class
from ..tabular_dataset_config import FeatureConfig

logger = logging.getLogger(__name__)
logger.setLevel('DEBUG')

class FeatDictEncoder(nn.Module):

    def __init__(
        self,
        feat_configs : Dict[Any, FeatureConfig],
        feature_groups : Optional[List[List[Any]]],
        feat_encode_size : Optional[int]
    ):
        super().__init__()
        fg_cfgs = []
        self.ft2gid = {}
        if feature_groups is not None:
            for fg in feature_groups:
                gid = len(fg_cfgs)
                fg_cfgs.append(feat_configs[fg[0]])
                for ft in fg:
                    self.ft2gid[ft] = gid
        for ft in sorted(feat_configs.keys()):
            cfg = feat_configs[ft]
            if ft not in self.ft2gid:
                self.ft2gid[ft] = len(fg_cfgs)
                fg_cfgs.append(cfg)

        # Create encoders.
        self.encoders = nn.ModuleList()
        for i, cfg in enumerate(fg_cfgs):
            encoder_class = get_encoder_class(cfg.dtype)
            self.encoders.append(encoder_class(cfg.extra_fields, feat_encode_size))

        self.out_size_dict = {
            ft : self.encoders[gid].out_size
            for ft, gid in self.ft2gid.items()
        }

    def forward(self, input_feat_dict : Dict[Any, torch.Tensor]) -> Dict[Any, torch.Tensor]:
        return {
            ft : self.encoders[self.ft2gid[ft]](val)
            for ft, val in input_feat_dict.items()
            # FIXME: what to do if a key in input_feat_dict does not exist in the encoders?
            # (e.g. itemId in Diginetica-clicks shouldn't be encoded?)
            if ft in self.ft2gid
        }

    def __repr__(self):
        super_repr = super().__repr__()
        ft_groups = [[] for i in range(len(self.encoders))]
        for ft, gid in self.ft2gid.items():
            ft_groups[gid].append(ft)
        ft_group_str = [str(fg) for fg in ft_groups]
        extra_repr = "  feat_groups=[\n"
        for fg in ft_groups:
            extra_repr += f"    {fg}\n"
        extra_repr += "  ]\n)"
        return super_repr[:-1] + extra_repr

FeatDict = Dict[str, Dict[str, torch.Tensor]]