from typing import Tuple, Dict, Optional, List, Any, Union
import pydantic
import torch
import torch.nn as nn
import torch.nn.functional as F

from . import registry

class MLPTabNNConfig(pydantic.BaseModel):
    hid_size : Optional[int] = 128
    dropout : Optional[float] = 0.5
    num_layers : Optional[int] = 3
    use_bn : Optional[bool] = False

@registry.tabnn
class MLPTabNN(nn.Module):
    name = "mlp"
    config_class = MLPTabNNConfig
    def __init__(self,
                 config : MLPTabNNConfig,
                 num_fields : int,
                 field_size : int,
                 out_size : int):
        super().__init__()
        self.config = config
        self.layers = nn.Sequential()
        in_size = num_fields * field_size
        self.layers += nn.Sequential(
            nn.Dropout(config.dropout),
            nn.Linear(in_size, config.hid_size, bias=not config.use_bn),
            nn.ReLU(),
        )
        if config.use_bn:
            self.layers.append(nn.BatchNorm1d(config.hid_size))
        for i in range(config.num_layers - 2):
            self.layers += nn.Sequential(
                nn.Dropout(config.dropout),
                nn.Linear(config.hid_size, config.hid_size, bias=not config.use_bn),
                nn.ReLU(),
            )
            if config.use_bn:
                self.layers.append(nn.BatchNorm1d(config.hid_size))
        self.layers.append(nn.Linear(config.hid_size, out_size))

    def forward(self, X : torch.Tensor) -> torch.Tensor:
        """Forward

        Input shape:
            X : (N, F, D_f)
                N is the batch size, F is the number of fields, D_f is field size.

        Output shape:
            H : (N, D_o), D_o is the output size.
        """
        X = X.view(X.shape[0], -1)
        return self.layers(X)
