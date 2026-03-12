import logging
from typing import Tuple, Dict, Optional, List, Any
from pathlib import Path
import wandb
from collections import defaultdict
from enum import Enum
import warnings
import pydantic
import re

import numpy as np
import pandas as pd
import torch
import xgboost as xgb
from dbinfer_bench import DBBRDBDataset, DBBColumnDType, DBBTaskType

from .base_tab import (
    TabularMLSolution,
    FitSummary,
    tabml_solution,
)
from .tabular_dataset_config import TabularDatasetConfig
from ..device import DeviceInfo
from .. import yaml_utils
from . import negative_sampler as NS
from ..evaluator import get_metric_fn

logger = logging.getLogger(__name__)
logger.setLevel('DEBUG')

__all__ = ['XGBSolution', 'XGBSolutionConfig']

class XGBSolutionConfig(pydantic.BaseModel):
    lr : float = 0.01
    gamma : float = 0.0
    max_depth : int = 6
    min_child_weight : int = 5
    subsample : float = 1.0
    colsample_bytree: float = 1.0
    # Boosting round control
    num_boost_round : int = 10000
    early_stopping_rounds: int = 50
    # Whether to use foreign keys as features.
    use_foreign_key_feature : Optional[bool] = False
    # Negative samples per positive
    negative_sampling_ratio : Optional[int] = 5


@tabml_solution
class XGBSolution(TabularMLSolution):
    """AutoGluon solution class."""
    config_class = XGBSolutionConfig
    name = "xgb"

    def __init__(
        self,
        solution_config : XGBSolutionConfig,
        data_config : TabularDatasetConfig
    ):
        super().__init__(solution_config, data_config)
        self.booster = None

    def fit(
        self,
        dataset : DBBRDBDataset,
        task_name : str,
        ckpt_path : Path,
        device : DeviceInfo
    ) -> FitSummary:
        task = dataset.get_task(task_name)
        train_feat_store, valid_feat_store = task.train_set, task.validation_set

        if self.data_config.task.task_type == DBBTaskType.retrieval:
            train_feat_store = self.negative_sampling(train_feat_store)

        train_df, train_y, train_query_idx = self.adjust_data(train_feat_store)
        valid_df, valid_y, valid_query_idx = self.adjust_data(valid_feat_store)

        dtrain = xgb.DMatrix(train_df, label=train_y, enable_categorical=True)
        dval = xgb.DMatrix(valid_df, label=valid_y, enable_categorical=True)

        # Hyperparameters
        params = {
            'eta' : self.solution_config.lr,
            'gamma' : self.solution_config.gamma,
            'subsample' : self.solution_config.subsample,
            'colsample_bytree': self.solution_config.colsample_bytree,
            'max_depth' : self.solution_config.max_depth,
            'min_child_weight': self.solution_config.min_child_weight,
            # NOTE: The threshold used by XGB to control when a categorical value should be
            # treated as one hot. Setting this to very large number to always enable it.
            'max_cat_to_onehot': 1_000_000_000,
        }
        if self.data_config.task.task_type == DBBTaskType.classification:
            params['objective'] = 'multi:softmax'
            params['num_class'] = self.data_config.task.num_classes
            params['eval_metric'] = 'mlogloss'
        elif self.data_config.task.task_type == DBBTaskType.regression:
            params['objective'] = 'reg:squarederror'
            params['eval_metric'] = 'logloss'
        elif self.data_config.task.task_type == DBBTaskType.retrieval:
            params['objective'] = 'binary:logistic'
            params['eval_metric'] = 'logloss'

        self.booster = xgb.train(
            params,
            dtrain,
            num_boost_round=self.solution_config.num_boost_round,
            evals=[(dtrain, 'train'), (dval, 'eval')],
            early_stopping_rounds=self.solution_config.early_stopping_rounds
        )

        self.checkpoint(ckpt_path)

        train_metric = self.get_metric(dtrain, train_query_idx)
        val_metric = self.get_metric(dval, valid_query_idx)

        wandb.log({'val_metric' : val_metric})

        summary = FitSummary()
        summary.val_metric = val_metric
        summary.train_metric = train_metric

        return summary

    def evaluate(
        self,
        table : Dict[str, np.ndarray],
        device : DeviceInfo,
    ) -> float:
        df, y, query_idx = self.adjust_data(table)
        data = xgb.DMatrix(df, label=y, enable_categorical=True)
        return self.get_metric(data, query_idx)

    def checkpoint(self, ckpt_path : Path) -> None:
        ckpt_path = Path(ckpt_path)
        yaml_utils.save_pyd(self.solution_config, ckpt_path / 'solution_config.yaml')
        yaml_utils.save_pyd(self.data_config, ckpt_path / 'data_config.yaml')
        if self.booster is not None:
            self.booster.save_model(ckpt_path / 'xgb_model.json')

    def load_from_checkpoint(self, ckpt_path : Path) -> None:
        ckpt_path = Path(ckpt_path)
        self.solution_config = yaml_utils.load_pyd(
            self.config_class, ckpt_path / 'solution_config.yaml')
        self.data_config = yaml_utils.load_pyd(
            TabularDatasetConfig, ckpt_path / 'data_config.yaml')
        model_path = ckpt_path / 'xgb_model.json'
        if model_path.exists():
            self.booster = xgb.Booster()
            self.booster.load_model(model_path)
        else:
            self.booster = None

    def adjust_data(
        self, 
        feat_dict : Dict[str, np.ndarray]
    ) -> Tuple[pd.DataFrame, np.ndarray, Optional[np.ndarray]]:
        """Adjust data for XGB. Return feature and label."""
        logger.info("Adapting features for XGB ...")

        feat_dict = dict(feat_dict)  # shallow copy to allow inplace mutation
        y = None
        query_idx = None

        if self.data_config.task.task_type == DBBTaskType.retrieval:
            y = feat_dict.pop(self.data_config.task.key_prediction_label_column)
            query_idx = feat_dict.pop(self.data_config.task.key_prediction_query_idx_column)
        else:
            y = feat_dict.pop(self.data_config.task.target_column)

        df = pd.DataFrame()
        for name, feat in feat_dict.items():
            if name not in self.data_config.features:
                continue
            dtype = self.data_config.features[name].dtype
            new_name = re.sub(r'[<>\[\]]', '_', name)
            if (dtype == DBBColumnDType.category_t
                or (self.solution_config.use_foreign_key_feature
                    and dtype == DBBColumnDType.foreign_key)):
                df[new_name] = pd.Series(feat).astype('category')
            elif dtype == DBBColumnDType.float_t:
                in_size = self.data_config.features[name].extra_fields['in_size']
                if in_size == 1:
                    df[new_name] = pd.Series(feat).astype('float')
                else:
                    # Split 2d array
                    assert feat.ndim == 2
                    for i in range(feat.shape[1]):
                        df[f'{new_name}_{i}'] = pd.Series(feat[:,i]).astype('float')
            else:
                logger.info(f"Ignore feature '{name}' of type {dtype}")

        return df, y, query_idx

    def negative_sampling(self, array_dict : Dict[str, np.ndarray]) -> Dict[str, np.ndarray]:
        target_column_name = self.data_config.task.target_column
        target_column_capacity = (
            self.data_config.features[target_column_name].extra_fields['capacity']
        )
        array_dict = {k : torch.tensor(v) for k, v in array_dict.items()}
        array_dict = NS.negative_sampling(
            array_dict,
            self.solution_config.negative_sampling_ratio,
            target_column_name,
            target_column_capacity,
            self.data_config.task.key_prediction_label_column,
            self.data_config.task.key_prediction_query_idx_column,
            shuffle_rest_columns=True
        )
        return array_dict

    def get_metric(self, data : xgb.DMatrix, query_idx : Optional[np.ndarray]) -> float:
        metric_fn = get_metric_fn(self.data_config.task)
        pred = self.booster.predict(data, output_margin=True, strict_shape=True)
        if self.data_config.task.task_type in [DBBTaskType.retrieval, DBBTaskType.regression]:
            pred = pred.reshape(-1)
        label = data.get_label()
        pred, label = torch.tensor(pred), torch.tensor(label)
        if query_idx is not None:
            query_idx = torch.tensor(query_idx)
        metric = metric_fn(query_idx, pred, label).item()
        return metric
