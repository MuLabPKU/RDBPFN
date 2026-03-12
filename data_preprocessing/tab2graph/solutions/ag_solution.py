import logging
from typing import Tuple, Dict, Optional, List, Any
from pathlib import Path
import wandb
from collections import defaultdict
from enum import Enum
import warnings
import pydantic

import numpy as np
import pandas as pd
import torch
from autogluon.tabular import TabularDataset, TabularPredictor
from autogluon.common.features.feature_metadata import FeatureMetadata as AGFeatMeta
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

__all__ = ['AGSolution', 'AGSolutionConfig']

HP_MODE_FULL = {
    "NN_TORCH": {},
    "GBM": [
        {"extra_trees": True, "ag_args": {"name_suffix": "XT"}},
        {},
        "GBMLarge",
    ],
    "CAT": {},
    "XGB": {},
    "FASTAI": {},
    "RF": [
        {"criterion": "gini", "ag_args": {"name_suffix": "Gini", "problem_types": ["binary", "multiclass"]}},
        {"criterion": "entropy", "ag_args": {"name_suffix": "Entr", "problem_types": ["binary", "multiclass"]}},
        {"criterion": "squared_error", "ag_args": {"name_suffix": "MSE", "problem_types": ["regression", "quantile"]}},
    ],
    "XT": [
        {"criterion": "gini", "ag_args": {"name_suffix": "Gini", "problem_types": ["binary", "multiclass"]}},
        {"criterion": "entropy", "ag_args": {"name_suffix": "Entr", "problem_types": ["binary", "multiclass"]}},
        {"criterion": "squared_error", "ag_args": {"name_suffix": "MSE", "problem_types": ["regression", "quantile"]}},
    ],
}

HP_MODE_XGB = {
    "XGB": {},
}

HP_MODE_NN = {
    "NN_TORCH": {},
}

class AGMode(str, Enum):
    # Run AG's default setting with all available models.
    full = "full"
    # Only run xgboost.
    xgb = "xgb"
    # Only run neural network.
    nn = "nn"

class AGSolutionConfig(pydantic.BaseModel):
    mode : AGMode = "xgb"
    # Whether to enable bagging and stacking
    use_ensembling : Optional[bool] = True
    # Whether to use default feature generator
    use_feature_generator : Optional[bool] = False
    # Whether to use foreign keys as features.
    use_foreign_key_feature : Optional[bool] = False
    # Time budget
    time_limit_sec : Optional[int] = 36000
    # Negative samples per positive
    negative_sampling_ratio : Optional[int] = 5


@tabml_solution
class AGSolution(TabularMLSolution):
    """AutoGluon solution class."""
    config_class = AGSolutionConfig
    name = "ag"

    def __init__(
        self,
        solution_config : AGSolutionConfig,
        data_config : TabularDatasetConfig
    ):
        super().__init__(solution_config, data_config)

    def fit(
        self,
        dataset : DBBRDBDataset,
        task_name : str,
        ckpt_path : Path,
        device : DeviceInfo
    ) -> FitSummary:
        warnings.warn("Don't use metrics computed by AG. Use final metrics reported by"
                      "the current framework instead. ") 
        _dataset = dataset.get_task(task_name)
        train_feat_store, valid_feat_store = \
            _dataset.train_set, _dataset.validation_set

        train_feat_dict, feat_meta = self.adjust_features(train_feat_store)
        train_df = pd.DataFrame(train_feat_dict, copy=False)
        valid_feat_dict, _ = self.adjust_features(valid_feat_store)
        valid_df = pd.DataFrame(valid_feat_dict, copy=False)

        train_set = TabularDataset(train_df)
        valid_set = TabularDataset(valid_df)

        self.predictor = TabularPredictor(
            label=self.get_label_name(),
            path=ckpt_path,
            verbosity=4
        )

        if self.data_config.task.task_type == DBBTaskType.retrieval:
            train_set = self.negative_sampling(train_set)

        extra_kwargs = {}
        if not self.solution_config.use_ensembling:
            extra_kwargs['auto_stack'] = False
            extra_kwargs['num_bag_folds'] = 0
            extra_kwargs['num_bag_sets'] = 1
            extra_kwargs['num_stack_levels'] = 0
        else:
            extra_kwargs['auto_stack'] = True
            extra_kwargs['use_bag_holdout'] = True
        
        if not self.solution_config.use_feature_generator:
            extra_kwargs['feature_generator'] = None

        if self.solution_config.mode == AGMode.full:
            hparams = HP_MODE_FULL
        elif self.solution_config.mode == AGMode.xgb:
            hparams = HP_MODE_XGB
        elif self.solution_config.mode == AGMode.nn:
            hparams = HP_MODE_NN
        else:
            raise ValueError(f"Unknown AG mode: {self.solution_config.mode}")

        self.predictor.fit(
            self.get_feat_label(train_set),
            tuning_data=self.get_feat_label(valid_set),
            num_gpus=len(device.gpu_devices),
            time_limit=self.solution_config.time_limit_sec,
            hyperparameters=hparams,
            feature_metadata=feat_meta,
            **extra_kwargs
        )
        self.checkpoint(ckpt_path)

        train_metric = self.get_metric(train_set)
        val_metric = self.get_metric(valid_set)

        wandb.log({'val_metric' : val_metric})

        logger.info(f"Best model: {self.predictor.get_model_best()}")

        summary = FitSummary()
        summary.val_metric = val_metric
        summary.train_metric = train_metric

        return summary

    def evaluate(
        self,
        table : Dict[str, np.ndarray],
        device : DeviceInfo,
    ) -> float:
        warnings.warn("Don't use metrics computed by AG. Use final metrics reported by"
                      "the current framework instead. ") 

        feat_dict, _ = self.adjust_features(table)
        test_df = pd.DataFrame(feat_dict, copy=False)
        test_set = TabularDataset(test_df)

        logger.info(self.predictor.leaderboard(test_set))
        metric = self.get_metric(test_set)
        return metric

    def checkpoint(self, ckpt_path : Path) -> None:
        ckpt_path = Path(ckpt_path)
        yaml_utils.save_pyd(self.solution_config, ckpt_path / 'solution_config.yaml')
        yaml_utils.save_pyd(self.data_config, ckpt_path / 'data_config.yaml')

        if self.predictor is not None:
            self.predictor.save()

    def load_from_checkpoint(self, ckpt_path : Path) -> None:
        ckpt_path = Path(ckpt_path)
        self.solution_config = yaml_utils.load_pyd(
            self.config_class, ckpt_path / 'solution_config.yaml')
        self.data_config = yaml_utils.load_pyd(
            TabularDatasetConfig, ckpt_path / 'data_config.yaml')
        self.predictor = TabularPredictor.load(ckpt_path)

    def adjust_features(
        self, 
        feat_dict : Dict[str, np.ndarray]
    ) -> Tuple[Dict[str, np.ndarray], AGFeatMeta]:
        """Adjust features suitable for AG. Construct feature metadata."""
        logger.info("Adapting features for AG ...")
        type_map_raw = {}
        type_group_map_special = defaultdict(list)
        new_feat_dict = {}
        for name, feat in feat_dict.items():
            if name in [
                self.data_config.task.key_prediction_label_column,
                self.data_config.task.key_prediction_query_idx_column,
            ]:
                new_feat_dict[name] = feat
                type_map_raw[name] = 'category'
            else:
                dtype = self.data_config.features[name].dtype
                if dtype in [DBBColumnDType.category_t, 
                             DBBColumnDType.primary_key, 
                             DBBColumnDType.foreign_key]:
                    new_feat_dict[name] = feat
                    type_map_raw[name] = 'category'
                elif dtype == DBBColumnDType.timestamp_t:
                    new_feat_dict[name] = feat
                    type_map_raw[name] = 'int'
                elif dtype == DBBColumnDType.float_t:
                    in_size = self.data_config.features[name].extra_fields['in_size']
                    if in_size == 1:
                        new_feat_dict[name] = feat
                        type_map_raw[name] = 'float'
                    else:
                        # Split 2d array
                        assert feat.ndim == 2
                        for i in range(feat.shape[1]):
                            new_name = f'{name}_{i}'
                            new_feat_dict[new_name] = feat[:,i]
                            type_map_raw[new_name] = 'float'
                elif dtype == DBBColumnDType.datetime_t:
                    new_feat_dict[name] = feat
                    type_map_raw[name] = 'datetime'
                elif dtype == DBBColumnDType.text_t:
                    new_feat_dict[name] = feat
                    type_map_raw[name] = 'object'
                    type_group_map_special['text'].append(name)
                else:
                    raise ValueError(f"Unknown feature type: {dtype}")
        feat_meta = AGFeatMeta(type_map_raw=type_map_raw,
                               type_group_map_special=type_group_map_special)
        return new_feat_dict, feat_meta

    def negative_sampling(self, df : TabularDataset) -> TabularDataset:
        target_column_name = self.data_config.task.target_column
        target_column_capacity = (
            self.data_config.features[target_column_name].extra_fields['capacity']
        )
        df_dict = {}
        for column in df.columns:
            df_dict[column] = torch.tensor(df[column].values)
        df_dict = NS.negative_sampling(
            df_dict,
            self.solution_config.negative_sampling_ratio,
            target_column_name,
            target_column_capacity,
            self.data_config.task.key_prediction_label_column,
            self.data_config.task.key_prediction_query_idx_column,
            shuffle_rest_columns=True
        )
        df = pd.DataFrame.from_dict(df_dict)
        return df

    def get_label_name(self) -> str:
        if self.data_config.task.task_type == DBBTaskType.retrieval:
            return self.data_config.task.key_prediction_label_column
        else:
            return self.data_config.task.target_column

    def get_feat_label(self, df : TabularDataset) -> TabularDataset:
        feat_names = []
        for feat_name in df.columns:
            old_name = feat_name
            if feat_name.split('_')[-1].isdigit():
                feat_name = feat_name.rsplit('_', 1)[0]
            if feat_name in [
                self.data_config.task.target_column,
                self.data_config.task.key_prediction_label_column,
                self.data_config.task.key_prediction_query_idx_column,
            ]:
                # Skip target columns.
                continue

            dtype = self.data_config.features[feat_name].dtype
            if dtype in [
                DBBColumnDType.primary_key,
            ]:
                continue

            if (
                not self.solution_config.use_foreign_key_feature
                and dtype == DBBColumnDType.foreign_key
            ):
                continue

            feat_names.append(old_name)

        _df = df[feat_names].copy()
        _df[self.get_label_name()] = df[self.get_label_name()].copy()
        return _df

    def get_metric(self, df : TabularDataset) -> float:
        metric_fn = get_metric_fn(self.data_config.task)
        label_column = self.get_label_name()
        if self.data_config.task.task_type == DBBTaskType.retrieval:
            pred = self.predictor.predict_proba(self.get_feat_label(df))
            pred = pred[1].values
        else:
            pred = self.predictor.predict_proba(self.get_feat_label(df))
            pred = pred.values
        
        label = df[label_column].values
        pred, label = torch.tensor(pred), torch.tensor(label)
        if self.data_config.task.task_type == DBBTaskType.retrieval:
            index_column = self.data_config.task.key_prediction_query_idx_column
            index = df[index_column].values
            index = torch.tensor(index)
        else:
            index = None
        metric = metric_fn(index, pred, label).item()
        return metric
