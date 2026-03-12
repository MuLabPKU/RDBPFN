"""Tabular solution base classes."""
from enum import Enum
import logging
from typing import Tuple, Dict, Optional, List, Any
from pathlib import Path
import abc
import pydantic
import numpy as np
import pandas as pd

from dbinfer_bench import DBBRDBDataset

from .tabular_dataset_config import TabularDatasetConfig
from ..device import DeviceInfo
from .utils import make_feature_importance_df

__all__ = [
    'FitSummary',
    'SweepChoice',
    'TabularMLSolutionConfig',
    'TabularMLSolution',
    'tabml_solution',
    'get_tabml_solution_class',
    'get_tabml_solution_choice',
]

logger = logging.getLogger(__name__)
logger.setLevel('DEBUG')

class FitSummary:

    def __init__(self):
        self.train_metric = None
        self.val_metric = None

class SweepChoice(str, Enum):
    random = "random"
    grid = "grid"
    bayes = "bayes"

class TabularMLSolutionConfig(pydantic.BaseModel):
    lr : float = 0.001
    batch_size : int = 1024
    eval_batch_size : int = 1024
    negative_sampling_ratio : Optional[int] = 5
    patience : Optional[int] = 15
    epochs : Optional[int] = 200
    embed_keys : Optional[List[str]] = []
    time_budget : Optional[float] = 0  # Unit is second. 0 means unlimited budget.

class TabularMLSolution:

    config_class = TabularMLSolutionConfig
    name = "base_table"

    def __init__(
        self,
        solution_config : pydantic.BaseModel,
        data_config : TabularDatasetConfig,
    ):
        self.solution_config = solution_config
        self.data_config = data_config

    @staticmethod
    @abc.abstractstaticmethod
    def create_from_dataset(
        config : pydantic.BaseModel,
        dataset : DBBRDBDataset,
    ):
        pass

    @abc.abstractmethod
    def fit(
        self,
        dataset : DBBRDBDataset,
        task_name : str,
        ckpt_path : Path,
        device : DeviceInfo
    ) -> FitSummary:
        pass

    @abc.abstractmethod
    def evaluate(
        self,
        table : Dict[str, np.ndarray],
        device : DeviceInfo,
    ) -> float:
        pass

    def feature_importance(
        self,
        table : Dict[str, np.ndarray],
        device : DeviceInfo,
    ) -> pd.DataFrame:
        main_metric = self.evaluate(table, device)
        shuffled_metrics = {}
        for key in table.keys():
            if key == self.data_config.task.target_column:
                continue
            shuffled_table = table.copy()
            shuffled_table[key] = np.random.permutation(shuffled_table[key])
            shuffled_metrics[key] = self.evaluate(shuffled_table, device)
        df = make_feature_importance_df(main_metric, shuffled_metrics)
        return df

    @abc.abstractmethod
    def checkpoint(self, ckpt_path):
        pass

    @abc.abstractmethod
    def load_from_checkpoint(self, ckpt_path):
        pass

_TabML_SOLUTION_REGISTRY = {}

def tabml_solution(solution_class, solution_name : str = None):
    global _TabML_SOLUTION_REGISTRY
    if solution_name is None:
        solution_name = solution_class.name
    _TabML_SOLUTION_REGISTRY[solution_name] = solution_class
    return solution_class

def get_tabml_solution_class(name : str):
    global _TabML_SOLUTION_REGISTRY
    solution_class = _TabML_SOLUTION_REGISTRY.get(name, None)
    if solution_class is None:
        raise ValueError(f"Cannot find the solution class of name {name}.")
    return solution_class

def get_tabml_solution_choice():
    """Get an enum class of all the available TabML solutions."""
    names = _TabML_SOLUTION_REGISTRY.keys()
    return Enum("TabMLSolutionChoice", {name.upper() : name for name in names})
