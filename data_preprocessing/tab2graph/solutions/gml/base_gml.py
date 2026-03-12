"""Graph solution base classes."""
from enum import Enum
import logging
from typing import Tuple, Dict, Optional, List, Any
from pathlib import Path
import abc
import pydantic

# Import DGL conditionally only for graph functionality
# This import is protected at the package level
import dgl.graphbolt as gb
from dbinfer_bench import DBBGraphDataset

from ...device import DeviceInfo
from ..base_tab import FitSummary

__all__ = [
    'GraphMLSolutionConfig',
    'GraphMLSolution',
    'gml_solution',
    'get_gml_solution_class',
    'get_gml_solution_choice',
]

logger = logging.getLogger(__name__)
logger.setLevel('DEBUG')

class GraphMLSolutionConfig(pydantic.BaseModel):
    lr : float = 0.001
    batch_size : int = 512
    eval_batch_size : int = 512
    feat_encode_size : Optional[int] = 8
    fanouts : List[int] = [10, 10]
    eval_fanouts : Optional[List[int]] = None
    negative_sampling_ratio : Optional[int] = 5
    patience : Optional[int] = 15
    epochs : Optional[int] = 200
    embed_ntypes : Optional[List[str]] = []
    enable_temporal_sampling : Optional[bool] = True
    time_budget : Optional[float] = 0  # Unit is second. 0 means unlimited budget.

class GraphMLSolution:

    config_class = GraphMLSolutionConfig
    name = "base_gml"

    @staticmethod
    @abc.abstractstaticmethod
    def create_from_dataset(
        config : pydantic.BaseModel,
        dataset : DBBGraphDataset,
    ):
        pass

    @abc.abstractmethod
    def fit(
        self,
        dataset : DBBGraphDataset,
        task_name : str,
        ckpt_path : Path,
        device : DeviceInfo
    ) -> FitSummary:
        pass

    @abc.abstractmethod
    def evaluate(
        self,
        item_set_dict : gb.ItemSetDict,
        graph : gb.sampling_graph.SamplingGraph,
        feat_store : gb.FeatureStore,
        device : DeviceInfo,
    ) -> float:
        pass

    @abc.abstractmethod
    def checkpoint(self, ckpt_path):
        pass

    @abc.abstractmethod
    def load_from_checkpoint(self, ckpt_path):
        pass


_GML_SOLUTION_REGISTRY = {}

def gml_solution(solution_class):
    global _GML_SOLUTION_REGISTRY
    _GML_SOLUTION_REGISTRY[solution_class.name] = solution_class
    return solution_class

def get_gml_solution_class(name : str):
    global _GML_SOLUTION_REGISTRY
    solution_class = _GML_SOLUTION_REGISTRY.get(name, None)
    if solution_class is None:
        raise ValueError(f"Cannot find the solution class of name {name}.")
    return solution_class

def get_gml_solution_choice():
    """Get an enum class of all the available GML solutions."""
    names = _GML_SOLUTION_REGISTRY.keys()
    return Enum("GMLSolutionChoice", {name.upper() : name for name in names})
