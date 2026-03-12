# Always available imports
from .base_tab import *
from .tabnn_solution import *
from .ag_solution import *
from .tabular_dataset_config import *
from .xgb_solution import *
from .tabpfn_solution import *

# Conditionally import graph-based modules only if DGL is available
try:
    import dgl
    # Graph ML base classes and functions
    from .gml import *
    _GRAPH_ML_AVAILABLE = True
except ImportError:
    _GRAPH_ML_AVAILABLE = False
    # Define dummy functions for missing graph functionality
    def get_gml_solution_class(name):
        raise ImportError("DGL is required for graph ML functionality but is not installed. Please install DGL to use graph-based solutions.")
    
    def get_gml_solution_choice():
        raise ImportError("DGL is required for graph ML functionality but is not installed. Please install DGL to use graph-based solutions.")
    
    def parse_config_from_graph_dataset(*args, **kwargs):
        raise ImportError("DGL is required for graph dataset functionality but is not installed. Please install DGL to use graph-based datasets.")
