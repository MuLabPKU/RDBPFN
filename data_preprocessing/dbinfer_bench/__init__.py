from .dataset_meta import *
from .download import *
from .rdb_dataset import *
from .version import *

# Conditionally import graph-related modules only if DGL is available
try:
    import dgl
    from .graph_dataset import *
    from .ondisk_dataset_creator import *
    _GRAPH_AVAILABLE = True
except ImportError:
    _GRAPH_AVAILABLE = False
    # Optionally define dummy classes or pass
