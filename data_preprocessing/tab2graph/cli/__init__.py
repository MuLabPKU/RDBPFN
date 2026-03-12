# Always available CLI commands
from .builtin_dataset import list_builtin
from .evaluate_tab import evaluate_tab
from .fit_tab import fit_tab
from .preprocess import preprocess
from .sweep_tab import sweep_tab

__all__ = [
    "list_builtin",
    "evaluate_tab",
    "fit_tab",
    "preprocess",
    "sweep_tab",
]

# Conditionally import graph-related CLI commands only if DGL is available
try:
    import dgl
    from .evaluate_gml import evaluate_gml
    from .fit_gml import fit_gml
    from .construct_graph import construct_graph
    from .get_node_embed import get_node_embed
    from .sweep_gml import sweep_gml

    __all__ += [
        "evaluate_gml",
        "fit_gml",
        "construct_graph",
        "get_node_embed",
        "sweep_gml",
    ]
    _GRAPH_CLI_AVAILABLE = True
except ImportError:
    _GRAPH_CLI_AVAILABLE = False
