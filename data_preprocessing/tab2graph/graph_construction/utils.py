from enum import Enum

class GraphConstructionChoice(str, Enum):
    r2ne = "r2ne"
    r2n = "r2n"

def get_graph_construction_class(graph_construction_name):
    if graph_construction_name == "r2ne":
        from .er_graph_construction import ERGraphConstruction
        graph_construction_class = ERGraphConstruction
    elif graph_construction_name == "r2n":
        from .rdb2graph import RDB2Graph
        graph_construction_class = RDB2Graph
    else:
        raise ValueError("Unknown graph construction name:", graph_construction_name)
    return graph_construction_class
