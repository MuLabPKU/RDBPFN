import pydantic

from .er_graph_construction import (
    ERGraphConstruction,
    ERGraphConstructionConfig
)

class RDB2GraphConfig(pydantic.BaseModel):
    # Whether to construct a relation table as edges.
    # If not, all tables will be constructed as nodes.
    relation_table_as_edge : bool = False


class RDB2Graph(ERGraphConstruction):

    config_class = RDB2GraphConfig
    name = "r2n"
