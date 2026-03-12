#!/usr/bin/env python3
"""
Simplified task generation with clear separation of concerns:
1. Schema graph (nodes + edges with specified directions)
2. Instance graph generation (focal entity + topo order + PK-FK constraints)
3. Label generation (from instance graph)
"""

import pandas as pd
from typing import Dict, List, Set, Optional, Any
from dataclasses import dataclass
from enum import Enum

from .table_generation import Table, DataType, DataTypeConfig


@dataclass
class SchemaEdge:
    """Represents a directed edge in the schema graph."""

    from_table: str
    to_table: str
    from_column: int
    to_column: int

    def __repr__(self):
        return f"SchemaEdge({self.from_table}[{self.from_column}] -> {self.to_table}[{self.to_column}])"


class SchemaGraph:
    """Simple schema graph with nodes and directed edges."""

    def __init__(self):
        self.nodes: Dict[str, Table] = {}
        self.edges: List[SchemaEdge] = []
        self.adjacency: Dict[str, List[str]] = {}

    def add_node(self, table_name: str, table: Table):
        """Add a table node to the schema graph."""
        self.nodes[table_name] = table
        self.adjacency[table_name] = []

    def add_edge(self, edge: SchemaEdge):
        """Add a directed edge to the schema graph."""
        self.edges.append(edge)

        # Update adjacency list
        if edge.from_table not in self.adjacency:
            self.adjacency[edge.from_table] = []
        if edge.to_table not in self.adjacency:
            self.adjacency[edge.to_table] = []

        if edge.to_table not in self.adjacency[edge.from_table]:
            self.adjacency[edge.from_table].append(edge.to_table)

    def get_parents(self, table_name: str) -> List[str]:
        """Get parent tables (tables that this table depends on)."""
        parents = []
        for edge in self.edges:
            if edge.to_table == table_name:
                parents.append(edge.from_table)
        return parents

    def get_children(self, table_name: str) -> List[str]:
        """Get child tables (tables that depend on this table)."""
        children = []
        for edge in self.edges:
            if edge.from_table == table_name:
                children.append(edge.to_table)
        return children

    def is_dag(self) -> bool:
        """Check if the schema graph is a DAG."""
        import networkx as nx

        G = nx.DiGraph()
        for table_name in self.nodes.keys():
            G.add_node(table_name)
        for edge in self.edges:
            G.add_edge(edge.from_table, edge.to_table)

        return nx.is_directed_acyclic_graph(G)

    def get_topological_order(self) -> List[str]:
        """Get topological ordering of tables."""
        import networkx as nx

        if not self.is_dag():
            raise ValueError("Schema graph is not a DAG, cannot get topological order")

        G = nx.DiGraph()
        for table_name in self.nodes.keys():
            G.add_node(table_name)
        for edge in self.edges:
            G.add_edge(edge.from_table, edge.to_table)

        return list(nx.topological_sort(G))

    def get_edge(self, from_table: str, to_table: str) -> Optional[SchemaEdge]:
        """Get the edge between two tables."""
        for edge in self.edges:
            if edge.from_table == from_table and edge.to_table == to_table:
                return edge
        return None


@dataclass
class FocalEntity:
    """Represents a focal entity (root of instance graph)."""

    table_name: str
    record_id: int

    def get_record(self, table: Table) -> pd.Series:
        """Get the actual record from the table."""
        if table.dataframe is None:
            table.generate_dataframe()
        return table.dataframe.iloc[self.record_id]


class InstanceGraph:
    """Represents an instance graph generated from a schema graph."""

    def __init__(self, focal_entity: FocalEntity, schema_graph: SchemaGraph):
        self.focal_entity = focal_entity
        self.schema_graph = schema_graph
        self.records: Dict[str, List[pd.Series]] = {}
        self.leaf_tables: Set[str] = set()

    def generate(self, rdb) -> None:
        """
        Generate instance graph by following topological order and respecting PK-FK constraints.

        Parameters
        ----------
        rdb : RDB
            The relational database containing the actual data
        """
        # Start with focal entity
        focal_table = rdb.tables[self.focal_entity.table_name]
        focal_record = self.focal_entity.get_record(focal_table)
        self.records[self.focal_entity.table_name] = [focal_record]

        # Get topological order
        try:
            topo_order = self.schema_graph.get_topological_order()
            # Start from focal entity
            focal_index = topo_order.index(self.focal_entity.table_name)
            assert (
                focal_index == 0
            ), "Focal entity must be the first table in the topological order"
            processing_order = topo_order[focal_index:]
        except ValueError:
            raise ValueError(
                "Schema graph is not a DAG, cannot generate instance graph"
            )

        # Process tables in topological order
        for table_name in processing_order:
            # Find records for this table based on PK-FK constraints
            self._find_records_for_table(rdb, table_name)

    def _find_records_for_table(self, rdb, table_name: str) -> None:
        """Find records for a table based on PK-FK constraints from parent tables."""
        if table_name not in rdb.tables:
            return

        target_table = rdb.tables[table_name]
        if target_table.dataframe is None:
            target_table.generate_dataframe()

        # Get all parent tables
        parent_tables = self.schema_graph.get_parents(table_name)

        if not parent_tables:
            # No parents, skip this table
            return

        # Start with all records in the target table
        candidate_records = target_table.dataframe.copy()

        # Apply PK-FK constraints from each parent
        for parent_table in parent_tables:
            if parent_table not in self.records:
                candidate_records = pd.DataFrame()
                break  # Skip if parent not in our instance graph

            parent_records = self.records[parent_table]
            if not parent_records:
                candidate_records = pd.DataFrame()
                break  # Skip if parent not in our instance graph

            # Get the edge between parent and target
            edge = self.schema_graph.get_edge(parent_table, table_name)
            if edge is None:
                candidate_records = pd.DataFrame()
                break  # Skip if parent not in our instance graph

            # Apply PK-FK constraint
            candidate_records = self._apply_pk_fk_constraint(
                candidate_records, target_table, parent_records, parent_table, edge, rdb
            )

            if candidate_records.empty:
                break  # No records match this parent

        # Store the filtered records
        if not candidate_records.empty:
            self.records[table_name] = []
            for _, record in candidate_records.iterrows():
                self.records[table_name].append(record)

    def _apply_pk_fk_constraint(
        self, candidate_records, target_table, parent_records, parent_table, edge, rdb
    ):
        """Apply PK-FK constraint between parent and target tables."""
        # Get parent key values
        parent_key_values = set()
        for parent_record in parent_records:
            parent_key_values.add(parent_record.iloc[edge.from_column])

        # Filter target records based on PK-FK relationship
        target_column = target_table.column_names[edge.to_column]
        filtered_records = candidate_records[
            candidate_records[target_column].isin(parent_key_values)
        ]

        return filtered_records

    def get_records(self, table_name: str) -> List[pd.Series]:
        """Get records for a specific table."""
        return self.records.get(table_name, [])

    def __repr__(self):
        table_counts = {table: len(records) for table, records in self.records.items()}
        return f"InstanceGraph(focal={self.focal_entity}, records={table_counts}, leaves={self.leaf_tables})"


# Label generation classes (simplified)
class DirectAttributeTarget:
    """Direct attribute prediction target."""

    def __init__(self, table_name: str, column_name: str):
        self.table_name = table_name
        self.column_name = column_name

    def compute_label(self, instance_graph: InstanceGraph) -> Any:
        """Compute label from instance graph."""

        records = instance_graph.get_records(self.table_name)
        if not records:
            raise ValueError(f"No records found for table {self.table_name}")

        # Return the value from the first record
        return records[0][self.column_name]


class RelationalAggregationTarget:
    """Relational aggregation prediction target."""

    def __init__(
        self,
        table_name: str,
        column_name: str,
        aggregation_func: str = "count",
    ):
        self.table_name = table_name
        self.column_name = column_name
        self.aggregation_func = aggregation_func

    def compute_label(self, instance_graph: InstanceGraph) -> Any:
        """Compute label from instance graph."""
        records = instance_graph.get_records(self.table_name)
        if not records:
            return 0

        # Simple aggregation
        if self.aggregation_func == "count":
            return len(records)
        elif self.aggregation_func == "sum":
            return sum(record[self.column_name] for record in records)
        elif self.aggregation_func == "avg":
            values = [record[self.column_name] for record in records]
            return sum(values) / len(values) if values else 0
        else:
            raise ValueError(f"Unknown aggregation function: {self.aggregation_func}")


# Utility functions
def create_schema_graph_from_rdb(rdb, edges: List[SchemaEdge]) -> SchemaGraph:
    """Create a schema graph from RDB with specified edges."""
    schema_graph = SchemaGraph()

    # Add specified edges
    for edge in edges:
        # If start table is not in the schema graph, add it
        if edge.from_table not in schema_graph.nodes:
            schema_graph.add_node(edge.from_table, rdb.tables[edge.from_table])
        # If end table is not in the schema graph, add it
        if edge.to_table not in schema_graph.nodes:
            schema_graph.add_node(edge.to_table, rdb.tables[edge.to_table])
        schema_graph.add_edge(edge)

    return schema_graph


def generate_focal_entities(
    rdb, table_name: str, num_samples: int = 10
) -> List[FocalEntity]:
    """Generate random focal entities from a table."""
    import random

    table = rdb.tables[table_name]
    if table.dataframe is None:
        table.generate_dataframe()

    focal_entities = []
    for _ in range(num_samples):
        record_id = random.randint(0, len(table.dataframe) - 1)
        focal_entity = FocalEntity(table_name, record_id)
        focal_entities.append(focal_entity)

    return focal_entities


def generate_instance_graphs(
    rdb, schema_graph: SchemaGraph, focal_entities: List[FocalEntity]
) -> List[InstanceGraph]:
    """Generate instance graphs from schema graph and focal entities."""
    instance_graphs = []

    for focal_entity in focal_entities:
        instance_graph = InstanceGraph(focal_entity, schema_graph)
        instance_graph.generate(rdb)
        instance_graphs.append(instance_graph)

    return instance_graphs
