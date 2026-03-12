import os
import random
import numpy as np
import pandas as pd
import torch
from typing import Dict, List, Optional, Tuple, Union, Set, Any
from enum import Enum
from dataclasses import dataclass

from .dataset_meta import (
    DBBTaskType,
    DBBTaskEvalMetric,
    DBBTaskMeta,
    DBBColumnSchema,
    DBBColumnDType,
    DBBTableDataFormat,
)
from .table_generation import (
    convert_to_dbb_column_dtype,
    DataType,
    Table,
    TaskGenerationSchema,
    Relationship,
)


class TaskType(Enum):
    """Types of tasks that can be generated"""

    SINGLE_TABLE_PREDICTION = "single_table_prediction"
    DIRECT_ATTRIBUTE_PREDICTION = "direct_attribute_prediction"
    RELATIONAL_AGGREGATION_PREDICTION = "relational_aggregation_prediction"


class AggregationFunction(Enum):
    """Available aggregation functions"""

    COUNT = "count"
    SUM = "sum"
    AVG = "avg"
    MAX = "max"
    MIN = "min"
    STD = "std"


class PredicateOperator(Enum):
    """Available predicate operators"""

    GREATER_THAN = ">"
    LESS_THAN = "<"
    GREATER_EQUAL = ">="
    LESS_EQUAL = "<="
    EQUAL = "=="
    NOT_EQUAL = "!="
    IN = "in"
    NOT_IN = "not_in"


class SchemaEdgeDirection(Enum):
    """Available schema edge directions"""

    PK_TO_FK = "pk_to_fk"
    FK_TO_PK = "fk_to_pk"


@dataclass
class SchemaEdge:
    """Represents a directed edge in the schema graph."""

    from_table: str
    to_table: str
    from_column: int
    to_column: int
    direction: SchemaEdgeDirection

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

    def generate_target_table_name(self) -> str:
        """Generate a target table name."""
        # Now we only random pick a leaf node as the target table
        leaf_nodes = [node for node in self.nodes.keys() if not self.get_children(node)]
        return random.choice(leaf_nodes)

    def compute_table_row_num(self) -> Dict[str, str]:
        """Compute the number of rows for each table. Root Table starts with 1 row.
        Then each table based on parent table's row number and edge type.
        """
        order = self.get_topological_order()
        num_rows_dict = {}

        for table_name in order:
            num_rows = "1"
            if table_name == order[0]:
                num_rows_dict[table_name] = "1"
            else:
                parent_tables = self.get_parents(table_name)
                for parent_table in parent_tables:
                    edge = self.get_edge(parent_table, table_name)
                    if edge.direction == SchemaEdgeDirection.PK_TO_FK:
                        num_rows = "2"
                        break
                    elif num_rows_dict[parent_table] == "2":
                        num_rows = "2"
                        break
            num_rows_dict[table_name] = num_rows

        return num_rows_dict

    def compute_possible_task_types(self, target_table_name: str) -> TaskType:
        """Compute possible task types based on schema graph structure."""

        num_rows_dict = self.compute_table_row_num()
        num_rows = int(num_rows_dict[target_table_name])
        if num_rows == "1":
            return TaskType.DIRECT_ATTRIBUTE_PREDICTION
        else:
            return TaskType.RELATIONAL_AGGREGATION_PREDICTION


@dataclass
class FocalEntity:
    """Represents a focal entity (root of instance graph)."""

    table_name: str
    record_id: int

    def get_record(self, table: Table) -> pd.Series:
        """Get the actual record from the table."""
        if table.dataframe is None:
            table.generate_dataframe()
        return table.dataframe.iloc[[self.record_id]]


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
        self.records[self.focal_entity.table_name] = focal_record
        # self.records[self.focal_entity.table_name] = [focal_record]

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
            if parent_records.empty:
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
            self.records[table_name] = candidate_records
            # self.records[table_name] = []
            # for _, record in candidate_records.iterrows():
            #     self.records[table_name].append(record)

    def _apply_pk_fk_constraint(
        self, candidate_records, target_table, parent_records, parent_table, edge, rdb
    ):
        """Apply PK-FK constraint between parent and target tables."""
        try:
            # Ensure edge.from_column is an integer
            column_index = int(edge.from_column)

            # Handle both DataFrame and Series for parent_records
            if isinstance(parent_records, pd.Series):
                # For Series, we need to get the value at the specified column_index
                # The Series represents a single record (row) with multiple columns
                if column_index >= len(parent_records):
                    return candidate_records  # Return unfiltered records
                parent_key_values = set([int(parent_records.iloc[column_index])])
            else:
                # For DataFrame, validate column index
                if column_index >= parent_records.shape[1]:
                    return candidate_records  # Return unfiltered records

                # Get parent key values using iloc for robustness
                parent_key_values = set(
                    parent_records.iloc[:, column_index].astype(int).tolist()
                )

        except Exception:
            return candidate_records  # Return unfiltered records

        # Filter target records based on PK-FK relationship
        target_column = target_table.column_names[edge.to_column]
        filtered_records = candidate_records[
            candidate_records[target_column].isin(parent_key_values)
        ]

        return filtered_records

    def get_records(self, table_name: str) -> List[pd.Series]:
        """Get records for a specific table."""
        return self.records.get(table_name, pd.DataFrame())

    def __repr__(self):
        table_counts = {table: len(records) for table, records in self.records.items()}
        return f"InstanceGraph(focal={self.focal_entity}, records={table_counts}, leaves={self.leaf_tables})"


class TargetNodeSet:
    """Represents a set of target nodes for aggregation in an instance graph."""

    def __init__(
        self, table_name: str, join_column: str, filter_condition: Optional[Dict] = None
    ):
        """
        Initialize a target node set.

        Parameters
        ----------
        table_name : str
            Name of the table to select records from
        join_column : str
            Column name to join with the focal entity. Current not used.
        filter_condition : Optional[Dict]
            Optional filter condition for the records
        """
        self.table_name = table_name
        self.join_column = join_column
        self.filter_condition = filter_condition or {}

    def __repr__(self):
        return f"TargetNodeSet(table_name={self.table_name}, join_column={self.join_column}, filter_condition={self.filter_condition})"

    def get_records(self, instance_graph: InstanceGraph) -> List[pd.Series]:
        """
        Get records from the instance graph that match the target node set criteria.

        Parameters
        ----------
        instance_graph : InstanceGraph
            The instance graph to extract records from

        Returns
        -------
        List[pd.Series]
            List of records matching the criteria
        """
        # Get all records for the target table
        all_records = instance_graph.get_records(self.table_name)

        if all_records.empty:
            return pd.DataFrame()

        return all_records

        # Apply filter conditions if specified
        # TODO: handle filter condition
        # filtered_records = []
        # for record in all_records:
        #     if self._matches_filter(record):
        #         filtered_records.append(record)

        # return filtered_records

    def _matches_filter(self, record: pd.Series) -> bool:
        """Check if a record matches the filter condition."""
        for column, condition in self.filter_condition.items():
            if column not in record.index:
                return False

            value = record[column]
            if isinstance(condition, dict):
                # Complex condition like {"operator": ">", "value": 10}
                operator = condition.get("operator", "==")
                target_value = condition.get("value")

                if not self._evaluate_condition(value, operator, target_value):
                    return False
            else:
                # Simple equality condition
                if value != condition:
                    return False

        return True

    def _evaluate_condition(self, value, operator: str, target_value) -> bool:
        """Evaluate a single condition."""
        if operator == ">":
            return value > target_value
        elif operator == "<":
            return value < target_value
        elif operator == ">=":
            return value >= target_value
        elif operator == "<=":
            return value <= target_value
        elif operator == "==":
            return value == target_value
        elif operator == "!=":
            return value != target_value
        elif operator == "in":
            return value in target_value
        elif operator == "not_in":
            return value not in target_value
        else:
            return False


class AggregationProcessor:
    """Handles aggregation operations on target node sets."""

    @staticmethod
    def apply_aggregation(
        records: pd.DataFrame, column: str, agg_func: AggregationFunction
    ) -> float:
        """
        Apply aggregation function to a DataFrame.

        Parameters
        ----------
        records : pd.DataFrame
            DataFrame containing records to aggregate
        column : str
            Column name to aggregate
        agg_func : AggregationFunction
            Aggregation function to apply

        Returns
        -------
        float
            Aggregated value
        """
        if records.empty:
            return 0.0

        # Check if column exists in the DataFrame
        if column not in records.columns:
            return 0.0

        # Extract values from the specified column
        values = records[column].dropna()

        if len(values) == 0:
            return 0.0

        if agg_func == AggregationFunction.COUNT:
            return float(len(values))
        elif agg_func == AggregationFunction.SUM:
            return float(values.sum())
        elif agg_func == AggregationFunction.AVG:
            return float(values.mean())
        elif agg_func == AggregationFunction.MAX:
            return float(values.max())
        elif agg_func == AggregationFunction.MIN:
            return float(values.min())
        elif agg_func == AggregationFunction.STD:
            return float(values.std())
        else:
            raise ValueError(f"Unknown aggregation function: {agg_func}")


class PredicateFunction:
    """Handles predicate operations for converting aggregated values to labels."""

    def __init__(
        self, operator: PredicateOperator, threshold: Union[float, int, str, List]
    ):
        """
        Initialize a predicate function.

        Parameters
        ----------
        operator : PredicateOperator
            The comparison operator
        threshold : Union[float, int, str, List]
            The threshold value(s) for comparison
        """
        self.operator = operator
        self.threshold = threshold

    def apply(self, value: Union[float, int, str]) -> bool:
        """
        Apply the predicate function to a value.

        Parameters
        ----------
        value : Union[float, int, str]
            The value to evaluate

        Returns
        -------
        bool
            Result of the predicate evaluation
        """
        if self.operator == PredicateOperator.GREATER_THAN:
            return value > self.threshold
        elif self.operator == PredicateOperator.LESS_THAN:
            return value < self.threshold
        elif self.operator == PredicateOperator.GREATER_EQUAL:
            return value >= self.threshold
        elif self.operator == PredicateOperator.LESS_EQUAL:
            return value <= self.threshold
        elif self.operator == PredicateOperator.EQUAL:
            return value == self.threshold
        elif self.operator == PredicateOperator.NOT_EQUAL:
            return value != self.threshold
        elif self.operator == PredicateOperator.IN:
            return value in self.threshold
        elif self.operator == PredicateOperator.NOT_IN:
            return value not in self.threshold
        else:
            raise ValueError(f"Unknown predicate operator: {self.operator}")


class DirectAttributeTarget:
    """Type A: Direct attribute prediction target."""

    def __init__(self, table_name: str, column_name: str):
        """
        Initialize a direct attribute target.

        Parameters
        ----------
        table_name : str
            Name of the table containing the target column
        column_name : str
            Name of the column to predict
        """
        self.table_name = table_name
        self.column_name = column_name
        self.target_type = TaskType.DIRECT_ATTRIBUTE_PREDICTION

    def compute_label(self, instance_graph: InstanceGraph) -> Any:
        """Compute label from instance graph."""

        records = instance_graph.get_records(self.table_name)
        if records.empty:
            return None
            # raise ValueError(f"No records found for table {self.table_name}")

        # For directed attribute prediction, we should only have one record
        if len(records) != 1:
            print(
                f"Warning: Expected 1 record for table {self.table_name}, got {len(records)}"
            )
            # raise ValueError(f"Expected 1 record for table {self.table_name}, got {len(records)}")

        # Return the value from the first record
        return records.iloc[0][self.column_name]

    def __repr__(self):
        return f"DirectAttributeTarget(table_name={self.table_name}, column_name={self.column_name})"


class RelationalAggregationTarget:
    """Type B: Relational aggregation prediction target."""

    def __init__(
        self,
        target_node_set: TargetNodeSet,
        aggregation_column: str,
        aggregation_func: AggregationFunction,
        predicate_func: PredicateFunction,
    ):
        """
        Initialize a relational aggregation target.

        Parameters
        ----------
        target_node_set : TargetNodeSet
            The set of nodes to aggregate over
        aggregation_column : str
            Column name to aggregate
        aggregation_func : AggregationFunction
            Aggregation function to apply
        predicate_func : PredicateFunction
            Predicate function to convert aggregated value to label
        """
        self.target_node_set = target_node_set
        self.aggregation_column = aggregation_column
        self.aggregation_func = aggregation_func
        self.predicate_func = predicate_func
        self.target_type = TaskType.RELATIONAL_AGGREGATION_PREDICTION

    def compute_label(self, instance_graph: InstanceGraph) -> bool:
        """
        Apply aggregation and predicate to target nodes.

        Parameters
        ----------
        instance_graph : InstanceGraph
            The instance graph to extract data from

        Returns
        -------
        bool
            The label after applying aggregation and predicate
        """
        # Get target records
        target_records = self.target_node_set.get_records(instance_graph)
        if target_records.empty:
            aggregated_value = 0
            # raise ValueError(f"No records found for table {self.target_node_set.table_name}")
        else:
            # Apply aggregation
            aggregated_value = AggregationProcessor.apply_aggregation(
                target_records, self.aggregation_column, self.aggregation_func
            )

        # Apply predicate
        label = int(self.predicate_func.apply(aggregated_value))

        return label

    def __repr__(self):
        return (
            f"RelationalAggregationTarget(table={self.target_node_set.table_name}, "
            f"column={self.aggregation_column}, func={self.aggregation_func.value}, "
            f"predicate={self.predicate_func.operator.value}, threshold={self.predicate_func.threshold})"
        )


AggregationFunctionList = [
    AggregationFunction.COUNT,
    AggregationFunction.SUM,
    AggregationFunction.AVG,
    AggregationFunction.MAX,
    AggregationFunction.MIN,
    # AggregationFunction.STD,
]

PredicateFunctionList = [
    PredicateFunction(PredicateOperator.GREATER_THAN, 1),
    PredicateFunction(PredicateOperator.LESS_THAN, 1),
    PredicateFunction(PredicateOperator.GREATER_EQUAL, 1),
    PredicateFunction(PredicateOperator.LESS_EQUAL, 1),
    # PredicateFunction(PredicateOperator.EQUAL, 1),
    # PredicateFunction(PredicateOperator.NOT_EQUAL, 1),
    # PredicateFunction(PredicateOperator.IN, 1),
    # PredicateFunction(PredicateOperator.NOT_IN, 1),
]
