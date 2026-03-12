#!/usr/bin/env python3
"""
Task Generation Module

This module defines Task classes and utilities for generating prediction tasks
on RDB tables. Currently supports single-table column prediction tasks.
"""

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
from .task_generation_utils import (
    TaskType,
    AggregationFunction,
    PredicateFunction,
    PredicateOperator,
    TargetNodeSet,
    DirectAttributeTarget,
    RelationalAggregationTarget,
    SchemaEdge,
    SchemaGraph,
    InstanceGraph,
    FocalEntity,
    AggregationFunctionList,
    PredicateFunctionList,
)


class Task:
    """
    A task defines a prediction problem on RDB data.
    Each task is associated with key tables and a specific prediction target.
    """

    def __init__(
        self,
        task_name: str,
        task_type: TaskType,
        key_tables: Union[str, List[str]],
        target_table: str,
        real_name_for_target_table: str,
        target_column: str,
        real_name_for_target_column: str,
        evaluation_metric: DBBTaskEvalMetric,
        dbb_task_type: DBBTaskType,
        num_classes: Optional[int] = None,
        description: Optional[str] = None,
        time_column_name: Optional[str] = None,
        # Below are optional fields for complex tasks
        target_node_set: Optional[TargetNodeSet] = None,
        target_computation: Union[
            DirectAttributeTarget, RelationalAggregationTarget
        ] = None,
        schema_graph: Optional[SchemaGraph] = None,
    ):
        """
        Initialize a Task.

        Parameters
        ----------
        task_name : str
            Unique name for the task
        task_type : TaskType
            Type of the task (currently only SINGLE_TABLE_PREDICTION)
        key_tables : Union[str, List[str]]
            Key table(s) for this task. For single table prediction, this is a single table name.
            For future multi-table tasks, this can be a list of table names.
        target_table : str
            Name of the table containing the target column
        real_name_for_target_table : str
            Name of the table containing the actual primary key column
        target_column : str
            Name of the column to predict. Note that this is always the column name in the target table.
        real_name_for_target_column : str
            Name of the column to predict in the generated task table
        evaluation_metric : DBBTaskEvalMetric
            Evaluation metric for the task
        dbb_task_type : DBBTaskType
            4DBInfer task type (classification/regression)
        num_classes : Optional[int]
            Number of classes for classification tasks
        description : Optional[str]
            Optional description of the task
        time_column_name : Optional[str]
            Name of the time column in the key table
        """
        self.task_name = task_name
        self.task_type = task_type

        # Handle key tables - normalize to list format
        if isinstance(key_tables, str):
            self.key_tables = [key_tables]
        else:
            self.key_tables = key_tables

        self.target_table = target_table
        self.real_name_for_target_table = real_name_for_target_table
        self.target_column = target_column
        self.real_name_for_target_column = real_name_for_target_column
        self.evaluation_metric = evaluation_metric
        self.dbb_task_type = dbb_task_type
        self.num_classes = num_classes
        self.description = description
        self.time_column_name = time_column_name
        self.target_node_set = target_node_set
        self.target_computation = target_computation
        self.schema_graph = schema_graph

        # Validate that target table is in key tables for single table prediction
        if self.task_type == TaskType.SINGLE_TABLE_PREDICTION:
            if len(self.key_tables) != 1:
                raise ValueError(
                    "Single table prediction requires exactly one key table"
                )
            if self.target_table != self.key_tables[0]:
                raise ValueError(
                    "For single table prediction, target table must be the same as key table"
                )
        elif self.task_type == TaskType.DIRECT_ATTRIBUTE_PREDICTION:
            if self.target_computation is None:
                raise ValueError(
                    "Direct attribute prediction requires a target computation"
                )
        elif self.task_type == TaskType.RELATIONAL_AGGREGATION_PREDICTION:
            if self.target_computation is None:
                raise ValueError(
                    "Relational aggregation prediction requires a target computation"
                )

        # Task metadata and data will be generated by TaskDataGenerator
        self.task_metadata = None  # Stores column specifications, features vs labels
        self.unified_dataframe = None  # Combined features and labels dataframe
        self.task_data = None  # Train/validation/test splits

    def __repr__(self):
        return (
            f"Task(name={self.task_name}, type={self.task_type}, "
            f"key_tables={self.key_tables}, target={self.target_table}.{self.target_column}, "
            f"metric={self.evaluation_metric}), target_node_set={self.target_node_set}, target_computation={self.target_computation})"
        )

    def is_classification_task(self) -> bool:
        """Check if this is a classification task"""
        return self.dbb_task_type == DBBTaskType.classification

    def is_regression_task(self) -> bool:
        """Check if this is a regression task"""
        return self.dbb_task_type == DBBTaskType.regression

    def get_primary_key_table(self) -> str:
        """Get the primary key table for this task (first key table)"""
        return self.key_tables[0]

    def set_task_column_metadata(self, metadata: Dict):
        """
        Set task metadata generated by TaskDataGenerator.

        Parameters
        ----------
        metadata : Dict
            Task metadata containing column specifications and feature/label info
        """
        self.task_metadata = metadata

    def set_unified_dataframe(self, dataframe: pd.DataFrame):
        """
        Set unified dataframe containing all required features and labels.

        Parameters
        ----------
        dataframe : pd.DataFrame
            Unified dataframe with features and labels
        """
        self.unified_dataframe = dataframe

    def set_task_data(self, task_data: Dict):
        """
        Set task data generated by TaskDataGenerator.

        Parameters
        ----------
        task_data : Dict
            Task data dictionary containing splits and features
        """
        self.task_data = task_data

    def to_dbb_task_meta(self, source_file: str) -> DBBTaskMeta:
        """
        Convert this task to 4DBInfer task metadata format.

        Parameters
        ----------
        source_file : str
            Path to the task data file

        Returns
        -------
        DBBTaskMeta
            Task metadata in 4DBInfer format
        """
        # Create column schemas for the task data
        column_schemas = []
        for k, v in self.task_metadata["column_metadata"].items():
            column_schema = DBBColumnSchema(
                name=k,
                dtype=convert_to_dbb_column_dtype(v["data_type"]),
                **({"in_size": 1} if v["data_type"] == DataType.FLOAT else {}),
                **(
                    {"num_categories": v["num_categories"]}
                    if v["data_type"] == DataType.CATEGORICAL
                    else {}
                ),
                **(
                    {
                        # * Here we assume the parent table's primary key name is the same as the foreign key name
                        # * This is not always true for real table, but it is true for the generated synthetic tables
                        "link_to": f"{v['parent_table']}.{v['name']}"
                    }
                    if v["data_type"] == DataType.FOREIGN_KEY
                    else {}
                ),
            )
            column_schemas.append(column_schema)

        if (
            self.task_type == TaskType.DIRECT_ATTRIBUTE_PREDICTION
            or self.task_type == TaskType.RELATIONAL_AGGREGATION_PREDICTION
        ):
            column_schemas.append(
                DBBColumnSchema(
                    name=self.real_name_for_target_column,
                    dtype=DBBColumnDType.category_t,
                )
            )

        # Create task metadata
        task_meta = DBBTaskMeta(
            name=self.task_name,
            source=source_file,
            format=DBBTableDataFormat.PARQUET,
            columns=column_schemas,
            evaluation_metric=self.evaluation_metric,
            target_column=self.real_name_for_target_column,
            target_table=self.real_name_for_target_table,
            task_type=self.dbb_task_type,
            time_column=self.time_column_name,
        )

        if self.is_classification_task() and self.num_classes:
            task_meta.num_classes = self.num_classes

        return task_meta


class TaskDataGenerator:
    """
    Generates task data from key tables for specific task types.
    Handles metadata generation, feature combination, and data splitting separately.
    """

    def __init__(self, rdb, random_seed: int = 42):
        """
        Initialize TaskDataGenerator.

        Parameters
        ----------
        rdb : RDB
            The RDB object containing the tables
        random_seed : int
            Random seed for reproducible data generation
        """
        self.rdb = rdb
        self.random_seed = random_seed
        random.seed(random_seed)
        np.random.seed(random_seed)
        torch.manual_seed(random_seed)

    def _compute_column_metadata(
        self,
        task: Task,
        key_table: Table,
        all_columns: List[str],
        feature_columns: List[str],
        label_column: str,
    ) -> Dict:
        """
        Compute column metadata for a task.
        """
        column_metadata = {}
        for i, col_name in enumerate(all_columns):
            data_type_config = key_table.data_type_configs[i]
            col_meta = {
                "name": col_name,
                "data_type": data_type_config.data_type,
                "is_feature": col_name in feature_columns,
                "is_label": col_name == label_column,
                "column_index": i,
            }
            if data_type_config.data_type == DataType.CATEGORICAL:
                col_meta["num_categories"] = data_type_config.config.get(
                    "num_categories", 2
                )
            elif data_type_config.data_type == DataType.FOREIGN_KEY:
                col_meta["parent_table"] = data_type_config.config.get("parent_table")
            column_metadata[col_name] = col_meta
        return column_metadata

    def generate_task_column_metadata(
        self,
        task: Task,
        key_tables: Dict[str, Table],
    ) -> Dict:
        """
        Generate task column metadata specifying columns, features, and labels.

        Parameters
        ----------
        task : Task
            Task object containing task specifications
        key_tables : Dict[str, Table]
            Dictionary mapping table names to Table objects

        Returns
        -------
        Dict
            Task column metadata containing column specifications
        """
        if task.task_type == TaskType.SINGLE_TABLE_PREDICTION:
            # Get the single key table
            key_table_name = task.key_tables[0]
            if key_table_name not in key_tables:
                raise ValueError(f"Key table '{key_table_name}' not found")

            key_table = key_tables[key_table_name]
            all_columns = key_table.column_names.copy()
            feature_columns = [col for col in all_columns if col != task.target_column]
            label_column = task.target_column

            column_metadata = self._compute_column_metadata(
                task, key_table, all_columns, feature_columns, label_column
            )

            task_column_metadata = {
                "key_table": key_table_name,
                "all_columns": all_columns,
                "feature_columns": feature_columns,
                "label_column": label_column,
                "column_metadata": column_metadata,
            }

            return task_column_metadata
        elif (
            task.task_type == TaskType.DIRECT_ATTRIBUTE_PREDICTION
            or task.task_type == TaskType.RELATIONAL_AGGREGATION_PREDICTION
        ):
            # TODO: Handle multiple key tables
            key_table_name = task.key_tables[0]
            if key_table_name not in key_tables:
                raise ValueError(f"Key table '{key_table_name}' not found")

            key_table = key_tables[key_table_name]
            all_columns = key_table.column_names.copy()
            feature_columns = [col for col in all_columns if col != task.target_column]
            label_column = task.target_column

            column_metadata = self._compute_column_metadata(
                task, key_table, all_columns, feature_columns, label_column
            )

            task_column_metadata = {
                "key_table": key_table_name,
                "all_columns": all_columns,
                "feature_columns": feature_columns,
                "label_column": label_column,
                "column_metadata": column_metadata,
            }

            return task_column_metadata

    def generate_instance_graphs_and_compute_labels(
        self,
        task: Task,
        key_table_name: str,
        key_table: Table,
        num_samples: int = 10,
    ) -> Dict:
        """
        Generate instance graphs and compute labels for a task.
        """
        # random select some idx from key_table
        assert num_samples <= key_table.dataframe.shape[0]
        idx = random.sample(range(key_table.dataframe.shape[0]), num_samples)
        combined_df = key_table.dataframe.iloc[idx]
        removed_idx = []
        labels = []
        for i in idx:
            instance_graph = InstanceGraph(
                FocalEntity(key_table_name, i), task.schema_graph
            )
            instance_graph.generate(self.rdb)
            label = task.target_computation.compute_label(instance_graph)
            if label is None:
                removed_idx.append(i)
                continue
            labels.append(label)
        # check if all labels are the same
        if len(set(labels)) == 1:
            return None
        combined_df = combined_df.drop(removed_idx)
        combined_df[task.real_name_for_target_column] = labels
        return combined_df

    def combine_features_and_labels(
        self,
        task: Task,
        key_tables: Dict[str, Table],
    ) -> pd.DataFrame:
        """
        Combine required features and labels into a unified dataframe.

        Parameters
        ----------
        task : Task
            Task object with metadata
        key_tables : Dict[str, Table]
            Dictionary mapping table names to Table objects

        Returns
        -------
        pd.DataFrame
            Unified dataframe with features and labels
        """
        if task.task_metadata is None:
            raise ValueError(
                "Task metadata not generated. Call generate_task_metadata first."
            )

        key_table_name = task.task_metadata["key_table"]
        key_table = key_tables[key_table_name]

        # Ensure table has dataframe
        if key_table.dataframe is None:
            key_table.generate_dataframe()

        # Get required columns
        if task.task_type == TaskType.SINGLE_TABLE_PREDICTION:
            required_columns = task.task_metadata["feature_columns"] + [
                task.task_metadata["label_column"]
            ]
            # Create unified dataframe with only required columns
            unified_df = key_table.dataframe[required_columns].copy()

        elif (
            task.task_type == TaskType.DIRECT_ATTRIBUTE_PREDICTION
            or task.task_type == TaskType.RELATIONAL_AGGREGATION_PREDICTION
        ):
            # Randomly generate instance graphs and compute labels
            unified_df = self.generate_instance_graphs_and_compute_labels(
                task, key_table_name, key_table, key_table.num_rows
            )

        return unified_df

    def split_task_data(
        self,
        task: Task,
        train_ratio: float = 0.8,
        valid_ratio: float = 0.1,
    ) -> Dict:
        """
        Split unified dataframe into train/validation/test sets.

        Parameters
        ----------
        task : Task
            Task object with unified dataframe
        train_ratio : float
            Ratio of data to use for training
        valid_ratio : float
            Ratio of data to use for validation

        Returns
        -------
        Dict
            Task data dictionary containing splits
        """
        if task.unified_dataframe is None:
            raise ValueError(
                "Unified dataframe not created. Call combine_features_and_labels first."
            )

        if task.task_metadata is None:
            raise ValueError(
                "Task metadata not generated. Call generate_task_metadata first."
            )

        num_rows = len(task.unified_dataframe)
        num_train = int(num_rows * train_ratio)
        num_valid = int(num_rows * valid_ratio)

        # Create random indices for train/valid/test split
        indices = torch.randperm(num_rows)
        train_indices = indices[:num_train]
        valid_indices = indices[num_train : num_train + num_valid]
        test_indices = indices[num_train + num_valid :]

        # Extract target column (labels)
        real_target_column = task.real_name_for_target_column
        all_labels = task.unified_dataframe[real_target_column]

        # Create input features (all columns except target)
        feature_columns = task.task_metadata["feature_columns"]
        all_features = task.unified_dataframe[feature_columns]

        # Split into train/valid/test
        train_features = all_features.iloc[train_indices]
        train_labels = all_labels.iloc[train_indices]
        valid_features = all_features.iloc[valid_indices]
        valid_labels = all_labels.iloc[valid_indices]
        test_features = all_features.iloc[test_indices]
        test_labels = all_labels.iloc[test_indices]

        # Full dataframe splits
        train_df = task.unified_dataframe.iloc[train_indices]
        validation_df = task.unified_dataframe.iloc[valid_indices]
        test_df = task.unified_dataframe.iloc[test_indices]

        # Store task data
        task_data = {
            "train_features": train_features,
            "train_labels": train_labels,
            "validation_features": valid_features,
            "validation_labels": valid_labels,
            "test_features": test_features,
            "test_labels": test_labels,
            "train_indices": train_indices,
            "validation_indices": valid_indices,
            "test_indices": test_indices,
            "train_df": train_df,
            "test_df": test_df,
            "validation_df": validation_df,
            "feature_columns": feature_columns,
            "target_column": real_target_column,
        }

        return task_data

    def generate_task_data(
        self,
        task: Task,
        key_tables: Dict[str, Table],
        train_ratio: float = 0.8,
        valid_ratio: float = 0.1,
    ) -> Dict:
        """
        Generate complete task data: metadata -> unified dataframe -> splits.

        Parameters
        ----------
        task : Task
            Task object containing task specifications
        key_tables : Dict[str, Table]
            Dictionary mapping table names to Table objects
        train_ratio : float
            Ratio of data to use for training
        valid_ratio : float
            Ratio of data to use for validation

        Returns
        -------
        Dict
            Task data dictionary containing splits and features
        """
        # Step 1: Generate task metadata
        task_column_metadata = self.generate_task_column_metadata(task, key_tables)
        task.set_task_column_metadata(task_column_metadata)

        # Step 2: Combine features and labels into unified dataframe
        unified_df = self.combine_features_and_labels(task, key_tables)
        if unified_df is None:
            return None
        task.set_unified_dataframe(unified_df)

        # Step 3: Split the unified dataframe
        task_data = self.split_task_data(task, train_ratio, valid_ratio)

        return task_data

    def save_task_data(self, task_data: Dict, file_path: str):
        """
        Save task data to files in 4DBInfer format.

        Parameters
        ----------
        task_data : Dict
            Task data dictionary
        file_path : str
            Directory path to save task data
        """
        if task_data is None:
            raise ValueError("Task data is None")

        os.makedirs(file_path, exist_ok=True)

        # Save train/valid/test data
        task_data["train_df"].to_parquet(
            os.path.join(file_path, "train.parquet"), index=False
        )
        task_data["validation_df"].to_parquet(
            os.path.join(file_path, "validation.parquet"), index=False
        )
        task_data["test_df"].to_parquet(
            os.path.join(file_path, "test.parquet"), index=False
        )


class TaskGenerator:
    """
    Main interface for generating tasks and task data for RDB tables.
    Uses TaskDataGenerator internally for data generation.
    """

    def __init__(self, rdb, random_seed: int = 42):
        """
        Initialize TaskGenerator.

        Parameters
        ----------
        rdb : RDB
            RDB object
        random_seed : int
            Random seed for reproducible task generation
        """
        self.rdb = rdb
        self.random_seed = random_seed
        random.seed(random_seed)
        np.random.seed(random_seed)
        torch.manual_seed(random_seed)

        # Initialize data generator
        self.data_generator = TaskDataGenerator(rdb=rdb, random_seed=random_seed)

    def generate_single_table_prediction_tasks(
        self,
        table_name: str,
        table,
        num_tasks: int = 1,
        exclude_primary_foreign_keys: bool = True,
        exclude_timestamp_columns: bool = True,
    ) -> List[Task]:
        """
        Generate single table prediction tasks for a table.

        Parameters
        ----------
        table_name : str
            Name of the table
        table : Table
            Table object
        num_tasks : int
            Number of tasks to generate
        exclude_primary_foreign_keys : bool
            Whether to exclude primary and foreign key columns as targets
        exclude_timestamp_columns : bool
            Whether to exclude timestamp columns as targets

        Returns
        -------
        List[Task]
            List of generated tasks
        """
        tasks = []

        # Find candidate target columns
        candidate_columns = []
        for i, (col_name, data_type_config) in enumerate(
            zip(table.column_names, table.data_type_configs)
        ):
            if exclude_primary_foreign_keys:
                if data_type_config.data_type in [
                    DataType.PRIMARY_KEY,
                    DataType.FOREIGN_KEY,
                ]:
                    continue
            if exclude_timestamp_columns:
                if data_type_config.data_type == DataType.TIMESTAMP:
                    continue
            candidate_columns.append((i, col_name, data_type_config))

        if len(candidate_columns) == 0:
            print(f"Warning: No candidate columns found for table {table_name}")
            return tasks

        # Generate tasks by randomly selecting target columns
        for _ in range(min(num_tasks, len(candidate_columns))):
            # Select a random target column (without replacement)
            selected_idx = random.randint(0, len(candidate_columns) - 1)
            target_col_idx, target_col_name, target_data_type_config = (
                candidate_columns.pop(selected_idx)
            )

            # Determine task type and evaluation metric based on data type
            if target_data_type_config.data_type == DataType.CATEGORICAL:
                dbb_task_type = DBBTaskType.classification
                evaluation_metric = DBBTaskEvalMetric.accuracy
                num_classes = target_data_type_config.config.get("num_categories", 2)
            elif target_data_type_config.data_type == DataType.FLOAT:
                dbb_task_type = DBBTaskType.regression
                evaluation_metric = DBBTaskEvalMetric.mse
                num_classes = None
            else:
                # Not a target column
                continue

            time_column_name = None
            if table.time_column is not None:
                time_column_name = table.column_names[table.time_column]

            # Create task with key table concept
            task_name = f"{table_name}_{target_col_name}_task"
            task = Task(
                task_name=task_name,
                task_type=TaskType.SINGLE_TABLE_PREDICTION,
                key_tables=table_name,  # Single key table for single table prediction
                target_table=table_name,
                target_column=target_col_name,
                real_name_for_target_table=table_name,
                real_name_for_target_column=target_col_name,
                evaluation_metric=evaluation_metric,
                dbb_task_type=dbb_task_type,
                num_classes=num_classes,
                description=f"Predict {target_col_name} column in {table_name} table",
                time_column_name=time_column_name,
            )

            tasks.append(task)

        return tasks

    def generate_tasks_for_rdb_per_table(
        self,
        rdb,
        exclude_small_tables: bool = True,
        min_table_size: int = 10,
    ) -> List[Task]:
        """
        Generate tasks for each table in an RDB.

        Parameters
        ----------
        rdb : RDB
            RDB object
        exclude_small_tables : bool
            Whether to exclude small tables
        min_table_size : int
            Minimum table size to consider for task generation

        Returns
        -------
        List[Task]
            List of generated tasks
        """

        if not rdb.tables:
            return []

        all_tasks = []

        for table_name, table in rdb.tables.items():
            if exclude_small_tables and table.num_rows < min_table_size:
                continue
            tasks = self.generate_single_table_prediction_tasks(table_name, table)
            if len(tasks) > 0:
                all_tasks.extend(tasks)
                print(f"Generated {len(tasks)} tasks for table {table_name}")

        return all_tasks

    def generate_tasks_for_rdb(
        self,
        rdb,
        tasks_per_rdb: int = 3,
        exclude_small_tables: bool = True,
        min_table_size: int = 10,
    ) -> List[Task]:
        """
        Generate tasks for all tables in an RDB.

        Parameters
        ----------
        rdb : RDB
            RDB object
        tasks_per_table : int
            Number of tasks to generate per table
        exclude_small_tables : bool
            Whether to exclude small tables
        min_table_size : int
            Minimum table size to consider for task generation

        Returns
        -------
        List[Task]
            List of generated tasks
        """

        if not rdb.tables:
            return []

        all_tasks = []

        for i in range(tasks_per_rdb):
            print(f"Generating {i+1}th tasks for RDB...")
            table_name, table = random.choice(list(rdb.tables.items()))
            # Skip small tables if requested
            if exclude_small_tables and table.num_rows < min_table_size:
                print(f"Skipping table {table_name} (only {table.num_rows} rows)")
                continue

            # Generate single table prediction tasks for this table
            tasks = self.generate_single_table_prediction_tasks(
                table_name, table, num_tasks=1
            )

            if len(tasks) > 0:
                all_tasks.extend(tasks)
                print(f"Generated {len(tasks)} tasks for table {table_name}")

        return all_tasks

    def generate_tasks_for_rdb_with_complex_tasks(
        self,
        rdb,
        tasks_per_rdb: int = 3,
        exclude_small_tables: bool = True,
        min_table_size: int = 10,
    ) -> List[Task]:
        """Generate tasks for an RDB with complex tasks.

        Args:
            rdb (_type_): RDB object
            tasks_per_rdb (int, optional): Number of tasks to generate per RDB. Defaults to 3.
            exclude_small_tables (bool, optional): Whether to exclude small tables. Defaults to True.
            min_table_size (int, optional): Minimum table size to consider for task generation. Defaults to 10.

        Returns:
            List[Task]: List of generated tasks
        """

        all_tasks = []

        for i in range(tasks_per_rdb):
            print(f"Generating {i+1}th tasks for RDB...")
            focal_table_name, schema_graph = (
                self.generate_random_focal_table_and_schema(
                    rdb=rdb,
                    max_neighbors=2,
                    exclude_small_tables=exclude_small_tables,
                    min_table_size=min_table_size,
                )
            )
            # ! Currently, we only accept one type of schema graph including 3 nodes, otherwise, we skip it.
            if len(schema_graph.nodes) != 3:
                print(f"Skipping schema graph with {len(schema_graph.nodes)} nodes")
                continue
            target_table_name = schema_graph.generate_target_table_name()
            # print(f"Selected target table: {target_table_name}")
            # TODO: Implement the compute_possible_task_types method in SchemaGraph
            task_type = schema_graph.compute_possible_task_types(target_table_name)
            if task_type == TaskType.DIRECT_ATTRIBUTE_PREDICTION:
                target_column_name = random.choice(
                    rdb.tables[target_table_name].get_feature_columns(
                        only_categorical=True
                    )
                )
            elif task_type == TaskType.RELATIONAL_AGGREGATION_PREDICTION:
                target_column_name = random.choice(
                    rdb.tables[target_table_name].get_feature_columns(only_float=True)
                )
            else:
                raise ValueError(f"Invalid task type: {task_type}")
            # print(f"Selected target column: {target_column_name}")
            # TODO: The filter condition is not implemented yet
            target_node_set = TargetNodeSet(
                target_table_name, target_column_name, filter_condition=None
            )
            target_computation = None
            if task_type == TaskType.DIRECT_ATTRIBUTE_PREDICTION:
                target_computation = DirectAttributeTarget(
                    target_table_name, target_column_name
                )
            elif task_type == TaskType.RELATIONAL_AGGREGATION_PREDICTION:
                target_computation = RelationalAggregationTarget(
                    target_node_set,
                    target_column_name,
                    random.choice(AggregationFunctionList),
                    random.choice(PredicateFunctionList),
                )

            # Determine evaluation metric and DBB task type based on target computation
            if isinstance(target_computation, DirectAttributeTarget):
                # For direct attribute, we need to check the column type
                target_table = rdb.tables[target_table_name]
                target_col_idx = target_table.column_names.index(target_column_name)
                target_data_type = target_table.data_type_configs[
                    target_col_idx
                ].data_type

                if target_data_type == DataType.CATEGORICAL:
                    evaluation_metric = DBBTaskEvalMetric.accuracy
                    dbb_task_type = DBBTaskType.classification
                    num_classes = target_table.data_type_configs[
                        target_col_idx
                    ].config.get("num_categories", 2)
                else:
                    evaluation_metric = DBBTaskEvalMetric.mse
                    dbb_task_type = DBBTaskType.regression
                    num_classes = None
            else:
                # For relational aggregation, it's always binary classification
                evaluation_metric = DBBTaskEvalMetric.accuracy
                dbb_task_type = DBBTaskType.classification
                num_classes = 2

            task = Task(
                task_name=f"complex_task_{focal_table_name}_{task_type.value}_{i+1}",
                task_type=task_type,
                key_tables=[focal_table_name],
                target_table=target_table_name,
                real_name_for_target_table=focal_table_name,
                target_column=target_column_name,
                real_name_for_target_column=(
                    target_column_name
                    if task_type == TaskType.SINGLE_TABLE_PREDICTION
                    else "labels"
                ),
                evaluation_metric=evaluation_metric,
                dbb_task_type=dbb_task_type,
                num_classes=num_classes,
                target_node_set=target_node_set,
                target_computation=target_computation,
                schema_graph=schema_graph,
                description=f"Complex task: {task_type.value} on {target_table_name}.{target_column_name}",
            )
            all_tasks.append(task)

        return all_tasks

    def generate_task_data(
        self,
        tasks: List[Task],
        rdb,
        train_ratio: float = 0.8,
        valid_ratio: float = 0.1,
    ) -> None:
        """
        Generate complete data for all tasks via TaskDataGenerator.generate_task_data.

        Parameters
        ----------
        tasks : List[Task]
            List of tasks to generate data for
        rdb : RDB
            RDB object containing the tables
        train_ratio : float
            Ratio of data to use for training
        valid_ratio : float
            Ratio of data to use for validation
        """
        for task in tasks:
            task_data = self.data_generator.generate_task_data(
                task=task,
                key_tables=rdb.tables,
                train_ratio=train_ratio,
                valid_ratio=valid_ratio,
            )
            if task_data is None:
                print(f"Warning: Task {task.task_name} has no data, skipping")
                continue
            task.set_task_data(task_data)

            # Create and store task generation schema
            primary_table_name = task.get_primary_key_table()
            if primary_table_name in rdb.table_generation_schemas:
                primary_table_schema = rdb.table_generation_schemas[primary_table_name]

                task_schema = TaskGenerationSchema(
                    task_name=task.task_name,
                    primary_table_name=primary_table_name,
                    primary_table_generation_schema=primary_table_schema,
                    target_column=task.target_column,
                    task_type=task.task_type.value,
                )
                rdb.task_generation_schemas.append(task_schema)

            print(f"Generated data for task: {task.task_name}")

    def save_all_task_data(
        self,
        tasks: List[Task],
        base_path: str,
    ) -> List[DBBTaskMeta]:
        """
        Save all task data to files and return task metadata.

        Parameters
        ----------
        tasks : List[Task]
            List of tasks with generated data
        base_path : str
            Base directory to save task data

        Returns
        -------
        List[DBBTaskMeta]
            List of task metadata objects
        """
        task_metas = []

        for task in tasks:
            if task.task_data is None:
                print(f"Warning: Task {task.task_name} has no data, skipping")
                continue

            # Create task directory
            task_file_dir = os.path.join(base_path, task.task_name)
            os.makedirs(task_file_dir, exist_ok=True)

            # Save task data
            self.data_generator.save_task_data(task.task_data, task_file_dir)

            # Create task metadata
            source_file = f"{task.task_name}/{{split}}.parquet"
            task_meta = task.to_dbb_task_meta(source_file)
            task_metas.append(task_meta)

        print(f"Saved {len(task_metas)} tasks to {base_path}")
        return task_metas

    def generate_random_focal_table_and_schema(
        self,
        rdb,
        max_neighbors: int = 2,
        exclude_small_tables: bool = True,
        min_table_size: int = 10,
    ) -> Tuple[str, SchemaGraph]:
        """
        Randomly select a focal entity table and sample neighbors to create a schema graph.

        Parameters
        ----------
        rdb : RDB
            The RDB object containing tables and relationships
        max_neighbors : int
            Maximum number of neighbor tables to select (default: 2)
        exclude_small_tables : bool
            Whether to exclude small tables from selection (default: True)
        min_table_size : int
            Minimum table size to consider for focal entity selection (default: 10)

        Returns
        -------
        Tuple[str, SchemaGraph]
            A tuple containing:
            - Name of the randomly selected focal table
            - Schema graph containing the focal table and its selected neighbors
        """
        # Get candidate tables for focal entity selection
        candidate_tables = []
        for table_name, table in rdb.tables.items():
            if exclude_small_tables and table.num_rows < min_table_size:
                continue
            candidate_tables.append(table_name)

        if not candidate_tables:
            raise ValueError(
                f"No suitable tables found for focal entity selection "
                f"(min_size={min_table_size}, exclude_small={exclude_small_tables})"
            )

        # Randomly select a focal entity table
        focal_table_name = random.choice(candidate_tables)
        # print(f"Selected focal table: {focal_table_name}")

        # * Currently, we only do 2-hop 1-neighbor schema graph
        schema_graph = rdb.create_multi_hop_schema_graph(
            focal_table_name, num_hops=2, neighbors_each_hop=1
        )

        # # Select neighbor tables
        # neighbor_tables = rdb.select_neighbor_tables(focal_table_name, max_neighbors)
        # print(f"Selected neighbor tables: {neighbor_tables}")

        # # Create sub-schema graph
        # schema_graph = rdb.create_sub_schema_graph(focal_table_name, neighbor_tables)
        # print(
        #     f"Created schema graph with {len(schema_graph.nodes)} tables and {len(schema_graph.edges)} edges"
        # )

        return focal_table_name, schema_graph
