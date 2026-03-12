"""
Table Definition Module

This module provides classes and utilities for generating relational database tables,
managing relationships, and creating prediction tasks.
"""

from .table_generation import (
    DataType,
    DataTypeConfig,
    ColumnDataProcessor,
    Table,
    Relationship,
    TableGenerator,
    RDB,
)

from .task_generation import (
    TaskType,
    Task,
    TaskDataGenerator,
    TaskGenerator,
)

from .dataset_meta import (
    DBBColumnDType,
    DBBColumnSchema,
    DBBTableDataFormat,
    DBBTableSchema,
    DBBTaskType,
    DBBTaskEvalMetric,
    DBBTaskMeta,
    DBBColumnID,
    DBBRelationship,
    DBBRDBDatasetMeta,
)

from .yaml_utils import (
    save_pyd,
    load_pyd,
)

__all__ = [
    # Core table generation
    "DataType",
    "DataTypeConfig",
    "ColumnDataProcessor",
    "Table",
    "Relationship",
    "TableGenerator",
    "RDB",
    # Task generation
    "TaskType",
    "Task",
    "TaskDataGenerator",
    "TaskGenerator",
    # 4DBInfer metadata schemas
    "DBBColumnDType",
    "DBBColumnSchema",
    "DBBTableDataFormat",
    "DBBTableSchema",
    "DBBTaskType",
    "DBBTaskEvalMetric",
    "DBBTaskMeta",
    "DBBColumnID",
    "DBBRelationship",
    "DBBRDBDatasetMeta",
    # Utilities
    "save_pyd",
    "load_pyd",
]
