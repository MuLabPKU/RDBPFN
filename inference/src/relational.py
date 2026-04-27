from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Mapping, Sequence

import numpy as np
import pandas as pd
import torch

from .predictor import RDBPFNClassifier


TASK_TABLE_NAME = "__rdbpfn_task__"
TASK_INDEX_COLUMN = "__rdbpfn_task_id__"
SYNTHETIC_INDEX_COLUMN = "__rdbpfn_row_id__"


@dataclass(frozen=True)
class Relationship:
    parent_table: str
    parent_column: str
    child_table: str
    child_column: str


@dataclass(frozen=True)
class TableSpec:
    index: str | None = None
    time_index: str | None = None


@dataclass(frozen=True)
class RelationalDatabase:
    tables: Mapping[str, pd.DataFrame]
    relationships: Sequence[Relationship | tuple[str, str, str, str]] = ()
    table_specs: Mapping[str, TableSpec | Mapping[str, str | None]] = field(
        default_factory=dict
    )

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "relationships",
            [_coerce_relationship(rel) for rel in self.relationships],
        )
        object.__setattr__(
            self,
            "table_specs",
            {
                name: _coerce_table_spec(spec)
                for name, spec in self.table_specs.items()
            },
        )


@dataclass(frozen=True)
class RDBTaskSpec:
    target_table: str
    entity_id: str
    target: str
    time_column: str | None = None


@dataclass(frozen=True)
class RDBDFSConfig:
    max_depth: int = 2
    agg_primitives: Sequence[str] = (
        "count",
        "mean",
        "sum",
        "min",
        "max",
        "mode",
    )
    trans_primitives: Sequence[str] = ()


@dataclass
class RelationalFeatureSynthesizer:
    task: RDBTaskSpec
    dfs_config: RDBDFSConfig = field(default_factory=RDBDFSConfig)

    def __post_init__(self) -> None:
        self.features_: list | None = None
        self.feature_names_: list[str] | None = None

    def fit_transform(
        self,
        rdb: RelationalDatabase,
        task_rows: pd.DataFrame,
    ) -> pd.DataFrame:
        ft = _import_featuretools()
        entityset, task_ids, cutoff_time, ignore_columns = self._build_entityset(
            ft,
            rdb,
            task_rows,
        )
        feature_matrix, features = ft.dfs(
            entityset=entityset,
            target_dataframe_name=TASK_TABLE_NAME,
            instance_ids=None if cutoff_time is not None else task_ids,
            cutoff_time=cutoff_time,
            max_depth=self.dfs_config.max_depth,
            agg_primitives=list(self.dfs_config.agg_primitives),
            trans_primitives=list(self.dfs_config.trans_primitives),
            ignore_columns=ignore_columns,
            features_only=False,
        )
        feature_matrix = self._finalize_feature_matrix(feature_matrix, task_ids)
        if feature_matrix.shape[1] == 0:
            raise ValueError(
                "DFS did not produce any usable relational features. "
                "Check relationships, target table, and max_depth."
            )
        self.features_ = features
        self.feature_names_ = [str(name) for name in feature_matrix.columns]
        return feature_matrix

    def transform(
        self,
        rdb: RelationalDatabase,
        task_rows: pd.DataFrame,
    ) -> pd.DataFrame:
        if self.features_ is None or self.feature_names_ is None:
            raise RuntimeError("Call fit_transform() before transform().")
        ft = _import_featuretools()
        entityset, task_ids, cutoff_time, _ = self._build_entityset(
            ft,
            rdb,
            task_rows,
        )
        feature_matrix = ft.calculate_feature_matrix(
            features=self.features_,
            entityset=entityset,
            instance_ids=None if cutoff_time is not None else task_ids,
            cutoff_time=cutoff_time,
        )
        feature_matrix = self._finalize_feature_matrix(feature_matrix, task_ids)
        missing = [name for name in self.feature_names_ if name not in feature_matrix]
        if missing:
            raise ValueError(f"Missing synthesized feature columns: {missing}")
        return feature_matrix[self.feature_names_]

    def _build_entityset(
        self,
        ft,
        rdb: RelationalDatabase,
        task_rows: pd.DataFrame,
    ):
        tables = rdb.tables
        _validate_tables(tables)
        self._validate_relationships(rdb)
        if self.task.target_table not in tables:
            raise ValueError(f"Target table '{self.task.target_table}' not found.")
        if self.task.entity_id not in task_rows.columns:
            raise ValueError(
                f"Task rows must include entity id column '{self.task.entity_id}'."
            )

        normalized_tables = {
            name: table.copy(deep=False).reset_index(drop=True)
            for name, table in tables.items()
        }
        normalized_tables = self._ensure_task_entities_in_target_table(
            normalized_tables,
            task_rows,
        )

        entityset = ft.EntitySet(id="rdbpfn_inference")
        indexes = self._infer_indexes(rdb, normalized_tables)
        relationship_key_columns = self._relationship_key_columns(rdb)
        ignore_columns: dict[str, list[str]] = {}

        for table_name, table in normalized_tables.items():
            index = indexes[table_name]
            dataframe = table.drop(columns=[self.task.target], errors="ignore").copy()
            if index == SYNTHETIC_INDEX_COLUMN:
                dataframe[index] = np.arange(len(dataframe), dtype=np.int64)
            _validate_index(dataframe, table_name, index)

            spec = rdb.table_specs.get(table_name, TableSpec())
            time_index = spec.time_index if spec.time_index in dataframe.columns else None
            ignored = set(relationship_key_columns.get(table_name, set()))
            ignored.add(self.task.target)
            ignore_columns[table_name] = [
                col for col in ignored if col in dataframe.columns
            ]
            entityset.add_dataframe(
                dataframe_name=table_name,
                dataframe=dataframe,
                index=index,
                time_index=time_index,
            )

        task_frame = pd.DataFrame(
            {
                TASK_INDEX_COLUMN: np.arange(len(task_rows), dtype=np.int64),
                self.task.entity_id: task_rows[self.task.entity_id].to_numpy(copy=True),
            }
        )
        entityset.add_dataframe(
            dataframe_name=TASK_TABLE_NAME,
            dataframe=task_frame,
            index=TASK_INDEX_COLUMN,
        )
        ignore_columns[TASK_TABLE_NAME] = [self.task.entity_id]

        for rel in rdb.relationships:
            entityset.add_relationship(
                rel.parent_table,
                rel.parent_column,
                rel.child_table,
                rel.child_column,
            )
        entityset.add_relationship(
            self.task.target_table,
            self.task.entity_id,
            TASK_TABLE_NAME,
            self.task.entity_id,
        )

        cutoff_time = None
        if self.task.time_column is not None:
            if self.task.time_column not in task_rows.columns:
                raise ValueError(
                    f"Task rows must include time column '{self.task.time_column}'."
                )
            cutoff_time = pd.DataFrame(
                {
                    "instance_id": task_frame[TASK_INDEX_COLUMN].to_numpy(copy=True),
                    "time": pd.to_datetime(task_rows[self.task.time_column]).to_numpy(),
                }
            )

        return (
            entityset,
            task_frame[TASK_INDEX_COLUMN].tolist(),
            cutoff_time,
            ignore_columns,
        )

    def _validate_relationships(self, rdb: RelationalDatabase) -> None:
        tables = rdb.tables
        for rel in rdb.relationships:
            if rel.parent_table not in tables:
                raise ValueError(
                    f"Relationship parent table '{rel.parent_table}' not found."
                )
            if rel.child_table not in tables:
                raise ValueError(
                    f"Relationship child table '{rel.child_table}' not found."
                )
            if rel.parent_column not in tables[rel.parent_table]:
                raise ValueError(
                    f"Relationship parent column '{rel.parent_table}."
                    f"{rel.parent_column}' not found."
                )
            if rel.child_column not in tables[rel.child_table]:
                raise ValueError(
                    f"Relationship child column '{rel.child_table}."
                    f"{rel.child_column}' not found."
                )

    def _infer_indexes(
        self,
        rdb: RelationalDatabase,
        tables: Mapping[str, pd.DataFrame],
    ) -> dict[str, str]:
        parent_indexes: dict[str, set[str]] = {}
        for rel in rdb.relationships:
            parent_indexes.setdefault(rel.parent_table, set()).add(rel.parent_column)

        indexes = {}
        for table_name, table in tables.items():
            spec = rdb.table_specs.get(table_name, TableSpec())
            if table_name == self.task.target_table:
                indexes[table_name] = self.task.entity_id
            elif spec.index is not None:
                indexes[table_name] = spec.index
            elif len(parent_indexes.get(table_name, set())) == 1:
                indexes[table_name] = next(iter(parent_indexes[table_name]))
            elif len(parent_indexes.get(table_name, set())) > 1:
                raise ValueError(
                    f"Table '{table_name}' is a parent through multiple columns. "
                    "Keep the simple inference path to one primary key per table."
                )
            else:
                indexes[table_name] = SYNTHETIC_INDEX_COLUMN

            if (
                indexes[table_name] != SYNTHETIC_INDEX_COLUMN
                and indexes[table_name] not in table
            ):
                raise ValueError(
                    f"Index column '{indexes[table_name]}' not found in table '{table_name}'."
                )
        for rel in rdb.relationships:
            if indexes[rel.parent_table] != rel.parent_column:
                raise ValueError(
                    f"Relationship parent {rel.parent_table}.{rel.parent_column} "
                    f"must be the index for table '{rel.parent_table}'."
                )
        return indexes

    def _relationship_key_columns(self, rdb: RelationalDatabase) -> dict[str, set[str]]:
        columns: dict[str, set[str]] = {}
        for rel in rdb.relationships:
            columns.setdefault(rel.parent_table, set()).add(rel.parent_column)
            columns.setdefault(rel.child_table, set()).add(rel.child_column)
        columns.setdefault(self.task.target_table, set()).add(self.task.entity_id)
        columns.setdefault(TASK_TABLE_NAME, set()).add(self.task.entity_id)
        return columns

    def _ensure_task_entities_in_target_table(
        self,
        tables: Mapping[str, pd.DataFrame],
        task_rows: pd.DataFrame,
    ) -> dict[str, pd.DataFrame]:
        target = tables[self.task.target_table]
        existing = set(target[self.task.entity_id].dropna().tolist())
        needed = task_rows[~task_rows[self.task.entity_id].isin(existing)]
        if needed.empty:
            return dict(tables)

        appendable_columns = [col for col in target.columns if col in needed.columns]
        if self.task.entity_id not in appendable_columns:
            raise ValueError(
                f"Task rows cannot be linked to target table '{self.task.target_table}'."
            )
        additions = needed[appendable_columns].copy()
        for column in target.columns:
            if column not in additions:
                additions[column] = np.nan
        additions = additions[target.columns]
        updated = dict(tables)
        updated[self.task.target_table] = pd.concat(
            [target, additions],
            axis=0,
            ignore_index=True,
        )
        return updated

    @staticmethod
    def _finalize_feature_matrix(
        feature_matrix: pd.DataFrame,
        task_ids: Sequence[int],
    ) -> pd.DataFrame:
        feature_matrix = feature_matrix.reindex(task_ids)
        feature_matrix = feature_matrix.reset_index(drop=True)
        feature_matrix.columns = [str(col) for col in feature_matrix.columns]
        return feature_matrix


class RDBPFNRelationalClassifier:
    def __init__(
        self,
        classifier: RDBPFNClassifier,
        *,
        dfs_config: RDBDFSConfig | None = None,
    ):
        self.classifier = classifier
        self.dfs_config = dfs_config or RDBDFSConfig()
        self.task_: RDBTaskSpec | None = None
        self.synthesizer_: RelationalFeatureSynthesizer | None = None

    @classmethod
    def from_pretrained(
        cls,
        identifier: str | Path | None = None,
        *,
        device: str | torch.device | None = None,
        dfs_config: RDBDFSConfig | None = None,
    ) -> "RDBPFNRelationalClassifier":
        classifier = RDBPFNClassifier.from_pretrained(identifier, device=device)
        return cls(classifier, dfs_config=dfs_config)

    @property
    def classes_(self):
        return self.classifier.classes_

    @property
    def feature_names_(self) -> list[str] | None:
        return None if self.synthesizer_ is None else self.synthesizer_.feature_names_

    def fit(
        self,
        rdb: RelationalDatabase,
        task: RDBTaskSpec,
        task_rows: pd.DataFrame | None = None,
        *,
        dfs_config: RDBDFSConfig | None = None,
    ) -> "RDBPFNRelationalClassifier":
        self.task_ = task
        if dfs_config is not None:
            self.dfs_config = dfs_config
        self.synthesizer_ = RelationalFeatureSynthesizer(
            task=task,
            dfs_config=self.dfs_config,
        )
        train_rows = self._resolve_train_rows(rdb, task, task_rows)
        y = train_rows[task.target].to_numpy(copy=True)
        feature_frame = self.synthesizer_.fit_transform(rdb, train_rows)
        self.classifier.fit(feature_frame, y)
        return self

    def predict_proba(
        self,
        rdb: RelationalDatabase,
        task_rows: pd.DataFrame | None = None,
        *,
        task: RDBTaskSpec | None = None,
        chunk_size: int | None = None,
    ) -> np.ndarray:
        fitted_task = self._resolve_task(task)
        if self.synthesizer_ is None:
            raise RuntimeError("Call fit() before prediction.")
        prediction_rows = self._resolve_prediction_rows(rdb, fitted_task, task_rows)
        feature_frame = self.synthesizer_.transform(rdb, prediction_rows)
        return self.classifier.predict_proba(feature_frame, chunk_size=chunk_size)

    def predict(
        self,
        rdb: RelationalDatabase,
        task_rows: pd.DataFrame | None = None,
        *,
        task: RDBTaskSpec | None = None,
        chunk_size: int | None = None,
    ) -> np.ndarray:
        probabilities = self.predict_proba(
            rdb,
            task_rows,
            task=task,
            chunk_size=chunk_size,
        )
        indices = probabilities.argmax(axis=1)
        if self.classifier.classes_ is None:
            raise RuntimeError("Model has not been fit.")
        return self.classifier.classes_[indices]

    def _resolve_train_rows(
        self,
        rdb: RelationalDatabase,
        task: RDBTaskSpec,
        task_rows: pd.DataFrame | None,
    ) -> pd.DataFrame:
        rows = task_rows if task_rows is not None else rdb.tables[task.target_table]
        if task.target not in rows:
            raise ValueError(
                f"Training task rows must include target column '{task.target}'."
            )
        rows = rows[rows[task.target].notna()].copy()
        if rows.empty:
            raise ValueError("No labeled training rows found.")
        return rows

    def _resolve_prediction_rows(
        self,
        rdb: RelationalDatabase,
        task: RDBTaskSpec,
        task_rows: pd.DataFrame | None,
    ) -> pd.DataFrame:
        if task_rows is not None:
            return task_rows.copy()
        rows = rdb.tables[task.target_table]
        if task.target in rows:
            rows = rows[rows[task.target].isna()].copy()
            if rows.empty:
                raise ValueError(
                    "No unlabeled prediction rows found. Pass task_rows explicitly."
                )
            return rows
        return rows.copy()

    def _resolve_task(self, task: RDBTaskSpec | None) -> RDBTaskSpec:
        if self.task_ is None:
            raise RuntimeError("Call fit() before prediction.")
        if task is not None and task != self.task_:
            raise ValueError("Prediction task must match the task used in fit().")
        return self.task_


def _coerce_relationship(
    relationship: Relationship | tuple[str, str, str, str],
) -> Relationship:
    if isinstance(relationship, Relationship):
        return relationship
    if len(relationship) != 4:
        raise ValueError("Relationships must be 4-tuples.")
    return Relationship(*relationship)


def _coerce_table_spec(spec: TableSpec | Mapping[str, str | None]) -> TableSpec:
    if isinstance(spec, TableSpec):
        return spec
    return TableSpec(index=spec.get("index"), time_index=spec.get("time_index"))


def _validate_tables(tables: Mapping[str, pd.DataFrame]) -> None:
    if not tables:
        raise ValueError("At least one table is required.")
    for name, table in tables.items():
        if not isinstance(table, pd.DataFrame):
            raise TypeError(f"Table '{name}' must be a pandas DataFrame.")


def _validate_index(dataframe: pd.DataFrame, table_name: str, index: str) -> None:
    if index not in dataframe:
        raise ValueError(f"Index column '{index}' not found in table '{table_name}'.")
    if dataframe[index].isna().any():
        raise ValueError(f"Index column '{index}' in table '{table_name}' has nulls.")
    if dataframe[index].duplicated().any():
        raise ValueError(
            f"Index column '{index}' in table '{table_name}' must be unique."
        )


def _import_featuretools():
    try:
        import featuretools as ft
    except ModuleNotFoundError as exc:
        raise ImportError(
            "Relational inference requires Featuretools. Install it with "
            "`pip install featuretools` in the inference environment."
        ) from exc
    return ft
