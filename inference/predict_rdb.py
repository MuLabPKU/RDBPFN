from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parent
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from src.relational import (
    RDBDFSConfig,
    RDBPFNRelationalClassifier,
    RDBTaskSpec,
    RelationalDatabase,
    Relationship,
)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Predict on a small relational database using DFS features."
    )
    parser.add_argument(
        "--table",
        action="append",
        required=True,
        metavar="NAME=PATH",
        help="Input table as csv/parquet. Repeat for each table.",
    )
    parser.add_argument(
        "--relationship",
        action="append",
        default=[],
        metavar="PARENT,PK,CHILD,FK",
        help="Relationship tuple. Repeat for each relationship.",
    )
    parser.add_argument("--target-table", required=True, help="Target/entity table.")
    parser.add_argument("--entity-id", required=True, help="Entity id column.")
    parser.add_argument("--target", required=True, help="Training label column.")
    parser.add_argument(
        "--time-column",
        default=None,
        help="Optional cutoff-time column in task rows.",
    )
    parser.add_argument(
        "--test-task-rows",
        default=None,
        help=(
            "Optional csv/parquet with rows to predict. If omitted, rows in the "
            "target table with missing target values are predicted."
        ),
    )
    parser.add_argument("--output", required=True, help="Output csv path.")
    parser.add_argument(
        "--checkpoint",
        default="RDBPFN",
        help="Checkpoint name or local checkpoint path.",
    )
    parser.add_argument("--device", default=None, help="Torch device, for example cpu or cuda.")
    parser.add_argument("--max-depth", type=int, default=2, help="DFS max depth.")
    args = parser.parse_args()

    tables = _load_tables(args.table)
    relationships = [_parse_relationship(value) for value in args.relationship]
    rdb = RelationalDatabase(tables=tables, relationships=relationships)
    task = RDBTaskSpec(
        target_table=args.target_table,
        entity_id=args.entity_id,
        target=args.target,
        time_column=args.time_column,
    )
    dfs_config = RDBDFSConfig(max_depth=args.max_depth)
    test_task_rows = (
        _load_table(args.test_task_rows) if args.test_task_rows is not None else None
    )

    classifier = RDBPFNRelationalClassifier.from_pretrained(
        identifier=args.checkpoint,
        device=args.device,
    )
    classifier.fit(rdb, task, dfs_config=dfs_config)
    probabilities = classifier.predict_proba(rdb, test_task_rows)
    predictions = classifier.classes_[probabilities.argmax(axis=1)]

    output = (
        test_task_rows.copy()
        if test_task_rows is not None
        else _default_prediction_rows(tables[args.target_table], args.target)
    )
    output["prediction"] = predictions
    for index, class_name in enumerate(classifier.classes_):
        output[f"prob_{class_name}"] = probabilities[:, index]

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output.to_csv(output_path, index=False)


def _load_tables(values: list[str]) -> dict[str, pd.DataFrame]:
    tables = {}
    for value in values:
        if "=" not in value:
            raise ValueError(f"Invalid --table value '{value}'. Use NAME=PATH.")
        name, path = value.split("=", 1)
        if not name:
            raise ValueError(f"Invalid --table value '{value}'. Table name is empty.")
        tables[name] = _load_table(path)
    return tables


def _load_table(path: str | Path) -> pd.DataFrame:
    path = Path(path)
    if path.suffix.lower() == ".csv":
        return pd.read_csv(path)
    if path.suffix.lower() == ".parquet":
        return pd.read_parquet(path)
    raise ValueError(f"Unsupported table format '{path.suffix}'. Use csv or parquet.")


def _parse_relationship(value: str) -> Relationship:
    parts = [part.strip() for part in value.split(",")]
    if len(parts) != 4 or any(not part for part in parts):
        raise ValueError(
            f"Invalid --relationship value '{value}'. Use PARENT,PK,CHILD,FK."
        )
    return Relationship(*parts)


def _default_prediction_rows(target_table: pd.DataFrame, target: str) -> pd.DataFrame:
    if target not in target_table:
        return target_table.copy()
    return target_table[target_table[target].isna()].copy()


if __name__ == "__main__":
    main()
