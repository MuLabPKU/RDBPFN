from __future__ import annotations

from pathlib import Path

import pandas as pd

from src.relational import (
    RDBDFSConfig,
    RDBPFNRelationalClassifier,
    RDBTaskSpec,
    RelationalDatabase,
    Relationship,
)


def main() -> None:
    example_dir = Path(__file__).resolve().parent / "examples" / "rdb"
    customers = pd.read_csv(example_dir / "customers.csv")
    orders = pd.read_csv(example_dir / "orders.csv")

    rdb = RelationalDatabase(
        tables={
            "customers": customers,
            "orders": orders,
        },
        relationships=[
            Relationship("customers", "customer_id", "orders", "customer_id"),
        ],
    )
    task = RDBTaskSpec(
        target_table="customers",
        entity_id="customer_id",
        target="label",
    )

    clf = RDBPFNRelationalClassifier.from_pretrained("RDBPFN")
    clf.fit(rdb, task, dfs_config=RDBDFSConfig(max_depth=2))

    test_rows = customers[customers["label"].isna()].copy()
    prob = clf.predict_proba(rdb, test_rows)
    pred = clf.predict(rdb, test_rows)

    output = test_rows[["customer_id", "segment", "signup_days"]].copy()
    output["prediction"] = pred
    for index, class_name in enumerate(clf.classes_):
        output[f"prob_{class_name}"] = prob[:, index]
    print(output.to_string(index=False))


if __name__ == "__main__":
    main()
