from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parent
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from src.predictor import RDBPFNClassifier


def main() -> None:
    parser = argparse.ArgumentParser(description="Predict on train/test csv or parquet tables.")
    parser.add_argument("--train", required=True, help="Path to train csv/parquet.")
    parser.add_argument("--test", required=True, help="Path to test csv/parquet.")
    parser.add_argument("--target", required=True, help="Target column name.")
    parser.add_argument("--output", required=True, help="Output csv path.")
    parser.add_argument("--checkpoint", default="RDBPFN", help="Checkpoint name or local checkpoint path.")
    parser.add_argument("--device", default=None, help="Torch device, for example cpu or cuda.")
    args = parser.parse_args()

    classifier = RDBPFNClassifier.from_pretrained(
        identifier=args.checkpoint,
        device=args.device,
    )
    train_df = _load_table(args.train)
    test_df = _load_table(args.test)
    classifier.fit(train_df, target=args.target)

    features = (
        test_df.drop(columns=[args.target])
        if args.target in test_df.columns
        else test_df
    )
    probabilities = classifier.predict_proba(features)
    predictions = classifier.predict(features)

    output = features.copy()
    output["prediction"] = predictions
    for index, class_name in enumerate(classifier.classes_):
        output[f"prob_{class_name}"] = probabilities[:, index]
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output.to_csv(output_path, index=False)


def _load_table(path: str | Path) -> pd.DataFrame:
    path = Path(path)
    if path.suffix.lower() == ".csv":
        return pd.read_csv(path)
    if path.suffix.lower() == ".parquet":
        return pd.read_parquet(path)
    raise ValueError(f"Unsupported table format '{path.suffix}'. Use csv or parquet.")


if __name__ == "__main__":
    main()
