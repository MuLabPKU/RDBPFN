"""Validate and package one context configuration locally, without uploading."""

import argparse
import json
import shutil
import zipfile
from pathlib import Path

from huggingface_hub import snapshot_download
from relbench.hf import RELBENCH_HF
from relbench.submit import LEADERBOARD_TASKS, evaluate_task

from .eval_relbench_context import parse_config
from .predict_relbench import sha256, write_json


def collect_sources(runs, configuration):
    """Require complete, compatible runs and unchanged prediction files."""
    parse_config(configuration)
    reference = None
    sources = {}
    for root in runs:
        report = json.loads((root / "run.json").read_text())
        if report["status"] != "complete" or report["split"] != "test":
            raise ValueError(f"Expected a completed test run: {root}")
        identity = {key: report[key] for key in (
            "official_repo", "official_revision", "checkpoint_sha256",
            "model_config_sha256", "code_sha256", "dtype",
        )}
        identity["seed"] = report["settings"]["seed"]
        if identity["official_repo"] != RELBENCH_HF:
            raise ValueError(f"Unexpected official dataset repository: {root}")
        if reference is not None and identity != reference:
            raise ValueError("Input runs have different data, checkpoint, source or seed")
        reference = identity
        for row in report["results"]:
            if row["configuration"] != configuration:
                continue
            task = row["task"]
            if task in sources:
                raise ValueError(f"Duplicate task: {task}")
            source = root / task.replace("/", "__") / f"{configuration}.test.csv"
            if sha256(source) != row["prediction_sha256"]:
                raise ValueError(f"Prediction changed since inference: {source}")
            sources[task] = source
    if set(sources) != set(LEADERBOARD_TASKS["classification"]):
        raise ValueError("Expected exactly the 12 classification tasks")
    return reference, sources


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--runs", nargs="+", type=Path, required=True)
    parser.add_argument("--configuration", default="4096x8")
    parser.add_argument("--output-root", type=Path, required=True)
    args = parser.parse_args()
    if args.output_root.exists():
        parser.error("Choose a new output directory; existing results are never overwritten")
    reference, sources = collect_sources(args.runs, args.configuration)
    patterns = set()
    for task in sources:
        dataset, name = task.split("/")
        patterns.update([f"{dataset}/manifest.yaml",
                         f"{dataset}/tasks/{name}/manifest.yaml",
                         f"{dataset}/tasks/{name}/test.parquet"])
    snapshot = Path(snapshot_download(
        RELBENCH_HF, repo_type="dataset", revision=reference["official_revision"],
        allow_patterns=sorted(patterns),
    ))
    preds = args.output_root / "preds"
    preds.mkdir(parents=True)
    scores = []
    for task in LEADERBOARD_TASKS["classification"]:
        target = preds / (task.replace("/", "__") + ".csv")
        shutil.copyfile(sources[task], target)
        metrics = evaluate_task(task, target, dataset=snapshot / task.split("/")[0])
        scores.append({"task": task, "roc_auc": float(metrics["roc_auc"]),
                       "prediction_sha256": sha256(target)})
        print(f"Validated {task}: AUROC={metrics['roc_auc']:.8f}", flush=True)
    archive = args.output_root / "submission-classification.zip"
    with zipfile.ZipFile(archive, "w", compression=zipfile.ZIP_DEFLATED) as out:
        for path in sorted(preds.glob("*.csv")):
            out.write(path, arcname=path.name)
    with zipfile.ZipFile(archive) as out:
        if out.testzip() is not None or len(out.namelist()) != len(sources):
            raise ValueError("Archive integrity check failed")
    validation = {
        "status": "locally_validated", "classification_tasks_valid": len(scores),
        "configuration": args.configuration, **reference,
        "uploaded": False, "issue_created": False,
        "scores": scores,
        "mean_auroc_percent": 100 * sum(s["roc_auc"] for s in scores) / len(scores),
        "archive_sha256": sha256(archive),
    }
    write_json(args.output_root / "validation.json", validation)
    print(f"Local validation complete: {validation['mean_auroc_percent']:.8f}%")


if __name__ == "__main__":
    main()
