from __future__ import annotations

import argparse
import math
import sys
from pathlib import Path

import h5py
import numpy as np
from tqdm import tqdm


def load_h5_datasets(path: Path) -> dict[str, np.ndarray]:
    with h5py.File(path, "r") as handle:
        return {name: handle[name][...] for name in handle.keys()}


def dominant_ratio(values: np.ndarray, atol: float) -> float:
    if values.size == 0:
        return 0.0
    sorted_vals = np.sort(values)
    best_count = 1
    current_count = 1
    current_value = sorted_vals[0]
    for val in sorted_vals[1:]:
        if abs(val - current_value) <= atol:
            current_count += 1
        else:
            best_count = max(best_count, current_count)
            current_value = val
            current_count = 1
    best_count = max(best_count, current_count)
    return best_count / sorted_vals.size


def count_high_ratio_columns(
    data_matrix: np.ndarray, ratio_threshold: float, atol: float
) -> int:
    rows, cols = data_matrix.shape
    flagged = 0
    for col_idx in range(cols):
        column = data_matrix[:, col_idx]
        column = column[~np.isnan(column)]
        if column.size == 0:
            continue
        ratio = dominant_ratio(column, atol)
        if ratio >= ratio_threshold:
            flagged += 1
    return flagged


def expected_repetitive_columns(
    flagged: int,
    inspected_cols: int,
    sampled_cols: int,
    safety_factor: float,
) -> float:
    if inspected_cols == 0 or sampled_cols == 0:
        return 0.0
    # flagged columns observed among inspected columns approximates
    # the repetitive fraction across all available features.
    repetitive_fraction = flagged / inspected_cols
    expected = repetitive_fraction * sampled_cols
    return expected * safety_factor


def filter_datasets(
    arrays: dict[str, np.ndarray],
    ratio_threshold: float,
    max_expected_repetitive: float,
    atol: float,
    sampled_columns: int,
    safety_factor: float,
) -> tuple[np.ndarray, list[tuple[int, int, float, float]]]:
    x_data = arrays["X"]
    num_rows = arrays["num_datapoints"]
    num_cols = arrays["num_features"]
    available_cols = arrays.get("num_available_features", num_cols)
    num_datasets = x_data.shape[0]
    keep_mask = np.ones(num_datasets, dtype=bool)
    stats: list[tuple[int, int, float, float]] = []

    iterator = tqdm(range(num_datasets), desc="Filtering datasets", unit="task")
    for idx in iterator:
        rows = int(num_rows[idx])
        stored_cols = int(num_cols[idx])
        available = int(available_cols[idx])
        if rows == 0 or stored_cols == 0 or sampled_columns <= 0:
            continue
        inspect_cols = min(stored_cols, available, x_data.shape[2])
        if inspect_cols == 0:
            continue
        matrix = x_data[idx, :rows, :inspect_cols]
        flagged = count_high_ratio_columns(matrix, ratio_threshold, atol)
        target_sample = min(sampled_columns, max(1, available))
        expected = expected_repetitive_columns(
            flagged,
            inspected_cols=inspect_cols,
            sampled_cols=target_sample,
            safety_factor=safety_factor,
        )
        stats.append((idx, flagged, expected, target_sample))
        if expected > max_expected_repetitive:
            keep_mask[idx] = False
    return keep_mask, stats


def write_filtered_h5(
    arrays: dict[str, np.ndarray],
    keep_mask: np.ndarray,
    output_path: Path,
) -> None:
    keep_indices = np.nonzero(keep_mask)[0]
    num_datasets = arrays["X"].shape[0]
    if keep_indices.size == 0:
        raise RuntimeError("No datasets left after filtering.")
    with h5py.File(output_path, "w") as dst:
        for name, data in arrays.items():
            array = np.asarray(data)
            if array.shape and array.shape[0] == num_datasets:
                dst.create_dataset(name, data=array[keep_indices])
            else:
                dst.create_dataset(name, data=array)


def parse_args(argv: list[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Filter HDF5 datasets using expected repetitive columns after column sampling."
        )
    )
    parser.add_argument("input", type=Path, help="Input HDF5 dataset.")
    parser.add_argument("output", type=Path, help="Filtered output HDF5 dataset.")
    parser.add_argument(
        "--ratio",
        type=float,
        default=0.99,
        help="Minimum dominant-value ratio to consider a column repetitive (default: 0.99).",
    )
    parser.add_argument(
        "--max-expected-columns",
        type=float,
        default=0.0,
        help="Maximum expected repetitive columns per sampled subset (default: 0).",
    )
    parser.add_argument(
        "--sampled-columns",
        type=int,
        required=True,
        help="Number of columns randomly sampled during training (used-columns).",
    )
    parser.add_argument(
        "--safety-factor",
        type=float,
        default=1.25,
        help=(
            "Multiplier applied to the expected repetitive count to account for variance "
            "(default: 1.25)."
        ),
    )
    parser.add_argument(
        "--atol",
        type=float,
        default=1e-6,
        help="Absolute tolerance when comparing floats (default: 1e-6).",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv or sys.argv[1:])
    if args.sampled_columns <= 0:
        print("--sampled-columns must be positive.", file=sys.stderr)
        return 1
    if not args.input.exists():
        print(f"Input file {args.input} does not exist.", file=sys.stderr)
        return 1
    arrays = load_h5_datasets(args.input)
    keep_mask, stats = filter_datasets(
        arrays,
        ratio_threshold=args.ratio,
        max_expected_repetitive=args.max_expected_columns,
        atol=args.atol,
        sampled_columns=args.sampled_columns,
        safety_factor=args.safety_factor,
    )
    removed = int((~keep_mask).sum())
    total = keep_mask.size
    print(
        f"Identified {removed} / {total} datasets with expected repetitive columns "
        f"exceeding {args.max_expected_columns:.2f} "
        f"(sampled={args.sampled_columns}, ratio>={args.ratio:.2f}, safety={args.safety_factor})."
    )
    if removed == total:
        print("All datasets would be removed; aborting.", file=sys.stderr)
        return 1
    write_filtered_h5(arrays, keep_mask, args.output)
    print(f"Wrote filtered dataset to {args.output} with {total - removed} entries.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
