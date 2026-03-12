from __future__ import annotations

import argparse
from pathlib import Path
from typing import Tuple

import h5py
import numpy as np
import torch
import torch.nn.functional as F


def sparse2dense(
    sparse_tensor: torch.Tensor,
    row_lengths: torch.Tensor,
    max_len: int | None = None,
    dtype: torch.dtype = torch.float32,
) -> torch.Tensor:
    if sparse_tensor.dim() != 1:
        raise ValueError("sparse_tensor must be 1D")
    if row_lengths.sum().item() != sparse_tensor.numel():
        raise ValueError("row_lengths must sum to sparse tensor length")
    num_rows = row_lengths.shape[0]
    max_len = max_len or int(row_lengths.max().item())
    dense = torch.zeros(num_rows, max_len, dtype=dtype, device=sparse_tensor.device)
    indices = torch.arange(max_len, device=sparse_tensor.device)
    mask = indices.unsqueeze(0) < row_lengths.unsqueeze(1)
    dense[mask] = sparse_tensor.to(dtype)
    return dense


def load_batch_as_tensor(batch_path: Path, device: str) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    batch = torch.load(batch_path, map_location=device, weights_only=True)
    X = batch["X"]
    y = batch["y"]
    d = batch["d"].to("cpu")
    seq_lens = batch["seq_lens"].to("cpu")
    train_sizes = batch["train_sizes"].to("cpu")
    batch_size = int(batch["batch_size"])

    if seq_lens.numel() == 1:
        seq_lens = seq_lens.repeat(batch_size)
    if train_sizes.numel() == 1:
        train_sizes = train_sizes.repeat(batch_size)

    if hasattr(X, "is_nested") and X.is_nested:
        X = X.to_padded_tensor(0.0)
        y = y.to_padded_tensor(0.0)
    else:
        seq_len = int(seq_lens[0].item())
        dense = sparse2dense(
            X, d.repeat_interleave(seq_len), dtype=torch.float32
        ).view(batch_size, seq_len, -1)
        X = dense
        y = y.view(batch_size, seq_len)

    return X.cpu(), y.cpu(), d.cpu(), seq_lens.cpu(), train_sizes.cpu()


def _encode_categorical_columns_inplace(
    sample: torch.Tensor,
    num_features: int,
    max_categories: int,
) -> np.ndarray:
    view = sample[:, :num_features]
    arr = view.numpy()
    mask = np.zeros(num_features, dtype=np.uint8)
    for col_idx in range(num_features):
        column = arr[:, col_idx]
        valid_mask = ~np.isnan(column)
        valid_values = column[valid_mask]
        if valid_values.size == 0:
            continue
        unique_values, counts = np.unique(valid_values, return_counts=True)
        if unique_values.size == 0 or unique_values.size > max_categories:
            continue
        order = np.lexsort((unique_values, -counts))
        sorted_values = unique_values[order]
        mapping = {value: idx for idx, value in enumerate(sorted_values)}
        encoded = np.array([mapping[val] for val in valid_values], dtype=np.float32)
        column_encoded = column.copy()
        column_encoded[valid_mask] = encoded
        arr[:, col_idx] = column_encoded
        mask[col_idx] = 1
    return mask


def main():
    parser = argparse.ArgumentParser(
        description="Merge fixed-size ICL batches into a single HDF5 file using RAM concatenation."
    )
    parser.add_argument("--input-dir", type=Path, required=True, help="Directory with batch_*.pt files.")
    parser.add_argument("--output", type=Path, default=Path("icl_merged.h5"), help="Output HDF5 path.")
    parser.add_argument("--seq-len", type=int, required=True, help="Expected sequence length.")
    parser.add_argument("--num-features", type=int, required=True, help="Expected number of features.")
    parser.add_argument("--max-num-classes", type=int, required=True, help="Max number of classes.")
    parser.add_argument("--device", type=str, default="cpu", help="Device to load torch tensors.")
    parser.add_argument(
        "--allow-fewer-features",
        action="store_true",
        help="Allow batches with fewer feature columns and pad zeros at the end.",
    )
    parser.add_argument(
        "--max-category-cardinality",
        type=int,
        default=10,
        help="Maximum number of unique values to treat a feature as categorical.",
    )
    args = parser.parse_args()

    batch_files = sorted(args.input_dir.glob("batch_*.pt"))
    if not batch_files:
        raise SystemExit(f"No batch_*.pt files found in {args.input_dir}")

    tensors_X = []
    tensors_y = []
    tensors_num_features = []
    tensors_seq_len = []
    tensors_split = []
    category_masks = []

    total = 0
    for batch_path in batch_files:
        X, y, d, seq_lens, train_sizes = load_batch_as_tensor(batch_path, args.device)
        if X.shape[1] != args.seq_len:
            print(
                f"Skipping batch {batch_path} because seq len {X.shape[1]} "
                f"!= expected {args.seq_len}"
            )
            continue
        if X.shape[2] > args.num_features:
            print(
                f"Skipping batch {batch_path} because it has {X.shape[2]} features "
                f"but expected at most {args.num_features}"
            )
            continue
        if X.shape[2] < args.num_features:
            if not args.allow_fewer_features:
                print(
                    f"Skipping batch {batch_path} because it has {X.shape[2]} features "
                    f"but expected {args.num_features} (set --allow-fewer-features to pad)."
                )
                continue
            pad_cols = args.num_features - X.shape[2]
            X = F.pad(X, (0, pad_cols))
        batch_size = X.shape[0]
        masks_batch = []
        for i in range(batch_size):
            feature_count = int(d[i].item())
            mask = _encode_categorical_columns_inplace(
                X[i], feature_count, args.max_category_cardinality
            )
            masks_batch.append(mask)
        tensors_X.append(X)
        tensors_y.append(y)
        tensors_num_features.append(d)
        tensors_seq_len.append(seq_lens)
        tensors_split.append(train_sizes)
        category_masks.extend(masks_batch)
        total += batch_size
        print(f"Loaded {total} datasets", end="\r", flush=True)

    print(f"\nConcatenating {total} datasets in memory...")
    X_all = torch.cat(tensors_X, dim=0).numpy().astype(np.float32, copy=False)
    y_all = torch.cat(tensors_y, dim=0).numpy().astype(np.int32, copy=False)
    num_features = torch.cat(tensors_num_features, dim=0).numpy().astype(np.int32, copy=False)
    num_datapoints = torch.cat(tensors_seq_len, dim=0).numpy().astype(np.int32, copy=False)
    split_pos = torch.cat(tensors_split, dim=0).numpy().astype(np.int32, copy=False)
    category_mask_array = np.zeros((total, args.num_features), dtype=np.uint8)
    for idx, mask in enumerate(category_masks):
        length = mask.shape[0]
        category_mask_array[idx, :length] = mask

    args.output.parent.mkdir(parents=True, exist_ok=True)
    total_rows = X_all.shape[0]
    chunk_rows = min(4, total_rows) if total_rows > 0 else 1
    with h5py.File(args.output, "w") as h5:
        h5.create_dataset(
            "X",
            data=X_all,
            dtype="float32",
            compression="lzf",
            chunks=(chunk_rows, args.seq_len, args.num_features),
        )
        h5.create_dataset(
            "y",
            data=y_all,
            dtype="int32",
            compression="lzf",
            chunks=(chunk_rows, args.seq_len),
        )
        h5.create_dataset("num_features", data=num_features, dtype="int32")
        h5.create_dataset("num_datapoints", data=num_datapoints, dtype="int32")
        h5.create_dataset("single_eval_pos", data=split_pos, dtype="int32")
        h5.create_dataset(
            "feature_is_categorical",
            data=category_mask_array,
            dtype="uint8",
        )
        h5.create_dataset(
            "max_num_classes",
            data=np.array([args.max_num_classes], dtype="int32"),
        )
    print(f"Wrote merged prior to {args.output}")


if __name__ == "__main__":
    main()
