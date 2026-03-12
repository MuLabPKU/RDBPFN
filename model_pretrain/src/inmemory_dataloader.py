from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import List, Sequence

import h5py
import numpy as np
import torch
from torch.utils.data import Dataset

logger = logging.getLogger(__name__)


class InMemoryDataset(Dataset):
    """
    Map-style dataset that loads an entire HDF5 prior dump into host memory.
    Each sample is returned as a dictionary matching the structure consumed
    by the training loop, so callers can wrap this dataset with a regular
    PyTorch DataLoader.
    """

    def __init__(self, filename: str, start_index: int = 0, output_log: bool = False):
        self.filename = filename
        self.start_index = int(max(0, start_index))
        self.output_log = output_log
        self._file = h5py.File(filename, "r")
        # Keep datasets as file-backed handles to avoid copying the full dump into RAM.
        self.X = self._file["X"]
        self.y = self._file["y"]
        self.num_features = np.array(self._file["num_features"])
        self.num_datapoints = np.array(self._file["num_datapoints"])
        self.single_eval_pos = np.array(self._file["single_eval_pos"])
        self.num_available_features = (
            np.array(self._file["num_available_features"])
            if "num_available_features" in self._file
            else None
        )
        self.feature_is_categorical = (
            np.array(self._file["feature_is_categorical"])
            if "feature_is_categorical" in self._file
            else None
        )

        self.dataset_size = self.X.shape[0]
        if self.output_log:
            logger.info(
                "Ready with %d samples from %s (X=%s, y=%s)",
                self.dataset_size,
                filename,
                self.X.shape,
                self.y.shape,
            )

    def close(self):
        file_handle = getattr(self, "_file", None)
        if file_handle is not None:
            try:
                file_handle.close()
            finally:
                self._file = None

    def __del__(self):
        try:
            self.close()
        except Exception:
            pass

    def __len__(self) -> int:
        return self.dataset_size

    def __getitem__(self, index: int) -> dict:
        if self.dataset_size == 0:
            raise IndexError("Dataset is empty.")

        idx = (int(index) + self.start_index) % self.dataset_size

        num_feat = int(self.num_features[idx])
        num_rows = int(self.num_datapoints[idx])

        sample = {
            "x": torch.from_numpy(self.X[idx, :num_rows, :num_feat]),
            "y": torch.from_numpy(self.y[idx, :num_rows]),
            "train_test_split_index": int(self.single_eval_pos[idx]),
            "num_features": torch.tensor(num_feat, dtype=torch.long),
            "num_available_features": torch.tensor(
                (
                    int(self.num_available_features[idx])
                    if self.num_available_features is not None
                    else num_feat
                ),
                dtype=torch.long,
            ),
        }

        # if self.feature_is_categorical is not None:
        #     mask = self.feature_is_categorical[idx, :num_feat].astype(np.float32)
        #     sample["category_mask"] = torch.from_numpy(mask.copy())

        return sample


def collate_batch(batch: List[dict]) -> dict:
    """
    Pad-and-stack collate to handle different sequence lengths / feature counts.
    """

    batch_size = len(batch)
    max_rows = max(item["x"].shape[0] for item in batch)
    max_features = max(item["x"].shape[1] for item in batch)

    dtype_x = batch[0]["x"].dtype
    dtype_y = batch[0]["y"].dtype
    x_out = torch.zeros(batch_size, max_rows, max_features, dtype=dtype_x)
    y_out = torch.zeros(batch_size, max_rows, dtype=dtype_y)
    num_features = torch.zeros(batch_size, dtype=torch.long)
    num_available = torch.zeros(batch_size, dtype=torch.long)

    # has_category_mask = all("category_mask" in item for item in batch)
    # category_mask_out = None
    # if has_category_mask:
    #     category_mask_out = torch.zeros(
    #         batch_size,
    #         max_features,
    #         dtype=batch[0]["category_mask"].dtype,
    #     )

    for idx, sample in enumerate(batch):
        rows, feats = sample["x"].shape
        x_out[idx, :rows, :feats] = sample["x"]
        y_out[idx, :rows] = sample["y"]
        num_features[idx] = sample["num_features"]
        num_available[idx] = sample["num_available_features"]

        # if has_category_mask:
        #     category_mask_out[idx, :feats] = sample["category_mask"]

    collated = {
        "x": x_out,
        "y": y_out,
        "train_test_split_index": batch[0]["train_test_split_index"],
        "num_features": num_features,
        "num_available_features": num_available,
    }

    # if has_category_mask and category_mask_out is not None:
    #     collated["category_mask"] = category_mask_out

    return collated


@dataclass(frozen=True)
class _SamplePointer:
    dataset_idx: int
    sample_idx: int


class JointDataset(Dataset):
    """
    Dataset that joins multiple in-memory datasets using weighted sampling.
    Sampling decisions are made at the step level so that every batch draws
    `batch_size` sequential samples from the selected dataset.
    """

    def __init__(
        self,
        datasets: Sequence[Dataset],
        steps_per_epoch: int,
        batch_size: int = 1,
        weights: Sequence[float] | None = None,
        seed: int = 0,
        shard_id: int = 0,
        num_shards: int = 1,
    ):
        if not datasets:
            raise ValueError("JointDataset requires at least one dataset.")
        if steps_per_epoch <= 0:
            raise ValueError("steps_per_epoch must be positive.")
        if batch_size <= 0:
            raise ValueError("batch_size must be positive.")

        self.datasets = list(datasets)
        self.batch_size = int(batch_size)
        self.global_steps = int(steps_per_epoch)
        self.seed = int(seed)
        self.epoch = 0
        self.num_shards = max(1, int(num_shards))
        self.shard_id = int(shard_id) % self.num_shards
        self.steps_per_shard = 0

        lengths = [len(ds) for ds in self.datasets]
        if any(length <= 0 for length in lengths):
            raise ValueError("All datasets must contain at least one sample.")

        if weights is None:
            probs = np.ones(len(self.datasets), dtype=np.float64)
        else:
            probs = np.asarray(weights, dtype=np.float64)
            if probs.shape != (len(self.datasets),):
                raise ValueError("weights must match number of datasets.")
            probs = np.clip(probs, 0.0, None)
            if probs.sum() == 0:
                probs[:] = 1.0
        self.probs = probs / probs.sum()
        self._schedule: list[_SamplePointer] = []

        self._regenerate_schedule()

    def _regenerate_schedule(self):
        rng = np.random.default_rng(self.seed + self.epoch)
        dataset_choices = rng.choice(
            len(self.datasets),
            size=self.global_steps,
            p=self.probs,
        )

        pointers = np.zeros(len(self.datasets), dtype=np.int64)
        schedule: list[_SamplePointer] = []
        shard_positions = range(self.shard_id, self.global_steps, self.num_shards)

        for pos in shard_positions:
            dataset_idx = int(dataset_choices[pos])
            dataset_len = len(self.datasets[dataset_idx])

            for _ in range(self.batch_size):
                sample_idx = int(pointers[dataset_idx])
                schedule.append(_SamplePointer(dataset_idx, sample_idx))
                pointers[dataset_idx] = (sample_idx + 1) % dataset_len

        self._schedule = schedule
        self.steps_per_shard = len(schedule) // self.batch_size

    def set_epoch(self, epoch: int):
        self.epoch = int(epoch)
        self._regenerate_schedule()

    def __len__(self) -> int:
        return len(self._schedule)

    def __getitem__(self, index: int) -> dict:
        if not self._schedule:
            raise IndexError("JointDataset has no scheduled samples.")

        pointer = self._schedule[int(index) % len(self._schedule)]
        sample = self.datasets[pointer.dataset_idx][pointer.sample_idx]
        sample["source_id"] = pointer.dataset_idx
        return sample
