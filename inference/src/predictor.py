from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from .checkpoint import load_model
from .metrics import summarize_classification
from .preprocessing import FlatPreprocessor, dataframe_to_xy


MAX_CONTEXT_SIZE = 1024
MAX_PREDICT_CHUNK_SIZE = 2000
MAX_ENSEMBLE_CONTEXT_SOURCE_SIZE = 10000


class RDBPFNClassifier:
    def __init__(self, model: torch.nn.Module, device: torch.device):
        self.model = model.to(device)
        self.device = device
        self.preprocessor = FlatPreprocessor()
        self.classes_: np.ndarray | None = None
        self._class_to_index: dict[object, int] | None = None
        self.context_ensembles_: list[list[tuple[np.ndarray, np.ndarray]]] | None = None
        self.model.eval()

    @classmethod
    def from_pretrained(
        cls,
        identifier: str | Path | None = None,
        *,
        device: str | torch.device | None = None,
    ) -> "RDBPFNClassifier":
        resolved_device = _resolve_device(device)
        model = load_model(identifier, device=resolved_device)
        return cls(model=model, device=resolved_device)

    def fit(self, X, y=None, *, target: str | None = None) -> "RDBPFNClassifier":
        if target is not None:
            X, y = dataframe_to_xy(X, target)
        if y is None:
            raise ValueError("fit() requires either y or target= for dataframe input.")
        X_train = self.preprocessor.fit_transform(X)
        encoded, classes, class_to_index = _fit_label_encoder(y)
        self.classes_ = classes
        self._class_to_index = class_to_index
        self.context_ensembles_ = _build_class_context_ensembles(
            X_train,
            encoded,
            num_classes=len(classes),
            max_context_size=MAX_CONTEXT_SIZE,
            max_ensemble_source_size=MAX_ENSEMBLE_CONTEXT_SOURCE_SIZE,
        )
        return self

    def predict_proba(self, X, *, chunk_size: int | None = None) -> np.ndarray:
        self._ensure_fitted()
        X_test = self.preprocessor.transform(X)
        if chunk_size is not None and chunk_size != MAX_PREDICT_CHUNK_SIZE:
            raise ValueError(
                f"chunk_size is fixed to {MAX_PREDICT_CHUNK_SIZE} in inference v1."
            )
        if len(X_test) <= MAX_PREDICT_CHUNK_SIZE:
            return self._predict_proba_chunk(X_test)

        probabilities = []
        for start in range(0, len(X_test), MAX_PREDICT_CHUNK_SIZE):
            stop = start + MAX_PREDICT_CHUNK_SIZE
            probabilities.append(self._predict_proba_chunk(X_test[start:stop]))
        return np.concatenate(probabilities, axis=0)

    def predict(self, X, *, chunk_size: int | None = None) -> np.ndarray:
        probabilities = self.predict_proba(X, chunk_size=chunk_size)
        indices = probabilities.argmax(axis=1)
        if self.classes_ is None:
            raise RuntimeError("Model has not been fit.")
        return self.classes_[indices]

    def evaluate(
        self,
        X_train,
        y_train,
        X_test,
        y_test,
        *,
        metric: str = "auroc",
        chunk_size: int | None = None,
    ) -> dict[str, float]:
        self.fit(X_train, y_train)
        probabilities = self.predict_proba(X_test, chunk_size=chunk_size)
        pred_indices = probabilities.argmax(axis=1)
        y_true = self._transform_labels(y_test)
        return summarize_classification(
            y_true,
            probabilities,
            pred_indices,
            primary_metric=metric,
        )

    def _predict_proba_chunk(self, X_test: np.ndarray) -> np.ndarray:
        self._ensure_fitted()
        if self.context_ensembles_ is None or self.classes_ is None:
            raise RuntimeError("Model has not been fit.")

        if len(self.classes_) == 2:
            ensemble_probs = []
            for X_train_context, y_train_context in self.context_ensembles_[0]:
                ensemble_probs.append(
                    self._predict_proba_with_context(
                        X_train_context,
                        y_train_context,
                        X_test,
                    )
                )
            return np.mean(ensemble_probs, axis=0)

        class_scores = []
        for class_context_ensemble in self.context_ensembles_:
            ensemble_positive_probs = []
            for X_train_context, y_train_context in class_context_ensemble:
                probs = self._predict_proba_with_context(
                    X_train_context,
                    y_train_context,
                    X_test,
                )
                ensemble_positive_probs.append(probs[:, 1])
            class_scores.append(np.mean(ensemble_positive_probs, axis=0))

        scores = np.column_stack(class_scores).astype(np.float32, copy=False)
        row_sums = scores.sum(axis=1, keepdims=True)
        zero_mask = row_sums.squeeze(1) <= 0
        row_sums = np.where(row_sums <= 0, 1.0, row_sums)
        probabilities = scores / row_sums
        if np.any(zero_mask):
            probabilities[zero_mask] = 1.0 / probabilities.shape[1]
        return probabilities

    def _predict_proba_with_context(
        self,
        X_train_context: np.ndarray,
        y_train_context: np.ndarray,
        X_test: np.ndarray,
    ) -> np.ndarray:
        x = np.concatenate((X_train_context, X_test), axis=0)
        y = y_train_context
        with torch.no_grad():
            x_tensor = (
                torch.from_numpy(x).unsqueeze(0).to(torch.float32).to(self.device)
            )
            y_tensor = (
                torch.from_numpy(y).unsqueeze(0).to(torch.float32).to(self.device)
            )
            out = self.model(
                (x_tensor, y_tensor),
                train_test_split_index=len(X_train_context),
            ).squeeze(0)
            if self.classes_ is None:
                raise RuntimeError("Model has not been fit.")
            out = out[:, : len(self.classes_)]
            probabilities = F.softmax(out, dim=1)
            return probabilities.cpu().numpy()

    def _transform_labels(self, y) -> np.ndarray:
        if self._class_to_index is None:
            raise RuntimeError("Model has not been fit.")
        values = np.asarray(y)
        try:
            encoded = np.array(
                [
                    self._class_to_index[
                        value.item() if hasattr(value, "item") else value
                    ]
                    for value in values
                ],
                dtype=np.int64,
            )
        except KeyError as exc:
            raise ValueError(
                f"Unknown label encountered during evaluation: {exc.args[0]}"
            ) from exc
        return encoded

    def _ensure_fitted(self) -> None:
        if self.context_ensembles_ is None or self.classes_ is None:
            raise RuntimeError("Call fit() before prediction.")


def _fit_label_encoder(y) -> tuple[np.ndarray, np.ndarray, dict[object, int]]:
    values = np.asarray(y)
    if values.ndim != 1:
        values = values.reshape(-1)
    unique_values = np.unique(values)
    class_to_index = {
        value.item() if hasattr(value, "item") else value: index
        for index, value in enumerate(unique_values)
    }
    encoded = np.array(
        [class_to_index[value.item() if hasattr(value, "item") else value] for value in values],
        dtype=np.int64,
    )
    classes = unique_values.copy()
    return encoded, classes, class_to_index


def _build_class_context_ensembles(
    X_train: np.ndarray,
    y_train_encoded: np.ndarray,
    *,
    num_classes: int,
    max_context_size: int,
    max_ensemble_source_size: int,
) -> list[list[tuple[np.ndarray, np.ndarray]]]:
    if num_classes == 2:
        return [
            _build_context_ensemble(
                X_train,
                y_train_encoded,
                max_context_size=max_context_size,
                max_ensemble_source_size=max_ensemble_source_size,
            )
        ]

    context_ensembles = []
    for class_index in range(num_classes):
        binary_labels = (y_train_encoded == class_index).astype(np.int64, copy=False)
        context_ensembles.append(
            _build_context_ensemble(
                X_train,
                binary_labels,
                max_context_size=max_context_size,
                max_ensemble_source_size=max_ensemble_source_size,
            )
        )
    return context_ensembles


def _build_context_ensemble(
    X_train: np.ndarray,
    y_train: np.ndarray,
    *,
    max_context_size: int,
    max_ensemble_source_size: int,
) -> list[tuple[np.ndarray, np.ndarray]]:
    if len(X_train) <= max_context_size:
        return [(X_train, y_train)]

    if len(X_train) < max_ensemble_source_size:
        ensemble_count = _compute_ensemble_count(len(X_train), max_context_size)
        rng = np.random.default_rng(0)
        return [
            _sample_context_subset(
                X_train,
                y_train,
                max_context_size=max_context_size,
                rng=rng,
            )
            for _ in range(ensemble_count)
        ]

    rng = np.random.default_rng(0)
    return [
        _sample_context_subset(
            X_train,
            y_train,
            max_context_size=max_context_size,
            rng=rng,
        )
    ]


def _compute_ensemble_count(num_rows: int, max_context_size: int) -> int:
    if num_rows <= max_context_size:
        return 1
    return int(np.ceil(num_rows / max_context_size))


def _sample_context_subset(
    X_train: np.ndarray,
    y_train: np.ndarray,
    *,
    max_context_size: int,
    rng: np.random.Generator,
) -> tuple[np.ndarray, np.ndarray]:
    class_values, class_counts = np.unique(y_train, return_counts=True)
    allocations = np.floor(
        class_counts.astype(np.float64) * max_context_size / len(y_train)
    ).astype(np.int64)
    allocations = np.maximum(allocations, 1)
    allocations = np.minimum(allocations, class_counts)

    while allocations.sum() > max_context_size:
        reducible = np.where(allocations > 1)[0]
        if reducible.size == 0:
            break
        target = reducible[np.argmax(allocations[reducible])]
        allocations[target] -= 1

    while allocations.sum() < max_context_size:
        expandable = np.where(allocations < class_counts)[0]
        if expandable.size == 0:
            break
        target = expandable[np.argmax(class_counts[expandable] - allocations[expandable])]
        allocations[target] += 1

    selected_indices = []
    for class_value, allocation in zip(class_values, allocations):
        class_indices = np.flatnonzero(y_train == class_value)
        if allocation >= len(class_indices):
            selected_indices.append(class_indices)
            continue
        sampled = rng.choice(class_indices, size=allocation, replace=False)
        selected_indices.append(np.sort(sampled))

    kept = np.concatenate(selected_indices)
    kept.sort()
    return X_train[kept], y_train[kept]


def _resolve_device(device: str | torch.device | None) -> torch.device:
    if device is None:
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(device)
