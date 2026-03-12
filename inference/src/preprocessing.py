from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd


@dataclass
class ColumnTransform:
    name: str
    kind: str
    fill_value: float
    category_to_index: dict[str, int] | None = None


class FlatPreprocessor:
    def __init__(self):
        self._feature_names: list[str] | None = None
        self._transforms: list[ColumnTransform] | None = None
        self._array_fill_values: np.ndarray | None = None

    @property
    def feature_names(self) -> list[str] | None:
        return self._feature_names

    def fit_transform(self, X) -> np.ndarray:
        if isinstance(X, pd.DataFrame):
            matrix, transforms = _fit_dataframe(X)
            self._feature_names = [transform.name for transform in transforms]
            self._transforms = transforms
            self._array_fill_values = None
            return matrix

        matrix = _coerce_array(X)
        fill_values = _compute_numeric_fill_values(matrix)
        self._feature_names = None
        self._transforms = None
        self._array_fill_values = fill_values
        return _fill_numeric_array(matrix, fill_values)

    def transform(self, X) -> np.ndarray:
        if self._transforms is not None:
            if not isinstance(X, pd.DataFrame):
                raise TypeError("Expected a pandas DataFrame for prediction.")
            return _transform_dataframe(X, self._transforms)

        if isinstance(X, pd.DataFrame):
            raise TypeError("Expected a numpy-compatible array for prediction.")
        if self._array_fill_values is None:
            raise RuntimeError("Preprocessor has not been fit.")
        matrix = _coerce_array(X)
        if matrix.shape[1] != len(self._array_fill_values):
            raise ValueError(
                f"Expected {len(self._array_fill_values)} features, got {matrix.shape[1]}."
            )
        return _fill_numeric_array(matrix, self._array_fill_values)


def dataframe_to_xy(df: pd.DataFrame, target: str) -> tuple[pd.DataFrame, np.ndarray]:
    if target not in df.columns:
        raise ValueError(f"Target column '{target}' not found.")
    y = df[target].to_numpy(copy=True)
    X = df.drop(columns=[target])
    return X, y


def _fit_dataframe(df: pd.DataFrame) -> tuple[np.ndarray, list[ColumnTransform]]:
    transformed_columns = []
    transforms: list[ColumnTransform] = []

    for column_name in df.columns:
        series = df[column_name]
        if pd.api.types.is_numeric_dtype(series) or pd.api.types.is_bool_dtype(series):
            numeric = pd.to_numeric(series, errors="coerce").astype(np.float32)
            fill_value = _nanmedian_or_zero(numeric.to_numpy(dtype=np.float32, copy=False))
            transformed_columns.append(
                numeric.fillna(fill_value).to_numpy(dtype=np.float32, copy=False)
            )
            transforms.append(
                ColumnTransform(
                    name=column_name,
                    kind="numeric",
                    fill_value=float(fill_value),
                )
            )
            continue

        values = series.astype("string")
        unique_values = pd.unique(values.dropna())
        mapping = {str(value): idx for idx, value in enumerate(unique_values)}
        encoded = values.map(lambda value: mapping.get(str(value), -1) if pd.notna(value) else -1)
        transformed_columns.append(encoded.to_numpy(dtype=np.float32, copy=False))
        transforms.append(
            ColumnTransform(
                name=column_name,
                kind="category",
                fill_value=-1.0,
                category_to_index=mapping,
            )
        )

    matrix = np.column_stack(transformed_columns).astype(np.float32, copy=False)
    return matrix, transforms


def _transform_dataframe(df: pd.DataFrame, transforms: list[ColumnTransform]) -> np.ndarray:
    expected = [transform.name for transform in transforms]
    missing = [name for name in expected if name not in df.columns]
    if missing:
        raise ValueError(f"Missing feature columns: {missing}")

    extras = [name for name in df.columns if name not in expected]
    if extras:
        raise ValueError(f"Unexpected feature columns: {extras}")

    transformed_columns = []
    for transform in transforms:
        series = df[transform.name]
        if transform.kind == "numeric":
            numeric = pd.to_numeric(series, errors="coerce").astype(np.float32)
            transformed_columns.append(
                numeric.fillna(transform.fill_value).to_numpy(dtype=np.float32, copy=False)
            )
            continue

        values = series.astype("string")
        mapping = transform.category_to_index or {}
        encoded = values.map(lambda value: mapping.get(str(value), -1) if pd.notna(value) else -1)
        transformed_columns.append(encoded.to_numpy(dtype=np.float32, copy=False))

    return np.column_stack(transformed_columns).astype(np.float32, copy=False)


def _coerce_array(X) -> np.ndarray:
    array = np.asarray(X, dtype=np.float32)
    if array.ndim != 2:
        raise ValueError(f"Expected a 2D feature matrix, got shape {array.shape}.")
    return array


def _compute_numeric_fill_values(matrix: np.ndarray) -> np.ndarray:
    fill_values = np.zeros(matrix.shape[1], dtype=np.float32)
    for index in range(matrix.shape[1]):
        fill_values[index] = _nanmedian_or_zero(matrix[:, index])
    return fill_values


def _fill_numeric_array(matrix: np.ndarray, fill_values: np.ndarray) -> np.ndarray:
    filled = matrix.copy()
    for index, fill_value in enumerate(fill_values):
        mask = np.isnan(filled[:, index])
        if mask.any():
            filled[mask, index] = fill_value
    return filled.astype(np.float32, copy=False)


def _nanmedian_or_zero(values: np.ndarray) -> float:
    median = np.nanmedian(values)
    if np.isnan(median):
        return 0.0
    return float(median)
