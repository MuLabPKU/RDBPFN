from __future__ import annotations


import pandas as pd
import math
import random
from enum import Enum
from typing import Dict, Any, List, Tuple, Union
import os
import datetime
import warnings

import networkx as nx
import numpy as np
import torch
from torch import nn
import json
import yaml
from src.table_def.yaml_utils import load_pyd, save_pyd

from src.prior.mlp_scm import MLPSCM, MASK_TYPE
from src.prior.row_gnn import RowGraphBuilder, RowGNNRunner
from src.table_def.dataset_meta import (
    DBBRDBDatasetMeta,
    DBBTableSchema,
    DBBColumnSchema,
    DBBColumnDType,
    DBBTableDataFormat,
    DTYPE_EXTRA_FIELDS,
)
from src.prior.hp_sampling import HpSamplerList
from src.prior.prior_config import DEFAULT_SAMPLED_HP


# Utility functions for robust data preprocessing
def torch_nanstd(input, dim=None, keepdim=False, ddof=0, *, dtype=None) -> torch.Tensor:
    """Calculates the standard deviation of a tensor, ignoring NaNs, using NumPy internally.

    Parameters
    ----------
    input : Tensor
        The input tensor.
    dim : int or tuple[int], optional
        The dimension or dimensions to reduce. Defaults to None (reduce all dimensions).
    keepdim : bool, optional
        Whether the output tensor has `dim` retained or not. Defaults to False.
    ddof : int, optional
        Delta Degrees of Freedom.
    dtype : torch.dtype, optional
        The desired data type of returned tensor. Defaults to None.

    Returns
    -------
    Tensor
        The standard deviation.
    """
    device = input.device
    np_input = input.cpu().numpy()

    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=RuntimeWarning)
        std = np.nanstd(np_input, axis=dim, dtype=dtype, keepdims=keepdim, ddof=ddof)

    return torch.from_numpy(std).to(dtype=torch.float, device=device)


def outlier_removing(input: torch.Tensor, threshold: float = 4.0) -> torch.Tensor:
    """Clamps outliers in the input tensor based on a specified number of standard deviations.

    Parameters
    ----------
    input : Tensor
        Input tensor of shape (T,) or (T, H).
    threshold : float, optional, default=4.0
        Number of standard deviations to use as the cutoff.

    Returns
    -------
    Tensor
        The tensor with outliers clamped.
    """
    # Ensure input is at least 1D
    if input.ndim == 0:
        return input

    # First stage: Identify outliers using initial statistics
    mean = torch.nanmean(input, dim=0)
    std = torch_nanstd(input, dim=0, ddof=1 if input.shape[0] > 1 else 0).clip(min=1e-6)
    cut_off = std * threshold
    lower, upper = mean - cut_off, mean + cut_off

    # Create mask for non-outlier, non-NaN values
    mask = (lower <= input) & (input <= upper) & ~torch.isnan(input)

    # Second pass using only non-outlier values for mean/std
    masked_input = torch.where(mask, input, torch.nan)
    masked_mean = torch.nanmean(masked_input, dim=0)
    masked_std = torch_nanstd(
        masked_input, dim=0, ddof=1 if input.shape[0] > 1 else 0
    ).clip(min=1e-6)

    # Handle cases where a column had <= 1 valid value after masking
    masked_mean = torch.where(torch.isnan(masked_mean), mean, masked_mean)
    masked_std = torch.where(torch.isnan(masked_std), torch.zeros_like(std), masked_std)

    # Recalculate cutoff with robust estimates
    cut_off = masked_std * threshold
    lower, upper = masked_mean - cut_off, masked_mean + cut_off

    # Replace NaN bounds with +/- inf
    lower = torch.nan_to_num(lower, nan=-torch.inf)
    upper = torch.nan_to_num(upper, nan=torch.inf)

    return input.clamp(min=lower, max=upper)


# Generation schema classes
class TableGenerationSchema:
    """Schema information for how a table was generated."""

    def __init__(
        self,
        table_name: str,
        generation_type: str,
        parent_tables: List[str] = None,
        uses_timestamp: bool = False,
        eta: float = None,
        num_parents: int = 0,
        is_timestamp_table: bool = False,
        scm_params: Dict = None,
        masks: Dict = None,
    ):
        self.table_name = table_name
        self.generation_type = (
            generation_type  # "self_generated", "parent_based", "timestamp_based"
        )
        self.parent_tables = parent_tables or []
        self.uses_timestamp = uses_timestamp
        self.eta = eta
        self.num_parents = num_parents
        self.is_timestamp_table = is_timestamp_table
        self.scm_params = scm_params or {}
        self.masks = masks or {}

    def to_dict(self) -> Dict:
        """Convert to dictionary for YAML serialization."""
        return {
            "table_name": self.table_name,
            "generation_type": self.generation_type,
            "parent_tables": self.parent_tables,
            "uses_timestamp": self.uses_timestamp,
            "eta": self.eta,
            "num_parents": self.num_parents,
            "is_timestamp_table": self.is_timestamp_table,
            "scm_params": self.scm_params,
            "masks": self.masks,
        }


class TaskGenerationSchema:
    """Schema information for how a task was generated."""

    def __init__(
        self,
        task_name: str,
        primary_table_name: str,
        primary_table_generation_schema: TableGenerationSchema,
        target_column: str,
        task_type: str,
    ):
        self.task_name = task_name
        self.primary_table_name = primary_table_name
        self.primary_table_generation_schema = primary_table_generation_schema
        self.target_column = target_column
        self.task_type = task_type

    def to_dict(self) -> Dict:
        """Convert to dictionary for YAML serialization."""
        return {
            "task_name": self.task_name,
            "primary_table_name": self.primary_table_name,
            "primary_table_generation_schema": self.primary_table_generation_schema.to_dict(),
            "target_column": self.target_column,
            "task_type": self.task_type,
        }


# This file is used to generate tables and RDBs.
# To implement this, we firstly define the table class and RDB class.
# Then we use a RDB generator to generate RDBs.


class DataType(Enum):
    FLOAT = "float"
    CATEGORICAL = "categorical"
    TIMESTAMP = "timestamp"
    PRIMARY_KEY = "primary_key"
    FOREIGN_KEY = "foreign_key"


def convert_to_dbb_column_dtype(data_type: DataType) -> DBBColumnDType:
    if data_type == DataType.FLOAT:
        return DBBColumnDType.float_t
    elif data_type == DataType.CATEGORICAL:
        return DBBColumnDType.category_t
    elif data_type == DataType.TIMESTAMP:
        return DBBColumnDType.datetime_t
    elif data_type == DataType.PRIMARY_KEY:
        return DBBColumnDType.primary_key
    elif data_type == DataType.FOREIGN_KEY:
        return DBBColumnDType.foreign_key
    else:
        raise ValueError(f"Invalid data type: {data_type}")


class DataTypeConfig:
    """Configuration class for data type conversion parameters."""

    def __init__(self, data_type: DataType, **kwargs):
        self.data_type = data_type
        self.config = kwargs

    @classmethod
    def float_config(cls, normalize: bool = False, **kwargs) -> "DataTypeConfig":
        """Create float data type configuration.

        Parameters
        ----------
        normalize : bool, default=False
            Whether to normalize float values to standard scale
        """
        config = {"normalize": normalize}
        config.update(kwargs)
        return cls(DataType.FLOAT, **config)

    @classmethod
    def categorical_config(
        cls,
        num_categories: int = None,
        strategy: str = "random",  # "quantile", "rank", "value", "random"
        balanced: bool = False,
        ordered_prob: float = 0.0,  # Probability of keeping natural order
        max_categories: int = 10,
        **kwargs,
    ) -> "DataTypeConfig":
        """Create categorical data type configuration with flexible options.

        Parameters
        ----------
        num_categories : int, optional
            Number of categories. If None, uses gamma distribution
        strategy : str, default="quantile"
            Conversion strategy: "quantile", "rank", "value", or "random"
            - "quantile": Fixed boundaries at quantiles (deterministic)
            - "rank": Random boundaries sampled from data points
            - "value": Random boundaries from normal distribution
            - "random": Randomly selects one of the above strategies
        balanced : bool, default=False
            For binary categories, ensure balanced distribution
        ordered_prob : float, default=0.3
            Probability of keeping natural class order
        max_categories : int, default=10
            Maximum number of categories when auto-determining
        """
        if num_categories is None:
            # Use gamma distribution for more natural category count selection
            num_categories = min(
                max(round(random.gammavariate(1, 10)), 2), max_categories
            )
        config = {
            "num_categories": num_categories,
            "strategy": strategy,
            "balanced": balanced,
            "ordered_prob": ordered_prob,
            "max_categories": max_categories,
        }
        config.update(kwargs)
        return cls(DataType.CATEGORICAL, **config)

    @classmethod
    def timestamp_config(cls, **kwargs) -> "DataTypeConfig":
        """Create timestamp data type configuration."""
        return cls(DataType.TIMESTAMP, **kwargs)

    @classmethod
    def primary_key_config(cls, **kwargs) -> "DataTypeConfig":
        """Create primary key data type configuration."""
        return cls(DataType.PRIMARY_KEY, **kwargs)

    @classmethod
    def foreign_key_config(cls, parent_table: str = None, **kwargs) -> "DataTypeConfig":
        """Create foreign key data type configuration."""
        config = {"parent_table": parent_table}
        config.update(kwargs)
        return cls(DataType.FOREIGN_KEY, **config)


class ColumnDataProcessor:
    def __init__(self, use_preprocessing: bool = True, outlier_threshold: float = 4.0):
        """Initialize column data processor.

        Parameters
        ----------
        use_preprocessing : bool, default=True
            Whether to apply outlier removal preprocessing
        outlier_threshold : float, default=4.0
            Number of standard deviations for outlier detection
        """
        self.use_preprocessing = use_preprocessing
        self.outlier_threshold = outlier_threshold

    def process_data(
        self, raw_data: torch.Tensor, data_type_config: DataTypeConfig
    ) -> torch.Tensor:
        """
        Generate data based on data type configuration.

        Parameters:
        -----------
        raw_data : torch.Tensor
            Raw continuous data
        data_type_config : DataTypeConfig
            Configuration specifying data type and parameters

        Returns:
        --------
        torch.Tensor
            Processed data according to configuration
        """
        data_type = data_type_config.data_type
        config = data_type_config.config

        # For primary key, currently we require the data strictly increasing from [0, num_rows - 1]
        if data_type == DataType.PRIMARY_KEY:
            assert torch.all(raw_data == torch.arange(raw_data.shape[0]).unsqueeze(1))
            return raw_data
        # For foreign key, no modification is needed.
        if data_type == DataType.FOREIGN_KEY:
            return raw_data

        # Apply preprocessing for numeric types (except timestamp which has its own handling)
        if self.use_preprocessing and data_type != DataType.TIMESTAMP:
            raw_data = self._preprocess(raw_data, data_type_config)

        # Type-specific conversion
        if data_type == DataType.FLOAT:
            normalize = config.get("normalize", False)
            if normalize:
                return self._normalize_float(raw_data)
            return raw_data
        elif data_type == DataType.CATEGORICAL:
            return self._convert_to_categorical(raw_data, config)
        elif data_type == DataType.TIMESTAMP:
            return self._convert_to_datetime64(raw_data)
        else:
            raise ValueError(f"Invalid data type: {data_type}")

    def _preprocess(
        self, raw_data: torch.Tensor, data_type_config: DataTypeConfig
    ) -> torch.Tensor:
        """Preprocess raw data with outlier removal.

        Parameters
        ----------
        raw_data : torch.Tensor
            Raw data tensor
        data_type_config : DataTypeConfig
            Configuration for this column

        Returns
        -------
        torch.Tensor
            Preprocessed data with outliers handled
        """
        # For categorical, be more conservative with outlier removal
        if data_type_config.data_type == DataType.CATEGORICAL:
            threshold = 3.0  # Less aggressive
        else:
            threshold = self.outlier_threshold

        # Add dimension if 1D for outlier_removing function
        if raw_data.ndim == 1:
            processed = outlier_removing(
                raw_data.unsqueeze(-1), threshold=threshold
            ).squeeze(-1)
        else:
            processed = outlier_removing(raw_data, threshold=threshold)

        return processed

    def _normalize_float(self, raw_data: torch.Tensor) -> torch.Tensor:
        """Normalize float data to standard scale.

        Parameters
        ----------
        raw_data : torch.Tensor
            Raw float data

        Returns
        -------
        torch.Tensor
            Normalized float data
        """
        finite_mask = torch.isfinite(raw_data)
        if not torch.any(finite_mask):
            return torch.zeros_like(raw_data)

        finite_data = raw_data[finite_mask]
        mean = torch.mean(finite_data)
        std = torch.std(finite_data).clip(min=1e-6)

        normalized = (raw_data - mean) / std
        return torch.clip(normalized, min=-100, max=100)

    def _convert_to_categorical(
        self, raw_data: torch.Tensor, config: dict
    ) -> torch.Tensor:
        """
        Convert continuous data to categorical data with multiple strategies.

        Parameters:
        -----------
        raw_data : torch.Tensor
            Continuous data tensor
        config : dict
            Configuration dictionary with keys:
            - num_categories: Number of categories (auto-determined if None)
            - strategy: Conversion strategy ("quantile", "rank", "value", "random")
            - balanced: For binary, use median-based balancing
            - ordered_prob: Probability of keeping natural order

        Returns:
        --------
        torch.Tensor
            Categorical data with integer labels
        """
        num_categories = config.get("num_categories", None)
        if num_categories is None:
            # Use gamma distribution for natural category count
            num_categories = min(
                max(round(random.gammavariate(1, 10)), 2),
                config.get("max_categories", 10),
            )

        # Special case: balanced binary
        if num_categories == 2 and config.get("balanced", False):
            finite_mask = torch.isfinite(raw_data)
            if torch.any(finite_mask):
                median = torch.median(raw_data[finite_mask])
                return (raw_data > median).long()
            else:
                return torch.zeros_like(raw_data, dtype=torch.long)

        # Choose conversion strategy
        strategy = config.get("strategy", "random")

        # Random strategy selection
        if strategy == "random":
            strategy = random.choice(["quantile", "rank", "value"])

        if strategy == "quantile":
            categorical = self._quantile_binning(raw_data, num_categories)
        elif strategy == "rank":
            categorical = self._rank_based_binning(raw_data, num_categories)
        elif strategy == "value":
            categorical = self._value_based_binning(raw_data, num_categories)
        else:
            categorical = self._quantile_binning(raw_data, num_categories)

        # Optional: permute classes for diversity
        ordered_prob = config.get("ordered_prob", 0.0)
        if ordered_prob == 0.0 or random.random() > ordered_prob:
            categorical = self._permute_classes(categorical, num_categories)

        return categorical

    def _permute_classes(
        self, classes: torch.Tensor, num_categories: int
    ) -> torch.Tensor:
        """Randomly permute class labels.

        Parameters
        ----------
        classes : torch.Tensor
            Class labels
        num_categories : int
            Number of categories

        Returns
        -------
        torch.Tensor
            Permuted class labels
        """
        unique_vals = torch.unique(classes)
        num_unique = len(unique_vals)

        if num_unique <= 1:
            return classes

        device = classes.device
        perm = torch.randperm(num_unique, device=device)

        # Create a mapping from old to new classes
        permuted = classes.clone()
        for i, val in enumerate(unique_vals):
            permuted[classes == val] = perm[i]

        return permuted

    def _quantile_binning(
        self, raw_data: torch.Tensor, num_categories: int
    ) -> torch.Tensor:
        """
        Convert continuous data to categorical using quantile-based binning.
        This ensures roughly equal distribution across categories.
        """
        # Clamp extreme values to prevent numerical issues
        # Use finite values only for min/max computation
        finite_mask = torch.isfinite(raw_data)
        if not torch.any(finite_mask):
            # All values are NaN/Inf - return zeros
            return torch.zeros_like(raw_data, dtype=torch.long)

        # Replace NaN/Inf with median of finite values
        finite_data = raw_data[finite_mask]
        if len(finite_data) > 0:
            median_val = torch.median(finite_data)
            clamped_data = torch.where(finite_mask, raw_data, median_val)
        else:
            clamped_data = torch.zeros_like(raw_data)

        # Further clamp to reasonable range to prevent quantile issues
        # Use 99.9th percentile to handle outliers
        if len(finite_data) > 1:
            clamped_data = torch.clamp(clamped_data, min=-10000, max=10000)

        # Calculate quantile thresholds
        quantiles = torch.linspace(0, 1, num_categories + 1)
        try:
            thresholds = torch.quantile(clamped_data, quantiles)
        except RuntimeError:
            # Fallback: use uniform binning if quantile fails
            min_val = clamped_data.min()
            max_val = clamped_data.max()
            thresholds = torch.linspace(min_val, max_val, num_categories + 1)

        # Assign categories based on thresholds
        categorical_data = torch.zeros_like(raw_data, dtype=torch.long)
        for i in range(num_categories):
            if i == 0:
                # First category: values <= first threshold
                mask = clamped_data <= thresholds[i + 1]
            elif i == num_categories - 1:
                # Last category: values > last threshold
                mask = clamped_data > thresholds[i]
            else:
                # Middle categories: values between thresholds
                mask = (clamped_data > thresholds[i]) & (
                    clamped_data <= thresholds[i + 1]
                )

            categorical_data[mask] = i

        return categorical_data

    def _threshold_binning(
        self, raw_data: torch.Tensor, num_categories: int
    ) -> torch.Tensor:
        """
        Convert continuous data to categorical using random threshold-based binning.
        This creates more varied category distributions.
        """
        # Clamp extreme values to prevent numerical issues
        finite_mask = torch.isfinite(raw_data)
        if not torch.any(finite_mask):
            # All values are NaN/Inf - return zeros
            return torch.zeros_like(raw_data, dtype=torch.long)

        # Replace NaN/Inf with median of finite values
        finite_data = raw_data[finite_mask]
        if len(finite_data) > 0:
            median_val = torch.median(finite_data)
            clamped_data = torch.where(finite_mask, raw_data, median_val)
        else:
            clamped_data = torch.zeros_like(raw_data)

        # Clamp to reasonable range using percentiles
        if len(finite_data) > 1:
            q001 = torch.quantile(finite_data, 0.001)
            q999 = torch.quantile(finite_data, 0.999)
            clamped_data = torch.clamp(clamped_data, min=q001, max=q999)

        # Generate random thresholds within the data range
        min_val = clamped_data.min()
        max_val = clamped_data.max()

        # Handle case where all values are the same
        if min_val == max_val:
            return torch.zeros_like(raw_data, dtype=torch.long)

        thresholds = torch.sort(
            torch.rand(num_categories - 1) * (max_val - min_val) + min_val
        )[0]

        # Assign categories based on thresholds
        categorical_data = torch.zeros_like(raw_data, dtype=torch.long)
        categorical_data[clamped_data <= thresholds[0]] = 0

        for i in range(1, num_categories - 1):
            mask = (clamped_data > thresholds[i - 1]) & (clamped_data <= thresholds[i])
            categorical_data[mask] = i

        categorical_data[clamped_data > thresholds[-1]] = num_categories - 1

        return categorical_data

    def _rank_based_binning(
        self, raw_data: torch.Tensor, num_categories: int
    ) -> torch.Tensor:
        """
        Convert continuous data to categorical using rank-based binning.
        Boundaries are sampled from actual data points.

        Parameters
        ----------
        raw_data : torch.Tensor
            Continuous data tensor
        num_categories : int
            Number of categories

        Returns
        -------
        torch.Tensor
            Categorical data with integer labels
        """
        # Handle edge cases
        finite_mask = torch.isfinite(raw_data)
        if not torch.any(finite_mask):
            return torch.zeros_like(raw_data, dtype=torch.long)

        T = raw_data.shape[0]
        device = raw_data.device

        # Replace NaN/Inf with median
        finite_data = raw_data[finite_mask]
        if len(finite_data) > 0:
            median_val = torch.median(finite_data)
            clamped_data = torch.where(finite_mask, raw_data, median_val)
        else:
            return torch.zeros_like(raw_data, dtype=torch.long)

        # Sample random data points as boundaries
        boundary_indices = torch.randint(0, T, (num_categories - 1,), device=device)
        boundaries = clamped_data[boundary_indices].sort()[0]

        # Handle case where all boundaries are the same
        if boundaries.min() == boundaries.max():
            # Fall back to quantile-based
            return self._quantile_binning(raw_data, num_categories)

        # Compare and assign classes
        classes = (clamped_data.unsqueeze(-1) > boundaries.unsqueeze(0)).sum(dim=1)

        return classes.long()

    def _value_based_binning(
        self, raw_data: torch.Tensor, num_categories: int
    ) -> torch.Tensor:
        """
        Convert continuous data to categorical using value-based binning.
        Boundaries are generated from normal distribution.

        Parameters
        ----------
        raw_data : torch.Tensor
            Continuous data tensor
        num_categories : int
            Number of categories

        Returns
        -------
        torch.Tensor
            Categorical data with integer labels
        """
        # Handle edge cases
        finite_mask = torch.isfinite(raw_data)
        if not torch.any(finite_mask):
            return torch.zeros_like(raw_data, dtype=torch.long)

        device = raw_data.device

        # Replace NaN/Inf with median
        finite_data = raw_data[finite_mask]
        if len(finite_data) > 0:
            median_val = torch.median(finite_data)
            clamped_data = torch.where(finite_mask, raw_data, median_val)
        else:
            return torch.zeros_like(raw_data, dtype=torch.long)

        # Standardize data
        mean = torch.mean(finite_data)
        std = torch.std(finite_data).clip(min=1e-6)
        standardized = (clamped_data - mean) / std

        # Generate random boundaries from normal distribution
        boundaries = torch.randn(num_categories - 1, device=device).sort()[0]

        # Assign classes
        classes = (standardized.unsqueeze(-1) > boundaries.unsqueeze(0)).sum(dim=1)

        return classes.long()

    def _convert_to_datetime64(self, raw_data: torch.Tensor) -> torch.Tensor:
        """
        Convert continuous data to np.datetime64 format.

        Generates a random time range between 1970 and 2020, then normalizes the
        float data to this range and converts to np.datetime64 values with day precision.

        The data is stored internally as days since epoch (1970-01-01) to maintain
        tensor compatibility, but can be converted back to datetime64[D] format
        when generating dataframes.

        Parameters:
        -----------
        raw_data : torch.Tensor
            Continuous data tensor

        Returns:
        --------
        torch.Tensor
            Days since epoch (1970-01-01) as int64 tensor
        """
        # Clamp extreme values to prevent numerical issues
        finite_mask = torch.isfinite(raw_data)
        if not torch.any(finite_mask):
            # All values are NaN/Inf - return middle of date range
            min_datetime = np.datetime64("1970-01-01", "D")
            max_datetime = np.datetime64("2020-12-31", "D")
            min_days = (min_datetime - np.datetime64("1970-01-01", "D")).astype(int)
            max_days = (max_datetime - np.datetime64("1970-01-01", "D")).astype(int)
            middle_days = (min_days + max_days) // 2
            return torch.full_like(raw_data, middle_days, dtype=torch.long)

        # Replace NaN/Inf with median of finite values
        finite_data = raw_data[finite_mask]
        if len(finite_data) > 0:
            median_val = torch.median(finite_data)
            clamped_data = torch.where(finite_mask, raw_data, median_val)
        else:
            clamped_data = torch.zeros_like(raw_data)

        # Clamp to reasonable range using percentiles to handle outliers
        if len(finite_data) > 1:
            clamped_data = torch.clamp(clamped_data, min=-100000, max=100000)

        # Define datetime bounds (1970-01-01 to 2020-12-31)
        min_year = 1970
        max_year = 2020

        # Create np.datetime64 bounds with day precision
        min_datetime = np.datetime64(f"{min_year}-01-01", "D")
        max_datetime = np.datetime64(f"{max_year}-12-31", "D")

        # Generate random datetime range within the bounds for this column
        # Convert to days since epoch for easier random generation
        min_days = (min_datetime - np.datetime64("1970-01-01", "D")).astype(int)
        max_days = (max_datetime - np.datetime64("1970-01-01", "D")).astype(int)

        # Each column gets its own random time range within the overall bounds
        range_start_days = random.randint(
            min_days, max_days - 1
        )  # Leave at least 1 day range
        range_end_days = random.randint(
            range_start_days + 1, max_days
        )  # At least 1 day after start

        # Normalize clamped data to [0, 1] range
        raw_min = clamped_data.min()
        raw_max = clamped_data.max()

        if (
            raw_max == raw_min
            or not torch.isfinite(raw_max)
            or not torch.isfinite(raw_min)
        ):
            # Handle case where all values are the same or still have issues
            normalized_data = torch.full_like(clamped_data, 0.5)
        else:
            normalized_data = (clamped_data - raw_min) / (raw_max - raw_min)
            # Clamp normalized data to [0, 1] range in case of numerical errors
            normalized_data = torch.clamp(normalized_data, 0.0, 1.0)

        # Scale to datetime range (in days)
        day_range = range_end_days - range_start_days
        scaled_days = range_start_days + (normalized_data * day_range)

        # Convert to integer days (datetime64[D] precision)
        # Clamp to valid date range
        days_since_epoch = scaled_days.long()
        days_since_epoch = torch.clamp(days_since_epoch, min=min_days, max=max_days)

        return days_since_epoch


class Table:
    def __init__(
        self,
        num_rows: int,
        num_cols: int,
        num_features: int,
        column_names: List[str],
        data_type_configs: List[DataTypeConfig],
        time_column: int = None,
        device: str = "cpu",
        use_preprocessing: bool = True,
        outlier_threshold: float = 4.0,
    ):
        self.num_rows = num_rows
        self.num_cols = num_cols
        # The difference between num_cols and num_features is that num_features do not count the ID column.
        self.num_features = num_features
        self.column_names = column_names
        self.data_type_configs = data_type_configs
        self.time_column = time_column
        self.device = device

        # Initialize column data processors with preprocessing options
        self.column_data_processors = [
            ColumnDataProcessor(
                use_preprocessing=use_preprocessing, outlier_threshold=outlier_threshold
            )
            for i in range(num_cols)
        ]
        self.dataframe = None

        # valid the number of columns and features
        assert num_cols == len(data_type_configs)
        assert num_features == len(
            [
                data_type_config
                for data_type_config in data_type_configs
                if data_type_config.data_type != DataType.PRIMARY_KEY
                and data_type_config.data_type != DataType.FOREIGN_KEY
            ]
        )

        # valid the first column and only the first column is primary key
        assert data_type_configs[0].data_type == DataType.PRIMARY_KEY
        for i in range(1, len(data_type_configs)):
            if data_type_configs[i].data_type == DataType.PRIMARY_KEY:
                raise ValueError("Only one primary key is allowed")

        # valid if foreign key exists, they must be consective from the second column
        for i in range(2, len(data_type_configs)):
            if data_type_configs[i].data_type == DataType.FOREIGN_KEY:
                assert data_type_configs[i - 1].data_type == DataType.FOREIGN_KEY

    def __repr__(self):
        return f"""
        Table(num_rows={self.num_rows}, num_cols={self.num_cols}, num_features={self.num_features}, column_names={self.column_names},
        data_type_configs={self.data_type_configs}, time_column={self.time_column}, device={self.device})
    """

    @property
    def is_time_table(self) -> bool:
        return self.time_column is not None

    def simple_copy_data(self, raw_data: torch.Tensor) -> torch.Tensor:
        """
        Simple copy data from raw data.
        """
        assert raw_data.shape[0] == self.num_rows
        assert raw_data.shape[1] == self.num_cols
        self.data = raw_data

    def generate_dataframe(self) -> pd.DataFrame:
        """
        Generate a dataframe from the table data.
        """
        data = self.data.cpu().numpy()
        df_dict = {}
        for i in range(self.num_cols):
            column_name = self.column_names[i]
            if self.data_type_configs[i].data_type == DataType.PRIMARY_KEY:
                df_dict[column_name] = data[:, i].astype(np.int32)
            elif self.data_type_configs[i].data_type == DataType.FOREIGN_KEY:
                df_dict[column_name] = data[:, i].astype(np.int32)
            elif self.data_type_configs[i].data_type == DataType.CATEGORICAL:
                df_dict[column_name] = np.round(data[:, i]).astype(np.int32)
            elif self.data_type_configs[i].data_type == DataType.TIMESTAMP:
                # Convert days since epoch back to datetime64 format
                days_since_epoch = data[:, i].astype(np.int64)
                datetime64_values = np.datetime64("1970-01-01", "D") + days_since_epoch
                df_dict[column_name] = datetime64_values
            else:
                # For float data types
                df_dict[column_name] = data[:, i]

        self.dataframe = pd.DataFrame(df_dict)
        return self.dataframe

    def process_data(
        self, raw_data: torch.Tensor, FK_ids=None, parent_tables: List[str] = None
    ) -> torch.Tensor:
        """
        Generate table data by converting raw data according to specified data type configurations.

        Parameters:
        -----------
        raw_data : torch.Tensor or Dict[MASK_TYPE, torch.Tensor]
            Raw continuous data of shape (num_rows, num_features)
            OR Dict if from temporal sampling (contains TIMESTAMP and X keys)

        Returns:
        --------
        torch.Tensor
            Processed data with appropriate data types
        """
        # Handle dict input from temporal sampling
        from src.prior.utils import MASK_TYPE

        timestamp_values = None
        if isinstance(raw_data, dict):
            # Extract timestamp if available
            if MASK_TYPE.TIMESTAMP in raw_data:
                timestamp_values = raw_data[MASK_TYPE.TIMESTAMP]
            raw_data = raw_data[MASK_TYPE.X]

        assert raw_data.shape[0] == self.num_rows
        assert raw_data.shape[1] == self.num_features

        self.data = torch.zeros(self.num_rows, self.num_cols, device=self.device)

        # Set PK first
        self.data[:, 0] = torch.arange(self.num_rows, device=self.device)

        # Set FKs
        if FK_ids is not None:
            for i in range(0, FK_ids.shape[1]):
                FK_id = FK_ids[:, i]
                for j in range(1, self.num_cols):
                    if self.data_type_configs[j].data_type == DataType.FOREIGN_KEY:
                        if self.data_type_configs[j].parent_table == parent_tables[i]:
                            self.data[:, j] = FK_id
                            break

        # Set timestamp column if available
        if timestamp_values is not None and self.time_column is not None:
            # Process timestamp values through the timestamp processor
            self.data[:, self.time_column] = self.column_data_processors[
                self.time_column
            ].process_data(
                timestamp_values.squeeze(), self.data_type_configs[self.time_column]
            )

        # Process remaining features (skip timestamp column if already set)
        for i in range(self.num_features):
            processer_idx = i + self.num_cols - self.num_features

            # Skip if this is the timestamp column (already processed)
            if self.time_column is not None and processer_idx == self.time_column:
                continue

            data_type_config = self.data_type_configs[processer_idx]
            self.data[:, processer_idx] = self.column_data_processors[
                processer_idx
            ].process_data(raw_data[:, i], data_type_config)

        return self.data

    def save_to_file(self, file_path: str) -> None:
        # save to a csv file
        if self.dataframe is None:
            self.generate_dataframe()
        self.dataframe.to_csv(file_path, index=False)

    def get_feature_columns(
        self, only_categorical: bool = False, only_float: bool = False
    ) -> List[str]:
        """
        Get the feature columns of the table.
        """
        assert not (only_categorical and only_float)
        if only_categorical:
            return [
                col
                for (i, col) in enumerate(self.column_names)
                if self.data_type_configs[i].data_type == DataType.CATEGORICAL
            ]
        if only_float:
            return [
                col
                for (i, col) in enumerate(self.column_names)
                if self.data_type_configs[i].data_type == DataType.FLOAT
            ]
        return [
            col
            for (i, col) in enumerate(self.column_names)
            if self.data_type_configs[i].data_type == DataType.FLOAT
            or self.data_type_configs[i].data_type == DataType.CATEGORICAL
        ]


class Relationship:
    """Represents a relationship between tables (can be PK-FK or FK-PK)."""

    def __init__(
        self, from_table: str, from_column: int, to_table: str, to_column: int
    ):
        """
        Initialize a relationship.

        Parameters:
        -----------
        from_table : str
            Name of the source table
        from_column : int
            Column index in from_table
        to_table : str
            Name of the target table
        to_column : int
            Column index in to_table
        """
        self.from_table = from_table
        self.from_column = from_column
        self.to_table = to_table
        self.to_column = to_column

    def is_pk_to_fk(self, rdb) -> bool:
        """
        Check if this is a PK→FK relationship.

        Parameters:
        -----------
        rdb : RDB
            The RDB containing the tables

        Returns:
        --------
        bool
            True if from_table has PK and to_table has FK
        """
        from_table = rdb.tables[self.from_table]
        to_table = rdb.tables[self.to_table]

        from_is_pk = (
            from_table.data_type_configs[self.from_column].data_type
            == DataType.PRIMARY_KEY
        )
        to_is_fk = (
            to_table.data_type_configs[self.to_column].data_type == DataType.FOREIGN_KEY
        )

        return from_is_pk and to_is_fk

    def is_fk_to_pk(self, rdb) -> bool:
        """
        Check if this is a FK→PK relationship.

        Parameters:
        -----------
        rdb : RDB
            The RDB containing the tables

        Returns:
        --------
        bool
            True if from_table has FK and to_table has PK
        """
        from_table = rdb.tables[self.from_table]
        to_table = rdb.tables[self.to_table]

        from_is_fk = (
            from_table.data_type_configs[self.from_column].data_type
            == DataType.FOREIGN_KEY
        )
        to_is_pk = (
            to_table.data_type_configs[self.to_column].data_type == DataType.PRIMARY_KEY
        )

        return from_is_fk and to_is_pk

    def get_direction(self, rdb) -> str:
        """
        Get the relationship direction.

        Parameters:
        -----------
        rdb : RDB
            The RDB containing the tables

        Returns:
        --------
        str
            'pk_to_fk', 'fk_to_pk', or 'unknown'
        """
        if self.is_pk_to_fk(rdb):
            return "pk_to_fk"
        elif self.is_fk_to_pk(rdb):
            return "fk_to_pk"
        else:
            return "unknown"

    def __repr__(self):
        return f"Relationship({self.from_table}[{self.from_column}] -> {self.to_table}[{self.to_column}])"

    def to_dict(self):
        return {
            "from_table": self.from_table,
            "from_column": self.from_column,
            "to_table": self.to_table,
            "to_column": self.to_column,
        }


class TableGenerator:
    def __init__(
        self,
        table_name: str,
        num_rows: int,
        num_cols: int,
        num_features: int,
        device: str,
    ):
        self.table_name = table_name
        self.output_causes = 10  # set to 10 for now
        self.num_rows = num_rows
        self.num_cols = num_cols
        self.num_features = num_features
        self.device = device

        # Storage for full SCM outputs and index mappings
        self.full_scm_outputs = None  # Full flattened outputs from all MLP layers
        self.mask_idx_dict = {}  # Index dictionary for accessing different mask types
        self.all_scm_outputs = {}  # Dictionary containing all masked outputs
        self.row_embeddings = None
        self.row_embedding_layout: List[Tuple[MASK_TYPE, int, int]] = []
        self.pending_outputs: Dict[MASK_TYPE, torch.Tensor] | None = None
        self.pending_fk_ids: torch.Tensor | None = None
        self.pending_parent_tables: List[str] | None = None

    def init_table_SCM(
        self,
        **kwargs: Dict[str, Any],
        # seq_len: int,
        # num_layers: int,
        # hidden_dim: int,
        # num_outputs: int,
        # num_causes: int,
        # mlp_activations: nn.Module,
        # other_causes: int,
        # sampling_ratio: float,
        # masks: Dict[str, int],
        # device: str,
        # use_timestamp_sampling: bool = False,
        # eta: float = 1.0,
    ) -> None:
        self.table_SCM = MLPSCM(
            **kwargs,
            # seq_len=seq_len,
            # num_layers=num_layers,
            # hidden_dim=hidden_dim,
            # num_outputs=num_outputs,
            # num_causes=num_causes,
            # mlp_activations=mlp_activations,
            # other_causes=other_causes,
            # sampling_ratio=sampling_ratio,
            # masks=masks,
            # device=device,
            # use_timestamp_sampling=use_timestamp_sampling,
            # # Enhanced sampling parameters
            # eta=eta,
        )

        # Store the actual generated indices from the SCM (not the input mask numbers)
        self.mask_idx_dict = {}
        for mask_type, indices in self.table_SCM.masks.items():
            # Convert MASK_TYPE enum to string for consistent access
            mask_key = (
                mask_type.value if hasattr(mask_type, "value") else str(mask_type)
            )
            self.mask_idx_dict[mask_key] = indices

    def generate_data(self, **kwargs) -> torch.Tensor:
        # currently, only support parent_data_list
        with torch.no_grad():
            if "parent_data_list" in kwargs:
                parent_data_list = kwargs["parent_data_list"]

                # Check if we should use timestamp-based sampling
                if (
                    hasattr(self.table_SCM, "use_timestamp_sampling")
                    and self.table_SCM.use_timestamp_sampling
                ):
                    # print("Using enhanced temporal sampling")
                    # Always use enhanced temporal sampling when timestamps are enabled
                    X, FK_ids, outputs_flat = (
                        self.table_SCM.forward_with_enhanced_temporal_sampling(
                            parent_data_list
                        )
                    )
                else:
                    # print("Using Plain Parent method")
                    # Use original method
                    X, FK_ids, outputs_flat = self.table_SCM.forward_with_input(
                        parent_data_list
                    )

                # Save full SCM outputs and all masked outputs
                self.all_scm_outputs = X.copy()

                # Return the full X dict (includes TIMESTAMP if available)
                return X, FK_ids
            else:
                # print("Using Non-parent method")
                X, outputs_flat = self.table_SCM.forward_without_input()

                # Save full SCM outputs and all masked outputs
                self.all_scm_outputs = X.copy()

                # Return the full X dict
                return X

    def cache_pending_outputs(
        self,
        outputs: Dict[MASK_TYPE, torch.Tensor],
        fk_ids: torch.Tensor | None,
        parent_tables: List[str],
    ) -> None:
        self.pending_outputs = self.all_scm_outputs if self.all_scm_outputs else outputs
        self.pending_fk_ids = fk_ids
        self.pending_parent_tables = list(parent_tables) if parent_tables else []
        self.row_embeddings = self._build_row_embedding()

    def _build_row_embedding(self) -> torch.Tensor | None:
        if self.pending_outputs is None:
            return None

        components = []
        layout: List[Tuple[MASK_TYPE, int, int]] = []
        cursor = 0

        for mask_type in [MASK_TYPE.X, MASK_TYPE.CAUSAL_OUTPUT]:
            if mask_type not in self.pending_outputs:
                continue
            tensor = self.pending_outputs[mask_type]
            if tensor.ndim == 1:
                tensor = tensor.unsqueeze(-1)
            width = tensor.shape[1]
            components.append(tensor)
            layout.append((mask_type, cursor, cursor + width))
            cursor += width

        if not components:
            self.row_embedding_layout = []
            return None

        self.row_embedding_layout = layout
        return torch.cat(components, dim=-1)

    def update_row_embeddings(self, refined_embeddings: torch.Tensor) -> None:
        if refined_embeddings is None or not self.row_embedding_layout:
            return

        self.row_embeddings = refined_embeddings
        if self.pending_outputs is None:
            return

        for mask_type, start, end in self.row_embedding_layout:
            updated_slice = refined_embeddings[:, start:end]
            if mask_type in self.pending_outputs:
                self.pending_outputs[mask_type] = updated_slice
            if mask_type in self.all_scm_outputs:
                self.all_scm_outputs[mask_type] = updated_slice


class RDB:
    """Relational Database class that manages multiple tables and their relationships."""

    def __init__(self, name: str = "default_rdb"):
        """
        Initialize an RDB.

        Parameters:
        -----------
        name : str
            Name of the relational database
        """
        self.name = name
        self.table_names: List[str] = []
        self.relationships: List[Relationship] = []
        self.graph: nx.DiGraph = None
        self.tables: Dict[str, Table] = {}
        self.table_generators: Dict[str, TableGenerator] = {}
        # self.device = "cuda:0"  # TODO: Change to GPU if available
        self.device = "cpu"  # TODO: Change to GPU if available

        # Generation schema tracking
        self.table_generation_schemas: Dict[str, TableGenerationSchema] = {}
        self.task_generation_schemas: List[TaskGenerationSchema] = []
        self.row_gnn_runner: RowGNNRunner | None = None

    def add_table(self, table_name: str, table: Table) -> None:
        """
        Add a table to the RDB.

        Parameters:
        -----------
        table_name : str
            Name of the table
        table : Table
            Table object to add
        """
        if table_name in self.tables:
            raise ValueError(
                f"Table '{table_name}' already exists in RDB '{self.name}'"
            )

        self.tables[table_name] = table
        self.table_names.append(table_name)
        self.table_generators[table_name] = TableGenerator(
            table_name, table.num_rows, table.num_cols, table.num_features, self.device
        )
        if self.graph is not None:
            self.graph = None

    def add_relationship(self, relationship: Relationship) -> None:
        """
        Add a relationship between tables. Supports both PK→FK and FK→PK directions.

        Parameters:
        -----------
        relationship : Relationship
            Relationship object to add
        """
        # Validate that both tables exist
        if relationship.from_table not in self.tables:
            raise ValueError(f"Table '{relationship.from_table}' not found in RDB")
        if relationship.to_table not in self.tables:
            raise ValueError(f"Table '{relationship.to_table}' not found in RDB")

        # Validate column indices
        from_table = self.tables[relationship.from_table]
        to_table = self.tables[relationship.to_table]

        if relationship.from_column >= from_table.num_cols:
            raise ValueError(
                f"Column {relationship.from_column} out of range for table '{relationship.from_table}'"
            )
        if relationship.to_column >= to_table.num_cols:
            raise ValueError(
                f"Column {relationship.to_column} out of range for table '{relationship.to_table}'"
            )

        # Validate that it's a valid relationship (PK-FK or FK-PK)
        direction = relationship.get_direction(self)
        if direction == "unknown":
            raise ValueError(
                f"Invalid relationship: {relationship.from_table}[{relationship.from_column}] -> "
                f"{relationship.to_table}[{relationship.to_column}]. "
                f"Must be PK→FK or FK→PK relationship."
            )

        self.relationships.append(relationship)

        # Set parent table for foreign key columns
        if direction == "fk_to_pk":
            # from_table has FK, to_table has PK
            from_table.data_type_configs[relationship.from_column].parent_table = (
                relationship.to_table
            )
        elif direction == "pk_to_fk":
            # from_table has PK, to_table has FK
            to_table.data_type_configs[relationship.to_column].parent_table = (
                relationship.from_table
            )

        print(f"Added {direction} relationship: {relationship}")
        if self.graph is not None:
            self.graph = None

    def get_table(self, table_name: str) -> Table:
        """
        Get a table by name.

        Parameters:
        -----------
        table_name : str
            Name of the table to retrieve

        Returns:
        --------
        Table
            The requested table
        """
        if table_name not in self.tables:
            raise ValueError(f"Table '{table_name}' not found in RDB '{self.name}'")
        return self.tables[table_name]

    def get_relationships_for_table(self, table_name: str) -> List[Relationship]:
        """
        Get all relationships involving a specific table.

        Parameters:
        -----------
        table_name : str
            Name of the table

        Returns:
        --------
        List[Relationship]
            List of relationships involving the table
        """
        return [
            rel
            for rel in self.relationships
            if rel.from_table == table_name or rel.to_table == table_name
        ]

    def get_foreign_keys_for_table(self, table_name: str) -> List[Relationship]:
        """
        Get all foreign key relationships where the given table is the source.

        Parameters:
        -----------
        table_name : str
            Name of the table

        Returns:
        --------
        List[Relationship]
            List of foreign key relationships from the table
        """
        return [rel for rel in self.relationships if rel.from_table == table_name]

    def get_primary_keys_for_table(self, table_name: str) -> List[Relationship]:
        """
        Get all primary key relationships where the given table is the target.

        Parameters:
        -----------
        table_name : str
            Name of the table

        Returns:
        --------
        List[Relationship]
            List of primary key relationships to the table
        """
        return [rel for rel in self.relationships if rel.to_table == table_name]

    def enable_row_gnn(
        self,
        hidden_dim: int = 32,
        num_layers: int = 1,
        num_steps: int = 1,
        device: str = "cpu",
    ) -> None:
        """
        Enable row-level GNN refinement with the specified configuration.
        """
        self.row_gnn_runner = RowGNNRunner(
            hidden_dim=hidden_dim,
            num_layers=num_layers,
            num_steps=num_steps,
            device=device,
        )

    def init_table_SCMs(
        self,
        seed: int = 42,
    ) -> None:
        """
        Initialize the SCMs for all tables.
        """
        if self.graph is None:
            self.convert_to_graph()
        hpsampler = HpSamplerList(DEFAULT_SAMPLED_HP, device=self.device, seed=seed)

        for table_name in nx.topological_sort(self.graph):
            table = self.tables[table_name]
            parent_tables = [
                rel.to_table for rel in self.get_foreign_keys_for_table(table_name)
            ]
            seq_len = table.num_rows
            # num_outputs = num_causes  # Set to 10 for now
            # num_causes = 10  # Set to 10 for now
            masks = {MASK_TYPE.X: table.num_features}

            # Check if we should use timestamp-based sampling
            # based on whether the table has a timestamp column
            if table.is_time_table:
                use_timestamp_sampling = True
            else:
                use_timestamp_sampling = False

            if parent_tables:
                other_causes = np.sum(
                    [
                        self.table_generators[parent_table].table_SCM.num_outputs
                        for parent_table in parent_tables
                    ]
                )
                masks[MASK_TYPE.EDGE_PROB] = 1

                # Add timestamp mask for timestamp-based sampling
                if use_timestamp_sampling:
                    sampling_ratio = 1.0  # Set to 1 for now
                    masks[MASK_TYPE.TIMESTAMP] = 1
                else:
                    sampling_ratio = 10.0  # Set to 10 for now
            else:
                other_causes = 0
                sampling_ratio = 1.0

            # Determine generation type
            if len(parent_tables) == 0:
                generation_type = "self_generated"
            elif use_timestamp_sampling:
                generation_type = "timestamp_based"
            else:
                generation_type = "parent_based"

            # Create and store table generation schema
            # 1st dict: sampled parameters from hpsampling
            sampled_scm = hpsampler.sample()
            sampled_scm_params = {
                k: v() if callable(v) else v for k, v in sampled_scm.items()
            }

            # 2nd dict: base parameters (without masks)
            base_params = {
                "seq_len": seq_len,
                # "num_outputs": num_outputs,
                # "num_causes": num_causes,
                "other_causes": other_causes,
                "sampling_ratio": sampling_ratio,
                "use_timestamp_sampling": use_timestamp_sampling,
            }

            # 3rd dict: combine sampled and base parameters
            combined_params = sampled_scm_params.copy()
            for k, v in base_params.items():
                if k not in combined_params:
                    combined_params[k] = v
                else:
                    raise ValueError(f"Parameter {k} is already sampled")

            # Create separate mask dictionaries for different purposes
            # For SCM: use original MASK_TYPE enums
            scm_masks = masks.copy()
            # For schema: convert MASK_TYPE to strings for serialization
            schema_masks = {str(k): v for k, v in masks.items()}

            table_schema = TableGenerationSchema(
                table_name=table_name,
                generation_type=generation_type,
                parent_tables=parent_tables,
                uses_timestamp=use_timestamp_sampling,
                eta=combined_params["eta"],
                num_parents=len(parent_tables),
                is_timestamp_table=table.is_time_table,
                scm_params=combined_params,
                masks=schema_masks,
            )
            self.table_generation_schemas[table_name] = table_schema

            self.table_generators[table_name].init_table_SCM(
                **combined_params,
                masks=scm_masks,
                device=self.device,
            )

    def generate_one_table_data_from_SCM(
        self, table_name: str, parent_tables: List[str]
    ) -> torch.Tensor:
        """
        Generate data for a single table from a SCM and cache the outputs.
        """
        table_generator = self.table_generators[table_name]

        if parent_tables:
            parent_data_list = []
            for parent_table in parent_tables:
                # Use all SCM outputs instead of just latent embeddings
                parent_data = self.table_generators[parent_table].all_scm_outputs
                parent_data_list.append(parent_data)
            # Returns X_dict (dict with TIMESTAMP if available), FK_ids
            X_dict, FK_ids = table_generator.generate_data(
                parent_data_list=parent_data_list
            )
        else:
            # Returns X_dict (dict)
            X_dict = table_generator.generate_data()
            FK_ids = None

        table_generator.cache_pending_outputs(X_dict, FK_ids, parent_tables)

        return

    def generate_all_data_from_SCM(self) -> Dict[str, torch.Tensor]:
        """
        Generate data for all tables in the RDB from the graph, according to topological order.

        Returns:
        --------
        None
        """

        if self.graph is None:
            self.convert_to_graph()

        for table_name in nx.topological_sort(self.graph):
            parent_tables = [
                rel.to_table for rel in self.get_foreign_keys_for_table(table_name)
            ]
            # print(f"Generating data for table {table_name}...")
            self.generate_one_table_data_from_SCM(table_name, parent_tables)

        self._run_row_gnn_if_enabled()
        self._materialize_tables_from_pending()

        return

    def _run_row_gnn_if_enabled(self) -> None:
        if self.row_gnn_runner is None:
            return

        gnn_device = (
            self.row_gnn_runner.device if self.row_gnn_runner is not None else self.device
        )
        print(f"Running row GNN on device: {gnn_device}")
        builder = RowGraphBuilder(self.table_generators, device=gnn_device)
        graph = builder.build()
        if graph is None:
            return

        refined_embeddings = self.row_gnn_runner.run(graph)
        if refined_embeddings is None:
            return

        for table_name, embeddings in refined_embeddings.items():
            if table_name not in self.table_generators:
                continue
            table_generator = self.table_generators[table_name]
            table_generator.update_row_embeddings(embeddings)

    def _materialize_tables_from_pending(self) -> None:
        for table_name in self.table_names:
            generator = self.table_generators[table_name]
            outputs = generator.pending_outputs
            if outputs is None:
                continue

            fk_ids = generator.pending_fk_ids
            parent_tables = generator.pending_parent_tables or []
            detached_outputs = {
                key: value.detach() if isinstance(value, torch.Tensor) else value
                for key, value in outputs.items()
            }
            self.tables[table_name].process_data(
                detached_outputs, fk_ids, parent_tables
            )

            # Clear caches for next generation
            generator.pending_outputs = None
            generator.pending_fk_ids = None
            generator.pending_parent_tables = None

    def validate_data_quality(self, threshold: float = 0.5) -> tuple[bool, dict]:
        """
        Validate data quality by checking for degenerate columns (all same values).
        Checks each table individually - if ANY table has >threshold constant columns,
        the entire RDB is invalid.

        Parameters
        ----------
        threshold : float, default=0.5
            Maximum fraction of numeric/categorical columns that can be constant per table.
            If any table has more than this fraction with constant values, the RDB is invalid.

        Returns
        -------
        tuple[bool, dict]
            - is_valid: True if ALL tables pass quality check
            - stats: Dictionary with validation statistics
        """
        total_checkable_columns = 0
        constant_columns = 0
        invalid_tables = []
        table_stats = []

        for table_name, table in self.tables.items():
            if not hasattr(table, "data") or table.data is None:
                continue

            table_checkable_cols = 0
            table_constant_cols = []

            for col_idx, data_type_config in enumerate(table.data_type_configs):
                # Only check numeric and categorical columns
                if data_type_config.data_type in [DataType.FLOAT, DataType.CATEGORICAL]:
                    table_checkable_cols += 1
                    total_checkable_columns += 1
                    col_data = table.data[:, col_idx]

                    # Check if all values are the same
                    if self._is_constant_column(col_data):
                        constant_columns += 1
                        table_constant_cols.append(
                            {
                                "column_name": table.column_names[col_idx],
                                "column_idx": col_idx,
                                "data_type": data_type_config.data_type.value,
                            }
                        )

            # Calculate per-table fraction
            table_constant_fraction = (
                len(table_constant_cols) / table_checkable_cols
                if table_checkable_cols > 0
                else 0.0
            )

            # Check if this table is invalid
            table_is_valid = table_constant_fraction <= threshold

            table_info = {
                "table_name": table_name,
                "checkable_columns": table_checkable_cols,
                "constant_columns": len(table_constant_cols),
                "constant_fraction": table_constant_fraction,
                "is_valid": table_is_valid,
                "constant_column_details": table_constant_cols,
            }
            table_stats.append(table_info)

            if not table_is_valid:
                invalid_tables.append(table_info)

        # RDB is valid only if ALL tables are valid
        is_valid = len(invalid_tables) == 0

        # Calculate overall fraction for informational purposes
        overall_constant_fraction = (
            constant_columns / total_checkable_columns
            if total_checkable_columns > 0
            else 0.0
        )

        stats = {
            "is_valid": is_valid,
            "total_tables": len(self.tables),
            "invalid_tables_count": len(invalid_tables),
            "total_checkable_columns": total_checkable_columns,
            "constant_columns": constant_columns,
            "overall_constant_fraction": overall_constant_fraction,
            "threshold": threshold,
            "table_stats": table_stats,
            "invalid_tables": invalid_tables,
        }

        return is_valid, stats

    def _is_constant_column(self, col_data: torch.Tensor) -> bool:
        """
        Check if a column has all the same values (constant/degenerate).

        Parameters
        ----------
        col_data : torch.Tensor
            Column data to check

        Returns
        -------
        bool
            True if all values are the same (ignoring NaN)
        """
        # Handle NaN values
        finite_mask = torch.isfinite(col_data)
        if not torch.any(finite_mask):
            # All NaN - consider this constant
            return True

        finite_data = col_data[finite_mask]

        # Check if all finite values are the same
        unique_values = torch.unique(finite_data)
        return len(unique_values) == 1

    def get_table_info(self) -> Dict[str, Any]:
        """
        Get information about all tables and relationships in the RDB.

        Returns:
        --------
        Dict[str, Any]
            Dictionary containing RDB information
        """
        info = {
            "name": self.name,
            "num_tables": len(self.tables),
            "num_relationships": len(self.relationships),
            "tables": {},
            "relationships": [],
        }

        for table_name, table in self.tables.items():
            info["tables"][table_name] = {
                "num_features": table.num_features,
                "num_rows": table.num_rows,
                "is_time_table": table.is_time_table,
                "time_column": table.time_column,
                "data_types": [
                    config.data_type.value for config in table.data_type_configs
                ],
            }

        for rel in self.relationships:
            info["relationships"].append(
                {
                    "from_table": rel.from_table,
                    "from_column": rel.from_column,
                    "to_table": rel.to_table,
                    "to_column": rel.to_column,
                }
            )

        return info

    def convert_to_graph(self) -> nx.DiGraph:
        """
        Convert the RDB to a graph.
        """
        self.graph = nx.DiGraph()
        for table_name, table in self.tables.items():
            self.graph.add_node(table_name)
        for relationship in self.relationships:
            self.graph.add_edge(relationship.to_table, relationship.from_table)
        return

    def __repr__(self):
        return f"RDB(name='{self.name}', tables={list(self.tables.keys())}, relationships={len(self.relationships)})"

    def save_to_file(self, file_path: str) -> None:
        # for each table, save to a csv file
        for table_name, table in self.tables.items():
            table.save_to_file(f"{file_path}/{table_name}.csv")

        # save the relationships to a json file
        with open(f"{file_path}/relationships.json", "w") as f:
            json.dump(
                [relationship.to_dict() for relationship in self.relationships], f
            )

    def save_to_4dbinfer_dataset(self, file_path: str) -> None:
        """
        Save the RDB to 4DBInfer dataset format.

        Parameters:
        -----------
        file_path : str
            Directory path where the dataset will be saved
        """
        # Create the dataset directory
        os.makedirs(file_path, exist_ok=True)

        # Create table schemas list
        table_schemas = []

        for table_name, table in self.tables.items():
            # Create column schemas
            column_schemas = []

            for i, (col_name, data_type_config) in enumerate(
                zip(table.column_names, table.data_type_configs)
            ):
                # Map our data types to DBB data types
                if data_type_config.data_type == DataType.PRIMARY_KEY:
                    dbb_dtype = DBBColumnDType.primary_key
                    extra_fields = {"capacity": table.num_rows}
                elif data_type_config.data_type == DataType.FOREIGN_KEY:
                    dbb_dtype = DBBColumnDType.foreign_key
                    # Find the parent table for this foreign key
                    parent_table = data_type_config.config.get(
                        "parent_table", "unknown"
                    )
                    extra_fields = {
                        "link_to": f"{parent_table}.{self.tables[parent_table].column_names[0]}",
                    }
                elif data_type_config.data_type == DataType.CATEGORICAL:
                    dbb_dtype = DBBColumnDType.category_t
                    num_categories = data_type_config.config.get("num_categories", 10)
                    extra_fields = {"num_categories": num_categories}
                elif data_type_config.data_type == DataType.TIMESTAMP:
                    dbb_dtype = DBBColumnDType.datetime_t
                    extra_fields = {}
                elif data_type_config.data_type == DataType.FLOAT:
                    dbb_dtype = DBBColumnDType.float_t
                    extra_fields = {"in_size": 1}
                else:
                    # Default to float for unknown types
                    dbb_dtype = DBBColumnDType.float_t
                    extra_fields = {"in_size": 1}

                # Create column schema with extra fields
                column_schema = DBBColumnSchema(
                    name=col_name, dtype=dbb_dtype, **extra_fields
                )
                column_schemas.append(column_schema)

            # Save table data as parquet
            table_file_name = f"{table_name}.parquet"
            table_file_path = os.path.join(file_path, table_file_name)

            # Convert table data to pandas DataFrame and save as parquet
            data_np = table.data.cpu().numpy()
            df_dict = {}
            for i, col_name in enumerate(table.column_names):
                if table.data_type_configs[i].data_type in [
                    DataType.PRIMARY_KEY,
                    DataType.FOREIGN_KEY,
                    DataType.CATEGORICAL,
                ]:
                    df_dict[col_name] = data_np[:, i].astype(np.int32)
                elif table.data_type_configs[i].data_type == DataType.TIMESTAMP:
                    # Convert days since epoch back to datetime64 format
                    days_since_epoch = data_np[:, i].astype(np.int64)
                    datetime64_values = (
                        np.datetime64("1970-01-01", "D") + days_since_epoch
                    )
                    df_dict[col_name] = datetime64_values
                else:
                    df_dict[col_name] = data_np[:, i].astype(np.float32)

            df = pd.DataFrame(df_dict)
            df.to_parquet(table_file_path, index=False)

            # Determine time column name if it exists
            time_column_name = None
            if table.time_column is not None:
                time_column_name = table.column_names[table.time_column]

            # Create table schema
            table_schema = DBBTableSchema(
                name=table_name,
                source=table_file_name,
                format=DBBTableDataFormat.PARQUET,
                columns=column_schemas,
                time_column=time_column_name,
            )
            table_schemas.append(table_schema)

        # Create dataset metadata
        dataset_meta = DBBRDBDatasetMeta(
            dataset_name=self.name, tables=table_schemas, tasks=[]  # No tasks for now
        )

        # Save metadata as JSON
        metadata_file_path = os.path.join(file_path, "metadata.yaml")
        save_pyd(dataset_meta, metadata_file_path)

        print(f"Saved 4DBInfer dataset to {file_path}")
        print(f"  - {len(self.tables)} tables")
        print(f"  - {len(self.relationships)} relationships")
        print(f"  - Dataset metadata: {metadata_file_path}")

    def initialize_tasks(
        self,
        task_generator=None,
        tasks_per_rdb: int = 1,
        exclude_small_tables: bool = True,
        min_table_size: int = 10,
        train_ratio: float = 0.8,
        valid_ratio: float = 0.1,
    ) -> List:
        """
        Initialize TaskGenerator and generate tasks via TaskDataGenerator.generate_task_data.

        Parameters
        ----------
        task_generator : TaskGenerator, optional
            TaskGenerator instance. If None, creates a new one.
        tasks_per_rdb : int
            Number of tasks to generate per table
        exclude_small_tables : bool
            Whether to exclude small tables from task generation
        min_table_size : int
            Minimum table size to consider for task generation
        train_ratio : float
            Ratio of data to use for training
        valid_ratio : float
            Ratio of data to use for validation

        Returns
        -------
        List[Task]
            List of generated tasks with data
        """
        # Import here to avoid circular imports
        from .task_generation import TaskGenerator

        # Generate dataframes for all tables if not already done
        for table_name, table in self.tables.items():
            if (
                table.dataframe is None
                and hasattr(table, "data")
                and table.data is not None
            ):
                table.generate_dataframe()

        # Initialize task generator if not provided
        if task_generator is None:
            task_generator = TaskGenerator(rdb=self, random_seed=42)

        # Store task generator for reuse
        self.task_generator = task_generator

        # Generate tasks
        print("Generating tasks...")
        # tasks = task_generator.generate_tasks_for_rdb(
        #     self,
        #     tasks_per_rdb=tasks_per_rdb,
        #     exclude_small_tables=exclude_small_tables,
        #     min_table_size=min_table_size,
        # )
        tasks = task_generator.generate_tasks_for_rdb_per_table(
            self,
            exclude_small_tables=exclude_small_tables,
            min_table_size=min_table_size,
        )

        if len(tasks) == 0:
            print("No tasks generated")
            self.tasks = []
            return []

        print(f"Generated {len(tasks)} tasks")

        # Generate complete task data through TaskGenerator (single entrypoint)
        task_generator.generate_task_data(
            tasks=tasks,
            rdb=self,
            train_ratio=train_ratio,
            valid_ratio=valid_ratio,
        )

        # Store tasks in RDB
        self.tasks = tasks
        print(f"Initialized {len(tasks)} tasks with data")

        return tasks

    def initialize_tasks_with_complex_tasks(
        self,
        task_generator=None,
        tasks_per_rdb: int = 3,
        exclude_small_tables: bool = True,
        min_table_size: int = 10,
        train_ratio: float = 0.8,
        valid_ratio: float = 0.1,
    ) -> List:
        """
        Initialize TaskGenerator and generate tasks via TaskDataGenerator.generate_task_data.

        Parameters
        ----------
        task_generator : TaskGenerator, optional
            TaskGenerator instance. If None, creates a new one.
        tasks_per_rdb : int
            Number of tasks to generate per table
        exclude_small_tables : bool
            Whether to exclude small tables from task generation
        min_table_size : int
            Minimum table size to consider for task generation
        train_ratio : float
            Ratio of data to use for training
        valid_ratio : float
            Ratio of data to use for validation

        Returns
        -------
        List[Task]
            List of generated tasks with data
        """
        # Import here to avoid circular imports
        from .task_generation import TaskGenerator

        # Generate dataframes for all tables if not already done
        for table_name, table in self.tables.items():
            if (
                table.dataframe is None
                and hasattr(table, "data")
                and table.data is not None
            ):
                table.generate_dataframe()

        # Initialize task generator if not provided
        if task_generator is None:
            task_generator = TaskGenerator(rdb=self, random_seed=42)

        # Store task generator for reuse
        self.task_generator = task_generator

        # Generate tasks
        print("Generating tasks...")
        tasks = task_generator.generate_tasks_for_rdb_with_complex_tasks(
            self,
            tasks_per_rdb=tasks_per_rdb,
            exclude_small_tables=exclude_small_tables,
            min_table_size=min_table_size,
        )

        if len(tasks) == 0:
            print("No tasks generated")
            self.tasks = []
            return []

        print(f"Generated {len(tasks)} tasks")

        # Generate complete task data through TaskGenerator (single entrypoint)
        task_generator.generate_task_data(
            tasks=tasks,
            rdb=self,
            train_ratio=train_ratio,
            valid_ratio=valid_ratio,
        )

        # Store tasks in RDB
        self.tasks = tasks
        print(f"Initialized {len(tasks)} tasks with data")

        return tasks

    def save_to_4dbinfer_dataset_with_tasks(
        self,
        file_path: str,
    ) -> None:
        """
        Save the RDB and tasks to 4DBInfer dataset format.
        Tasks must be already initialized using initialize_tasks().

        Parameters
        ----------
        file_path : str
            Directory path where the dataset will be saved
        """
        # First save the regular RDB data
        self.save_to_4dbinfer_dataset(file_path)

        # Check if tasks are initialized
        if not hasattr(self, "tasks") or len(self.tasks) == 0:
            print(
                "No tasks found. Use initialize_tasks() first or saving RDB without tasks"
            )
            return

        if not hasattr(self, "task_generator"):
            raise ValueError("TaskGenerator not found, creating new one for saving")
            print("Warning: TaskGenerator not found, creating new one for saving")
            from .task_generation import TaskGenerator

            self.task_generator = TaskGenerator()

        # Save task data and get metadata
        print("Saving task data...")
        task_metas = self.task_generator.save_all_task_data(self.tasks, file_path)

        # Update dataset metadata to include tasks
        metadata_file_path = os.path.join(file_path, "metadata.yaml")

        # Read existing metadata
        dataset_meta = load_pyd(DBBRDBDatasetMeta, metadata_file_path)

        # Add tasks to metadata
        dataset_meta.tasks = task_metas

        # Save updated metadata
        save_pyd(dataset_meta, metadata_file_path)

        # Save generation schemas
        self.save_generation_schemas(file_path)

        print(f"Successfully saved RDB with {len(task_metas)} tasks to {file_path}")

    def save_generation_schemas(self, file_path: str) -> None:
        """
        Save generation schemas to a separate YAML file.

        Parameters
        ----------
        file_path : str
            Directory path where the generation schemas will be saved
        """
        os.makedirs(file_path, exist_ok=True)

        # Prepare generation schemas data
        generation_schemas = {
            "rdb_name": self.name,
            "table_generation_schemas": {
                table_name: schema.to_dict()
                for table_name, schema in self.table_generation_schemas.items()
            },
            "task_generation_schemas": [
                schema.to_dict() for schema in self.task_generation_schemas
            ],
        }

        # Save to YAML file
        schema_file_path = os.path.join(file_path, "generation_schemas.yaml")
        with open(schema_file_path, "w") as f:
            yaml.dump(generation_schemas, f, default_flow_style=False, sort_keys=False)

        print(f"Saved generation schemas to {schema_file_path}")
        print(f"  - {len(self.table_generation_schemas)} table schemas")
        print(f"  - {len(self.task_generation_schemas)} task schemas")

    def create_schema_graph(self):
        """
        Create a schema graph from the RDB's tables and relationships.

        Returns
        -------
        SchemaGraph
            Schema graph representing the RDB structure
        """
        from .task_generation_utils import SchemaGraph

        schema_graph = SchemaGraph()

        # Add all tables to the schema graph
        for table_name, table in self.tables.items():
            schema_graph.add_table(table_name, table)

        # Add all relationships to the schema graph
        for relationship in self.relationships:
            schema_graph.add_relationship(relationship)

        return schema_graph

    def select_neighbor_tables(
        self, center_table: str, max_neighbors: int = 2, random_seed: int = None
    ):
        """
        Randomly select 0 to max_neighbors neighbor tables for the given center table.

        Parameters
        ----------
        center_table : str
            The center table to find neighbors for
        max_neighbors : int
            Maximum number of neighbors to select (default: 2)
        random_seed : int, optional
            Random seed for reproducible sampling. If None, uses current random state.

        Returns
        -------
        List[str]
            List of selected neighbor table names
        """
        if center_table not in self.tables:
            raise ValueError(f"Table '{center_table}' not found in RDB")

        # Get all possible neighbors
        all_neighbors = []

        # Find tables that have relationships with the center table
        for relationship in self.relationships:
            if relationship.from_table == center_table:
                all_neighbors.append(relationship.to_table)
            elif relationship.to_table == center_table:
                all_neighbors.append(relationship.from_table)

        # Randomly select max_neighbors neighbors
        if all_neighbors:
            num_neighbors = min(max_neighbors, len(all_neighbors))
            # Ensure we have enough neighbors to sample
            if num_neighbors > 0:
                selected_neighbors = random.sample(all_neighbors, num_neighbors)
            else:
                selected_neighbors = []
        else:
            selected_neighbors = []

        return selected_neighbors

    def create_sub_schema_graph(self, center_table: str, neighbor_tables: List[str]):
        """
        Create a sub-schema graph containing only the center table and selected neighbors.

        Parameters
        ----------
        center_table : str
            The center table
        neighbor_tables : List[str]
            List of neighbor table names

        Returns
        -------
        SchemaGraph
            Sub-schema graph with only the specified tables and their relationships
        """
        from .task_generation_utils import SchemaGraph, SchemaEdge

        sub_schema = SchemaGraph()
        selected_tables = [center_table] + neighbor_tables

        # Add selected tables as nodes
        for table_name in selected_tables:
            if table_name in self.tables:
                sub_schema.add_node(table_name, self.tables[table_name])

        # Add relationships between selected tables only
        for relationship in self.relationships:
            if (
                relationship.from_table in selected_tables
                and relationship.to_table in selected_tables
            ):
                # Convert Relationship to SchemaEdge
                # Always from center table to neighbor table
                if relationship.from_table == center_table:
                    schema_edge = SchemaEdge(
                        from_table=relationship.from_table,
                        to_table=relationship.to_table,
                        from_column=relationship.from_column,
                        to_column=relationship.to_column,
                    )
                else:
                    schema_edge = SchemaEdge(
                        from_table=relationship.to_table,
                        to_table=relationship.from_table,
                        from_column=relationship.to_column,
                        to_column=relationship.from_column,
                    )
                sub_schema.add_edge(schema_edge)

        return sub_schema

    def create_multi_hop_schema_graph(
        self, center_table: str, num_hops: int = 2, neighbors_each_hop: int = 1
    ):
        """
        Create a multi-hop schema graph containing the center table and selected neighbors.
        """
        from .task_generation_utils import SchemaGraph, SchemaEdge, SchemaEdgeDirection

        schema_graph = SchemaGraph()
        schema_graph.add_node(center_table, self.tables[center_table])
        current_centers = [center_table]
        new_centers = []
        past_tables = set()
        past_tables.add(center_table)
        for hop in range(num_hops):
            for center in current_centers:
                neighbors = self.select_neighbor_tables(center, neighbors_each_hop)
                for neighbor in neighbors:
                    if neighbor not in past_tables:
                        schema_graph.add_node(neighbor, self.tables[neighbor])
                        for relationship in self.relationships:
                            if (
                                relationship.from_table == center
                                and relationship.to_table in neighbors
                            ):
                                schema_edge = SchemaEdge(
                                    from_table=relationship.from_table,
                                    to_table=relationship.to_table,
                                    from_column=relationship.from_column,
                                    to_column=relationship.to_column,
                                    direction=SchemaEdgeDirection.PK_TO_FK,
                                )
                                schema_graph.add_edge(schema_edge)
                            elif (
                                relationship.to_table == center
                                and relationship.from_table in neighbors
                            ):
                                schema_edge = SchemaEdge(
                                    from_table=relationship.to_table,
                                    to_table=relationship.from_table,
                                    from_column=relationship.to_column,
                                    to_column=relationship.from_column,
                                    direction=SchemaEdgeDirection.FK_TO_PK,
                                )
                                schema_graph.add_edge(schema_edge)

                        past_tables.add(neighbor)
                        new_centers.append(neighbor)

            current_centers = new_centers
            new_centers = []

        return schema_graph


if __name__ == "__main__":
    # Test the RDB class with multiple tables and relationships
    print("\n" + "=" * 60)
    print("Testing RDB class with multiple tables and relationships...")
    print("=" * 60)

    # Create an RDB instance
    rdb = RDB("test_database")

    # Create tables for a simple e-commerce database
    # Table 1: Customers (id, name, email, age)
    customers_table = Table(
        num_rows=50,
        num_cols=4,
        num_features=3,
        column_names=["id", "name", "email", "age"],
        data_type_configs=[
            DataTypeConfig.primary_key_config(),  # id (PK)
            DataTypeConfig.categorical_config(num_categories=20),  # name
            DataTypeConfig.categorical_config(num_categories=30),  # email
            DataTypeConfig.float_config(),  # age
        ],
    )

    # Table 2: Products (id, name, price, category_id)
    products_table = Table(
        num_rows=100,
        num_cols=4,
        num_features=3,
        column_names=["id", "name", "price", "category_id"],
        data_type_configs=[
            DataTypeConfig.primary_key_config(),  # id (PK)
            DataTypeConfig.categorical_config(num_categories=25),  # name
            DataTypeConfig.float_config(),  # price
            DataTypeConfig.categorical_config(num_categories=10),  # category_id (FK)
        ],
    )

    # Table 3: Orders (id, customer_id, product_id, quantity, timestamp)
    orders_table = Table(
        num_rows=200,
        num_cols=5,
        num_features=2,
        column_names=["id", "customer_id", "product_id", "quantity", "timestamp"],
        data_type_configs=[
            DataTypeConfig.primary_key_config(),  # id (PK)
            DataTypeConfig.foreign_key_config(),  # customer_id (FK)
            DataTypeConfig.foreign_key_config(),  # product_id (FK)
            DataTypeConfig.float_config(),  # quantity
            DataTypeConfig.timestamp_config(),  # timestamp
        ],
        time_column=4,  # timestamp column
    )

    # Add tables to RDB
    rdb.add_table("customers", customers_table)
    rdb.add_table("products", products_table)
    rdb.add_table("orders", orders_table)

    # Define relationships
    # Orders.customer_id -> Customers.id
    customer_order_rel = Relationship("orders", 1, "customers", 0)
    # Orders.product_id -> Products.id
    product_order_rel = Relationship("orders", 2, "products", 0)
    # Products.category_id -> Categories.id (we'll create a categories table)
    # For now, let's assume category_id is just a categorical field

    # Add relationships
    rdb.add_relationship(customer_order_rel)
    rdb.add_relationship(product_order_rel)

    # Generate raw data for all tables
    raw_data_dict = {
        "customers": torch.randn(50, 4),
        "products": torch.randn(100, 4),
        "orders": torch.randn(200, 5),
    }

    # Initialize SCMs and generate processed data
    rdb.init_table_SCMs()
    processed_data_dict = rdb.generate_all_data_from_SCM()

    # Print RDB information
    print(f"RDB: {rdb}")
    print(f"Tables: {list(rdb.tables.keys())}")
    print(f"Relationships: {len(rdb.relationships)}")

    # Print table information
    for table_name, table in rdb.tables.items():
        print(f"\nTable '{table_name}':")
        print(f"  Shape: {processed_data_dict[table_name].shape}")
        print(
            f"  Data types: {[config.data_type.value for config in table.data_type_configs]}"
        )
        if table.is_time_table:
            print(f"  Time column: {table.time_column}")

    # Print relationship information
    print("\nRelationships:")
    for rel in rdb.relationships:
        print(f"  {rel}")

    # Test relationship validation
    print(
        f"\nRelationship validation: {rdb.validate_relationships(processed_data_dict)}"
    )

    # Test getting relationships for specific tables
    print(
        f"\nForeign keys for 'orders': {len(rdb.get_foreign_keys_for_table('orders'))}"
    )
    print(
        f"Primary keys for 'customers': {len(rdb.get_primary_keys_for_table('customers'))}"
    )

    # Test getting table info
    info = rdb.get_table_info()
    print("\nRDB Info:")
    print(f"  Name: {info['name']}")
    print(f"  Number of tables: {info['num_tables']}")
    print(f"  Number of relationships: {info['num_relationships']}")

    # Show sample data from each table
    print("\nSample data:")
    for table_name, data in processed_data_dict.items():
        print(f"\n{table_name} (first 5 rows):")
        print(data[:5])

    print("\nRDB test completed successfully!")

    def create_schema_graph(self):
        """
        Create a schema graph from the RDB's tables and relationships.

        Returns
        -------
        SchemaGraph
            Schema graph representing the RDB structure
        """
        from .task_generation_utils import SchemaGraph

        schema_graph = SchemaGraph()

        # Add all tables to the schema graph
        for table_name, table in self.tables.items():
            schema_graph.add_table(table_name, table)

        # Add all relationships to the schema graph
        for relationship in self.relationships:
            schema_graph.add_relationship(relationship)

        return schema_graph

    def select_neighbor_tables(
        self, center_table: str, max_neighbors: int = 2
    ) -> List[str]:
        """
        Randomly select 0 to max_neighbors neighbor tables for the given center table.

        Parameters
        ----------
        center_table : str
            The center table to find neighbors for
        max_neighbors : int
            Maximum number of neighbors to select (default: 2)

        Returns
        -------
        List[str]
            List of selected neighbor table names
        """
        if center_table not in self.tables:
            raise ValueError(f"Table '{center_table}' not found in RDB")

        # Get all possible neighbors
        all_neighbors = []

        # Find tables that have relationships with the center table
        for relationship in self.relationships:
            if relationship.from_table == center_table:
                all_neighbors.append(relationship.to_table)
            elif relationship.to_table == center_table:
                all_neighbors.append(relationship.from_table)

        # Remove duplicates
        all_neighbors = list(set(all_neighbors))

        # Randomly select 0 to max_neighbors neighbors
        num_neighbors = random.randint(0, min(max_neighbors, len(all_neighbors)))
        selected_neighbors = random.sample(all_neighbors, num_neighbors)

        return selected_neighbors

    def create_sub_schema_graph(self, center_table: str, neighbor_tables: List[str]):
        """
        Create a sub-schema graph containing only the center table and selected neighbors.

        Parameters
        ----------
        center_table : str
            The center table
        neighbor_tables : List[str]
            List of neighbor table names

        Returns
        -------
        SchemaGraph
            Sub-schema graph with only the specified tables and their relationships
        """
        from .task_generation_utils import SchemaGraph

        sub_schema = SchemaGraph(rdb=self)  # Pass RDB reference for direction checking
        selected_tables = [center_table] + neighbor_tables

        # Add selected tables
        for table_name in selected_tables:
            if table_name in self.tables:
                sub_schema.add_table(table_name, self.tables[table_name])

        # Add relationships between selected tables only
        for relationship in self.relationships:
            if (
                relationship.from_table in selected_tables
                and relationship.to_table in selected_tables
            ):
                sub_schema.add_relationship(relationship)

        return sub_schema

    def generate_focal_entities(self, table_name: str, num_samples: int = 10):
        """
        Generate random focal entities (samples) from the specified table.

        Parameters
        ----------
        table_name : str
            Name of the table to sample from
        num_samples : int
            Number of focal entities to generate

        Returns
        -------
        List[FocalEntity]
            List of focal entities
        """
        from .task_generation import FocalEntity

        if table_name not in self.tables:
            raise ValueError(f"Table '{table_name}' not found in RDB")

        table = self.tables[table_name]
        num_rows = table.num_rows

        # Generate random record indices
        if num_samples >= num_rows:
            # If we want more samples than available, use all records
            record_indices = list(range(num_rows))
        else:
            # Randomly sample without replacement
            record_indices = random.sample(range(num_rows), num_samples)

        # Create focal entities
        focal_entities = []
        for record_id in record_indices:
            focal_entity = FocalEntity(table_name, record_id)
            focal_entities.append(focal_entity)

        return focal_entities

    def generate_instance_graphs(
        self, center_table: str, neighbor_tables: List[str], num_samples: int = 10
    ):
        """
        Generate instance graphs for the given center table and neighbor tables.

        Parameters
        ----------
        center_table : str
            The center table name
        neighbor_tables : List[str]
            List of neighbor table names
        num_samples : int
            Number of focal entities to sample from the center table

        Returns
        -------
        List[InstanceGraph]
            List of instance graphs
        """
        from .task_generation import InstanceGraph

        # Create sub-schema graph
        sub_schema = self.create_sub_schema_graph(center_table, neighbor_tables)

        # Generate focal entities
        focal_entities = self.generate_focal_entities(center_table, num_samples)

        # Generate instance graphs
        instance_graphs = []
        for focal_entity in focal_entities:
            instance_graph = InstanceGraph(focal_entity, sub_schema)
            instance_graph.build_from_focal_entity(self)
            instance_graphs.append(instance_graph)

        return instance_graphs


def create_simple_ecommerce_rdb() -> RDB:
    """
    Create a simple e-commerce RDB with customers, products, and orders tables.

    Returns:
    --------
    RDB
        A configured RDB with tables and relationships
    """
    # Create RDB
    rdb = RDB("ecommerce_db")

    # Create tables
    customers_table = Table(
        num_rows=20,
        num_cols=3,
        num_features=2,
        column_names=["customer_id", "name", "age"],
        data_type_configs=[
            DataTypeConfig.primary_key_config(),  # customer_id (PK)
            DataTypeConfig.categorical_config(num_categories=10),  # name
            DataTypeConfig.float_config(),  # age
        ],
    )

    products_table = Table(
        num_rows=15,
        num_cols=3,
        num_features=2,
        column_names=["product_id", "name", "price"],
        data_type_configs=[
            DataTypeConfig.primary_key_config(),  # product_id (PK)
            DataTypeConfig.categorical_config(num_categories=8),  # name
            DataTypeConfig.float_config(),  # price
        ],
    )

    orders_table = Table(
        num_rows=30,
        num_cols=4,
        num_features=1,
        column_names=["order_id", "customer_id", "product_id", "quantity"],
        data_type_configs=[
            DataTypeConfig.primary_key_config(),  # order_id (PK)
            DataTypeConfig.foreign_key_config(),  # customer_id (FK)
            DataTypeConfig.foreign_key_config(),  # product_id (FK)
            DataTypeConfig.float_config(),  # quantity
        ],
    )

    # Add tables to RDB
    rdb.add_table("customers", customers_table)
    rdb.add_table("products", products_table)
    rdb.add_table("orders", orders_table)

    # Add relationships
    rdb.add_relationship(
        Relationship("orders", 1, "customers", 0)
    )  # orders.customer_id -> customers.customer_id
    rdb.add_relationship(
        Relationship("orders", 2, "products", 0)
    )  # orders.product_id -> products.product_id

    return rdb


def example_usage():
    """
    Example usage of the RDB class.
    """
    print("Creating simple e-commerce RDB...")
    rdb = create_simple_ecommerce_rdb()

    # Initialize SCMs and generate processed data
    rdb.init_table_SCMs()
    processed_data = rdb.generate_all_data_from_SCM()

    # Print summary
    print(f"RDB: {rdb.name}")
    print(f"Tables: {list(rdb.tables.keys())}")
    print(f"Relationships: {len(rdb.relationships)}")

    # Validate relationships
    is_valid = rdb.validate_relationships(processed_data)
    print(f"Relationships valid: {is_valid}")

    return rdb, processed_data
