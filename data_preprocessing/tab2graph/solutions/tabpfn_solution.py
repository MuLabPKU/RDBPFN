import logging
from typing import Tuple, Dict, Optional, List, Any, Literal
from pathlib import Path
import wandb
from enum import Enum
import warnings
import pydantic

import numpy as np
import pandas as pd
import torch
from tqdm import tqdm
from dbinfer_bench import DBBRDBDataset, DBBColumnDType, DBBTaskType
from tabpfn import TabPFNClassifier, TabPFNRegressor

from .base_tab import (
    TabularMLSolution,
    FitSummary,
    tabml_solution,
)
from .tabular_dataset_config import TabularDatasetConfig
from ..device import DeviceInfo
from .. import yaml_utils
from . import negative_sampler as NS
from ..evaluator import get_metric_fn

logger = logging.getLogger(__name__)
logger.setLevel("DEBUG")

np.random.seed(42)

__all__ = ["TabPFNSolution", "TabPFNSolutionConfig"]


class TabPFNSolutionConfig(pydantic.BaseModel):
    # Number of estimators in the TabPFN model.
    n_estimators: int = 8
    # Maximum number of training samples to use. None means no limit.
    max_train_samples: Optional[int] = None
    # Strategy for selecting training samples.
    training_sample_selection_strategy: Literal["graph", "random", "stratified"] = "random"
    # Whether to use foreign keys as features.
    use_foreign_key_feature: bool = False
    # Negative samples per positive
    negative_sampling_ratio: int = 5
    # Batch size for evaluation
    eval_batch_size: int = 10000
    # Use auto TabPFN classifier/regressor
    use_auto_tabpfn: bool = False
    # Feature selection parameters
    max_feature_product: Optional[int] = 2000000  # Maximum #rows * #features. None means no limit.
    feature_selection_seed: int = 42  # Random seed for feature selection


@tabml_solution
class TabPFNSolution(TabularMLSolution):
    """Scikit-learn Random Forest solution class."""

    config_class = TabPFNSolutionConfig
    name = "tabpfn"

    def __init__(self, solution_config: TabPFNSolutionConfig, data_config: TabularDatasetConfig):
        super().__init__(solution_config, data_config)
        self.predictor = None
        self.selected_feature_indices = None  # Store selected feature indices for consistency

    def fit(
        self, dataset: DBBRDBDataset, task_name: str, ckpt_path: Path, device: DeviceInfo
    ) -> FitSummary:
        _dataset = dataset.get_task(task_name)
        train_feat_store, valid_feat_store = _dataset.train_set, _dataset.validation_set

        if self.data_config.task.task_type == DBBTaskType.retrieval:
            train_feat_store = self.negative_sampling(train_feat_store)

        train_feat_dict, feat_meta = self.adjust_features(train_feat_store)
        train_df = pd.DataFrame(train_feat_dict, copy=False)

        # Prepare features and labels
        X_train, y_train = self.extract_features_and_label(train_df, is_training=True)

        self.X_train = X_train
        self.y_train = y_train

        self.select_and_fit(target_set=train_df)

        self.checkpoint(ckpt_path)

        # Calculate metrics
        # train_metric = self.calculate_metric(X_train, y_train, train_df)
        train_metric = 0.0

        # Log to wandb
        logger.info(f"Training metric: {train_metric}")

        summary = FitSummary()
        summary.val_metric = 0.0  # Placeholder, as we don't calculate validation metric here
        summary.train_metric = train_metric

        return summary

    def select_and_fit(
        self,
        target_set: pd.DataFrame,
    ):
        # Initialize the appropriate model based on the task
        if self.data_config.task.task_type in [DBBTaskType.classification, DBBTaskType.retrieval]:
            if self.solution_config.use_auto_tabpfn:
                # Use AutoTabPFNClassifier for classification and retrieval tasks
                logger.info("Using AutoTabPFNClassifier for classification/retrieval task.")
                from tabpfn_extensions.post_hoc_ensembles.sklearn_interface import (
                    AutoTabPFNClassifier,
                )

                self.predictor = AutoTabPFNClassifier(max_time=120, device="cuda")
            else:
                self.predictor = TabPFNClassifier(
                    n_estimators=self.solution_config.n_estimators,
                    n_jobs=-1,
                    random_state=42,
                    inference_config={"SUBSAMPLE_SAMPLES": 10000},
                    ignore_pretraining_limits=True,
                    # inference_precision=torch.float32,
                    # fit_mode="batched",
                )
        elif self.data_config.task.task_type == DBBTaskType.regression:
            if self.solution_config.use_auto_tabpfn:
                # Use AutoTabPFNRegressor for regression tasks
                logger.info("Using AutoTabPFNRegressor for regression task.")
                from tabpfn_extensions.post_hoc_ensembles.sklearn_interface import (
                    AutoTabPFNRegressor,
                )

                self.predictor = AutoTabPFNRegressor(max_time=120, device="cuda")
            else:
                self.predictor = TabPFNRegressor(
                    n_estimators=self.solution_config.n_estimators,
                    n_jobs=-1,
                    random_state=42,
                    inference_config={"SUBSAMPLE_SAMPLES": 10000},
                    ignore_pretraining_limits=True,
                )
        else:
            raise ValueError(f"Unsupported task type: {self.data_config.task.task_type}")

        if self.solution_config.max_train_samples:
            # Select training samples
            selected_X_train, selected_y_train, _ = self.select_training_samples(
                self.X_train,
                self.y_train,
                target_set=target_set,
                max_samples=self.solution_config.max_train_samples,
                strategy=self.solution_config.training_sample_selection_strategy,
            )
        else:
            # Use all training samples
            selected_X_train = self.X_train
            selected_y_train = self.y_train

        logger.info(
            f"Fitting on {len(selected_X_train)} samples. # of features: {selected_X_train.shape[1]}"
        )
        logger.info(
            f"Label distribution in training set: {np.unique(selected_y_train, return_counts=True)}"
        )

        # Fit the model
        print(selected_X_train.shape)
        print(selected_y_train.shape)
        self.predictor.fit(selected_X_train, selected_y_train)

    def evaluate(
        self,
        table: Dict[str, np.ndarray],
        device: DeviceInfo,
    ) -> float:
        feat_dict, _ = self.adjust_features(table)
        test_df = pd.DataFrame(feat_dict, copy=False)

        X_test, y_test = self.extract_features_and_label(test_df, is_training=False)

        # Select a random subset of X_test
        # idx = np.random.choice(
        #     len(X_test), min(100000, len(X_test)), replace=False)
        # X_test = X_test.iloc[idx]
        # y_test = y_test[idx]
        # test_df = test_df.iloc[idx]
        # logger.info(f"Evaluating on {len(X_test)} samples. # of features: {X_test.shape[1]}")

        metric = self.calculate_metric(X_test, y_test, test_df)
        logger.info("Evaluation metric: %.4f", metric)
        return metric

    def checkpoint(self, ckpt_path: Path) -> None:
        # Minimally implemented - just saving configuration
        ckpt_path = Path(ckpt_path)
        ckpt_path.mkdir(parents=True, exist_ok=True)
        yaml_utils.save_pyd(self.solution_config, ckpt_path / "solution_config.yaml")
        yaml_utils.save_pyd(self.data_config, ckpt_path / "data_config.yaml")

        # Save selected feature indices for consistency between training and evaluation
        if self.selected_feature_indices is not None:
            np.save(ckpt_path / "selected_feature_indices.npy", self.selected_feature_indices)

        # Note: For production, you'd want to save the model using joblib or pickle
        # but we're skipping that per requirements

    def load_from_checkpoint(self, ckpt_path: Path) -> None:
        # Minimally implemented - just loading configuration
        ckpt_path = Path(ckpt_path)
        self.solution_config = yaml_utils.load_pyd(
            self.config_class, ckpt_path / "solution_config.yaml"
        )
        self.data_config = yaml_utils.load_pyd(TabularDatasetConfig, ckpt_path / "data_config.yaml")

        # Load selected feature indices for consistency between training and evaluation
        feature_indices_path = ckpt_path / "selected_feature_indices.npy"
        if feature_indices_path.exists():
            self.selected_feature_indices = np.load(feature_indices_path).tolist()

        # Note: For production, you'd want to load the model using joblib or pickle
        # but we're skipping that per requirements

    def adjust_features(
        self, feat_dict: Dict[str, np.ndarray]
    ) -> Tuple[Dict[str, np.ndarray], Dict]:
        """Adjust features suitable for sklearn models."""
        logger.info("Adapting features ...")
        new_feat_dict = {}
        feat_meta = {}  # Simple metadata placeholder

        for name, feat in feat_dict.items():
            if name not in self.data_config.features:
                logger.info(f"Ignoring feature '{name}' not in data config")
                continue

            if name in [
                self.data_config.task.key_prediction_label_column,
                self.data_config.task.key_prediction_query_idx_column,
                self.data_config.task.target_column,
            ]:
                new_feat_dict[name] = feat
            else:
                dtype = self.data_config.features[name].dtype
                if dtype in [
                    DBBColumnDType.category_t,
                    DBBColumnDType.primary_key,
                    DBBColumnDType.foreign_key,
                ]:
                    new_feat_dict[name] = feat
                elif dtype == DBBColumnDType.timestamp_t:
                    new_feat_dict[name] = feat
                elif dtype == DBBColumnDType.float_t:
                    in_size = self.data_config.features[name].extra_fields.get("in_size", 1)
                    if in_size == 1:
                        new_feat_dict[name] = feat
                    else:
                        continue  # Skip multi-dimensional features for now
                else:
                    logger.info(f"Ignore feature '{name}' of type {dtype}")

        return new_feat_dict, feat_meta

    def extract_features_and_label(
        self, df: pd.DataFrame, is_training: bool = True
    ) -> Tuple[pd.DataFrame, np.ndarray]:
        """Extract features and labels from the dataframe."""
        label_name = self.get_label_name()

        # Get feature names excluding target/label columns
        feature_cols = []
        for col in df.columns:
            if col in [
                self.data_config.task.target_column,
                self.data_config.task.key_prediction_label_column,
                self.data_config.task.key_prediction_query_idx_column,
            ]:
                continue

            # Skip primary keys
            if (
                col in self.data_config.features
                and self.data_config.features[col].dtype == DBBColumnDType.primary_key
            ):
                continue

            # Skip foreign keys if configured
            if (
                not self.solution_config.use_foreign_key_feature
                and col in self.data_config.features
                and self.data_config.features[col].dtype == DBBColumnDType.foreign_key
            ):
                continue

            feature_cols.append(col)

        # Apply feature selection if needed
        selected_feature_cols = self.select_features_if_needed(feature_cols, len(df), is_training)

        X = df[selected_feature_cols]  # Return as DataFrame instead of .values
        y = df[label_name].values

        return X, y

    def select_features_if_needed(
        self, feature_columns: List[str], num_rows: int, is_training: bool = True
    ) -> List[str]:
        """
        Select features if the product of #rows * #features exceeds max_feature_product.

        Args:
            feature_columns: List of feature column names
            num_rows: Number of rows in the dataset
            is_training: Whether this is called during training (True) or evaluation (False)

        Returns:
            List of selected feature column names
        """
        print(f"feature_columns: {feature_columns}")
        print(f"num_rows: {num_rows}")
        print(f"is_training: {is_training}")
        print(f"max_feature_product: {self.solution_config.max_feature_product}")
        print(f"selected_feature_indices: {self.selected_feature_indices}")

        # Check if we have a feature limit
        if self.solution_config.max_feature_product is None:
            return feature_columns

        num_features = len(feature_columns)
        current_product = num_rows * num_features

        logger.info(
            f"Current feature product: {current_product} (rows: {num_rows}, features: {num_features})"
        )

        # If we're under the limit, return all features
        if current_product <= self.solution_config.max_feature_product:
            if is_training:
                # Store all feature indices for consistency
                self.selected_feature_indices = list(range(num_features))
                return feature_columns

        # Calculate maximum number of features we can use
        max_features = self.solution_config.max_feature_product // num_rows
        max_features = max(1, max_features)  # Ensure at least 1 feature

        logger.info(
            f"Feature product {current_product} exceeds limit {self.solution_config.max_feature_product}. "
            f"Selecting {max_features} out of {num_features} features."
        )

        if is_training:
            # During training, randomly select features and store indices
            np.random.seed(self.solution_config.feature_selection_seed)
            selected_indices = np.random.choice(
                num_features, size=max_features, replace=False
            ).tolist()
            selected_indices.sort()  # Sort for reproducibility
            self.selected_feature_indices = selected_indices
        else:
            # During evaluation, use the same features as training
            if self.selected_feature_indices is None:
                raise ValueError(
                    "No feature indices stored from training. "
                    "Make sure to call this method during training first."
                )
            selected_indices = self.selected_feature_indices

        # Return selected feature columns
        selected_features = [feature_columns[i] for i in selected_indices]

        logger.info(
            f"Selected features ({len(selected_features)}): {selected_features[:10]}{'...' if len(selected_features) > 10 else ''}"
        )

        return selected_features

    def get_label_name(self) -> str:
        """Get the appropriate label column name based on task type."""
        if self.data_config.task.task_type == DBBTaskType.retrieval:
            return self.data_config.task.key_prediction_label_column
        else:
            return self.data_config.task.target_column

    def calculate_metric(self, X: pd.DataFrame, y: np.ndarray, df: pd.DataFrame) -> float:
        """Calculate the appropriate metric based on task type using batch processing."""
        metric_fn = get_metric_fn(self.data_config.task)
        batch_size = self.solution_config.eval_batch_size

        # Process data in batches to avoid memory issues
        num_samples = len(X)
        num_batches = (num_samples + batch_size - 1) // batch_size  # Ceiling division

        all_preds = []

        # Process each batch
        for i in tqdm(range(num_batches), desc="Computing predictions in batches"):
            start_idx = i * batch_size
            end_idx = min((i + 1) * batch_size, num_samples)

            X_batch = X[start_idx:end_idx]

            # self.select_and_fit(target_set=X_batch)

            # Get predictions based on task type
            if self.data_config.task.task_type == DBBTaskType.classification:
                # For classification tasks
                batch_pred = self.predictor.predict_proba(X_batch)
                all_preds.append(batch_pred)
            elif self.data_config.task.task_type == DBBTaskType.retrieval:
                # For retrieval tasks
                batch_pred = self.predictor.predict_proba(X_batch)
                batch_pred = batch_pred[:, 1]
                # batch_pred = np.full((X_batch.shape[0],), 0.5)  # Dummy prediction for retrieval
                all_preds.append(batch_pred)
            else:
                # For regression tasks
                batch_pred = self.predictor.predict(X_batch)
                all_preds.append(batch_pred)

        # Combine all batch predictions
        if self.data_config.task.task_type == DBBTaskType.classification:
            # For multi-class classification, need to handle the 2D array
            pred = np.vstack(all_preds)
        else:
            # For binary classification or regression
            pred = np.concatenate(all_preds)

        pred, label = torch.tensor(pred), torch.tensor(y)

        if self.data_config.task.task_type == DBBTaskType.retrieval:
            index_column = self.data_config.task.key_prediction_query_idx_column
            index = df[index_column].values
            index = torch.tensor(index)
        else:
            index = None

        metric = metric_fn(index, pred, label).item()
        return metric

    def select_training_samples(
        self,
        X: pd.DataFrame,
        y: np.ndarray,
        target_set: pd.DataFrame,
        max_samples: int,
        strategy: Literal["graph", "random", "stratified"] = "graph",
    ) -> Tuple[pd.DataFrame, np.ndarray, np.ndarray]:
        """Select a subset of training samples based on the task type and max_samples.

        Args:
            X: Feature DataFrame
            y: Target labels/values
            max_samples: Maximum number of samples to retain

        Returns:
            Tuple of (selected_X, selected_y, index)
        """
        if strategy == "graph":
            if not self.solution_config.use_foreign_key_feature:
                raise ValueError(
                    "Graph-based sampling strategy requires use_foreign_key_feature=True. "
                    "Set this in the solution configuration."
                )
            return self.select_samples_graph_based(X, y, target_set, max_samples)
        elif strategy == "random":
            return self.downsample_training_set(X, y, max_samples, stratified_sampling=False)
        elif strategy == "stratified":
            return self.downsample_training_set(X, y, max_samples, stratified_sampling=True)
        else:
            raise ValueError(f"Unsupported sampling strategy: {strategy}")

    def select_samples_graph_based(
        self,
        X: pd.DataFrame,
        y: np.ndarray,
        target_set: pd.DataFrame,
        max_samples: int,
    ) -> Tuple[pd.DataFrame, np.ndarray, np.ndarray]:
        """Select training samples based on graph connectivity through foreign keys.

        Algorithm:
        1. For each foreign key column:
           a. Find unique IDs in the target set
           b. Select rows from X that have matching IDs (assuming they already have earlier timestamps)
        2. Sample from these connected rows up to max_samples limit

        Args:
            X: Feature DataFrame
            y: Target labels/values
            target_set: DataFrame containing the target set (e.g., validation/test set)
            max_samples: Maximum number of samples to select

        Returns:
            Tuple of (selected_X, selected_y, selected_indices)
        """
        logger.info("Using graph-based training sample selection strategy")

        # If X has fewer samples than max_samples, return all
        if len(X) <= max_samples:
            return X, y, None

        # Identify foreign key columns
        foreign_key_cols = []
        for col in X.columns:
            if (
                col in self.data_config.features
                and self.data_config.features[col].dtype == DBBColumnDType.foreign_key
            ):
                foreign_key_cols.append(col)

        if not foreign_key_cols:
            logger.warning("No foreign key columns found. Falling back to random sampling.")
            return self.downsample_training_set(X, y, max_samples, stratified_sampling=False)

        logger.info(f"Found {len(foreign_key_cols)} foreign key columns: {foreign_key_cols}")

        # Find valid rows for each foreign key column
        all_valid_indices = []
        samples_per_fk = max_samples // len(foreign_key_cols)

        for fk_col in foreign_key_cols:
            # Extract unique foreign keys from target set
            if fk_col in target_set.columns:
                target_fk_values = set(target_set[fk_col].unique())
                logger.info(
                    f"Column {fk_col}: Found {len(target_fk_values)} unique foreign keys in target set"
                )

                # Use pandas's isin for vectorized matching instead of iterating
                matching_mask = X[fk_col].isin(target_fk_values)
                matching_indices = np.where(matching_mask)[0]

                logger.info(f"Found {len(matching_indices)} matching indices for column {fk_col}")

                # Sample from matching indices if necessary
                if len(matching_indices) > samples_per_fk:
                    # If time column exists, prioritize the latest samples
                    time_column = self.data_config.task.time_column
                    if time_column and time_column in X.columns:
                        logger.info(
                            f"Using time column '{time_column}' to select the latest samples"
                        )
                        # Get time values for the matching indices
                        times = X.iloc[matching_indices][time_column].values
                        # Get indices of samples sorted by time (descending)
                        sorted_indices = np.argsort(-times)  # Negate for descending order
                        # Select the top samples_per_fk samples (the most recent ones)
                        latest_indices = sorted_indices[:samples_per_fk]
                        matching_indices = matching_indices[latest_indices]
                    else:
                        # If no time column, randomly sample as before
                        matching_indices = np.random.choice(
                            matching_indices, samples_per_fk, replace=False
                        )

                # Add to our collection of valid indices
                all_valid_indices.append(matching_indices)

        # Flatten the list of arrays into a single array of indices
        all_valid_indices = np.unique(np.concatenate(all_valid_indices))

        # If we couldn't find enough connected samples, supplement with random samples
        if len(all_valid_indices) < max_samples:
            logger.info(
                f"Found only {len(all_valid_indices)} connected samples, adding random samples to reach {max_samples}"
            )

            remaining_samples = max_samples - len(all_valid_indices)
            additional_indices = np.random.choice(len(X), remaining_samples, replace=False)
            selected_indices = np.concatenate([all_valid_indices, additional_indices])
        elif len(all_valid_indices) > max_samples:
            selected_indices = np.random.choice(all_valid_indices, max_samples, replace=False)
        else:
            selected_indices = all_valid_indices

        logger.info(f"Final selected sample size: {len(selected_indices)}")

        # Return selected samples
        return X.iloc[selected_indices], y[selected_indices], selected_indices

    def negative_sampling(self, array_dict: Dict[str, np.ndarray]) -> Dict[str, np.ndarray]:
        target_column_name = self.data_config.task.target_column
        target_column_capacity = self.data_config.features[target_column_name].extra_fields[
            "capacity"
        ]
        array_dict = {k: torch.tensor(v) for k, v in array_dict.items()}
        array_dict = NS.negative_sampling(
            array_dict,
            self.solution_config.negative_sampling_ratio,
            target_column_name,
            target_column_capacity,
            self.data_config.task.key_prediction_label_column,
            self.data_config.task.key_prediction_query_idx_column,
            shuffle_rest_columns=True,
        )
        return array_dict

    def downsample_training_set(
        self, X: pd.DataFrame, y: np.ndarray, max_samples: int, stratified_sampling=False
    ) -> Tuple[pd.DataFrame, np.ndarray, np.ndarray]:
        """Downsample data to max_samples while preserving class balance for classification tasks.

        Args:
            X: Feature DataFrame
            y: Target labels/values
            max_samples: Maximum number of samples to retain
            stratified_sampling: If True, maintain class balance. If False, sample randomly
                                 but ensure at least one sample per class for classification tasks.

        Returns:
            Tuple of (downsampled_X, downsampled_y, idx)
        """
        if len(X) <= max_samples:
            return X, y, None

        logger.info(f"Downsampling training set from {len(X)} to {max_samples} samples.")

        if self.data_config.task.task_type == DBBTaskType.regression:
            # For regression tasks, we can just sample randomly
            idx = np.random.choice(len(X), max_samples, replace=False)
            return X.iloc[idx], y[idx], idx

        # For classification tasks, ensure at least one sample per class regardless of stratified_sampling
        if not stratified_sampling:
            # Even with random sampling, ensure at least one sample per class
            unique_labels = np.unique(y)
            n_classes = len(unique_labels)

            # First, pick one sample from each class
            selected_indices = []
            for label in unique_labels:
                class_indices = np.where(y == label)[0]
                if len(class_indices) > 0:
                    selected_idx = np.random.choice(class_indices, 1)[0]
                    selected_indices.append(selected_idx)

            # Then randomly sample the rest
            remaining_samples = max_samples - len(selected_indices)
            if remaining_samples > 0:
                # Create mask to exclude already selected indices
                mask = np.ones(len(X), dtype=bool)
                mask[selected_indices] = False
                eligible_indices = np.where(mask)[0]

                if len(eligible_indices) > 0:
                    additional_indices = np.random.choice(
                        eligible_indices,
                        min(remaining_samples, len(eligible_indices)),
                        replace=False,
                    )
                    selected_indices.extend(additional_indices)

            # Shuffle to avoid having all samples of one class together
            np.random.shuffle(selected_indices)
            idx = np.array(selected_indices)

            # Log the class distribution in the sample
            sampled_dist = np.unique(y[idx], return_counts=True)
            logger.info(f"Random sample class distribution: {dict(zip(*sampled_dist))}")

            return X.iloc[idx], y[idx], idx

        else:
            # Get the unique labels and their counts
            unique_labels, label_counts = np.unique(y, return_counts=True)
            logger.info(f"Original label distribution: {dict(zip(unique_labels, label_counts))}")

            # Calculate samples per class, ensuring all classes have at least one sample
            n_classes = len(unique_labels)
            # Determine samples per class with a minimum of 1
            samples_per_class = max(1, max_samples // n_classes)

            # Sample from each class
            balanced_indices = []
            remaining_indices = []

            for label in unique_labels:
                class_indices = np.where(y == label)[0]
                if len(class_indices) == 0:
                    continue

                # If we have fewer samples than samples_per_class, take all of them (no replacement)
                if len(class_indices) <= samples_per_class:
                    balanced_indices.extend(class_indices)
                else:
                    # Sample without replacement
                    sampled_indices = np.random.choice(
                        class_indices, samples_per_class, replace=False
                    )
                    balanced_indices.extend(sampled_indices)

                    # Store unused indices for potential additional sampling
                    mask = np.ones(len(class_indices), dtype=bool)
                    mask[np.isin(class_indices, sampled_indices)] = False
                    remaining_indices.extend(class_indices[mask])

            # If we haven't reached max_samples, sample from remaining indices
            samples_needed = max_samples - len(balanced_indices)
            if samples_needed > 0 and len(remaining_indices) > 0:
                # Sample without replacement if possible, otherwise with replacement
                additional_samples = np.random.choice(
                    remaining_indices, min(samples_needed, len(remaining_indices)), replace=False
                )
                balanced_indices.extend(additional_samples)
                logger.info(
                    f"Added {len(additional_samples)} additional samples to balance the dataset"
                )

            # Shuffle the indices to avoid having all samples of one class together
            np.random.shuffle(balanced_indices)

            # Limit to max_samples in case we somehow exceeded it
            balanced_indices = balanced_indices[:max_samples]
            idx = np.array(balanced_indices)

            # Log the new distribution
            new_label_counts = np.unique(y[idx], return_counts=True)
            logger.info(f"Balanced label distribution: {dict(zip(*new_label_counts))}")

            # Apply the sampling
            return X.iloc[idx], y[idx], idx
