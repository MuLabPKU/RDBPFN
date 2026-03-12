"""Classifier factory functions for evaluation."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any
import sys
import importlib

import torch
from omegaconf import OmegaConf
from sklearn.ensemble import RandomForestClassifier
import numpy as np

try:
    from xgboost import XGBClassifier
except ImportError:
    XGBClassifier = None

try:
    from tabpfn import TabPFNClassifier
    from tabpfn.constants import ModelVersion
except ImportError:
    TabPFNClassifier = None

try:
    from tabicl import TabICLClassifier
except ImportError:
    TabICLClassifier = None

try:
    from autogluon.tabular import TabularPredictor
except ImportError:
    TabularPredictor = None

LimiXPredictor = None

try:
    from huggingface_hub import hf_hub_download
except ImportError:
    hf_hub_download = None

from .models import ModelConfig, build_model, load_checkpoint, build_classifier


def _build_nanopfn_factory(
    model_config: ModelConfig, device: torch.device, checkpoint_path: Path | None
):
    """Build a factory function that creates NanoPFN classifiers."""
    if checkpoint_path is None:
        raise ValueError("checkpoint_path required for NanoTabPFN evaluation")
    if not checkpoint_path.exists() or not checkpoint_path.is_file():
        raise ValueError(
            f"checkpoint_path must be an existing file, got {checkpoint_path}"
        )
    model = build_model(model_config)
    load_checkpoint(model, checkpoint_path, str(device))
    model.to(device)
    model.eval()

    def factory():
        return build_classifier(model, device, model_config)

    return factory


def _build_random_forest_factory():
    """Build a factory function that creates RandomForest classifiers."""

    def factory():
        return RandomForestClassifier(
            n_estimators=500,
            max_depth=None,
            n_jobs=-1,
            random_state=0,
        )

    return factory


def _build_xgboost_factory():
    """Build a factory function that creates XGBoost classifiers."""
    if XGBClassifier is None:
        raise ImportError("xgboost is not installed.")

    def factory():
        return XGBClassifier(
            tree_method="hist",
            n_estimators=5000,
            learning_rate=0.01,
            max_depth=12,
            subsample=0.8,
            colsample_bytree=0.8,
            n_jobs=-1,
        )

    return factory


def _build_tabpfn_factory(
    model_version: ModelVersion, config: "TabPFNConfig | None" = None
):
    """Build a factory function that creates TabPFN classifiers."""
    if TabPFNClassifier is None:
        raise ImportError("tabpfn is not installed.")

    def factory():
        if model_version == ModelVersion.V2:
            if config is None or config.n_estimators is None:
                return TabPFNClassifier.create_default_for_version(ModelVersion.V2)
            return TabPFNClassifier.create_default_for_version(
                ModelVersion.V2, n_estimators=config.n_estimators
            )
        if model_version == ModelVersion.V2_5:
            if config is None or config.n_estimators is None:
                return TabPFNClassifier()
            return TabPFNClassifier(n_estimators=config.n_estimators)
        raise ValueError(f"Unknown model version {model_version}")

    return factory


def _build_tabicl_factory(model_version, config: "TabICLConfig | None" = None):
    """Build a factory function that creates TabiCL classifiers."""
    if TabICLClassifier is None:
        raise ImportError("tabicl is not installed.")

    def factory():
        if model_version == "v1.1":
            if config is None or config.n_estimators is None:
                return TabICLClassifier()
            return TabICLClassifier(n_estimators=config.n_estimators)
        if model_version == "v1":
            kwargs = {"checkpoint_version": "tabicl-classifier-v1-0208.ckpt"}
            if config is not None and config.n_estimators is not None:
                kwargs["n_estimators"] = config.n_estimators
            return TabICLClassifier(**kwargs)
        raise ValueError(f"Unknown model version {model_version}")

    return factory


@dataclass
class AutoGluonConfig:
    preset: str | None = None
    time_limit: int | None = None
    hyperparameters: dict[str, Any] | None = None


@dataclass
class TabPFNConfig:
    n_estimators: int | None = None


@dataclass
class TabICLConfig:
    n_estimators: int | None = None


@dataclass
class LimiXConfig:
    model_path: str | None = None
    inference_config: str | None = None
    config_name: str | None = None
    variant: str | None = None


class _LimiXClassifier:
    """Wrapper to make LimiX predictor sklearn-compatible."""

    def __init__(self, predictor: Any):
        self._predictor = predictor
        self._X_train: np.ndarray | None = None
        self._y_train: np.ndarray | None = None

    def fit(self, X, y):
        self._X_train = np.asarray(X, dtype=np.float32)
        self._y_train = np.asarray(y, dtype=np.int64)
        return self

    def predict_proba(self, X):
        if self._X_train is None or self._y_train is None:
            raise ValueError("LimiXClassifier must be fit before predict_proba().")
        X_test = np.asarray(X, dtype=np.float32)
        return self._predictor.predict(
            self._X_train, self._y_train, X_test, task_type="Classification"
        )


class _AutoGluonClassifier:
    """Wrapper to make AutoGluon TabularPredictor sklearn-compatible."""

    def __init__(
        self,
        preset: str | None = None,
        time_limit: int | None = None,
        hyperparameters: dict[str, Any] | None = None,
    ):
        self._predictor = None
        self._preset = preset
        self._time_limit = time_limit
        self._hyperparameters = hyperparameters

    def fit(self, X, y):
        import pandas as pd

        df = pd.DataFrame(X)
        df["label"] = y
        fit_kwargs = {}
        if self._preset:
            fit_kwargs["presets"] = self._preset
        if self._time_limit is not None:
            fit_kwargs["time_limit"] = self._time_limit
        if self._hyperparameters:
            hyperparams = (
                OmegaConf.to_container(self._hyperparameters, resolve=True)
                if OmegaConf.is_config(self._hyperparameters)
                else self._hyperparameters
            )
            fit_kwargs["hyperparameters"] = hyperparams
        self._predictor = TabularPredictor(
            label="label",
            verbosity=0,
        ).fit(
            df,
            **fit_kwargs,
        )
        return self

    def predict_proba(self, X):
        import pandas as pd

        df = pd.DataFrame(X)
        proba = self._predictor.predict_proba(df)
        return proba.values


def _build_autogluon_factory(config: AutoGluonConfig | None = None):
    """Build a factory function that creates AutoGluon classifiers."""
    if TabularPredictor is None:
        raise ImportError("autogluon is not installed.")

    def factory():
        if config is None:
            return _AutoGluonClassifier()
        return _AutoGluonClassifier(
            preset=config.preset,
            time_limit=config.time_limit,
            hyperparameters=config.hyperparameters,
        )

    return factory


def _build_limix_factory(
    model_name: str,
    config: LimiXConfig | None = None,
    device: torch.device | None = None,
):
    """Build a factory function that creates LimiX classifiers."""
    if LimiXPredictor is None:
        limix_root = Path(__file__).resolve().parents[1] / "LimiX"
        if limix_root.exists():
            sys.path.insert(0, str(limix_root))
        try:
            module = importlib.import_module("inference.predictor")
            _LimiXPredictor = getattr(module, "LimiXPredictor")
        except Exception as exc:  # pylint: disable=broad-except
            raise ImportError("LimiX is not available in this environment.") from exc
    else:
        _LimiXPredictor = LimiXPredictor

    if hf_hub_download is None:
        raise ImportError("huggingface_hub is required for LimiX model download.")

    repo_root = Path(__file__).resolve().parents[1]
    limix_root = repo_root / "LimiX"
    # default_config = limix_root / "config/cls_default_noretrieval.json"
    default_config = limix_root / "config/cls_default_1est.json"
    resolved_config = config or LimiXConfig()
    variant = resolved_config.variant
    if variant is None:
        if "16" in model_name:
            variant = "16M"
        elif "2" in model_name:
            variant = "2M"
        else:
            variant = "16M"

    if resolved_config.inference_config:
        inference_config = Path(resolved_config.inference_config)
    elif resolved_config.config_name:
        inference_config = limix_root / "config" / resolved_config.config_name
    else:
        inference_config = default_config
    if not inference_config.is_absolute():
        candidate = (repo_root / inference_config).resolve()
        if candidate.exists():
            inference_config = candidate
        else:
            inference_config = (limix_root / inference_config).resolve()
    if not inference_config.exists():
        raise ValueError(f"LimiX inference_config not found: {inference_config}")

    model_path = resolved_config.model_path
    if model_path is None:
        if variant == "2M":
            repo_id = "stableai-org/LimiX-2M"
            filename = "LimiX-2M.ckpt"
        else:
            repo_id = "stableai-org/LimiX-16M"
            filename = "LimiX-16M.ckpt"
        cache_dir = limix_root / "cache"
        cache_dir.mkdir(parents=True, exist_ok=True)
        model_path = hf_hub_download(
            repo_id=repo_id, filename=filename, local_dir=str(cache_dir)
        )
    else:
        model_path_path = Path(model_path).expanduser()
        if not model_path_path.is_absolute():
            model_path_path = repo_root / model_path_path
        model_path_path = model_path_path.resolve()
        if not model_path_path.exists():
            raise ValueError(f"LimiX model_path not found: {model_path_path}")
        model_path = str(model_path_path)

    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def factory():
        predictor_kwargs = {}
        predictor = _LimiXPredictor(
            device=device,
            model_path=str(model_path),
            inference_config=str(inference_config),
        )
        return _LimiXClassifier(predictor)

    return factory


def build_classifier_factory(
    model_name: str,
    model_config: ModelConfig | None = None,
    autogluon_config: AutoGluonConfig | None = None,
    tabpfn_config: TabPFNConfig | None = None,
    tabicl_config: TabICLConfig | None = None,
    limix_config: LimiXConfig | None = None,
    device: torch.device | None = None,
    checkpoint_path: Path | None = None,
):
    """Build a classifier factory based on model name.

    Args:
        model_name: Name of the model (nanopfn, random_forest, xgboost, tabpfn, limix)
        model_config: Model configuration (used for nanopfn)
        device: Torch device (used for nanopfn)
        checkpoint_path: Path to checkpoint file (used for nanopfn)

    Returns:
        A factory function that creates classifier instances.
    """
    name = model_name.lower()
    if name == "nanopfn":
        return _build_nanopfn_factory(model_config, device, checkpoint_path)
    if name == "random_forest":
        return _build_random_forest_factory()
    if name == "xgboost":
        return _build_xgboost_factory()
    if name == "tabpfnv2.5":
        return _build_tabpfn_factory(ModelVersion.V2_5, tabpfn_config)
    if name == "tabpfnv2":
        return _build_tabpfn_factory(ModelVersion.V2, tabpfn_config)
    if name == "tabiclv1.1":
        return _build_tabicl_factory("v1.1", tabicl_config)
    if name == "tabiclv1":
        return _build_tabicl_factory("v1", tabicl_config)
    if name == "autogluon":
        return _build_autogluon_factory(autogluon_config)
    if name in {"limix", "limix2m", "limix16m"}:
        return _build_limix_factory(name, limix_config, device)
    raise ValueError(f"Unknown model name {model_name}")
