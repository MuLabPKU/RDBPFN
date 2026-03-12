from pathlib import Path
import typer
import logging
import wandb
import os
import numpy as np

import dbinfer_bench as dbb
from ..device import DeviceInfo, get_device_info
from ..solutions import (
    get_tabml_solution_class,
    parse_config_from_tabular_dataset,
    get_tabml_solution_choice,
)
from .. import yaml_utils
from .fit_utils import _fit_main

logger = logging.getLogger(__name__)
logger.setLevel('DEBUG')

TabMLSolutionChoice = get_tabml_solution_choice()

def evaluate_tab(
    dataset_path : str = typer.Argument(
        ...,
        help=("Path to the dataset folder or one of the built-in datasets. "
              "Use the list-builtin command to list all the built-in datasets.")
    ),
    task_name : str = typer.Argument(
        ...,
        help=("Name of the task to fit the solution.")
    ),
    solution_name : TabMLSolutionChoice = typer.Argument(
        ...,
        help="Solution name"
    ),
    checkpoint_path : str = typer.Argument(
        ...,
        help="Path to the workspace. "
        " The workspace should contain data_config.yaml, solution_config.yaml, and model.pt."
    ),
    feature_importance_path : str = typer.Option(
        None,
        "--feature-importance-path",
        help="Compute feature importance and output the result as a CSV at the given path."
    ),
    split : str = typer.Option(
        "test",
        "--split",
        help="The split to evaluate. Can be train, validation or test (default)."
    ),
):
    config_path = os.path.join(checkpoint_path, "solution_config.yaml")
    data_config_path = os.path.join(checkpoint_path, "data_config.yaml")

    solution_class = get_tabml_solution_class(solution_name.value)

    logger.info(f"Load solution configuration file: {config_path}.")
    solution_config = yaml_utils.load_pyd(solution_class.config_class, config_path)
    logger.debug(f"Solution config:\n{solution_config.json()}")

    logger.info("Loading data ...")
    dataset = dbb.load_rdb_data(dataset_path)

    data_config = parse_config_from_tabular_dataset(dataset, task_name)
    preserve_data_config = yaml_utils.load_pyd(type(data_config), data_config_path)
    assert data_config == preserve_data_config, "Preserved data config is different from the original one."
    logger.debug(f"Data config:\n{data_config.json()}")

    logger.info("Creating solution ...")
    solution = solution_class(solution_config, data_config)
    device = get_device_info()

    def _invoke_test(solution, run_ckpt_path : Path, device : DeviceInfo):
        solution.load_from_checkpoint(run_ckpt_path)
        assert split in ['train', 'validation', 'test']
        split_set = getattr(dataset.get_task(task_name), split + '_set')
        metric = solution.evaluate(split_set, device)
        if feature_importance_path:
            df = solution.feature_importance(split_set, device)
            df.to_csv(feature_importance_path)
            logger.info(f'Feature importance\n{df}')
        return metric

    logger.info("Testing ...")
    test_metric = _invoke_test(solution, checkpoint_path, device)
    logger.info(f"{split} metric: {test_metric:.4f}")
