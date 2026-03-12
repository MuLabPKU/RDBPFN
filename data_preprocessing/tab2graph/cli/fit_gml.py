from pathlib import Path
import typer
import logging
import wandb
import os
import numpy as np
import dbinfer_bench as dbb

from ..device import DeviceInfo
from ..solutions import (
    get_gml_solution_class,
    parse_config_from_graph_dataset,
    get_gml_solution_choice,
)
from .. import yaml_utils
from .fit_utils import _fit_main

logger = logging.getLogger(__name__)
logger.setLevel('DEBUG')

GMLSolutionChoice = get_gml_solution_choice()

def fit_gml(
    dataset_path : str = typer.Argument(
        ...,
        help=("Path to the dataset folder or one of the built-in datasets. "
              "Use the list-builtin command to list all the built-in datasets.")
    ),
    task_name : str = typer.Argument(
        ...,
        help=("Name of the task to fit the solution.")
    ),
    solution_name : GMLSolutionChoice = typer.Argument(
        ...,
        help="Solution name"
    ),
    config_path : Path = typer.Option(
        None,
        "--config_path", "-c",
        help="Solution configuration path. Use default if not specified."
    ),
    checkpoint_path : str = typer.Option(
        None, 
        "--checkpoint_path", "-p",
        help="Checkpoint path."
    ),
    enable_wandb : bool = typer.Option(
        True,
        "--enable-wandb/--disable-wandb",
        help="Enable Weight&Bias for logging."
    ),
    launch_by_wandb_agent : bool = typer.Option(
        False,
        "--launch-by-wandb-agent/--not-launch-by-wandb-agent",
        help=("This command is launched by `wandb agent`. "
              "Its configuration will be loaded from W&B.")
    ),
    num_runs : int = typer.Option(
        1,
        "--num-runs", "-n",
        help="Number of runs."
    ),
):
    solution_class = get_gml_solution_class(solution_name.value)
    if config_path is None:
        logger.info("No solution configuration file provided. Use default configuration.")
        solution_config = solution_class.config_class()
    else:
        logger.info(f"Load solution configuration file: {config_path}.")
        solution_config = yaml_utils.load_pyd(solution_class.config_class, config_path)

    logger.info("Loading data ...")
    dataset = dbb.load_graph_data(dataset_path)

    data_config = parse_config_from_graph_dataset(dataset, task_name)
    logger.debug(f"Data config:\n{data_config.json()}")

    def _invoke_fit(solution, run_ckpt_path : Path, device : DeviceInfo):
        summary = solution.fit(dataset, task_name, run_ckpt_path, device)
        return summary

    def _invoke_test(solution, run_ckpt_path : Path, device : DeviceInfo):
        solution.load_from_checkpoint(run_ckpt_path)
        val_metric = solution.evaluate(
            dataset.graph_tasks[task_name].validation_set,
            dataset.graph,
            dataset.feature,
            device,
        )
        test_metric = solution.evaluate(
            dataset.graph_tasks[task_name].test_set,
            dataset.graph,
            dataset.feature,
            device,
        )
        return val_metric, test_metric

    _fit_main(
        solution_class,
        dataset,
        task_name,
        data_config,
        solution_config,
        checkpoint_path,
        enable_wandb,
        launch_by_wandb_agent,
        num_runs,
        _invoke_fit,
        _invoke_test
    )
