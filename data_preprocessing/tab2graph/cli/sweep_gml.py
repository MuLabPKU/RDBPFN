from pathlib import Path
import typer
import logging
import wandb
import os
import numpy as np
import yaml
import tempfile
import subprocess

import dbinfer_bench as dbb

from ..solutions import (
    get_gml_solution_class,
    get_gml_solution_choice,
    SweepChoice,
)
from ..device import DeviceInfo

logger = logging.getLogger(__name__)
logger.setLevel('DEBUG')

def sweep_gml(
    dataset_path : str = typer.Argument(
        ...,
        help=("Path to the dataset folder or one of the built-in datasets. "
              "Use the list-builtin command to list all the built-in datasets.")
    ),
    task_name : str = typer.Argument(
        ...,
        help=("Name of the task to fit the solution.")
    ),
    solution_name : str = typer.Argument(
        ...,
        help="Solution name"
    ),
    config_path : str = typer.Argument(
        ...,
        help="Sweep configuration path."
    ),
    sweep_method : SweepChoice = typer.Option(
        "bayes",
        "--sweep_method", "-m",
        help="Sweep (hyper-parameter search) method."
    ),
    run_cap : int = typer.Option(
        100,
        "--run_cap", "-c",
        help=("Number of hyper-parameter combinations to try. "
              "Setting it to -1 means exhuasting all the combinations. "
              "However, this may cause the sweep to run forever.")
    ),
    checkpoint_path : str = typer.Option(
        None, 
        "--checkpoint_path", "-p",
        help="Checkpoint path."
    ),
    sweep_project_name : str = typer.Option(
        "TGIF-Sweeps",
        "--sweep_project_name",
        help="Sweep project name."
    ),
):
    # This function should not be available if DGL is not installed
    # The function import itself will fail at the package level
    _create_sweep(
        "gml",
        dataset_path,
        task_name,
        solution_name,
        config_path,
        str(sweep_method.value),
        run_cap,
        checkpoint_path,
        sweep_project_name,
    )

def _create_sweep(
    model_type : str,
    dataset_path : str,
    task_name : str,
    solution_name : str,
    config_path : str,
    sweep_method : str,
    run_cap : int,
    checkpoint_path : str,
    sweep_project_name : str,
):
    # Load parameters from the user-provided sweep configuration file
    with open(config_path, 'r') as file:
        parameters = yaml.safe_load(file)
    # Add parameters for logging purposes.
    parameters.update({
        "dataset" : {
            "distribution" : "constant",
            "value" : dataset_path,
        },
        "task" : {
            "distribution" : "constant",
            "value" : task_name,
        },
        "solution" : {
            "distribution" : "constant",
            "value" : solution_name,
        },
    })

    # Agent command.
    command = [
        "python", "-m", "tab2graph.main",
        f"fit-{model_type}",
        dataset_path,
        task_name,
        solution_name,
        "--launch-by-wandb-agent",
    ]
    if checkpoint_path is not None:
        command += ["-p", checkpoint_path]

    # Prepare the new sweep configuration and write it to a temporary file
    sweep_config = {
        "program": "tab2graph.main",
        "command": command,
        "method": sweep_method,
        "metric": {
            "name": "val_metric.max",
            "goal": "maximize"
        },
        "run_cap": run_cap,
        "parameters": parameters,
    }

    temp_sweep_file = tempfile.NamedTemporaryFile(delete=False, suffix='.yaml')
    with open(temp_sweep_file.name, 'w') as file:
        yaml.dump(sweep_config, file)

    # Execute the command to create the sweep
    ret = subprocess.run([
        "wandb", "sweep",
        "--project", sweep_project_name,
        temp_sweep_file.name], check=True)

    # Close and remove the temporary file
    temp_sweep_file.close()
    os.unlink(temp_sweep_file.name)
