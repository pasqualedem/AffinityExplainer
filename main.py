from datetime import datetime
import os
import click
import yaml

from affex.evaluate import evaluate
from affex.utils.logger import get_logger
from affex.utils.utils import load_yaml
from affex.utils.grid import ParallelRun, create_experiment


OUT_FOLDER = "out"


@click.group()
def cli():
    """Run a refinement or a grid"""
    pass


@cli.command("grid")
@click.option(
    "--parameters",
    default=None,
    help="Path to the file containing the parameters for a grid search",
)
@click.option(
    "--parallel",
    default=False,
    is_flag=True,
    help="Run the grid in parallel",
)
@click.option(
    "--only_create",
    default=False,
    is_flag=True,
    help="Only create the slurm scripts",
)
def grid(parameters, parallel, only_create=False):
    parameters = load_yaml(parameters)
    grid_name = parameters.pop("grid")
    current_time = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    grid_name = f"{current_time}_{grid_name}"
    log_folder = os.path.join(OUT_FOLDER, grid_name)
    
    runs_parameters = create_experiment(parameters)
    
    os.makedirs(log_folder)
    with open(os.path.join(log_folder, "hyperparams.yaml"), "w") as f:
        yaml.dump(parameters, f)

    grid_logger = get_logger("Grid", f"{log_folder}/grid.log")
    grid_logger.info(f"Running {len(runs_parameters)} runs")
    for i, run_parameters in enumerate(runs_parameters):
        run_name = f"{log_folder}/run_{i}"
        if parallel:
            run = ParallelRun(
                run_parameters,
                multi_gpu=False,
                logger=grid_logger,
                run_name=run_name,
            )
            run.launch(
                only_create=only_create,
                script_args=[
                    "--disable_log_params",
                    "--disable_log_on_file",
                ],
            )
        else:
            grid_logger.info(f"Running run {i+1}/{len(runs_parameters)}")
            evaluate(run_parameters, run_name=run_name)


@cli.command("run")
@click.option(
    "--parameters",
    default=None,
    help="Path to the file containing the parameters for a single run",
)
@click.option("--run_name", default=None, help="Name of the run")
@click.option(
    "--disable_log_params",
    default=False,
    is_flag=True,
    help="Disable Log the parameters",
)
@click.option(
    "--disable_log_on_file", default=False, is_flag=True, help="Disable Log on file"
)
def run(
    parameters,
    run_name=None,
    disable_log_params=False,
    disable_log_on_file=False,
):
    parameters = load_yaml(parameters)
    evaluate(
        parameters,
        run_name,
        not disable_log_params,
        not disable_log_on_file,
    )


if __name__ == "__main__":
    cli()
