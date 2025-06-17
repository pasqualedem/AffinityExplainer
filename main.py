import copy
from datetime import datetime
import os
import click
import pandas as pd
from tqdm import tqdm
import yaml


from affex.collect import runs_collect
from affex.data import get_dataloaders
from affex.data.utils import BatchKeys
from affex.evaluate import evaluate
from affex.utils.logger import get_logger
from affex.utils.utils import load_yaml
from affex.utils.grid import create_experiment
from affex.utils.run import ParallelRun


OUT_FOLDER = "out"


@click.group()
def cli():
    """Run a refinement or a grid"""
    pass


def manage_multiprocess_run(run_parameters, run_name, logger):
    """
    Manage the multiprocess run.
    This function is used to launch the run in parallel or sequentially.
    """
    if "num_processes" in run_parameters["dataloader"]:
        multi_runs = [
                copy.deepcopy(run_parameters) for _ in range(run_parameters["dataloader"]["num_processes"])
            ]
        run_names = [
            f"{run_name}/p_{str(i).zfill(3)}" for i in range(run_parameters["dataloader"]["num_processes"])
        ]
        
        logger.info(f"Running {len(multi_runs)} processes in parallel")
        for i, run_parameters in enumerate(multi_runs):
            run_name = f"{run_names[i]}"
            multi_runs[i]["dataloader"]["process_id"] = i
            os.makedirs(run_name, exist_ok=True)
    else:
        multi_runs = [run_parameters]
        run_names = [run_name]
        logger.info("Running in single process mode")

    return multi_runs, run_names


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
    grid_name = parameters["grid"]
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
        os.makedirs(run_name, exist_ok=True)
        multi_runs, run_names = manage_multiprocess_run(
            run_parameters, run_name, grid_logger
        )
        for k, (subrun_parameters, subrun_name) in enumerate(
            zip(multi_runs, run_names)
        ):
            if parallel:
                run = ParallelRun(
                    subrun_parameters,
                    multi_gpu=False,
                    logger=grid_logger,
                    run_name=subrun_name,
                )
                run.launch(
                    only_create=only_create,
                    script_args=[
                        "--disable_log_params",
                        "--disable_log_on_file",
                    ],
                )
            else:
                if len(multi_runs) > 1:
                    grid_logger.info(
                        f"Running subrun {k+1}/{len(multi_runs)} in run {i+1}/{len(runs_parameters)}"
                    )
                else:
                    grid_logger.info(f"Running run {i+1}/{len(runs_parameters)}")
                evaluate(subrun_parameters, run_name=subrun_name)


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
@click.option(
    "--run_name",
    default=None,
    help="Name of the run, if not provided, it will be generated based on the current time",
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
    
    
@cli.command("collect")
@click.option(
    "-f", "--folder",
    default=None,
    help="Path to the folder containing the runs to collect",
)
@click.option(
    "--no_wandb",
    default=False,
    is_flag=True,
    help="Disable wandb collection",
)
@click.option(
    "--overwrite",
    default=False,
    is_flag=True,
    help="Overwrite existing runs on wandb",
)
def collect(folder, no_wandb, overwrite):
    assert os.path.exists(folder), f"Folder {folder} does not exist"
    assert os.path.isdir(folder), f"{folder} is not a directory"
    
    runs_collect(folder=folder, use_wandb=not no_wandb, overwrite=overwrite)


@cli.command("generate")
@click.option(
    "-p", "--parameters",
    default=None,
    help="Path to the file containing the parameters for the dataset generation",
)
def generate(parameters):
    parameters = load_yaml(parameters)
    assert "dataset" in parameters, "Parameters file must contain a 'dataset' key"
    
    outfolder = parameters["output_dir"]
    if not os.path.exists(outfolder):
        os.makedirs(outfolder)

    _, val, _ = get_dataloaders(
        copy.deepcopy(parameters["dataset"]),
        copy.deepcopy(parameters["dataloader"]),
        num_processes=1,
    )
    datasets = {}
    logger = get_logger("Dataset Generation")

    for dataset_name, val_dataloader in val.items():
        logger.info(f"Creating {dataset_name}")
        dataset = pd.DataFrame(columns=[BatchKeys.IMAGE_IDS.value, BatchKeys.CLASSES.value])
        bar = tqdm(
            val_dataloader,
            total=len(val_dataloader),
            desc=f"Creating dataset {dataset_name}",
            unit="batch",
        )
        for batch in bar:
            batch, _ = batch
            batch, gt = batch
            image_ids = batch[BatchKeys.IMAGE_IDS]
            class_ids = batch[BatchKeys.CLASSES]
            batch_df = pd.DataFrame(
                {
                    BatchKeys.IMAGE_IDS.value: [image_ids],
                    BatchKeys.CLASSES.value: [class_ids],
                }
            )
            dataset = pd.concat([dataset, batch_df], ignore_index=True)
        logger.info(f"Dataset {dataset_name} created")
        datasets[dataset_name] = dataset
    
    for dataset_name, dataset in datasets.items():
        dataset_file = os.path.join(outfolder, f"{dataset_name}.csv")
        dataset.to_csv(dataset_file, index=False)
        logger.info(f"Dataset {dataset_name} saved to {dataset_file}")

if __name__ == "__main__":
    cli()
