import pandas as pd

import os
import wandb
from collections import defaultdict

import wandb.errors

from affex.utils.utils import load_yaml, write_yaml


def log_to_wandb(folder, run_params, hyperparams, datasets_results, overwrite=False):
    """
    Log the collected datasets to Weights & Biases.
    
    Args:
        params (dict): Parameters of the experiment.
        datasets_results (dict): Dictionary containing dataset names as keys and their results as values.
    """
    print("Logging to Weights & Biases...")
    if os.path.exists(os.path.join(folder, "wandb.txt")):
        print("overwrite:", overwrite)
        if not overwrite:
            print("WARNING: Weights & Biases run already exists, use `overwrite=True` to overwrite.")
            return
        else:
            print("Weights & Biases run already exists, overwriting.")
            # Read the existing run id
            with open(os.path.join(folder, "wandb.txt"), "r") as f:
                run_id = f.read().strip().split(": ")[1]
            # Get the existing run
            try:
                existing_run = wandb.Api().run(f"affex/{run_id}")
                # Delete the existing run
                existing_run.delete()
                # Remove the existing wandb.txt file
            except wandb.errors.errors.CommError as e:
                print(f"Error deleting existing run: {e}, continuing without deleting.")
            os.remove(os.path.join(folder, "wandb.txt"))
            
    group = hyperparams.get("grid", None)
    
    wandb.init(project="affex", config=run_params, group=group)
    
    for dataset_name, dataframe in datasets_results.items():
        wandb.log({dataset_name: wandb.Table(dataframe=dataframe)})
        
    # Log the mean values as a separate results
    for dataset_name, dataframe in datasets_results.items():
        for col in dataframe.columns:
            # Remove nans from the column
            dataframe[col] = dataframe[col].dropna()
            # Get the mean of the column
            mean_value = dataframe[col].mean()
            # Log the mean value to Weights & Biases
            wandb.summary[f"{dataset_name}_{col.replace("_aucs","")}"] = mean_value
            
    # Get run id
    run_id = wandb.run.id
    wandb.finish()
    
    with open(os.path.join(folder, "wandb.txt"), "w") as f:
        f.write(f"Run ID: {run_id}\n")
    
    print("Logged to Weights & Biases.")


def collect_single_job(folder, params, hyperparams, datasets, use_wandb, overwrite):
    write_yaml(params, os.path.join(folder, "params.yaml"))

    print(f"Found {len(datasets)} datasets in {folder}")
    
    datasets_results = {}

    for dataset_name, csv_file in datasets.items():
        print(f"Dataset: {dataset_name}")
        datasets_results[dataset_name] = pd.read_csv(csv_file)
    
    if use_wandb:
        log_to_wandb(folder, params, hyperparams, datasets_results, overwrite=overwrite)

    
def collect_multi_job(folder, params, hyperparams, datasets, use_wandb, overwrite):
    write_yaml(params, os.path.join(folder, "params.yaml"))

    print(f"Found {len(datasets)} datasets in {folder}")
    
    datasets_results = {}

    for dataset_name, csv_files in datasets.items():
        print(f"Dataset: {dataset_name}")
        dataframes = [
            pd.read_csv(csv_file) for csv_file in csv_files
        ]
        dataframe = pd.concat(dataframes, ignore_index=True)
        dataframe.to_csv(
            os.path.join(folder, f"{dataset_name}.csv"),
            index=False,
        )
        datasets_results[dataset_name] = dataframe
    
    if use_wandb:
        log_to_wandb(folder, params, hyperparams, datasets_results, overwrite=overwrite)


def runs_collect(folder, use_wandb=True, overwrite=False):
    folders = [f for f in os.listdir(folder) if os.path.isdir(os.path.join(folder, f))]

    # Match all folders with name p_xxx
    processes = [
        f for f in folders if f.startswith("p_")
    ]
    # Match all folders with name run_x
    runs = [
        f for f in folders if f.startswith("run_")
    ]

    if len(processes) > 0 and len(runs) == 0:
        print("Collecting proocesses from subdirectories...")
        hyperparams = load_yaml(os.path.join(folder, "..", "hyperparams.yaml"))

        datasets = defaultdict(list)
        for subfolder in processes:
            subfolder_path = os.path.join(folder, subfolder)
            csvs = [
                f for f in os.listdir(subfolder_path) if f.endswith(".csv") and f.startswith("scores")
            ]
            for csv in csvs:
                dataset_name = csv.replace("scores_", "").replace(".csv", "")
                datasets[dataset_name].append(os.path.join(subfolder_path, csv))

        params = load_yaml(os.path.join(folder, processes[0], "params.yaml"))
        del params["process_id"]
        params["num_processes"] = len(processes)

        collect_multi_job(folder, params, hyperparams, datasets, use_wandb, overwrite)
            
        print(f"Collected {len(processes)} processes from {folder}")
    elif len(runs) > 0 and len(processes) == 0:
        print("Collecting runs from subdirectories...")
        for run in runs:
            run_path = os.path.join(folder, run)
            print(f"Collecting run: {run_path}")
            runs_collect(run_path, use_wandb=use_wandb, overwrite=overwrite)
    elif len(processes) == 0 and len(runs) == 0:
        files = [f for f in os.listdir(folder) if os.path.isfile(os.path.join(folder, f)) and f.endswith(".csv")]
        score_files = [f for f in files if f.startswith("scores_")]
        if len(score_files) > 0:
            print("Collecting scores from files in the folder...")
            hyperparams = load_yaml(os.path.join(folder, "..", "hyperparams.yaml"))

            datasets = {}
            for score_file in score_files:
                dataset_name = score_file.replace("scores_", "").replace(".csv", "")
                datasets[dataset_name] = os.path.join(folder, score_file)

            params = load_yaml(os.path.join(folder, "params.yaml"))
            collect_single_job(folder, params, hyperparams, datasets, use_wandb, overwrite)
        else:
            print("No subdirectories and no score file found in the folder.")
    else:
        print("Both processes and runs found in the folder, please specify one type to collect, something is wrong.")