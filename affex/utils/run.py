import os
import subprocess
import sys
import copy

import pandas as pd

from affex.data import get_dataloaders
from affex.data.utils import BatchKeys
from affex.utils.utils import PrintLogger, write_yaml


class ParallelRun:
    slurm_command = "sbatch"
    slurm_multi_gpu_script = "slurm/launch_run_multi_gpu"
    slurm_script_first_parameter = "--parameters="
    out_extension = "log"
    param_extension = "yaml"
    slurm_stderr = "-e"
    slurm_stdout = "-o"

    def __init__(
        self,
        params: dict,
        multi_gpu=False,
        logger=None,
        run_name=None,
        slurm_script=None,
    ):
        self.params = params
        self.multi_gpu = multi_gpu
        self.logger = logger or PrintLogger()
        self.run_name = run_name
        self.slurm_script = slurm_script or "slurm/launch_run"
        if "." not in sys.path:
            sys.path.extend(".")

    def manage_processing(self):
        if "num_processes" in self.params and self.params["num_processes"] > 1:
            datasets = self.create_dataset()
            multi_runs = [
                copy.deepcopy(self.params) for _ in range(self.params["num_processes"])
            ]
        else:
            multi_runs = [self.params]
            datasets = None
        return multi_runs, datasets

    def launch(self, only_create=False, script_args=[]):
        out_file = f"{self.run_name}/log.{self.out_extension}"
        param_file = f"{self.run_name}/params.{self.param_extension}"

        multi_runs, datasets = self.manage_processing()
        if len(multi_runs) > 1:
            self.logger.info(f"Running {len(multi_runs)} processes in parallel")
            for i, run_parameters in enumerate(multi_runs):
                run_name = f"{self.run_name}/p_{str(i).zfill(3)}"
                run_parameters["process_id"] = i
                os.makedirs(run_name, exist_ok=True)
                out_file = f"{run_name}/log.{self.out_extension}"
                param_file = f"{run_name}/params.{self.param_extension}"
                for dataset_name, dataset_shard in datasets.items():
                    dataset_file = f"{run_name}/{dataset_name}.csv"
                    dataset_shard[i].to_csv(dataset_file, index=False)

                self.launch_process(
                    run_parameters, out_file, param_file, only_create, script_args
                )
        else:
            self.logger.info("Running a single process")
            self.launch_process(
                self.params, out_file, param_file, only_create, script_args
            )

    def launch_process(
        self, params, out_file, param_file, only_create=False, script_args=[]
    ):
        write_yaml(params, param_file)
        slurm_script = (
            self.slurm_multi_gpu_script if self.multi_gpu else self.slurm_script
        )
        command = [
            self.slurm_command,
            self.slurm_stdout,
            out_file,
            self.slurm_stderr,
            out_file,
            slurm_script,
            self.slurm_script_first_parameter + param_file,
            *script_args,
        ]
        if only_create:
            self.logger.info(f"Creating command: {' '.join(command)}")
        else:
            self.logger.info(f"Launching command: {' '.join(command)}")
            subprocess.run(command)

    def create_dataset(self):
        """
        Create a dataset for the run parameters
        :param run_parameters: parameters for the run
        :param dataset_name: name of the dataset to create
        :return: None
        """
        _, val, _ = get_dataloaders(
            copy.deepcopy(self.params["dataset"]),
            copy.deepcopy(self.params["dataloader"]),
            num_processes=1,
        )
        datasets = {}

        num_processes = self.params.get("num_processes", 1)
        for dataset_name, val_dataloader in val.items():
            self.logger.info(f"Creating {dataset_name}")
            dataset_shards = [
                pd.DataFrame(columns=[BatchKeys.IMAGE_IDS.value, BatchKeys.CLASSES.value])
                for _ in range(num_processes)
            ]
            for i, batch in enumerate(val_dataloader):
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
                dataset_shards[i % num_processes] = pd.concat(
                    [dataset_shards[i % num_processes], batch_df], ignore_index=True
                )
            self.logger.info(f"Dataset {dataset_name} created")
            datasets[dataset_name] = dataset_shards
        return datasets
