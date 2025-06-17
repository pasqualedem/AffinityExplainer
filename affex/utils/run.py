import os
import subprocess
import sys
import copy

from affex.utils.utils import PrintLogger, write_yaml


class ParallelRun:
    slurm_command = "sbatch"
    slurm_multi_gpu_script = "slurm/launch_run_multi_gpu"
    slurm_script_first_parameter = "--parameters="
    slurm_script_run_name_parameter = "--run_name="
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

    def launch(self, only_create=False, script_args=[]):
        out_file = f"{self.run_name}/log.{self.out_extension}"
        param_file = f"{self.run_name}/params.{self.param_extension}"

        self.logger.info("Running a single process")
        self.launch_process(
            self.params, self.run_name, out_file, param_file, only_create, script_args
        )

    def launch_process(
        self, params, run_name, out_file, param_file, only_create=False, script_args=[]
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
            self.slurm_script_run_name_parameter + run_name,
            *script_args,
        ]
        if only_create:
            self.logger.info(f"Creating command: {' '.join(command)}")
        else:
            self.logger.info(f"Launching command: {' '.join(command)}")
            subprocess.run(command)

