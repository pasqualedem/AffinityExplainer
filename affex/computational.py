OUT_FOLDER = "out"

import copy
import os
import random
import uuid
from matplotlib import pyplot as plt
import numpy as np
import pandas as pd
import torch
from tqdm import tqdm
import wandb
import yaml
import lovely_tensors as lt

lt.monkey_patch()

from affex.data import get_dataloaders
from affex.data.utils import BatchKeys
from affex.explainer import build_explainer, get_explanation_mask
from affex.models import build_model_preconfigured
from affex.substitution import Substitutor
from affex.utils.logger import get_logger
from affex.utils.utils import to_device


def sync_cuda():
    if torch.cuda.is_available():
        torch.cuda.synchronize()


def set_seed(seed):
    """Set the seed for reproducibility."""
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    random.seed(seed)
    np.random.seed(seed)


def evaluate_computational(parameters, run_name=None, log_params=True, log_on_file=True):
    set_seed(parameters.get("seed", 42))

    if run_name is None:
        run_name = str(uuid.uuid4())[:8]
        run_name = os.path.join(OUT_FOLDER, run_name)
        os.makedirs(run_name, exist_ok=True)
    # model filename is log filename but with .pt instead of .log
    params_filename = run_name + "/params.yaml"
    if log_params:
        with open(params_filename, "w") as f:
            yaml.dump(parameters, f)

    log_filename = run_name + "/log.log" if log_on_file else None
    logger = get_logger("Eval", log_filename)
    logger.info("parameters:")
    logger.info(parameters)

    wandb.init(project="affex", config=parameters, group="Computational")

    device = parameters.get("device", "cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Running on {device}")

    model, image_size = build_model_preconfigured(model_name=parameters["model"])
    model.eval()
    model.to(device)
    log_frequency = parameters.get("log_frequency", 50)

    for k in parameters["dataset"]["datasets"]:
        parameters["dataset"]["datasets"][k]["image_size"] = image_size
    
    if "preprocess" not in parameters["dataset"]:
        parameters["dataset"]["preprocess"] = {} 
    parameters["dataset"]["preprocess"]["image_size"] = image_size

    _, val, _ = get_dataloaders(
        copy.deepcopy(parameters["dataset"]),
        copy.deepcopy(parameters["dataloader"]),
        num_processes=1,
    )

    explainer = build_explainer(
        model=model,
        name=parameters["explainer"]["name"],
        params={k: v for k, v in parameters["explainer"].items() if k != "name"},
        device=device,
    )

    masking_type = parameters["explanation_masking"]
    num_steps = parameters["num_steps"]
    warmup_steps = parameters["warmup_steps"]
    total_steps = num_steps + warmup_steps
    
    explanation_size = parameters.get("explanation_size", image_size)
    
    explanation_times = []
    forward_memories = []
    explanation_memories = []
    forward_times = []

    for dataset_name, val_dataloader in val.items():
        logger.info(f"Evaluating {dataset_name} dataset")
        bar = tqdm(
            enumerate(val_dataloader),
            total=total_steps,
            desc=f"Evaluating {dataset_name} - Warmup",
            unit="batch",
        )
        
        for i, batch in bar:
            batch, _ = batch

            substitutor = Substitutor(substitute=False)
            substitutor.reset(batch=batch)
            input_dict, gt = next(substitutor)
            input_dict = to_device(input_dict, device)
            gt = to_device(gt, device)
            
            
            torch.cuda.reset_peak_memory_stats(device=device)
            
            start_forward = torch.cuda.Event(enable_timing=True)
            end_forward = torch.cuda.Event(enable_timing=True)
            
            start_forward.record()
            with torch.no_grad():
                result = model(input_dict, postprocess=False)
            end_forward.record()
            torch.cuda.synchronize()
            elapsed_forward_time = start_forward.elapsed_time(end_forward)
                
            forward_memory = torch.cuda.max_memory_allocated(device=device) / (1024 ** 2)  # in MB

            explanation_size = explanation_size or input_dict[BatchKeys.IMAGES].shape[-2:]
            explanation_mask = get_explanation_mask(input_dict, gt, result, explanation_size, masking_type)
            
            start_explanation = torch.cuda.Event(enable_timing=True)
            end_explanation = torch.cuda.Event(enable_timing=True)
            
            # Reset memory stats before explanation
            torch.cuda.reset_peak_memory_stats(device=device)
            
            start_explanation.record()
            _ = explainer.explain(
                input_dict=input_dict,
                explanation_mask=explanation_mask,
                explanation_size=explanation_size,
            )
            end_explanation.record()
            torch.cuda.synchronize()
            elapsed_explanation_time = start_explanation.elapsed_time(end_explanation)

            explanation_memory = torch.cuda.max_memory_allocated(device=device) / (1024 ** 2)  # in MB

            if i >= warmup_steps:
                bar.set_description(f"Explanation time: {elapsed_explanation_time:.2f} ms")
                explanation_times.append(elapsed_explanation_time)
                forward_memories.append(forward_memory)
                explanation_memories.append(explanation_memory)
                forward_times.append(elapsed_forward_time)
                
            if i == total_steps - 1:
                break  # Limit to total_steps batches for faster evaluation
            
        mean_explanation_time = np.mean(explanation_times)
        std_explanation_time = np.std(explanation_times)
        mean_forward_memory = np.mean(forward_memories)
        std_forward_memory = np.std(forward_memories)
        mean_explanation_memory = np.mean(explanation_memories)
        std_explanation_memory = np.std(explanation_memories)
        mean_forward_time = np.mean(forward_times)
        std_forward_time = np.std(forward_times)
        
        logger.info(f"Forward time over {num_steps} runs: {mean_forward_time:.2f} ± {std_forward_time:.2f} ms")
        logger.info(f"Explanation time over {num_steps} runs: {mean_explanation_time:.2f} ± {std_explanation_time:.2f} ms")
        logger.info(f"Forward memory: {mean_forward_memory:.2f} ± {std_forward_memory:.2f} MB")
        logger.info(f"Explanation memory: {mean_explanation_memory:.2f} ± {std_explanation_memory:.2f} MB")
        
        wandb.log({
            f"explanation_time_mean": mean_explanation_time,
            f"explanation_time_std": std_explanation_time,
            f"forward_memory_mean": mean_forward_memory,
            f"forward_memory_std": std_forward_memory,
            f"explanation_memory_mean": mean_explanation_memory,
            f"explanation_memory_std": std_explanation_memory,
            f"forward_time_mean": mean_forward_time,
            f"forward_time_std": std_forward_time,
        })
    logger.info("Evaluation completed.")
    wandb.finish()


