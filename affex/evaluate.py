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
import yaml
import lovely_tensors as lt

from affex.utils.segmentation import create_rgb_segmentation, unnormalize
lt.monkey_patch()

from affex.data import get_dataloaders
from affex.data.utils import BatchKeys
from affex.explainer import build_explainer, get_explanation_mask
from affex.metrics import FSSCausalMetric
from affex.models import build_model, build_model_preconfigured
from affex.substitution import Substitutor
from affex.utils.logger import get_logger
from affex.utils.utils import ResultDict, to_device

from torchmetrics import MetricCollection


def set_seed(seed):
    """Set the seed for reproducibility."""
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    random.seed(seed)
    np.random.seed(seed)


def log_step(input_dict, gt, results, explanation, metrics, outfolder, batch_idx):
    """Log the step results."""
    outfolder = os.path.join(outfolder, f"batch_{batch_idx}")
    os.makedirs(outfolder, exist_ok=True)
    
    images = input_dict[BatchKeys.IMAGES]
    unnormalize(images).rgb.fig.savefig(os.path.join(outfolder, "images.png"))
    
    logits = results[ResultDict.LOGITS]
    seg = create_rgb_segmentation(logits.cpu())
    seg.rgb.fig.savefig(os.path.join(outfolder, "segmentation.png"))
    
    gt = create_rgb_segmentation(gt.cpu())
    gt.rgb.fig.savefig(os.path.join(outfolder, "ground_truth.png"))
    
    explanation.chans.fig.savefig(os.path.join(outfolder, "explanation.png"))
    
    metrics_folders = os.path.join(outfolder, "metrics")
    os.makedirs(metrics_folders, exist_ok=True)
    for metric_name, metric_value in metrics.items():
        mid_statuses = metric_value.mid_statuses
        for mid_status in mid_statuses:
            j, mig_images, mid_masks, top_pred = mid_status
            unnormalize(mig_images).rgb.fig.savefig(os.path.join(metrics_folders, f"{metric_name}_{j}_img.png"))
            mid_masks.chans.fig.savefig(os.path.join(metrics_folders, f"{metric_name}_{j}_mask.png"))
            
        scores = metric_value.compute()["scores"]
        scores_df = pd.DataFrame(scores)
        scores_df.to_csv(os.path.join(metrics_folders, f"{metric_name}_scores.csv"), index=False)
        
        # Plot scores
        batch_element = 0  # Change this to plot a different batch element
        plt.plot(np.arange(scores.shape[0]) / scores.shape[0], scores[:, batch_element])
        plt.fill_between(np.arange(scores.shape[0]) / scores.shape[0], 0, scores[:, batch_element], alpha=0.4)
        plt.xlim(-0.1, 1.1)
        plt.ylim(0, 1.05)
        plt.xlabel(f'Pixels changed')
        plt.ylabel('Score')
        plt.title(f'{metric_name}')
        plt.savefig(os.path.join(metrics_folders, f"{metric_name}_scores.svg"))
        plt.clf()
        plt.close()


def evaluate(parameters, run_name=None, log_params=True, log_on_file=True):
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

    device = "cuda" if torch.cuda.is_available() else "cpu"
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
    explanation_size = parameters.get("explanation_size", image_size)
    metrics = MetricCollection(
        {
            "iauc": FSSCausalMetric(
                model=model,
                mode="ins",
                **parameters["metric"]
            ),
            "dauc": FSSCausalMetric(
                model=model,
                mode="del",
                **parameters["metric"]
            ),
        }
    )

    for dataset_name, val_dataloader in val.items():
        logger.info(f"Evaluating {dataset_name} dataset")
        bar = tqdm(
            enumerate(val_dataloader),
            total=len(val_dataloader),
            desc=f"Evaluating {dataset_name}",
            unit="batch",
        )
        
        for i, batch in bar:
            batch, _ = batch

            substitutor = Substitutor(substitute=False)
            substitutor.reset(batch=batch)
            input_dict, gt = next(substitutor)
            input_dict = to_device(input_dict, device)
            
            
            bar.set_description(f"Calculating prediction")
            with torch.no_grad():
                result = model(input_dict, postprocess=False)

            explanation_size = explanation_size or input_dict[BatchKeys.IMAGES].shape[-2:]
            explanation_mask = get_explanation_mask(input_dict, gt, result, explanation_size, masking_type)

            bar.set_description(f"Calculating explanation")
            explanation = explainer.explain(
                input_dict=input_dict,
                explanation_mask=explanation_mask,
                explanation_size=explanation_size,
            )
            
            assert len(explanation) == 1, "Only support one class for now"
            explanation = explanation[0]
            
            bar.set_description(f"Calculating metrics")
            metrics.update(
                input_dict=input_dict,
                explanation=explanation,
                explanation_mask=explanation_mask,
            )
            scores = metrics.compute()
            dauc = scores["dauc_auc"]
            iauc = scores["iauc_auc"]
            
            logger.info(f"iAUC: {iauc:.4f}, dAUC: {dauc:.4f}")
        
            if i % log_frequency == 0:
                log_step(input_dict, gt, result, explanation, metrics, run_name, i)
            
            daucs = scores["dauc_aucs"]
            iaucs = scores["iauc_aucs"]
                
            scores_df = pd.DataFrame(
                {
                    "dauc_aucs": torch.tensor(daucs).tolist(),
                    "iauc_aucs": torch.tensor(iaucs).tolist(),
                }
            )
            scores_df.to_csv(os.path.join(run_name, f"scores_{dataset_name}.csv"), index=False)
