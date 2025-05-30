OUT_FOLDER = "out"

import copy
import os
import uuid
import torch
import yaml

from affex.data import get_dataloaders
from affex.data.utils import BatchKeys
from affex.explainer import build_explainer, get_explanation_mask
from affex.metrics import FSSCausalMetric
from affex.models import build_model, build_model_preconfigured
from affex.substitution import Substitutor
from affex.utils.logger import get_logger
from affex.utils.utils import to_device

from torchmetrics import MetricCollection


def set_seed(seed):
    """Set the seed for reproducibility."""
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def evaluate(parameters, run_name=None, log_params=True, log_on_file=True):
    set_seed(parameters.get("seed", 42))

    if run_name is None:
        run_name = str(uuid.uuid4())[:8] + ".log"
        run_name = os.path.join(OUT_FOLDER, run_name)
        os.makedirs(OUT_FOLDER, exist_ok=True)
    # model filename is log filename but with .pt instead of .log
    params_filename = run_name + ".yaml"
    if log_params:
        with open(params_filename, "w") as f:
            yaml.dump(parameters, f)

    log_filename = run_name + ".log" if log_on_file else None
    logger = get_logger("Eval", log_filename)
    logger.info("parameters:")
    logger.info(parameters)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    logger.info(f"Running on {device}")

    model, image_size = build_model_preconfigured(model_name=parameters["model"])
    model.eval()
    model.to(device)

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
                step=parameters["metric"]["step"],
                substrate_fn=parameters["metric"]["substrate_fn"],
            ),
            "dauc": FSSCausalMetric(
                model=model,
                mode="del",
                step=parameters["metric"]["step"],
                substrate_fn=parameters["metric"]["substrate_fn"],
            ),
        }
    )

    for dataset_name, val_dataloader in val.items():
        logger.info(f"Evaluating {dataset_name} dataset")
        for i, batch in enumerate(val_dataloader):
            logger.info(f"Processing batch {i}")
            batch, dataset_name = batch

            substitutor = Substitutor(substitute=False)
            substitutor.reset(batch=batch)
            input_dict, gt = next(substitutor)
            input_dict = to_device(input_dict, device)
            
            with torch.no_grad():
                result = model(input_dict)

            explanation_size = explanation_size or input_dict[BatchKeys.IMAGES].shape[-2:]
            explanation_mask = get_explanation_mask(input_dict, gt, result, explanation_size, masking_type)

            explanation = explainer.explain(
                input_dict=input_dict,
                explanation_mask=explanation_mask,
                explanation_size=explanation_size,
            )
            assert len(explanation) == 1, "Only support one class for now"
            explanation = explanation[0]
            metrics.update(
                input_dict=input_dict,
                explanation=explanation,
                explanation_mask=explanation_mask,
            )
