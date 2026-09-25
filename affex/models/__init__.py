import os

import torch

from collections import namedtuple

from .dcama import build_dcama
from .dmtnet import build_dmtnet
from .insid3 import build_insid3
from ..assets import checkpoints_dir, ensure
from .matcher import build_matcher
from .gfsam import build_gfsam
from .sansa import build_sansa_model
from .panet_head import build_panet

ComposedOutput = namedtuple("ComposedOutput", ["main", "aux"])


MODEL_REGISTRY = {
    "dcama": build_dcama,
    "dmtnet": build_dmtnet,
    "insid3": build_insid3,
    "matcher": build_matcher,
    "gfsam": build_gfsam,
    "sansa": build_sansa_model,
    "panet": build_panet,
}

def build_model(params):
    name = params["name"]
    params = {k: v for k, v in params.items() if k != "name"}
    return MODEL_REGISTRY[name](**params)


def get_dcama(dataset="pascal", val_fold_idx=0, use_pe=True, **kwargs):
    name = "dcama"
    backbone = ensure("dcama-backbone")
    if (dataset, val_fold_idx) == ("pascal", 0):
        model_ckpt = ensure("dcama-pascal-fold0")
    else:
        model_ckpt = os.path.join(checkpoints_dir(), f"dcama/{dataset}/swin_fold{val_fold_idx}.pt")
    params = dict(
        backbone_checkpoint=backbone,
        model_checkpoint=model_ckpt,
        pe=use_pe,
    )
    image_size = 384
    return MODEL_REGISTRY[name](**params), image_size




def get_dmtnet(**kwargs):
    name = "dmtnet"
    params = dict(
        model_checkpoint=ensure("dmtnet"),
        voting=False,
    )
    image_size = 400
    dmtnet = MODEL_REGISTRY[name](**params)
    return dmtnet, image_size


def get_insid3(model_size="large", image_size=1024, mask_refiner="bilinear", differentiable=False, **kwargs):
    name = "insid3"
    import inspect
    accepted = set(inspect.signature(MODEL_REGISTRY[name]).parameters.keys())
    params = dict(
        model_size=model_size,
        image_size=image_size,
        mask_refiner=mask_refiner,
        differentiable=differentiable,
        **{k: v for k, v in kwargs.items() if k in accepted},
    )
    model = MODEL_REGISTRY[name](**params)
    return model, image_size


def get_matcher(sam_size="vit_b", image_size=518, **kwargs):
    model = MODEL_REGISTRY["matcher"](sam_size=sam_size, image_size=image_size, **kwargs)
    return model, image_size


def get_gfsam(sam_size="vit_b", image_size=1024, **kwargs):
    model = MODEL_REGISTRY["gfsam"](sam_size=sam_size, image_size=image_size, **kwargs)
    return model, image_size


def get_sansa(sam2_version="large", image_size=1024, **kwargs):
    model = MODEL_REGISTRY["sansa"](sam2_version=sam2_version, image_size=image_size, **kwargs)
    return model, image_size


def get_matcher_prompt(sam_size="vit_b", image_size=518, **kwargs):
    """Decoder-aware Matcher: AffEx reads only the support<->query matches that survive into
    SAM point prompts (Match composed with the SAM decoder's prompt selection)."""
    model = MODEL_REGISTRY["matcher"](sam_size=sam_size, image_size=image_size,
                                      attribution_mode="prompt", **kwargs)
    return model, image_size


def get_panet(dataset="pascal", val_fold_idx=0, **kwargs):
    name = "panet"
    params = dict(
        backbone_checkpoint=ensure("dcama-backbone"),
        model_checkpoint=os.path.join(checkpoints_dir(), f"panet/{dataset}/swin_fold{val_fold_idx}.pt"),
    )
    image_size = 384
    return MODEL_REGISTRY[name](**params), image_size


SUPPORTED_MODELS = {
    "dcama": get_dcama,
    "panet": get_panet,
    "dmtnet": get_dmtnet,
    "insid3": get_insid3,
    "matcher": get_matcher,
    "matcher_prompt": get_matcher_prompt,
    "gfsam": get_gfsam,
    "sansa": get_sansa,
}


def build_model_preconfigured(model_name, **kwargs):
    return SUPPORTED_MODELS[model_name](**kwargs)