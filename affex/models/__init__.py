# Code taken from https://github.com/facebookresearch/segment-anything
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
import torch

from collections import namedtuple
from torchvision.models import resnet50
from transformers import AutoImageProcessor

from affex.models.bam import build_bam
from affex.models.hdmnet import build_hdmnet
from affex.models.la.build_lam import build_lam, build_lam_vit_mae_b

from .dcama import build_dcama
from .dummy import build_dummy
from .dmtnet import build_dmtnet
from .patnet import build_patnet

ComposedOutput = namedtuple("ComposedOutput", ["main", "aux"])


def build_deit():
    processor = AutoImageProcessor.from_pretrained("facebook/deit-base-distilled-patch16-224")
    model = torch.hub.load('facebookresearch/deit:main',
                           'deit_tiny_patch16_224', pretrained=True)
    return model, processor


def build_resnet50():
    processor = AutoImageProcessor.from_pretrained("microsoft/resnet-50")
    model = resnet50(pretrained=True)
    return model, processor


MODEL_REGISTRY = {
    "dcama": build_dcama,
    "dummy": build_dummy,
    "deit": build_deit,
    "dmtnet": build_dmtnet,
    "patnet": build_patnet,
    "resnet50": build_resnet50,
    "hdmnet": build_hdmnet,
    "bam": build_bam,
    "la": build_lam_vit_mae_b,
}

def build_model(params):
    name = params["name"]
    params = {k: v for k, v in params.items() if k != "name"}
    return MODEL_REGISTRY[name](**params)


def get_dcama(dataset="pascal", val_fold_idx=0, use_pe=True, **kwargs):
    name = "dcama"
    params = dict(
        backbone_checkpoint="checkpoints/dcama/swin_base_patch4_window12_384.pth",
        model_checkpoint=f"checkpoints/dcama/{dataset}/swin_fold{val_fold_idx}.pt",
        pe=use_pe,
    )
    image_size = 384
    return MODEL_REGISTRY[name](**params), image_size


def get_bam(dataset="pascal", k_shots=1, val_fold_idx=0, **kwargs):
    name = "bam"
    params = dict(
        shots=k_shots,
        val_fold_idx=val_fold_idx,
        dataset=dataset,
    )
    image_size = 641
    bam = MODEL_REGISTRY[name](**params)
    return bam, image_size


def get_hdmnet(k_shots=1, val_fold_idx=0, **kwargs):
    name = "hdmnet"
    params = dict(
        shots=k_shots,
        val_fold_idx=val_fold_idx,
    )
    image_size = 641
    hdmnet = MODEL_REGISTRY[name](**params)
    return hdmnet, image_size


def get_dmtnet(**kwargs):
    name = "dmtnet"
    params = dict(
        model_checkpoint="checkpoints/dmtnet.pt",
    )
    image_size = 400
    dmtnet = MODEL_REGISTRY[name](**params)
    return dmtnet, image_size


SUPPORTED_MODELS = {
    "dcama": get_dcama,
    "bam": get_bam,
    "hdmnet": get_hdmnet,
    "dmtnet": get_dmtnet,
}


def build_model_preconfigured(model_name, **kwargs):
    return SUPPORTED_MODELS[model_name](**kwargs)