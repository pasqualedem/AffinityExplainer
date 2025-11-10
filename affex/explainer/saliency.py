from einops import rearrange
import numpy as np

from ..utils.torch import clone_input_dict

from ..data.utils import BatchKeys, min_max_scale
from ..utils.utils import ResultDict

from .blur_ig import FSSBlurIG
from .xrai import FSSXRAI

import torch
import torch.nn as nn

import saliency.core as saliency

import inspect
from typing import Dict, List, Tuple


def make_baseline(baseline_type, shape):
    if baseline_type == "zero":
        baseline = np.zeros(shape)
    elif baseline_type == "random":
        baseline = np.random.rand(*shape)
    else:
        raise ValueError(
            f"Baseline {baseline_type} not supported"
        )
    return baseline


class SaliencyExplainer(nn.Module):
    methods = {
        "integrated_gradients": (
            saliency.IntegratedGradients,
            {},
            dict(batch_size=1, steps=25, baseline="zero"),
        ),
        "guided_ig": (
            saliency.GuidedIG,
            {},
            dict(x_steps=25, max_dist=1.0, fraction=0.5),
        ),
        "blur_ig": (
            FSSBlurIG,
            {},
            dict(max_sigma=50, steps=25, grad_step=0.01, sqrt=False),
        ),
        "xrai": (
            FSSXRAI,
            {},
            dict(batch_size=1, baseline="zero"),
        ),
    }

    def __init__(
        self,
        model: nn.Module,
        name: str = "integrated gradients",
        **kwargs,
    ):
        super(SaliencyExplainer, self).__init__()
        self.model = model

        method, default_init_kwargs, forward_kwargs = self.methods[name]

        passed_init_kwargs = set(
            inspect.signature(method.__init__).parameters.keys()
        ).intersection(kwargs.keys())
        passed_attribute_kwargs = set(
            inspect.signature(method.GetMask).parameters.keys()
        ).intersection(kwargs.keys())
        # warning if any of the passed kwargs is not used
        unused_kwargs = set(kwargs.keys()) - (
            passed_init_kwargs | passed_attribute_kwargs
        )
        if len(unused_kwargs) > 0:
            print(f"Warning: the following kwargs are not used: {unused_kwargs}")

        init_kwargs = {
            **default_init_kwargs,
            **{k: kwargs[k] for k in passed_init_kwargs},
        }
        self.method = method(**init_kwargs)
        self.method_kwargs = {
            **forward_kwargs,
            **{k: kwargs[k] for k in passed_attribute_kwargs},
        }

        self.device = next(model.parameters()).device

    def prepare(self, explanation_mask):
        self.explanation_mask = explanation_mask

    def call_model_function(self, images, call_model_args=None, expected_keys=None):
        images = torch.tensor(images, device=self.device, dtype=torch.float32)
        images.requires_grad = True
        input_dict = call_model_args["input_dict"]

        input_dict[BatchKeys.IMAGES] = images
        target_class_idx = call_model_args["target"]
        output = self.model(input_dict, postprocess=False)[ResultDict.LOGITS]
        m = torch.nn.Softmax(dim=1)
        output = m(output)
        if saliency.base.INPUT_OUTPUT_GRADIENTS in expected_keys:
            outputs = output[:, target_class_idx, self.explanation_mask].mean()
            grads = torch.autograd.grad(
                outputs, images, grad_outputs=torch.ones_like(outputs)
            )
            # grads = torch.movedim(grads[0], 1, 3)
            grads = grads[0]
            gradients = grads.cpu().detach().numpy()
            return {saliency.base.INPUT_OUTPUT_GRADIENTS: gradients}
        else:
            raise ValueError(f"Expected keys {expected_keys} not supported")
            one_hot = torch.zeros_like(output)
            one_hot[:, target_class_idx] = 1
            model.zero_grad()
            output.backward(gradient=one_hot, retain_graph=True)
            return conv_layer_outputs

    def forward(
        self, input_dict: Dict[str, torch.Tensor], batched_input: torch.Tensor
    ) -> List[Dict[str, torch.Tensor]]:

        results = self.model(input_dict, postprocess=False)
        logits = results[ResultDict.LOGITS]
        out = logits[:, :, self.explanation_mask].mean(dim=-1)
        return out

    def explain(self, input_dict, explanation_mask, chosen_classes=None, **kwargs):
        self.prepare(explanation_mask)
        explanations = []
        if chosen_classes is None:
            chosen_classes = input_dict[BatchKeys.PROMPT_MASKS].shape[2]

        images = input_dict[BatchKeys.IMAGES].detach().cpu()

        assert len(images) == 1, "Only support batch size 1 for now"
        images = images[0].cpu().numpy()
        
        curr_method_kwargs = self.method_kwargs.copy()

        if "baseline" in curr_method_kwargs:

            if type(self.method) == FSSXRAI:
                shape = images.shape
                baseline = make_baseline(self.method_kwargs["baseline"], shape)
                curr_method_kwargs["baselines"] = [baseline]
                curr_method_kwargs.pop("baseline")
            else:
                curr_method_kwargs["baseline"] = make_baseline(
                    curr_method_kwargs["baseline"], images[0].shape
                )

        for chosen_class in range(1, chosen_classes):
            attribution = self.method.GetMask(
                images,
                call_model_function=self.call_model_function,
                call_model_args={
                    "target": chosen_class,
                    "input_dict": clone_input_dict(input_dict),
                },
                **curr_method_kwargs,
            )
            
            if type(self.method) == FSSXRAI:
                explanation = min_max_scale(torch.tensor(attribution).unsqueeze(0), quantile=0.99, clamp=True)
            else:
                # Attribution post-processing
                explanation = torch.tensor(attribution)[1:].unsqueeze(0)
                explanation = min_max_scale(
                    torch.abs(explanation).mean(dim=2), quantile=0.99, clamp=True
                )
            explanations.append(explanation)

        return explanations
