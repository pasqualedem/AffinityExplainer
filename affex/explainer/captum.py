from einops import rearrange


from .lime import FSSLime
from .shap import FSSGradientShap
from ..data.utils import BatchKeys, min_max_scale
from ..utils.utils import ResultDict

import math
import torch
import torch.nn as nn
from captum.attr import DeepLift, IntegratedGradients, LayerGradCam, Saliency


import inspect
from typing import Dict, List, Tuple


class LamLayerGradCam(LayerGradCam):
    def attribute(self, *args, **kwargs):
        print(args)
        inputs = args[0]
        b, m, c, h, w = inputs[0].shape
        attrs = super().attribute(*args, **kwargs)
        size = int(math.sqrt(attrs[0].shape[0]))
        print(attrs)
        return (rearrange(attrs.sum(dim=-1), " b (h w) -> b h w", h=size),)


GRADIENT_KEYS = [
    BatchKeys.IMAGES,
    BatchKeys.PROMPT_MASKS,
]
NON_GRADIENT_KEYS = [
    BatchKeys.FLAG_MASKS,
    BatchKeys.FLAG_EXAMPLES,
    BatchKeys.DIMS,
]


class CaptumExplainer(nn.Module):
    methods = {
        "integrated_gradients": (
            IntegratedGradients,
            {},
            {"n_steps": 25, "internal_batch_size": 1},
        ),
        "saliency": (Saliency, {}, {}),
        "gradcam": (LamLayerGradCam, {}, {"attr_dim_summation": False}),
        "gradient_shap": (FSSGradientShap, {}, {}),
        "deep_lift": (DeepLift, {}, {}),
        "lime": (FSSLime, {"use_baselines": False}, {}),
    }

    def __init__(
        self,
        model: nn.Module,
        name: str = "integrated gradients",
        layer: str = None,
        **kwargs,
    ):
        super(CaptumExplainer, self).__init__()
        self.model = model

        method, default_init_kwargs, forward_kwargs = self.methods[name]
        if method == LamLayerGradCam:
            default_init_kwargs["layer"] = layer

        passed_init_kwargs = set(inspect.signature(method.__init__).parameters.keys()).intersection(kwargs.keys())
        passed_attribute_kwargs = set(inspect.signature(method.attribute).parameters.keys()).intersection(kwargs.keys())
        # warning if any of the passed kwargs is not used
        unused_kwargs = set(kwargs.keys()) - (passed_init_kwargs | passed_attribute_kwargs)
        if len(unused_kwargs) > 0:
            print(f"Warning: the following kwargs are not used: {unused_kwargs}")

        init_kwargs = {**default_init_kwargs, **{k: kwargs[k] for k in passed_init_kwargs}}
        self.method = method(self, **init_kwargs)
        self.method_kwargs = {**forward_kwargs, **{k: kwargs[k] for k in passed_attribute_kwargs}}

    def prepare(self, explanation_mask):
        self.explanation_mask = explanation_mask

    def forward(
        self, *batched_input_tuple: Tuple[torch.Tensor, ...]
    ) -> List[Dict[str, torch.Tensor]]:

        tuple_mapping = batched_input_tuple[-1]

        # Prepare batched_input in the format expected by self._forward
        batched_input = {
            key: batched_input_tuple[value] for key, value in tuple_mapping.items()
        }

        # Some methods (LIME) may affect the inputs, so we need to re-prepare the batched_input
        if self.method.__class__.__name__ in ["FSSLime"]:
            if (
                not self.method.current_valid_masks.all()
            ):  # If the valid masks have changed, we need to remove the removed from the flag_examples
                flag_examples = batched_input[BatchKeys.FLAG_EXAMPLES]
                flag_examples = flag_examples[:, self.method.current_valid_masks]
                batched_input[BatchKeys.FLAG_EXAMPLES] = flag_examples

        results = self.model(batched_input, postprocess=False)
        logits = results[ResultDict.LOGITS]
        out = logits[:, :, self.explanation_mask].mean(dim=-1)
        return out

    def explain(self, input_dict, explanation_mask, chosen_classes=None, **kwargs):
        self.prepare(explanation_mask)
        # Get a tuple from the batch
        main_input = {
            key: input_dict[key] for key in input_dict.keys() if key in GRADIENT_KEYS
        }

        additional_input = {
            key: input_dict[key]
            for key in input_dict.keys()
            if key in NON_GRADIENT_KEYS
        }
        tuple_mapping = {key: i for i, key in enumerate(main_input.keys())} | {
            key: i + len(main_input) for i, key in enumerate(additional_input.keys())
        }
        main_input = tuple(main_input.values())
        additional_input = tuple(additional_input.values())
        additional_input += (tuple_mapping,)

        explanations = []
        if chosen_classes is None:
            chosen_classes = input_dict[BatchKeys.PROMPT_MASKS].shape[2]

        for chosen_class in range(1, chosen_classes):
            attribution_tuple = self.method.attribute(
                main_input,
                additional_forward_args=additional_input,
                target=chosen_class,
                **self.method_kwargs,
            )
            attribution_dict = dict(zip(tuple_mapping.keys(), attribution_tuple))
            explanation = attribution_dict[BatchKeys.IMAGES]
            explanation = explanation[:, 1:]
            explanation = min_max_scale(explanation.mean(dim=2))
            explanations.append(explanation)

        return explanations




