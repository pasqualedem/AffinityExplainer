import math
from typing import Dict, List, Tuple
import torch
from einops import rearrange, repeat

import torch.nn.functional as F
import torch.nn as nn
from torchvision.transforms import functional as TvT
from torchvision.transforms.functional import resize
from captum.attr import IntegratedGradients, Saliency, LayerGradCam, GradientShap, DeepLift


from affex.data.utils import BatchKeys, min_max_scale
from affex.lime import FSSLime
from affex.shap import FSSGradientShap
from affex.utils.segmentation import unnormalize
from affex.utils.utils import ResultDict
from affex.models.dcama import DCAMAMultiClass
from affex.models.dmtnet import DMTNetMultiClass


GRADIENT_KEYS = [
    BatchKeys.IMAGES,
    BatchKeys.PROMPT_MASKS,
]
NON_GRADIENT_KEYS = [
    BatchKeys.FLAG_MASKS,
    BatchKeys.FLAG_EXAMPLES,
    BatchKeys.DIMS,
]

class LamLayerGradCam(LayerGradCam):
    def attribute(self, *args, **kwargs):
        print(args)
        inputs = args[0]
        b, m, c, h, w = inputs[0].shape
        attrs = super().attribute(*args, **kwargs)
        size = int(math.sqrt(attrs[0].shape[0]))
        print(attrs)
        return (rearrange(attrs.sum(dim=-1), " b (h w) -> b h w", h=size),)
    

class TraditionalExplainer(nn.Module):
    methods = {
        "integrated_gradients": (IntegratedGradients, {}, {"n_steps": 25, "internal_batch_size": 1}),
        "saliency": (Saliency, {}, {}),
        "gradcam": (LamLayerGradCam, {}, {"attr_dim_summation": False}),
        "gradient_shap": (FSSGradientShap, {}, {}),
        "deep_lift": (DeepLift, {}, {}),
        "lime": (FSSLime, {}, {"use_baselines": True})
    }
    def __init__(self, model: nn.Module, method: str = "integrated gradients", layer: str = None, **kwargs):
        super(TraditionalExplainer, self).__init__()
        self.model = model
        
        method, init_kwargs, forward_kwargs = self.methods[method]
        if method == LamLayerGradCam:
            init_kwargs["layer"] = layer
        self.method = method(self, **init_kwargs, **kwargs)
        self.method_kwargs = forward_kwargs

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

        results = self.model(batched_input, postprocess=False)
        logits = results[ResultDict.LOGITS]
        out = logits[:, :, self.explanation_mask].mean(dim=-1)
        return out

    def explain(self, input_dict, explanation_mask, chosen_classes=None, **kwargs):
        self.prepare(explanation_mask)
        # Get a tuple from the batch
        main_input = {key: input_dict[key] for key in input_dict.keys() if key in GRADIENT_KEYS}

        additional_input = {
            key: input_dict[key] for key in input_dict.keys() if key in NON_GRADIENT_KEYS
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


def dmtnet_preprocess_attentions(attentions):
    processed_attentions = []
    for class_attns in attentions:
        num_shots = len(class_attns)
        level_attentions = [
            torch.cat(
                [
                    class_attns[i][0][j] + class_attns[i][1][j]
                    for i in range(num_shots)
                ],
                dim=-2
            )
            for j in range(len(class_attns[0][0]))
        ]

        processed_level_attentions = []
        for level_attn in level_attentions:
            level_attn = rearrange(level_attn, "b c h1 w1 nh2 w2 -> b c (h1 w1) (nh2 w2)")
            for i in range(level_attn.shape[1]):
                processed_level_attentions.append(level_attn[:, i])
        processed_attentions.append(processed_level_attentions)
    return processed_attentions


def dcama_preprocess_attentions(attentions):
        processed_attentions = []
        for class_attns in attentions:
            processed_level_attentions = []
            for level_attn in class_attns:
                level_attn = level_attn.mean(dim=1)
                processed_level_attentions.append(level_attn)
            processed_attentions.append(processed_level_attentions)
        return processed_attentions


MODEL_EXPLAINER_REGISTRY = {
    DMTNetMultiClass: dmtnet_preprocess_attentions,
    DCAMAMultiClass: dcama_preprocess_attentions
}


def preprocess_attentions(model, attentions):
    return MODEL_EXPLAINER_REGISTRY[model.__class__](attentions)


def get_explanation_mask(input_dict, gt, result, target_shape, masking_type="logits"):
    
    n_ways = input_dict[BatchKeys.PROMPT_MASKS].shape[2] - 1
    
    if masking_type == "logits":    
        logits = F.interpolate(
                result[ResultDict.LOGITS],
                size=target_shape,
                mode="bilinear",
                align_corners=False,
                antialias=False,
            ).argmax(dim=1)
        explanation_mask = F.one_hot(logits, num_classes=n_ways+1).permute(0, 3, 1, 2)[0].bool()[1]
    elif masking_type == "gt" or masking_type == "ground_truth":
        gt = F.interpolate(
            gt.float().unsqueeze(1),
            size=(target_shape, target_shape),
            mode="nearest",
        )[:, 0]
        gt[gt == -100] = 0  # Convert -100 to 0 for ground truth
        explanation_mask = F.one_hot(gt.long(), num_classes=n_ways+1).permute(0, 3, 1, 2)[0].bool()[1]
    elif masking_type == "all":
        explanation_mask = torch.ones((target_shape, target_shape), dtype=torch.bool)
    
    return explanation_mask


class AffinityExplainer:
    def __init__(self, model, aggregation_method="feature_ablation", use_softmax=True):
        self.model = model
        self.aggregation_method = aggregation_method
        self.use_softmax = use_softmax
        if not any(isinstance(model, cls) for cls in MODEL_EXPLAINER_REGISTRY.keys()):
            raise ValueError(f"Model {model.__class__.__name__} is not supported for explanations. Supported models: {list(EXPLAINER_REGISTRY.keys())}")
        assert hasattr(model, "feature_ablation"), f"Model {model.__class__.__name__} does not have a feature_ablation method for explanations."

    def explain(self, input_dict, result=None, explanation_mask="logits", explanation_size=None, selected_classes=None, gt=None):
        
        if result is None:
            with torch.no_grad():
                result = self.model(input_dict, postprocess=False)
        
        masks = input_dict[BatchKeys.PROMPT_MASKS]
        flag_examples = input_dict[BatchKeys.FLAG_EXAMPLES]
        num_classes = masks.shape[2] - 1
        
        if selected_classes is None:
            selected_classes = list(range(num_classes))
        
        if explanation_size is None:
            explanation_size = input_dict[BatchKeys.IMAGES][:, 0].shape[2:]
            
        if explanation_mask == "logits":
            explanation_mask = get_explanation_mask(input_dict, gt=None, result=result, target_shape=explanation_size, masking_type=explanation_mask)
        elif explanation_mask == "gt" or explanation_mask == "ground_truth":
            assert gt is not None, "Ground truth (gt) must be provided when using 'gt' or 'ground_truth' masking type."
            explanation_mask = get_explanation_mask(input_dict, gt=gt, result=result, target_shape=explanation_size, masking_type=explanation_mask)

        if len(explanation_mask.shape) == 2: # We need class dimension
            explanation_mask = repeat(explanation_mask, "h w -> c h w", c=num_classes)
        
        attns = preprocess_attentions(self.model, result[ResultDict.ATTENTIONS])
        explanations = []
        for chosen_class in selected_classes:
            class_attns = attns[chosen_class]
            class_examples = flag_examples[:, :, chosen_class + 1]
            mask = masks[:, :, chosen_class + 1, ::][class_examples]
            class_shots = mask.shape[0]

            support_mask = resize(mask, explanation_size, interpolation=TvT.InterpolationMode.NEAREST).float()
            support_mask = rearrange(support_mask, "n h w -> h (n w)")
            support_mask = 2 * support_mask - 1

            level_contributions = []
            level_predictions = []
            for level_contribution in class_attns:
                hw = level_contribution.shape[-2]
                h = w = int(hw ** 0.5)
                
                # Get the attention map for the chosen class 
                if self.use_softmax:
                    level_contribution = F.softmax(level_contribution, dim=-1)
                # mask_level = rearrange(resize(mask, explanation_size, interpolation=TvT.InterpolationMode.NEAREST), "n h w -> h (n w)")
                
                # Transpose and resize the attention map
                level_contribution = rearrange(level_contribution, "b (hq wq) (n hs ws) -> (b hs ws n) hq wq", n=class_shots, hs=h, ws=w, hq=h, wq=w)
                level_contribution = resize(level_contribution, explanation_size)
                # reshaped_level_attn = rearrange(level_attn, "b (hq wq) (n hs ws) -> (b hq wq n) hs ws", n=class_shots, hs=h, ws=w, hq=h, wq=w)
                # reshaped_level_attn = resize(reshaped_level_attn, explanation_size)
                
                # Normalize the attention map
                norm = level_contribution.sum(dim=(-1, -2), keepdim=True).add_(1e-6) # In place normalization
                level_contribution.div_(norm)
                # level_contribution = level_contribution / (level_contribution.sum(dim=(-1, -2), keepdim=True) + 1e-6) # Not in place normalization
                # reshaped_level_attn = reshaped_level_attn / (reshaped_level_attn.sum(dim=(-1, -2), keepdim=True) + 1e-6)
                # reshaped_level_attn = rearrange(reshaped_level_attn, "(b n) h w -> b h (n w)", n=class_shots)
                # level_prediction = rearrange((reshaped_level_attn * mask_level).sum(dim=(-1, -2)), "(h w) -> 1 h w", h=h, w=w)
                
                # Get the mean contribution for the chosen class
                level_contribution = level_contribution[:, explanation_mask[chosen_class]].mean(dim=1)
                level_contribution = rearrange(level_contribution, "(b hs ws n) -> (b n) hs ws", hs=h, ws=w, n=class_shots)
                level_contribution = resize(level_contribution, explanation_size, interpolation=TvT.InterpolationMode.BILINEAR, antialias=False)
                level_contribution = rearrange(level_contribution, "(b n) h w -> b n h w", n=class_shots)
                level_contribution = level_contribution / (level_contribution.sum(dim=(-1, -2, -3), keepdim=True) + 1e-6)
                level_contributions.append(level_contribution)
                # level_predictions.append(level_prediction)

            contrib_seq = torch.stack(level_contributions, dim=1)  # B C N H W

            if self.aggregation_method == "feature_ablation":
                cmask_contrib = self.model.feature_ablation(result, chosen_class, explanation_mask, n_shots=class_shots, explanation_size=explanation_size)
                if cmask_contrib is None:
                    cmask_contrib = torch.full((contrib_seq.shape[0],), 1 / contrib_seq.shape[0])
                cmask_contrib = rearrange(cmask_contrib, "c -> 1 c 1 1 1") # B C N H W
                weighted_contrib = min_max_scale((contrib_seq * cmask_contrib).sum(dim=1))
            elif self.aggregation_method == "mean":
                mean_contrib = contrib_seq.mean(dim=1)
                mean_contrib = min_max_scale(mean_contrib)
                weighted_contrib = mean_contrib
            else:
                raise ValueError(f"Aggregation method {self.aggregation_method} is not supported. Supported methods: ['feature_ablation', 'mean']")

            # explanations.append((mean_contrib, weighted_contrib, contrib_seq, level_predictions, support_mask, class_shots))
            explanations.append(weighted_contrib)
        return explanations
    
    
class RandomExplainer:
    def __init__(self, model):
        self.model = model
        
    def explain(self, input_dict, result=None, explanation_mask="logits", explanation_size=None, selected_classes=None):
        images = input_dict[BatchKeys.IMAGES]
        support_images = images[:, 1:]
        
        masks = input_dict[BatchKeys.PROMPT_MASKS]
        num_classes = masks.shape[2] - 1
        if selected_classes is None:
            selected_classes = list(range(num_classes))
        
        explanations = []
        
        for chosen_class in selected_classes:
            explanation = torch.rand_like(support_images)[:, :, 0]
            explanations.append(explanation)
        
        return explanations
    
    
EXPLAINER_REGISTRY = {
    "integrated_gradients": TraditionalExplainer,
    "saliency": TraditionalExplainer,
    "gradcam": TraditionalExplainer,
    "gradient_shap": TraditionalExplainer,
    "deep_lift": TraditionalExplainer,
    "affinity": AffinityExplainer,
    "random": RandomExplainer,
    "lime": TraditionalExplainer,
}


def build_explainer(model, name, params, device="cpu"):
    params = params.copy()
    
    if name not in EXPLAINER_REGISTRY:
        raise ValueError(f"Explainer {name} is not supported. Supported explainers: {list(EXPLAINER_REGISTRY.keys())}")
    
    explainer_class = EXPLAINER_REGISTRY[name]
    if name == "affinity" or name == "random":
        return explainer_class(model, **params)
    
    return explainer_class(model=model, method=name, **params).to(device)