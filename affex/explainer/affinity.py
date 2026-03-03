import torch
from einops import rearrange, repeat
import torch.nn.functional as F
from torchvision.transforms import functional as TvT
from torchvision.transforms.functional import resize
from torchvision import transforms as T

from ..data.utils import BatchKeys, min_max_scale
from ..models.dcama import DCAMAMultiClass
from ..models.dmtnet import DMTNetMultiClass
from ..utils.utils import ResultDict



def dmtnet_preprocess_attentions(attentions):
    processed_attentions = []
    for class_attns in attentions:
        num_shots = len(class_attns)
        level_attentions = [
            torch.cat(
                [class_attns[i][0][j] + class_attns[i][1][j] for i in range(num_shots)],
                dim=-2,
            )
            for j in range(len(class_attns[0][0]))
        ]

        processed_level_attentions = []
        for level_attn in level_attentions:
            level_attn = rearrange(
                level_attn, "b c h1 w1 nh2 w2 -> b c (h1 w1) (nh2 w2)"
            )
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
    DMTNetMultiClass.__name__: dmtnet_preprocess_attentions,
    DCAMAMultiClass.__name__: dcama_preprocess_attentions,
}


def preprocess_attentions(model, attentions):
    return MODEL_EXPLAINER_REGISTRY[model.__class__.__name__](attentions)


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
        explanation_mask = (
            F.one_hot(logits, num_classes=n_ways + 1).permute(0, 3, 1, 2)[0].bool()[1]
        )
    elif masking_type == "gt" or masking_type == "ground_truth":
        gt = F.interpolate(
            gt.float().unsqueeze(1),
            size=(target_shape, target_shape),
            mode="nearest",
        )[:, 0]
        gt[gt == -100] = 0  # Convert -100 to 0 for ground truth
        explanation_mask = (
            F.one_hot(gt.long(), num_classes=n_ways + 1)
            .permute(0, 3, 1, 2)[0]
            .bool()[1]
        )
    elif masking_type == "all":
        explanation_mask = torch.ones((target_shape, target_shape), dtype=torch.bool)

    return explanation_mask


class AffinityExplainer:
    def __init__(self, model, aggregation_method="feature_ablation", explanation_size=None, use_softmax=True, masking=False, mask_blur_kernel_size=31, mask_blur_sigma=50):
        self.model = model
        self.aggregation_method = aggregation_method
        self.use_softmax = use_softmax
        self.masking = masking
        self.blur = T.GaussianBlur(kernel_size=mask_blur_kernel_size, sigma=mask_blur_sigma)

        if isinstance(explanation_size, int):
            explanation_size = (explanation_size, explanation_size)
            
        self.explanation_size = explanation_size
        if not any(model.__class__.__name__ == cls for cls in MODEL_EXPLAINER_REGISTRY.keys()):
            raise ValueError(
                f"Model {model.__class__.__name__} is not supported for explanations. Supported models: {list(MODEL_EXPLAINER_REGISTRY.keys())}"
            )
        assert hasattr(
            model, "feature_ablation"
        ), f"Model {model.__class__.__name__} does not have a feature_ablation method for explanations."

    def explain(
        self,
        input_dict,
        result=None,
        explanation_mask="logits",
        selected_classes=None,
        gt=None,
    ):

        if result is None:
            with torch.no_grad():
                result = self.model(input_dict, postprocess=False)

        masks = input_dict[BatchKeys.PROMPT_MASKS]
        flag_examples = input_dict[BatchKeys.FLAG_EXAMPLES]
        num_classes = masks.shape[2] - 1

        if selected_classes is None:
            selected_classes = list(range(num_classes))

        image_size = input_dict[BatchKeys.IMAGES].shape[-2:]
        if self.explanation_size is None:
            explanation_size = image_size
        else:
            explanation_size = self.explanation_size

        if explanation_mask == "logits":
            explanation_mask = get_explanation_mask(
                input_dict,
                gt=None,
                result=result,
                target_shape=explanation_size,
                masking_type=explanation_mask,
            )
        elif explanation_mask == "gt" or explanation_mask == "ground_truth":
            assert (
                gt is not None
            ), "Ground truth (gt) must be provided when using 'gt' or 'ground_truth' masking type."
            explanation_mask = get_explanation_mask(
                input_dict,
                gt=gt,
                result=result,
                target_shape=explanation_size,
                masking_type=explanation_mask,
            )
        elif isinstance(explanation_mask, torch.Tensor):
            if explanation_mask.shape[-2:] != explanation_size:
                explanation_mask = resize(
                    explanation_mask.float().unsqueeze(0),
                    explanation_size,
                    interpolation=TvT.InterpolationMode.NEAREST,
                ).bool().squeeze(0)

        if len(explanation_mask.shape) == 2:  # We need class dimension
            explanation_mask = repeat(explanation_mask, "h w -> c h w", c=num_classes)

        attns = preprocess_attentions(self.model, result[ResultDict.ATTENTIONS])
        explanations = []
        for chosen_class in selected_classes:
            class_attns = attns[chosen_class]
            class_examples = flag_examples[:, :, chosen_class + 1]
            mask = masks[:, :, chosen_class + 1, ::][class_examples]
            class_shots = mask.shape[0]

            mask = resize(
                mask, explanation_size, interpolation=TvT.InterpolationMode.NEAREST
            ).float()

            level_contributions = []
            level_predictions = []
            for level_contribution in class_attns:
                hw = level_contribution.shape[-2]
                h = w = int(hw**0.5)

                # Get the attention map for the chosen class
                if self.use_softmax:
                    level_contribution = F.softmax(level_contribution, dim=-1)
                # mask_level = rearrange(resize(mask, explanation_size, interpolation=TvT.InterpolationMode.NEAREST), "n h w -> h (n w)")

                # Transpose and resize the attention map
                level_contribution = rearrange(
                    level_contribution,
                    "b (hq wq) (n hs ws) -> (b hs ws n) hq wq",
                    n=class_shots,
                    hs=h,
                    ws=w,
                    hq=h,
                    wq=w,
                )
                level_contribution = resize(level_contribution, explanation_size)
                # reshaped_level_attn = rearrange(level_attn, "b (hq wq) (n hs ws) -> (b hq wq n) hs ws", n=class_shots, hs=h, ws=w, hq=h, wq=w)
                # reshaped_level_attn = resize(reshaped_level_attn, explanation_size)

                # Normalize the attention map
                norm = level_contribution.sum(dim=(-1, -2), keepdim=True).add_(
                    1e-6
                )  # In place normalization
                level_contribution.div_(norm)
                # level_contribution = level_contribution / (level_contribution.sum(dim=(-1, -2), keepdim=True) + 1e-6) # Not in place normalization
                # reshaped_level_attn = reshaped_level_attn / (reshaped_level_attn.sum(dim=(-1, -2), keepdim=True) + 1e-6)
                # reshaped_level_attn = rearrange(reshaped_level_attn, "(b n) h w -> b h (n w)", n=class_shots)
                # level_prediction = rearrange((reshaped_level_attn * mask_level).sum(dim=(-1, -2)), "(h w) -> 1 h w", h=h, w=w)

                # Get the mean contribution for the chosen class
                level_contribution = level_contribution[
                    :, explanation_mask[chosen_class]
                ].mean(dim=1)
                level_contribution = rearrange(
                    level_contribution,
                    "(b hs ws n) -> (b n) hs ws",
                    hs=h,
                    ws=w,
                    n=class_shots,
                )
                level_contribution = resize(
                    level_contribution,
                    explanation_size,
                    interpolation=TvT.InterpolationMode.BILINEAR,
                    antialias=False,
                )
                level_contribution = rearrange(
                    level_contribution, "(b n) h w -> b n h w", n=class_shots
                )

                # if self.masking:
                #     level_support_mask = self.blur(mask)
                #     if self.masking == "sign":
                #         level_support_mask = 2 * level_support_mask - 1
                #     level_contribution = level_contribution * level_support_mask

                level_contribution = level_contribution / (
                    level_contribution.sum(dim=(-1, -2, -3), keepdim=True) + 1e-6
                )
                level_contributions.append(level_contribution)

            contrib_seq = torch.stack(level_contributions, dim=1)  # B C N H W

            if self.aggregation_method == "feature_ablation":
                cmask_contrib = self.model.feature_ablation(
                    result,
                    chosen_class,
                    explanation_mask,
                    n_shots=class_shots,
                    explanation_size=explanation_size,
                )
                if cmask_contrib is None:
                    cmask_contrib = torch.full(
                        (contrib_seq.shape[0],), 1 / contrib_seq.shape[0]
                    )
                cmask_contrib = rearrange(cmask_contrib, "c -> 1 c 1 1 1")  # B C N H W
                weighted_contrib = min_max_scale(
                    (contrib_seq * cmask_contrib).sum(dim=1)
                )
            elif self.aggregation_method == "mean":
                mean_contrib = contrib_seq.mean(dim=1)
                mean_contrib = min_max_scale(mean_contrib)
                weighted_contrib = mean_contrib
            else:
                raise ValueError(
                    f"Aggregation method {self.aggregation_method} is not supported. Supported methods: ['feature_ablation', 'mean']"
                )

            if weighted_contrib.shape[-2:] != image_size:
                weighted_contrib = resize(
                    weighted_contrib,
                    image_size,
                    interpolation=TvT.InterpolationMode.BILINEAR,
                    antialias=False,
                )
                
            if self.masking:
                curr_mask = self.blur(mask)
                if self.masking == "sign":
                    curr_mask = 2 * curr_mask - 1
                weighted_contrib = min_max_scale(weighted_contrib * curr_mask)

            # explanations.append((mean_contrib, weighted_contrib, contrib_seq, level_predictions, support_mask, class_shots))
            explanations.append(weighted_contrib)
            
        return explanations
    
    
class MaskedAffinityExplainer(AffinityExplainer):
    def __init__(self, model, **kwargs):
        super().__init__(model, **kwargs, masking=True)


class SignedAffinityExplainer(AffinityExplainer):
    def __init__(self, model, **kwargs):
        super().__init__(model, **kwargs, masking="sign")