import torch
from einops import rearrange

import torch.nn.functional as F
from torchvision.transforms import functional as TvT
from torchvision.transforms.functional import resize


from affex.data.utils import min_max_scale
from affex.utils.utils import ResultDict
from affex.models.dcama import DCAMAMultiClass
from affex.models.dmtnet import DMTNetMultiClass


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


EXPLAINER_REGISTRY = {
    DMTNetMultiClass: dmtnet_preprocess_attentions,
    DCAMAMultiClass: dcama_preprocess_attentions
}


def preprocess_attentions(model, attentions):
    return EXPLAINER_REGISTRY[model.__class__](attentions)


def calculate_explanations(model, result, masks, flag_examples, explanation_mask, num_classes, explanation_size):
    attns = preprocess_attentions(model, result[ResultDict.ATTENTIONS])
    explanations = []
    for chosen_class in range(num_classes):
        class_attns = attns[chosen_class]
        class_examples = flag_examples[:, :, chosen_class + 1]
        mask = masks[:, :, chosen_class + 1, ::][class_examples]
        class_shots = mask.shape[0]

        support_mask = resize(mask, explanation_size, interpolation=TvT.InterpolationMode.NEAREST).float()
        support_mask = rearrange(support_mask, "n h w -> h (n w)")
        support_mask = 2 * support_mask - 1

        level_contributions = []
        level_predictions = []
        for level_attn in class_attns:
            hw = level_attn.shape[-2]
            h = w = int(hw ** 0.5)
            level_attn = F.softmax(level_attn, dim=-1)
            mask_level = rearrange(resize(mask, explanation_size, interpolation=TvT.InterpolationMode.NEAREST), "n h w -> h (n w)")
            transposed_level_attn = rearrange(level_attn, "b (hq wq) (n hs ws) -> (b hs ws n) hq wq", n=class_shots, hs=h, ws=w, hq=h, wq=w)
            reshaped_level_attn = rearrange(level_attn, "b (hq wq) (n hs ws) -> (b hq wq n) hs ws", n=class_shots, hs=h, ws=w, hq=h, wq=w)
            transposed_level_attn = resize(transposed_level_attn, explanation_size)
            reshaped_level_attn = resize(reshaped_level_attn, explanation_size)
            normalized_level_attn = transposed_level_attn / (transposed_level_attn.sum(dim=(-1, -2), keepdim=True) + 1e-6)
            reshaped_level_attn = reshaped_level_attn / (reshaped_level_attn.sum(dim=(-1, -2), keepdim=True) + 1e-6)
            reshaped_level_attn = rearrange(reshaped_level_attn, "(b n) h w -> b h (n w)", n=class_shots)
            level_prediction = rearrange((reshaped_level_attn * mask_level).sum(dim=(-1, -2)), "(h w) -> 1 h w", h=h, w=w)
            level_contribution = normalized_level_attn[:, explanation_mask[chosen_class + 1]].mean(dim=1)
            level_contribution = rearrange(level_contribution, "(b hs ws n) -> (b n) hs ws", hs=h, ws=w, n=class_shots)
            resized_level_contribution = resize(level_contribution, explanation_size, interpolation=TvT.InterpolationMode.BILINEAR, antialias=False)
            resized_level_contribution = rearrange(resized_level_contribution, "(b n) h w -> b h (n w)", n=class_shots)
            normalized_level_contribution = resized_level_contribution / (resized_level_contribution.sum(dim=(-1, -2), keepdim=True) + 1e-6)
            level_contributions.append(normalized_level_contribution)
            level_predictions.append(level_prediction)

        contrib_seq = torch.stack(level_contributions)
        mean_contrib = contrib_seq.mean(dim=0)

        cmask_contrib = model.feature_ablation(result, chosen_class, explanation_mask, n_shots=class_shots, image_size=explanation_size)
        if cmask_contrib is None:
            cmask_contrib = torch.full((contrib_seq.shape[0],), 1 / contrib_seq.shape[0])

        mean_contrib = min_max_scale(mean_contrib)

        cmask_contrib = rearrange(cmask_contrib, "c -> c 1 1 1")
        weighted_contrib = min_max_scale((contrib_seq * cmask_contrib).sum(dim=0))

        explanations.append((mean_contrib, weighted_contrib, contrib_seq, level_predictions, support_mask, class_shots))
    return explanations