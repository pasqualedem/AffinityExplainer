from ..data.utils import BatchKeys, min_max_scale


import torch


class RandomExplainer:
    def __init__(self, model):
        self.model = model

    def explain(
        self,
        input_dict,
        result=None,
        explanation_mask="logits",
        explanation_size=None,
        selected_classes=None,
    ):
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
    
    
class GaussianNoiseMask:
    """
    Computes the explanation as random Gaussian noise on the support masks.
    """
    
    def __init__(self, model, mean=0.0, std=0.01):
        self.model = model
        self.mean = mean
        self.std = std

    def explain(
        self,
        input_dict,
        result=None,
        explanation_mask="logits",
        explanation_size=None,
        selected_classes=None,
    ):
        images = input_dict[BatchKeys.IMAGES]
        support_images = images[:, 1:]

        masks = input_dict[BatchKeys.PROMPT_MASKS]
        num_classes = masks.shape[2] - 1
        if selected_classes is None:
            selected_classes = list(range(num_classes))

        explanations = []

        for chosen_class in selected_classes:
            explanation = (torch.randn_like(masks[:, :, chosen_class+1]) * self.std + self.mean) + masks[:, :, chosen_class+1]
            # Normalize to [0, 1]
            explanation = min_max_scale(explanation, quantile=0.99, clamp=True)
            
            explanations.append(explanation)

        return explanations