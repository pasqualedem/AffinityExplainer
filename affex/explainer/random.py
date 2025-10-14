from affex.data.utils import BatchKeys


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