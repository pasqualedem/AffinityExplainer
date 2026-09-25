r"""Native prototype read-out explainer (reviewer M83E comparison).

For prototype-based heads (PANetMultiClass) this is the attribution the
prototypical-parts literature obtains by upsampling the pre-pooling similarity
maps: each support pixel is scored by the cosine similarity of its (projected)
feature to the foreground prototype, per level, and the level maps are averaged
at image resolution. Computed inside the model forward and read out here.
"""
import torch
import torch.nn.functional as F

from ..data.utils import BatchKeys, min_max_scale
from ..utils.utils import ResultDict


class PrototypeReadoutExplainer:
    def __init__(self, model, explanation_size=None, **kwargs):
        self.model = model
        assert hasattr(model, 'model'), 'PrototypeReadoutExplainer needs a head module'

    def explain(self, input_dict, result=None, explanation_mask="logits",
                selected_classes=None, gt=None):
        if result is None:
            with torch.no_grad():
                result = self.model(input_dict, postprocess=False)

        readouts = result['prototype_readout']
        num_classes = input_dict[BatchKeys.PROMPT_MASKS].shape[2] - 1
        if selected_classes is None:
            selected_classes = list(range(num_classes))
        image_size = input_dict[BatchKeys.IMAGES].shape[-2:]

        explanations = []
        for chosen_class in selected_classes:
            levels = readouts[chosen_class]  # list of B,N,h,w
            resized = [
                F.interpolate(l, size=image_size, mode='bilinear', align_corners=False)
                for l in levels
            ]
            explanations.append(min_max_scale(torch.stack(resized).mean(dim=0)))
        return explanations
