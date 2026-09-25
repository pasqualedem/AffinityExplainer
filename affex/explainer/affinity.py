import torch
from einops import rearrange, repeat
import torch.nn.functional as F
import torchvision
from torchvision.transforms import functional as TvT
from torchvision.transforms.functional import resize
from torchvision import transforms as T

from ..data.utils import BatchKeys, min_max_scale
from ..models.dcama import DCAMAMultiClass
from ..models.dmtnet import DMTNetMultiClass
from ..models.insid3 import INSID3MultiClass
from ..models.matcher import MatcherMultiClass
from ..models.gfsam import GFSAMMultiClass
from ..models.sansa import SANSAMultiClass
from ..models.panet_head import PANetMultiClass
from ..utils.utils import ResultDict


def dilate_mask(mask: torch.Tensor, radius: int = 1, kernel_size: int = 3):
    """
    Dilata una maschera binaria senza smoothing.

    Args:
        mask: [B,H,W] binaria (0/1)
        radius: numero di iterazioni di dilatazione
        kernel_size: dimensione kernel per dilatazione (disco approssimato con ones)

    Returns:
        [B,H,W] maschera dilatata binaria
    """
    if mask.dim() != 3:
        raise ValueError("Input must be [B,H,W]")

    device = mask.device
    mask = mask.float().unsqueeze(1)  # [B,1,H,W]

    # Kernel pieno
    kernel = torch.ones(1, 1, kernel_size, kernel_size, device=device)
    padding = kernel_size // 2

    for _ in range(radius):
        dil = F.conv2d(mask, kernel, padding=padding)
        mask = (dil > 0).float()  # ogni pixel con almeno un vicino attivo diventa 1

    return mask.squeeze(1)
    

class DilateMaskTransform:
    def __init__(self, radius: int = 1, kernel_size: int = 3):
        self.radius = radius
        self.kernel_size = kernel_size

    def __call__(self, mask):
        """
        mask: Tensor [H,W] o [B,H,W]
        """
        return dilate_mask(mask, self.radius, self.kernel_size)


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


def insid3_preprocess_attentions(attentions):
    """INSID3 already emits per-class attentions as a list of levels, each shaped
    `b (hq wq) (n hs ws)` (no attention-head axis to average). Pass through unchanged."""
    return attentions


MODEL_EXPLAINER_REGISTRY = {
    DMTNetMultiClass.__name__: dmtnet_preprocess_attentions,
    DCAMAMultiClass.__name__: dcama_preprocess_attentions,
    INSID3MultiClass.__name__: insid3_preprocess_attentions,
    MatcherMultiClass.__name__: insid3_preprocess_attentions,  # pass-through, same layout
    GFSAMMultiClass.__name__: insid3_preprocess_attentions,    # pass-through, same layout
    SANSAMultiClass.__name__: insid3_preprocess_attentions,    # pass-through, same layout
    PANetMultiClass.__name__: insid3_preprocess_attentions,    # pass-through, same layout
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
    def __init__(self, model, aggregation_method="feature_ablation", explanation_size=None, use_softmax=True, masking=False, mask_blur_kernel_size=1, mask_blur_sigma=50, mask_dilation_radius=0, mask_dilation_kernel=None):
        self.model = model
        self.aggregation_method = aggregation_method
        self.use_softmax = use_softmax
        self.masking = masking
        self.mask_dilation_radius = mask_dilation_radius
        self.mask_dilation_kernel = mask_dilation_kernel
        if self.mask_dilation_radius > 0 and self.mask_dilation_kernel is not None:
            self.dilation = DilateMaskTransform(radius=self.mask_dilation_radius, kernel_size=self.mask_dilation_kernel)
        else:
            self.dilation = T.Lambda(lambda x: x)  # Identity transform if no dilation is applied
        
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
                # The support mask is only needed by the mask-based modes; the
                # "smooth" control never touches it.
                if self.masking and self.masking != "smooth":
                    mask = resize(
                        mask,
                        image_size,
                        interpolation=TvT.InterpolationMode.NEAREST,
                    )

            # Attribution post-processing modes (self.masking):
            #   False        -> raw unmasked attribution (no smoothing, no mask)
            #   "smooth"     -> blur the attribution map itself, no mask (ynLC control:
            #                   isolates the smoothing/interior-concentration effect)
            #   True         -> multiply by the smoothed support-mask magnitude in [0, 1]
            #   "sign"       -> multiply by the smoothed support-mask sign in [-1, 1]
            #   "reverse_sign" -> multiply by the reversed sign in [1, -1]
            if self.masking == "smooth":
                weighted_contrib = min_max_scale(self.blur(weighted_contrib))
            elif self.masking:
                curr_mask = self.blur(self.dilation(mask))
                if self.masking == "sign":
                    curr_mask = 2 * curr_mask - 1
                elif self.masking == "reverse_sign":
                    curr_mask = 1 - 2 * curr_mask
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
        
        
class ReverseSignedAffinityExplainer(AffinityExplainer):
    def __init__(self, model, **kwargs):
        super().__init__(model, **kwargs, masking="reverse_sign")


class SmoothedAffinityExplainer(AffinityExplainer):
    """Smoothed-unmasked control (reviewer ynLC): applies the same Gaussian blur used by
    the masked variants to the raw unmasked attribution map, without any support mask or
    sign. Tests whether the masked variant's advantage comes from smoothing/interior
    concentration rather than from the support-mask information."""

    def __init__(self, model, **kwargs):
        super().__init__(model, **kwargs, masking="smooth")


class EncoderAffinityExplainer(AffinityExplainer):
    """
    Extends AffinityExplainer by tracing input-pixel attributions through the encoder.

    For each feature vector at position (h, w) in encoder layer l, computes which input
    pixels contributed most by differentiating the mean of that feature vector w.r.t. the
    input image:

        scalar(h, w) = mean_over_batch_and_channels(F_l[:, :, h, w])
        attr_map(h, w) = |d scalar(h,w) / d input|.sum(over_input_channels)

    This yields one attribution map per feature vector. These maps are then weighted by
    the affinity explanation (which tells us how much each feature position matters for
    the prediction) and summed to produce a single per-class encoder attribution map.
    The final output optionally multiplies the encoder and affinity maps.
    """

    def __init__(self, model, combine_with_affinity=True, **kwargs):
        super().__init__(model, **kwargs)
        self.combine_with_affinity = combine_with_affinity
        self.layer_attr_volumes = None  # [H_l, W_l, B, H_in, W_in] per layer after explain()

    def _extract_query_feats(self, query_img):
        """Run the encoder on the query image and return per-layer feature maps."""
        model = self.model
        if isinstance(model, DCAMAMultiClass):
            return model.extract_feats(query_img)
        elif isinstance(model, DMTNetMultiClass):
            return model.extract_feats(
                query_img,
                model.backbone,
                model.feat_ids,
                getattr(model, "bottleneck_ids", None),
                getattr(model, "lids", None),
            )
        raise ValueError(
            f"EncoderAffinityExplainer does not support {model.__class__.__name__}. "
            f"Supported models: {list(MODEL_EXPLAINER_REGISTRY.keys())}"
        )

    def _compute_encoder_attributions(self, encoder_feats, query_img):
        """
        For each feature vector at position (h, w) in each encoder layer, compute the
        gradient of mean(feat[:, :, h, w]) w.r.t. the query image. The gradient's
        absolute value summed over input channels gives the per-pixel attribution for
        that feature vector.

        Returns: List[Tensor[H_l, W_l, B, H_in, W_in]] — one volume per encoder layer.
        """
        B, _, H_in, W_in = query_img.shape
        n_layers = len(encoder_feats)

        h_max, w_max = encoder_feats[0].shape[2:]
        
        encoder_attribution = torch.zeros(h_max, w_max, B, H_in, W_in, device=query_img.device)

        for layer_idx, feat_map in enumerate(encoder_feats):
            _, _, H_l, W_l = feat_map.shape
            feat_mean = feat_map.mean(dim=1)  # [B, H_l, W_l] — mean over channels

            pos_attrs = []
            n_positions = H_l * W_l
            for pos_idx in range(n_positions):
                h, w = divmod(pos_idx, W_l)
                # Scalar: mean over the batch of the feature mean at position (h, w).
                scalar = feat_mean[:, h, w].mean()
                is_last = (layer_idx == n_layers - 1) and (pos_idx == n_positions - 1)
                grad = torch.autograd.grad(
                    scalar, query_img, retain_graph=not is_last, create_graph=False
                )[0]  # [B, C_in, H_in, W_in]
                pos_attrs.append(grad.abs().sum(dim=1))  # [B, H_in, W_in]

            layer_attribution = torch.stack(pos_attrs).reshape(H_l, W_l, B, H_in, W_in)
            layer_attribution = rearrange(layer_attribution, "h w b h_in w_in -> h_in w_in b h w")
            if layer_attribution.shape[0] != h_max or layer_attribution.shape[1] != w_max:
                # resize to max spatial dimensions across layers for easier later weighting
                layer_attribution = F.interpolate(
                    layer_attribution, size=(1, h_max, w_max),
                    mode="bilinear", align_corners=False,
                )
            encoder_attribution += layer_attribution  # Sum contributions from all layers

        return encoder_attribution

    def explain(
        self,
        input_dict,
        result=None,
        explanation_mask="logits",
        selected_classes=None,
        gt=None,
    ):
        # Step 1: Compute affinity explanations via parent class (no gradients needed).
        affinity_explanations = super().explain(
            input_dict,
            result=result,
            explanation_mask=explanation_mask,
            selected_classes=selected_classes,
            gt=gt,
        )

        masks = input_dict[BatchKeys.PROMPT_MASKS]
        num_classes = masks.shape[2] - 1
        if selected_classes is None:
            selected_classes = list(range(num_classes))

        image_size = input_dict[BatchKeys.IMAGES].shape[-2:]
        B = input_dict[BatchKeys.IMAGES].shape[0]

        # Step 2: Run the encoder with gradient tracking so we can backpropagate
        # from each feature vector's mean back to individual input pixels.
        query_img = input_dict[BatchKeys.IMAGES][:, 0].detach().requires_grad_(True)
        encoder_feats = self._extract_query_feats(query_img)

        # Step 3: For every feature vector (h, w) in every encoder layer, compute
        # |d mean(feat[:,:,h,w]) / d input|.sum(channels). This gives one attribution
        # map per feature position, stored as [H_l, W_l, B, H_in, W_in] per layer.
        self.layer_attr_volumes = self._compute_encoder_attributions(encoder_feats, query_img)

        # Step 4: For each class, weight every feature-vector's attribution map by its
        # affinity importance and sum — collapsing [H_l, W_l] into a single map.
        encoder_maps = []
        for class_idx in range(len(selected_classes)):
            affinity_expl = affinity_explanations[class_idx].detach()
            # Parent keeps a per-shot dimension [B, N, H, W]; average over shots.
            if affinity_expl.dim() == 4:
                affinity_expl = affinity_expl.mean(dim=1)  # [B, H_q, W_q]

            encoder_map = torch.zeros(B, *image_size, device=query_img.device)

            for feat_map, attr_vol in zip(encoder_feats, self.layer_attr_volumes):
                H_l, W_l = feat_map.shape[-2:]
                # Resize affinity to match this layer's spatial resolution.
                resized_affinity = F.interpolate(
                    affinity_expl.float().unsqueeze(1),
                    size=(H_l, W_l),
                    mode="bilinear",
                    align_corners=False,
                ).squeeze(1)  # [B, H_l, W_l]

                # attr_vol: [H_l, W_l, B, H_in, W_in]
                # Broadcast affinity [B, H_l, W_l] → [H_l, W_l, B, 1, 1] and weight.
                weights = resized_affinity.permute(1, 2, 0).unsqueeze(-1).unsqueeze(-1)
                encoder_map += (attr_vol * weights).sum(dim=(0, 1))  # [B, H_in, W_in]

            if encoder_map.shape[-2:] != image_size:
                encoder_map = F.interpolate(
                    encoder_map.unsqueeze(1), size=image_size,
                    mode="bilinear", align_corners=False,
                ).squeeze(1)

            encoder_maps.append(min_max_scale(encoder_map))

        if not self.combine_with_affinity:
            return encoder_maps

        # Step 5: Combine affinity and encoder attributions element-wise.
        # Affinity captures which feature positions matter; encoder gradients capture
        # which input pixels created those features.
        combined = []
        for aff, enc in zip(affinity_explanations, encoder_maps):
            if aff.dim() == 4:
                aff = aff.mean(dim=1)
            combined.append(min_max_scale(aff * enc))
        return combined