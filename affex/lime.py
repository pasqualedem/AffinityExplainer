import torch
import torch.nn.functional as F

from captum.attr import Lime
from captum._utils.typing import TensorOrTupleOfTensorsGeneric
from captum._utils.models.linear_model import (
    SkLearnLasso,
    SkLearnRidge,
    SGDLinearRegression,
)
from skimage.segmentation import slic


def flatten_tensor_or_tuple(inp: TensorOrTupleOfTensorsGeneric) -> torch.Tensor:
    if isinstance(inp, torch.Tensor):
        return inp.flatten()
    return torch.cat([single_inp.flatten() for single_inp in inp])


def fss_similarity_func(
    original_input, perturbed_input, perturbed_interpretable_input=None, **kwargs
):
    """
    Custom similarity function for LIME in Few-Shot Segmentation.
    Computes similarity based on both images and masks.

    Expected original_input: tuple(images, masks)
        - images: [B, 1+M, C, H, W]   (query + supports)
        - masks:  [B, M, Cmask, H, W]
    Returns:
        Similarity score [1, N] where N is number of perturbed samples
    """
    orig_masks, orig_images = original_input
    pert_masks, pert_images = perturbed_input

    support_orig_images = orig_images[:, 1:]  # [B, M, C, H, W]
    support_pert_images = pert_images[:, 1:]  # [B, M, C, H, W]

    alpha = kwargs.get("alpha", 0.2)  # weight for images vs masks

    # Flatten images and masks
    orig_images_flat = flatten_tensor_or_tuple(support_orig_images)
    pert_images_flat = flatten_tensor_or_tuple(support_pert_images)

    # Compute Cosine distances for images
    image_distances = F.cosine_similarity(
        orig_images_flat.unsqueeze(0), pert_images_flat, dim=1
    )  # [N]
    image_distances = 1 - image_distances  # convert similarity to distance
    # Mask distances is how many ones are removed
    mask_distances = (orig_masks - pert_masks).sum() / orig_masks.sum()

    total_distances = alpha * image_distances + (1 - alpha) * mask_distances

    # Convert distances to similarities using exponential kernel
    kernel_width = kwargs.get("kernel_width", 1.0)  # default kernel width
    similarities = torch.exp(-(total_distances**2) / (kernel_width**2))

    return similarities.squeeze(0)  # [N]


def compute_feature_mask(
    image: torch.Tensor, num_segments: int = 100, compactness: float = 10.0
):
    """
    Compute interpretable feature mask (superpixels) for LIME on the query image.

    Args:
        image: Tensor [C, H, W], normalized to [0, 1] or [0, 255]
        num_segments: approximate number of superpixels
        compactness: tradeoff between color proximity and spatial proximity

    Returns:
        feature_mask: torch.LongTensor [H, W] with integers in [0, K-1]
    """
    img_np = image.detach().cpu().permute(1, 2, 0).numpy()
    segments = slic(
        img_np, n_segments=num_segments, compactness=compactness, start_label=0
    )
    return torch.from_numpy(segments).long().to(image.device)


def fss_perturb_func(original_input, **kwargs):
    """
    Custom perturbation function for LIME in Few-Shot Segmentation.
    Produces interpretable, structured perturbations rather than random noise.

    Expected original_input: tuple(images, masks)
        - images: [B, 1+M, C, H, W]   (query + supports)
        - masks:  [B, M, Cmask, H, W]
    Returns:
        Binary vector [1, num_interp_features]
    """
    masks, images = original_input

    num_interp_features = kwargs.get("num_interp_features", None)
    feature_mask = kwargs.get("feature_mask", None)

    # --- If feature_mask is provided, perturb based on interpretable regions ---
    if feature_mask is not None:
        feature_mask = feature_mask[0]  # Get support feature mask [H, W]

        # Each unique integer in feature_mask corresponds to an interpretable feature
        unique_features = torch.unique(feature_mask)
        unique_features = unique_features[unique_features >= 0]  # exclude -1 if used
        K = len(unique_features)  # exclude special value for query

        # Binary activation vector: 1 = feature on, 0 = off
        active = torch.randint(0, 2, (1, K)).float().to(images.device)
        # # Add the query support features as always active
        active = torch.cat([active, torch.ones(1, 1).to(active.device)], dim=1)

        return active

    # --- Fallback: simple random binary vector if no feature_mask ---
    if num_interp_features is None:
        # if unspecified, fall back to a small fixed number
        num_interp_features = 8
    return torch.randint(0, 2, (1, num_interp_features), device=images.device).float()


class FSSLime(Lime):
    def __init__(
        self,
        forward_func,
        interpretable_model=None,
        similarity_func=None,
        perturb_func=None,
        model_type: str = None,
        use_baselines: bool = False,
        kernel_width: float = 1.0,
        image_weight: float = 0.2,
    ):
        """
        Few-Shot Semantic Segmentation (FSS) variant of LIME.

        Automatically handles baselines for both query/support images and support masks.

        Args:
            forward_func: model forward function
            interpretable_model: optional custom surrogate model
            similarity_func: optional similarity kernel
            perturb_func: optional sampling function
            model_type: "ridge", "lasso", or "linear" (SGD)
            use_baselines: if True, generate and use baselines for images and masks;
                           if False, do not use baselines (standard LIME behavior)
            kernel_width: float = 1.0, kernel width for similarity function
            image_weight: float = 0.2, weight for image similarity vs mask similarity
        """

        if interpretable_model is None:
            if model_type == "ridge":
                interpretable_model = SkLearnRidge(alpha=1.0)
            else:
                interpretable_model = SkLearnLasso(alpha=0.01)

        if similarity_func is None:
            similarity_func = (
                lambda orig, pert, interpretable_inp, **kwargs: fss_similarity_func(
                    orig,
                    pert,
                    interpretable_inp,
                    alpha=image_weight,
                    kernel_width=kernel_width,
                    **kwargs,
                )
            )

        if perturb_func is None:
            perturb_func = fss_perturb_func

        super().__init__(
            forward_func=forward_func,
            interpretable_model=interpretable_model,
            similarity_func=similarity_func,
            perturb_func=perturb_func,
        )
        self.use_baselines = use_baselines

    # ---- Baseline generation ----
    def generate_baselines(self, images, masks):
        """
        Create baselines for both images and masks.
        - For images: blurred query + mean of supports.
        - For masks: deterministic zero tensor.
        """
        B, M_plus_1, C, H, W = images.shape
        _, M, _, _, _ = masks.shape

        # Query = first image, supports = rest
        query = images[:, 0]
        supports = images[:, 1:]

        # Blurred query as baseline
        blurred = torch.nn.functional.avg_pool2d(
            query, kernel_size=3, stride=1, padding=1
        )

        # Mean of support images as baseline
        mean_support = supports.mean(dim=1, keepdim=True)

        baseline_images = (
            0.5 * blurred.unsqueeze(1) + 0.5 * mean_support
        )  # soft interpolation
        baseline_masks = torch.zeros_like(masks)  # zeroed support masks

        return baseline_images, baseline_masks

    # ---- Override LIME attribution ----
    def attribute(self, *args, **kwargs):
        """
        Builds baselines from FSS batch dict, reorders according to `ordering`,
        and calls standard LIME attribution.
        """
        batch_dict = args[0]
        ordering = kwargs["additional_forward_args"][-1]

        images = batch_dict[ordering["images"]]  # [B, 1+M, C, H, W]
        masks = batch_dict[ordering["prompt_masks"]]  # [B, M, H, W]

        baselines_images, baselines_masks = self.generate_baselines(images, masks)

        if self.use_baselines:
            # Remove redundant singleton dimension if present
            baselines_images = baselines_images.squeeze(1)
            baselines_masks = baselines_masks.squeeze(1)

            # Pack tuple in correct order
            baselines_tuple = [None] * 2
            baselines_tuple[ordering["images"]] = baselines_images
            baselines_tuple[ordering["prompt_masks"]] = baselines_masks
            baselines_tuple = tuple(baselines_tuple)

            kwargs["baselines"] = baselines_tuple

        self.interpretable_model = self.interpretable_model.to(images.device)
        support_images = images[:, 1:]
        support_feature_mask = (
            torch.stack(
                [
                    torch.stack(
                        [
                            compute_feature_mask(support_images[i, j], num_segments=80)
                            for j in range(support_images.shape[1])
                        ],
                        dim=0,
                    )
                    for i in range(support_images.shape[0])
                ],
                dim=0,
            )
            .unsqueeze(2)
            .repeat(1, 1, 3, 1, 1)
        )  # [B, M, rgb, H, W]
        # query_feature_mask has special value that LIME ignores it (max value + 1)
        query_feature_mask = torch.ones_like(support_feature_mask[:, 0]) * (
            support_feature_mask.max() + 1
        )
        feature_mask = torch.cat(
            [query_feature_mask.unsqueeze(1), support_feature_mask], dim=1
        )

        # Support_mask feature_mask is the same as support_feature_mask
        support_masks_feature_mask = (
            support_feature_mask[:, :, 0]
            .unsqueeze(2)
            .repeat(1, 1, masks.shape[2], 1, 1)
        )
        feature_mask = (support_masks_feature_mask, feature_mask)

        return super().attribute(*args, feature_mask=feature_mask, **kwargs)
