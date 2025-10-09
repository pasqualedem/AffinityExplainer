import torch
from captum.attr import Lime
from captum._utils.typing import TensorOrTupleOfTensorsGeneric
from captum.attr._core.lime import get_exp_kernel_similarity_function, default_perturb_func
from captum._utils.models.linear_model import SkLearnLasso

class FSSLime(Lime):
    def __init__(
        self,
        forward_func,
        interpretable_model=None,
        similarity_func=None,
        perturb_func=None,
        use_baselines=True,
        **kwargs,
    ):
        """
        Few-Shot Semantic Segmentation (FSS) variant of LIME.

        Automatically handles baselines for both query/support images and support masks.
        """

        if interpretable_model is None:
            interpretable_model = SkLearnLasso(alpha=0.01)
        if similarity_func is None:
            similarity_func = get_exp_kernel_similarity_function()
        if perturb_func is None:
            perturb_func = default_perturb_func
            
        self.use_baselines = use_baselines

        super().__init__(
            forward_func=forward_func,
            interpretable_model=interpretable_model,
            similarity_func=similarity_func,
            perturb_func=perturb_func,
        )

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
        query = images[:, 0:1]
        supports = images[:, 1:]

        # Blurred query as baseline
        blurred = torch.nn.functional.avg_pool2d(query, kernel_size=3, stride=1, padding=1)

        # Mean of support images as baseline
        mean_support = supports.mean(dim=1, keepdim=True)

        baseline_images = 0.5 * blurred + 0.5 * mean_support  # soft interpolation
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

        images = batch_dict[ordering["images"]]        # [B, 1+M, C, H, W]
        masks = batch_dict[ordering["prompt_masks"]]   # [B, M, H, W]

        baselines_images, baselines_masks = self.generate_baselines(images, masks)

        if self.use_baselines:
            # Remove redundant singleton dimension if present
            baselines_images = baselines_images.squeeze(1)
            baselines_masks = baselines_masks.squeeze(1)

            # Pack tuple in correct order
            baselines_tuple = [None] * len(ordering)
            baselines_tuple[ordering["images"]] = baselines_images
            baselines_tuple[ordering["prompt_masks"]] = baselines_masks
            baselines_tuple = tuple(baselines_tuple)

            kwargs["baselines"] = baselines_tuple
        
        return super().attribute(*args, **kwargs)
