import torch
import torch.nn.functional as F
import random
from captum.attr import GradientShap
from affex.data.utils import BatchKeys


class FSSGradientShap(GradientShap):
    """
    Gradient SHAP for Few-Shot Semantic Segmentation.

    Expects input as a tuple: (images, masks)
      - images: [B, 1+M, Cin, H, W] (query + M supports)
      - masks:  [B, M, Cmask, H, W] (support masks, first channel = background)
    """

    def __init__(
        self,
        model,
        num_baselines: int = 20,
        mask_dilation: int = 3,
        baseline_mix_weights=(0.5, 0.25, 0.15, 0.1, 0.0),
        mask_zero_prob: float = 0.5,
        random_seed: int = 42,
        **kwargs,
    ):
        super().__init__(model)
        self.num_baselines = num_baselines
        self.mask_dilation = mask_dilation
        self.baseline_mix_weights = baseline_mix_weights
        self.mask_zero_prob = mask_zero_prob
        self.kinds = ['both_masked', 'support_masked', 'query_masked', 'support_swap', 'zeroed']
        random.seed(random_seed)

    # ---------------- Helper methods ---------------- #

    def _dilate_mask(self, mask: torch.Tensor, pad=None):
        if pad is None:
            pad = self.mask_dilation
        m = mask.unsqueeze(1).float()  # [B,1,H,W]
        m = F.max_pool2d(m, kernel_size=2 * pad + 1, stride=1, padding=pad)
        return (m.squeeze(1) > 0.5)

    def _sample_background_fill(self, img: torch.Tensor, mask: torch.Tensor):
        Cimg, H, W = img.shape
        non_masked = (~mask).flatten()
        if non_masked.sum() == 0:
            return torch.zeros_like(img)
        bg_pixels = img.view(Cimg, -1)[:, non_masked]
        mean_bg = bg_pixels.mean(dim=1, keepdim=True)
        fill = mean_bg.view(Cimg, 1, 1).expand(Cimg, H, W)
        result = img.clone()
        result[:, mask] = fill[:, mask]
        return result

    # ---------------- Core baseline generation ---------------- #

    def _generate_single_baseline(self, images, masks, kind):
        """
        Generate a single baseline for one batch.
        Returns tuple: (images_baseline, masks_baseline)
        """
        B, total, Cin, H, W = images.shape
        _, M, Cmask, _, _ = masks.shape
        baseline_images = images.clone()
        baseline_masks = masks.clone()

        for b in range(B):
            query = images[b, 0]        # [Cin,H,W]
            supports = images[b, 1:]    # [M,Cin,H,W]
            support_masks = masks[b]    # [M,Cmask,H,W]

            # Combine all non-background channels into binary foreground
            support_masks_binary = (support_masks[:, 1:, :, :].sum(dim=1) > 0)  # [M,H,W]

            # ---------------- Images baselines ---------------- #
            if kind in ['both_masked', 'support_masked']:
                for m in range(M):
                    smask = self._dilate_mask(support_masks_binary[m].unsqueeze(0))[0]
                    filled = self._sample_background_fill(supports[m], smask)
                    baseline_images[b, m + 1] = filled

            if kind in ['query_masked', 'both_masked']:
                mean_val = query.mean(dim=(1, 2), keepdim=True)
                baseline_images[b, 0] = mean_val.expand_as(query)

            if kind == 'support_swap' and M > 1:
                baseline_images[b, 1], baseline_images[b, 2] = (
                    baseline_images[b, 2].clone(),
                    baseline_images[b, 1].clone(),
                )

            if kind == 'zeroed':
                for m in range(M):
                    smask = self._dilate_mask(support_masks_binary[m].unsqueeze(0))[0]
                    baseline_images[b, m + 1][:, smask] = 0.0
                baseline_images[b, 0] = torch.zeros_like(query)

            # ---------------- Masks baselines ---------------- #
            # Zero all foreground channels for supports
            for m in range(M):
                if random.random() < self.mask_zero_prob:
                    baseline_masks[b, m, 1:, :, :] = 0.0  # keep background channel as is

        return baseline_images, baseline_masks

    def generate_baselines(self, images, masks):
        """
        Generate baseline samples for Gradient SHAP.
        Returns tuple of tensors:
          - images_baselines: [num_baselines, B, 1+M, Cin, H, W]
          - masks_baselines:  [num_baselines, B, M, Cmask, H, W]
        """
        baselines_images = []
        baselines_masks = []

        for _ in range(self.num_baselines):
            kind = random.choices(self.kinds, weights=self.baseline_mix_weights, k=1)[0]
            img_base, mask_base = self._generate_single_baseline(images, masks, kind)
            baselines_images.append(img_base)
            baselines_masks.append(mask_base)

        baselines_images = torch.stack(baselines_images, dim=0)
        baselines_masks = torch.stack(baselines_masks, dim=0)

        return baselines_images, baselines_masks

    # ---------------- Override attribute ---------------- #

    def attribute(self, *args, **kwargs):
        """
        Override attribute to dynamically build baselines using batch dictionary.
        """
        batch_dict = args[0]
        ordering = kwargs["additional_forward_args"][-1]

        images = batch_dict[ordering[BatchKeys.IMAGES]]       # [B, 1+M, Cin, H, W]
        masks = batch_dict[ordering[BatchKeys.PROMPT_MASKS]]  # [B, M, Cmask, H, W]

        baselines_images, baselines_masks = self.generate_baselines(images, masks)
        
        # Remove batch dimension for captum baselines
        baselines_images = baselines_images.squeeze(1)
        baselines_masks = baselines_masks.squeeze(1)

        # Pack baselines as tuple in the correct order
        baselines_tuple = [None] * 2
        baselines_tuple[ordering[BatchKeys.IMAGES]] = baselines_images
        baselines_tuple[ordering[BatchKeys.PROMPT_MASKS]] = baselines_masks
        baselines_tuple = tuple(baselines_tuple)

        kwargs["baselines"] = baselines_tuple
        return super().attribute(*args, **kwargs, n_samples=1)
