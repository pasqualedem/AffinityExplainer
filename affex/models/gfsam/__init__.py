"""AffEx wrapper around GF-SAM (Graph-based Few-shot Segment Anything, NeurIPS 2024).

GF-SAM (DINOv2 + SAM, training-free) computes an explicit dense support<->query
correspondence
    similarity = cos(support_patches, query_patches)   (GFSAM.generate_pixelwise_comparison)
which is exactly the affinity AffEx reads. The final mask is produced by SAM point-prompting
followed by graph-based Point-Mask Clustering — a heavy Seg stage. This lets us test AffEx on a
second modern, VFM matching pipeline that the HzER reviewer explicitly cited.

The upstream code is vendored under `affex/models/gfsam/` (MIT, see LICENSE.gfsam);
nothing outside this repository has to be cloned.
"""

import os
from argparse import Namespace

import torch
import torch.nn as nn

from ...data.utils import BatchKeys
from ...utils.utils import ResultDict
from ...utils.segmentation import unnormalize
from ...assets import ensure
from .gfsam import build_model


def _weights_dir():
    """Where the DINOv2 and SAM checkpoints live.

    Default is `checkpoints/` in the repo, which is what `scripts/setup.sh` fills;
    AFFEX_CHECKPOINTS overrides it for a shared read-only copy on a cluster.
    """
    return os.environ.get("AFFEX_CHECKPOINTS", "checkpoints")


def _default_gfsam_args(sam_size="vit_b", device="cuda"):
    return Namespace(
        device=device, dinov2_size="vit_large", sam_size=sam_size,
        dinov2_weights=ensure("dinov2-vitl14"),
        sam_weights=ensure("sam-vit-b" if sam_size == "vit_b" else "sam-vit-h"),
    )


class GFSAMMultiClass(nn.Module):
    def __init__(self, sam_size="vit_b", device="cuda", image_size=1024, **overrides):
        super().__init__()
        self.args = _default_gfsam_args(sam_size=sam_size, device=device)
        for k, v in overrides.items():
            setattr(self.args, k, v)
        self.gfsam = build_model(self.args)
        self.image_size = image_size
        self._device = device

    def _preprocess_masks(self, masks):
        # drop the background channel: [B, M, C+1, H, W] -> [B, M, C, H, W]
        return masks[:, :, 1:, ...]

    def _run_gfsam(self, sup_imgs01, sup_masks, query01):
        """Run GF-SAM on one class; return (pred_mask[H,W], sim[support_patches, query_patches])."""
        g = self.gfsam
        g.set_reference(sup_imgs01.unsqueeze(0), sup_masks.unsqueeze(0))  # (1,ns,3,H,W),(1,ns,H,W)
        g.set_target(query01)
        pred_mask, _ = g.predict()  # [1, H, W]
        # dense support<->query correspondence, list of ns tensors [1, supp_hw, query_hw]
        coms = getattr(g, "_pixelwise_coms_positive", None)
        sim = torch.cat([c[0] for c in coms], dim=0) if coms else None  # [n*supp_hw, query_hw]
        g.clear()
        pred = pred_mask.float().reshape(pred_mask.shape[-2], pred_mask.shape[-1])
        return pred, (sim.detach() if sim is not None else None)

    @torch.no_grad()
    def forward(self, x, postprocess=True):
        images = x[BatchKeys.IMAGES]  # [B, 1+M, 3, H, W], ImageNet-normalized
        masks = self._preprocess_masks(x[BatchKeys.PROMPT_MASKS])  # [B, M, C, H, W]
        B, _, _, H, W = images.shape
        assert B == 1, "GF-SAM wrapper tested with batch size 1"

        query01 = unnormalize(images[:, 0]).clamp(0, 1)  # [1,3,H,W] in [0,1]
        supports = images[:, 1:]                          # [1, M, 3, H, W]
        num_classes = masks.shape[2]

        decisions, attentions = [], []
        for c in range(num_classes):
            class_examples = x[BatchKeys.FLAG_EXAMPLES][:, :, c + 1]  # [B, M]
            n_shots = int(class_examples.sum().item())
            if n_shots == 0:
                decisions.append(torch.zeros((H, W), device=images.device))
                attentions.append(None)
                continue
            sup_imgs01 = unnormalize(supports[class_examples]).clamp(0, 1)  # [ns,3,H,W]
            sup_masks = masks[:, :, c][class_examples].float()             # [ns,H,W]
            try:
                pred, sim = self._run_gfsam(sup_imgs01, sup_masks, query01)
            except Exception:
                pred = torch.zeros((H, W), device=images.device)
                sim = None
            if sim is None:
                # GF-SAM failed on this episode: emit a zero correspondence of the right shape
                # so the explainer stays valid (episode contributes an empty attribution).
                fs = self.gfsam.encoder_feat_size
                sim = torch.zeros((n_shots * fs * fs, fs * fs), device=images.device)
            decisions.append(pred)
            # sim: (n*supp_hw, query_hw) -> AffEx layout (1, query_hw, n*supp_hw); single "level"
            attentions.append([sim.t().unsqueeze(0).contiguous()])

        decisions = torch.stack(decisions, dim=0).unsqueeze(0)  # [1, C, H, W]
        fg = decisions.clamp(0, 1)
        if fg.size(1) == 1:
            logits = torch.cat([1 - fg, fg], dim=1)  # [1, 2, H, W]
        else:
            bg = (1 - fg.max(dim=1, keepdim=True)[0])
            logits = torch.cat([bg, fg], dim=1)

        return {
            ResultDict.LOGITS: logits,
            ResultDict.ATTENTIONS: attentions,
        }

    def feature_ablation(self, result, chosen_class, mask, n_shots=None, explanation_size=None):
        """Single matching level -> uniform weight (equivalent to mean aggregation)."""
        n_levels = len(result[ResultDict.ATTENTIONS][chosen_class])
        return torch.full((n_levels,), 1.0 / n_levels, device=result[ResultDict.LOGITS].device)


def build_gfsam(sam_size="vit_b", device="cuda", image_size=1024, **kwargs):
    return GFSAMMultiClass(sam_size=sam_size, device=device, image_size=image_size, **kwargs)
