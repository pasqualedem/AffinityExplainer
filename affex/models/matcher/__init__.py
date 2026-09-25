"""AffEx wrapper around Matcher (training-free DINOv2 + SAM one-shot segmentation).

Matcher exposes an explicit dense support<->query correspondence
    S = ref_feats @ tar_feat.t()   (support_patches x query_patches)
which is exactly the affinity AffEx reads, while the final mask is produced by a heavy
SAM decoder. This lets us test whether AffEx's matching-stage attributions stay useful
when a promptable decoder produces the output.

The upstream code is vendored under `affex/models/matcher/` (BSD-2-Clause, see
LICENSE.matcher); nothing outside this repository has to be cloned.
"""

import os
from argparse import Namespace

import torch
import torch.nn as nn

from ...data.utils import BatchKeys
from ...utils.utils import ResultDict
from ...utils.segmentation import unnormalize
from ...assets import ensure
from .matcher import build_matcher_oss


def _default_matcher_args(sam_size="vit_b", device="cuda"):
    """Upstream's default configuration, with the weights resolved through `assets`."""
    return Namespace(
        device=device, dinov2_size="vit_large", sam_size=sam_size,
        dinov2_weights=ensure("dinov2-vitl14"),
        sam_weights=ensure("sam-vit-b" if sam_size == "vit_b" else "sam-vit-h"),
        points_per_side=64, pred_iou_thresh=0.88, stability_score_thresh=0.95,
        sel_stability_score_thresh=0.0, iou_filter=0.0, box_nms_thresh=1.0,
        output_layer=3, dense_multimask_output=0, use_dense_mask=0, multimask_output=0,
        num_centers=8, use_box=False, use_points_or_centers=True,
        sample_range=(1, 6), max_sample_iterations=64, alpha=1.0, beta=0.0, exp=0.0,
        emd_filter=0.0, purity_filter=0.0, coverage_filter=0.0, use_score_filter=False,
        deep_score_norm_filter=0.1, deep_score_filter=0.33, topk_scores_threshold=0.0,
        num_merging_mask=9,
    )


class MatcherMultiClass(nn.Module):
    def __init__(self, sam_size="vit_b", device="cuda", image_size=518,
                 attribution_mode="dense", **overrides):
        super().__init__()
        self.args = _default_matcher_args(sam_size=sam_size, device=device)
        for k, v in overrides.items():
            setattr(self.args, k, v)
        self.matcher = build_matcher_oss(self.args)
        self.image_size = image_size
        self._device = device
        # "dense": AffEx reads the full matching matrix S (Match stage only).
        # "prompt": decoder-aware AffEx — S restricted to the support<->query pairs whose
        #   Hungarian match survives into SAM point prompts (the Match->Seg interface), i.e.
        #   AffEx composed with the SAM decoder's prompt selection.
        self.attribution_mode = attribution_mode

    def _preprocess_masks(self, masks):
        # drop the background channel: [B, M, C+1, H, W] -> [B, M, C, H, W]
        return masks[:, :, 1:, ...]

    def _run_matcher(self, sup_imgs01, sup_masks, query01):
        """Run Matcher on one class; return (pred_mask[H,W], S[support_patches, query_patches])."""
        m = self.matcher
        m.set_reference(sup_imgs01.unsqueeze(0), sup_masks.unsqueeze(0))  # (1,ns,3,H,W),(1,ns,H,W)
        m.set_target(query01)
        ref_feats, tar_feat = m.extract_img_feats()
        all_points, box, S, C, _ = m.patch_level_matching(ref_feats=ref_feats, tar_feat=tar_feat)
        points = m.clustering(all_points) if not m.use_points_or_centers else all_points
        m.set_rps()
        pred = m.mask_generation(m.tar_img_np, points, box, all_points, m.ref_masks_pool, C)

        attr = S.detach()
        if self.attribution_mode == "prompt":
            # Decoder-aware: keep only the (support, query) matches that survived into SAM
            # prompts; zero everything else. AffEx then attributes exactly the support pixels
            # whose matching determines the prompts the SAM decoder actually consumes.
            sparse = torch.zeros_like(attr)
            sup_idx = getattr(m, "_matched_sup_idx", None)
            qry_idx = getattr(m, "_matched_qry_idx", None)
            if sup_idx is not None and qry_idx is not None and len(sup_idx) > 0:
                sup_idx = sup_idx.to(attr.device).long()
                qry_idx = qry_idx.to(attr.device).long()
                sparse[sup_idx, qry_idx] = attr[sup_idx, qry_idx]
            attr = sparse
        return pred.float().reshape(pred.shape[-2], pred.shape[-1]), attr

    @torch.no_grad()
    def forward(self, x, postprocess=True):
        images = x[BatchKeys.IMAGES]  # [B, 1+M, 3, H, W], ImageNet-normalized
        masks = self._preprocess_masks(x[BatchKeys.PROMPT_MASKS])  # [B, M, C, H, W]
        B, _, _, H, W = images.shape
        assert B == 1, "Matcher wrapper tested with batch size 1"

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
                pred, S = self._run_matcher(sup_imgs01, sup_masks, query01)
            except Exception:
                pred = torch.zeros((H, W), device=images.device)
                S = None
            if S is None:
                # Matcher failed on this episode: emit a zero correspondence of the right shape
                # so the explainer stays valid (episode contributes an empty attribution).
                fs = self.matcher.encoder_feat_size
                S = torch.zeros((n_shots * fs * fs, fs * fs), device=images.device)
            decisions.append(pred)
            # S: (n*hs*ws, hq*wq) -> AffEx layout (1, hq*wq, n*hs*ws); single "level"
            attentions.append([S.t().unsqueeze(0).contiguous()])

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


def build_matcher(sam_size="vit_b", device="cuda", image_size=518, **kwargs):
    return MatcherMultiClass(sam_size=sam_size, device=device, image_size=image_size, **kwargs)
