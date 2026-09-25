"""AffEx wrapper around SANSA (Unleashing the Hidden Semantics in SAM2 for FSS, NeurIPS'25).

SANSA turns SAM2 into a few-shot segmenter: support frames (with mask prompts) populate SAM2's
memory bank, then the query frame is decoded via SAM2 memory attention. SAM2's per-frame vision
features are spatially dense, so we read the dense support<->query correspondence as the cosine
similarity of the SAM2 features (the same "matching" AffEx reads for Matcher/GF-SAM). The final mask
comes from SAM2's memory attention + mask decoder — a heavy Seg stage. This tests AffEx on the
reviewer's SAM2 / Memory-Encoder cited model.

The upstream code is vendored under `affex/models/sansa/` (MIT, see LICENSE.sansa);
nothing outside this repository has to be cloned.
"""

import os
import sys
from typing import Any, Dict, List

import torch
import torch.nn as nn
import torch.nn.functional as F

from ...data.utils import BatchKeys
from ...assets import ensure
from .sansa import build_sansa
from ...utils.utils import ResultDict


class SANSAMultiClass(nn.Module):
    def __init__(self, sam2_version="large", device="cuda", image_size=1024,
                 adapter_ckpt=None, channel_factor=0.8, differentiable=False, **overrides):
        super().__init__()
        # differentiable=True keeps the autograd graph through SAM2 (memory attention + mask
        # decoder are differentiable end-to-end), enabling gradient-based explainers; the
        # default no_grad path is kept for the (much lighter) AffEx/perturbation runs.
        self.differentiable = differentiable

        # channel_factor 0.8 matches the released "SANSA Universal" adapter checkpoint.
        self.sansa = build_sansa(sam2_version=sam2_version, channel_factor=channel_factor,
                                 device=device)
        ckpt = adapter_ckpt or ensure("sansa-adapter")
        if os.path.exists(ckpt):
            sd = torch.load(ckpt, map_location="cpu")
            sd = sd.get("model", sd)
            self.sansa.load_state_dict(sd, strict=False)
        self.sansa.eval()
        self.image_size = image_size
        self._device = device

    def _preprocess_masks(self, masks):
        # drop the background channel: [B, M, C+1, H, W] -> [B, M, C, H, W]
        return masks[:, :, 1:, ...]

    def _run_sansa(self, sup_imgs, sup_masks, query_img):
        """sup_imgs [ns,3,H,W] (ImageNet-norm), sup_masks [ns,H,W], query_img [1,3,H,W].
        Returns (pred[H,W] logits, sim[ns*hw, hw])."""
        from .promptable_utils import build_prompt_dict
        ns = sup_imgs.shape[0]
        imgs = torch.cat([sup_imgs, query_img], dim=0).unsqueeze(0)  # [1, ns+1, 3, H, W]
        prompt_dict = build_prompt_dict(sup_masks.unsqueeze(0), "mask", n_shots=ns,
                                        train_mode=False, device=imgs.device)
        out = self.sansa(imgs, prompt_dict)
        pred = out["pred_masks"][-1]  # query frame logits [H, W]

        # dense correspondence from SAM2 highest-level vision features
        bo = self.sansa._backbone_output
        T = self.sansa._num_frames
        qf = bo.get_current_feats(T - 1)[-1].squeeze(1)  # [hw, C]
        qf = F.normalize(qf, dim=-1)
        sims = []
        for s in range(ns):
            sf = bo.get_current_feats(s)[-1].squeeze(1)   # [hw, C]
            sf = F.normalize(sf, dim=-1)
            sims.append(sf @ qf.t())                       # [hw(sup), hw(qry)]
        sim = torch.cat(sims, dim=0)                        # [ns*hw, hw]
        return pred.float(), sim.detach()

    def forward(self, x, postprocess=True):
        with torch.set_grad_enabled(self.differentiable and torch.is_grad_enabled()):
            return self._forward_impl(x, postprocess=postprocess)

    def _forward_impl(self, x, postprocess=True):
        images = x[BatchKeys.IMAGES]  # [B, 1+M, 3, H, W], ImageNet-normalized
        masks = self._preprocess_masks(x[BatchKeys.PROMPT_MASKS])  # [B, M, C, H, W]
        B, _, _, H, W = images.shape
        if B > 1:
            # Captum methods stack extra samples on the batch dim (e.g. DeepLift's
            # input+baseline pair); process per-sample and concatenate, keeping the graph.
            outs = [
                self._forward_impl(
                    {k: (v[b:b + 1] if torch.is_tensor(v) and v.shape[:1] == (B,) else v)
                     for k, v in x.items()},
                    postprocess=postprocess,
                )
                for b in range(B)
            ]
            return {
                ResultDict.LOGITS: torch.cat([o[ResultDict.LOGITS] for o in outs], dim=0),
                ResultDict.ATTENTIONS: outs[0][ResultDict.ATTENTIONS],
            }

        query_img = images[:, 0]      # [1,3,H,W]
        supports = images[:, 1:]      # [1, M, 3, H, W]
        num_classes = masks.shape[2]

        decisions, attentions = [], []
        for c in range(num_classes):
            class_examples = x[BatchKeys.FLAG_EXAMPLES][:, :, c + 1]  # [B, M]
            n_shots = int(class_examples.sum().item())
            if n_shots == 0:
                decisions.append(torch.zeros((H, W), device=images.device))
                attentions.append(None)
                continue
            sup_imgs = supports[class_examples]              # [ns,3,H,W]
            sup_masks = masks[:, :, c][class_examples].float()  # [ns,H,W]
            try:
                pred, sim = self._run_sansa(sup_imgs, sup_masks, query_img)
                if pred.shape != (H, W):
                    pred = F.interpolate(pred[None, None], size=(H, W), mode="bilinear",
                                         align_corners=False)[0, 0]
                fg = pred.sigmoid()
            except torch.OutOfMemoryError:
                raise  # never degrade an OOM into a silent zero prediction
            except Exception as e:
                # Fallback for episodes SANSA cannot process (e.g. fully-perturbed supports
                # during insertion steps): a zero prediction is the semantically correct
                # output there. Anything else must be visible, so log and count it.
                if "CUDA" in str(e) or "cuda" in type(e).__name__:
                    raise
                self._fallback_count = getattr(self, "_fallback_count", 0) + 1
                import traceback
                print(f"[SANSA] episode fallback #{self._fallback_count} ({type(e).__name__}: {e})")
                traceback.print_exc()
                fg = torch.zeros((H, W), device=images.device)
                sim = None
            if sim is None:
                hw = (self.sansa.sam.image_size // 16) ** 2  # SAM2 feat grid (64x64 @1024)
                sim = torch.zeros((n_shots * hw, hw), device=images.device)
            decisions.append(fg)
            attentions.append([sim.t().unsqueeze(0).contiguous()])

        decisions = torch.stack(decisions, dim=0).unsqueeze(0)  # [1, C, H, W]
        fg = decisions.clamp(0, 1)
        if fg.size(1) == 1:
            logits = torch.cat([1 - fg, fg], dim=1)
        else:
            bg = (1 - fg.max(dim=1, keepdim=True)[0])
            logits = torch.cat([bg, fg], dim=1)

        # SANSA binarizes the mask prompt, so PROMPT_MASKS falls out of the autograd graph
        # (derivative 0 a.e.). Captum differentiates w.r.t. all gradient inputs and errors on
        # unused tensors; couple the masks with a zero term so they stay in the graph.
        prompt_masks = x[BatchKeys.PROMPT_MASKS]
        if self.differentiable and prompt_masks.requires_grad:
            logits = logits + 0.0 * prompt_masks.sum()

        return {
            ResultDict.LOGITS: logits,
            ResultDict.ATTENTIONS: attentions,
        }

    def feature_ablation(self, result, chosen_class, mask, n_shots=None, explanation_size=None):
        n_levels = len(result[ResultDict.ATTENTIONS][chosen_class])
        return torch.full((n_levels,), 1.0 / n_levels, device=result[ResultDict.LOGITS].device)


def build_sansa_model(sam2_version="large", device="cuda", image_size=1024, **kwargs):
    return SANSAMultiClass(sam2_version=sam2_version, device=device, image_size=image_size, **kwargs)
