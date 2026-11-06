"""Similarity-level interventions for the appendix experiments.

`push_down_topk` is a deletion applied to the affinities instead of to the support
pixels: for every query position, its highest in-mask support similarities are lowered,
while the out-of-mask similarities and the image itself are left untouched.
"""
import math

import torch


def push_down_topk(S, in_mask, kappa, tau):
    """Lower the top in-mask similarities of every query position.

    Args:
        S: [..., Nq, Ns] similarities, support axis last.
        in_mask: bool tensor broadcastable to S with a singleton query axis, [..., 1, Ns]:
            the support mask as the model sees it on its own similarity grid.
        kappa: fraction of the in-mask support entries lowered at each query position.
        tau: each lowered entry drops by exactly tau * std(S) (the delta of Appendix I). A fixed drop keeps the curve
            nested in kappa; a random drop would let the in-mask maximum survive through
            entries that happen to be lowered little.
    """
    n_in = int(in_mask.sum(-1).max().item())
    k = min(math.ceil(kappa * n_in), S.shape[-1])
    if k == 0 or tau == 0:
        return S
    valid = S > -1e8  # ignore masked fills
    sigma = S[valid].float().std()
    in_mask = in_mask.expand_as(S)
    ranked = S.masked_fill(~in_mask, float("-inf"))
    idx = ranked.topk(k, dim=-1).indices
    # Masks with fewer than k in-mask entries: never lower an out-of-mask entry.
    drop = torch.gather(in_mask, -1, idx).to(S.dtype) * (tau * sigma)
    return S.scatter_add(-1, idx, -drop.to(S.dtype))
