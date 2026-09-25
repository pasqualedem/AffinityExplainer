r""" PANet-style prototype head on the (frozen) DCAMA backbone.

Same-encoder / different-head comparison requested by reviewer M83E: the encoder is
byte-identical to DCAMA's (same Swin checkpoint, frozen exactly as in DCAMA training);
only the head differs. The head follows PANet: masked average pooling of the support
features into foreground/background prototypes, cosine similarity to every query
position, scaled by a learned temperature. A per-level 1x1 projection (identity init)
is the only trained module, matching DCAMA's head-only training regime.

Levels used: the last block of the three deepest backbone stages (1/8, 1/16, 1/32),
the same feature range DCAMA reads. Per-level dense query-support cosine similarities
are exposed as ResultDict.ATTENTIONS in the `b (hq wq) (n hs ws)` layout so the
AffinityExplainer applies unchanged.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange

from affex.models.dcama.dcama import AbstractDCAMA, DCAMAMultiClass
from affex.utils.utils import ResultDict
from affex.data.utils import sum_scale


class PANetHeadModule(nn.Module):
    def __init__(self, in_channels, stack_ids, temperature=20.0):
        super().__init__()
        self.level_ids = [stack_ids[1] - 1, stack_ids[2] - 1, stack_ids[3] - 1]
        chans = [in_channels[1], in_channels[2], in_channels[3]]
        self.projs = nn.ModuleList([nn.Conv2d(c, c, 1) for c in chans])
        for proj in self.projs:
            nn.init.eye_(proj.weight.view(proj.weight.shape[0], -1))
            nn.init.zeros_(proj.bias)
        self.temperature = nn.Parameter(torch.tensor(float(temperature)))

    def forward(self, query_feats, support_feats, support_mask, nshot=1):
        sfeats = [support_feats] if nshot == 1 else support_feats
        smasks = support_mask.unsqueeze(1) if support_mask.dim() == 3 else support_mask

        level_logits, attns, proto_sims = [], [], []
        for i, lidx in enumerate(self.level_ids):
            qf = F.normalize(self.projs[i](query_feats[lidx]), dim=1)
            h, w = qf.shape[-2:]

            sf, masks = [], []
            for k in range(nshot):
                sf.append(F.normalize(self.projs[i](sfeats[k][lidx]), dim=1))
                masks.append(F.interpolate(smasks[:, k].unsqueeze(1).float(), (h, w),
                                           mode='bilinear', align_corners=True))
            sf = torch.stack(sf, dim=1)      # B N C h w
            masks = torch.stack(masks, dim=1)  # B N 1 h w

            proto_fg = (sf * masks).sum(dim=(1, 3, 4)) / (masks.sum(dim=(1, 3, 4)) + 1e-6)
            proto_bg = (sf * (1 - masks)).sum(dim=(1, 3, 4)) / ((1 - masks).sum(dim=(1, 3, 4)) + 1e-6)
            proto_fg = F.normalize(proto_fg, dim=1)
            proto_bg = F.normalize(proto_bg, dim=1)

            sim_fg = torch.einsum('bchw,bc->bhw', qf, proto_fg)
            sim_bg = torch.einsum('bchw,bc->bhw', qf, proto_bg)
            level_logits.append(self.temperature * torch.stack([sim_bg, sim_fg], dim=1))

            attn = torch.einsum('bcq,bcs->bqs', qf.flatten(2),
                                rearrange(sf, 'b n c h w -> b c (n h w)'))
            attns.append(self.temperature.detach() * attn)

            # Native prototype read-out (reviewer M83E): per-support-pixel similarity to
            # the foreground prototype, i.e. the pre-pooling activation map upsampled.
            proto_sims.append(torch.einsum('bnchw,bc->bnhw', sf, proto_fg))

        out_size = level_logits[0].shape[-2:]
        resized = [F.interpolate(l, out_size, mode='bilinear', align_corners=True)
                   for l in level_logits]
        logits = torch.stack(resized).mean(dim=0)

        return {
            ResultDict.LOGITS: logits,
            ResultDict.ATTENTIONS: attns,
            ResultDict.COARSE_MASKS: resized,
            ResultDict.QUERY_FEATS: query_feats,
            ResultDict.SUPPORT_FEATS: support_feats,
            ResultDict.NSHOT: nshot,
            'prototype_readout': proto_sims,
        }


class PANetMultiClass(DCAMAMultiClass):
    def __init__(self, backbone, pretrained_path, image_size=384, temperature=20.0, voting=None):
        self.predict = None
        self.generate_class_embeddings = None
        self.image_size = image_size
        AbstractDCAMA.__init__(self, backbone, pretrained_path,
                               use_original_imgsize=False, voting=voting)
        self.model = PANetHeadModule(self.feat_channels, self.stack_ids, temperature)
        self.cross_entropy_loss = nn.CrossEntropyLoss()

    def forward_episode(self, query_img, support_img, support_mask, query_mask=None):
        # Training entry point (DCAMAMultiClass.forward keeps the multiclass dict API).
        result = self.forward_1shot(query_img, support_img, support_mask)
        if query_mask is not None:
            result[ResultDict.LOSS] = self.compute_objective(result, query_mask)
        return result

    def forward_1shot(self, query_img, support_img, support_mask):
        # The backbone is frozen (as in DCAMA training); skip building its graph.
        with torch.no_grad():
            query_feats = self.extract_feats(query_img)
            support_feats = self.extract_feats(support_img)
        return self.model(query_feats, support_feats, support_mask.clone())

    def compute_objective(self, result, gt_mask):
        logits = result[ResultDict.LOGITS]
        logits = F.interpolate(logits, gt_mask.shape[-2:], mode='bilinear', align_corners=True)
        return self.cross_entropy_loss(logits, gt_mask.long())

    def get_learnable_params(self, train_params=None):
        return self.model.parameters()

    def pred_layer(self, *args, **kwargs):
        raise NotImplementedError("PANet head has no mixer stack; see feature_ablation.")

    def feature_ablation(self, result, chosen_class, mask, n_shots=None, explanation_size=None):
        level_logits = result[ResultDict.COARSE_MASKS][chosen_class]
        if explanation_size is not None:
            level_logits = [
                F.interpolate(l, size=explanation_size, mode="bilinear", align_corners=False)
                if l.shape[-2] != explanation_size else l
                for l in level_logits
            ]
        with torch.no_grad():
            orig = torch.stack(level_logits).mean(dim=0)
            orig_out = orig[:, :, mask[chosen_class]].mean(dim=2)
            diffs = []
            for i in range(len(level_logits)):
                others = [l for j, l in enumerate(level_logits) if j != i]
                abl_out = torch.stack(others).mean(dim=0)[:, :, mask[chosen_class]].mean(dim=2)
                diffs.append(orig_out - abl_out)
        return sum_scale(torch.cat([torch.abs(d[:, 1]) for d in diffs]))


def build_panet(
    backbone: str = "swin",
    backbone_checkpoint: str = "checkpoints/dcama/swin_base_patch4_window12_384.pth",
    model_checkpoint: str = None,
    image_size: int = 384,
    temperature: float = 20.0,
    voting=None,
):
    model = PANetMultiClass(backbone, backbone_checkpoint, image_size=image_size,
                            temperature=temperature, voting=voting)
    if model_checkpoint is not None:
        state_dict = torch.load(model_checkpoint, map_location="cpu")
        model.model.load_state_dict(state_dict)
    return model
