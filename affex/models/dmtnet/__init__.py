import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import repeat, rearrange

from affex.data.utils import BatchKeys, sum_scale
from affex.models.dmtnet.dmtnet import DMTNetwork
from affex.utils.utils import ResultDict


def build_dmtnet(backbone="resnet50", model_checkpoint="checkpoints/dmtnet.pt", voting=True):
    model = DMTNetMultiClass(backbone, voting=voting)
    src_dict = torch.load(model_checkpoint, map_location="cpu")
    src_dict = {k[len("module."):]: v for k, v in src_dict.items()}
    model.load_state_dict(src_dict)
    return model


class DMTNetMultiClass(DMTNetwork):
    def __init__(self, *args, **kwargs):
        self.predict = None
        self.generate_class_embeddings = None
        super().__init__(*args, **kwargs)

    def _preprocess_masks(self, masks, dims):
        B, N, C, H, W = masks.size()
        # remove bg from masks
        masks = masks[:, :, 1:, ::]
        mask_size = 256

        # Repeat dims along class dimension
        support_dims = dims[:, 1:]
        repeated_dims = repeat(support_dims, "b n d -> (b n c) d", c=C)
        masks = rearrange(masks, "b n c h w -> (b n c) h w")

        # Remove padding from masks
        # pad_dims = [get_preprocess_shape(h, w, mask_size) for h, w in repeated_dims]
        # masks = [mask[:h, :w] for mask, (h, w) in zip(masks, pad_dims)]
        # masks = torch.cat(
        #     [
        #         F.interpolate(
        #             torch.unsqueeze(mask, 0).unsqueeze(0),
        #             size=(self.image_size, self.image_size),
        #             mode="nearest",
        #         )[0]
        #         for mask in masks
        #     ]
        # )
        return rearrange(masks, "(b n c) h w -> b n c h w", b=B, n=N)

    def forward(self, x, postprocess=True):

        masks = self._preprocess_masks(x[BatchKeys.PROMPT_MASKS], x[BatchKeys.DIMS])
        B = masks.size(0)
        # assert B == 1, "Only tested with batch size = 1"
        voting_masks = []
        fg_logits_masks = []
        attentions = []
        # get logits for each class
        for c in range(masks.size(2)):
            class_examples = x[BatchKeys.FLAG_EXAMPLES][:, :, c + 1]
            n_shots = class_examples.sum().item() // B
            class_input_dict = {
                "query_img": x[BatchKeys.IMAGES][:, 0],
                "support_imgs": rearrange(x[BatchKeys.IMAGES][:, 1:][class_examples], "(b m) c h w -> b m c h w", b=B),
                "support_masks": rearrange(masks[:, :, c, ::][class_examples], "(b m) h w -> b m h w", b=B),
            }
            if n_shots == 1:
                result = self.predict_mask_1shot(
                    class_input_dict["query_img"],
                    class_input_dict["support_imgs"][:, 0],
                    class_input_dict["support_masks"][:, 0],
                )
                logit_mask, bg_logit_mask, pred_mask = result[ResultDict.LOGITS]
                attentions.append([result[ResultDict.ATTENTIONS]])
                fg_logits_masks.append(logit_mask)
            else:
                result = self.predict_mask_nshot(class_input_dict, n_shots)
                (voting_mask, logit_mask_orig, bg_logit_mask_orig) = result[ResultDict.LOGITS]
                attentions.append(result[ResultDict.ATTENTIONS])
                voting_masks.append(voting_mask)
                
        if fg_logits_masks:
            raw_logits = torch.stack(fg_logits_masks, dim=1)
            raw_logits = F.softmax(raw_logits, dim=2)
            fg_logits = raw_logits[:, :, 1, ::]
            bg_logits = raw_logits[:, :, 0, ::]
            bg_positions = fg_logits.argmax(dim=1)
            bg_logits = torch.gather(bg_logits, 1, bg_positions.unsqueeze(1))
            logits = torch.cat([bg_logits, fg_logits], dim=1)
        elif not self.vote_prediction:
            if masks.size(2) == 1:
                logits = voting_masks[0]
            else:
                multiclass_logits = torch.stack(voting_masks, dim=1) # voting masks are actually logits
                fg_logits = multiclass_logits[:, :, 1, ::]
                bg_logits = multiclass_logits[:, :, 0, ::]
                bg_positions = fg_logits.argmax(dim=1)
                bg_logits = torch.gather(bg_logits, 1, bg_positions.unsqueeze(1))
                logits = torch.cat([bg_logits, fg_logits], dim=1)
        else:
            votes = torch.stack([class_res for class_res in voting_masks], dim=1)
            preds = (votes.argmax(dim=1)+1) * (votes > 0.5).max(dim=1).values
            logits = rearrange(F.one_hot(preds, num_classes=len(voting_masks)+1), "b h w c -> b c h w").float()
            
        if postprocess:
            logits = self.postprocess_masks(logits, x["dims"])

        return {
            ResultDict.LOGITS: logits,
            ResultDict.ATTENTIONS: attentions,
        }

    def postprocess_masks(self, logits, dims):
        max_dims = torch.max(dims.view(-1, 2), 0).values.tolist()
        dims = dims[:, 0, :]  # get real sizes of the query images
        logits = [
            F.interpolate(
                torch.unsqueeze(logit, 0),
                size=dim.tolist(),
                mode="bilinear",
                align_corners=False,
            )
            for logit, dim in zip(logits, dims)
        ]

        logits = torch.cat(
            [
                F.pad(
                    mask,
                    (
                        0,
                        max_dims[1] - dims[i, 1],
                        0,
                        max_dims[0] - dims[i, 0],
                    ),
                    mode="constant",
                    value=float("-inf"),
                )
                for i, mask in enumerate(logits)
            ]
        )
        return logits
    
    def pred_layer(self, corr, bg_corr, image_size):
        logit_mask = self.hpn_learner(corr)
        logit_mask = F.interpolate(logit_mask, image_size, mode='bilinear', align_corners=True)
        bg_logit_mask = self.hpn_learner(bg_corr)
        bg_logit_mask = F.interpolate(bg_logit_mask, image_size, mode='bilinear', align_corners=True)
        return logit_mask, bg_logit_mask

    def feature_ablation(self, result, chosen_class, mask, n_shots, explanation_size, **kwargs):
        attentions = result[ResultDict.ATTENTIONS]
        with torch.no_grad():
            for shot in range(n_shots):
                corr, bg_corr = attentions[chosen_class][shot]
                orig_out, bg_orig_out = self.pred_layer(corr, bg_corr, explanation_size)
                orig_out = orig_out[:, :, mask[chosen_class]]
                bg_orig_out = bg_orig_out[:, :, mask[chosen_class]] 
        diffs = []
        for shot in range(n_shots):
            shot_diffs = []
            corr, bg_corr = attentions[chosen_class][shot]
            for i in range(len(corr)):
                for j in range(corr[i].shape[1]):
                    new_input = corr[i].clone()
                    new_input[:, j] = 0
                    new_expl_input = [*corr[0:i], *[new_input], *corr[i+1:]]
                    
                    new_input_bg = bg_corr[i].clone()
                    new_input_bg[:, j] = 0
                    new_expl_input_bg = [*bg_corr[0:i], *[new_input_bg], *bg_corr[i+1:]]
                    new_expl_input = [new_expl_input, new_expl_input_bg]
                    
                    with torch.no_grad():
                        new_out , new_out_bg = self.pred_layer(*new_expl_input, explanation_size)
                    new_out = new_out[:, :, mask[chosen_class]]
                    new_out_bg = new_out_bg[:, :, mask[chosen_class]]
                    
                    diff = orig_out - new_out
                    diff_bg = bg_orig_out - new_out_bg
                    diff = ((torch.abs(diff) + torch.abs(diff_bg)) / 2).mean()
                    shot_diffs.append(diff)
            diffs.append(shot_diffs)
        
        diffs = list(zip(*diffs))
        diffs = [torch.stack(diff, dim=0).mean() for diff in diffs]
        
        abl_attr = sum_scale(torch.stack(diffs))
        return abl_attr
