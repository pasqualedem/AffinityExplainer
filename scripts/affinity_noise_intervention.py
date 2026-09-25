"""Affinity-noise intervention (appendix experiment).

Adds zero-mean noise of relative strength tau to the support-query affinities ON THE
DECISION PATH of each model, keeping the in-mask/out-mask structure intact on average,
and measures how the final segmentation reacts.

Hypothesis (decoder-dominated claim): Matcher and GF-SAM, whose SAM decoder consumes the
whole prompted region, stay flat as tau grows; matching-dominated models (INSID3, DCAMA)
degrade early because their per-query affinity mass is the decision.

Per (model, tau): mean mIoU vs ground truth and mean agreement (IoU of the predicted
foreground vs the tau=0 prediction of the same episode). Noise is reseeded per episode
for reproducibility.

Usage: .venv/bin/python affinity_noise_intervention.py --model matcher [--episodes-csv val_pascal5i_N1K1_60] [--taus 0,0.25,0.5,1,2,4]
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))  # repo root, for `affex`
import argparse
import copy
import csv
import os

import torch
import torch.nn.functional as F
from torchmetrics.classification import MulticlassJaccardIndex
from tqdm import tqdm

from affex.data import get_dataloaders
from affex.models import build_model_preconfigured
from affex.substitution import Substitutor
from affex.utils.torch import to_device
from affex.utils.utils import ResultDict


def dataset_params(csv_name, image_size):
    return {
        "datasets": {
            csv_name: {
                "name": "pascal", "data_dir": "data/pascal", "split": "val",
                "val_fold_idx": None, "n_folds": 4, "n_shots": 1, "n_ways": 1,
                "do_subsample": False, "val_num_samples": int(csv_name.rsplit("_", 1)[1]),
                "maintain_gt_shape": False, "image_size": image_size,
            }
        },
        "common": {"remove_small_annotations": True, "custom_preprocess": False,
                   "maintain_gt_shape": False},
        "preprocess": {"image_size": image_size},
    }


def set_noise(model, model_name, tau):
    if model_name == "dcama":
        import affex.models.dcama.transformer as T
        T.SIM_NOISE_STD = tau
        return
    targets = {id(model): model}
    for attr in ("matcher", "gfsam", "insid3", "model", "sansa"):
        if hasattr(model, attr):
            o = getattr(model, attr)
            targets[id(o)] = o
    if isinstance(model, torch.nn.Module):
        for m in model.modules():
            targets[id(m)] = m
    for t in targets.values():
        try:
            t.sim_noise_std = tau
        except Exception:
            pass


def fg_pred(model, input_dict):
    with torch.no_grad():
        result = model(input_dict, postprocess=False)
    logits = result[ResultDict.LOGITS]
    return torch.argmax(F.softmax(logits.float(), dim=1), dim=1)  # [B, H, W]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--episodes-csv", default="val_pascal5i_N1K1_60")
    ap.add_argument("--taus", default="0,0.25,0.5,1,2,4")
    ap.add_argument("--out", default="out/affinity_noise")
    args = ap.parse_args()
    taus = [float(t) for t in args.taus.split(",")]
    assert taus[0] == 0.0, "first tau must be 0 (reference prediction)"
    device = "cuda"

    model, image_size = build_model_preconfigured(model_name=args.model)
    model.eval().to(device)

    val = get_dataloaders(
        dataset_params(args.episodes_csv, image_size),
        {"num_workers": 2, "batch_size": 1, "csv_folder": "data_csv"},
        None,
    )
    (dataset_name, loader), = val.items()

    os.makedirs(args.out, exist_ok=True)
    out_csv = os.path.join(args.out, f"{args.model}_{args.episodes_csv}.csv")
    rows = []
    for i, batch in tqdm(enumerate(loader), total=len(loader), desc=args.model):
        batch, _ = batch
        substitutor = Substitutor(substitute=False)
        substitutor.reset(batch=batch)
        input_dict, gt = next(substitutor)
        input_dict = to_device(input_dict, device)
        gt = to_device(gt, device)
        num_classes = int(gt.max().item()) + 1 if gt.max() > 0 else 2

        ref = None
        for tau in taus:
            set_noise(model, args.model, tau)
            torch.manual_seed(1000 * i + int(tau * 100))
            pred = fg_pred(model, input_dict)
            if pred.shape[-2:] != gt.shape[-2:]:
                pred = F.interpolate(pred[None].float(), size=gt.shape[-2:], mode="nearest")[0].long()
            jac = MulticlassJaccardIndex(num_classes=max(num_classes, 2), ignore_index=-100,
                                         average="none").to(device)
            miou = jac(pred, gt.long())[1:].mean().item()  # fg classes only
            if tau == 0.0:
                ref = pred
                agree = 1.0
            else:
                inter = ((pred == 1) & (ref == 1)).sum().item()
                union = ((pred == 1) | (ref == 1)).sum().item()
                agree = inter / union if union > 0 else 1.0
            rows.append({"episode": i, "tau": tau, "miou": miou, "agreement": agree})
        set_noise(model, args.model, 0.0)

    with open(out_csv, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["episode", "tau", "miou", "agreement"])
        w.writeheader()
        w.writerows(rows)

    print(f"\n{args.model} ({args.episodes_csv}):")
    for tau in taus:
        sel = [r for r in rows if r["tau"] == tau]
        m = sum(r["miou"] for r in sel) / len(sel)
        a = sum(r["agreement"] for r in sel) / len(sel)
        print(f"  tau={tau:<5} mIoU={m:.4f}  agreement={a:.4f}  n={len(sel)}")


if __name__ == "__main__":
    main()
