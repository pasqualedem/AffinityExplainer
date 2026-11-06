"""Top-k push-down intervention (appendix experiment).

A deletion applied to the affinities instead of to the support pixels. At every query
position, the top ceil(kappa * |F|) in-mask support similarities are lowered by
exactly tau * std(S) (kappa and tau are the kappa and delta of the paper's Appendix I);
out-of-mask similarities and the images are untouched.
The in-mask set F is the support mask as each model sees it on its own similarity grid.

Hypothesis: a matching-dominated model reacts early, because its decision rests on
those similarities; a decoder-dominated model reacts little, because the rest of the
support mask still points it at the object.

All kappa values are evaluated in the same pass over the episodes, like a deletion curve.
Per (model, kappa): mean mIoU vs ground truth and mean agreement (foreground IoU vs the
kappa=0 prediction of the same episode). Every forward pass is seeded (torch, numpy and
Python's random), so Matcher's prompt sampling is deterministic.

Usage: .venv/bin/python scripts/affinity_topk_intervention.py --model matcher \
           [--episodes-csv val_pascal5i_N1K1] [--kappas 0,0.01,0.05,0.1,0.25,0.5] [--taus 2,4] \
           [--shard 0/4]
Rows are appended per episode, so an interrupted run resumes where it stopped.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))  # repo root, for `affex`
import argparse
import csv
import os
import random

import numpy as np
import torch
import torch.nn.functional as F
from torchmetrics.classification import MulticlassJaccardIndex
from tqdm import tqdm

from affex.data import get_dataloaders
from affex.models import build_model_preconfigured
from affex.substitution import Substitutor
from affex.utils.torch import to_device
from affex.utils.utils import ResultDict


def set_topk(model, model_name, value):
    """value = (kappa, tau), or None to switch the intervention off."""
    if model_name == "dcama":
        import affex.models.dcama.transformer as T
        T.SIM_TOPK_DOWN = value
        return
    targets = {id(model): model}
    for attr in ("matcher", "gfsam", "insid3", "model"):
        if hasattr(model, attr):
            o = getattr(model, attr)
            targets[id(o)] = o
    if isinstance(model, torch.nn.Module):
        for m in model.modules():
            targets[id(m)] = m
    for t in targets.values():
        try:
            t.sim_topk_down = value
        except Exception:
            pass


def seed_all(seed):
    random.seed(seed)
    np.random.seed(seed % (2 ** 32))
    torch.manual_seed(seed)


def episode_params(csv_name, image_size):
    """Pascal 1-shot dataset parameters for an episode list, with the episode count read
    from the list itself."""
    with open(os.path.join("data_csv", f"{csv_name}.csv")) as f:
        n = sum(1 for _ in f) - 1
    return {
        "datasets": {
            csv_name: {
                "name": "pascal", "data_dir": "data/pascal", "split": "val",
                "val_fold_idx": None, "n_folds": 4, "n_shots": 1, "n_ways": 1,
                "do_subsample": False, "val_num_samples": n,
                "maintain_gt_shape": False, "image_size": image_size,
            }
        },
        "common": {"remove_small_annotations": True, "custom_preprocess": False,
                   "maintain_gt_shape": False},
        "preprocess": {"image_size": image_size},
    }


def fg_pred(model, input_dict):
    with torch.no_grad():
        result = model(input_dict, postprocess=False)
    logits = result[ResultDict.LOGITS]
    return torch.argmax(F.softmax(logits.float(), dim=1), dim=1)  # [B, H, W]


FIELDS = ["episode", "tau", "kappa", "miou", "agreement"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--episodes-csv", default="val_pascal5i_N1K1")
    ap.add_argument("--kappas", default="0,0.01,0.05,0.1,0.25,0.5")
    ap.add_argument("--taus", default="2,4")
    ap.add_argument("--shard", default="0/1", help="i/n: run the episodes with index %% n == i")
    ap.add_argument("--out", default="out/affinity_topk")
    args = ap.parse_args()
    kappas = [float(k) for k in args.kappas.split(",")]
    assert kappas[0] == 0.0, "first kappa must be 0 (reference prediction)"
    taus = [float(t) for t in args.taus.split(",")]
    shard, n_shards = (int(x) for x in args.shard.split("/"))
    device = "cuda"

    os.makedirs(args.out, exist_ok=True)
    out_csv = os.path.join(args.out, f"{args.model}_{args.episodes_csv}_shard{shard}of{n_shards}.csv")
    # Resume: an episode counts as done once all its rows are on disk.
    per_episode = 1 + len(taus) * (len(kappas) - 1)
    done = set()
    if os.path.exists(out_csv):
        with open(out_csv) as f:
            counts = {}
            for r in csv.DictReader(f):
                counts[r["episode"]] = counts.get(r["episode"], 0) + 1
        done = {int(e) for e, c in counts.items() if c == per_episode}
    fresh = not os.path.exists(out_csv)
    fh = open(out_csv, "a", newline="")
    writer = csv.DictWriter(fh, fieldnames=FIELDS)
    if fresh:
        writer.writeheader()

    model, image_size = build_model_preconfigured(model_name=args.model)
    model.eval().to(device)
    val = get_dataloaders(
        episode_params(args.episodes_csv, image_size),
        {"num_workers": 2, "batch_size": 1, "csv_folder": "data_csv"},
        None,
    )
    (dataset_name, loader), = val.items()

    todo = [i for i in range(len(loader)) if i % n_shards == shard and i not in done]
    bar = tqdm(total=len(todo), desc=f"{args.model} {shard}/{n_shards}")
    for i, batch in enumerate(loader):
        if i % n_shards != shard or i in done:
            continue
        batch, _ = batch
        substitutor = Substitutor(substitute=False)
        substitutor.reset(batch=batch)
        input_dict, gt = next(substitutor)
        input_dict = to_device(input_dict, device)
        gt = to_device(gt, device)
        num_classes = int(gt.max().item()) + 1 if gt.max() > 0 else 2
        jac = MulticlassJaccardIndex(num_classes=max(num_classes, 2), ignore_index=-100,
                                     average="none").to(device)

        def run(value):
            set_topk(model, args.model, value)
            seed_all(1000 * i)  # same prompt-sampling seed for every setting of an episode
            pred = fg_pred(model, input_dict)
            if pred.shape[-2:] != gt.shape[-2:]:
                pred = F.interpolate(pred[None].float(), size=gt.shape[-2:], mode="nearest")[0].long()
            return pred, jac(pred, gt.long())[1:].mean().item()  # fg classes only

        ref, miou = run(None)
        rows = [{"episode": i, "tau": 0.0, "kappa": 0.0, "miou": miou, "agreement": 1.0}]
        for tau in taus:
            for kappa in kappas[1:]:
                pred, miou = run((kappa, tau))
                inter = ((pred == 1) & (ref == 1)).sum().item()
                union = ((pred == 1) | (ref == 1)).sum().item()
                rows.append({"episode": i, "tau": tau, "kappa": kappa, "miou": miou,
                             "agreement": inter / union if union > 0 else 1.0})
        set_topk(model, args.model, None)
        writer.writerows(rows)
        fh.flush()
        bar.update(1)
    bar.close()
    fh.close()
    print(f"done: {out_csv}")


if __name__ == "__main__":
    main()
