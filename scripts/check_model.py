r"""Check that a model still computes what it used to.

Refactors like vendoring an upstream repository or moving files should not change a
model's output. This runs one episode and compares against a saved reference, so a
port can be validated in a minute instead of rerunning an experiment.

    # before the change
    python scripts/check_model.py gfsam --save refs/gfsam.npz
    # after the change
    python scripts/check_model.py gfsam --compare refs/gfsam.npz

Two distances are reported: the fraction of mask pixels that differ, and the mean
absolute difference between attribution maps relative to the reference's own scale.
Both are 0 for a deterministic model that is unchanged. Models that resample internally
(Matcher redraws SAM prompts) never reach 0, so measure their noise floor first with
--repeat and accept a comparison that lands inside it.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))  # repo root, for `affex`

import argparse
from statistics import mean

import numpy as np
import torch
import torch.nn.functional as F

from affex.data import get_dataloaders
from affex.explainer import build_explainer
from affex.explainer.affinity import get_explanation_mask
from affex.models import build_model_preconfigured
from affex.substitution import Substitutor
from affex.utils.torch import to_device
from affex.utils.utils import ResultDict

# The paper's AffEx settings, so the check exercises the configuration we report.
AFFEX = dict(explanation_size=64, mask_blur_kernel_size=7, mask_blur_sigma=50,
             mask_dilation_radius=5, mask_dilation_kernel=7)


def load_episode(dataset, image_size, shots, index, device):
    name = ("val_coco20i" if dataset == "coco" else "val_pascal5i") + f"_N1K{shots}"
    common = {"remove_small_annotations": True, "custom_preprocess": False,
              "maintain_gt_shape": False}
    if dataset == "coco":
        cfg = {"name": "coco", "split": "val", "n_shots": shots, "n_ways": 1,
               "instances_path": "data/coco/annotations/instances_val2014.json",
               "img_dir": "data/coco/train_val_2017", "all_example_categories": False,
               "do_subsample": False, "add_box_noise": False, "val_fold_idx": None,
               "n_folds": None}
    else:
        cfg = {"name": "pascal", "split": "val", "data_dir": "data/pascal", "n_shots": shots,
               "n_ways": 1, "n_folds": 4, "val_fold_idx": None, "do_subsample": False,
               "val_num_samples": 1000, "maintain_gt_shape": False}
        common["ignore_borders"] = True
    cfg["image_size"] = image_size

    loader = get_dataloaders(
        {"datasets": {name: cfg}, "common": common, "preprocess": {"image_size": image_size}},
        {"num_workers": 0, "batch_size": 1, "csv_folder": "data_csv"}, num_processes=1)
    loader = next(iter(loader.values()))
    sampler = loader.batch_sampler
    start = sum(sampler.batch_sizes[:index])
    meta = {k: v[index] for k, v in sampler.batch_metadata.items()}
    batch = loader.collate_fn([loader.dataset[(start + j, meta)]
                               for j in range(sampler.batch_sizes[index])])
    sub = Substitutor(substitute=True)
    sub.reset(batch=batch[0])
    chosen, gt = next(sub)
    return to_device(chosen, device), gt.to(device)


def run_once(model, batch, gt, explainer_name, device):
    """One forward pass plus one attribution, as numpy."""
    with torch.no_grad():
        result = model(batch, postprocess=False)
    logits = F.interpolate(result[ResultDict.LOGITS], size=gt.shape[-2:],
                           mode="bilinear", align_corners=False)
    pred = logits.argmax(dim=1) == 1
    valid = gt >= 0
    union = ((pred | (gt == 1)) & valid).sum().item()
    miou = (pred & (gt == 1) & valid).sum().item() / union if union else float("nan")

    params = dict(AFFEX) if "affinity" in explainer_name else {}
    explainer = build_explainer(name=explainer_name, model=model, params=params, device=device)
    size = AFFEX["explanation_size"] if params else gt.shape[-1]
    mask = get_explanation_mask(batch, gt=None, result=result,
                                target_shape=(size, size), masking_type="logits")
    attr = explainer.explain(input_dict=batch, explanation_mask=mask)
    attr = attr[0] if isinstance(attr, (list, tuple)) else attr
    return {"pred": pred[0].cpu().numpy(),
            "attr": attr.detach().float().cpu().numpy(),
            "miou": miou}


def mask_mae(a, b):
    """Fraction of mask pixels that differ."""
    return float(np.abs(a.astype(np.float32) - b.astype(np.float32)).mean())


def attr_mae(a, b):
    """Mean absolute difference between attributions, in units of the first one's scale."""
    return float(np.abs(a - b).mean() / max(float(np.abs(a).mean()), 1e-12))


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("model")
    ap.add_argument("--dataset", default="pascal", choices=["pascal", "coco"])
    ap.add_argument("--shots", type=int, default=1)
    ap.add_argument("--episode", type=int, default=0)
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--explainer", default="signed_affinity")
    ap.add_argument("--repeat", type=int, default=1,
                    help="Run N times and report the spread between runs, which is the "
                         "noise floor any comparison has to beat.")
    ap.add_argument("--save", help="Write this run as a reference .npz")
    ap.add_argument("--compare", help="Compare this run against a reference .npz")
    args = ap.parse_args()

    torch.manual_seed(0)
    model, image_size = build_model_preconfigured(model_name=args.model)
    model.eval().to(args.device)
    batch, gt = load_episode(args.dataset, image_size, args.shots, args.episode, args.device)

    runs = [run_once(model, batch, gt, args.explainer, args.device)
            for _ in range(max(1, args.repeat))]

    print(f"{args.model} on {args.dataset} {args.shots}-shot, episode {args.episode}")
    for i, r in enumerate(runs):
        label = f"  run {i}:" if args.repeat > 1 else "  result:"
        print(f"{label} {int(r['pred'].sum()):>8d} foreground pixels, mIoU {r['miou']:.4f}")

    if args.repeat > 1:
        print(f"  run to run: mask MAE {mean(mask_mae(x['pred'], y['pred']) for x, y in zip(runs, runs[1:])):.6f}"
              f", attribution MAE {mean(attr_mae(x['attr'], y['attr']) for x, y in zip(runs, runs[1:])):.6f}")

    if args.save:
        Path(args.save).parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(args.save, pred=runs[0]["pred"], attr=runs[0]["attr"])
        print(f"  saved reference to {args.save}")

    if args.compare:
        ref = np.load(args.compare)
        m, a = mask_mae(ref["pred"], runs[0]["pred"]), attr_mae(ref["attr"], runs[0]["attr"])
        print(f"  vs {args.compare}: mask MAE {m:.6f}, attribution MAE {a:.6f}"
              f"{'  (identical)' if m == 0 and a == 0 else ''}")


if __name__ == "__main__":
    main()
