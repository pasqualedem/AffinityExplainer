"""Insertion and deletion on one episode, exported as the demo app's ZIP does.

The app (affex/app.py) runs these curves inside a browser session, which is dropped when
the tab sleeps or the page reruns. For slow models such as Matcher (about 8 s per forward
pass, 150 passes for the two curves) this script runs the same pipeline unattended:

  * the episode list the app uses (data_csv/val_<dataset>_N1K<shots>.csv), picked by
    index, or by index among the episodes of one class;
  * AffEx (signed_affinity) with the per-model settings of the paper, explaining the
    predicted query mask;
  * IAUC and DAUC with mIoU, recording every step.

Output, in --out: {iauc,dauc}_curve.{csv,svg,pdf,png} and, for every step,
{iauc,dauc}_step{j}_{probs,segmentation,shot<k>}.png. Python's, NumPy's and torch's
random generators are seeded once, so Matcher's prompt sampling is reproducible.

Usage: .venv/bin/python scripts/insertion_deletion_example.py --model matcher \
           --dataset coco --shots 5 --episode 12 [--class-name toaster] [--steps 75] --out <dir>
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))  # repo root, for `affex`
import argparse
import os
import random
from io import BytesIO

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from PIL import Image

from affex.data import get_dataloaders
from affex.data.utils import BatchKeys
from affex.explainer import build_explainer
from affex.metrics import FSSCausalMetric
from affex.models import build_model_preconfigured
from affex.substitution import Substitutor
from affex.utils.segmentation import unnormalize
from affex.utils.torch import to_device
from affex.utils.utils import ResultDict

# AffEx settings of the paper, as in the app (AFFINITY_BY_MODEL).
AFFINITY = dict(explanation_size=256, mask_blur_kernel_size=7, mask_blur_sigma=50,
                mask_dilation_radius=5, mask_dilation_kernel=7)
AFFINITY_BY_MODEL = {
    "insid3": dict(AFFINITY, explanation_size=64),
    "sansa": dict(AFFINITY, explanation_size=64),
    "gfsam": dict(AFFINITY, explanation_size=37),
    "matcher": dict(AFFINITY, explanation_size=37),
    "dcama": dict(AFFINITY, mask_dilation_radius=0, mask_dilation_kernel=None),
    "dmtnet": dict(AFFINITY, mask_dilation_radius=0, mask_dilation_kernel=None),
}


def dataset_config(dataset, image_size, shots):
    if dataset == "coco":
        params = {"name": "coco", "instances_path": "data/coco/annotations/instances_val2014.json",
                  "all_example_categories": False,
                  "img_dir": "data/coco/train_val_2017" if os.path.exists("data/coco/train_val_2017") else None,
                  "n_shots": shots, "n_ways": 1, "do_subsample": False, "add_box_noise": False,
                  "val_fold_idx": None, "n_folds": None, "split": "val", "image_size": image_size}
        common = {"remove_small_annotations": True, "custom_preprocess": False, "maintain_gt_shape": False}
        key = f"val_coco20i_N1K{shots}"
    else:
        params = {"name": "pascal", "data_dir": "data/pascal", "split": "val", "val_fold_idx": None,
                  "n_folds": 4, "n_shots": shots, "n_ways": 1, "do_subsample": False,
                  "val_num_samples": 1000, "maintain_gt_shape": False, "image_size": image_size}
        common = {"remove_small_annotations": True, "ignore_borders": True, "custom_preprocess": False}
        key = f"val_pascal5i_N1K{shots}"
    return {"datasets": {key: params}, "common": common, "preprocess": {"image_size": image_size}}


def fetch_batch(loader, index):
    """The app's direct episode lookup (no iteration up to `index`)."""
    sampler = loader.batch_sampler
    size = sampler.batch_sizes[index]
    start = sum(sampler.batch_sizes[:index])
    metadata = {k: v[index] for k, v in sampler.batch_metadata.items()}
    return loader.collate_fn([loader.dataset[(start + j, metadata)] for j in range(size)])


def class_episodes(loader, class_name):
    """Indices of the episodes whose query class is `class_name`, as the app's class filter."""
    sampler = loader.batch_sampler
    categories = next(iter(loader.dataset.categories.values()))
    out = []
    for b, classes in enumerate(sampler.batch_metadata[BatchKeys.CLASSES]):
        query = classes[0] if classes else []
        if class_name in [categories[c]["name"] for c in query if c in categories]:
            out.append(b)
    return out


# Rendering, identical to the app's helpers.
def tint_foreground(image_3chw, mask_hw, tint=(0.2, -0.1, -0.1), tint_border=(0.5, -0.3, -0.3)):
    thickness = 3
    image_3chw[:, mask_hw.cpu().bool()] += torch.tensor(tint).unsqueeze(-1)
    image_3chw = torch.clamp(image_3chw, 0, 1)
    query_mask = (mask_hw > 0.5).unsqueeze(0).unsqueeze(0).float()
    kernel = 2 * thickness + 1
    erosion = -F.max_pool2d(-query_mask, kernel_size=kernel, stride=1, padding=thickness)
    border_mask = ((query_mask - erosion) > 0).squeeze(0).squeeze(0).cpu().bool()
    image_3chw[:, border_mask] += torch.tensor(tint_border).unsqueeze(-1)
    return torch.clamp(image_3chw, 0, 1)


def to_pil(tensor):
    return Image.fromarray((tensor * 255).permute(1, 2, 0).detach().type(torch.uint8).cpu().numpy())


def heatmap(tensor):
    rgba = plt.get_cmap("jet")(tensor.detach().cpu().numpy())
    return Image.fromarray((rgba[..., :3] * 255).astype(np.uint8))


def curve_figure(scores, name, auc_value, measure="miou"):
    x = np.linspace(0, 1, len(scores))
    y = np.asarray(scores, dtype=float)
    fig, ax = plt.subplots(figsize=(45 / 5, 15 / 5), dpi=150)
    ax.plot(x, y, linewidth=2)
    ax.fill_between(x, 0, y, alpha=0.3)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1.05)
    ax.set_xlabel(f"Pixels {'inserted' if name == 'iauc' else 'deleted'}")
    ax.set_ylabel(f"Score ({'mIoU' if measure == 'miou' else 'Logits'})")
    ax.set_title(f"{name.upper()} Curve")
    ax.text(0.8, 0.8, f"AUC = {(float(auc_value) * 100):.2f}", fontsize=12,
            bbox=dict(facecolor="white", alpha=0.6))
    ax.grid(True, linestyle="--", linewidth=0.7, alpha=0.6)
    for spine in ["top", "right"]:
        ax.spines[spine].set_visible(False)
    return fig


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--dataset", default="coco", choices=["coco", "pascal"])
    ap.add_argument("--shots", type=int, default=5)
    ap.add_argument("--episode", type=int, required=True,
                    help="episode index, or index among the episodes of --class-name")
    ap.add_argument("--class-name", default="")
    ap.add_argument("--steps", type=int, default=75)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)
    device = "cuda"

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    model, image_size = build_model_preconfigured(model_name=args.model)
    model.eval().to(device)
    (dataset_name, loader), = get_dataloaders(
        dataset_config(args.dataset, image_size, args.shots),
        {"num_workers": 0, "batch_size": 1, "csv_folder": "data_csv"}, num_processes=1,
    ).items()
    index = args.episode
    if args.class_name:
        matches = class_episodes(loader, args.class_name)
        index = matches[args.episode]
        print(f"class '{args.class_name}': {len(matches)} episodes, using episode {index}")
    raw, _ = fetch_batch(loader, index)
    substitutor = Substitutor(substitute=True)
    substitutor.reset(batch=raw)
    batch, gt = next(substitutor)
    batch, gt = to_device(batch, device), gt.to(device)

    # The explained region is the predicted query mask, as in the app.
    with torch.no_grad():
        result = model(batch, postprocess=False)
    logits = F.interpolate(result[ResultDict.LOGITS], size=batch[BatchKeys.IMAGES][:, 0].shape[2:],
                           mode="bilinear", align_corners=False, antialias=False)
    region = F.one_hot(logits.argmax(dim=1), num_classes=2).permute(0, 3, 1, 2)[0].bool()[1]
    explainer = build_explainer(name="signed_affinity", model=model,
                                params=AFFINITY_BY_MODEL.get(args.model, AFFINITY), device=device)
    explanation = explainer.explain(input_dict=batch, explanation_mask=region)[0]
    print(f"episode {index}: predicted region {int(region.sum())} px")

    for name, mode in (("iauc", "ins"), ("dauc", "del")):
        metric = FSSCausalMetric(model=model, mode=mode, n_steps=args.steps, measure="miou",
                                 n_mid_statuses=args.steps, mid_statuses_distribution="linear")
        for _, i, _ in metric.evaluate_interactive(batch, explanation, region, gt=gt):
            if i % 10 == 0:
                print(f"{name}: step {i}/{metric.computed_n_steps}", flush=True)
        scores = metric.scores
        pd.DataFrame(scores.numpy()).to_csv(os.path.join(args.out, f"{name}_curve.csv"), index=False)
        fig = curve_figure(scores[:, 0].numpy(), name, metric.xauc)
        for ext in ("svg", "pdf", "png"):
            fig.savefig(os.path.join(args.out, f"{name}_curve.{ext}"), bbox_inches="tight")
        plt.close(fig)
        for j, images, masks, probs, _ in metric.mid_statuses:
            prefix = os.path.join(args.out, f"{name}_step{j}")
            heatmap(probs[0, 1].cpu()).save(f"{prefix}_probs.png", compress_level=1)
            seg = tint_foreground(unnormalize(images[0, 0])[0].cpu(), probs.argmax(dim=1)[0].cpu().bool())
            to_pil(seg).save(f"{prefix}_segmentation.png", compress_level=1)
            for k in range(masks.shape[1]):
                shot = tint_foreground(unnormalize(images[0, 1 + k])[0].cpu(), masks[0, k, 1].cpu().bool())
                to_pil(shot).save(f"{prefix}_shot{k}.png", compress_level=1)
        print(f"{name}: AUC {100 * float(metric.xauc):.2f}, {len(metric.mid_statuses)} steps exported", flush=True)
    print(f"done: {args.out}")


if __name__ == "__main__":
    main()
