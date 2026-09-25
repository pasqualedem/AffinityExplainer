"""Self-agreement control for the affinity-noise intervention.

Matcher picks its SAM point prompts with random.sample (Matcher.py, iterative prompt
search), which the intervention runner never seeds, so two clean forwards on the same
episode need not agree. Any agreement drop under affinity noise must be read against
this floor: a model whose tau=0 self-agreement is already 0.8 has not been shown to be
sensitive to the noise when it reaches 0.8 at tau=0.25.

Runs the same 60 episodes twice at tau=0 and reports mean IoU between the two clean
predictions. Writes out/affinity_noise/<model>_<csv>_control.csv.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))  # repo root, for `affex`
import argparse
import csv
import os

import torch
import torch.nn.functional as F
from tqdm import tqdm

from affex.data import get_dataloaders
from affex.models import build_model_preconfigured
from affex.substitution import Substitutor
from affex.utils.torch import to_device

from affinity_noise_intervention import dataset_params, set_noise, fg_pred


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--episodes-csv", default="val_pascal5i_N1K1_60")
    ap.add_argument("--out", default="out/affinity_noise")
    args = ap.parse_args()
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
    out_csv = os.path.join(args.out, f"{args.model}_{args.episodes_csv}_control.csv")
    rows = []
    for i, batch in tqdm(enumerate(loader), total=len(loader), desc=f"{args.model} control"):
        batch, _ = batch
        substitutor = Substitutor(substitute=False)
        substitutor.reset(batch=batch)
        input_dict, gt = next(substitutor)
        input_dict = to_device(input_dict, device)
        gt = to_device(gt, device)

        set_noise(model, args.model, 0.0)
        preds = []
        for rep in range(2):
            # Same torch seeding as the intervention run at tau=0; the python RNG is
            # left alone, exactly as in the experiment being controlled.
            torch.manual_seed(1000 * i)
            preds.append(fg_pred(model, input_dict))
        a, b = preds
        inter = ((a == 1) & (b == 1)).sum().item()
        union = ((a == 1) | (b == 1)).sum().item()
        rows.append({"episode": i, "self_agreement": inter / union if union > 0 else 1.0})

    with open(out_csv, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["episode", "self_agreement"])
        w.writeheader()
        w.writerows(rows)

    mean = sum(r["self_agreement"] for r in rows) / len(rows)
    print(f"\n{args.model} ({args.episodes_csv}): tau=0 self-agreement={mean:.4f} n={len(rows)}")


if __name__ == "__main__":
    main()
