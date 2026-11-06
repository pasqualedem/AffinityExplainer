"""Summarize the top-k push-down intervention (scripts/affinity_topk_intervention.py).

Merges the shards of each model and prints, per drop size tau and fraction kappa, the
paired change in mIoU against the same episode's unperturbed prediction (mean and
standard error over episodes), the mean agreement, and the episode count.

Usage: .venv/bin/python scripts/summarize_affinity_topk.py [--episodes-csv val_pascal5i_N1K1]
           [--models dcama,insid3,gfsam,matcher] [--out out/affinity_topk]
"""
import argparse
import glob
import os

import numpy as np
import pandas as pd


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--episodes-csv", default="val_pascal5i_N1K1")
    ap.add_argument("--models", default="dcama,insid3,gfsam,matcher")
    ap.add_argument("--out", default="out/affinity_topk")
    args = ap.parse_args()

    for model in args.models.split(","):
        files = glob.glob(os.path.join(args.out, f"{model}_{args.episodes_csv}_shard*of*.csv"))
        if not files:
            print(f"{model}: no results")
            continue
        d = pd.concat([pd.read_csv(f) for f in files]).drop_duplicates(["episode", "tau", "kappa"])
        base = d[d.kappa == 0].set_index("episode").miou
        print(f"\n{model}: {base.size} episodes, clean mIoU {base.mean():.3f}")
        for tau in sorted(t for t in d.tau.unique() if t > 0):
            for kappa in sorted(k for k in d.kappa.unique() if k > 0):
                sel = d[(d.tau == tau) & (d.kappa == kappa)].set_index("episode")
                delta = (sel.miou - base.reindex(sel.index)).dropna()
                se = delta.std(ddof=1) / np.sqrt(len(delta))
                print(f"  tau={tau:g} kappa={kappa:<5g} dmIoU={delta.mean():+.3f} +- {se:.3f}"
                      f"  agreement={sel.agreement.mean():.3f}  n={len(delta)}")


if __name__ == "__main__":
    main()
