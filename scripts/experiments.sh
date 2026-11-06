#!/usr/bin/env bash
# Every experiment in the paper, in the order the tables appear.
#
# Each line is independent, so a single table can be reproduced by running its group
# alone. Weights and datasets are fetched on first use, except the two licence-gated
# checkpoints (DINOv3 for INSID3, the SANSA adapter), which have to be accepted and
# placed by hand; those runs fail with instructions if they are missing.
set -e

run() { uv run python main.py grid --parameters "$1"; }

# Tables 1-3: mIoULoss@p for every model and attribution method. DCAMA and DMTNet share
# parameters/pascal and parameters/coco, which also hold their Table 6 runs; then INSID3 and
# SANSA (Table 1), the PANet prototype head (Table 2, trained first with
# scripts/train_panet.py), GF-SAM and Matcher (Table 3).
for cfg in parameters/pascal/*.yaml parameters/coco/*.yaml; do
    run "$cfg"
done
for model in insid3 sansa panet gfsam matcher; do
    for cfg in parameters/$model/*.yaml; do
        run "$cfg"
    done
done

# Table 6 (Appendix D): insertion and deletion curves of the other models.
for cfg in parameters/*/insertion_deletion/*.yaml; do
    run "$cfg"
done

# Gradient baselines, which only the differentiable models support.
for cfg in parameters/*/gradients/*.yaml; do
    run "$cfg"
done

# Appendix: ablations and the per-step curves.
for cfg in parameters/ablation/*.yaml parameters/curves/*.yaml; do
    run "$cfg"
done

# Appendix I: top-k push-down of the in-mask similarities, then the summary table.
for model in dcama insid3 gfsam matcher; do
    uv run python scripts/affinity_topk_intervention.py --model "$model"
done
uv run python scripts/summarize_affinity_topk.py

# Computational cost (Appendix A).
for cfg in parameters/computational/*.yaml; do
    uv run python main.py grid --parameters "$cfg" --function computational
done
