#!/usr/bin/env bash
# Every experiment in the paper, in the order the tables appear.
#
# Each line is independent, so a single table can be reproduced by running its group
# alone. Weights and datasets are fetched on first use, except the two licence-gated
# checkpoints (DINOv3 for INSID3, the SANSA adapter), which have to be accepted and
# placed by hand; those runs fail with instructions if they are missing.
set -e

run() { uv run python main.py grid --parameters "$1"; }

# Table 1: mIoULoss@p, every model and every attribution method.
for model in dcama dmtnet panet insid3 sansa gfsam matcher; do
    for cfg in parameters/$model/*.yaml; do
        [ -e "$cfg" ] && run "$cfg"
    done
done
for cfg in parameters/pascal/*.yaml parameters/coco/*.yaml; do
    run "$cfg"
done

# Table 5: insertion and deletion curves.
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

# Computational cost.
for cfg in parameters/computational/*.yaml; do
    uv run python main.py grid --parameters "$cfg" --function computational
done
