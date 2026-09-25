#!/usr/bin/env bash
# Rebuild the fixed episode lists in data_csv/.
#
# They are committed, so this is only needed to define new episode sets. Regenerating
# them draws different episodes and the numbers in the paper no longer apply.
set -e

for cfg in parameters/data/*.yaml; do
    uv run python main.py generate -p "$cfg"
done
