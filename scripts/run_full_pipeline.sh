#!/usr/bin/env bash
# Full 15-stage research campaign. Heavy: ~6 GPU-hours on a 4 GB GPU
# (stage 1 retrains every configuration from scratch).
#
# For reproducing the published analyses from the committed runs/ artifacts
# without retraining, use scripts/reproduce_main_results.sh instead.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

echo "This retrains EVERYTHING (~6 GPU-hours). Continue? [y/N]"
read -r reply
if [[ "$reply" != "y" && "$reply" != "Y" ]]; then
  echo "Aborted. Use scripts/reproduce_main_results.sh to analyze existing artifacts."
  exit 0
fi

python -m experiments.run_pipeline
