#!/usr/bin/env bash
# Fast CPU smoke test: unit tests + short training run + geometry analysis.
# Intended to run in < 5 minutes on CPU (this is what CI also runs).
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-}"  # CPU unless caller sets GPU

echo "==> [1/3] Unit tests (CPU)"
python -m pytest tests/ -q

echo "==> [2/3] Short boundary-starvation training run with logging"
SMOKE_DIR="runs/ci_smoke"
rm -rf "$SMOKE_DIR"
mkdir -p "$SMOKE_DIR"
python - <<EOF
import yaml, json
from pathlib import Path
cfg = yaml.safe_load(open("configs/failure_boundary_starvation.yaml"))
cfg["run"].update({"name": "ci_smoke", "output_dir": "$SMOKE_DIR",
                   "device": "cpu", "seed": 7})
cfg["training"].update({"steps": 60, "checkpoint_every": 50, "log_every": 10})
cfg["logging"].update({"save_activations": True, "log_gradients": False,
                       "log_diagnostics": False})
Path("$SMOKE_DIR").mkdir(parents=True, exist_ok=True)
Path("$SMOKE_DIR/ci_smoke.yaml").write_text(yaml.safe_dump(cfg))
EOF
python -m experiments.train --config "$SMOKE_DIR/ci_smoke.yaml" \
  --steps 60 --log-activations --act-save-raw

echo "==> [3/3] Effective-rank analysis on the smoke run"
python - <<EOF
from pathlib import Path
from analysis.effective_rank import analyze_run_activations
stats = analyze_run_activations(Path("$SMOKE_DIR/ci_smoke"), "layers.1")
assert stats and stats.get("n_samples", 0) > 0, "no activation stats"
assert "participation_ratio" in stats
print("smoke geometry:", {k: stats[k] for k in
      ("n_samples", "dim", "participation_ratio")})
EOF

echo "==> Cleaning smoke artifacts"
rm -rf "$SMOKE_DIR"

echo "==> SMOKE TEST PASSED"
