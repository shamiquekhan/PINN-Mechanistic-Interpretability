#!/usr/bin/env bash
# Create a reproducible environment for the PINN-MI framework.
#
# Usage:
#   bash scripts/setup_env.sh            # exact pinned environment (requirements.lock)
#   bash scripts/setup_env.sh --dev      # development minimums (requirements.txt)
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
VENV="${PINN_MI_VENV:-$ROOT/.venv}"

cd "$ROOT"

if [[ "${1:-}" == "--dev" ]]; then
  REQ="requirements.txt"
else
  REQ="requirements.lock"
fi

echo "==> Creating virtual environment at: $VENV"
python3 -m venv "$VENV"
# shellcheck disable=SC1091
source "$VENV/bin/activate"

echo "==> Upgrading pip"
pip install --upgrade pip

echo "==> Installing dependencies from $REQ"
if [[ "$REQ" == "requirements.lock" ]]; then
  # torch with CUDA 12.4 build lives on the PyTorch index; try the lock
  # first (works on machines with matching index config), fall back to the
  # documented two-step install.
  if ! pip install -r "$REQ"; then
    echo "==> Lock install failed (likely the +cu124 local version tag)."
    echo "    Installing torch from the PyTorch cu124 index, then the rest."
    pip install torch==2.6.0 --index-url https://download.pytorch.org/whl/cu124
    grep -v '^torch==' "$REQ" > /tmp/reqs_rest.txt
    pip install -r /tmp/reqs_rest.txt
  fi
else
  pip install -r "$REQ"
fi

pip install -e .

echo "==> Environment ready."
echo "    Activate with: source $VENV/bin/activate"
echo "    Verify:        python -m pytest tests/ -q   (139 tests)"
