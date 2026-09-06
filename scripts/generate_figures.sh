#!/usr/bin/env bash
# Generate all publication figures from the committed runs/ artifacts.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT/figures"

echo "==> Generating publication figures (figures/generated/)"
for f in figure_*.py; do
  echo "---- $f"
  python "$f"
done

echo "==> Figures written to figures/generated/"
ls -1 generated/*.png
