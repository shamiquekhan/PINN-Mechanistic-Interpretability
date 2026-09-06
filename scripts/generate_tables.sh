#!/usr/bin/env bash
# Generate LaTeX publication tables from the committed runs/ artifacts.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

echo "==> Generating publication tables (paper/tables/)"
python scripts/generate_tables.py

echo "==> Tables written:"
ls -1 paper/tables/*.tex
