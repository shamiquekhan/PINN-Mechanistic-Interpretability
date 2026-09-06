"""Shared helpers for publication figures (no hand-entered numbers)."""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

PROJECT_ROOT = Path(__file__).resolve().parent.parent
RUNS = PROJECT_ROOT / "runs"
OUT_DIR = Path(__file__).resolve().parent / "generated"

# Shared style: modest, grayscale-safe, publication-oriented.
STYLE = {
    "figure.dpi": 160,
    "savefig.dpi": 300,
    "font.size": 9,
    "axes.titlesize": 10,
    "axes.labelsize": 9,
    "legend.fontsize": 8,
    "axes.spines.top": False,
    "axes.spines.right": False,
}

# Palette (colorblind-safe ordering).
C = {
    "main": "#1f77b4",
    "alt": "#d62728",
    "third": "#2ca02c",
    "fourth": "#9467bd",
    "grey": "#7f7f7f",
    "light": "#bbbbbb",
}


def load_artifact(rel_path: str) -> dict:
    """Load a JSON artifact or raise a clear error."""
    p = RUNS / rel_path
    if not p.exists():
        raise FileNotFoundError(
            f"Missing artifact {rel_path!r} — run the corresponding pipeline "
            "stage first (see docs/experiment_matrix.md)."
        )
    return json.loads(p.read_text())


def start_fig(nrows: int = 1, ncols: int = 1, figsize=(6.0, 3.2)):
    plt.rcParams.update(STYLE)
    fig, axes = plt.subplots(nrows, ncols, figsize=figsize)
    return fig, axes


def save_fig(fig, name: str):
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    out = OUT_DIR / f"{name}.png"
    fig.tight_layout()
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)
    print(f"  wrote {out.relative_to(PROJECT_ROOT)}")
    return out
