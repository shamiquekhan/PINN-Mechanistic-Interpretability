"""
Physics-Feature Dictionary exporter — loads SAE features, associations,
and cross-seed matching results, then renders a human-readable markdown report.
"""
from __future__ import annotations
import json
from pathlib import Path
from typing import Dict, List, Optional


def render_feature_dictionary_markdown(dictionary: List[Dict], out_path: Path):
    lines = [
        "# Physics-Feature Dictionary",
        "",
        "| Feature ID | Layer | Freq | Stable | Top Correlation | Candidate Interpretation | Tier |",
        "|---|---|---|---|---|---|---|",
    ]

    for f in dictionary:
        fid = f.get("feature_id")
        layer = f.get("layer", "")
        freq = f.get("activation_frequency", 0.0)
        stable = "✓" if f.get("stable_across_seeds") else "✗"
        interp = f.get("candidate_interpretation", "unassigned")
        tier = f.get("confidence_tier", "candidate")

        corrs = f.get("correlations", {})
        top_corr_str = "none"
        if corrs:
            best_m, best_c = max(corrs.items(), key=lambda kv: abs(kv[1]), default=("none", 0.0))
            top_corr_str = f"{best_m} ({best_c:+.2f})"

        lines.append(f"| {fid} | {layer} | {freq:.3f} | {stable} | {top_corr_str} | {interp} | {tier} |")

    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text("\n".join(lines))
    print(f"Physics-Feature Dictionary Markdown report generated at {out_path}")
