"""R1b reconciliation table — the watertight basis x criteria matrix.

Aggregates runs/r1_physSAE/r1_report.json (the committed R1 artifact)
into the side-by-side table the PhysSAE reconciliation requires: every
dictionary family x every causal-evidence criterion, on the same frozen
checkpoints, with the random-basis control column that the PhysSAE
battery does not run between bases.

Criteria (the level ladder, docs/final_claim_ladder_v4.md):
  L1 alignment      — max |r| with the physical-concept panel
                      (permutation-calibrated);
  L4 ESF80 conc.    — spatial concentration of the ablation footprint
                      vs PCA/ICA (PhysSAE's headline) AND vs random
                      directions (the between-basis control);
  L3 E_T specificity— matched-deletion effect-magnitude survivors
                      (Bonferroni/BH-FDR counts);
  L0 reconstruction — held-out reconstruction MSE (descriptive).

Output: runs/r1_physSAE/reconciliation_table.json (+ markdown to stdout).
Reads only the committed artifact — no recomputation.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Dict

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parent.parent
RUNS = PROJECT_ROOT / "runs"
ARTIFACT = RUNS / "r1_physSAE" / "r1_report.json"

BASIS_ORDER = [
    "relul1_sae_s0", "relul1_sae_s1", "relul1_sae_s2",
    "topk_sae", "pca", "ica", "random",
]
PRETTY = {
    "relul1_sae_s0": "ReLU+L1 SAE (s0)",
    "relul1_sae_s1": "ReLU+L1 SAE (s1)",
    "relul1_sae_s2": "ReLU+L1 SAE (s2)",
    "topk_sae": "TopK SAE",
    "pca": "PCA",
    "ica": "ICA",
    "random": "Random dirs",
}


def aggregate() -> Dict:
    r = json.loads(ARTIFACT.read_text())
    rows = {}
    for sub_name, sub in r["substrates"].items():
        if "seeds" not in sub:
            continue
        for seed, seed_data in sub["seeds"].items():
            if "evaluations" not in seed_data:
                continue
            for basis, ev in seed_data["evaluations"].items():
                et = ev["et_battery"]
                row = rows.setdefault(basis, {
                    "n_runs": 0,
                    "alignment_r": [],
                    "permutation_z": [],
                    "esf80_advantage": [],
                    "esf80_top": [],
                    "esf80_vs_pca": [],
                    "esf80_vs_random": [],
                    "bonferroni_survivors": [],
                    "bh_fdr_survivors": [],
                    "reconstruction_mse": [],
                })
                row["n_runs"] += 1
                row["alignment_r"].append(ev["max_alignment_r"])
                row["permutation_z"].append(ev["permutation_z"])
                row["esf80_advantage"].append(
                    ev["mean_esf80_advantage"])
                row["esf80_top"].append(np.mean(
                    [c["esf80_top"] for c in
                     ev["per_concept"].values()]))
                row["bonferroni_survivors"].append(
                    et["bonferroni_survivors"])
                row["bh_fdr_survivors"].append(et["bh_fdr_survivors"])
                row["reconstruction_mse"].append(
                    ev["reconstruction_eval_mse"])

    # between-basis ESF80 ratios need per-run pairing: SAE-vs-PCA and
    # SAE-vs-random on the SAME (substrate, seed) cell. Re-walk with the
    # cell index preserved.
    cell_esf = {}   # (sub, seed) -> {basis: mean esf80_top}
    for sub_name, sub in r["substrates"].items():
        if "seeds" not in sub:
            continue
        for seed, seed_data in sub["seeds"].items():
            if "evaluations" not in seed_data:
                continue
            cell_esf[(sub_name, seed)] = {
                b: float(np.mean([c["esf80_top"] for c in
                                  ev["per_concept"].values()]))
                for b, ev in seed_data["evaluations"].items()
            }

    summary = {}
    for basis, row in rows.items():
        n = row["n_runs"]
        # concentration ratios: lower ESF80 = more concentrated, so
        # ratio > 1 means the BASIS IN THE NUMERATOR is more concentrated
        vs_pca, vs_rand = [], []
        for (sub, seed), cells in cell_esf.items():
            if basis in cells and "pca" in cells:
                vs_pca.append(cells["pca"] / max(cells[basis], 1e-12))
            if basis in cells and "random" in cells:
                vs_rand.append(cells["random"] / max(cells[basis], 1e-12))
        summary[basis] = {
            "n_runs": n,
            "L1_alignment_max_abs_r": {
                "mean": float(np.mean(row["alignment_r"])),
                "min": float(np.min(row["alignment_r"])),
                "max": float(np.max(row["alignment_r"])),
            },
            "L1_permutation_z_min": float(np.min(row["permutation_z"])),
            "L4_esf80_footprint_mean": float(np.mean(row["esf80_top"])),
            "L4_concentration_vs_pca": (
                float(np.mean(vs_pca)) if vs_pca else None),
            "L4_concentration_vs_random": (
                float(np.mean(vs_rand)) if vs_rand else None),
            "L4_matched_atom_advantage_mean": float(
                np.mean(row["esf80_advantage"])),
            "L3_et_bonferroni_survivors_per_run": row[
                "bonferroni_survivors"],
            "L3_et_bh_fdr_survivors_per_run": row["bh_fdr_survivors"],
            "L3_et_total_survivors": int(np.sum(
                row["bonferroni_survivors"])),
            "L0_reconstruction_mse_mean": float(
                np.mean(row["reconstruction_mse"])),
        }
    return summary


def to_markdown(summary: Dict) -> str:
    hdr = ("| Basis | L1 max \\|r\\| (min–max) | L4 ESF80 footprint | "
           "conc. vs PCA | conc. vs random | L3 E_T Bonf survivors "
           "(per run) | L3 total | L0 recon MSE |")
    sep = "|---|---|---:|---:|---:|---|---:|---:|"
    lines = [hdr, sep]
    for b in BASIS_ORDER:
        if b not in summary:
            continue
        s = summary[b]
        al = s["L1_alignment_max_abs_r"]
        runs = s["L3_et_bonferroni_survivors_per_run"]
        conc_pca = s["L4_concentration_vs_pca"]
        conc_rnd = s["L4_concentration_vs_random"]
        cpca = f"{conc_pca:.2f}x" if conc_pca is not None else "—"
        crnd = f"{conc_rnd:.2f}x" if conc_rnd is not None else "—"
        lines.append(
            f"| {PRETTY[b]} "
            f"| {al['min']:.2f}–{al['max']:.2f} "
            f"| {s['L4_esf80_footprint_mean']:.3f} "
            f"| {cpca} "
            f"| {crnd} "
            f"| {runs} "
            f"| {s['L3_et_total_survivors']} "
            f"| {s['L0_reconstruction_mse_mean']:.2e} |")
    return "\n".join(lines)


if __name__ == "__main__":
    summary = aggregate()
    out = {
        "stage": "r1_reconciliation_table",
        "source_artifact": "runs/r1_physSAE/r1_report.json",
        "n_runs_per_basis": {b: s["n_runs"] for b, s in summary.items()},
        "criteria": {
            "L1_alignment": "max |r| with the physical-concept panel "
                            "(permutation-null calibrated)",
            "L4_esf80": "spatial concentration of the ablation "
                        "footprint; footprint = mean ESF80 of the "
                        "top atom's effect field; concentration ratios "
                        ">1 mean the numerator basis is more "
                        "concentrated",
            "L3_et": "matched-deletion effect-magnitude battery, "
                     "Bonferroni/BH survivors out of 8 candidates "
                     "per run",
            "L0_recon": "held-out reconstruction MSE (descriptive)",
        },
        "table": summary,
    }
    out_path = RUNS / "r1_physSAE" / "reconciliation_table.json"
    out_path.write_text(json.dumps(out, indent=2))
    print(f"wrote {out_path}")
    print()
    print(to_markdown(summary))
