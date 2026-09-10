"""R4: SAE-seed robustness on the core Fourier condition (v4.1).

Preregistered in docs/preregistration.md §R4 (committed 2026-09-11,
BEFORE this run): 3 PINN seeds x 3 SAE seeds x {TopK, ReLU+L1} at
n_freq=32; Hungarian-matched cross-seed cosine (the PhysSAE §2.10
protocol) alongside reconstruction — separating PINN stochasticity
from SAE training stochasticity, and dictionary non-uniqueness from
regime-level verdict stability.

Pre-written decision rule: dictionary non-uniqueness (low cross-seed
cosine) WITH stable regime-level verdicts (reconstruction advantage,
and — if R2's battery is run per seed — its null) => the dissociation
is recorded: dictionary identity is not the unit of scientific claim;
the regime is.

Machinery note: the H18 checkpoints at n_freq=32 are the substrate
(runs/architecture_boundary/fourier_w64_seed*) — no retraining; the
SAE seeds are the only stochasticity under test.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, List

import numpy as np
import torch

from experiments.architecture_boundary import (
    WIDTH, _collect_representations,
)

PROJECT_ROOT = Path(__file__).resolve().parent.parent
RUNS = PROJECT_ROOT / "runs"
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

PINN_SEEDS = [7, 42, 123]
SAE_SEEDS = [0, 1, 2]


def _train_topk(train_reps, seed):
    from experiments.operator_boundary import train_topk_sae
    return train_topk_sae(train_reps, train_reps, expansion=4, topk=8,
                          steps=1500, seed=seed)["sae"]


def _train_relul1(train_reps, seed):
    from sae.model import SparseAutoencoder
    d_h = train_reps.shape[1]
    torch.manual_seed(seed)
    if DEVICE.type == "cuda":
        torch.cuda.manual_seed_all(seed)
    sae = SparseAutoencoder(
        input_dim=d_h, latent_expansion=512 // d_h, sparsity_coeff=0.02,
        activation_mode="relul1", topk=0, decoder_normalize=True).to(DEVICE)
    opt = torch.optim.Adam(sae.parameters(), lr=1e-3)
    bank = train_reps.to(DEVICE)
    n = bank.shape[0]
    rng = torch.Generator(device="cpu").manual_seed(seed + 777)
    for step in range(3000):
        idx = torch.randint(0, n, (min(4096, n),), generator=rng).to(DEVICE)
        opt.zero_grad()
        loss, _ = sae.loss(bank[idx])
        loss.backward()
        opt.step()
        sae._normalise_decoder()
    sae.eval()
    return sae


def _decoder_dirs(sae) -> np.ndarray:
    W = sae.W_d.weight.detach().cpu().numpy()      # (d, D)
    return W / (np.linalg.norm(W, axis=0, keepdims=True) + 1e-12)


def _hungarian_cosine(D1: np.ndarray, D2: np.ndarray) -> Dict:
    """Optimal-assignment matched cosine (PhysSAE §2.10 protocol)."""
    from scipy.optimize import linear_sum_assignment
    C = np.abs(D1.T @ D2)                            # (D1_cols, D2_cols)
    row, col = linear_sum_assignment(-C)
    matched = C[row, col]
    return {
        "mean_matched_cosine": float(matched.mean()),
        "median_matched_cosine": float(np.median(matched)),
        "frac_gt_0.7": float((matched > 0.7).mean()),
        "frac_gt_0.9": float((matched > 0.9).mean()),
    }


def _recon_metrics(sae, train_reps, test_reps) -> Dict:
    import torch.nn.functional as F
    with torch.no_grad():
        _, rec = sae(test_reps.to(DEVICE))
        mse = float(F.mse_loss(rec, test_reps.to(DEVICE)))
    return {"test_recon_mse": mse}


def _k_matched_pca(train_reps, test_reps) -> float:
    from experiments.operator_boundary import k_matched_pca_reconstruction
    return k_matched_pca_reconstruction(train_reps, test_reps, 8)


def run_r4(smoke: bool = False) -> Dict:
    print("\n=======================================================")
    print("R4: SAE-Seed Robustness on the Core Fourier Condition")
    print("=======================================================")

    sae_seeds = [0] if smoke else SAE_SEEDS
    out_root = RUNS / "r4_sae_seed_robustness"
    out_root.mkdir(parents=True, exist_ok=True)

    results: Dict[str, Dict] = {}
    for pin_seed in PINN_SEEDS:
        run_dir = RUNS / "architecture_boundary" / f"fourier_w64_seed{pin_seed}"
        if not (run_dir / "activations.jsonl").exists():
            print(f"  missing {run_dir} — skipping")
            continue
        reps = _collect_representations(run_dir)
        n = reps.shape[0]
        perm = torch.randperm(n, generator=torch.Generator()
                              .manual_seed(pin_seed))
        train_reps, test_reps = (reps[perm[:int(0.8 * n)]],
                                 reps[perm[int(0.8 * n):]])
        pca_mse = _k_matched_pca(train_reps, test_reps)

        for fam, trainer in [("topk", _train_topk),
                              ("relul1", _train_relul1)]:
            saes = {}
            for s in sae_seeds:
                print(f"  PINN seed {pin_seed} / {fam} / SAE seed {s}")
                saes[s] = trainer(train_reps, s)
            # pairwise Hungarian cosine between SAE seeds
            pair_stats = []
            keys = sorted(saes)
            for i in range(len(keys)):
                for j in range(i + 1, len(keys)):
                    D1, D2 = _decoder_dirs(saes[keys[i]]), \
                        _decoder_dirs(saes[keys[j]])
                    m = min(D1.shape[1], D2.shape[1])
                    pair_stats.append(
                        _hungarian_cosine(D1[:, :m], D2[:, :m]))
            # per-seed reconstruction verdicts
            recon = {str(s): _recon_metrics(saes[s], train_reps, test_reps)
                     for s in keys}
            ratios = [pca_mse / max(r["test_recon_mse"], 1e-12)
                      for r in recon.values()]
            entry = {
                "pca_k_mse": pca_mse,
                "per_sae_seed_recon": recon,
                "per_sae_seed_pca_over_sae_ratio": [round(x, 2)
                                                     for x in ratios],
                "sae_beats_pca_all_seeds": bool(all(x > 1 for x in ratios)),
                "ratio_mean": float(np.mean(ratios)),
                "ratio_std": float(np.std(ratios)),
                "cross_seed_cosine": pair_stats,
                "cross_seed_cosine_mean": (
                    float(np.mean([p["mean_matched_cosine"]
                                   for p in pair_stats]))
                    if pair_stats else None),
            }
            results.setdefault(fam, {})[str(pin_seed)] = entry

    # aggregate
    summary = {}
    for fam, per_pinn in results.items():
        ratios = [v["ratio_mean"] for v in per_pinn.values()]
        cosines = [v["cross_seed_cosine_mean"] for v in per_pinn.values()
                   if v["cross_seed_cosine_mean"] is not None]
        verdict_stable = all(v["sae_beats_pca_all_seeds"]
                             for v in per_pinn.values())
        summary[fam] = {
            "n_pinn_seeds": len(per_pinn),
            "ratio_mean_across_pinn": float(np.mean(ratios)),
            "ratio_min": float(np.min(ratios)),
            "ratio_max": float(np.max(ratios)),
            "mean_cross_sae_seed_cosine": (float(np.mean(cosines))
                                           if cosines else None),
            "reconstruction_verdict_stable": bool(verdict_stable),
            "dictionary_non_uniqueness_dissociation": bool(
                cosines and np.mean(cosines) < 0.5 and verdict_stable),
        }

    report = {
        "stage": "r4_sae_seed_robustness",
        "hypotheses": "R4 (docs/preregistration.md §R4; v4.1)",
        "protocol": {"pinn_seeds": PINN_SEEDS,
                     "sae_seeds": sae_seeds,
                     "families": ["topk_k8_exp4", "relul1_D512"],
                     "condition": "H18 Fourier n_freq=32 checkpoints",
                     "matching": "Hungarian (PhysSAE §2.10)"},
        "results": results,
        "summary": summary,
        "prewritten_rule_note": (
            "dictionary non-uniqueness (low cross-seed cosine) WITH stable "
            "regime-level verdicts => the dissociation is recorded: "
            "dictionary identity is not the unit of scientific claim; the "
            "regime is"),
    }
    out_path = out_root / "r4_report.json"
    out_path.write_text(json.dumps(report, indent=2, default=str))
    print("\nSummary:", json.dumps(summary, indent=1))
    print(f"R4 report: {out_path}")
    return report


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--smoke", action="store_true")
    args = ap.parse_args()
    run_r4(smoke=args.smoke)
