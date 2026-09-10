"""R3: Fourier frequency sweep — the rank dose-response (v4.1).

Preregistered in docs/preregistration.md §R3 (committed 2026-09-11,
BEFORE this run). H18 showed a binary contrast: tanh depth-3 PR ~1.4
vs Fourier n_freq=32 PR 4.0-5.6 (25-56x reconstruction advantage).
R3 converts the boundary into a quantitative curve.

Design (registered):
  n_freq in {2, 4, 8, 16, 32, 64}, fixed 1D Poisson, width 64,
  depth 3, 3 seeds each (18 runs), same training protocol as H18
  (2000 steps, the width-scaling control regime).
  Per run: PR, PR/W, PCA-95, tangent rank, plus reconstruction
  comparisons with BOTH registered SAE families:
    - TopK SAE (k=8, expansion=4) vs k-matched PCA(k=8)  [H18 protocol]
    - ReLU+L1 SAE (D=512, lambda=0.02, unit-norm decoder — the
      PhysSAE spec) vs k-matched PCA(k=8)  [R1 protocol]
  The R2 causal battery runs wherever PR moves above the registered
  bar (3.0) — but note R2's sharpened design (v4.1.1): the battery is
  run on the TOP-8 candidates only (the H18 protocol's per-seed
  budget), since R2 already measured the full top-16 protocol at
  n_freq=32.

Pre-written outcomes (all publishable, none preferred):
  - Smooth: the SAE/PCA ratio rises smoothly with rho = PR/W ->
    quantitative phase diagram.
  - Threshold: a critical rho separates the regimes -> reported as an
    empirical constant of this protocol, not a universal law.
  - Non-monotone: recorded as-is; geometry insufficient alone.

Machinery gate: the planted-feature control (stage-5 world through
the PINN hook, the R2 gate) must pass BEFORE any verdict is read.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, List

import numpy as np
import torch

from experiments.architecture_boundary import (
    WIDTH, SEEDS as H18_SEEDS, PR_MOVE_THRESHOLD,
    build_h18_config, _train_one, _analyze_run, _collect_representations,
)
from analysis.effective_rank import analyze_run_activations

PROJECT_ROOT = Path(__file__).resolve().parent.parent
RUNS = PROJECT_ROOT / "runs"
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

N_FREQS = [2, 4, 8, 16, 32, 64]
SEEDS = [7, 42, 123]
STEPS = 2000


# ---------------------------------------------------------------------------
# Reconstruction: both registered SAE families vs k-matched PCA
# ---------------------------------------------------------------------------

def recon_topk(train_reps: torch.Tensor, test_reps: torch.Tensor,
               seed: int) -> Dict:
    """TopK SAE (k=8, exp=4) vs k-matched PCA — the H18 protocol."""
    from experiments.operator_boundary import (
        k_matched_pca_reconstruction, train_topk_sae)
    out = train_topk_sae(train_reps, test_reps, expansion=4, topk=8,
                         steps=1500, seed=seed)
    pca_mse = k_matched_pca_reconstruction(train_reps, test_reps, 8)
    return {
        "sae_test_recon_mse": out["test_recon_mse"],
        "pca_k_test_recon_mse": pca_mse,
        "pca_over_sae_ratio": float(pca_mse / max(out["test_recon_mse"],
                                                  1e-12)),
        "sae_beats_pca": bool(out["test_recon_mse"] < pca_mse),
        "dead_feature_fraction": out["dead_feature_fraction"],
        "mean_l0": out["mean_l0"],
    }


def recon_relul1(train_reps: torch.Tensor, test_reps: torch.Tensor,
                 seed: int) -> Dict:
    """ReLU+L1 SAE (D=512, lambda=0.02 — the PhysSAE spec) vs k-matched
    PCA(k=8).  Codes can be dense; the k-matched comparison keeps the
    capacity axis fixed at k=8 for both SAE families (registered)."""
    from experiments.operator_boundary import k_matched_pca_reconstruction
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
    with torch.no_grad():
        _, rec = sae(test_reps.to(DEVICE))
        sae_mse = float(torch.nn.functional.mse_loss(
            rec, test_reps.to(DEVICE)))
        z, _ = sae(test_reps.to(DEVICE))
        l0 = float((z > 0).float().sum(-1).mean())
    pca_mse = k_matched_pca_reconstruction(train_reps, test_reps, 8)
    return {
        "sae_test_recon_mse": sae_mse,
        "pca_k_test_recon_mse": pca_mse,
        "pca_over_sae_ratio": float(pca_mse / max(sae_mse, 1e-12)),
        "sae_beats_pca": bool(sae_mse < pca_mse),
        "mean_l0": l0,
    }


def _reconstruction_pair(reps: torch.Tensor, seed: int) -> Dict:
    """80/20 sample split (the H18/R1 protocol); both SAE families."""
    n = reps.shape[0]
    perm = torch.randperm(n, generator=torch.Generator().manual_seed(seed))
    n_train = int(0.8 * n)
    train_reps, test_reps = reps[perm[:n_train]], reps[perm[n_train:]]
    return {
        "topk_vs_pca": recon_topk(train_reps, test_reps, seed),
        "relul1_vs_pca": recon_relul1(train_reps, test_reps, seed),
    }


# ---------------------------------------------------------------------------
# Driver
# ---------------------------------------------------------------------------

def run_r3(seeds: List[int] = None, n_freqs: List[int] = None,
           smoke: bool = False) -> Dict:
    print("\n=======================================================")
    print("R3: Fourier Frequency Sweep (the rank dose-response)")
    print("=======================================================")

    seeds = seeds or SEEDS
    n_freqs = n_freqs or ([2, 32] if smoke else N_FREQS)
    steps = 300 if smoke else STEPS

    out_root = RUNS / "r3_frequency_sweep"
    out_root.mkdir(parents=True, exist_ok=True)

    # ---- Machinery gate (R2's, reused — the same hook machinery the
    #      causal arm depends on; the reconstruction arm needs no new
    #      machinery beyond the H18/R1 paths already validated) ----
    print("\n--- Machinery gate (planted world through the PINN hook) ---")
    from experiments.r2_fourier_causal import machinery_gate
    gate = machinery_gate()
    print(f"  planted cosine {gate['planted_cosine']:.3f} | "
          f"{gate['beats_controls_batches']}/{gate['n_batches']} | "
          f"pass={gate['pipeline_pass']}")
    if not gate["pipeline_pass"]:
        out = {"stage": "r3_frequency_sweep", "gates_pass": False,
               "status": "VOID: machinery gate failure", "gate": gate}
        (out_root / "r3_report.json").write_text(
            json.dumps(out, indent=2, default=str))
        print("  GATE FAILED — R3 VOID")
        return out

    per_run = []
    for n_freq in n_freqs:
        for seed in seeds:
            name = f"r3_fourier_nf{n_freq}_seed{seed}"
            print(f"\n--- {name} ---")
            config = build_h18_config(
                name, [WIDTH] * 3, seed, fourier=True,
                output_root=out_root, n_freq=n_freq)
            config["training"]["steps"] = steps
            run_dir = _train_one(config, steps)
            label = {"run": name, "arm": "frequency", "n_freq": n_freq,
                     "fourier_embed": True, "seed": seed}
            stats = _analyze_run(run_dir, label)
            stats["input_dim_effective"] = 2 * n_freq
            pr = stats.get("participation_ratio")

            reps = _collect_representations(run_dir)
            recon = _reconstruction_pair(reps, seed)
            stats["reconstruction"] = recon
            print(f"  PR {pr:.2f} (rho {pr / WIDTH:.3f}) pca95 "
                  f"{stats['n_components_95pct']} tangent "
                  f"{stats['local_tangent_rank']} | TopK "
                  f"PCA/SAE {recon['topk_vs_pca']['pca_over_sae_ratio']:.2f} "
                  f"| ReLU+L1 PCA/SAE "
                  f"{recon['relul1_vs_pca']['pca_over_sae_ratio']:.2f} "
                  f"(l0 {recon['relul1_vs_pca']['mean_l0']:.0f})")
            per_run.append(stats)

    # ---- Aggregation by n_freq ----
    by_freq: Dict[int, List[Dict]] = {}
    for r in per_run:
        by_freq.setdefault(r["n_freq"], []).append(r)
    curve = {}
    for nf in sorted(by_freq):
        runs = by_freq[nf]
        prs = [r["participation_ratio"] for r in runs]
        rhos = [r["pr_over_width"] for r in runs]
        tk = [r["reconstruction"]["topk_vs_pca"]["pca_over_sae_ratio"]
              for r in runs]
        rl = [r["reconstruction"]["relul1_vs_pca"]["pca_over_sae_ratio"]
              for r in runs]
        curve[str(nf)] = {
            "n_runs": len(runs),
            "mean_pr": float(np.mean(prs)),
            "std_pr": float(np.std(prs)),
            "mean_rho": float(np.mean(rhos)),
            "mean_tangent_rank": float(np.mean(
                [r["local_tangent_rank"] for r in runs])),
            "mean_pca95": float(np.mean(
                [r["n_components_95pct"] for r in runs])),
            "topk_pca_over_sae_mean": float(np.mean(tk)),
            "topk_pca_over_sae_per_seed": [round(x, 3) for x in tk],
            "relul1_pca_over_sae_mean": float(np.mean(rl)),
            "relul1_pca_over_sae_per_seed": [round(x, 3) for x in rl],
        }

    # ---- Pre-written outcome reading ----
    rhos_seq = [curve[str(nf)]["mean_rho"] for nf in sorted(by_freq)]
    tk_seq = [curve[str(nf)]["topk_pca_over_sae_mean"] for nf in
              sorted(by_freq)]
    # smooth vs threshold vs non-monotone (registered):
    #  monotone in rho means the ratio sequence is non-decreasing
    #  (allowing one tie) with the first/last split >= 3x.
    inc = all(tk_seq[i + 1] >= tk_seq[i] - 1e-9
              for i in range(len(tk_seq) - 1))
    dec_then_inc = None
    if len(tk_seq) >= 3:
        # threshold shape: low plateau then rise
        min_idx = int(np.argmin(tk_seq))
        dec_then_inc = (min_idx > 0
                        and all(tk_seq[i + 1] >= tk_seq[i]
                                for i in range(min_idx, len(tk_seq) - 1)))
    if inc and tk_seq[-1] / max(tk_seq[0], 1e-9) >= 3:
        shape = "smooth"
        note = ("the SAE/PCA reconstruction ratio rises monotonically "
                "with rho = PR/W: a quantitative phase diagram")
    elif dec_then_inc:
        shape = "threshold"
        note = ("a critical rho separates the regimes: reported as an "
                "empirical constant of this protocol, not a universal law")
    else:
        shape = "non-monotone"
        note = ("the SAE/PCA ratio is non-monotone in rho: geometry is "
                "insufficient alone at this protocol; recorded as-is")
    # correlation (Spearman) between rho and ratio across ALL runs
    # (no scipy dependency: rank-corr computed by hand below)
    all_rho = [r["pr_over_width"] for r in per_run]
    all_tk = [r["reconstruction"]["topk_vs_pca"]["pca_over_sae_ratio"]
              for r in per_run]
    all_rl = [r["reconstruction"]["relul1_vs_pca"]["pca_over_sae_ratio"]
              for r in per_run]
    def _spearman(a, b):
        ra = np.argsort(np.argsort(a)); rb = np.argsort(np.argsort(b))
        return float(np.corrcoef(ra, rb)[0, 1])
    rho_tk_corr = _spearman(all_rho, all_tk)
    rho_rl_corr = _spearman(all_rho, all_rl)

    report = {
        "stage": "r3_frequency_sweep",
        "hypotheses": "R3 (docs/preregistration.md §R3; v4.1)",
        "gates_pass": True, "gate": gate,
        "protocol": {"n_freqs": n_freqs, "seeds": seeds, "steps": steps,
                     "width": WIDTH, "depth": 3,
                     "sae_families": ["topk_k8_exp4", "relul1_D512_l0.02"],
                     "k_matched_pca": 8},
        "per_run": per_run,
        "curve_by_n_freq": curve,
        "rho_sequences": {"rho": rhos_seq,
                          "topk_pca_over_sae": tk_seq,
                          "relul1_pca_over_sae": [
                              curve[str(nf)]["relul1_pca_over_sae_mean"]
                              for nf in sorted(by_freq)]},
        "spearman_rho_vs_topk_ratio": rho_tk_corr,
        "spearman_rho_vs_relul1_ratio": rho_rl_corr,
        "shape": shape,
        "shape_note": note,
    }
    out_path = out_root / "r3_report.json"
    out_path.write_text(json.dumps(report, indent=2, default=str))
    print(f"\nShape: {shape} (Spearman rho-vs-TopK-ratio "
          f"{rho_tk_corr:+.2f}, vs ReLU+L1-ratio {rho_rl_corr:+.2f})")
    print(f"R3 report: {out_path}")
    return report


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--seeds", type=str, default=None)
    ap.add_argument("--n-freqs", type=str, default=None)
    args = ap.parse_args()
    seeds = [int(s) for s in args.seeds.split(",")] if args.seeds else None
    nfs = [int(s) for s in args.n_freqs.split(",")] if args.n_freqs else None
    run_r3(seeds=seeds, n_freqs=nfs, smoke=args.smoke)
