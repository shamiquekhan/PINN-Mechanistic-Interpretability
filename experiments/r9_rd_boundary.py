"""R9: Cross-PDE replication of the within-PINN boundary — steady 1D
reaction-diffusion (v4.2).

Preregistered in docs/preregistration.md §R9 (registered 2026-09-12,
BEFORE this implementation and any run). The R5 protocol transplanted
to a qualitatively different PDE family: steady 1D reaction-diffusion
(-eps u'' + mu u = f, eps=0.01, mu=1 — the stiff spatial-vs-reaction
regime), the review's Track B choice. Tanh vs Fourier (n_freq=32)
arms, width 64 / depth 3, 3 seeds, 2,000 steps (the H18/R3/R5 control
regime); per run: PR, rho, PCA-95, tangent rank, rel L2
(manufactured-solution reference), TopK-vs-k-matched-PCA(k=8)
reconstruction; the R2-style causal battery ONLY where PR crosses the
registered move-bar (3.0).

Pre-written outcomes: R9a replicates (Fourier PR >= 3.0 above tanh,
SAE beats PCA where PR moves) => three-family external validity
(steady-linear, time-dependent-nonlinear, steady-stiff). R9b does not
replicate => scope drawn honestly at the two families. R9c partial =>
recorded as-is.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, List

import numpy as np
import torch

from pinn.config import load_config
from pinn.model import MLP
from pinn.pdes import make_pde

PROJECT_ROOT = Path(__file__).resolve().parent.parent
RUNS = PROJECT_ROOT / "runs"
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

SEEDS = [7, 42, 123]
STEPS = 2000
WIDTH = 64
PR_MOVE_BAR = 3.0


def _rd_config(name: str, seed: int, fourier: bool,
               output_root: Path) -> Dict:
    """Steady 1D reaction-diffusion PINN (eps=0.01, mu=1 — the stiff
    regime from the stage-2 baseline family), width 64, depth 3 — the
    H18 shape. fourier_embed n_freq=32 applies to the 1-d input."""
    return {
        "run": {
            "name": name,
            "output_dir": str(output_root),
            "seed": seed,
            "deterministic": True,
            "device": "cuda" if DEVICE.type == "cuda" else "cpu",
            "dtype": "float32",
            "failure_label": "unlabeled",
        },
        "pde": {
            "name": "reaction_diffusion_1d",
            "domain": [-1.0, 1.0],
            # Non-degenerate spec (the fresh-campaign §3 fix class):
            # mu=1 (reaction_rate), f=1 (forcing) -> u_p = 1 with stiff
            # boundary layers at eps=0.01. NOTE: 'source' is the LEGACY
            # alias for reaction_rate — setting source with forcing=0
            # yields exact == 0 (degenerate), which the first smoke
            # execution caught (recorded in preregistration §R9).
            "source": 1.0,
            "forcing": 1.0,
            "diffusion": 0.01,
            "reaction_rate": 1.0,
            "boundary_values": [0.0, 0.0],
            "validation_points": 1001,
        },
        "model": {
            "input_dim": 1,
            "output_dim": 1,
            "hidden_layers": [WIDTH] * 3,
            "activation": "tanh",
            "init": "xavier",
            "fourier_embed": fourier,
            "fourier_n_freq": 32,
            "fourier_scale": 1.0,
        },
        "training": {
            "optimizer": "adam",
            "learning_rate": 1e-3,
            "steps": STEPS,
            "interior_points": 256,
            "boundary_points": 2,
            "log_every": 100,
            "checkpoint_every": STEPS,
            "lambda_pde": 1.0,
            "lambda_bc": 1.0,
            "resample_every": 0,
            "spatial_bias": 0.0,
        },
        "logging": {
            "save_activations": True,
            "activation_layers": [1],
            "save_pointwise_residuals": True,
            "log_gradients": False,
            "grad_log_every": 100,
            "log_diagnostics": False,
            "diag_log_every": 100,
        },
    }


def _rel_l2(model, pde) -> float:
    with torch.no_grad():
        n = 400
        x = torch.linspace(pde.left, pde.right, n, device=DEVICE) \
            .unsqueeze(1)
        pred = model(x)
        exact = pde.exact(x)
        return float(torch.linalg.vector_norm(pred - exact)
                     / torch.linalg.vector_norm(exact).clamp(min=1e-12))


def run_r9(seeds: List[int] = None, smoke: bool = False) -> Dict:
    print("\n" + "=" * 60)
    print("R9: Within-PINN Boundary on Reaction-Diffusion")
    print("=" * 60)

    seeds = seeds or ([7] if smoke else SEEDS)
    steps = 300 if smoke else STEPS
    out_root = RUNS / "r9_rd_boundary"
    out_root.mkdir(parents=True, exist_ok=True)

    from experiments.architecture_boundary import (
        _train_one, _analyze_run, _collect_representations,
        _reconstruction_comparison, WIDTH as H18W,
    )

    per_run = []
    for arm, fourier in [("fourier", True), ("tanh", False)]:
        for seed in seeds:
            name = f"r9_{arm}_seed{seed}"
            print(f"\n--- {name} ---")
            cfg = _rd_config(name, seed, fourier, out_root)
            cfg["training"]["steps"] = steps
            run_dir = _train_one(cfg, steps)

            cfg_obj = load_config(Path(cfg["run"]["output_dir"]) / name
                                  / "config.json")
            pde = make_pde(cfg_obj.pde)
            model = MLP(cfg_obj.model.input_dim, cfg_obj.model.output_dim,
                        cfg_obj.model.hidden_layers,
                        cfg_obj.model.activation,
                        fourier_embed=cfg_obj.model.fourier_embed,
                        fourier_n_freq=cfg_obj.model.fourier_n_freq,
                        fourier_seed=seed).to(DEVICE)
            ckpts = sorted(run_dir.glob("checkpoint_*.pt"))
            ckpt = torch.load(ckpts[-1], map_location=DEVICE,
                              weights_only=True)
            model.load_state_dict(ckpt["model"])
            model.eval()
            rel = _rel_l2(model, pde)
            print(f"  rel L2 (manufactured reference): {rel:.4f}")

            label = {"run": name, "arm": arm, "seed": seed,
                     "fourier_embed": fourier}
            stats = _analyze_run(run_dir, label)
            stats["rel_l2"] = rel
            pr = stats.get("participation_ratio")
            stats["pr_moved"] = bool(pr and pr > PR_MOVE_BAR)
            reps = _collect_representations(run_dir)
            stats["reconstruction"] = _reconstruction_comparison(reps, seed)
            print(f"  PR {pr:.2f} (rho {pr / H18W:.3f}) pca95 "
                  f"{stats['n_components_95pct']} tangent "
                  f"{stats['local_tangent_rank']} | PCA/SAE "
                  f"{stats['reconstruction']['pca_over_sae_ratio']:.2f}")
            per_run.append(stats)

    # ---- verdict (pre-written) ----
    by_arm: Dict[str, List[Dict]] = {}
    for r in per_run:
        by_arm.setdefault(r["arm"], []).append(r)
    fourier = by_arm.get("fourier", [])
    tanh = by_arm.get("tanh", [])
    f_pr = [r["participation_ratio"] for r in fourier]
    t_pr = [r["participation_ratio"] for r in tanh]
    f_ratio = [r["reconstruction"]["pca_over_sae_ratio"] for r in fourier]
    t_ratio = [r["reconstruction"]["pca_over_sae_ratio"] for r in tanh]
    f_crosses = bool(np.mean(f_pr) > PR_MOVE_BAR)
    f_above_t = bool(np.mean(f_pr) > np.mean(t_pr))
    sae_wins = bool(np.mean(f_ratio) > 1 and all(x > 1 for x in f_ratio))
    sae_wins_consistent = bool(all(x > 1 for x in f_ratio))

    if f_crosses and f_above_t and sae_wins:
        verdict = "R9a"
        note = ("the within-PINN boundary replicates on the steady-stiff "
                "family: the Fourier arm crosses into the high-rank regime "
                "above the tanh twin and the SAE beats k-matched PCA there "
                "— three-family external validity (steady-linear, "
                "time-dependent-nonlinear, steady-stiff)")
    elif f_crosses and f_above_t and not sae_wins_consistent:
        verdict = "R9c"
        note = ("partial: rank crosses but the reconstruction advantage is "
                "inconsistent across seeds — recorded as-is with per-seed "
                "numbers")
    else:
        verdict = "R9b"
        note = ("the boundary does not replicate on this family: the scope "
                "is drawn honestly at the families where it holds "
                "(Poisson, Burgers) + the operator regime; the "
                "geometry-vs-compression link is task-dependent — itself "
                "a recorded finding about which task structures enrich "
                "the covariance spectrum")

    report = {
        "stage": "r9_rd_boundary",
        "hypotheses": "R9 (docs/preregistration.md §R9; v4.2)",
        "protocol": {"seeds": seeds, "steps": steps, "width": WIDTH,
                     "depth": 3, "pde": "reaction_diffusion_1d (eps=0.01, "
                     "mu=1, manufactured reference)", "move_bar":
                     PR_MOVE_BAR},
        "per_run": per_run,
        "aggregate": {
            "fourier_mean_pr": float(np.mean(f_pr)) if f_pr else None,
            "tanh_mean_pr": float(np.mean(t_pr)) if t_pr else None,
            "fourier_mean_pca_over_sae": float(np.mean(f_ratio))
            if f_ratio else None,
            "tanh_mean_pca_over_sae": float(np.mean(t_ratio))
            if t_ratio else None,
            "fourier_crosses": f_crosses,
            "fourier_above_tanh": f_above_t,
            "sae_wins_fourier": sae_wins,
        },
        "verdict": verdict,
        "verdict_note": note,
    }
    out_path = out_root / "r9_report.json"
    out_path.write_text(json.dumps(report, indent=2, default=str))
    print(f"\nVerdict: {verdict} — {note}")
    print(f"R9 report: {out_path}")
    return report


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--seeds", type=str, default=None)
    args = ap.parse_args()
    seeds = [int(s) for s in args.seeds.split(",")] if args.seeds else None
    run_r9(seeds=seeds, smoke=args.smoke)
