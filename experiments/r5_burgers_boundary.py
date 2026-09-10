"""R5: Within-PINN boundary on a second PDE — Burgers tanh vs Fourier (v4.1).

Preregistered in docs/preregistration.md §R5 (committed 2026-09-11,
BEFORE this run): tanh vs Fourier (n_freq=32) on one time-dependent
family — Burgers, chosen because it spans partial convergence and has
localized concepts (the shock) matching the PhysSAE concept panel.
3 seeds, both SAE families, rank diagnostic + reconstruction; the R2
battery wherever PR moves.

Pre-written decision rule: replicates (the Fourier arm crosses into
the high-rank regime and the SAE beats k-matched PCA there) => the
within-PINN boundary has external validity beyond 1D Poisson. Does not
=> scope drawn at steady tasks + the operator regime, recorded
honestly.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, List

import numpy as np
import torch

from pinn.config import PDEConfig, load_config
from pinn.model import MLP
from pinn.pdes import make_pde
from pinn.reproducibility import set_seed

PROJECT_ROOT = Path(__file__).resolve().parent.parent
RUNS = PROJECT_ROOT / "runs"
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

SEEDS = [7, 42, 123]
STEPS = 2000
WIDTH = 64


def _burgers_config(name: str, seed: int, fourier: bool,
                    output_root: Path) -> Dict:
    """Burgers (t,x) PINN, width 64, depth 3 — the H18 shape on a
    time-dependent family.  fourier_embed n_freq=32 applies to the
    2-d input (t, x): the embedding sees both coordinates."""
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
            "name": "burgers_1d",
            "domain": [-1.0, 1.0],
            "time_domain": [0.0, 1.0],
            "source": 1.0,
            "forcing": 0.0,
            "diffusion": 0.01,
            "boundary_values": [0.0, 0.0],
            "validation_points": 400,
        },
        "model": {
            "input_dim": 2,
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
            "boundary_points": 66,
            "log_every": 100,
            "checkpoint_every": STEPS,
            "lambda_pde": 1.0,
            "lambda_bc": 100.0,
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
        n = 200
        t = torch.linspace(pde.t0, pde.tT, n, device=DEVICE)
        x = torch.linspace(pde.left, pde.right, n, device=DEVICE)
        gt, gx = torch.meshgrid(t, x, indexing="ij")
        xv = torch.stack([gt.flatten(), gx.flatten()], dim=1)
        pred = model(xv)
        exact = pde.exact(xv)
        return float(torch.linalg.vector_norm(pred - exact)
                     / torch.linalg.vector_norm(exact).clamp(min=1e-12))


def run_r5(seeds: List[int] = None, smoke: bool = False) -> Dict:
    print("\n=======================================================")
    print("R5: Within-PINN Boundary on Burgers (tanh vs Fourier)")
    print("=======================================================")

    seeds = seeds or ([7] if smoke else SEEDS)
    steps = 300 if smoke else STEPS
    out_root = RUNS / "r5_burgers_boundary"
    out_root.mkdir(parents=True, exist_ok=True)

    from experiments.architecture_boundary import (
        _train_one, _analyze_run, _collect_representations,
        _reconstruction_comparison, WIDTH as H18W,
    )
    # _analyze_run/_reconstruction_comparison use layers.1 — compatible
    # (width-64 layers.1 regardless of the 2-d input or embedding).

    per_run = []
    for arm, fourier in [("fourier", True), ("tanh", False)]:
        for seed in seeds:
            name = f"r5_{arm}_seed{seed}"
            print(f"\n--- {name} ---")
            cfg = _burgers_config(name, seed, fourier, out_root)
            cfg["training"]["steps"] = steps
            run_dir = _train_one(cfg, steps)

            # rel L2 vs the independent spectral reference
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
            print(f"  space-time rel L2: {rel:.4f}")

            label = {"run": name, "arm": arm, "seed": seed,
                     "fourier_embed": fourier}
            stats = _analyze_run(run_dir, label)
            stats["rel_l2"] = rel
            pr = stats.get("participation_ratio")
            stats["pr_moved"] = bool(pr and pr > 3.0)
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
    crosses = bool(np.mean(f_pr) > 3.0 and np.mean(t_pr) < 3.0)
    sae_wins = bool(np.mean(f_ratio) > 1 and all(x > 1 for x in f_ratio))
    if crosses and sae_wins:
        verdict = "R5a"
        note = ("the within-PINN boundary replicates on a time-dependent "
                "family: the Fourier arm crosses into the high-rank regime "
                "and the SAE beats k-matched PCA there")
    elif crosses and not sae_wins:
        verdict = "R5b-partial"
        note = ("rank crosses but the reconstruction advantage does not "
                "transfer to this family — geometry necessary, not "
                "sufficient for the compression advantage either; recorded "
                "as-is")
    else:
        verdict = "R5b"
        note = ("the boundary does not replicate on this family: scope is "
                "drawn at steady tasks + the operator regime, recorded "
                "honestly")

    report = {
        "stage": "r5_burgers_boundary",
        "hypotheses": "R5 (docs/preregistration.md §R5; v4.1)",
        "protocol": {"seeds": seeds, "steps": steps, "width": WIDTH,
                     "depth": 3, "pde": "burgers_1d (viscous 0.01, "
                     "spectral reference)", "move_bar": 3.0},
        "per_run": per_run,
        "aggregate": {
            "fourier_mean_pr": float(np.mean(f_pr)) if f_pr else None,
            "tanh_mean_pr": float(np.mean(t_pr)) if t_pr else None,
            "fourier_mean_pca_over_sae": float(np.mean(f_ratio))
            if f_ratio else None,
            "tanh_mean_pca_over_sae": float(np.mean(t_ratio))
            if t_ratio else None,
            "fourier_crosses": crosses,
            "sae_wins_fourier": sae_wins,
        },
        "verdict": verdict,
        "verdict_note": note,
    }
    out_path = out_root / "r5_report.json"
    out_path.write_text(json.dumps(report, indent=2, default=str))
    print(f"\nVerdict: {verdict} — {note}")
    print(f"R5 report: {out_path}")
    return report


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--seeds", type=str, default=None)
    args = ap.parse_args()
    seeds = [int(s) for s in args.seeds.split(",")] if args.seeds else None
    run_r5(seeds=seeds, smoke=args.smoke)
