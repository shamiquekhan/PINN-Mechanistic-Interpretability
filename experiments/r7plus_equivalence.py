"""R7+: power-upgraded SAE/PCA equivalence replication (v4.2).

Preregistered in docs/preregistration.md §R7+ (registered 2026-09-12,
AFTER R8's power calculation, before any training). R7b was
inconclusive: the paired SAE-PCA causal-strength difference (+0.0042)
lies inside the preregistered delta=0.01 margin, but the 90% CI
straddles the boundary. R8's cluster-level power calculation: 80%
power at delta=0.01 needs 163 checkpoints — infeasible in one session
on this hardware (GTX 1650), so the REGISTERED FALLBACK applies: an
intermediate n of new boundary-starvation models (the next seeds in
the established block), the IDENTICAL stage-5/8 batteries (same SAE,
same probe directions, same candidates, same controls), then the TOST
on the COMBINED pair set with the R8 cluster bootstrap as the primary
interval — and the ACHIEVED power reported, never read as equivalence
if inconclusive.

Intermediate n (registered): 20 new seeds (combined 31 checkpoints,
~25% of the 163 for 80% power; ~2x the current 11). Achieved power
is computed and reported from the combined cluster SD.

Machinery gate: the planted-feature positive control must pass at the
combined scale before any verdict (the stage-5/R2 gate).
"""
from __future__ import annotations

import json
import math
from collections import defaultdict
from pathlib import Path
from typing import Dict, List

import numpy as np
import torch

from experiments.run_pipeline import DEVICE, RUNS

PROJECT_ROOT = Path(__file__).resolve().parent.parent

NEW_SEEDS = [11, 22, 33, 44, 55, 66, 88, 99, 111, 222,
             333, 444, 5555, 6666, 7777, 8888, 9999, 1234, 2345, 3456]
STEPS = 5000
DELTA = 0.01
ALPHA = 0.05
N_BOOT = 5000


def _boundary_starvation_config(name: str, seed: int) -> Dict:
    """The exact stage-2 boundary-starvation spec (the config the
    current 11 checkpoints were trained under)."""
    return {
        "run": {
            "name": name,
            "output_dir": "runs",
            "seed": seed,
            "deterministic": True,
            "device": "cuda" if DEVICE.type == "cuda" else "cpu",
            "dtype": "float32",
        },
        "pde": {
            "name": "poisson_1d",
            "domain": [-1.0, 1.0],
            "source": 1.0,
            "forcing": 0.0,
            "diffusion": 0.01,
            "boundary_values": [0.0, 0.0],
            "validation_points": 1001,
        },
        "model": {
            "input_dim": 1,
            "output_dim": 1,
            "hidden_layers": [64, 64, 64],
            "activation": "tanh",
            "init": "xavier",
        },
        "training": {
            "optimizer": "adam",
            "learning_rate": 0.001,
            "steps": STEPS,
            "interior_points": 256,
            "boundary_points": 2,
            "log_every": 500,
            "checkpoint_every": STEPS,
            "lambda_pde": 100.0,
            "lambda_bc": 0.01,
            "resample_every": 0,
        },
        "logging": {
            "save_activations": True,
            "activation_layers": [1],
            "save_pointwise_residuals": False,
            "log_gradients": False,
            "log_diagnostics": False,
        },
    }


def _train_new_seeds(seeds: List[int], steps: int = STEPS) -> List[Path]:
    """Train the new boundary-starvation models through the canonical
    training path (subprocess, like _train_one)."""
    from experiments.architecture_boundary import _train_one
    run_dirs = []
    for seed in seeds:
        name = f"boundary_starvation_seed{seed}"
        existing = RUNS / name
        ckpts = sorted(existing.glob("checkpoint_*.pt")) if existing.exists() else []
        if ckpts:
            print(f"  seed {seed}: checkpoints exist — reusing")
            run_dirs.append(existing)
            continue
        print(f"  seed {seed}: training ({steps} steps)...")
        cfg = _boundary_starvation_config(name, seed)
        cfg["training"]["steps"] = steps
        run_dirs.append(_train_one(cfg, steps))
    return run_dirs


def _load_model_and_pde(run_dir: Path):
    from experiments.run_pipeline import _remap_legacy_state
    from pinn.config import load_config
    from pinn.model import MLP
    from pinn.pdes import make_pde
    cfg = load_config(run_dir / "config.json")
    model = MLP(cfg.model.input_dim, cfg.model.output_dim,
                cfg.model.hidden_layers, cfg.model.activation).to(DEVICE)
    ckpt = torch.load(sorted(run_dir.glob("checkpoint_*.pt"))[-1],
                      map_location=DEVICE, weights_only=True)
    state = ckpt["model"]
    if any(k.startswith("net.") for k in state):
        state = _remap_legacy_state(state)
    model.load_state_dict(state)
    model.eval()
    return model, make_pde(cfg.pde)


def _run_battery_on_checkpoints(run_dirs: List[Path]) -> Dict:
    """The IDENTICAL stage-5 + stage-8 batteries on every new checkpoint.

    Identity requirements (registered): same frozen SAE (the first
    sae_models_v2 replica), same dictionary candidates, same
    supervised probe — all trained on the COMMITTED discovery pool
    with the new runs EXCLUDED (they are evaluation checkpoints, not
    discovery data; including them would change the basis and void
    the pairing with the committed 88 pairs).
    """
    from sae.model import SparseAutoencoder
    from interventions.causal import run_inference_interventions
    from interventions.pca_battery import (
        run_pca_causal_battery, train_pca_probe_direction,
        pca_basis_to_device,
    )
    from analysis.pca_interpretability import fit_pca
    from experiments.run_pipeline import (
        find_logged_runs, build_multiview_metrics, train_failure_probe,
        make_probe_directions_for_features,
    )

    # discovery pool: committed logged runs, EXCLUDING the new seeds
    new_names = {d.name for d in run_dirs}
    pool = [d for d in find_logged_runs(width=64)
            if d.name not in new_names]
    samples, labels, _, _ = build_multiview_metrics(pool)
    data = torch.tensor(samples, dtype=torch.float32)
    print(f"    discovery pool: {len(pool)} committed runs "
          f"({data.shape[0]} rows) — new seeds excluded")

    # SAE arm: identical frozen SAE + candidates + probe (stage-5)
    sae_path = sorted((RUNS / "sae_models_v2").glob("*/sae.pt"))[0]
    sae = SparseAutoencoder.load(sae_path, DEVICE)
    dict_path = RUNS / "feature_dictionary" / "physics_feature_dictionary.json"
    candidates = list(range(8))
    if dict_path.exists():
        dictionary = json.loads(dict_path.read_text())
        candidates = [f["feature_id"] for f in dictionary[:8]]
    probe_dir = train_failure_probe(sae, data, labels, DEVICE)
    probe_dirs = make_probe_directions_for_features(
        probe_dir, candidate_features=candidates)

    # PCA arm: identical pooled basis + probe (stage-8)
    basis_cpu = fit_pca(data, n_components=8)
    pca_probe = train_pca_probe_direction(basis_cpu, data, labels,
                                          torch.device("cpu"))
    basis = pca_basis_to_device(basis_cpu, DEVICE)

    sae_rows, pca_rows = [], []
    for run_dir in run_dirs:
        name = run_dir.name
        model, pde = _load_model_and_pde(run_dir)
        print(f"    battery on {name}")

        rows = run_inference_interventions(
            model=model, sae=sae, pde=pde, device=DEVICE,
            dtype=torch.float32, candidate_features=candidates,
            target_loss="pde", layer_index=1,
            probe_directions=probe_dirs,
        )
        for r in rows:
            r["run"] = name
        sae_rows.extend(rows)

        pca_res = run_pca_causal_battery(
            model=model, basis=basis, pde=pde, device=DEVICE,
            dtype=torch.float32, candidate_components=list(range(8)),
            probe_direction=pca_probe, target_loss="pde",
            layer_index=1, alpha=0.0,
        )
        for r in pca_res:
            r["run"] = name
        pca_rows.extend(pca_res)

    return {"sae_rows": sae_rows, "pca_rows": pca_rows}


def run_r7plus(new_seeds: List[int] = None, smoke: bool = False) -> Dict:
    print("=" * 64)
    print("R7+: Power-Upgraded SAE/PCA Equivalence Replication")
    print("=" * 64)

    seeds = (new_seeds or NEW_SEEDS)[:1] if smoke else (new_seeds or NEW_SEEDS)
    n_new = len(seeds)
    print(f"  registered intermediate n: {n_new} new seeds "
          f"(combined {11 + n_new} checkpoints)")
    print(f"  full-power target (R8): 163 checkpoints — this run reaches "
          f"{(11 + n_new) / 163:.0%} of it; achieved power reported")

    # ---- 1. train the new checkpoints ----
    print(f"\n[1/3] Training {n_new} boundary-starvation models "
          f"({STEPS} steps each)...")
    new_dirs = _train_new_seeds(seeds, steps=300 if smoke else STEPS)

    # ---- 2. the identical batteries on ALL checkpoints ----
    print(f"\n[2/3] Running the identical stage-5/8 batteries on "
          f"{len(new_dirs)} new checkpoints...")
    battery = _run_battery_on_checkpoints(new_dirs)

    # ---- 3. combine with the committed pairs and TOST ----
    print("\n[3/3] Combining with the committed 88 pairs and testing...")
    pca = json.loads((RUNS / "pca_causal_results.json").read_text())
    committed_pairs = pca["head_to_head_vs_sae"]["pairs"]

    # pair the new rows per checkpoint: the stage-8 head-to-head pairs
    # POSITIONALLY (dictionary candidate i <-> PCA component i, per
    # run) — the committed pairs follow the same convention.
    new_sae = defaultdict(dict)
    new_pca = defaultdict(dict)
    for r in battery["sae_rows"]:
        new_sae[r["run"]][r["feature_idx"]] = \
            r["causal_score"]["causal_strength"]
    for r in battery["pca_rows"]:
        new_pca[r["run"]][r["feature_idx"]] = \
            r["causal_score"]["causal_strength"]

    per_ckpt = {}
    for run in sorted(set(new_sae) & set(new_pca)):
        # positional pairing: candidate order (sorted feature ids for
        # the SAE dictionary candidates) vs component order 0..7
        sae_keys = sorted(new_sae[run])
        pca_keys = sorted(new_pca[run])
        if len(sae_keys) != len(pca_keys):
            continue
        diffs = [new_pca[run][pk] - new_sae[run][sk]
                 for sk, pk in zip(sae_keys, pca_keys)]
        if diffs:
            per_ckpt[run] = float(np.mean(diffs))
    committed_means = defaultdict(list)
    for pr in committed_pairs:
        committed_means[pr["run"]].append(
            pr["pca_causal_strength"] - pr["sae_causal_strength"])
    for run, vals in committed_means.items():
        per_ckpt[run] = float(np.mean(vals))

    runs_sorted = sorted(per_ckpt)
    dbar = np.array([per_ckpt[r] for r in runs_sorted])
    m = len(runs_sorted)
    grand = float(dbar.mean())
    s_c = float(dbar.std(ddof=1))
    print(f"  combined: {m} checkpoints | grand mean diff {grand:+.5f} | "
          f"cluster SD {s_c:.5f}")

    # cluster bootstrap CI (R8 protocol)
    rng = np.random.default_rng(0)
    boots = [float(rng.choice(dbar, m, replace=True).mean())
             for _ in range(N_BOOT)]
    lo90 = float(np.percentile(boots, 5))
    hi90 = float(np.percentile(boots, 95))
    lo95, hi95 = (float(np.percentile(boots, 2.5)),
                  float(np.percentile(boots, 97.5)))

    # TOST on the combined set (cluster-level t-test, df = m-1)
    se_c = s_c / math.sqrt(m)
    if se_c > 0:
        t_low = (grand - (-DELTA)) / se_c
        t_high = (DELTA - grand) / se_c
        try:
            from scipy import stats as st
            p_low = 1.0 - st.t.cdf(t_low, m - 1)
            p_high = 1.0 - st.t.cdf(t_high, m - 1)
        except ImportError:
            z_low = (grand + DELTA) / se_c
            z_high = (DELTA - grand) / se_c
            p_low = 1.0 - 0.5 * (1 + math.erf(z_low / math.sqrt(2)))
            p_high = 1.0 - 0.5 * (1 + math.erf(z_high / math.sqrt(2)))
        p_tost = float(max(p_low, p_high))
    else:
        p_tost = 0.0 if abs(grand) < DELTA else 1.0
    equivalent = bool(p_tost <= ALPHA)

    # achieved power at the observed cluster SD (TOST power, the
    # standard two-sided-of-one-sided approximation)
    from scipy import stats as st
    z_a = 1.645
    se_c_pow = s_c / math.sqrt(m)
    achieved_power = float(
        st.norm.cdf(se_c_pow and (DELTA / se_c_pow - z_a))
        + st.norm.cdf(-(DELTA / se_c_pow + z_a)))

    # the pre-written trichotomy (R7 rule, corrected)
    lo90b, hi90b = lo90, hi90
    if equivalent:
        outcome = "R7+a"
        note = ("Equivalence CONFIRMED on the combined set at delta=0.01: "
                "the basis-independence claim upgrades to formal "
                "equivalence.")
    elif lo90b <= DELTA and hi90b >= -DELTA:
        outcome = "R7+b"
        note = ("Inconclusive at the combined n: the cluster 90% CI still "
                "overlaps the margin — recorded as the honest terminal "
                "state at this resource level, with achieved power "
                "stated.")
    else:
        outcome = "R7+c"
        note = ("Non-equivalence established: the cluster 90% CI lies "
                "entirely outside the margin — the claim rewords per "
                "the R7 outcome-3 rule.")

    print(f"\n  cluster 90% CI: [{lo90:+.5f}, {hi90:+.5f}] | "
          f"TOST p = {p_tost:.4f}")
    print(f"  achieved power at delta=0.01: {achieved_power:.2f}")
    print(f"  OUTCOME: {outcome}")
    print(f"    {note}")

    report = {
        "stage": "r7plus_equivalence_replication",
        "preregistration": "docs/preregistration.md §R7+ (registered "
                           "2026-09-12 after R8's power calculation)",
        "new_seeds": seeds,
        "n_checkpoints_combined": m,
        "power_context": {
            "full_power_target_80pct": 163,
            "fraction_of_target_reached": (m / 163),
            "achieved_power_at_delta": achieved_power,
        },
        "per_checkpoint_mean_diff": per_ckpt,
        "combined": {
            "grand_mean": grand,
            "cluster_sd": s_c,
            "cluster_ci90": [lo90, hi90],
            "cluster_ci95": [lo95, hi95],
            "tost_p": p_tost,
            "equivalent": equivalent,
        },
        "new_battery_rows": {
            "n_sae_rows": len(battery["sae_rows"]),
            "n_pca_rows": len(battery["pca_rows"]),
        },
        "outcome": outcome,
        "outcome_note": note,
    }
    out_dir = RUNS / "r7plus_equivalence"
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "r7plus_report.json").write_text(
        json.dumps(report, indent=2, default=str))
    (out_dir / "r7plus_battery_rows.json").write_text(
        json.dumps(battery, indent=2, default=str))
    print(f"\n  report: {out_dir / 'r7plus_report.json'}")
    return report


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--seeds", type=str, default=None,
                    help="comma-separated new seeds (default: the "
                         "registered 20)")
    args = ap.parse_args()
    seeds = [int(s) for s in args.seeds.split(",")] if args.seeds else None
    run_r7plus(new_seeds=seeds, smoke=args.smoke)
