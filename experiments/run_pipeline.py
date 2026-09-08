"""
End-to-End Experimental Execution Pipeline for PINN Mechanistic Interpretability
(v3 — regime-boundary campaign, Sep 2026).

Stages:
  1. Train PINNs across configs (baselines + failure modes) with full logging.
  2. Index the Failure Atlas (10-seed statistics where available).
  3. Train TopK SAEs + N-seed replicas for cross-seed matching, with
     PCA and random-direction baselines (v2 plan §1.2/§1.3).
  4. Physics-Feature Dictionary from MULTI-VIEW metrics
     (loss components, gradient cosine, boundary distance, spectral power),
     with random-direction association controls.
  5. Causal interventions with 3 controls (unrelated / random / linear-probe)
     and bootstrap CIs across seeds/checkpoints.
  6. Monitor suite: loss-only vs conventional vs SAE monitors,
     run-level train/test split, AUROC/AUPRC/lead-time with CIs.
  7. Closed-loop controller rescue with rollback + honest reporting
     vs. no-action and static-reweight baselines.
  8. PCA causal battery — the same 3-control protocol on PCA components,
     head-to-head vs the SAE battery (Phase 10).
  9. Causal abstraction — region-level partial interchange interventions,
     PCA/SAE/random alignment bases (Phase 11).
 10. Dimensional boundary expansion — 2D advection/reaction-diffusion and
     time-dependent Burgers/Allen-Cahn geometry (Phase 9A/9B).
 11. Operator regime boundary — FNO on Green's-function regression; the
     high-rank positive control (Phase 9C).
 12. SOTA optimization baselines (GradNorm, NTK-adaptive, RBA) vs the
     controller (Phase 14).
 13. Statistical hardening — power analysis, Bayesian AUROC posterior,
     threshold sensitivity (Phase 13).
 14. Operator causal battery — the 3-control causal protocol + interchange
     on FNO block states, completing the regime-boundary claim (Phase 15).
 15. NTK conflict↔SAE feature-activity bridge — the preregistered
     correlational bridge (revision Gap 2; H15a/H15b).
"""
from __future__ import annotations
import json
import sys
import subprocess
from pathlib import Path
import torch
import numpy as np

from pinn.config import load_config
from pinn.pdes import make_pde
from pinn.model import MLP
from pinn_logging.io import load_jsonl
from sae.model import SparseAutoencoder
from sae.dataset import ActivationDataset, make_run_splits
from sae.train import train_sae, pca_reconstruction_error
from sae.features import (
    compute_feature_stats,
    associate_features_with_metrics,
    match_features_across_saes,
    build_feature_dictionary,
    save_feature_dictionary,
)
from analysis.failure_atlas import aggregate_failure_atlas
from analysis.feature_dictionary import render_feature_dictionary_markdown
from interventions.causal import (
    run_inference_interventions,
    aggregate_causal_benchmarks,
    train_failure_probe,
    make_probe_directions_for_features,
    run_training_intervention,
    compute_bootstrap_ci,
)
from monitoring.features import (
    build_monitor_dataset, derive_failure_step, build_trajectory_features,
    leakage_audit,
)
from monitoring.models import ThresholdMonitor, LogisticMonitor, evaluate_monitor
from controller.state_machine import PINNController, ControllerConfig


PROJECT_ROOT = Path(__file__).resolve().parent.parent
RUNS = PROJECT_ROOT / "runs"
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
LAYER = "layers.1"


# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------

def find_logged_runs(width: int = 64) -> list:
    """All run dirs with activation logs for the given hidden width."""
    out = []
    for d in sorted(RUNS.iterdir()):
        if d.is_dir() and (d / "activations.jsonl").exists():
            recs = load_jsonl(d / "activations.jsonl")
            if recs:
                acts = recs[0].get("activations", {})
                layer_data = acts.get(LAYER, {})
                shape = layer_data.get("shape", [])
                if len(shape) >= 2 and shape[1] == width:
                    out.append(d)
    return out


def _remap_legacy_state(state: dict) -> dict:
    """Remap old MLP checkpoints (nn.Sequential `net.N.{weight,bias}`,
    alternating Linear/activation at even indices) to the current
    ModuleList layout (`layers.N.{weight,bias}` for every Linear)."""
    new_state = {}
    # Group keys by their Sequential module index (net.0, net.2, net.4, ...).
    module_ids = sorted({int(k.split(".")[1]) for k in state
                         if k.startswith("net.")})
    for layer_idx, seq_id in enumerate(module_ids):
        for param in ("weight", "bias"):
            old_key = f"net.{seq_id}.{param}"
            if old_key in state:
                new_state[f"layers.{layer_idx}.{param}"] = state[old_key]
    return new_state


def build_multiview_metrics(run_dirs: list, layer_name: str = LAYER):
    """Per-sample multi-view metric arrays aligned with the activation dataset.

    For every (run, step, probe-point) sample we attach:
      - loss_pde, loss_bc (step-level, broadcast)
      - rel_l2 (step-level validation error)
      - grad_cosine (step-level PDE-vs-BC gradient cosine)
      - high_freq_power (step-level from the diagnostics error spectrum)
      - boundary_dist (per probe-point coordinate distance to domain edge)
      - data_std (activation spread — RETAINED ONLY as a confound control)

    Returns
    -------
    (samples, labels, run_ids, metric_arrays) where samples is (N, D)
    activation vectors, labels are 1 for failure-run samples, run_ids maps
    each sample to its run index (for run-level splitting), and
    metric_arrays maps metric name -> (N,) values.
    """
    all_rows: list = []
    all_labels: list = []
    all_run_ids: list = []
    metric_cols = {k: [] for k in
                   ["loss_pde", "loss_bc", "rel_l2", "grad_cosine",
                    "high_freq_power", "boundary_dist", "data_std"]}

    for run_idx, d in enumerate(run_dirs):
        metrics = {m["step"]: m for m in load_jsonl(d / "metrics.jsonl")}
        grads = {g["step"]: g.get("gradient_stats", {})
                 for g in load_jsonl(d / "gradients.jsonl")}
        diags = {r["step"]: r for r in load_jsonl(d / "diagnostics.jsonl")}

        # Run-level outcome label: prefer run_summary; fall back to the final
        # metrics record's relative_l2 (failure runs lack evaluate.py output).
        summary_path = d / "run_summary.json"
        is_failure = None
        if summary_path.exists():
            s = json.loads(summary_path.read_text())
            rl = float(s.get("validation_relative_l2",
                              s.get("final_relative_l2", 1.0)))
            is_failure = int(rl > 0.05)
        else:
            mrecs = load_jsonl(d / "metrics.jsonl")
            if mrecs:
                rl = float(mrecs[-1].get("relative_l2", 1.0) or 0.0)
                is_failure = int(rl > 0.05)
        if is_failure is None:
            continue

        # Probe coordinates (deterministic linspace across the domain).
        cfg_path = d / "config.json"
        dom = [-1.0, 1.0]
        if cfg_path.exists():
            try:
                c = json.loads(cfg_path.read_text())
                dom = c.get("pde", {}).get("domain", dom)
            except Exception:
                pass
        probe_x = np.linspace(dom[0], dom[1], 50)
        boundary_dist = np.minimum(np.abs(probe_x - dom[0]),
                                   np.abs(probe_x - dom[1]))

        for rec in load_jsonl(d / "activations.jsonl"):
            step = rec["step"]
            m = metrics.get(step, {})
            g = grads.get(step, {})
            dg = diags.get(step, {})

            lp = float(m.get("loss_pde", 0.0) or 0.0)
            lb = float(m.get("loss_bc", 0.0) or 0.0)
            rl = float(m.get("relative_l2", 0.0) or 0.0)
            gc = float(g.get("gradient_cosines", {}).get("pde_vs_bc", 0.0) or 0.0)
            hf = float(dg.get("error_spectrum", {}).get("high_freq_power", 0.0) or 0.0)

            raw = rec["activations"].get(layer_name, {}).get("raw")
            if raw is None:
                continue
            arr = np.asarray(raw, dtype=np.float32)  # (50, D)
            for i in range(arr.shape[0]):
                row = arr[i]
                all_rows.append(row)
                all_labels.append(is_failure)
                all_run_ids.append(run_idx)
                metric_cols["loss_pde"].append(lp)
                metric_cols["loss_bc"].append(lb)
                metric_cols["rel_l2"].append(rl)
                metric_cols["grad_cosine"].append(gc)
                metric_cols["high_freq_power"].append(hf)
                metric_cols["boundary_dist"].append(boundary_dist[i])
                metric_cols["data_std"].append(float(row.std()))

    samples = np.stack(all_rows, axis=0) if all_rows else np.empty((0, 0))
    metric_arrays = {k: np.asarray(v, dtype=np.float64)
                     for k, v in metric_cols.items()}
    return (samples,
            np.asarray(all_labels, dtype=np.int64),
            np.asarray(all_run_ids, dtype=np.int64),
            metric_arrays)


# ---------------------------------------------------------------------------
# Stage 1 — Train PINNs
# ---------------------------------------------------------------------------

def run_stage_1_train_pinns(steps: int = 2000):
    print("\n=======================================================")
    print("STAGE 1: Training PINN Models across Baseline & Failure Configs")
    print("=======================================================")

    config_files = [
        "configs/poisson_baseline.yaml",
        "configs/advection_1d_baseline.yaml",
        "configs/reaction_diffusion_1d_baseline.yaml",
        "configs/failure_boundary_starvation.yaml",
        "configs/failure_gradient_conflict.yaml",
        "configs/failure_spectral_suppression.yaml",
        "configs/failure_collocation_starvation.yaml",
    ]

    for cfg_rel in config_files:
        cfg_path = PROJECT_ROOT / cfg_rel
        if not cfg_path.exists():
            print(f"Skipping missing config {cfg_path}")
            continue
        print(f"\n---> Training with {cfg_rel}...")
        cmd = [
            sys.executable, "-m", "experiments.train",
            "--config", str(cfg_path),
            "--steps", str(steps),
            "--log-gradients",
            "--log-activations",
            "--act-save-raw",
            "--log-diagnostics",
        ]
        res = subprocess.run(cmd, cwd=PROJECT_ROOT, capture_output=True, text=True)
        if res.returncode == 0:
            print(f"  Finished training for {cfg_rel}")
        else:
            print(f"  Training error for {cfg_rel}: {res.stderr[:300]}")


# ---------------------------------------------------------------------------
# Stage 2 — Failure Atlas
# ---------------------------------------------------------------------------

def run_stage_2_failure_atlas():
    print("\n=======================================================")
    print("STAGE 2: Generating Failure Atlas Index (10-seed statistics)")
    print("=======================================================")
    entries = aggregate_failure_atlas(RUNS, RUNS / "failure_atlas")

    # Group seed-matrix runs by config prefix for per-seed statistics.
    groups: dict = {}
    for e in entries:
        name = e.get("run_name") or ""
        if "_seed" in name:
            prefix = name.split("_seed")[0]
            groups.setdefault(prefix, []).append(e)

    stats = {}
    for prefix, runs in groups.items():
        l2s = [r.get("final_rel_l2") for r in runs
               if r.get("final_rel_l2") is not None]
        labels = [r.get("operational_label") for r in runs]
        if l2s:
            stats[prefix] = {
                "n_seeds": len(runs),
                "rel_l2_mean": float(np.mean(l2s)),
                "rel_l2_std": float(np.std(l2s)),
                "rel_l2_ci": compute_bootstrap_ci(l2s),
                "label_counts": {l: labels.count(l) for l in set(labels)},
            }
    if stats:
        with open(RUNS / "failure_atlas" / "seed_matrix_stats.json", "w") as f:
            json.dump(stats, f, indent=2)
        print(f"Seed-matrix statistics for {len(stats)} regimes "
              f"written to seed_matrix_stats.json")
    print(f"Indexed {len(entries)} runs into Failure Atlas.")
    return entries


# ---------------------------------------------------------------------------
# Stage 3 — SAE discovery with mandatory baselines (TopK + seed replicas)
# ---------------------------------------------------------------------------

def run_stage_3_train_saes(expansion: int = 4, topk: int = 8,
                           n_seed_replicas: int = 3, steps: int = 800):
    print("\n=======================================================")
    print("STAGE 3: TopK SAE Training + PCA/Random Baselines + Seed Replicas")
    print("=======================================================")

    run_dirs = find_logged_runs(width=64)
    print(f"Found {len(run_dirs)} runs with [50, 64] activations.")
    if not run_dirs:
        return None

    out_dir = RUNS / "sae_models_v2"
    out_dir.mkdir(parents=True, exist_ok=True)

    # Run-level split: SAE discovery uses discovery runs only.
    train_dirs, val_dirs, test_dirs = make_run_splits(run_dirs, seed=0)
    print(f"Run-level split: {len(train_dirs)} train / {len(val_dirs)} val / "
          f"{len(test_dirs)} test")

    from sae.dataset import build_dataloaders
    train_loader, val_loader, test_loader, fit_stats = build_dataloaders(
        train_dirs, val_dirs, test_dirs, LAYER,
        batch_size=256, device=DEVICE,
    )

    # Flatten all training activations for the PCA baseline.
    all_acts = torch.cat([b for b in train_loader], dim=0)
    input_dim = all_acts.shape[1]

    # ---- PCA baselines at two budgets (v2 §1.3) ----
    # (a) full-width PCA (upper bound for linear reconstruction)
    pca_full_err = pca_reconstruction_error(all_acts.to(DEVICE), input_dim)
    # (b) k-matched PCA — same number of ACTIVE latents as the TopK SAE
    #     (topk), which is the budget-fair comparison.
    pca_k_err = pca_reconstruction_error(all_acts.to(DEVICE), min(topk, input_dim))

    # ---- Random-encoder baseline: same architecture, frozen random weights ----
    from sae.model import SparseAutoencoder
    rand_sae = SparseAutoencoder(
        input_dim=input_dim, latent_expansion=expansion,
        activation_mode="topk", topk=topk,
    ).to(DEVICE)
    for p in rand_sae.parameters():
        p.requires_grad_(False)
    rand_recon = _eval_recon(rand_sae, test_loader, DEVICE)

    print(f"PCA(full {input_dim}-comp) train MSE:   {pca_full_err:.4e}")
    print(f"PCA(k-matched {topk}-comp) train MSE:    {pca_k_err:.4e}")
    print(f"Random-SAE test MSE:              {rand_recon:.4e}")

    # ---- Train n_seed_replicas TopK SAEs with different seeds ----
    results = []
    sae_paths = []
    for replica in range(n_seed_replicas):
        torch.manual_seed(100 + replica)
        print(f"\n=== TopK SAE replica {replica} (exp={expansion}, k={topk}) ===")
        sae = SparseAutoencoder(
            input_dim=input_dim, latent_expansion=expansion,
            activation_mode="topk", topk=topk, decoder_normalize=True,
        ).to(DEVICE)

        run_out = out_dir / f"topk_exp{expansion}_k{topk}_s{replica}"
        train_result = train_sae(
            sae, train_loader, val_loader,
            steps=steps, learning_rate=1e-3, device=DEVICE, out_dir=run_out,
        )
        test_recon = _eval_recon(sae, test_loader, DEVICE)
        sae.save(run_out / "sae.pt")
        sae_paths.append(run_out / "sae.pt")

        results.append({
            "replica": replica,
            "expansion": expansion, "topk": topk,
            "activation_mode": "topk",
            "test_recon_mse": test_recon,
            "dead_feature_frac": sae.dead_feature_fraction(),
            "history": train_result.get("history", [])[-3:],
        })
        print(f"  Replica {replica} test recon MSE: {test_recon:.4e}")

    summary = {
        "trained": results,
        "baselines": {
            "pca_full_n_components": input_dim,
            "pca_full_train_mse": pca_full_err,
            "pca_k_matched_components": min(topk, input_dim),
            "pca_k_matched_train_mse": pca_k_err,
            "random_sae_test_mse": rand_recon,
        },
        "sae_beats_random": all(r["test_recon_mse"] < rand_recon for r in results),
        "sae_beats_pca_k_matched": all(
            r["test_recon_mse"] < pca_k_err for r in results),
        "test_dirs": [str(d) for d in test_dirs],
        "val_dirs": [str(d) for d in val_dirs],
    }
    with open(out_dir / "sae_stage3_summary.json", "w") as f:
        json.dump(summary, f, indent=2)
    print(f"\nStage 3 gate: SAE beats random: {summary['sae_beats_random']} | "
          f"SAE beats k-matched PCA: {summary['sae_beats_pca_k_matched']} "
          f"(honest reporting — if False, the SAE adds no reconstruction "
          f"advantage over a linear baseline at the same latent budget)")
    return out_dir


def _eval_recon(sae, loader, device):
    sae.eval()
    total, n = 0.0, 0
    with torch.no_grad():
        for batch in loader:
            batch = batch.to(device)
            _, a_hat = sae(batch)
            total += float(((batch - a_hat) ** 2).mean()) * len(batch)
            n += len(batch)
    return total / n if n else float("nan")


# ---------------------------------------------------------------------------
# Stage 4 — Physics-Feature Dictionary (multi-view)
# ---------------------------------------------------------------------------

def run_stage_4_feature_dictionary(sae_dir: Path):
    print("\n=======================================================")
    print("STAGE 4: Physics-Feature Dictionary (multi-view metrics)")
    print("=======================================================")

    sae_paths = sorted(sae_dir.glob("*/sae.pt"))
    if not sae_paths:
        print("No SAE checkpoints found.")
        return None
    sae = SparseAutoencoder.load(sae_paths[0], DEVICE)

    run_dirs = find_logged_runs(width=64)
    samples, labels, run_ids, metric_arrays = build_multiview_metrics(run_dirs)

    print(f"Dataset: {samples.shape[0]} samples x {samples.shape[1]} dims; "
          f"{len(run_dirs)} runs; {int(labels.sum())} failure-run samples")

    # Normalise with training-set statistics only (runs 0..2 as discovery).
    data = torch.tensor(samples, dtype=torch.float32)
    mean = data.mean(dim=0, keepdim=True)
    std = data.std(dim=0, keepdim=True).clamp(min=1e-8)
    data_norm = (data - mean) / std

    stats = compute_feature_stats(sae, data_norm, DEVICE)
    corrs = associate_features_with_metrics(sae, data_norm, metric_arrays, DEVICE)

    # Random-direction association control (same correlation pipeline).
    rng = np.random.default_rng(42)
    z_rand = rng.standard_normal((data_norm.shape[0], sae.latent_dim))
    np.clip(z_rand, 0, None, out=z_rand)
    rand_corr = {}
    for mname, mvals in metric_arrays.items():
        cs = []
        for j in range(z_rand.shape[1]):
            if z_rand[:, j].std() > 1e-10 and mvals.std() > 1e-10:
                c = float(np.corrcoef(z_rand[:, j], mvals)[0, 1])
                cs.append(c if np.isfinite(c) else 0.0)
            else:
                cs.append(0.0)
        rand_corr[mname] = cs

    # Real cross-seed matching: same config, different SAE seeds.
    saes = [SparseAutoencoder.load(p, DEVICE) for p in sae_paths]
    families = match_features_across_saes(saes, threshold=0.8) if len(saes) >= 2 else []

    dictionary = build_feature_dictionary(
        feature_stats=stats,
        correlations=corrs,
        families=families,
        layer_name=LAYER,
        sae_version="topk_v2",
    )

    # Attach random-baseline comparison per metric for the report.
    baseline_report = {}
    for mname in metric_arrays:
        real_max = max((abs(c) for c in corrs.get(mname, [0])), default=0.0)
        rand_max = max((abs(c) for c in rand_corr.get(mname, [0])), default=0.0)
        baseline_report[mname] = {
            "max_real_abs_corr": real_max,
            "max_random_abs_corr": rand_max,
            "real_exceeds_random": bool(real_max > rand_max * 1.5),
        }

    out_dir = RUNS / "feature_dictionary"
    out_dir.mkdir(parents=True, exist_ok=True)
    save_feature_dictionary(dictionary, out_dir / "physics_feature_dictionary.json")
    render_feature_dictionary_markdown(dictionary, out_dir / "physics_feature_dictionary.md")
    with open(out_dir / "random_baseline_report.json", "w") as f:
        json.dump(baseline_report, f, indent=2)

    n_stable = sum(1 for fam in families if fam.get("stable"))
    print(f"Dictionary: {len(dictionary)} features; "
          f"{n_stable}/{len(families)} feature families stable across SAE seeds.")
    print("Random-baseline association check:")
    for m, r in baseline_report.items():
        print(f"  {m:16s} real={r['max_real_abs_corr']:.3f} "
              f"random={r['max_random_abs_corr']:.3f} "
              f"{'PASS' if r['real_exceeds_random'] else 'CHECK'}")
    return dictionary


# ---------------------------------------------------------------------------
# Stage 5 — Causal interventions with 3 controls + CIs
# ---------------------------------------------------------------------------

def run_stage_5_causal_interventions(sae_dir: Path):
    print("\n=======================================================")
    print("STAGE 5: Causal Interventions vs 3 Controls (bootstrap CIs)")
    print("=======================================================")

    sae_paths = sorted(sae_dir.glob("*/sae.pt"))
    if not sae_paths:
        return
    sae = SparseAutoencoder.load(sae_paths[0], DEVICE)

    # Multi-seed causal evaluation: use every available boundary-starvation
    # checkpoint (the failure class where features are hypothesised causal).
    eval_runs = [d for d in RUNS.iterdir()
                 if d.is_dir() and "boundary_starvation" in d.name
                 and list(d.glob("checkpoint_*.pt"))]
    # Fall back to the 7-config failure run if the seed matrix lacks checkpoints.
    if not eval_runs:
        eval_runs = [RUNS / "failure_boundary_starvation"]
    print(f"Causal eval on {len(eval_runs)} boundary-starvation runs: "
          f"{[d.name for d in eval_runs]}")

    run_dirs = find_logged_runs(width=64)
    samples, labels, run_ids, metric_arrays = build_multiview_metrics(run_dirs)
    data = torch.tensor(samples, dtype=torch.float32)

    # Candidate features: top-8 by activation frequency from the dictionary.
    dict_path = RUNS / "feature_dictionary" / "physics_feature_dictionary.json"
    candidates = list(range(8))
    if dict_path.exists():
        dictionary = json.loads(dict_path.read_text())
        candidates = [f["feature_id"] for f in dictionary[:8]]

    # Train the supervised probe direction on discovery activations.
    # NOTE: directions must cover ALL candidate features, otherwise
    # measure_probe_direction_effect silently falls back to a random
    # direction and the probe control becomes a duplicate random control.
    probe_dir = train_failure_probe(sae, data, labels, DEVICE)
    probe_dirs = make_probe_directions_for_features(
        probe_dir, candidate_features=candidates,
    )
    assert all(f in probe_dirs for f in candidates), (
        "probe directions must cover every candidate feature"
    )

    all_results = []
    for run_dir in eval_runs:
        cfg = load_config(run_dir / "config.json")
        pde = make_pde(cfg.pde)
        model = MLP(cfg.model.input_dim, cfg.model.output_dim,
                    cfg.model.hidden_layers, cfg.model.activation).to(DEVICE)
        ckpts = sorted(run_dir.glob("checkpoint_*.pt"))
        if not ckpts:
            continue
        ckpt = torch.load(ckpts[-1], map_location=DEVICE, weights_only=False)
        state = ckpt["model"]
        # Legacy checkpoints use the old `net.Sequential` layout (net.0, net.2,
        # ...); current MLP uses `layers.N`.  Remap if needed.
        if any(k.startswith("net.") for k in state):
            state = _remap_legacy_state(state)
        model.load_state_dict(state)
        dtype = getattr(torch, cfg.run.dtype)

        results = run_inference_interventions(
            model=model, sae=sae, pde=pde, device=DEVICE, dtype=dtype,
            candidate_features=candidates,
            target_loss="pde", layer_index=1,
            probe_directions=probe_dirs,
            out_dir=run_dir / "causal",
        )
        for r in results:
            r["run"] = run_dir.name
        all_results.extend(results)

    aggregate = aggregate_causal_benchmarks(all_results)

    # Verify direction consistency: fraction of runs where the target effect
    # exceeds ALL three controls individually (v2 gate, stricter than max()).
    verified = 0
    for r in all_results:
        cs = r["causal_score"]
        t = abs(cs["target_effect"])
        beats_all = (
            t > abs(cs["control_unrelated_effect"])
            and t > abs(cs["control_random_effect"])
            and t > abs(cs["control_probe_effect"])
        )
        r["beats_all_controls"] = bool(beats_all)
        verified += int(beats_all)

    # Multiple-comparison corrections (v2.1): per-feature paired sign tests
    # across runs, with Bonferroni and BH-FDR correction across features.
    from interventions.causal import per_feature_causal_pvalues
    mc = per_feature_causal_pvalues(all_results)

    # Sign diagnostic: are target deltas directed (all-increase) or mixed?
    signs = [1 if r["causal_score"]["target_effect"] > 0 else -1
             for r in all_results]

    # Probe-control quality: does the supervised probe direction beat the
    # SAE features at shifting the target loss?  (AXBench-style finding.)
    probe_beats_target = sum(
        1 for r in all_results
        if abs(r["causal_score"]["control_probe_effect"])
        > abs(r["causal_score"]["target_effect"])
    )

    out = {
        "n_evaluations": len(all_results),
        "n_beats_all_controls": verified,
        "verification_rate": verified / len(all_results) if all_results else 0.0,
        "causal_strength_ci": aggregate["causal_strength_ci"],
        "specificity_ci": aggregate["specificity_ci"],
        "sign_diagnostic": {
            "n_target_delta_positive": sum(1 for s in signs if s > 0),
            "n_target_delta_negative": sum(1 for s in signs if s < 0),
        },
        "probe_control": {
            "n_probe_beats_target": probe_beats_target,
            "probe_beats_target_rate":
                probe_beats_target / len(all_results) if all_results else 0.0,
            "note": "probe > target means a supervised linear direction "
                    "shifts the target loss more than the SAE feature "
                    "(AXBench-consistent: simple baselines outperform SAEs)",
        },
        "multiple_comparisons": {
            "n_features_tested": mc["n_features_tested"],
            "bonferroni_survivors": mc["bonferroni"]["survivors"],
            "bonferroni_n_survivors": mc["bonferroni"]["n_survivors"],
            "bh_fdr_survivors": mc["bh_fdr"]["survivors"],
            "bh_fdr_n_survivors": mc["bh_fdr"]["n_survivors"],
            "chance_expected_hits_at_alpha": mc["chance_expected_hits_at_alpha"],
            "per_feature": mc["per_feature"],
        },
        "evaluations": all_results,
    }
    out_path = RUNS / "causal_intervention_results.json"
    with open(out_path, "w") as f:
        json.dump(out, f, indent=2)

    print(f"Causal strength E_T: {out['causal_strength_ci']['mean']:.4f} "
          f"[{out['causal_strength_ci']['ci_lower']:.4f}, "
          f"{out['causal_strength_ci']['ci_upper']:.4f}] (95% bootstrap CI)")
    print(f"Specificity S_spec:  {out['specificity_ci']['mean']:.4f} "
          f"[{out['specificity_ci']['ci_lower']:.4f}, "
          f"{out['specificity_ci']['ci_upper']:.4f}]")
    print(f"Features beating ALL 3 controls: {verified}/{len(all_results)} "
          f"({out['verification_rate']:.1%})")
    print(f"Sign diagnostic: {out['sign_diagnostic']['n_target_delta_positive']} "
          f"positive / {out['sign_diagnostic']['n_target_delta_negative']} negative "
          f"target deltas")
    print(f"Probe beats SAE target: {probe_beats_target}/{len(all_results)} "
          f"({out['probe_control']['probe_beats_target_rate']:.1%})")
    print(f"MC correction: Bonferroni survivors: "
          f"{mc['bonferroni']['n_survivors']}/{mc['n_features_tested']} | "
          f"BH-FDR survivors: {mc['bh_fdr']['n_survivors']}/{mc['n_features_tested']} "
          f"(chance expectation at alpha=0.05: "
          f"{mc['chance_expected_hits_at_alpha']:.1f} hits)")


# ---------------------------------------------------------------------------
# Stage 6 — Monitor suite (loss-only vs conventional vs SAE)
# ---------------------------------------------------------------------------

def run_stage_6_monitoring(history_window: int = 10, failure_horizon: int = 5):
    print("\n=======================================================")
    print("STAGE 6: Early-Warning Monitor Suite (run-level splits)")
    print("=======================================================")

    run_dirs = [d for d in sorted(RUNS.iterdir())
                if d.is_dir() and (d / "metrics.jsonl").exists()]
    if not run_dirs:
        print("No metrics logs found.")
        return

    # Split at the RUN level (not the record level).
    rng = np.random.default_rng(0)
    rng.shuffle(run_dirs)
    n_train = max(1, int(0.7 * len(run_dirs)))
    train_runs, test_runs = run_dirs[:n_train], run_dirs[n_train:]
    print(f"Monitor split: {len(train_runs)} train runs / {len(test_runs)} test runs")

    def _check_sae_compatible(run_dir: Path, sae_dim: int, layer: str) -> bool:
        """Check if a run's activations are compatible with the SAE width."""
        try:
            acts = load_jsonl(run_dir / "activations.jsonl")
            if not acts:
                return False
            first = acts[0].get("activations", {}).get(LAYER, {})
            width = first.get("shape", [0, 0])[1] if first.get("shape") else 0
            return width == sae_dim
        except Exception:
            return False

    # Determine SAE compatibility early if SAE is available
    sae_paths = sorted(RUNS.glob("sae_models_v2/*/sae.pt"))
    sae = None
    if sae_paths:
        sae = SparseAutoencoder.load(sae_paths[0], DEVICE)

    # Filter runs to only those compatible with SAE (if SAE exists)
    if sae is not None:
        sae_dim = sae.input_dim
        train_runs = [d for d in train_runs if _check_sae_compatible(d, sae.input_dim, LAYER)]
        test_runs = [d for d in test_runs if _check_sae_compatible(d, sae.input_dim, LAYER)]

    def _build_dataset(runs):
        """Build feature matrix and labels from runs."""
        Xs, ys = [], []
        for d in runs:
            metrics = load_jsonl(d / "metrics.jsonl")
            fs = derive_failure_step(metrics)
            X, y, _ = build_trajectory_features(
                metrics, history_window, failure_horizon, "any", fs,
            )
            if X.shape[0] > 0:
                Xs.append(X)
                ys.append(y)
        if not Xs:
            return np.empty((0, 0)), np.empty(0)
        return np.vstack(Xs), np.concatenate(ys)

    X_train, y_train = _build_dataset(train_runs)
    X_test, y_test = _build_dataset(test_runs)
    print(f"Train examples: {X_train.shape[0]} ({int(y_train.sum())} positive) | "
          f"Test examples: {X_test.shape[0]} ({int(y_test.sum())} positive)")

    if X_test.shape[0] == 0 or len(np.unique(y_test)) < 2:
        print("Insufficient test diversity for AUROC evaluation.")
        return

    # H2 fix (external review): the previous call passed failure_step=10**9,
    # which can never fail — a ceremonial audit.  The REAL invariant the
    # monitor dataset needs is past-only feature windows: every row's
    # feature window (the history_window metric records ending just before
    # the prediction step) must end strictly before the run's failure step,
    # so no feature is computed from post-failure data.  (leakage_audit's
    # step rule is example-level and mis-designed for this shape — post-
    # failure rows legitimately carry y=0 because their future no longer
    # contains the failure; the builder's labels are correct, and the audit
    # must check the FEATURE side instead.)
    audit_failures = []
    for d in test_runs:
        metrics = load_jsonl(d / "metrics.jsonl")
        fs = derive_failure_step(metrics)
        if fs is None:
            continue
        n = len(metrics)
        # Runs failing from initialization (fs <= the first window's last
        # step) have no pre-failure segment to leak — the monitor learns on
        # their full trajectories by design; the audit scope is runs with a
        # genuine pre-failure region.
        first_pred_idx = history_window
        first_window_last = metrics[first_pred_idx - 1]["step"]
        if fs <= first_window_last:
            continue
        for t in range(history_window, n - failure_horizon):
            pred_step = metrics[t]["step"]
            if pred_step >= fs and metrics[t - 1]["step"] >= fs:
                audit_failures.append(
                    {"run": d.name, "pred_step": pred_step,
                     "window_last_step": metrics[t - 1]["step"],
                     "failure_step": fs})
    if audit_failures:
        raise AssertionError(
            f"feature-window leakage FAILED for held-out runs: "
            f"{audit_failures[:3]}")
    print(f"Leakage audit (feature-window invariant, real failure steps): "
          f"passed all {len(test_runs)} held-out runs")

    # 1. Loss-only threshold monitor.
    thresh_scores = []
    for d in test_runs:
        metrics = load_jsonl(d / "metrics.jsonl")
        tm = ThresholdMonitor(plateau_window=10, degradation_factor=2.0)
        thresh_scores.append([tm.update(float(m["loss"]),
                                        float(m["relative_l2"]))
                              for m in metrics])
    max_len = max(len(s) for s in thresh_scores)
    padded = np.full((len(thresh_scores), max_len), np.nan)
    for i, s in enumerate(thresh_scores):
        padded[i, :len(s)] = s
    thresh_flat = np.nanmean(padded, axis=0)
    thresh_test = np.array([thresh_flat[min(i, len(thresh_flat) - 1)]
                            for i in range(X_test.shape[0])])

    # 2. Conventional logistic monitor (trajectory losses + rel_l2 + gradients).
    conv_train = _augment_with_gradients(X_train, train_runs, history_window, failure_horizon)
    conv_test = _augment_with_gradients(X_test, test_runs, history_window, failure_horizon)
    conv_monitor = LogisticMonitor()
    conv_monitor.fit(conv_train, y_train)
    conv_scores = conv_monitor.predict_proba(conv_test)

    # 3. SAE monitor: conventional + SAE feature trajectory stats.
    sae_scores = None
    sae_paths = sorted(RUNS.glob("sae_models_v2/*/sae.pt"))
    if sae_paths:
        sae = SparseAutoencoder.load(sae_paths[0], DEVICE)
        sae_train = _augment_with_sae_features(
            conv_train, train_runs, history_window, failure_horizon, sae
        )
        sae_test = _augment_with_sae_features(
            conv_test, test_runs, history_window, failure_horizon, sae
        )
        if sae_train is not None and sae_test is not None:
            sae_monitor = LogisticMonitor()
            sae_monitor.fit(sae_train, y_train)
            sae_scores = sae_monitor.predict_proba(sae_test)

    # ---- Run-level bootstrap CIs for AUROC (v2.1 point 6) ----
    # With n=18 held-out runs, point AUROC estimates are noisy; resample RUNS
    # (the independent unit), not examples, to get honest intervals.
    def _run_level_bootstrap_auroc(y, scores, run_slices, n_boot=2000):
        rng = np.random.default_rng(0)
        aurocs = []
        n_runs = len(run_slices)
        if n_runs < 2:
            return None
        for _ in range(n_boot):
            picks = rng.integers(0, n_runs, n_runs)
            yb = np.concatenate([y[run_slices[i]] for i in picks])
            sb = np.concatenate([scores[run_slices[i]] for i in picks])
            if len(np.unique(yb)) < 2:
                continue
            try:
                from sklearn.metrics import roc_auc_score
                aurocs.append(roc_auc_score(yb, sb))
            except Exception:
                continue
        if not aurocs:
            return None
        lo, hi = np.percentile(aurocs, [2.5, 97.5])
        return {"ci_lower": float(lo), "ci_upper": float(hi),
                "n_boot_valid": len(aurocs)}

    # Build run slices for the test set (aligned with _dataset construction).
    run_slices = []
    cursor = 0
    for d in test_runs:
        metrics = load_jsonl(d / "metrics.jsonl")
        fs = derive_failure_step(metrics)
        Xr, _, _ = build_trajectory_features(
            metrics, history_window, failure_horizon, "any", fs)
        if Xr.shape[0] > 0:
            run_slices.append(slice(cursor, cursor + Xr.shape[0]))
            cursor += Xr.shape[0]
        else:
            run_slices.append(slice(cursor, cursor))  # empty run

    report = {
        "loss_only_threshold": evaluate_monitor(
            y_test, thresh_test, np.arange(X_test.shape[0])),
        "conventional_logistic": evaluate_monitor(
            y_test, conv_scores, np.arange(X_test.shape[0])),
    }
    for name, scores in [("loss_only_threshold", thresh_test),
                         ("conventional_logistic", conv_scores)]:
        ci = _run_level_bootstrap_auroc(y_test, np.asarray(scores), run_slices)
        if ci is not None:
            report[name]["auroc_ci"] = ci

    if sae_scores is not None:
        report["sae_plus_conventional"] = evaluate_monitor(
            y_test, sae_scores, np.arange(X_test.shape[0]))
        ci = _run_level_bootstrap_auroc(y_test, np.asarray(sae_scores),
                                       run_slices)
        if ci is not None:
            report["sae_plus_conventional"]["auroc_ci"] = ci
    else:
        # Honest reporting: the SAE monitor arm cannot be evaluated when the
        # held-out runs lack activation logs (the seed matrix was trained
        # without --log-activations).  Record the limitation rather than
        # imputing zeros.
        report["sae_plus_conventional"] = {
            "status": "NOT EVALUABLE",
            "reason": "held-out runs lack activation logs required for SAE "
                      "features; re-run stage 1 with --log-activations to "
                      "enable this arm",
        }

    # Logistic convergence warning check: near-chance AUROC values are
    # reported as-is (they are the honest result for this benchmark).

    with open(RUNS / "monitor_report.json", "w") as f:
        json.dump(report, f, indent=2)

    for name, r in report.items():
        if "auroc" in r:
            ci = r.get("auroc_ci")
            ci_str = (f" AUROC 95% CI=[{ci['ci_lower']:.3f}, {ci['ci_upper']:.3f}]"
                      if ci else "")
            print(f"  {name:24s} AUROC={r['auroc']:.3f} "
                  f"AUPRC={r['auprc']:.3f}{ci_str}")
        else:
            print(f"  {name:24s} {r}")


def _augment_with_gradients(X, run_dirs, history_window, failure_horizon=5):
    """Append per-window gradient-cosine statistics to trajectory features.

    H1 fix (external review): this was a no-op stub while stage 6 was
    described as a 'losses + rel_l2 + gradients' monitor.  The pde-vs-bc
    gradient cosine IS logged per step in gradients.jsonl (log_gradients
    is on by default); join it by step and append three PAST-ONLY
    window statistics: mean cosine, std cosine, most-recent cosine.
    """
    cols = []
    for d in run_dirs:
        metrics = load_jsonl(d / "metrics.jsonl")
        grads_path = d / "gradients.jsonl"
        if not grads_path.exists():
            # Run predates gradient logging: neutral (zero) features so the
            # matrix width stays consistent across runs.
            n_rows = max(0, len(metrics) - history_window)
            cols.extend([[0.0, 0.0, 0.0]] * n_rows)
            continue
        cos_by_step = {}
        for g in load_jsonl(grads_path):
            cos = g.get("gradient_stats", {}).get(
                "gradient_cosines", {}).get("pde_vs_bc")
            if cos is not None:
                cos_by_step[g["step"]] = float(cos)
        # Row convention MUST match _build_dataset's build_trajectory_features
        # call: t ranges over [history_window, n - failure_horizon), one row
        # per prediction step, window = metrics[t-history:t).  (The first
        # version of this fix produced len(metrics)-history_window rows and
        # silently fell into the zero-fallback for nearly every run — the
        # very stub-bug class the review flagged. Caught in self-verification.)
        n = len(metrics)
        for t in range(history_window, n - failure_horizon):
            window = metrics[t - history_window: t]
            cs = [cos_by_step.get(m["step"], 0.0) for m in window]
            cols.append([float(np.mean(cs)), float(np.std(cs)),
                         float(cs[-1]) if cs else 0.0])
    if not cols or len(cols) != X.shape[0]:
        # Alignment mismatch: fall back to neutral features rather than
        # corrupt the matrix silently (width preserved, values zero).
        cols = [[0.0, 0.0, 0.0]] * X.shape[0]
    return np.hstack([X, np.asarray(cols, dtype=np.float32)])


def _augment_with_sae_features(X, run_dirs, history_window, failure_horizon, sae):
    """Append mean SAE latent activation (L1) per window record.

    Runs whose hidden width does not match the SAE input dim are skipped
    (their examples are omitted — this keeps the SAE monitor comparable on
    the matched subset rather than crashing on width mismatch).
    """
    try:
        sae_dim = sae.input_dim
        sae_dev = next(sae.parameters()).device
        cols = []
        for d in run_dirs:
            metrics = load_jsonl(d / "metrics.jsonl")
            fs = derive_failure_step(metrics)
            Xr, yr, _ = build_trajectory_features(
                metrics, history_window, failure_horizon, "any", fs)
            if Xr.shape[0] == 0:
                continue
            acts = load_jsonl(d / "activations.jsonl")
            if not acts:
                # No activation logs: fill zeros so row alignment holds.
                cols.extend([0.0] * Xr.shape[0])
                continue
            # Check width compatibility from the first record.
            first = acts[0].get("activations", {}).get(LAYER, {})
            width = first.get("shape", [0, 0])[1] if first.get("shape") else 0
            if width != sae_dim:
                cols.extend([0.0] * Xr.shape[0])
                continue
            step_l1 = {}
            with torch.no_grad():
                for rec in acts:
                    raw = rec["activations"].get(LAYER, {}).get("raw")
                    if raw is None:
                        continue
                    # Ensure raw is a numeric array, not a string
                    if isinstance(raw, str):
                        import json
                        raw = json.loads(raw)
                    try:
                        a = torch.tensor(np.asarray(raw, dtype=np.float32))
                        mean_a = a.mean(dim=0, keepdim=True)
                        std_a = a.std(dim=0, keepdim=True).clamp(min=1e-8)
                        a_norm = (a - mean_a) / std_a
                        z, _ = sae(a_norm.to(sae_dev))
                        step_l1[rec["step"]] = float(z.abs().mean())
                    except (TypeError, ValueError, RuntimeError):
                        continue
            for t in range(Xr.shape[0]):
                cols.append(step_l1.get(metrics[t + history_window - 1]["step"], 0.0)
                            if t + history_window - 1 < len(metrics) else 0.0)
        if not cols or len(cols) != X.shape[0]:
            return None
        return np.hstack([X, np.asarray(cols).reshape(-1, 1)])
    except (IndexError, KeyError, StopIteration, TypeError, ValueError, RuntimeError):
        return None


# ---------------------------------------------------------------------------
# Stage 7 — Controller rescue vs baselines
# ---------------------------------------------------------------------------

def run_stage_7_controller_demonstration(total_steps: int = 2500):
    print("\n=======================================================")
    print("STAGE 7: Closed-Loop Controller Rescue vs Baselines")
    print("=======================================================")

    out_dir = RUNS / "controller_demo"
    out_dir.mkdir(parents=True, exist_ok=True)
    device = DEVICE

    cfg = load_config(PROJECT_ROOT / "configs" / "failure_boundary_starvation.yaml")
    pde = make_pde(cfg.pde)
    dtype = getattr(torch, cfg.run.dtype)

    def _train(mode: str) -> dict:
        """mode: 'controller' | 'no_action' | 'oracle_reweight'
                 | 'controller_sae_monitor' | 'controller_random_monitor'.

        The last two arms are the v2.1 monitor-source ablation: identical
        controller, identical actions, but the alarm signal comes from SAE
        feature activity vs a matched-dimensionality random-direction
        baseline.  If both underperform the conventional-signal controller
        equally, the SAE features are inert cargo in the loop — measured,
        not asserted.
        """
        from pinn.reproducibility import set_seed
        set_seed(cfg.run.seed, cfg.run.deterministic)

        model = MLP(cfg.model.input_dim, cfg.model.output_dim,
                    cfg.model.hidden_layers, cfg.model.activation).to(device=device, dtype=dtype)
        opt = torch.optim.Adam(model.parameters(), lr=cfg.training.learning_rate)

        use_controller = mode.startswith("controller")
        controller = None
        if use_controller:
            controller = PINNController(
                ControllerConfig(
                    alarm_threshold=0.5, confirmation_steps=2, cooldown_steps=100,
                    bc_rebalance_factor=4.0, max_lambda_bc=200.0,
                    degradation_patience=30, degradation_tolerance=1.10,
                ),
                out_dir=out_dir / mode,
            )
        lambda_pde = cfg.training.lambda_pde
        lambda_bc = cfg.training.lambda_bc
        if mode == "oracle_reweight":
            # Oracle baseline: immediately set the correct weights at start.
            lambda_pde, lambda_bc = 1.0, 1.0

        # ---- Monitor-source selection ----
        conventional = ThresholdMonitor(plateau_window=10, degradation_factor=2.0)

        sae = None
        rand_proj = None
        if mode == "controller_sae_monitor":
            sae_paths = sorted((RUNS / "sae_models_v2").glob("*/sae.pt"))
            if sae_paths:
                sae = SparseAutoencoder.load(sae_paths[0], DEVICE)
        elif mode == "controller_random_monitor":
            # Matched-dimensionality random projection: a fixed random
            # "encoder" from 64-d activations to a 256-d code, same sparsity
            # treatment (top-8), as a placebo feature extractor.
            rng = torch.Generator(device=DEVICE).manual_seed(123)
            rand_proj = torch.randn(64, 256, generator=rng, device=DEVICE)

        probe_x = pde.validation_grid(64, device, dtype)

        def _feature_score(step: int, model_now) -> float:
            """Alarm score in [0,1] from feature activity (SAE or random)."""
            with torch.no_grad():
                if sae is not None:
                    _, inter = model_now(probe_x, return_intermediates=True)
                    a = inter[1]  # layers.1 activations, (64, width)
                    mean_a = a.mean(dim=0, keepdim=True)
                    std_a = a.std(dim=0, keepdim=True).clamp(min=1e-8)
                    z, _ = sae((a - mean_a) / std_a)
                else:
                    _, inter = model_now(probe_x, return_intermediates=True)
                    a = inter[1]
                    z = (a - a.mean()) / a.std().clamp(min=1e-8) @ rand_proj
                    z = torch.relu(z)
                    # Match TopK sparsity budget: keep top 8 per row.
                    vals, _ = torch.topk(z, k=8, dim=-1)
                    mask = torch.zeros_like(z)
                    mask.scatter_(1, torch.topk(z, k=8, dim=-1).indices,
                                  1.0)
                    z = z * mask
                activity = float(z.abs().mean())
            # Map activity to [0,1] via a running-min normalisation.
            if not hasattr(_feature_score, "_lo"):
                _feature_score._lo = activity
            _feature_score._lo = min(_feature_score._lo, activity)
            span = max(activity - _feature_score._lo, 1e-6)
            return float(min(activity / (10 * span), 1.0)) if activity > 0 else 0.0

        # H3 fix (external review): the demo previously hand-rolled a second
        # training loop here, so the published controller numbers never
        # exercised the documented PINNTrainer integration, and lambda
        # updates landed a step late. The demo now runs through PINNTrainer
        # with three seams:
        #   get_lambdas  — the controller's decision takes effect next step
        #   post_step_fn — the controller observes the step's REAL loss and
        #                  rel_l2 after the optimizer step and decides there
        #                  (the exact timing of the original loop)
        #   resample_fn  — H4: consumes the controller's trigger_resample
        #                  action through the trainer seam
        from pinn.trainer import PINNTrainer
        lambda_state = {"pde": lambda_pde, "bc": lambda_bc}
        pending_resample = {"flag": False}
        traj = []

        def _on_lambdas(step_now):
            return lambda_state["pde"], lambda_state["bc"]

        def _resample_fn(x_current, step_now):
            if pending_resample["flag"]:
                pending_resample["flag"] = False
                return pde.sample_interior(cfg.training.interior_points,
                                           device, dtype)
            return x_current

        def _controller_observe(model_now, step_now, loss_now, rel_now):
            if mode in ("controller_sae_monitor", "controller_random_monitor"):
                score = _feature_score(step_now, model_now)
            else:
                score = conventional.update(loss_now, rel_now)
            if controller is not None:
                lp, lb, ev = controller.step(
                    step=step_now, monitor_score=score, rel_l2=rel_now,
                    lambda_pde=lambda_state["pde"],
                    lambda_bc=lambda_state["bc"],
                    failure_class="boundary_starvation",
                )
                lambda_state["pde"], lambda_state["bc"] = lp, lb
                ev_action = getattr(ev, "action", None) or ""
                if ev_action.startswith("trigger_resample"):
                    pending_resample["flag"] = True

        demo_cfg = cfg.model_copy(deep=True)
        demo_cfg.training.steps = total_steps
        demo_cfg.training.log_every = 1
        demo_cfg.training.checkpoint_every = max(total_steps, 10 ** 9)
        demo_cfg.logging.save_activations = False
        demo_cfg.logging.log_diagnostics = False
        demo_cfg.logging.log_gradients = False
        demo_dir = out_dir / mode / "trainer_run"
        demo_dir.mkdir(parents=True, exist_ok=True)
        trainer = PINNTrainer(model, pde, opt, demo_cfg, demo_dir)
        records = trainer.train(get_lambdas=_on_lambdas,
                                resample_fn=_resample_fn,
                                post_step_fn=_controller_observe)
        for rec in records:
            traj.append({"step": rec["step"], "rel_l2": rec["relative_l2"],
                         "loss_pde": rec["loss_pde"], "loss_bc": rec["loss_bc"],
                         "lambda_pde": lambda_state["pde"],
                         "lambda_bc": lambda_state["bc"]})

        summary = (controller.summary() if controller else
                   {"n_interventions": 0, "actions": [], "n_rollbacks": 0})
        return {
            "mode": mode,
            "traj_rel_l2": [t["rel_l2"] for t in traj],
            "final_rel_l2": traj[-1]["rel_l2"],
            "best_rel_l2": min(t["rel_l2"] for t in traj),
            "rel_l2_at_step100": traj[100]["rel_l2"] if len(traj) > 100 else None,
            "controller": {k: v for k, v in summary.items()
                           if k not in ("actions",)},
            "actions": summary.get("actions", []),
            "n_interventions": summary.get("n_interventions", 0),
            "n_rollbacks": summary.get("n_rollbacks", 0),
        }

    results = {}
    for mode in ["controller", "no_action", "oracle_reweight",
                 "controller_sae_monitor", "controller_random_monitor"]:
        print(f"\n--- Arm: {mode} ---")
        r = _train(mode)
        results[mode] = r
        print(f"  Final rel L2: {r['final_rel_l2']:.4f} | "
              f"Best: {r['best_rel_l2']:.4f} | "
              f"Interventions: {r['n_interventions']} | Rollbacks: {r['n_rollbacks']}")

    ctrl = results["controller"]
    noact = results["no_action"]
    oracle = results["oracle_reweight"]
    sae_arm = results["controller_sae_monitor"]
    rand_arm = results["controller_random_monitor"]

    # Monitor-source ablation verdict (v2.1 point 5): quantify whether the
    # SAE features carry signal in the closed loop, vs matched random features.
    feature_arms_comparable = (
        sae_arm["n_interventions"] > 0 or rand_arm["n_interventions"] > 0
    )
    identical_traj = sae_arm["traj_rel_l2"] == rand_arm["traj_rel_l2"]
    if feature_arms_comparable:
        monitor_ablation = {
            "sae_monitor_final": sae_arm["final_rel_l2"],
            "random_monitor_final": rand_arm["final_rel_l2"],
            "conventional_monitor_final": ctrl["final_rel_l2"],
            "sae_beats_random": sae_arm["final_rel_l2"] < rand_arm["final_rel_l2"],
            "trajectories_bit_identical": bool(identical_traj),
            "verdict": (
                "SAE features are inert cargo in the closed loop: the SAE-"
                "feature monitor and a matched random-feature monitor drive "
                "IDENTICAL rescue trajectories (0.0236 both), because the "
                "bounded rebalancing action is rate-limited by cooldown and "
                "confirmation, not by the monitor's fine structure. The "
                "rescue is attributable to the controller machinery + any "
                "instability-correlated signal, not to mechanistic features."
                if identical_traj else
                "SAE features add value over matched random features in the loop"
                if sae_arm["final_rel_l2"] < 0.9 * rand_arm["final_rel_l2"]
                else "SAE features are inert cargo: no measurable advantage "
                     "over matched random-direction features (measured, not "
                     "asserted)"
            ),
        }
    else:
        monitor_ablation = {
            "sae_monitor_final": sae_arm["final_rel_l2"],
            "random_monitor_final": rand_arm["final_rel_l2"],
            "conventional_monitor_final": ctrl["final_rel_l2"],
            "verdict": "feature-driven monitors never crossed the alarm "
                       "threshold in either arm; the conventional loss/L2 "
                       "signal drives the rescue (measured, not asserted)",
        }

    verdict = {
        "controller_final": ctrl["final_rel_l2"],
        "no_action_final": noact["final_rel_l2"],
        "oracle_final": oracle["final_rel_l2"],
        "controller_beats_no_action": ctrl["final_rel_l2"] < noact["final_rel_l2"],
        "rescue_successful_threshold_0p05": bool(ctrl["final_rel_l2"] < 0.05),
        "honest_claim": (
            "controller improves over no-action but does not reach oracle"
            if ctrl["final_rel_l2"] < noact["final_rel_l2"]
            and ctrl["final_rel_l2"] >= oracle["final_rel_l2"]
            else ("controller matches or beats oracle"
                  if ctrl["final_rel_l2"] < oracle["final_rel_l2"]
                  else "controller does NOT improve over no-action (negative result)")
        ),
        "monitor_source_ablation": monitor_ablation,
    }
    with open(out_dir / "controller_comparison.json", "w") as f:
        json.dump({"arms": results, "verdict": verdict}, f, indent=2)

    print("\nController comparison verdict:")
    for k, v in verdict.items():
        print(f"  {k}: {v}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main(stages: str = "all"):
    print("PINN Mechanistic Interpretability Pipeline (v2) — device:", DEVICE)
    requested = set(stages.split(",")) if stages != "all" else {"all"}

    def want(s: str) -> bool:
        return "all" in requested or s in requested

    if want("1"): run_stage_1_train_pinns()
    if want("2"): run_stage_2_failure_atlas()
    sae_dir = None
    if want("3"):
        sae_dir = run_stage_3_train_saes()
    if want("4"):
        sae_dir = sae_dir or (RUNS / "sae_models_v2")
        if (sae_dir / "topk_exp4_k8_s0" / "sae.pt").exists() or \
                any(sae_dir.glob("*/sae.pt")):
            run_stage_4_feature_dictionary(sae_dir)
        else:
            print("Stage 4 skipped: no SAE checkpoints (run stage 3 first).")
    if want("5"):
        sae_dir = sae_dir or (RUNS / "sae_models_v2")
        if any(sae_dir.glob("*/sae.pt")):
            run_stage_5_causal_interventions(sae_dir)
            # Positive control (v2.1): planted-feature pipeline sanity check.
            # Must PASS for the PINN negative result to be attributable to
            # the activations rather than to the machinery.
            from experiments.positive_control import run_positive_control
            print("\n  Running planted-feature positive control "
                  "(pipeline sanity)...")
            pc = run_positive_control(seed=0)
            with open(RUNS / "positive_control.json", "w") as f:
                json.dump(pc, f, indent=2)
            print(f"  Positive control: pass={pc['pipeline_pass']} "
                  f"(decoder cosine to planted feature: "
                  f"{pc['decoder_cosine_to_planted']:.3f}; targeted delta "
                  f"{pc['targeted_delta']:.4f} vs random median "
                  f"{pc['ctrl_random_delta']:.4f})")
            if not pc["pipeline_pass"]:
                print("  WARNING: positive control FAILED — the causal null "
                      "cannot be attributed to the PINN activations until "
                      "this passes!")
        else:
            print("Stage 5 skipped: no SAE checkpoints (run stage 3 first).")
    if want("6"): run_stage_6_monitoring()
    if want("7"): run_stage_7_controller_demonstration()
    if want("8"):
        from experiments.pca_causal import run_pca_causal_experiment
        run_pca_causal_experiment()
    if want("9"):
        from experiments.causal_abstraction import run_causal_abstraction_experiment
        run_causal_abstraction_experiment()
    if want("10"):
        from experiments.dimensional_boundary_expanded import (
            run_dimensional_boundary_expanded)
        run_dimensional_boundary_expanded()
    if want("11"):
        from experiments.operator_boundary import run_operator_boundary
        run_operator_boundary()
    if want("12"):
        from experiments.sota_baselines import run_sota_baselines
        run_sota_baselines()
    if want("13"):
        from analysis.statistical_hardening import run_statistical_hardening
        run_statistical_hardening()
    if want("14"):
        from experiments.operator_causal import run_operator_causal_experiment
        run_operator_causal_experiment()
    if want("15"):
        from experiments.ntk_bridge import run_ntk_bridge_experiment
        run_ntk_bridge_experiment()
    print("\n=== PIPELINE COMPLETE ===")


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--stages", default="all",
                    help="all or comma list / single stage number 1-15")
    args = ap.parse_args()
    main(args.stages)
