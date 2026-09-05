"""
End-to-End Experimental Execution Pipeline for PINN Mechanistic Interpretability.

Stages:
  1. Train PINNs across 7 configs (baselines + 4 failure modes) with full logging.
  2. Index the Failure Atlas.
  3. Train Sparse Autoencoders (SAEs) on recorded activation datasets.
  4. Extract Physics-Feature Dictionary (associations & cross-seed matching).
  5. Perform Causal Interventions & counterfactual evaluation.
  6. Train & evaluate Early-Warning Monitors.
  7. Run Closed-Loop Adaptive PINN Controller demonstration.
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
from sae.dataset import ActivationDataset
from sae.train import sweep_sae
from sae.features import compute_feature_stats, associate_features_with_metrics, match_features_across_saes, build_feature_dictionary, save_feature_dictionary
from analysis.failure_atlas import aggregate_failure_atlas
from analysis.feature_dictionary import render_feature_dictionary_markdown
from interventions.causal import run_inference_interventions
from monitoring.features import build_monitor_dataset, leakage_audit
from monitoring.models import ThresholdMonitor
from controller.state_machine import PINNController, ControllerConfig


PROJECT_ROOT = Path("/home/shamique/projects/Pinn research/pinn_mechanistic_starter")


def run_stage_1_train_pinns():
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
            "--steps", "2000",
            "--log-gradients",
            "--log-activations",
            "--act-save-raw",
            "--log-diagnostics",
        ]
        res = subprocess.run(cmd, cwd=PROJECT_ROOT, capture_output=True, text=True)
        if res.returncode == 0:
            print(f"  Successfully finished training for {cfg_rel}")
        else:
            print(f"  Training error for {cfg_rel}: {res.stderr[:300]}")


def run_stage_2_failure_atlas():
    print("\n=======================================================")
    print("STAGE 2: Generating Failure Atlas Index")
    print("=======================================================")
    runs_dir = PROJECT_ROOT / "runs"
    atlas_dir = runs_dir / "failure_atlas"
    entries = aggregate_failure_atlas(runs_dir, atlas_dir)
    print(f"Indexed {len(entries)} runs into Failure Atlas.")
    return entries


def run_stage_3_train_saes():
    print("\n=======================================================")
    print("STAGE 3: Training Sparse Autoencoders on Hidden Activations")
    print("=======================================================")
    runs_dir = PROJECT_ROOT / "runs"

    # Filter runs to width-64 hidden layers so activation shapes match
    valid_run_dirs = []
    for d in runs_dir.iterdir():
        if not d.is_dir() or not (d / "activations.jsonl").exists():
            continue
        # Check hidden dim in first record
        recs = load_jsonl(d / "activations.jsonl")
        if recs:
            acts = recs[0].get("activations", {})
            layer_data = acts.get("layers.1", acts.get("net.2", {}))
            shape = layer_data.get("shape", [])
            if len(shape) >= 2 and shape[1] == 64:
                valid_run_dirs.append(d)

    print(f"Found {len(valid_run_dirs)} runs with matching activation shape [50, 64].")
    if not valid_run_dirs:
        return None

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    out_dir = runs_dir / "sae_models"

    sae_models = sweep_sae(
        run_dirs=valid_run_dirs,
        layer_name="layers.1",
        expansion_values=[2, 4],
        sparsity_values=[1e-3, 1e-2],
        steps=500,
        batch_size=128,
        learning_rate=1e-3,
        device=device,
        out_dir=out_dir,
    )
    print(f"Trained {len(sae_models)} SAE configurations.")
    return out_dir


def run_stage_4_feature_dictionary(sae_dir: Path):
    print("\n=======================================================")
    print("STAGE 4: Building Physics-Feature Dictionary")
    print("=======================================================")
    sae_paths = list(sae_dir.glob("*/sae.pt"))
    if not sae_paths:
        sae_paths = list(sae_dir.glob("sae_*.pt"))
    if not sae_paths:
        print("No trained SAE checkpoints found for dictionary extraction.")
        return None

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    runs_dir = PROJECT_ROOT / "runs"
    run_dirs = [d for d in runs_dir.iterdir() if d.is_dir() and (d / "activations.jsonl").exists()]

    # Filter to width 64 runs
    valid_run_dirs = []
    for d in run_dirs:
        recs = load_jsonl(d / "activations.jsonl")
        if recs:
            acts = recs[0].get("activations", {})
            layer_data = acts.get("layers.1", acts.get("net.2", {}))
            shape = layer_data.get("shape", [])
            if len(shape) >= 2 and shape[1] == 64:
                valid_run_dirs.append(d)

    ds = ActivationDataset(valid_run_dirs, "layers.1", normalise=True)
    sae = SparseAutoencoder.load(sae_paths[0], device)

    stats = compute_feature_stats(sae, ds.data, device)
    metric_arrays = {"data_std": ds.data.cpu().numpy().std(axis=1)}
    corrs = associate_features_with_metrics(sae, ds.data, metric_arrays, device)

    loaded_saes = [SparseAutoencoder.load(p, device) for p in sae_paths[:3]]
    families = match_features_across_saes(loaded_saes) if len(loaded_saes) >= 2 else []

    dictionary = build_feature_dictionary(
        feature_stats=stats,
        correlations=corrs,
        families=families,
        layer_name="layers.1",
        sae_version="v1.0",
    )

    out_json = runs_dir / "feature_dictionary" / "physics_feature_dictionary.json"
    out_md = runs_dir / "feature_dictionary" / "physics_feature_dictionary.md"

    save_feature_dictionary(dictionary, out_json)
    render_feature_dictionary_markdown(dictionary, out_md)

    print(f"Feature Dictionary created with {len(dictionary)} features.")
    return dictionary


def run_stage_5_causal_interventions(sae_dir: Path):
    print("\n=======================================================")
    print("STAGE 5: Causal Intervention & Specificity Evaluation")
    print("=======================================================")
    sae_paths = list(sae_dir.glob("*/sae.pt"))
    if not sae_paths:
        sae_paths = list(sae_dir.glob("sae_*.pt"))
    if not sae_paths:
        print("No SAE path found for interventions.")
        return

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    run_dir = PROJECT_ROOT / "runs" / "poisson_baseline"
    if not (run_dir / "config.json").exists():
        print("No baseline run found for causal testing.")
        return

    cfg = load_config(run_dir / "config.json")
    pde = make_pde(cfg.pde)
    model = MLP(cfg.model.input_dim, cfg.model.output_dim, cfg.model.hidden_layers, cfg.model.activation).to(device)

    ckpts = sorted(run_dir.glob("checkpoint_*.pt"))
    if ckpts:
        ckpt = torch.load(ckpts[-1], map_location=device, weights_only=False)
        model.load_state_dict(ckpt["model"])

    sae = SparseAutoencoder.load(sae_paths[0], device)

    scores = run_inference_interventions(
        model=model,
        sae=sae,
        pde=pde,
        device=device,
        dtype=getattr(torch, cfg.run.dtype),
        candidate_features=[0, 1, 2, 3],
        target_loss="pde",
        layer_index=1,
    )

    out_path = PROJECT_ROOT / "runs" / "causal_intervention_results.json"
    with open(out_path, "w") as f:
        json.dump(scores, f, indent=2)
    print(f"Causal intervention evaluations written to {out_path}")


def run_stage_6_monitoring():
    print("\n=======================================================")
    print("STAGE 6: Early-Warning Monitoring Suite Evaluation")
    print("=======================================================")
    runs_dir = PROJECT_ROOT / "runs"
    run_dirs = [d for d in runs_dir.iterdir() if d.is_dir() and (d / "metrics.jsonl").exists()]

    if not run_dirs:
        print("No metrics logs found for monitor dataset generation.")
        return

    failure_steps = {
        d.stem: 500 if "failure" in d.stem or "starvation" in d.stem or "conflict" in d.stem else -1
        for d in run_dirs
    }

    X, y = build_monitor_dataset(run_dirs, failure_steps=failure_steps, history_window=10, failure_horizon=5)
    print(f"Monitor dataset built: X shape={X.shape}, y shape={y.shape}")

    thresh_mon = ThresholdMonitor(plateau_window=10, degradation_factor=2.0)
    metrics = load_jsonl(run_dirs[0] / "metrics.jsonl")
    scores = [thresh_mon.update(float(m["loss"]), float(m["relative_l2"])) for m in metrics]
    print(f"Threshold monitor active. Max alarm score on run 0: {max(scores):.4f}")


def run_stage_7_controller_demonstration():
    print("\n=======================================================")
    print("STAGE 7: Closed-Loop Adaptive Controller Rescue Run")
    print("=======================================================")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    out_dir = PROJECT_ROOT / "runs" / "controller_demo"
    out_dir.mkdir(parents=True, exist_ok=True)

    ctrl_cfg = ControllerConfig(alarm_threshold=0.5, confirmation_steps=2, cooldown_steps=100)
    controller = PINNController(ctrl_cfg, out_dir=out_dir)

    cfg = load_config(PROJECT_ROOT / "configs" / "failure_boundary_starvation.yaml")
    pde = make_pde(cfg.pde)
    model = MLP(cfg.model.input_dim, cfg.model.output_dim, cfg.model.hidden_layers, cfg.model.activation).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=cfg.training.learning_rate)

    lambda_pde = cfg.training.lambda_pde
    lambda_bc = cfg.training.lambda_bc
    monitor = ThresholdMonitor(plateau_window=10, degradation_factor=2.0)

    print("Running adaptive training with closed-loop controller...")
    rel_l2 = 1.0
    for step in range(500):
        opt.zero_grad()
        x_int = pde.sample_interior(cfg.training.interior_points, device, torch.float32)
        r_pde = pde.residual(model, x_int)
        loss_pde = (r_pde ** 2).mean()

        r_bc = pde.boundary_residual(model)
        loss_bc = (r_bc ** 2).mean()

        total_loss = lambda_pde * loss_pde + lambda_bc * loss_bc
        total_loss.backward()
        opt.step()

        with torch.no_grad():
            x_val = pde.validation_grid(100, device, torch.float32)
            u_pred = model(x_val)
            u_exact = pde.exact(x_val)
            rel_l2 = float(torch.linalg.vector_norm(u_pred - u_exact) / torch.linalg.vector_norm(u_exact))

        score = monitor.update(float(total_loss), rel_l2)
        lambda_pde, lambda_bc, event = controller.step(
            step=step,
            monitor_score=score,
            rel_l2=rel_l2,
            lambda_pde=lambda_pde,
            lambda_bc=lambda_bc,
            failure_class="boundary_starvation",
        )

    summary = controller.summary()
    print("Closed-Loop Controller Run Complete!")
    print(f"  Total interventions applied: {summary['n_interventions']}")
    print(f"  Actions taken: {summary['actions']}")
    print(f"  Final Relative L2 Error: {rel_l2:.6f}")


def main():
    print("Starting Full PINN Mechanistic Interpretability Research Pipeline...")
    run_stage_1_train_pinns()
    run_stage_2_failure_atlas()
    sae_dir = run_stage_3_train_saes()
    if sae_dir is not None:
        run_stage_4_feature_dictionary(sae_dir)
        run_stage_5_causal_interventions(sae_dir)
    run_stage_6_monitoring()
    run_stage_7_controller_demonstration()
    print("\n=======================================================")
    print("FULL RESEARCH PIPELINE EXECUTION COMPLETE!")
    print("=======================================================")


if __name__ == "__main__":
    main()
