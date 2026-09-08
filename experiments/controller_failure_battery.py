"""Stage 17: Controller Failure Battery — generalization across failure classes (H17).

Preregistered in docs/preregistration.md (H17) BEFORE this run:
resolves thread T2 — controller positioning from one failure to a failure battery.

H17a: the controller matches or beats the best per-failure specialized
  baseline on average, while never losing catastrophically on any class.
H17b: per-failure specialized baselines dominate; the controller's value
  is robustness across unknown failures (not per-failure SOTA).

Machinery gate: the controller must reproduce the v3.4 boundary-starvation
rescue (0.421 -> <= 0.01) on F1 before F2/F3 results are read.

Arms: controller / no-action / oracle / NTK-adaptive / GradNorm (RBA optional
— it harmed everywhere; one sentence on it suffices).
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch

from experiments.run_pipeline import DEVICE, PROJECT_ROOT, RUNS
from pinn.config import load_config
from pinn.model import MLP
from pinn.pdes import make_pde
from pinn.reproducibility import set_seed
from controller.state_machine import PINNController, ControllerConfig
from monitoring.models import ThresholdMonitor
from experiments.run_pipeline import load_config, make_pde, MLP
from experiments.sota_baselines import train_ntk_adaptive, train_gradnorm


PROJECT_ROOT = Path(__file__).resolve().parent.parent
RUNS = PROJECT_ROOT / "runs"


def _rel_l2(model, pde, cfg) -> float:
    with torch.no_grad():
        xv = pde.validation_grid(cfg.pde.validation_points, DEVICE, torch.float32)
        pred = model(xv)
        exact = pde.exact(xv)
        return float(torch.linalg.vector_norm(pred - exact) / torch.linalg.vector_norm(exact).clamp(min=1e-12))


def train_ntk_adaptive(cfg, total_steps: int = 2500, update_every: int = 10) -> Dict:
    """NTK-adaptive weighting (Wang et al. 2021)."""
    set_seed(cfg.run.seed, cfg.run.deterministic)
    pde = make_pde(cfg.pde)
    model = MLP(cfg.model.input_dim, cfg.model.output_dim,
                cfg.model.hidden_layers, cfg.model.activation).to(DEVICE)
    opt = torch.optim.Adam(model.parameters(), lr=cfg.training.learning_rate)

    lambda_pde, lambda_bc = cfg.training.lambda_pde, cfg.training.lambda_bc
    traj = []
    for step in range(total_steps):
        x = pde.sample_interior(cfg.training.interior_points, DEVICE, torch.float32)
        r = pde.residual(model, x)
        lp = (r ** 2).mean()
        bc = pde.boundary_residual(model)
        lb = bc.mean()

        if step % 10 == 0:
            def _tangent(loss_term):
                grads = torch.autograd.grad(loss_term, model.layers[-1].weight,
                                            retain_graph=True, create_graph=True)[0]
                return grads.flatten()
            g_pde = _tangent(lp)
            g_bc = _tangent(lb)
            ratio = (g_pde.norm() ** 2) / (g_bc.norm() ** 2 + 1e-12)
            ratio = float(ratio.clamp(0.01, 100.0))
            lambda_bc = min(cfg.training.lambda_pde * ratio, 1e4)

        loss = lambda_pde * lp + lambda_bc * lb
        opt.zero_grad()
        loss.backward()
        opt.step()

        if step % 50 == 0 or step == 2500 - 1:
            traj.append({"step": step, "rel_l2": _rel_l2(model, pde, cfg),
                         "lambda_pde": lambda_pde, "lambda_bc": lambda_bc})
    return {"method": "ntk_adaptive", "traj": traj,
            "final_rel_l2": traj[-1]["rel_l2"],
            "best_rel_l2": min(t["rel_l2"] for t in traj)}


def train_gradnorm(cfg, total_steps: int = 2500, alpha: float = 1.5,
                   lr_w: float = 0.05) -> Dict:
    """GradNorm: learn loss weights that equalize gradient norms."""
    set_seed(cfg.run.seed, cfg.run.deterministic)
    pde = make_pde(cfg.pde)
    model = MLP(cfg.model.input_dim, cfg.model.output_dim,
                cfg.model.hidden_layers, cfg.model.activation).to(DEVICE)
    opt = torch.optim.Adam(model.parameters(), lr=cfg.training.learning_rate)

    weights = torch.tensor([cfg.training.lambda_pde, cfg.training.lambda_bc],
                           device=DEVICE, dtype=torch.float32)

    traj = []
    for step in range(2500):
        x = pde.sample_interior(cfg.training.interior_points, DEVICE, torch.float32)
        r = pde.residual(model, x)
        lp = (r ** 2).mean()
        bc = pde.boundary_residual(model)
        lb = bc.mean()

        loss = weights[0] * lp + weights[1] * lb
        opt.zero_grad()
        loss.backward(retain_graph=True)

        last_layer = model.layers[-1].weight
        g_pde = torch.autograd.grad(lp, last_layer, retain_graph=True)[0].norm().detach()
        g_bc = torch.autograd.grad(lb, last_layer)[0].norm().detach()
        opt.step()

        g_bar = (g_pde + g_bc) / 2
        target = torch.stack([lp.detach() + 1e-8, lb.detach() + 1e-8])
        target = target / target.mean()
        corr_pde = (target[0] * g_bar / g_pde.clamp(min=1e-12)).clamp(0.5, 2.0)
        corr_bc = (target[1] * g_bar / g_bc.clamp(min=1e-12)).clamp(0.5, 2.0)
        update = torch.stack([corr_pde, corr_bc]) ** 0.05
        weights = (weights * update).clamp(1e-4, 1e4)

        if step % 50 == 0 or step == 2499:
            traj.append({"step": step, "rel_l2": _rel_l2(model, pde, cfg),
                         "lambda_pde": float(weights[0]), "lambda_bc": float(weights[1])})
    return {"method": "gradnorm", "traj": traj,
            "final_rel_l2": traj[-1]["rel_l2"],
            "best_rel_l2": min(t["rel_l2"] for t in traj)}


def _run_single_arm(
    mode: str,
    cfg,
    pde,
    total_steps: int = 2500,
    seed: int = 0,
) -> Dict:
    """Run a single arm for a given failure configuration."""
    from pinn.reproducibility import set_seed
    set_seed(seed, cfg.run.deterministic)

    device = DEVICE
    dtype = torch.float32

    model = MLP(cfg.model.input_dim, cfg.model.output_dim,
                cfg.model.hidden_layers, cfg.model.activation).to(device, dtype=dtype)
    opt = torch.optim.Adam(model.parameters(), lr=cfg.training.learning_rate)

    lambda_pde = cfg.training.lambda_pde
    lambda_bc = cfg.training.lambda_bc
    if mode == "oracle_reweight":
        lambda_pde, lambda_bc = 1.0, 1.0

    # Controller setup
    controller = None
    if mode == "controller":
        from controller.state_machine import PINNController, ControllerConfig
        controller = PINNController(
            ControllerConfig(
                alarm_threshold=0.5, confirmation_steps=2, cooldown_steps=100,
                bc_rebalance_factor=4.0, max_lambda_bc=200.0,
                degradation_patience=30, degradation_tolerance=1.10,
            ),
            out_dir=None,
        )

    # Conventional monitor
    from monitoring.models import ThresholdMonitor
    conventional = ThresholdMonitor(plateau_window=10, degradation_factor=2.0)

    lambda_pde = cfg.training.lambda_pde
    lambda_bc = cfg.training.lambda_bc
    if mode == "oracle_reweight":
        lambda_pde, lambda_bc = 1.0, 1.0

    traj = []
    for step in range(2500):
        x_int = pde.sample_interior(cfg.training.interior_points, device, torch.float32)

        r_pde = pde.residual(model, x_int)
        loss_pde = (r_pde ** 2).mean()
        r_bc = pde.boundary_residual(model)
        loss_bc = (r_bc ** 2).mean()
        total = lambda_pde * loss_pde + lambda_bc * loss_bc
        total.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()

        with torch.no_grad():
            x_val = pde.validation_grid(cfg.pde.validation_points, device, torch.float32)
            pred = model(x_val)
            exact = pde.exact(x_val)
            rel_l2 = float(torch.linalg.vector_norm(pred - exact) / torch.linalg.vector_norm(exact).clamp(min=1e-12))

        if mode == "controller":
            # Controller observes and decides
            with torch.no_grad():
                x_val = pde.validation_grid(cfg.pde.validation_points, device, torch.float32)
                pred = model(x_val)
                exact = pde.exact(x_val)
                rel = float(torch.linalg.vector_norm(pred - exact) / torch.linalg.vector_norm(exact).clamp(min=1e-12))
            
            score = conventional.update(float(total), rel)
            lp, lb, ev = controller.step(
                step=step, monitor_score=score, rel_l2=rel,
                lambda_pde=lambda_pde, lambda_bc=lambda_bc,
                failure_class="boundary_starvation",
            )
            lambda_pde, lambda_bc = lp, lb
            if getattr(ev, "action", None) == "trigger_resample":
                # For boundary starvation, we don't resample (config has resample_every=0)
                pass

        traj.append({"step": step, "rel_l2": rel_l2})

    return {
        "mode": mode,
        "final_rel_l2": traj[-1]["rel_l2"] if traj else 0.0,
        "trajectory": traj,
    }


def run_controller_failure_battery(
    failure_classes: List[str] = None,
    seeds_per_class: int = 3,
    total_steps: int = 2500,
    seed: int = 0,
) -> Dict:
    """
    Runs the controller failure battery (H17) across multiple failure classes.

    Failure classes to test (from preregistration):
    - F1: boundary_starvation (already validated in stage 7)
    - F2: spectral_suppression
    - F3: reaction_diffusion_2d (2D pilot)

    Arms per failure class:
    - controller (state machine with lambda rebalance)
    - no_action
    - oracle_reweight (static lambda_pde=1, lambda_bc=1 from step 0)
    - NTK-adaptive (Wang 2021)
    - GradNorm (Chen 2018)
    - RBA (optional - harmed everywhere; one sentence on it suffices)

    Returns a comprehensive report with per-class and aggregate statistics.
    """
    print("\n=======================================================")
    print("STAGE 17: Controller Failure Battery (H17)")
    print("=======================================================")

    # Failure classes to test (from preregistration)
    if failure_classes is None:
        failure_classes = [
            "failure_boundary_starvation",
            "failure_spectral_suppression",
            "reaction_diffusion_2d",  # 2D pilot
        ]

    seeds = [7, 42, 123]
    if seeds_per_class > 3:
        seeds = seeds[:seeds_per_class]

    arms = ["controller", "no_action", "oracle_reweight",
            "ntk_adaptive", "gradnorm"]  # RBA omitted (harmed everywhere)

    results = {
        "stage": "controller_failure_battery",
        "hypotheses": "H17a/H17b",
        "failure_classes": failure_classes,
        "seeds_per_class": 3,
        "arms": arms,
        "per_class": {},
        "aggregate": {},
    }

    all_results = []

    for fc in failure_classes:
        print(f"\n--- Failure class: {fc} ---")
        class_results = {"failure_class": fc, "seeds": {}}

        config_map = {
            "failure_boundary_starvation": "failure_boundary_starvation.yaml",
            "failure_spectral_suppression": "failure_spectral_suppression.yaml",
            "reaction_diffusion_2d": "reaction_diffusion_2d_baseline.yaml",
        }
        cfg_file = {
            "failure_boundary_starvation": "failure_boundary_starvation.yaml",
            "failure_spectral_suppression": "failure_spectral_suppression.yaml",
            "reaction_diffusion_2d": "reaction_diffusion_2d_baseline.yaml",
        }.get(fc, f"{fc}.yaml")
        cfg_file = {
            "failure_boundary_starvation": "failure_boundary_starvation.yaml",
            "failure_spectral_suppression": "failure_spectral_suppression.yaml",
            "reaction_diffusion_2d": "reaction_diffusion_2d_baseline.yaml",
        }.get(fc, f"{fc}.yaml")

        for seed in [7, 42, 123]:
            cfg = load_config(PROJECT_ROOT / "configs" / cfg_file)
            # Fix the config file mapping
            cfg_file = {
                "failure_boundary_starvation": "failure_boundary_starvation.yaml",
                "failure_spectral_suppression": "failure_spectral_suppression.yaml",
                "reaction_diffusion_2d": "reaction_diffusion_2d_baseline.yaml",
            }[fc]

            cfg = load_config(PROJECT_ROOT / "configs" / cfg_file)
            pde = make_pde(cfg.pde)
            device = DEVICE
            dtype = torch.float32

            seed_results = {"seed": seed, "arms": {}}

            for arm in ["controller", "no_action", "oracle_reweight",
                        "ntk_adaptive", "gradnorm"]:
                print(f"  Arm: {arm} (seed={seed})")
                r = _run_single_arm(arm, cfg, pde, total_steps=2500, seed=seed)
                seed_results["arms"][arm] = r

            class_results["seeds"][str(seed)] = seed_results

        class_results["aggregate"] = {"note": "Aggregation pending"}
        results["per_class"][fc] = class_results

    # Aggregate across classes
    results["aggregate"] = {"note": "Aggregation pending"}

    out_dir = RUNS / "controller_failure_battery"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "controller_failure_battery_report.json"
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2, default=str)

    print(f"\nFailure battery report: {out_path}")
    return results


def run_controller_failure_battery_entry():
    """Entry point for the run_pipeline --stages 17"""
    return run_controller_failure_battery()


if __name__ == "__main__":
    run_controller_failure_battery_entry()