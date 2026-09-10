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

Implementation notes (v3.4.3 machinery fix, made BEFORE any H17 verdict was
read — the original v3.4.2 driver had these defects, recorded here for
provenance):
  1. NTK/GradNorm arms previously routed through _run_single_arm (which
     contains no NTK/GradNorm logic) and silently trained as no-action
     clones; they now call the canonical, tested sota_baselines.py
     implementations that produced the published stage-12 numbers.
  2. failure_class was hard-coded to "boundary_starvation"; it is now
     parameterized per failure config (the controller's action branch
     must follow the run's failure label for the battery to test
     generalization at all).
  3. The preregistered F1 machinery gate (reproduce 0.421 -> <= 0.01)
     was not implemented; it now runs before F2/F3 and voids the battery
     on failure.
  4. Aggregation was a "pending" placeholder; per-class and cross-class
     statistics (per-seed values, means, deltas vs no-action) are now
     computed into the report.
  5. The controller/oracle/no-action arms now run through PINNTrainer
     (the H3 config-faithful seams used by stage 7) instead of a
     hand-rolled loop, matching the published controller protocol.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch

from pinn.config import load_config
from pinn.model import MLP
from pinn.pdes import make_pde
from pinn.reproducibility import set_seed
from pinn.trainer import PINNTrainer
from controller.state_machine import PINNController, ControllerConfig
from monitoring.models import ThresholdMonitor
from experiments.sota_baselines import train_ntk_adaptive, train_gradnorm

PROJECT_ROOT = Path(__file__).resolve().parent.parent
RUNS = PROJECT_ROOT / "runs"
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

SEEDS = [7, 42, 123]
ARMS = ["controller", "no_action", "oracle_reweight", "ntk_adaptive", "gradnorm"]

FAILURE_MATRIX = {
    "failure_boundary_starvation": {
        "config": "failure_boundary_starvation.yaml",
        "failure_class": "boundary_starvation",
    },
    "failure_spectral_suppression": {
        "config": "failure_spectral_suppression.yaml",
        "failure_class": "spectral_suppression",
    },
    "reaction_diffusion_2d": {
        "config": "reaction_diffusion_2d_baseline.yaml",
        # H17 preregistration: the RD-2D pilot enters as an unlabeled /
        # unknown-failure regime — the controller runs in its
        # failure-agnostic configuration, not a class-specific branch.
        "failure_class": "unknown",
    },
}

# Machinery gate (preregistered): the F1 controller rescue must reproduce
# the v3.4 boundary-starvation rescue (stage 7, seed 7: 0.421 -> 0.0002;
# gate bar <= 0.01) before F2/F3 results are read.
GATE_CLASS = "failure_boundary_starvation"
GATE_ARM = "controller"
GATE_SEED = 7          # the v3.4 stage-7 reference seed
GATE_THRESHOLD = 0.01


def _rel_l2(model, pde, cfg) -> float:
    with torch.no_grad():
        xv = pde.validation_grid(cfg.pde.validation_points, DEVICE,
                                 torch.float32)
        pred = model(xv)
        exact = pde.exact(xv)
        return float(torch.linalg.vector_norm(pred - exact)
                     / torch.linalg.vector_norm(exact).clamp(min=1e-12))


def _train_controller_or_baseline_arm(
    mode: str,
    cfg,
    failure_class: str,
    seed: int,
    total_steps: int,
    out_dir: Optional[Path] = None,
) -> Dict:
    """controller / no_action / oracle_reweight through PINNTrainer (H3 seams).

    Mirrors the stage-7 config-faithful protocol exactly: controller lambda
    decisions land via get_lambdas, observation via post_step_fn, triggers
    via resample_fn.  No hand-rolled loss loop.
    """
    set_seed(seed, cfg.run.deterministic)
    pde = make_pde(cfg.pde)
    model = MLP(cfg.model.input_dim, cfg.model.output_dim,
                cfg.model.hidden_layers, cfg.model.activation).to(
                    device=DEVICE, dtype=torch.float32)
    opt = torch.optim.Adam(model.parameters(), lr=cfg.training.learning_rate)

    controller = None
    if mode == "controller":
        controller = PINNController(
            ControllerConfig(
                alarm_threshold=0.5, confirmation_steps=2, cooldown_steps=100,
                bc_rebalance_factor=4.0, max_lambda_bc=200.0,
                degradation_patience=30, degradation_tolerance=1.10,
                verbose=False,
            ),
            out_dir=out_dir,
        )

    lambda_state = {"pde": cfg.training.lambda_pde, "bc": cfg.training.lambda_bc}
    if mode == "oracle_reweight":
        # Oracle baseline: the correct weights from step 0 (as stage 7).
        lambda_state["pde"], lambda_state["bc"] = 1.0, 1.0

    conventional = ThresholdMonitor(plateau_window=10, degradation_factor=2.0)
    pending_resample = {"flag": False}
    lambdas_seen = {"pde": [], "bc": []}

    def _on_lambdas(step_now):
        lambdas_seen["pde"].append(lambda_state["pde"])
        lambdas_seen["bc"].append(lambda_state["bc"])
        return lambda_state["pde"], lambda_state["bc"]

    def _resample_fn(x_current, step_now):
        if pending_resample["flag"]:
            pending_resample["flag"] = False
            return pde.sample_interior(cfg.training.interior_points,
                                       DEVICE, torch.float32)
        return x_current

    def _controller_observe(model_now, step_now, loss_now, rel_now):
        score = conventional.update(float(loss_now), rel_now)
        if controller is not None:
            lp, lb, ev = controller.step(
                step=step_now, monitor_score=score, rel_l2=rel_now,
                lambda_pde=lambda_state["pde"],
                lambda_bc=lambda_state["bc"],
                failure_class=failure_class,
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
    if out_dir is not None:
        run_dir = out_dir / "trainer_run"
    else:
        # Arms without controller event logging still need a trainer
        # out_dir for JSONL logs; use a scratch directory per arm/seed.
        run_dir = (RUNS / "controller_failure_battery" / "_scratch"
                   / f"{cfg.run.name}_{mode}_seed{seed}")
    run_dir.mkdir(parents=True, exist_ok=True)
    trainer = PINNTrainer(model, pde, opt, demo_cfg, run_dir)
    records = trainer.train(get_lambdas=_on_lambdas,
                            resample_fn=_resample_fn,
                            post_step_fn=_controller_observe)

    traj = [{"step": rec["step"], "rel_l2": rec["relative_l2"],
             "loss_pde": rec["loss_pde"], "loss_bc": rec["loss_bc"]}
            for rec in records]

    if mode == "controller" and controller is not None:
        summary = controller.summary()
        interventions = summary["n_interventions"]
        rollbacks = summary["n_rollbacks"]
        actions = summary.get("actions", [])
    else:
        interventions, rollbacks, actions = 0, 0, []

    final_rel_l2 = traj[-1]["rel_l2"]
    best_rel_l2 = min(t["rel_l2"] for t in traj)
    # Time-to-rescue: first step at or below the v3.4 rescue bar.
    rescue_steps = [t["step"] for t in traj if t["rel_l2"] <= GATE_THRESHOLD]
    time_to_rescue = rescue_steps[0] if rescue_steps else None

    return {
        "mode": mode,
        "seed": seed,
        "final_rel_l2": final_rel_l2,
        "best_rel_l2": best_rel_l2,
        "time_to_rescue_le_0p01": time_to_rescue,
        "n_interventions": interventions,
        "n_rollbacks": rollbacks,
        "n_actions": len(actions),
        "traj": traj,
        "final_pde_residual": traj[-1]["loss_pde"],
        "final_bc_residual": traj[-1]["loss_bc"],
    }


def _run_ntk_arm(cfg, seed: int, total_steps: int) -> Dict:
    """Canonical NTK-adaptive baseline (sota_baselines.py, stage-12 protocol)."""
    cfg = cfg.model_copy(deep=True)
    cfg.run.seed = seed
    r = train_ntk_adaptive(cfg, total_steps=total_steps)
    return {
        "mode": "ntk_adaptive", "seed": seed,
        "final_rel_l2": r["final_rel_l2"], "best_rel_l2": r["best_rel_l2"],
        "traj": r["traj"], "final_pde_residual": None, "final_bc_residual": None,
        "n_interventions": None, "n_rollbacks": None, "n_actions": None,
        "time_to_rescue_le_0p01": None,
    }


def _run_gradnorm_arm(cfg, seed: int, total_steps: int) -> Dict:
    """Canonical GradNorm baseline (sota_baselines.py, stage-12 protocol)."""
    cfg = cfg.model_copy(deep=True)
    cfg.run.seed = seed
    r = train_gradnorm(cfg, total_steps=total_steps)
    return {
        "mode": "gradnorm", "seed": seed,
        "final_rel_l2": r["final_rel_l2"], "best_rel_l2": r["best_rel_l2"],
        "traj": r["traj"], "final_pde_residual": None, "final_bc_residual": None,
        "n_interventions": None, "n_rollbacks": None, "n_actions": None,
        "time_to_rescue_le_0p01": None,
    }


def _arm_metrics(arm_results: List[Dict]) -> Dict:
    """Aggregate per-seed results for one (class, arm) cell."""
    finals = [r["final_rel_l2"] for r in arm_results]
    bests = [r["best_rel_l2"] for r in arm_results]
    return {
        "per_seed_final_rel_l2": [round(f, 6) for f in finals],
        "per_seed_best_rel_l2": [round(b, 6) for b in bests],
        "mean_final_rel_l2": float(np.mean(finals)),
        "std_final_rel_l2": float(np.std(finals)),
        "mean_best_rel_l2": float(np.mean(bests)),
        "all_finals": {str(r["seed"]): r["final_rel_l2"] for r in arm_results},
    }


def run_controller_failure_battery(
    failure_classes: Optional[List[str]] = None,
    seeds: Optional[List[int]] = None,
    total_steps: int = 2500,
    skip_gate: bool = False,
) -> Dict:
    """Run the preregistered H17 battery.

    Matrix: {boundary starvation, spectral suppression, RD-2D pilot}
            x {controller, no_action, oracle, NTK-adaptive, GradNorm}
            x 3 seeds.  The F1 machinery gate runs first and voids the
            battery on failure (unless skip_gate, smoke-test only).
    """
    print("\n=======================================================")
    print("STAGE 17: Controller Failure Battery (H17)")
    print("=======================================================")

    if failure_classes is None:
        failure_classes = list(FAILURE_MATRIX.keys())
    if seeds is None:
        seeds = list(SEEDS)

    results = {
        "stage": "controller_failure_battery",
        "hypotheses": "H17a/H17b (docs/preregistration.md §H17)",
        "protocol": "config-faithful (PINNTrainer H3 seams; canonical "
                    "sota_baselines arms; per-class failure_class routing)",
        "failure_classes": failure_classes,
        "seeds": seeds,
        "arms": ARMS,
        "total_steps": total_steps,
        "machinery_gate": {},
        "per_class": {},
        "aggregate": {},
    }

    # ------------------------------------------------------------------
    # Machinery gate (preregistered): F1 controller rescue must reproduce
    # the v3.4 result before F2/F3 are read.  A gate failure voids the run.
    # ------------------------------------------------------------------
    gate = {
        "class": GATE_CLASS,
        "arm": GATE_ARM,
        "seed": GATE_SEED,
        "threshold": GATE_THRESHOLD,
        "reference_v34": {"no_action": 0.421, "controller": 0.0002},
        "gate_seed_final_rel_l2": None,
        "all_seeds_final_rel_l2": {},
        "pass": False,
    }
    if not skip_gate:
        print(f"\n--- Machinery gate: {GATE_CLASS} controller rescue, "
              f"seed {GATE_SEED} (bar: final rel L2 <= {GATE_THRESHOLD}) ---")
        spec = FAILURE_MATRIX[GATE_CLASS]
        cfg = load_config(PROJECT_ROOT / "configs" / spec["config"])
        gate_dir = RUNS / "controller_failure_battery" / "gate"
        gate_arm_result = None
        for seed in seeds:
            r = _train_controller_or_baseline_arm(
                "controller", cfg, spec["failure_class"], seed,
                total_steps, out_dir=gate_dir / f"seed{seed}")
            gate["all_seeds_final_rel_l2"][str(seed)] = r["final_rel_l2"]
            print(f"  seed {seed}: controller final rel L2 = "
                  f"{r['final_rel_l2']:.6f}")
            if seed == GATE_SEED:
                gate_arm_result = r
        if gate_arm_result is not None:
            # The gate reuses this run as the F1/seed-7 controller cell —
            # no duplicate compute, no post-hoc re-run.
            results["_gate_cell_reused"] = gate_arm_result
            gate["gate_seed_final_rel_l2"] = gate_arm_result["final_rel_l2"]
            gate["pass"] = bool(
                gate_arm_result["final_rel_l2"] <= GATE_THRESHOLD)
        print(f"  GATE {'PASS' if gate['pass'] else 'FAIL — battery void'}")
    results["machinery_gate"] = gate

    if not skip_gate and not gate["pass"]:
        # Preregistered rule: a gate failure voids the run — record and exit.
        out_dir = RUNS / "controller_failure_battery"
        out_dir.mkdir(parents=True, exist_ok=True)
        out_path = out_dir / "controller_failure_battery_report.json"
        with open(out_path, "w") as f:
            json.dump(results, f, indent=2, default=str)
        print(f"\nMACHINERY GATE FAILED — battery voided. Report: {out_path}")
        return results

    # ------------------------------------------------------------------
    # Main battery
    # ------------------------------------------------------------------
    for fc in failure_classes:
        spec = FAILURE_MATRIX[fc]
        print(f"\n--- Failure class: {fc} (failure_class="
              f"{spec['failure_class']}) ---")
        cfg_file = spec["config"]
        class_results = {"failure_class": fc,
                         "config": cfg_file,
                         "controller_failure_class": spec["failure_class"],
                         "seeds": {}, "arms": {}}

        for seed in seeds:
            cfg = load_config(PROJECT_ROOT / "configs" / cfg_file)
            seed_dir = (RUNS / "controller_failure_battery" / fc /
                        f"seed{seed}")
            seed_results = {"seed": seed, "arms": {}}
            for arm in ARMS:
                # Gate-reuse: the F1/seed-7 controller run from the
                # machinery gate IS this cell (same config, seed, protocol).
                reuse = (fc == GATE_CLASS and arm == "controller"
                         and seed == GATE_SEED
                         and "_gate_cell_reused" in results
                         and not skip_gate)
                if reuse:
                    r = dict(results["_gate_cell_reused"])
                    print(f"  Arm: {arm} (seed={seed}) [gate-reused run]")
                elif arm == "ntk_adaptive":
                    print(f"  Arm: {arm} (seed={seed})")
                    r = _run_ntk_arm(cfg, seed, total_steps)
                elif arm == "gradnorm":
                    print(f"  Arm: {arm} (seed={seed})")
                    r = _run_gradnorm_arm(cfg, seed, total_steps)
                else:
                    print(f"  Arm: {arm} (seed={seed})")
                    arm_dir = seed_dir / arm if arm == "controller" else None
                    r = _train_controller_or_baseline_arm(
                        arm, cfg, spec["failure_class"], seed,
                        total_steps, out_dir=arm_dir)
                # Retain a downsampled recovery curve (every 50th step +
                # final) in the report; full curves live in the trainer
                # JSONL logs.
                if "traj" in r:
                    r["traj_downsampled"] = [
                        {"step": t["step"], "rel_l2": round(t["rel_l2"], 6)}
                        for t in r["traj"]
                        if t["step"] % 50 == 0 or t["step"] == total_steps - 1]
                    del r["traj"]
                seed_results["arms"][arm] = r
                print(f"    final rel L2: {r['final_rel_l2']:.6f} | "
                      f"best: {r['best_rel_l2']:.6f}")
            class_results["seeds"][str(seed)] = seed_results

        # Per-arm aggregation across seeds (per-seed values retained).
        for arm in ARMS:
            arm_results = [class_results["seeds"][str(s)]["arms"][arm]
                           for s in seeds]
            agg = _arm_metrics(arm_results)
            # Delta vs no-action (per the H17 analysis protocol):
            # dL2 = L2(no_action) - L2(method); >0 means the method helps.
            noact = [class_results["seeds"][str(s)]["arms"]["no_action"]
                     ["final_rel_l2"] for s in seeds]
            meth = [class_results["seeds"][str(s)]["arms"][arm]
                    ["final_rel_l2"] for s in seeds]
            agg["delta_vs_no_action"] = [
                round(n - m, 6) for n, m in zip(noact, meth)]
            rel_imp = [(n - m) / n if n > 1e-12 else None
                       for n, m in zip(noact, meth)]
            agg["relative_improvement_vs_no_action"] = [
                round(x, 6) if x is not None else None for x in rel_imp]
            class_results["arms"][arm] = agg

        # Winner per class (lowest mean final rel L2).
        means = {arm: class_results["arms"][arm]["mean_final_rel_l2"]
                 for arm in ARMS}
        winner = min(means, key=means.get)
        class_results["winner"] = winner
        class_results["mean_final_rel_l2_by_arm"] = {
            arm: round(means[arm], 6) for arm in ARMS}
        print(f"  Winner: {winner} "
              f"(mean finals: {class_results['mean_final_rel_l2_by_arm']})")

        results["per_class"][fc] = class_results

    # ------------------------------------------------------------------
    # Cross-class aggregation
    # ------------------------------------------------------------------
    agg = {"per_class_mean_final_rel_l2": {}, "per_class_winner": {},
           "controller_vs_specialized": {}}
    for fc, cr in results["per_class"].items():
        agg["per_class_mean_final_rel_l2"][fc] = cr["mean_final_rel_l2_by_arm"]
        agg["per_class_winner"][fc] = cr["winner"]

    # Controller vs each specialized arm: mean delta across classes
    # (positive = controller better on average).
    for arm in ["ntk_adaptive", "gradnorm", "no_action", "oracle_reweight"]:
        deltas = []
        for fc in results["per_class"]:
            m_ctrl = results["per_class"][fc]["arms"]["controller"]["mean_final_rel_l2"]
            m_arm = results["per_class"][fc]["arms"][arm]["mean_final_rel_l2"]
            deltas.append(m_arm - m_ctrl)
        agg["controller_vs_specialized"][arm] = {
            "mean_delta_specialized_minus_controller": float(np.mean(deltas)),
            "per_class_deltas": [round(d, 6) for d in deltas],
            "controller_better_on_average": bool(np.mean(deltas) > 0),
        }

    # Worst class for the controller (catastrophic-loss check for H17a).
    ctrl_means = [results["per_class"][fc]["arms"]["controller"]["mean_final_rel_l2"]
                  for fc in results["per_class"]]
    noact_means = [results["per_class"][fc]["arms"]["no_action"]["mean_final_rel_l2"]
                   for fc in results["per_class"]]
    agg["controller_worst_class_mean"] = float(max(ctrl_means))
    agg["controller_never_loses_to_no_action"] = bool(
        all(c <= n for c, n in zip(ctrl_means, noact_means)))

    results["aggregate"] = agg
    results.pop("_gate_cell_reused", None)  # scratch: cell lives in per_class

    out_dir = RUNS / "controller_failure_battery"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "controller_failure_battery_report.json"
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2, default=str)

    print(f"\nFailure battery report: {out_path}")
    return results


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser(
        description="Stage 17: controller failure battery (H17, preregistered)")
    ap.add_argument("--steps", type=int, default=2500)
    ap.add_argument("--classes", type=str, default=None,
                    help="comma list of failure classes (default: all 3)")
    ap.add_argument("--seeds", type=str, default=None,
                    help="comma list of seeds (default: 7,42,123)")
    ap.add_argument("--smoke", action="store_true",
                    help="reduced smoke test: 1 class x 1 seed, gate skipped")
    args = ap.parse_args()

    classes = args.classes.split(",") if args.classes else None
    seeds = [int(s) for s in args.seeds.split(",")] if args.seeds else None
    if args.smoke:
        run_controller_failure_battery(
            failure_classes=["failure_boundary_starvation"],
            seeds=[7], total_steps=200, skip_gate=True)
    else:
        run_controller_failure_battery(
            failure_classes=classes, seeds=seeds, total_steps=args.steps)
