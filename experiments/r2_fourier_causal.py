"""R2: Causal battery on the Fourier-feature PINN (v4.1, the third arrow).

Preregistered in docs/preregistration.md §R2 (committed 2026-09-11,
BEFORE this run). The H18 chain is: rank up -> SAE-reconstruction
advantage (25-56x on the Fourier PINN). The causal arrow is untested
within PINNs — this stage runs the preregistered matched-deletion E_T
battery + the H16 direction-reversal (crossover) machinery on the H18
Fourier-PINN dictionaries.

Design (registered):
  Substrate: the H18 Fourier-feature PINN (width 64, n_freq=32, 1D
  Poisson), 3 seeds — the regime where the SAE reconstruction advantage
  was measured — plus the tanh depth-3 twin as the low-rank control
  arm (expected null; registered for contrast).
  Dictionary: TopK SAE (k=8, expansion=4 — the exact family that showed
  the 25-56x reconstruction advantage), trained on layers.1 activations
  (the H18 measurement site).
  Battery (the H16 protocol, applied through the PINN hook machinery):
    - top-16 candidates by activity, MC-corrected across 16 tests
    - n = 20 held-out collocation batches per feature
    - matched-deletion controls (unrelated active atom + random active
      atom + supervised probe direction), same operator as the target
    - direction-reversal crossover: amplification (alpha=1.5) must move
      the target readout oppositely to ablation (alpha=0), consistent
      across >= 15/20 batches (exact binomial, two-sided, p < 0.05).
  Target readout: the PDE residual loss on held-out collocation
  batches (the stage-5 convention). Non-target readout: the boundary
  residual.

Machinery gate (registered): the planted-feature positive control
through the REAL PINN hook machinery must pass at the same n (the
stage-5 gate re-run at battery scale); a gate failure voids the run.

Decision rule (pre-written):
  >= 1 feature survives Bonferroni at n=20 AND satisfies the crossover
  criterion => the boundary covers interpretability, not just
  compression (rank -> reconstruction -> causality); recorded R2a.
  Otherwise => "superposition is necessary but not sufficient" is the
  recorded conclusion (R2b) — the reconstruction advantage does not
  carry effect-magnitude specificity, and the reconciliation with
  PhysSAE is that their evidence lives below the specificity bar.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, List

import numpy as np
import torch

from experiments.run_pipeline import DEVICE, RUNS
from pinn.config import load_config
from pinn.model import MLP
from pinn.pdes import make_pde
from pinn.reproducibility import set_seed
from sae.model import SparseAutoencoder
from interventions.engine import measure_intervention_effect
from interventions.causal import (
    compute_causal_score, per_feature_causal_pvalues,
)

PROJECT_ROOT = Path(__file__).resolve().parent.parent

SEEDS = [7, 42, 123]
N_CANDIDATES = 16
N_BATCHES = 20
CROSSOVER_BATCHES = 15
ALPHA_ABLATE = 0.0     # ablation: z_k <- 0
ALPHA_AMPLIFY = 1.5
PR_MOVE_THRESHOLD = 3.0


# ---------------------------------------------------------------------------
# Substrate: reload the H18 Fourier / tanh checkpoints
# ---------------------------------------------------------------------------

def _load_h18_run(run_name: str):
    """Load a trained H18 checkpoint + its PDE, and rebuild the model."""
    h18_dir = RUNS / "architecture_boundary" / run_name
    cfg = json.loads((h18_dir / "config.json").read_text())
    from pinn.config import ExperimentConfig
    cfg_obj = ExperimentConfig(**{k: v for k, v in cfg.items()
                                  if k != "run"} | {"run": cfg["run"]})
    model = MLP(cfg_obj.model.input_dim, cfg_obj.model.output_dim,
                cfg_obj.model.hidden_layers, cfg_obj.model.activation,
                fourier_embed=cfg_obj.model.fourier_embed,
                fourier_n_freq=cfg_obj.model.fourier_n_freq,
                fourier_scale=cfg_obj.model.fourier_scale,
                fourier_seed=cfg_obj.run.seed).to(DEVICE)
    ckpts = sorted(h18_dir.glob("checkpoint_*.pt"))
    ckpt = torch.load(ckpts[-1], map_location=DEVICE, weights_only=True)
    model.load_state_dict(ckpt["model"])
    model.eval()
    pde = make_pde(cfg_obj.pde)
    return model, pde, cfg_obj


def _collect_activations(model, pde, cfg, n_points: int = 8192,
                         seed: int = 0) -> torch.Tensor:
    """layers.1 activations on a dense interior grid (the H18 site)."""
    g = torch.Generator().manual_seed(seed)
    x = torch.rand(n_points, 1, generator=g) * 2 - 1
    x = x.to(DEVICE)
    acts = []
    with torch.no_grad():
        for i in range(0, n_points, 4096):
            _, inter = model(x[i:i + 4096], return_intermediates=True)
            acts.append(inter[1].detach().cpu())
    return torch.cat(acts, dim=0)


def _train_sae(acts: torch.Tensor, seed: int) -> SparseAutoencoder:
    """The H18/TopK SAE family (k=8, expansion=4) on layers.1 activations."""
    torch.manual_seed(seed)
    if DEVICE.type == "cuda":
        torch.cuda.manual_seed_all(seed)
    sae = SparseAutoencoder(input_dim=acts.shape[1], latent_expansion=4,
                            activation_mode="topk", topk=8,
                            decoder_normalize=True).to(DEVICE)
    opt = torch.optim.Adam(sae.parameters(), lr=1e-2)
    bank = acts.to(DEVICE)
    n = bank.shape[0]
    rng = torch.Generator(device="cpu").manual_seed(seed + 777)
    for step in range(2000):
        idx = torch.randint(0, n, (256,), generator=rng).to(DEVICE)
        opt.zero_grad()
        loss, _ = sae.loss(bank[idx])
        loss.backward()
        opt.step()
        if sae.decoder_normalize:
            sae._normalise_decoder()
    sae.eval()
    return sae


def _supervised_probe_direction(sae, acts, labels) -> torch.Tensor:
    """Ridge probe in latent space (the probe control)."""
    from sklearn.linear_model import Ridge
    from sklearn.preprocessing import StandardScaler
    with torch.no_grad():
        z, _ = sae(acts.to(DEVICE))
    z_np = z.cpu().numpy()
    z_np = StandardScaler().fit_transform(z_np)
    y = (labels - labels.mean()) / (labels.std() + 1e-9)
    ridge = Ridge(alpha=1.0).fit(z_np, y)
    d = torch.tensor(ridge.coef_, dtype=torch.float32)
    return d / d.norm().clamp(min=1e-8)


def _probe_labels(pde, model, acts: torch.Tensor, seed: int = 0) -> np.ndarray:
    """Supervision for the probe: the residual magnitude at each
    activation's source point.  _collect_activations samples x uniformly
    with a seeded generator; regenerate the SAME x here to pair each
    activation row with its point."""
    g = torch.Generator().manual_seed(seed)
    n_points = acts.shape[0]
    x = (torch.rand(n_points, 1, generator=g) * 2 - 1).to(DEVICE)
    x.requires_grad_(True)  # residual uses autograd derivatives
    with torch.enable_grad():
        r = pde.residual(model, x)
    return (r ** 2).squeeze(-1).detach().cpu().numpy()


# ---------------------------------------------------------------------------
# Machinery gate (registered): planted control through the PINN hook
# ---------------------------------------------------------------------------

def machinery_gate(seed: int = 0) -> Dict:
    """Planted-feature positive control through the REAL PINN hook at
    battery scale (n=20 batches): a synthetic MLP-with-SAE world where
    one feature is planted-causal for the readout; the battery must
    recover it. This is the stage-5/H16 gate logic instantiated on the
    exact measure_intervention_effect path R2 uses."""
    from experiments.positive_control import (
        PlantedFeatureWorld, _SyntheticModel, _SyntheticPDE,
    )
    torch.manual_seed(seed + 100)
    # The established stage-5 planted world (n_features = 2x overcomplete
    # budget, matching the TopK SAE's k=8/256 selectivity): the gate must
    # instantiate the EXACT machinery R2 uses on the EXACT world the
    # published positive control validated.
    n_dim, n_features, sparsity = 64, 128, 0.1
    w = PlantedFeatureWorld(n_dim=n_dim, n_features=n_features,
                            sparsity=sparsity, seed=seed)
    acts = w.sample_activations(10000, seed=seed + 1)
    # SAE through the same trainer R2 uses
    sae = _train_sae(acts, seed)
    # planted readout host + PDE stand-in
    d_t = w.dictionary[:, w.target_feature]
    d_o = w.dictionary[:, w.other_feature]
    probe_x = torch.arange(256, device=DEVICE, dtype=torch.float32) \
        .unsqueeze(1)
    host = _SyntheticModel(acts.to(DEVICE))
    pde = _SyntheticPDE(d_t.to(DEVICE), d_o.to(DEVICE), probe_x)

    # find the SAE feature best matching the planted target
    with torch.no_grad():
        z, _ = sae(acts.to(DEVICE))
    D = sae.W_d.weight.detach().cpu()          # (d, D_latent)
    D = D / D.norm(dim=0, keepdim=True).clamp(min=1e-8)
    cos = D.T @ d_t / (D.norm(dim=0) * d_t.norm() + 1e-12)
    k_star = int(torch.argmax(cos).item())

    # battery-scale check: target ablation beats controls across batches
    n_survive = 0
    for b in range(N_BATCHES):
        rows = []
        for mode in ("ablate", "unrelated_control", "random_direction"):
            r = measure_intervention_effect(
                host, sae, pde, DEVICE, torch.float32,
                feature_idx=k_star, mode=mode, n_interior=256,
                layer_index=0, alpha=ALPHA_ABLATE,
                baseline_mode="natural", random_seed=b,
            )
            rows.append(r)
        t = abs(rows[0]["delta_pde"])
        c = max(abs(rows[1]["delta_pde"]), abs(rows[2]["delta_pde"]))
        n_survive += int(t > c)
    gate_pass = bool(
        n_survive >= CROSSOVER_BATCHES
        and float(cos[k_star]) > 0.5)
    return {
        "planted_cosine": float(cos[k_star]),
        "beats_controls_batches": n_survive,
        "n_batches": N_BATCHES,
        "pipeline_pass": gate_pass,
    }


# ---------------------------------------------------------------------------
# R2 battery (H16 protocol through the PINN hook)
# ---------------------------------------------------------------------------

def r2_battery(model, pde, sae, acts: torch.Tensor, probe_dir,
               seed: int) -> List[Dict]:
    """Top-16 candidates x 20 batches x (ablate/amplify + 3 controls),
    through the standard PINN SAE hook at layers.1."""
    with torch.no_grad():
        z, _ = sae(acts.to(DEVICE))
    activity = z.abs().sum(dim=0)
    candidates = [int(i) for i in
                  torch.argsort(activity, descending=True)[:N_CANDIDATES]]

    rows: List[Dict] = []
    for b in range(N_BATCHES):
        for k in candidates:
            # target: ablation (alpha = 0 -> z_k <- 0)
            t_ab = measure_intervention_effect(
                model, sae, pde, DEVICE, torch.float32,
                feature_idx=k, mode="ablate", n_interior=256,
                layer_index=1, alpha=ALPHA_ABLATE,
                baseline_mode="natural", random_seed=b,
            )
            # target: amplification (alpha = 1.5 -> z_k *= 1.5)
            t_am = measure_intervention_effect(
                model, sae, pde, DEVICE, torch.float32,
                feature_idx=k, mode="amplify", n_interior=256,
                layer_index=1, alpha=ALPHA_AMPLIFY,
                baseline_mode="natural", random_seed=b,
            )
            # control 1: unrelated active atom (matched deletion)
            c_un = measure_intervention_effect(
                model, sae, pde, DEVICE, torch.float32,
                feature_idx=k, mode="unrelated_control", n_interior=256,
                layer_index=1, alpha=ALPHA_ABLATE,
                baseline_mode="natural", random_seed=b,
            )
            # control 2: random active atom (matched deletion)
            c_rd = measure_intervention_effect(
                model, sae, pde, DEVICE, torch.float32,
                feature_idx=k, mode="random_direction", n_interior=256,
                layer_index=1, alpha=ALPHA_ABLATE,
                baseline_mode="natural", random_seed=b,
            )
            # control 3: supervised probe direction
            from interventions.causal import measure_probe_direction_effect
            c_pr = measure_probe_direction_effect(
                model, sae, pde, DEVICE, torch.float32,
                feature_idx=k, probe_direction=probe_dir,
                n_interior=256, layer_index=1,
                baseline_mode="natural",
            )
            score = compute_causal_score(
                target_effect=t_ab["delta_pde"],
                control_unrelated_effect=c_un["delta_pde"],
                control_random_effect=c_rd["delta_pde"],
                control_probe_effect=c_pr["delta_pde"],
                non_target_effects=[t_ab["delta_bc"]],
            )
            # crossover: amplify must move the readout OPPOSITE to ablate
            crossover = bool(
                (t_ab["delta_pde"] > 0) != (t_am["delta_pde"] > 0))
            rows.append({
                "feature_idx": k, "batch": b, "seed": seed,
                "target_ablate_delta_pde": t_ab["delta_pde"],
                "target_amplify_delta_pde": t_am["delta_pde"],
                "ctrl_unrelated_delta_pde": c_un["delta_pde"],
                "ctrl_random_delta_pde": c_rd["delta_pde"],
                "ctrl_probe_delta_pde": c_pr["delta_pde"],
                "target_ablate_delta_bc": t_ab["delta_bc"],
                "crossover": crossover,
                "causal_score": score,
                "control_was_noop": bool(
                    c_un.get("control_was_noop", False)
                    or c_rd.get("control_was_noop", False)),
            })
        print(f"    batch {b + 1}/{N_BATCHES} done")
    return rows


def _exact_binomial_two_sided(k: int, n: int, p: float = 0.5) -> float:
    from math import comb
    if n <= 0:
        return float("nan")
    pk = comb(n, k) * (p ** k) * ((1 - p) ** (n - k))
    total = 0.0
    for i in range(n + 1):
        pi = comb(n, i) * (p ** i) * ((1 - p) ** (n - i))
        if pi <= pk + 1e-15:
            total += pi
    return min(1.0, total)


def analyze_r2(rows: List[Dict]) -> Dict:
    """H16 analysis: E_T CIs, per-feature sign tests with MC correction,
    crossover counts, and the pre-written decision rule."""
    from interventions.causal import per_feature_causal_pvalues
    # MC-corrected per-feature sign tests (target > strongest control)
    mc = per_feature_causal_pvalues(rows)
    # crossover counts per feature
    cross_counts: Dict[int, int] = {}
    for r in rows:
        cross_counts.setdefault(r["feature_idx"], 0)
        cross_counts[r["feature_idx"]] += int(r["crossover"])
    # Crossover criterion (registered, ONE-SIDED): a causal feature must
    # show the amplify-vs-ablate direction reversal CONSISTENTLY — >=
    # CROSSOVER_BATCHES of N_BATCHES.  0/N crossovers is a FAILURE of the
    # criterion, not a "significant" pattern: the two-sided binomial
    # would spuriously certify never-crossing features (P(0) is tiny).
    crossover_survivors = [k for k, c in cross_counts.items()
                           if c >= CROSSOVER_BATCHES]

    survivors = mc["bonferroni"]["survivors"]

    # decision rule (pre-written)
    both = [k for k in survivors if k in crossover_survivors]
    verdict = "R2a" if both else "R2b"

    return {
        "n_rows": len(rows),
        "n_candidates": mc["n_features_tested"],
        "et_ci": _bootstrap_ci([r["causal_score"]["causal_strength"]
                                for r in rows]),
        "beats_all_rate": float(np.mean([
            r["causal_score"]["target_effect"] > max(
                abs(r["causal_score"]["control_unrelated_effect"]),
                abs(r["causal_score"]["control_random_effect"]),
                abs(r["causal_score"]["control_probe_effect"]))
            for r in rows])),
        "bonferroni_survivors": survivors,
        "n_bonferroni": len(survivors),
        "crossover_counts": {str(k): c for k, c in cross_counts.items()},
        "crossover_survivors": crossover_survivors,
        "n_crossover": len(crossover_survivors),
        "both_criteria": both,
        "verdict": verdict,
        "verdict_note": (
            ">=1 feature survives Bonferroni AND the crossover criterion: "
            "the boundary covers interpretability (rank -> reconstruction "
            "-> causality)" if verdict == "R2a" else
            "no feature satisfies both the effect-magnitude criterion and "
            "the direction-reversal crossover: superposition is necessary "
            "but not sufficient — the reconstruction advantage does not "
            "carry effect-magnitude specificity, and the PhysSAE evidence "
            "lives below the specificity bar (the registered R2b reading)"
        ),
        "mc": mc,
    }


def _bootstrap_ci(vals: List[float], n_boot: int = 2000) -> Dict:
    arr = np.array(vals, dtype=np.float64)
    rng = np.random.default_rng(0)
    means = [float(np.mean(rng.choice(arr, len(arr)))) for _ in range(n_boot)]
    return {"mean": float(arr.mean()),
            "ci_lower": float(np.percentile(means, 2.5)),
            "ci_upper": float(np.percentile(means, 97.5))}


# ---------------------------------------------------------------------------
# Driver
# ---------------------------------------------------------------------------

def run_r2(seeds: List[int] = None, smoke: bool = False) -> Dict:
    global N_BATCHES
    print("\n=======================================================")
    print("R2: Causal Battery on the Fourier PINN (the third arrow)")
    print("=======================================================")

    n_batches = 3 if smoke else N_BATCHES
    n_candidates = 4 if smoke else N_CANDIDATES

    # ---- Machinery gate FIRST ----
    print("\n--- Machinery gate: planted control through the PINN hook ---")
    gate = machinery_gate()
    print(f"  planted cosine: {gate['planted_cosine']:.3f} | "
          f"beats-controls batches: {gate['beats_controls_batches']}/"
          f"{gate['n_batches']} | pass={gate['pipeline_pass']}")
    if not gate["pipeline_pass"]:
        out = {"stage": "r2_fourier_causal", "gates_pass": False,
               "gate": gate, "status": "VOID: machinery gate failure"}
        out_dir = RUNS / "r2_fourier_causal"
        out_dir.mkdir(parents=True, exist_ok=True)
        (out_dir / "r2_report.json").write_text(
            json.dumps(out, indent=2, default=str))
        print("  GATE FAILED — R2 VOID")
        return out

    seeds = seeds or SEEDS
    _orig = N_BATCHES
    N_BATCHES = n_batches  # analyze() reads the module-level constant
    report: Dict = {"stage": "r2_fourier_causal", "gates_pass": True,
                    "gate": gate,
                    "protocol": {
                        "candidates": n_candidates, "batches": n_batches,
                        "crossover_batches_bar": CROSSOVER_BATCHES
                        if not smoke else 2,
                        "alpha_ablate": ALPHA_ABLATE,
                        "alpha_amplify": ALPHA_AMPLIFY,
                        "site": "layers.1 (the H18 measurement site)",
                        "sae": "TopK k=8 exp=4 (the H18 family)",
                    },
                    "arms": {}}
    try:
        for arm, run_pattern in [
                ("fourier", "fourier_w64_seed{}"),
                ("tanh_control", "depth3_w64_seed{}")]:
            arm_rows = []
            arm_pr = []
            for seed in seeds:
                run_name = run_pattern.format(seed)
                print(f"\n--- {arm} / {run_name} ---")
                model, pde, cfg = _load_h18_run(run_name)
                acts = _collect_activations(model, pde, cfg, seed=seed)
                # rank check (the H18 site, per-seed)
                from analysis.effective_rank import analyze_run_activations
                h18_dir = RUNS / "architecture_boundary" / run_name
                stats = analyze_run_activations(h18_dir, "layers.1")
                arm_pr.append(stats.get("participation_ratio"))
                sae = _train_sae(acts, seed)
                labels = _probe_labels(pde, model, acts, seed=seed)
                probe_dir = _supervised_probe_direction(sae, acts, labels)

                rows = r2_battery(model, pde, sae, acts, probe_dir, seed)
                arm_rows.extend(rows)
            analysis = analyze_r2(arm_rows)
            analysis["per_seed_pr_layers1"] = arm_pr
            report["arms"][arm] = analysis
            print(f"\n  {arm}: E_T {analysis['et_ci']['mean']:.4f} "
                  f"[{analysis['et_ci']['ci_lower']:.4f}, "
                  f"{analysis['et_ci']['ci_upper']:.4f}] | "
                  f"bonf {analysis['n_bonferroni']}/"
                  f"{analysis['n_candidates']} | "
                  f"crossover {analysis['n_crossover']} | "
                  f"verdict {analysis['verdict']}")
    finally:
        N_BATCHES = _orig

    out_dir = RUNS / "r2_fourier_causal"
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "r2_report.json").write_text(
        json.dumps(report, indent=2, default=str))
    print(f"\nR2 report: {out_dir / 'r2_report.json'}")
    return report


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--seeds", type=str, default=None)
    args = ap.parse_args()
    seeds = [int(s) for s in args.seeds.split(",")] if args.seeds else None
    run_r2(seeds=seeds, smoke=args.smoke)
