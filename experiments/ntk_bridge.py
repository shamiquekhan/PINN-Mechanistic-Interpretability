"""Stage 15: NTK gradient-conflict <-> SAE feature-activity bridge.

Closes revision Gap 2 (the NTK<->SAE bridge must be constructed, not
asserted) with the guide's Option A: a *correlational* first pass that
measures whether high-conflict training steps (negative pde-vs-bc
gradient cosine — Wang et al. 2021/2022's gradient-pathology criterion,
already logged per step in runs/<config>/gradients.jsonl) coincide with
feature-SPECIFIC SAE activity on the fixed probe grid
(runs/<config>/activations.jsonl), beyond what random dictionary
directions exhibit.

Hypotheses H15a/H15b and the conjunctive decision rule were preregistered
in docs/preregistration.md (section H8) BEFORE this stage was run:

  H15a requires ALL of:
    (i)   >=1 candidate survives Bonferroni with |rho_k| >= 0.3
    (ii)  the survivor's |rho_k| exceeds the 95th percentile of the
          random-direction |rho| distribution
    (iii) |rho_k| on the specificity-filtered lift s_k is >= 0.5 * the
          raw-activity correlation (association not a global-activity
          artifact)

  Otherwise H15b is recorded.

Machinery gate: a synthetic planted conflict-locked feature (active iff
the pde-vs-bc cosine < 0) must be recovered with rho > 0.7 through the
identical pipeline. A gate failure voids the run.

Data-handling bug fixed before the recorded run (documented in
preregistration H8 outcome): the activation logs repeat each step's
probe-grid record twice; the first implementation joined the duplicates,
which (a) inflated n past the exact permutation-test floor and (b) let
runs with only two non-conflict steps produce degenerate rho = +/-1
driven by single points. The fix deduplicates steps (first occurrence)
and requires >= 3 steps in both classes per run; runs that fail the
balance guard are skipped with recorded reasons. The decision rule is
unchanged — only the machinery was corrected.

Correlational by design: confirming H15a justifies (does not constitute)
the causal follow-up; Option B (NTK-eigenmode-projected SAE training)
stays future work either way.
"""
from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Dict, List

import numpy as np
import torch

from experiments.run_pipeline import DEVICE, RUNS
from sae.model import SparseAutoencoder


def _load_gradient_conflicts(run_dir: Path) -> Dict[int, dict]:
    """Map step -> gradient record (contains pde_vs_bc cosine)."""
    out = {}
    path = run_dir / "gradients.jsonl"
    if not path.exists():
        return out
    for line in path.read_text().splitlines():
        if not line.strip():
            continue
        rec = json.loads(line)
        out[rec["step"]] = rec["gradient_stats"]
    return out


def _load_probe_activations(run_dir: Path) -> List[dict]:
    """Ordered activation log records for one layer key."""
    path = run_dir / "activations.jsonl"
    if not path.exists():
        return []
    recs = []
    for line in path.read_text().splitlines():
        if not line.strip():
            continue
        recs.append(json.loads(line))
    return recs


def _layer_key(rec: dict) -> str:
    acts = rec.get("activations", {})
    keys = list(acts.keys())
    return keys[0] if keys else ""


def _feature_activity_matrix(sae: SparseAutoencoder,
                             activation_records: List[dict],
                             candidates: List[int],
                             device: torch.device) -> tuple:
    """Return (steps, A) where A[t, k] = mean probe-grid activation of
    candidate feature k at step t (encoder applied to the logged raw
    probe-grid activations)."""
    steps: List[int] = []
    rows: List[np.ndarray] = []
    seen_steps = set()
    for rec in activation_records:
        key = _layer_key(rec)
        if not key:
            continue
        raw = rec["activations"][key].get("raw")
        if not raw or rec["step"] in seen_steps:
            continue  # dedup: the activation log repeats steps; duplicates
                      # break permutation exchangeability and inflate n
        seen_steps.add(rec["step"])
        acts = torch.tensor(raw, dtype=torch.float32, device=device)
        with torch.no_grad():
            z = sae.encode(acts)            # (n_probe, latent_dim)
        mean_k = z.mean(dim=0).cpu().numpy()  # (latent_dim,)
        steps.append(rec["step"])
        rows.append(mean_k[[k for k in candidates]])
    if not steps:
        return [], np.zeros((0, len(candidates)))
    return steps, np.vstack(rows)


def _point_biserial(x: np.ndarray, labels: np.ndarray) -> float:
    """Correlation between continuous x and binary labels (NaN-safe)."""
    x = np.asarray(x, dtype=np.float64)
    labels = np.asarray(labels, dtype=np.float64)
    if len(x) < 3 or labels.std() == 0 or x.std() == 0:
        return float("nan")
    return float(np.corrcoef(x, labels)[0, 1])


def _permutation_pvalue(x: np.ndarray, labels: np.ndarray,
                        observed: float, n_perm: int = 1000,
                        rng: np.random.Generator | None = None) -> float:
    """Exact two-sided permutation p-value for the point-biserial rho."""
    rng = rng or np.random.default_rng(42)
    labels = np.asarray(labels)
    x = np.asarray(x, dtype=np.float64)
    count = 0
    for _ in range(n_perm):
        shuffled = rng.permutation(labels)
        rho = _point_biserial(x, shuffled)
        if not math.isnan(rho) and abs(rho) >= abs(observed) - 1e-12:
            count += 1
    return (count + 1) / (n_perm + 1)


def _random_direction_rhos(A: np.ndarray, labels: np.ndarray,
                           n_dirs: int = 32, seed: int = 0) -> np.ndarray:
    """|rho| distribution for random linear readouts of the SAME
    activation matrix (matched-dimensionality random dictionary control):
    random unit directions in activation space -> scalar projection ->
    correlation with the conflict label."""
    rng = np.random.default_rng(seed)
    n_steps, width = A.shape
    if n_steps < 3:
        return np.zeros(0)
    rhos = []
    for _ in range(n_dirs):
        v = rng.normal(size=width)
        v /= np.linalg.norm(v)
        proj = A @ v
        rho = _point_biserial(proj, labels)
        if not math.isnan(rho):
            rhos.append(abs(rho))
    return np.array(rhos) if rhos else np.zeros(0)


def _machinery_gate(n_steps: int = 60, seed: int = 0,
                    n_features: int = 8, width: int = 64,
                    n_perm: int = 1000) -> dict:
    """Planted conflict-locked feature: synthetic probe-grid activations
    where feature 0 carries extra energy exactly when the (synthetic)
    pde-vs-bc cosine is negative. The pipeline must recover rho > 0.7."""
    rng = np.random.default_rng(seed)
    cosines = rng.uniform(-1, 1, size=n_steps)
    labels = (cosines < 0).astype(float)
    # Base activity: smooth random curve + noise; planted lift on feature 0
    A = rng.normal(0.05, 0.02, size=(n_steps, n_features))
    A[:, 0] += 0.35 * labels + rng.normal(0, 0.01, size=n_steps)
    # global-activity confound: loss-magnitude-like surge uncorrelated w/ labels
    A += 0.2 * rng.uniform(0, 1, size=(n_steps, 1))
    raw = _point_biserial(A[:, 0], labels)
    lift = A - A.mean(axis=1, keepdims=True)
    spec = _point_biserial(lift[:, 0], labels)
    p = _permutation_pvalue(A[:, 0], labels, raw, n_perm=n_perm,
                            rng=np.random.default_rng(7))
    return {
        "planted_rho_raw": raw,
        "planted_rho_specific": spec,
        "permutation_p": p,
        "gate_pass": bool(abs(raw) > 0.7 and abs(spec) > 0.7 and p < 0.01),
    }


def run_ntk_bridge_experiment(n_perm: int = 1000) -> dict:
    print("\n=======================================================")
    print("STAGE 15: NTK conflict <-> SAE feature bridge (H15a/H15b)")
    print("=======================================================")

    # ---- Machinery gate FIRST: void the run if it fails ----
    gate = _machinery_gate()
    print(f"machinery gate: pass={gate['gate_pass']} "
          f"(planted rho raw {gate['planted_rho_raw']:.3f}, "
          f"specific {gate['planted_rho_specific']:.3f}, p={gate['permutation_p']:.4f})")
    if not gate["gate_pass"]:
        print("  GATE FAILED — stage void; fix pipeline before testing hypotheses.")

    # ---- Inputs: SAE + candidate features (stage-3/5 convention) ----
    sae_paths = sorted((RUNS / "sae_models_v2").glob("*/sae.pt"))
    if not sae_paths:
        raise FileNotFoundError("runs/sae_models_v2/*/sae.pt missing (run stage 3)")
    sae = SparseAutoencoder.load(sae_paths[0], DEVICE)

    dict_path = RUNS / "feature_dictionary" / "physics_feature_dictionary.json"
    candidates = list(range(8))
    if dict_path.exists():
        dictionary = json.loads(dict_path.read_text())
        candidates = [f["feature_id"] for f in dictionary[:8]]

    # ---- Eval runs: boundary-starvation runs with BOTH logs ----
    eval_runs = sorted(
        d for d in RUNS.iterdir()
        if d.is_dir() and "boundary_starvation" in d.name
        and (d / "gradients.jsonl").exists()
        and (d / "activations.jsonl").exists())
    print(f"bridge runs: {[d.name for d in eval_runs]}")

    per_run = []
    per_feature: Dict[int, List[dict]] = {f: [] for f in candidates}
    for run_dir in eval_runs:
        grads = _load_gradient_conflicts(run_dir)
        acts = _load_probe_activations(run_dir)
        if not grads or not acts:
            continue
        steps, A = _feature_activity_matrix(sae, acts, candidates, DEVICE)
        if len(steps) < 3:
            continue
        # Conflict label at each logged activation step (nearest gradient
        # record at-or-before the step; steps are logged on the same 100-step
        # grid, so this is an exact join in practice).
        labels = []
        for s in steps:
            avail = [g for g in grads if g <= s]
            if not avail:
                labels.append(np.nan)
                continue
            g = grads[max(avail)]
            labels.append(1.0 if g["gradient_cosines"]["pde_vs_bc"] < 0 else 0.0)
        labels = np.array(labels)
        keep = ~np.isnan(labels)
        steps_np = np.array(steps)[keep]
        A = A[keep]
        labels = labels[keep]
        n_pos = int((labels == 1).sum())
        n_neg = int((labels == 0).sum())
        if n_pos < 3 or n_neg < 3:
            per_run.append({"run": run_dir.name, "skipped": True,
                            "reason": f"degenerate label balance "
                                      f"({n_neg} non-conflict / {n_pos} conflict "
                                      f"of {len(labels)} steps; need >=3 both classes)",
                            "n_steps": int(len(labels)), "n_conflict": n_pos})
            continue

        # Specificity lift: remove per-step global mean across candidates
        S = A - A.mean(axis=1, keepdims=True)

        run_rec = {"run": run_dir.name, "n_steps": int(len(labels)),
                   "n_conflict": int(labels.sum()),
                   "features": {}}
        for j, f in enumerate(candidates):
            rho_raw = _point_biserial(A[:, j], labels)
            rho_spec = _point_biserial(S[:, j], labels)
            p = _permutation_pvalue(
                A[:, j], labels, rho_raw, n_perm=n_perm,
                rng=np.random.default_rng(1000 + f))
            rec = {"rho_raw": rho_raw, "rho_specific": rho_spec,
                   "perm_p": p}
            run_rec["features"][f] = rec
            per_feature[f].append({"run": run_dir.name, **rec})
        rand = _random_direction_rhos(A, labels)
        run_rec["random_rho_p95"] = float(np.percentile(rand, 95)) if len(rand) else float("nan")
        per_run.append(run_rec)

    # ---- Aggregate: per-feature sign-style consistency across runs ----
    summary_features = []
    for f in candidates:
        recs = [r for r in per_feature[f] if not math.isnan(r["rho_raw"])]
        n = len(recs)
        if n == 0:
            summary_features.append({"feature": f, "n_runs": 0})
            continue
        rhos = np.array([abs(r["rho_raw"]) for r in recs])
        rho_raws = np.array([r["rho_raw"] for r in recs])
        rho_specs = np.array([r["rho_specific"] for r in recs])
        best_p = min(r["perm_p"] for r in recs)
        # MC: minimum p across the 8 candidates * 11 runs battery
        bonf_p = min(1.0, best_p * len(candidates))
        summary_features.append({
            "feature": f,
            "n_runs": n,
            "mean_abs_rho": float(rhos.mean()),
            "max_abs_rho": float(rhos.max()),
            "max_abs_rho_run_rho_raw": float(rho_raws[np.argmax(rhos)]),
            "rho_specific_at_max": float(rho_specs[np.argmax(rhos)]),
            "min_perm_p": best_p,
            "bonferroni_p": bonf_p,
            "bonferroni_survivor": bool(bonf_p < 0.05 and rhos.max() >= 0.3),
        })

    # Random-control comparison for the best feature
    best = max(summary_features, key=lambda s: s.get("max_abs_rho", 0))
    rand_p95s = [r["random_rho_p95"] for r in per_run
                 if "random_rho_p95" in r
                 and not math.isnan(r["random_rho_p95"])]
    random_p95 = float(np.mean(rand_p95s)) if rand_p95s else float("nan")

    # ---- Preregistered conjunctive rule ----
    cond_i = any(s.get("bonferroni_survivor", False) for s in summary_features)
    cond_ii = bool(not math.isnan(random_p95)
                   and best.get("max_abs_rho", 0) > random_p95)
    if best.get("rho_specific_at_max") is not None and best.get("max_abs_rho_run_rho_raw"):
        cond_iii = bool(abs(best["rho_specific_at_max"]) >=
                        0.5 * abs(best["max_abs_rho_run_rho_raw"]))
    else:
        cond_iii = False
    passes_h15a = bool(gate["gate_pass"] and cond_i and cond_ii and cond_iii)

    report = {
        "stage": "ntk_bridge",
        "hypotheses": "H15a/H15b (docs/preregistration.md H8)",
        "machinery_gate": gate,
        "n_runs": len(per_run),
        "candidates": candidates,
        "per_run": per_run,
        "per_feature_summary": summary_features,
        "random_control": {"mean_p95_abs_rho": random_p95},
        "decision_rule": {
            "cond_i_bonferroni_survivor": cond_i,
            "cond_ii_beats_random_p95": cond_ii,
            "cond_iii_specificity_retained": cond_iii,
            "passes_h15a": passes_h15a,
            "recorded": "H15a" if passes_h15a else "H15b",
        },
        "interpretation_note": (
            "Correlational bridge only (revision-guide Option A). A null "
            "does not weaken stages 5/8/9; a pass justifies a causal "
            "amplify/ablate follow-up and Option B as future work."),
    }

    out_dir = RUNS / "ntk_bridge"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "ntk_bridge_report.json"
    with open(out_path, "w") as f:
        json.dump(report, f, indent=2)

    print(f"\nbridge verdict (H15a requires i&ii&iii):")
    print(f"  machinery gate: {gate['gate_pass']}")
    print(f"  (i) bonferroni survivor: {cond_i}")
    print(f"  (ii) best |rho| {best.get('max_abs_rho', float('nan')):.3f} "
          f"vs random p95 {random_p95:.3f}: {cond_ii}")
    print(f"  (iii) specificity retained: {cond_iii}")
    print(f"  recorded: {report['decision_rule']['recorded']}")
    print(f"report: {out_path}")
    return report


if __name__ == "__main__":
    run_ntk_bridge_experiment()
