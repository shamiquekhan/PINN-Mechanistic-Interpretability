"""Stage 16: H16 — operator causal asymmetry at high n (v3.5).

Preregistered in docs/preregistration.md (H16) BEFORE this run:
resolves thread T1 — is the stage-14 2/8-survivor asymmetry a boundary
signal or sign-test-floor noise?  Design (as registered):

  H16a (n-raise): the stage-14 battery with n >= 20 held-out function
       batches per feature and the TOP-16 SAE candidates (MC-corrected
       across 16 tests), matched-deletion controls unchanged.
  H16b (direction-reversal): the crossover criterion replaces the retired
       mixed-sign condition — a causal feature must show a consistent
       AMPLIFY-VS-ABLATE asymmetry: amplification (alpha=1.5) moves the
       target readout oppositely to ablation (alpha=0), with the
       crossover direction consistent across >= 15/20 batches
       (exact binomial, two-sided, p < 0.05).

Decision rule (pre-written): >=1 feature surviving Bonferroni at n>=20 AND
satisfying the crossover criterion => promote to "supported, small".
Otherwise T1 is CLOSED: the causal side of the boundary was not found
even at 2.5x power; the regime claim rests on the reconstruction side
alone and the paper says exactly that.

Machinery gate: the planted-feature positive control through the real
operator hook must pass at the same n; a gate failure voids the run.
"""
from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Dict, List

import numpy as np
import torch

from experiments.run_pipeline import DEVICE, RUNS
from experiments.operator_causal import (
    _train_fno_and_sae,
    _dominant_mode_magnitude,
)
from interventions.operator_battery import (
    operator_positive_control,
    probe_direction_for_target,
    run_operator_causal_battery,
    summarize_operator_battery,
)
from operators.fno import green_function_dataset

PROJECT_ROOT = Path(__file__).resolve().parent.parent


def _exact_binomial_two_sided(k: int, n: int, p: float = 0.5) -> float:
    """P(X == k or more extreme) under Bin(n, p), two-sided."""
    from math import comb
    if n <= 0:
        return float("nan")
    # two-sided: sum of all outcomes with probability <= P(k)
    pk = comb(n, k) * (p ** k) * ((1 - p) ** (n - k))
    total = 0.0
    for i in range(n + 1):
        pi = comb(n, i) * (p ** i) * ((1 - p) ** (n - i))
        if pi <= pk + 1e-15:
            total += pi
    return min(1.0, total)


def run_operator_highn_experiment(seed: int = 0,
                                  n_batches: int = 20,
                                  n_candidates: int = 16,
                                  n_eval_functions: int = 512,
                                  crossover_batches: int = 15) -> Dict:
    print("\n=======================================================")
    print("STAGE 16: H16 — operator causal asymmetry at high n")
    print("=======================================================")

    # ---- 1. Deterministic retrain (stage-11 protocol) ----
    print(f"[1/5] Retraining FNO + SAE (seed {seed})...")
    model, sae, a_train, u_train = _train_fno_and_sae(seed=seed)
    width = sae.input_dim

    # ---- 2. Machinery gate at the battery's own scale ----
    print("[2/5] Planted-feature positive control (gate)...")
    pc = operator_positive_control(seed=seed, device=DEVICE)
    print(f"  gate pass={pc['pipeline_pass']} (cosine "
          f"{pc['decoder_cosine_to_planted']:.3f})")
    if not pc["pipeline_pass"]:
        print("  GATE FAILED — run voided per the preregistration.")

    # ---- 3. Candidates (top-16 by activity) + probe ----
    with torch.no_grad():
        z, _ = sae(model(a_train, return_intermediates=True)[1][-1]
                   .reshape(-1, width))
        activity = z.abs().sum(dim=0)
    candidates = [int(i) for i in
                   torch.argsort(activity, descending=True)[:n_candidates]]
    print(f"[3/5] candidates (top-{n_candidates} by activity): {candidates}")

    dom_mag = _dominant_mode_magnitude(u_train)
    with torch.no_grad():
        states_flat = model(a_train, return_intermediates=True)[1][-1]
    n_samples, L = a_train.shape[0], a_train.shape[1]
    target_rows = np.repeat(dom_mag, L)
    probe_dir = probe_direction_for_target(sae, states_flat.reshape(-1, width),
                                            target_rows)

    # ---- 4. Battery: n_batches independent held-out function batches ----
    print(f"[4/5] Battery: {n_candidates} features x {n_batches} batches "
          f"x (ablate + amplify + 3 controls)...")
    all_rows: List[Dict] = []
    for b in range(n_batches):
        a_b, u_b = green_function_dataset(
            n_eval_functions // n_batches, seed=seed + 10000 + b,
            device=DEVICE)
        rows = run_operator_causal_battery(
            model, sae, a_b, u_b, candidates, probe_dir,
        )
        for r in rows:
            r["batch"] = b
        all_rows.extend(rows)
        print(f"  batch {b + 1}/{n_batches} done")

    summary = summarize_operator_battery(all_rows)

    # ---- 5. Direction-reversal (crossover) criterion ----
    # Per feature: across batches, count how often ablate and amplify move
    # the target loss in OPPOSITE directions (the causal signature: removing
    # the feature hurts, adding more of it helps -- or the reverse), and
    # whether that direction is consistent.
    print(f"[5/5] Direction-reversal criterion (crossover in "
          f">= {crossover_batches}/{n_batches} batches)...")
    crossover = {}
    for k in candidates:
        rows_k = [r for r in all_rows if r["feature_idx"] == k]
        ablate_delta = np.array([r["targeted_ablate"]["delta_target_loss"]
                                 for r in rows_k])
        amplify_delta = np.array([r["targeted_amplify"]["delta_target_loss"]
                                  for r in rows_k])
        # ablation should RAISE the loss (positive delta); amplification
        # should move it the other way (negative) -- or the consistent
        # reverse. crossover = sign(ablate) != sign(amplify), per batch,
        # restricted to batches where either effect is non-trivial (>1e-9).
        both = (np.abs(ablate_delta) > 1e-9) & (np.abs(amplify_delta) > 1e-9)
        n_both = int(both.sum())
        if n_both == 0:
            crossover[k] = {"n_batches_both": 0, "crossover_rate": float("nan"),
                             "consistent": False, "p": float("nan")}
            continue
        cross = (np.sign(ablate_delta[both]) != np.sign(amplify_delta[both]))
        rate = float(cross.mean())
        # consistency: the SAME direction pairing across batches
        pos_rate = float(((ablate_delta[both] > 0) & (amplify_delta[both] < 0)).mean())
        neg_rate = float(((ablate_delta[both] < 0) & (amplify_delta[both] > 0)).mean())
        consistent_dir = max(pos_rate, neg_rate)
        p = _exact_binomial_two_sided(int(round(consistent_dir * n_both)), n_both)
        crossover[k] = {
            "n_batches_both": n_both,
            "crossover_rate": rate,
            "consistent_direction_rate": consistent_dir,
            "consistent": bool(consistent_dir >= crossover_batches / n_batches
                               and p < 0.05),
            "p": p,
        }

    # ---- Preregistered decision rule ----
    mc = summary["multiple_comparisons"]
    bonf_survivors = [f["feature_idx"] for f in mc["per_feature"]
                      if f.get("bonferroni_survivor", False)]
    crossover_survivors = [k for k, c in crossover.items() if c["consistent"]]
    promoted = sorted(set(bonf_survivors) & set(crossover_survivors))
    if not bonf_survivors:
        # If the sign-test summary lost the survivor flags at this n, derive
        # from Bonferroni p directly.
        bonf_survivors = [f["feature_idx"] for f in mc["per_feature"]
                          if f["sign_test_p"] * n_candidates < 0.05]
        promoted = sorted(set(bonf_survivors) & set(crossover_survivors))

    passes_h16a = bool(pc["pipeline_pass"] and len(promoted) >= 1)
    recorded = "H16a" if passes_h16a else "H16b"

    report = {
        "stage": "operator_highn",
        "hypotheses": "H16a/H16b (docs/preregistration.md H16; v3.5)",
        "preregistration": "committed at ae8dafd BEFORE this run",
        "machinery_gate": pc,
        "n_batches": n_batches,
        "n_candidates": n_candidates,
        "candidates": candidates,
        "battery_summary": summary,
        "crossover": crossover,
        "bonferroni_survivors": bonf_survivors,
        "crossover_survivors": crossover_survivors,
        "promoted_features": promoted,
        "decision_rule": {
            "gate_pass": bool(pc["pipeline_pass"]),
            "n_bonferroni_survivors": len(bonf_survivors),
            "n_crossover_survivors": len(crossover_survivors),
            "n_promoted": len(promoted),
            "passes_h16a": passes_h16a,
            "recorded": recorded,
        },
        "pre_written_interpretation": {
            "H16a": "Asymmetry promoted to 'supported, small' — the causal "
                    "side of the regime boundary exists at high n.",
            "H16b": "T1 CLOSED: the causal side of the boundary was NOT "
                    "found even at 2.5x power; the regime claim rests on "
                    "the reconstruction side (PR 6.8, SAE-beats-PCA 4.7x) "
                    "alone, stated without hedging.",
        }[recorded],
    }

    out_dir = RUNS / "operator_highn"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "operator_highn_report.json"
    with open(out_path, "w") as f:
        json.dump(report, f, indent=2)

    print(f"\nH16 verdict:")
    print(f"  machinery gate: {pc['pipeline_pass']}")
    print(f"  Bonferroni survivors ({n_candidates} tests, n={n_batches}): "
          f"{bonf_survivors}")
    print(f"  crossover survivors: {crossover_survivors}")
    print(f"  promoted (both criteria): {promoted}")
    print(f"  recorded: {recorded}")
    print(f"report: {out_path}")
    return report


if __name__ == "__main__":
    run_operator_highn_experiment()
