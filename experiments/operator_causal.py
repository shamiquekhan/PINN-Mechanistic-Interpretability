"""Stage 14: Operator causal battery — complete the regime boundary.

The v3 stage-11 result is reconstruction-level only (TopK SAE beats
k-matched PCA 4.7x on FNO block states).  This stage runs the SAME
3-control causal protocol (unrelated / random / probe; MC corrections) on
those block states, plus the causal-abstraction interchange, so the regime
boundary claim is causal on BOTH sides:

  H14a (boundary): SAE features are causally specific in the high-rank
       regime — at least one candidate survives Bonferroni/BH-FDR, E_T
       CI excludes zero, and the sign diagnostic is mixed (directional
       ablation effects, not generic reconstruction damage).
  H14b (broader null): SAE features fail causally here too — the negative
       result extends past low-rank PINNs to function-space operators.

Either outcome is informative and publishable; the decision rules were
preregistered in docs/preregistration.md BEFORE this stage was run.

Pipeline:
  1. Retrain the stage-11 FNO on the nonlinear Green's-function task
     (deterministic seeds; identical to stage 11) and train the TopK SAE
     on its last block states.
  2. MACHINERY GATE: planted-feature positive control through the real
     operator hook + battery.  Must PASS before any verdict is read.
  3. Candidates: top-8 SAE latents by total activation.  Probe control:
     ridge probe in latent space predicting the output's dominant Fourier
     magnitude (a physically meaningful supervised target).
  4. Battery on a fixed held-out function batch; per-feature exact sign
     tests across independent held-out batches (n_batches replicates the
     PINN batteries' run-level replication); Bonferroni + BH-FDR.
  5. Interchange: partial interchange of the SAE-aligned subspace of the
     block state between two input functions; movement of the target loss
     toward the donor's — vs a random orthonormal basis (the stage-9
     criterion, transplanted to the operator).
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, List

import numpy as np
import torch
import torch.nn.functional as F

from analysis.pca_interpretability import fit_pca
from experiments.run_pipeline import DEVICE, RUNS
from interventions.operator_battery import (
    OperatorSAEHook,
    measure_operator_intervention,
    operator_positive_control,
    probe_direction_for_target,
    run_operator_causal_battery,
    summarize_operator_battery,
)
from operators.fno import FNO1d, green_function_dataset
from sae.model import SparseAutoencoder

PROJECT_ROOT = Path(__file__).resolve().parent.parent


def _train_fno_and_sae(seed: int = 0, width: int = 64, modes: int = 12,
                       n_layers: int = 4, grid_points: int = 64,
                       n_train: int = 2048, steps: int = 4000,
                       sae_steps: int = 2000, topk: int = 8,
                       expansion: int = 4):
    """Deterministic retrain of the stage-11 operator + SAE (identical
    protocol) so stage 14 is self-contained."""
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    a_train, u_train = green_function_dataset(
        n_train, grid_points, seed=seed, device=DEVICE)
    model = FNO1d(in_channels=1, out_channels=1, width=width, modes=modes,
                  n_layers=n_layers).to(DEVICE)
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)
    sched = torch.optim.lr_scheduler.StepLR(opt, step_size=1000, gamma=0.5)
    for step in range(steps):
        idx = torch.randint(0, n_train, (64,), device=DEVICE)
        loss = F.mse_loss(model(a_train[idx]), u_train[idx])
        opt.zero_grad()
        loss.backward()
        opt.step()
        sched.step()
    model.eval()

    # Block states for SAE training: per-(sample, position) rows.
    with torch.no_grad():
        _, inter = model(a_train, return_intermediates=True)
        states = inter[-1].reshape(-1, width)
    sae = SparseAutoencoder(input_dim=width, latent_expansion=expansion,
                            activation_mode="topk", topk=topk,
                            decoder_normalize=True).to(DEVICE)
    opt_s = torch.optim.Adam(sae.parameters(), lr=1e-2)
    bank = states.to(DEVICE)
    rng = torch.Generator(device="cpu").manual_seed(seed + 777)
    for _ in range(sae_steps):
        idx = torch.randint(0, bank.shape[0], (256,), generator=rng).to(DEVICE)
        opt_s.zero_grad()
        loss, _ = sae.loss(bank[idx])
        loss.backward()
        opt_s.step()
        if sae.decoder_normalize:
            sae._normalise_decoder()
    sae.eval()
    return model, sae, a_train, u_train


def _dominant_mode_magnitude(u: torch.Tensor) -> np.ndarray:
    """Per-sample magnitude of the output's dominant Fourier mode (probe
    target: a physically meaningful spectral quantity)."""
    spec = torch.fft.rfft(u.squeeze(-1), dim=-1)
    mag = spec.abs()                     # (n, L/2+1)
    # Exclude DC (index 0) — dominant NON-trivial mode.
    if mag.shape[1] > 1:
        dom_idx = mag[:, 1:].argmax(dim=1) + 1
    else:
        dom_idx = torch.zeros(mag.shape[0], dtype=torch.long)
    return mag[torch.arange(mag.shape[0]), dom_idx].cpu().numpy()


def run_operator_causal_experiment(seed: int = 0, n_batches: int = 8,
                                   n_candidates: int = 8,
                                   n_eval_functions: int = 256) -> Dict:
    print("\n=======================================================")
    print("STAGE 14: Operator Causal Battery (completing the boundary)")
    print("=======================================================")

    # ---- 1. Deterministic retrain (stage-11 protocol) ----
    print("[1/5] Retraining FNO + SAE (stage-11 protocol, seed 0)...")
    model, sae, a_train, u_train = _train_fno_and_sae(seed=seed)
    width = sae.input_dim

    # Held-out evaluation functions (fixed batch across the battery).
    a_eval, u_eval = green_function_dataset(
        n_eval_functions, seed=seed + 999, device=DEVICE)
    with torch.no_grad():
        test_rel = float((
            torch.linalg.vector_norm(model(a_eval) - u_eval, dim=(1, 2))
            / torch.linalg.vector_norm(u_eval, dim=(1, 2)).clamp(min=1e-12)
        ).mean())
    print(f"  FNO test rel L2: {test_rel:.4f}")

    # ---- 2. Machinery gate ----
    print("[2/5] Planted-feature positive control (operator machinery gate)...")
    pc = operator_positive_control(seed=seed, device=DEVICE)
    print(f"  gate pass={pc['pipeline_pass']} "
          f"(decoder cosine {pc['decoder_cosine_to_planted']:.3f}; "
          f"targeted {pc['targeted_delta']:.4f} vs random "
          f"{pc['ctrl_random_delta']:.4f})")
    if not pc["pipeline_pass"]:
        print("  WARNING: machinery gate FAILED — stage-14 verdict is not "
              "interpretable; fix the pipeline before drawing conclusions.")

    # ---- 3. Candidates + probe direction ----
    print("[3/5] Selecting candidate features + training the probe...")
    with torch.no_grad():
        z, _ = sae(
            model(a_train, return_intermediates=True)[1][-1]
            .reshape(-1, width))
        activity = z.abs().sum(dim=0)
    candidates = [int(i) for i in
                  torch.argsort(activity, descending=True)[:n_candidates]]
    print(f"  candidates (top-{n_candidates} by activity): {candidates}")

    # Probe target: dominant Fourier-mode magnitude of the OUTPUT function,
    # broadcast to the block-state rows (b*L) via the sample axis.
    dom_mag = _dominant_mode_magnitude(u_train)         # (n,)
    with torch.no_grad():
        states_flat = model(a_train, return_intermediates=True)[1][-1]
    n_samples, L = a_train.shape[0], a_train.shape[1]
    target_rows = np.repeat(dom_mag, L)                 # (n*L,)
    probe_dir = probe_direction_for_target(sae, states_flat.reshape(-1, width),
                                           target_rows)
    print("  probe direction trained (dominant-mode magnitude target).")

    # ---- 4. Battery across independent held-out batches ----
    print(f"[4/5] Causal battery: {n_candidates} features x "
          f"{n_batches} held-out batches x 3 controls...")
    all_rows: List[Dict] = []
    for b in range(n_batches):
        a_b, u_b = green_function_dataset(
            n_eval_functions, seed=seed + 5000 + b, device=DEVICE)
        rows = run_operator_causal_battery(
            model, sae, a_b, u_b, candidates, probe_dir,
            layer_index=-1, alpha=0.0,
        )
        for r in rows:
            r["batch"] = b
        all_rows.extend(rows)
        strongest = max(abs(r["causal_score"]["causal_strength"])
                        for r in rows)
        print(f"  batch {b}: strongest |E_T| = {strongest:.4f}")

    summary = summarize_operator_battery(all_rows)

    # ---- 5. Interchange on the operator (stage-9 criterion) ----
    print("[5/5] Partial interchange (SAE subspace vs random basis)...")
    sae_basis = sae.W_d.weight.detach().T[candidates].to(DEVICE)   # rows
    sae_mean = sae.b_a.detach().flatten().to(DEVICE)
    g = torch.Generator().manual_seed(3)
    raw = torch.randn(width, len(candidates), generator=g)
    # Orthonormal k rows in width-space: qr of the (width, k) matrix gives
    # Q (width, k); its transpose is the desired (k, width) basis.
    # (qr of raw.T would return a (k, k) Q — wrong orientation.)
    q, _ = torch.linalg.qr(raw.to(DEVICE))
    rand_basis = q.T[:len(candidates)].to(DEVICE)
    data_mean = states_flat.reshape(-1, width).mean(dim=0).to(DEVICE)

    def _interchange_movement(basis, mean, n_pairs: int = 32, seed_b: int = 0):
        """Movement of target loss toward donor under partial interchange
        of the aligned subspace (region mean convention from stage 9)."""
        rng = np.random.default_rng(seed_b)
        moves = []
        for _ in range(n_pairs):
            i, j = rng.integers(0, n_eval_functions, 2)
            a_s, a_d = a_eval[i:i + 1], a_eval[j:j + 1]
            with torch.no_grad():
                _, inter = model(a_s, return_intermediates=True)
                s_state = inter[-1]                     # (1, L, width)
                _, inter_d = model(a_d, return_intermediates=True)
                d_mean = inter_d[-1].mean(dim=1, keepdim=True)  # (1,1,width)
            lp_s = _reg_only(model, a_s, u_eval[i:i + 1])
            lp_d = _reg_only(model, a_d, u_eval[j:j + 1])
            # Keep the (1, 1, k) shape so it broadcasts along the position
            # axis against the (1, L, k) source coefficients.
            cd = ((d_mean - mean) @ basis.T)[..., :len(candidates)]
            def hook(_m, _i, output):
                cs = (output - mean) @ basis.T          # (1, L, k)
                cs = torch.cat([cd.expand(-1, cs.shape[1], -1),
                                cs[..., len(candidates):]], dim=-1)
                return (cs @ basis + mean)
            h = model.acts[-1].register_forward_hook(hook)
            try:
                lp_i = _reg_only(model, a_s, u_eval[i:i + 1])
            finally:
                h.remove()
            denom = np.log(lp_d + 1e-12) - np.log(lp_s + 1e-12)
            numer = np.log(lp_i + 1e-12) - np.log(lp_s + 1e-12)
            if abs(denom) > 1e-9:
                moves.append(float(numer / denom))
        return float(np.mean(moves)) if moves else float("nan")

    def _reg_only(model, a, u):
        with torch.no_grad():
            return float(F.mse_loss(model(a), u).item())

    mov_sae = _interchange_movement(sae_basis, sae_mean, seed_b=0)
    mov_rand = _interchange_movement(rand_basis, data_mean, seed_b=0)
    print(f"  interchange movement: SAE {mov_sae:+.3f} vs random "
          f"{mov_rand:+.3f}")

    # ---- Verdict ----
    mc = summary["multiple_comparisons"]
    et_ci = summary["causal_strength_ci"]
    passes_h14a = (
        mc["bonferroni_n_survivors"] > 0
        and et_ci["ci_lower"] > 0
        and summary["sign_diagnostic"]["n_target_delta_negative"] > 0
    )
    verdict = {
        "machinery_gate_pass": bool(pc["pipeline_pass"]),
        "bonferroni_survivors": mc["bonferroni_n_survivors"],
        "bh_fdr_survivors": mc["bh_fdr_n_survivors"],
        "et_ci": et_ci,
        "sign_diagnostic": summary["sign_diagnostic"],
        "interchange": {
            "sae_movement": mov_sae,
            "random_movement": mov_rand,
            "sae_beats_random": bool(mov_sae > mov_rand),
        },
        "h14a_boundary_confirmed": bool(passes_h14a),
        "h14b_broader_null": bool(not passes_h14a),
        "verdict_text": (
            "H14a: SAE features are causally specific in the high-rank "
            "regime — the regime boundary is causal on both sides."
            if passes_h14a else
            "H14b: SAE features are NOT causally specific even in the "
            "high-rank regime — the null extends beyond low-rank PINNs "
            "(reconstruction advantage without causal validity)."
        ),
    }

    out = {
        "status": "complete",
        "protocol": ("3 controls (unrelated latent, median-of-5 random "
                     "directions, supervised ridge probe on dominant-mode "
                     "magnitude), per-feature exact sign tests across "
                     f"{n_batches} held-out function batches, Bonferroni + "
                     "BH-FDR across 8 candidates"),
        "fno_test_rel_l2": test_rel,
        "positive_control": pc,
        "candidates": candidates,
        "battery_summary": summary,
        "interchange": {
            "sae_movement": mov_sae,
            "random_movement": mov_rand,
        },
        "verdict": verdict,
        "evaluations": all_rows,
    }
    out_dir = RUNS / "operator_causal"
    out_dir.mkdir(parents=True, exist_ok=True)
    with open(out_dir / "operator_causal_report.json", "w") as f:
        json.dump(out, f, indent=2)
    torch.save(model.state_dict(), out_dir / "fno_stage14.pt")
    sae.save(out_dir / "sae_stage14.pt")

    print("\nStage 14 verdict:")
    for k, v in verdict.items():
        print(f"  {k}: {v}")
    print(f"\nReport: {out_dir / 'operator_causal_report.json'}")
    return out


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--batches", type=int, default=8)
    args = ap.parse_args()
    run_operator_causal_experiment(seed=args.seed, n_batches=args.batches)
