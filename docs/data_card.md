# Data Card — PINN Mechanistic-Interpretability Benchmark (v3)

## 1. Dataset summary

Training trajectories of small PINNs across controlled baseline and
failure-inducing configurations, with full per-step instrumentation:
per-loss metrics, gradient statistics, pointwise diagnostics, probe-point
activation logs (raw vectors), and periodic checkpoints.

- **Modalities:** metrics.jsonl, gradients.jsonl, diagnostics.jsonl,
  activations.jsonl (raw, layer-resolved), checkpoint_*.pt, config.json,
  run_summary.json.
- **Volume (v3):** 45 logged 1D runs (5 regimes × 10 seeds minus omissions)
  + 7 legacy failure configs + width-scaling matrix (6 widths) + 2D pilots
  (Poisson×5, AD×3, RD×3) + time-dependent (Burgers×3, Allen-Cahn×3) + FNO
  operator run. ≈ 65,000 probe-point activation vectors at width 64 alone.
- **Generation:** `experiments/train.py` via `experiments/run_pipeline.py`
  stages 1–2, qualification seed matrix, and the boundary scripts.

## 2. Regimes and labels

| Regime | Configuration | Operational label |
|---|---|---|
| success_baseline | λ=(1,1), Xavier | success (rel_l2 ≤ 0.05) |
| boundary_starvation | λ_pde=100, λ_bc=0.01 | failure: BC starved |
| gradient_conflict | lr=0.1 | mixed labels — **excluded from failure-class claims** |
| spectral_suppression | W=16, source=50 | failure: high-freq error |
| collocation_starvation | N_i=4, bias 0.8 | mostly success — **excluded** |

Labels are operational (threshold-based), not ground-truth physics.
Threshold sensitivity (0.02–0.20) is reported in
`runs/statistical_hardening/analysis_report.json`; the `success` regime's
label fraction is threshold-sensitive, which is why it carries no
failure-class claims.

## 3. Intended use

- Studying whether sparse-autoencoder interpretability transfers to small
  scientific networks (it does not, in this benchmark — see RESULTS.md).
- Reproducing the causal-battery protocol (3 controls + MC corrections).
- Regime-boundary research: low-rank PINN vs high-rank FNO representations.
- Early-warning monitor and closed-loop controller research on
  configuration-determined failure mixtures.

## 4. Out-of-scope uses

- Claims about SAEs on *wide* PINNs, PINN ensembles, or production-scale
  operator learners — not measured here.
- Converged-solution benchmarks: runs intentionally include failure and
  stress configurations.
- Any deployment decision from monitor AUROCs: the failure mixture is
  configuration-determined, not natural-field data.

## 5. Known limitations and confounds

- `data_std` correlates r=0.999 with run identity — flagged as a
  run-type confound, never used as evidence.
- Held-out monitor runs in the original split lacked activation logs;
  the SAE monitor arm is reported NOT EVALUABLE rather than imputed.
- The 2D/time-dependent/FNO geometry runs are pilots (3–5 seeds), not
  powered studies.
- Gradient-conflict and collocation-starvation regimes do not reproduce
  clean labels across seeds and are excluded from class-level claims.

## 6. Versioning

- Schema: `docs/schema.md`; per-run config manifests are stored inside each
  run directory (config.json) and inside checkpoints.
- This card describes v3 (2026-09-07): adds PCA battery, causal-abstraction,
  dimensional-boundary expansion, operator boundary, SOTA baselines,
  statistical hardening artifacts.
