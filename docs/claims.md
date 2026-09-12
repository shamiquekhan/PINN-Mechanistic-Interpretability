# Claims and Non-Claims (v4.3)

This document records the claims that are supported and unsupported by
the v4.3.0 evidence. Every positive claim is tied to a committed
artifact verified by `scripts/check_results_grounded.py`. For the full
evidence chains see `docs/final_claim_ladder_v4.md`; for the registered
hypotheses and decision rules see `docs/preregistration.md`.

---

## Established claims

### Representation geometry

| Claim | Evidence |
|---|---|
| Ordinary tanh PINN activations live at very low effective covariance rank (PR 1.14–2.51, mean 1.78) | `runs/effective_rank_analysis/effective_rank_report.json` |
| Fourier-feature PINN (n_freq=32, width 64) reaches PR 4.0–5.6, crossing into a higher-rank regime within the PINN family | `runs/architecture_boundary/architecture_boundary_report.json` |
| The same architecture boundary replicates on Burgers (PR 14–16) and reaction–diffusion (PR 4.3–6.0) | `runs/r5_burgers_boundary/r5_report.json`, `runs/r9_rd_boundary/r9_report.json` |
| FNO function-space representations reach PR 6.8 | `runs/operator_boundary/operator_boundary_report.json` |
| Local tangent rank exactly equals input dimension in all tested PDE families (provable bound, saturated) | `runs/dimensional_boundary_expanded/` |
| Depth lowers PR monotonically (2.05 → 1.15, depths 2–6); depth does not rescue superposition | `runs/architecture_boundary/architecture_boundary_report.json` |

### Compression advantage

| Claim | Evidence |
|---|---|
| In high-rank regimes the TopK SAE beats k-matched PCA by 25–56× on Fourier PINNs, 31–38× on Burgers, 28.5–32.8× on reaction–diffusion, and 4.7× on FNO | H18, R5a, R9a, stage 11 artifacts |
| In low-rank regimes SAE and PCA perform comparably (SAE loses ~50× at the tanh extreme) | stage 2, stage 5 artifacts |
| PR family identifies a validated gate but does not continuously predict compression magnitude (leave-one-arm-out CV R² all negative) | `runs/geometry_predictors/geometry_predictors_report.json` |

### Causal testing

| Claim | Evidence |
|---|---|
| No tested feature basis (SAE, PCA, ICA, or matched-random) satisfies the preregistered direction-specific causal criterion in the low-rank PINN regime | stages 5/8: 0/8 Bonferroni + BH-FDR on every basis |
| No feature satisfies the direction-reversal (amplify-vs-ablate) criterion in any high-rank regime (Fourier PINN: 0/45 crossover; FNO H16b: 0/16 crossover) | R2b, H16b artifacts |
| The dose-response curve is dose-symmetric (0 R6a in every high-rank regime) — no signed causal channel at any tested magnitude | `runs/r6_dose_response/r6_report.json` |
| The planted-feature positive control passes at 9.2× its matched control, confirming the pipeline detects real features | `runs/positive_control.json` |

### Statistical closure

| Claim | Evidence |
|---|---|
| ICC 0.192, DEFF 2.34, SE inflation 1.46× — the pseudoreplication question is answered with measurement | `runs/hierarchical_analysis/hierarchical_report.json` |
| Core nulls (paired SAE−PCA difference, SAE negative-side CI) survive checkpoint-clustered resampling; PCA negative-side CI does not survive — reworded per pre-written rule | R8b in preregistration |
| R7+b inconclusive at 31 combined checkpoints (mean −0.0049, 90% CI [−0.0174, +0.0062], power 0.38): formal equivalence not established, neither basis formally different from the other | `runs/r7plus_equivalence/r7plus_report.json` |

### Robustness

| Claim | Evidence |
|---|---|
| Cross-SAE-seed dictionaries are non-unique (Hungarian cosine 0.41) but 18/18 SAE>PCA verdicts are stable — the regime, not the dictionary, is the unit of claim | `runs/r4_sae_seed_robustness/r4_report.json` |
| SAE features are inert cargo in the closed-loop controller (SAE and random monitors produce bit-identical trajectories) | stage 7 monitor-source ablation |

---

## Negative findings (explicitly NOT claims of absence)

| What is NOT claimed | Why |
|---|---|
| SAE and PCA are formally equivalent at δ=0.01 | R7b inconclusive; CI overlaps the margin. Formal equivalence is not established. |
| SAE and PCA are clearly non-equivalent at δ=0.01 | CI also includes equality; the study does not claim either. |
| No causal features exist in PINNs ever | Only tested in this regime; the planted positive control passes; the machinery detects real features (decoder cosine 0.989, targeted ablation 9.2× random). |
| PR > 3 is a universal superposition threshold | Empirical diagnostic on this 1D Poisson / width-64 benchmark; not a theorem. |
| SAE reconstruction implies mechanism | Reconstruction alone does not imply direction-specific causality (R2b, R6b — both with matched controls). |
| Random bases are just arbitrary; SAE is special | R1 head-to-head shows alignment/concentration are generic over all bases; only causality is null everywhere — the dictionary is not privileged. |
| The controller improves optimization (per se) | The controller is failure-aware rescue machinery, not a universal optimization SOTA. |
| The monitor predicts failure from SAE features | SAE-augmented monitor is not significantly better than conventional (P = 0.51); conventional + loss-only trajectory achieves 0.875 AUROC; SAE features are inert cargo in the loop. |

---

## Scope and limitations

- All experiments run on 1D Poisson (10 seeds), Burgers (3 seeds),
  reaction–diffusion (3 seeds), and FNO Green's-function regression
  (seed-level). Generalization to wider PINNs, ensembles, higher-dimensional
  PDEs, or DeepONets is not tested and not claimed.
- The causal batteries at 11–20 runs/feature can detect ≥85.7%
  sign-agreement effects at 80% power. Subtler effects would require
  more checkpoints (stated as honest limitation, not hidden).
- Covariance participation ratio is an empirical diagnostic, not a
  theorem. The provable statement is the local tangent-rank bound.
- The full 163-checkpoint R7+ power extension is preregistered as
  optional future work. Its absence does not weaken the core claims
  above, which rest on the evidence in the artifacts listed.
