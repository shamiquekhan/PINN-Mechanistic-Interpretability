# Roadmap

Research direction for the PINN Mechanistic Interpretability framework.
Statuses: **Completed** (evidence in `runs/`), **Planned** (scoped,
preregistered before execution when work starts), **Future** (direction,
not yet scoped). This file exists so nobody mistakes a planned experiment
for an established result.

## Completed (v1 → v3)

- [x] PINN core + 8-family PDE suite (1D Poisson/Advection/RD, 2D
  Poisson/AD/RD, viscous Burgers, Allen-Cahn) with manufactured solutions
  and stable reference solvers.
- [x] 10-seed failure atlas with operational labels; reproducible failure
  regimes (boundary starvation, spectral suppression).
- [x] TopK SAE discovery with mandatory PCA/random baselines; multi-view
  physics-feature dictionary with random-direction control.
- [x] 3-control causal battery + multiple-comparison corrections +
  planted-feature positive control → **hardened negative causal result**.
- [x] PCA causal battery → **null is basis-independent** (0/8; same
  representational signature).
- [x] Causal-abstraction interchange battery → **no alignment (PCA, SAE)
  beats random for region identity**.
- [x] Geometry: tangent rank = input dim in all families/widths; covariance
  PR 1.14–2.51 (mean 1.78); width scaling; dimensional + time-dependent boundary.
- [x] Operator regime boundary: FNO on nonlinear Green's-function
  regression → PR/W 0.106, SAE beats k-matched PCA 4.7× — the
  superposition regime measured with the same protocol.
- [x] Leakage-audited monitors (conventional AUROC 0.859; no SAE
  advantage), closed-loop controller with rollback + monitor-source
  ablation, SOTA baseline comparison (NTK-adaptive beats controller on its
  target failure; GradNorm/RBA harm).
- [x] Statistical hardening: sign-test MDE (85.7% @ 80% power), Bayesian
  monitor posterior (P=0.53), threshold sensitivity.
- [x] Research packaging: preregistration, theory note, data card,
  experiment matrix, reproducibility docs, figures/tables scripts, CI,
  Docker, license (CC BY 4.0), citation metadata.

## Planned (next)

- [ ] **Stage-14 follow-up (preregister next):** replace the mixed-sign
  condition with a direction-reversal criterion (amplify-vs-ablate asymmetry)
  and re-run the operator battery; if the asymmetry confirms, upgrade the
  boundary claim from suggestive to confirmed.

- [x] **Operator causal battery (stage 14, v3.2):** preregistered (H14a/H14b
  at f6eedf9), executed, recorded. Preregistered conjunctive rule fires
  H14b (sign condition non-diagnostic), but by the MC-correction standard
  used for the PINN nulls the operator features pass (6/8 Bonferroni
  survivors, E_T CI positive) — recorded as a SUGGESTIVE causal asymmetry,
  not a confirmed causal boundary. See docs/preregistration.md §H7 and
  RESULTS.md §5A.7.
- [ ] **Wider function-space task suite:** Darcy-flow-style operators,
  DeepONet comparison, wider FNOs (W = 128–512) to map the ρ threshold
  more finely.
- [ ] **Power upgrade for causal batteries:** ~4× checkpoints (44 runs/
  feature) to bring the detectable sign-agreement below 70%; then re-run
  stages 5/8/9 on the full matrix.
- [ ] **Wider-PINN ensembles** (the remaining scoped-out regime): ensemble
  mean/variance representations may raise effective rank even at narrow
  widths.
- [ ] **Preregistered OSF archiving** of the stage 8–13 protocols (the
  repo-internal preregistration precedes them; external archival is the
  next step for the submission).
- [ ] **Time-dependent geometry follow-up:** Burgers/Allen-Cahn PR was
  *below* steady 2D (≈1.5) at short schedules — run long-schedule variants
  to test whether the explored state family widens with training time.
- [ ] **Human-expert validation pilot** (Phase 15 of the research guide):
  blinded PCA-component vs SAE-latent identification task with 5–10
  domain experts.

## Future (directions, unscooped)

- Rank-aware sparse dictionary learning: SAE variants whose inductive bias
  adapts to measured PR (e.g., latent budget ≈ PR rather than expansion×W).
- Theory: predicting covariance PR from task/optimizer/width — the
  empirical regularity deserves an attempt at a law.
- Cross-domain transfer: does the participation-ratio precheck generalize
  to quantum-state networks, weather emulators, and protein models where
  SAEs are being tried?
- Community benchmark release: leaderboard for causal-interpretability
  methods on the frozen stage-1/2 failure mixture.
