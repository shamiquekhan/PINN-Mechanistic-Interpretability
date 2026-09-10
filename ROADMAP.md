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
- [x] Leakage-audited monitors (conventional AUROC 0.875 post-review-fix, was 0.859; no SAE
  advantage), closed-loop controller with rollback + monitor-source
  ablation, SOTA baseline comparison (NTK-adaptive beats controller on its
  target failure; GradNorm/RBA harm).
- [x] Statistical hardening: sign-test MDE (85.7% @ 80% power), Bayesian
  monitor posterior (P=0.53), threshold sensitivity.
- [x] Research packaging: preregistration, theory note, data card,
  experiment matrix, reproducibility docs, figures/tables scripts, CI,
  Docker, license (CC BY 4.0), citation metadata.
- [x] **Operator causal battery (stage 14, v3.2):** preregistered (H14a/H14b
  at f6eedf9), executed, recorded. Preregistered conjunctive rule fires
  H14b (sign condition non-diagnostic), but by the MC-correction standard
  used for the PINN nulls the operator features pass (6/8 Bonferroni
  survivors pre-fix, 2/8 under matched-deletion controls — see
  docs/external_review_response.md; E_T CI positive both ways) — recorded as a SUGGESTIVE causal asymmetry,
  not a confirmed causal boundary. See docs/preregistration.md §H7 and
  RESULTS.md §5A.7.
- [x] **NTK conflict↔SAE bridge (stage 15, v3.3):** preregistered
  (H15a/H15b at 747f82d), executed, recorded H15b — 0/8 Bonferroni
  survivors, best |rho| 0.578 marginally exceeds the random-direction p95
  (0.494) but fails the preregistered Bonferroni bar (p = 0.56);
  a duplicated-step-records bug (degenerate rho=±1) was caught and fixed
  before the verdict was read. Revision Gap 2 closed with a measured null.
  See docs/revision_gap_audit.md, RESULTS.md §5A.8.
- [x] **Stage 16 (H16) — Operator causal asymmetry at high n (v3.5):** preregistered
  (H16a/H16b at ae8dafd), executed, recorded H16b --- 6/16 Bonferroni survivors
  at 20/20 batch consistency, 0/16 direction-reversal survivors; the operator
  causal asymmetry thread T1 is closed. The regime boundary claim rests on the
  reconstruction side (PR 6.8, SAE beats PCA 4.7x). See docs/preregistration.md §H16.

## Planned (next)

- [ ] **Stage-14 follow-up (preregister next):** replace the mixed-sign
  condition with a direction-reversal criterion (amplify-vs-ablate asymmetry)
  and re-run the operator battery; if the asymmetry confirms, upgrade the
  boundary claim from suggestive to confirmed.
- [ ] **Stage-15 redesign (registered in preregistration H8 outcome):**
  continuous conflict-magnitude label (or cross-regime run sampling) for the
  NTK↔SAE bridge — the binary label is structurally low-sensitivity in a
  chronically-conflicted regime; also Option B (SAE on
  NTK-eigenmode-projected activations).
- [x] **Stage 17 (H17) — Controller generalization failure battery (v3.5):**
  preregistered, executed 2026-09-10, recorded **H17a** — machinery gate
  PASS (F1 rescue reproduced, 0.421 → 0.000162); the controller wins its
  home class (0.00012, beating NTK-adaptive 0.0044) and loses to
  NTK-adaptive on spectral suppression (0.335 vs 0.0036) and the RD-2D
  pilot (1.98 vs 1.67); never catastrophic vs no-action (worst mean
  Δ +0.019); GradNorm harms F1 (3.82). Positioning: robustness, not
  per-failure SOTA. A v3.4.2 machinery incident (fake NTK/GradNorm arms,
  hard-coded failure_class, unimplemented gate, pending aggregation,
  hand-rolled loop) was fixed at fc77850 pre-verdict. Thread T2 closed.
  See docs/preregistration.md §H17, RESULTS.md §5A.9.
- [x] **Stage 18 (H18) — Fourier-feature PINN + depth sweep (v3.5):**
  preregistered, executed 2026-09-10, recorded **H18a** — the
  Fourier-feature PINN's activation PR rises to 4.0–5.6 (above the
  width-scaling envelope, FNO neighborhood) and there the same TopK
  SAE beats k-matched PCA **25–56×** (mean 40.7, 3/3 seeds): the
  superposition regime demonstrated *within PINNs*, rank diagnostic as
  predictor. Depth sweep: PR falls monotonically 2.05 → 1.15 — depth
  makes representations more degenerate, inverting the "too simple"
  attack. Tangent rank 1 everywhere (covariance/tangent dissociation).
  Pre-run fix: dead `fourier_embed` wiring repaired at f647dca.
  See docs/preregistration.md §H18, RESULTS.md §5A.10.
- [ ] **Stage 19 (H19) — Monitor label provenance audit:** RD2D label
  trace + loss-history trajectory monitor to replace the threshold
  floor baseline.
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
