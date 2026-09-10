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

- [ ] **Stage-14 follow-up (superseded by R2):** the direction-reversal
  criterion is now the registered H16 machinery and R2 applies it to the
  Fourier PINN; the standalone operator-battery re-run is deferred.
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
- [x] **Stage 19 (H19) — Monitor label provenance audit (v3.5):**
  preregistered, executed 2026-09-10, recorded — RD2D stage-10 runs never
  entered monitor training (structural exclusion traced); 2 artifact-
  labeled pool runs excluded (57→55); collocation never tested by the
  operational labeler. The loss-trajectory floor (logistic on past-only
  loss channels) scores AUROC 0.786 [0.615, 0.898], closing 78.2% of
  the threshold→conventional gap: the single-threshold floor was a
  straw man; the conventional arm's marginal value (0.875 vs 0.786,
  overlapping CIs) is thin but real. See docs/preregistration.md §H19,
  RESULTS.md §5A.11.

## Planned (v4.1 — PhysSAE reconciliation campaign)

Registered in docs/preregistration.md §R1–R5 (committed 2026-09-11
BEFORE any run); scientific prelude + confound decomposition:
docs/physSAE_reconciliation.md. Concurrent work: PhysSAE
(arXiv:2609.07061). The v3.5 results do not numerically contradict it —
the campaign measures whether the apparent conflict is regime, layer,
dictionary, or causal-criterion (see the reconciliation's §4 confound
table and §5 convergence point: their §4.8 independently replicates our
rank-collapse-with-convergence regularity from the failure side).

- [ ] **R1 — PhysSAE head-to-head on frozen checkpoints (highest
  priority):** matched-architecture Burgers/Allen–Cahn retraining
  (5×128, Adam+L-BFGS, w_BC=w_IC=100, 3 seeds) + H18 penultimate
  re-extraction; dictionaries {ReLU+L1 D=512 (3 SAE seeds), TopK k=8,
  PCA, ICA, random matched} × evaluations {alignment + permutation
  null + ESF80 + negative controls (pre-registered sign convention:
  advantage = ESF80_random − ESF80_top), E_T matched-deletion battery,
  reconstruction vs k-matched PCA}. Gates: planted-feature control
  through every new hook (penultimate extraction, ReLU+L1 trainer,
  ESF80). Outcomes R1a–R1d all pre-written.
- [x] **R1 — PhysSAE head-to-head on frozen checkpoints (v4.1):**
  preregistered, executed 2026-09-11, recorded **R1b** — machinery gates
  3/3 PASS; alignment replicates but is generic (random 0.85–0.96);
  ESF80 spatial concentration replicates (SAE 2.0–2.5× vs PCA/ICA,
  their headline) but is equally generic (SAE 2.0–2.3× vs random —
  the between-basis control their battery does not run); E_T
  effect-magnitude specificity null for every basis (random at the
  0/8 floor on all 6 runs). The causal criteria dissociate; both
  programs' claims stand at different level-ladder rungs. See
  docs/preregistration.md §R1 outcome, RESULTS.md §5A.12,
  docs/physSAE_reconciliation.md.
- [ ] **R2 — causal battery on the Fourier PINN:** the missing third
  arrow (rank → reconstruction advantage → causal specificity); H16
  direction-reversal machinery on the H18 Fourier dictionaries.
  Pre-written both ways: specificity found, or "superposition is
  necessary but not sufficient" as the reconciliation with PhysSAE.
  Sharpened by R1: the Fourier regime is the one place specificity
  could still emerge (all low-rank substrates are null).
- [ ] **R3 — Fourier frequency sweep:** n_freq ∈ {2,4,8,16,32,64},
  3 seeds, PR/PR-W/PCA-95 + reconstruction (both SAE families) + R2
  battery where PR moves. Pre-written: smooth / threshold /
  non-monotone.
- [ ] **R4 — SAE-seed robustness:** 3 PINN × 3 SAE seeds on the core
  Fourier condition; Hungarian-matched cross-seed cosine vs
  regime-verdict stability.
- [ ] **R5 — Burgers within-PINN boundary:** tanh vs Fourier on one
  time-dependent family (localized-concept panel available).

## Deferred (v4.2+)

- [ ] **Wider function-space task suite:** Darcy-flow-style operators,
  DeepONet comparison, wider FNOs (W = 128–512) to map the ρ threshold
  more finely.
- [ ] **Power upgrade for causal batteries:** ~4× checkpoints (44 runs/
  feature) — AFTER R1/R2 resolve the methodological comparison
  (mismatch is currently the larger threat than power).
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
