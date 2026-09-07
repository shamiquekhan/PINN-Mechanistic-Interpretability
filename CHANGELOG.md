# Changelog

All notable changes to the PINN Mechanistic Interpretability framework.
Format: keep-a-changelog style; research-status entries track the evidence
state separately from code changes.

## [3.1.0] — 2026-09-07 — "Fresh Full-Campaign Record"

Complete pipeline rerun (stages 1–13 from scratch), recorded in
`docs/fresh_campaign_record.md`. 18/19 headline checks reproduce within
tolerance; the single drift (pooled mean PR 1.34 → 1.78) is an explained
correction (see below). Artifacts + checksums regenerated
(`data/manifest.json` v3.1.0).

### Fixed (found by the rerun)

- `experiments/train.py` no longer crashes on `--log-diagnostics` for
  non-Poisson PDEs (graceful degradation) — stage 1 had silently dropped
  the advection and reaction-diffusion baselines in previous full runs.
- `configs/reaction_diffusion_1d_baseline.yaml`: forcing 0 → 1 (the old
  configuration's exact solution was u≡0, making relative-L2 a 0/0
  artifact; the fresh run converges to rel_l2 ≈ 0.0015).

### Changed (research record)

- Pooled 1D-suite mean PR updated 1.34 → 1.78: the degenerate RD run's
  trivial activations previously biased the pooled mean downward. PR/W
  stays 0.028 (low-rank verdict unchanged); FNO boundary unchanged (6.8);
  every causal/regime claim unaffected (see the drift table in
  `docs/fresh_campaign_record.md`).
- Monitor AUROCs on the fresh split: conventional 0.859, SAE+conventional
  0.864 (published: 0.872/0.878 — split-level noise; no SAE advantage
  either way).
- `scripts/reproduce_main_results.sh`: geometry check updated to the
  corrected low-rank band (1.0 < PR < 2.5).

## [3.0.0] — 2026-09-07 — "Regime-Boundary Campaign"

### Added (code)

- **Stage 8 — PCA causal battery** (`interventions/pca_battery.py`,
  `experiments/pca_causal.py`): the SAE causal protocol (3 controls,
  per-feature exact sign tests, Bonferroni + BH-FDR) mirrored
  component-for-component on PCA bases, with a supervised PCA-space probe
  control and head-to-head paired comparison against the SAE battery.
- **Stage 9 — causal abstraction** (`interventions/interchange.py`,
  `experiments/causal_abstraction.py`): region-level partial interchange
  interventions implementing the Geiger et al. 2021/2022 interchange
  criterion; PCA / SAE / random alignment bases scored by movement
  fraction with a run-level beats-random-every-run gate.
- **Stage 10 — expanded PDE suite** (`pinn/pdes.py`):
  `AdvectionDiffusion2D`, `ReactionDiffusion2D` (manufactured solutions),
  `Burgers1D` (integrating-factor spectral reference solver), `AllenCahn1D`
  (enforced initial conditions); new configs; time-space probe grids.
- **Stage 11 — operator regime boundary** (`operators/fno.py`,
  `experiments/operator_boundary.py`): dependency-free FNO1d with named
  block states; nonlinear parametric Green's-function regression dataset;
  geometry + TopK-SAE vs k-matched-PCA comparison on function-space
  representations.
- **Stage 12 — SOTA baselines** (`experiments/sota_baselines.py`): GradNorm,
  NTK-adaptive weighting, and RBA residual-attention trainers compared
  against the controller on boundary starvation.
- **Stage 13 — statistical hardening**
  (`analysis/statistical_hardening.py`): exact sign-test power analysis
  (MDE), Bayesian posterior for monitor AUROC difference, failure-label
  threshold sensitivity.
- Pipeline runner extended to 13 stages; 38 new unit tests (139 total).

### Added (research packaging)

- `docs/preregistration.md` (v3 hypotheses H1–H6 with decision rules and
  recorded outcomes), `docs/theory_activation_rank.md`,
  `docs/paper_draft.md`, `docs/data_card.md`, `docs/experiment_matrix.md`,
  `docs/reproducibility.md`, `docs/reproducibility_checklist.md`,
  `docs/glossary.md`, `docs/failure_taxonomy.md`.
- `figures/` (8 publication figure scripts reading only `runs/` JSONs),
  `scripts/` (setup, smoke, full pipeline, reproduce, figures, tables),
  `paper/` (LaTeX skeleton + generated tables), `data/` (manifest +
  SHA256 checksums), `runs/README.md`.
- `Dockerfile`, `LICENSE` (CC BY 4.0), `CITATION.cff`,
  `CHANGELOG.md`, `ROADMAP.md`, `CONTRIBUTING.md`, `SECURITY.md`,
  `CODE_OF_CONDUCT.md`, `.github/workflows/tests.yml`, `.github/workflows/lint.yml`,
  `requirements.lock`.

### Changed

- `experiments/train.py`: relative-$L_2$ for time-dependent PDEs now
  computed on the exact solution's support (t=T slice) instead of the full
  zero-padded grid.
- README/DOCUMENTATION/RESULTS rewritten around the v3 claim ladder.

### Fixed

- PCA probe-direction control silently no-opped through the `natural`
  short-circuit (probe mode now a first-class hook mode).
- Burgers spectral reference blew up with explicit diffusion stepping
  (replaced with a stable integrating-factor Heun scheme).
- Time-dependent PINNs collapsed to u≡0 (initial conditions now enforced in
  the boundary/IC residual).
- Time-dependent `relative_l2` inflated by the zero-support denominator.

### Research status

- **Negative causal result hardened AND generalized:** SAE features AND PCA
  components fail MC-corrected causal testing (0/8 each); no candidate
  alignment basis satisfies the causal-abstraction interchange criterion —
  the null is basis-independent.
- **Regime boundary measured on both sides:** PINN activations PR/W
  0.02–0.10 (PCA dominates); FNO function-space representations PR/W 0.106
  where the same TopK SAE beats k-matched PCA 4.7×.
- **Controller repositioned honestly:** NTK-adaptive weighting (0.0064)
  beats the controller (0.0166) on boundary starvation; GradNorm/RBA
  actively harm. Controller claims limited to failure-agnostic machinery
  with measured monitor-source ablation.
- Theory corrected: covariance rank is NOT bounded by input dimension
  (moment-curve counterexample recorded); the provable statement is the
  local tangent-rank bound, which the data saturates exactly in all eight
  PDE families.

## [2.1.0] — 2026-09-06 — "Negative-Result Hardening"

### Added

- Multiple-comparison corrections (Bonferroni, BH-FDR) across the 8
  candidate features with per-feature exact paired sign tests.
- Planted-feature positive control
  (`experiments/positive_control.py`) — validates the SAE + hook +
  intervention + scoring machinery end-to-end.
- Effective-rank / participation-ratio analysis
  (`analysis/effective_rank.py`): PR ≈ 1.34 of 64 — the mechanistic
  explanation of the null.
- Monitor-source ablation in the controller demo (SAE-feature monitor vs
  matched-dimensionality random-feature monitor): bit-identical rescue
  trajectories — SAE features are inert cargo.
- Run-level bootstrap CIs on monitor AUROC; probe-direction control
  coverage fix (the v2.0 run had a fallback bug).
- Width-scaling study (W = 16–512) and 2D Poisson dimensional-boundary
  pilot (tangent-rank machinery in `analysis/activation_manifold.py`).

### Research status

- Hardened negative mechanistic result: E_T = −0.0002, CI spans zero;
  0/8 survive corrections; all 88 target deltas positive
  (representational signature); positive control passes.

## [2.0.0] — 2026-09-05 — "v2 Revision"

TopK SAEs with 3 mandatory causal controls (unrelated / random / probe),
controller rollback fix (real pre-action lambda snapshots), 10-seed
failure-atlas statistics, run-level splits for SAE training, multi-view
physics-feature dictionary with random-direction association control,
leakage-audited monitor suite.

## [1.0.0] — 2026-09-04 — "Initial Framework"

PINN core (MLP, trainer, PDE suite: Poisson/Advection/Reaction-Diffusion 1D),
activation/gradient/diagnostics logging, ReLU+L1 SAE, 7-mode intervention
engine, threshold + logistic monitors, closed-loop state-machine controller,
failure atlas, 92-test suite.
