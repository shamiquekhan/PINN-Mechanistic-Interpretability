# Changelog

All notable changes to the PINN Mechanistic Interpretability framework.
Format: keep-a-changelog style; research-status entries track the evidence
state separately from code changes.

## [3.3.0] — 2026-09-08 — "Revision-Gap Audit + NTK Bridge (Stage 15)"

Driven by the four-gap revision review of the original proposal; gap
statuses audited against artifacts (docs/revision_gap_audit.md).

### Added

- `docs/revision_gap_audit.md`: gap-by-gap verdicts with receipts.
  Gap 1 (superposition precondition) CLOSED by the executed campaign;
  Gap 3 (spectral-bias baselines) closed in structure, citations
  actioned; Gap 4 (controller positioning) closed by stage 12 + the
  monitor-source ablation; Gap 2 OPEN → stage 15.
- `experiments/ntk_bridge.py` (stage 15, wired into run_pipeline):
  the guide's Option-A correlational bridge — feature-specific SAE
  activity vs Wang-et-al. gradient-conflict label (pde-vs-bc cosine),
  point-biserial + 1000-permutation exact tests, 32 random-direction
  control, Bonferroni + BH-FDR, planted-feature machinery gate,
  per-step deduplication and a >=3-steps-both-classes balance guard.
- `docs/preregistration.md` §H8: H15a/H15b + conjunctive decision rule,
  committed (747f82d) BEFORE the run.
- 9 unit tests (`tests/unit/test_ntk_bridge.py`; 155 total).
- Paper citations the review required: Xu et al. (F-Principle),
  Rahaman et al. (spectral bias), McClenny & Braga-Neto (self-adaptive
  PINNs), Wang et al. (causal training), Kim et al. (ROM autoencoder),
  + a flagged placeholder for the toy-model phase-transition analysis.
  Background section now benchmarks SAE diagnostics explicitly against
  the Fourier/NTK baseline literature.

### Research outcome (recorded as preregistered)

- Machinery gate: PASS. Preregistered conjunctive rule fires **H15b**:
  0/8 features survive Bonferroni; best |rho| = 0.578 (uncorrected
  p = 0.070) vs random-direction p95 = 0.494.
- Machinery incident, fixed BEFORE the verdict was read: the first
  implementation joined duplicated activation-log step records,
  inflating n past the exact permutation floor and producing degenerate
  rho = ±1 single-point artifacts (condition (i) initially "passed" at
  rho = 1.000). Post-fix numbers are the recorded ones; skipped runs
  and reasons are in the artifact.
- Design observation: 9/11 runs fail the label-balance guard —
  gradient conflict is CHRONIC in boundary starvation (75–100% of
  steps), so the binary label has near-zero within-run variation
  exactly where features were hypothesized causal. Continuous-magnitude
  redesign registered as future work, not run post-hoc.

## [3.2.1] — 2026-09-07 — "Submission Packaging"

No research-claim changes; every v3.2.0 verdict stands as recorded.

### Added

- `paper/main.tex`: full manuscript (complete prose, 9 figures, 7
  artifact-generated tables; compiled 12-page reference PDF `paper/main.pdf`).
  Claim strength matches the RESULTS.md claim ladder — the operator causal
  result is reported as suggestive (H14b fired), not confirmed.
- `paper/arxiv_package/`: self-contained arXiv submission bundle (tex, bib,
  figures, tables, reference PDF, submission checklist with bib-verification
  and claims-discipline notes).
- `docs/osf_upload/`: OSF/AsPredicted archival bundle — preregistration,
  experiment matrix, fresh-campaign record, plus `osf_manifest.json` with
  SHA256s and the git-SHA provenance chain binding each stage's
  preregistration to the commit that preceded its run.
- `PROJECT_STATUS.md`: arXiv/OSF moved from not-started to
  "package prepared, account-holder upload pending"; stale counts fixed
  (17/17 checksums, 9/9 figures, 7/7 tables).

### Changed

- `paper/README.md`: skeleton description → full-manifest description.
- `docs/fresh_campaign_record.md`: checksum count corrected to 17/17.


Preregistered (H14a/H14b committed at `f6eedf9` BEFORE the run), then
executed and recorded exactly as the decision rules fired.

### Added

- `interventions/operator_battery.py`: OperatorSAEHook (ablate/amplify/
  unrelated/random/probe on (batch, L, width) FNO block states),
  regression-loss target readout + Fourier-magnitude non-target readout,
  ridge probe on the output's dominant Fourier-mode magnitude, MC-corrected
  battery summary (reuses the PINN scoring machinery), and a
  planted-feature machinery gate with a one-sided relu readout.
- `experiments/operator_causal.py`: stage-14 driver (deterministic
  stage-11 FNO/SAE retrain, machinery gate, 8-batch replicated battery,
  partial-interchange vs random basis); wired as pipeline stage 14.
- `operators/fno.py`: block states routed through `self.acts[i]` so
  forward hooks fire (MLP-compatible convention).
- 7 unit tests (146 total). `PROJECT_STATUS.md` (verified/in-flight/
  not-started ledger with receipts). `scripts/check_results_grounded.py`
  + `.github/workflows/results_check.yml`: CI now FAILS if results docs
  cite numbers ungrounded in committed artifacts.

### Research outcome (recorded as preregistered, nuance documented)

- Machinery gate: PASS (targeted 0.0229 vs random control 0.0125).
- Preregistered conjunctive rule fires **H14b**: the mixed-sign condition
  (iii) failed (all 64 target deltas positive). Conditions (i) 6/8
  Bonferroni + BH-FDR survivors at 8/8 batch consistency, and (ii) E_T CI
  [+7.1e-6, +1.6e-5] excluding zero on the positive side, were MET.
- Comparative context: FNO features beat all 3 controls in 75% of
  evaluations (median target/control 1.2×) vs 14% (1.00×) for SAE-on-PINN
  and 32% (0.98×) for PCA-on-PINN.
- Recorded conclusion: by the MC-correction standard that established the
  PINN nulls, operator features ARE causally specific (6/8 vs 0/8
  everywhere on PINNs) — reported as a *suggestive causal asymmetry*
  across the regime boundary, NOT a confirmed causal boundary (weak
  effect sizes; non-diagnostic sign condition documented as a design
  lesson for future preregistrations). Full record:
  docs/preregistration.md §H7; RESULTS.md §5A.7.

### Changed

- README.md fully re-grounded: stale v3.0-era numbers replaced with the
  fresh v3.1 artifacts (AUROC 0.859/0.864; PR mean 1.78 range 1.14–2.51;
  146 tests), stage-14 status recorded honestly, qualification table
  re-read from `runs/<config>/metrics.jsonl`.

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
