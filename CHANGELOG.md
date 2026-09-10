# Changelog

All notable changes to the PINN Mechanistic Interpretability framework.
Format: keep-a-changelog style; research-status entries track the evidence
state separately from code changes.

# Changelog

All notable changes to the PINN Mechanistic Interpretability framework.
Format: keep-a-changelog style; research-status entries track the evidence
state separately from code changes.

## [4.1.3] — 2026-09-11 — "R3/R4/R5 Complete: The v4.1 Campaign Closes"

### R3 — frequency sweep (18 runs): saturating rank response
- PR rises 2.05 -> 4.04 (nf 2->8) then SATURATES at ~4.8 (rho 0.075)
  for n_freq >= 16: covariance rank caps at what the width-64 layer
  can express; more frequencies add no rank
- The SAE/PCA ratio keeps climbing PAST the plateau (TopK 6->21->41->37x;
  ReLU+L1 24->63->125->137x): the advantage is not a function of rho
  alone — geometry gates, embedding richness drives compression
- As-scored shape: "non-monotone" (the pre-written classifier demands
  strict non-decrease; nf64's dip vs nf32 is within seed noise); as-read:
  a saturating dose-response, recorded exactly
- Spearman rho-vs-ratio +0.87/+0.88 across 18 runs; tangent rank 1 at
  every frequency

### R4 — SAE-seed robustness: the dissociation measured
- Cross-SAE-seed Hungarian cosine 0.41 (both families): dictionaries
  genuinely non-unique — independently replicating PhysSAE 4.7's ~0.35
  on our substrates
- 18/18 seed-combinations stable SAE>PCA verdicts (TopK 29-47x,
  ReLU+L1 79-167x): the regime, not the dictionary, is the unit of
  claim; the H18/R3 advantage is not a one-dictionary artifact

### R5 — Burgers boundary (R5a): external validity
- The within-PINN boundary replicates on a time-dependent family — and
  more strongly than Poisson: Fourier (t,x) PR 14.0-15.8 (rho 0.25,
  above the FNO's 6.8), SAE beats PCA 31-38x every seed
- tanh twin stays low-rank (PR ~1.8, honest note: mild 2-4x advantage
  from (t,x) structure); tangent 1; rel L2 comparable — architectural

### Campaign summary (R1-R5, all preregistered, all gates passed)
The v4.1 reconciliation campaign closes with every registered question
answered: R1b (PhysSAE evidence generic), R2b (causality absent at
every rank), R3 (saturating geometry + rising compression), R4
(dictionary vs regime dissociation), R5a (boundary external validity).
The chain: rank -> compression advantage (measured, robust, replicates)
-> causal specificity (ABSENT everywhere tested). Superposition is
necessary but not sufficient.

Also: docs/final_claim_ladder_v4.md (binding claim wording + the
L1-L5 level ladder); OSF bundle refreshed to v4.1.

## [4.1.1] — 2026-09-11 — "R2 Run: Superposition Is Necessary but Not Sufficient (R2b)"

R2 — the causal battery on the Fourier PINN (the third arrow), run per
the §R2 preregistration through the standard PINN hook at the H18
measurement site, H16 protocol (top-16 x n=20 x matched-deletion +
probe controls + one-sided crossover at >=15/20). Gate PASS (the
stage-5 planted world through the same path: cosine 0.989, 20/20).

- Fourier arm (PR 4.0-5.6 — the 25-56x reconstruction regime):
  E_T -15.89 [-17.46, -14.27]; 4/45 Bonferroni survivors ALL on the
  negative side (target ablations move the residual LESS than
  controls — the representational signature); 0/45 crossover
  survivors; beats-all 22%.
- Tanh control arm (PR 1.39-1.48): the identical null at smaller
  scale (E_T -0.058; 0 crossovers).
- Verdict R2b as pre-written: the within-PINN boundary is measured
  on both sides with BOTH criteria — crossing into the high-rank
  regime transfers the reconstruction advantage but NOT causal
  specificity. The chain rank -> reconstruction -> causality: two
  arrows measured, the third consistently absent across every regime
  tested (stages 5/8, H16b, R2b).
- Analysis correction, made BEFORE any verdict was recorded: the
  crossover criterion applied ONE-SIDED (>=15/20) as registered; a
  first two-sided implementation spuriously certified never-crossing
  features (P(0/20) small). Raw counts unchanged: every feature
  crosses 0/20 in both arms.

Also: README evidence section updated to the v4.1 state (within-PINN
boundary thesis, R1b, honest engineering claims); test-count
receipts 155 -> 171; RESULTS.md 5A.13; checksums 23/23.

## [4.1.0] — 2026-09-11 — "R1 Head-to-Head Run: R1b (Criteria Dissociate) + v4.0 Paper Rebuild"

### R1 — PhysSAE head-to-head (preregistered, gates passed, run, recorded R1b)

Matched-architecture Burgers/Allen–Cahn PINNs (5×128 tanh, Adam+L-BFGS,
w_BC=w_IC=100, 3 seeds each), penultimate activations, five dictionaries
(their ReLU+L1 D=512 × 3 seeds, our TopK, PCA, ICA, random), both metric
families on the same frozen checkpoints:

- Alignment replicates and is GENERIC: every basis incl. random
  directions aligns with physical concepts (max |r| 0.77-0.99, random
  0.85-0.96, Z>16 all runs)
- Spatial concentration replicates and is GENERIC: SAE ablation
  footprints 2.0-2.5× more ESF80-concentrated than PCA/ICA (their
  headline pattern) — but equally more concentrated than RANDOM
  directions (2.0-2.3×); the between-basis random control (not in
  their battery) removes the SAE-specific reading
- Effect-magnitude specificity NULL for every basis: random at the
  0/8 chance floor on all 6 runs (battery calibrated); survivors
  scatter without basis-specificity
- Verdict R1b as pre-written: the causal criteria dissociate; both
  papers' claims stand at different level-ladder rungs; the
  rank-collapse regularity was discovered independently by both
  programs

### Paper rebuilt around the geometry thesis (v4.0)

- New title: "When Sparse Features Become Mechanistic: Representation
  Geometry and Causal Interpretability in PINNs"
- Abstract rewritten: the "when" thesis; H18 within-PINN boundary as the
  centerpiece; PhysSAE reconciliation; H17/H19 as bounded secondary
  findings
- New sections: architecture boundary (H18, centerpiece figure 10 —
  three-panel phase diagram), PhysSAE reconciliation with the R1
  outcome, controller battery (H17), monitor floor (H19)
- Limitations extended: Fourier scope, PR bar operational not
  universal, covariance-vs-tangent wording
- 16 pages, 0 errors, 0 undefined citations; figure 10 generated
  from committed artifacts

Also: PhysSAE cited (patil2026physSAE) + Tancik Fourier cite added;
RESULTS.md §5A.12; preregistration §R1 outcome recorded; checksums 22/22;
171 tests; grounding OK.

## [3.6.0] — 2026-09-11 — "PhysSAE Reconciliation: v4.1 Campaign Registered (No Runs)"

Concurrent work surfaced during release preparation: PhysSAE
(arXiv:2609.07061v1, submitted 2026-09-07) reports positive SAE-PINN
alignment and spatially concentrated causal footprints across six
time-dependent PDE families. Full literature reconciliation added
(docs/physSAE_reconciliation.md) with the confound decomposition
(layer / dictionary / causal criterion / PDE regime / SAE-evidence chain)
and the level ladder mapping their terminology onto ours.

Key reconciliation findings (no experiments run — analysis of both papers):
- No numerical contradiction exists: the studies ask different causal
  questions (spatial-concentration ESF80 vs effect-magnitude specificity
  E_T), on different layers (penultimate vs middle), with different
  dictionaries (ReLU+L1 D=512 L0≈150 vs TopK k=8), on partially
  different PDE regimes.
- CONVERGENCE: PhysSAE §4.8 independently replicates our
  rank-collapse-with-convergence regularity from the failure side
  (failed PINNs higher-rank/diffuse; well-trained lower-rank/structured).
- Recorded factually: three internal-consistency notes on PhysSAE v1
  (abstract "0.04-0.44" vs Table 4 -0.025..+0.125; §2.8/§4.4 sign-
  convention contradiction; mean top-5 promised but unreported). Our
  replication pre-registers its own choices.
- The linear structure of their penultimate intervention (Δu =
  α·z_k·(W_out d_k), exactly proportional to the atom's activation
  field) makes ESF80 an activation-geometry statistic; whether that
  constitutes direction-specific causality is exactly what R1 measures.

v4.1 campaign registered BEFORE any run (docs/preregistration.md §R1-R5):
- R1 PhysSAE head-to-head on frozen checkpoints (matched-architecture
  Burgers/Allen-Cahn + H18 penultimate re-extraction; 5 dictionaries ×
  both metric families; planted-feature gates through every new hook)
- R2 causal battery on the Fourier PINN (the missing third arrow)
- R3 Fourier frequency sweep (phase diagram, pre-written smooth/
  threshold/non-monotone)
- R4 SAE-seed robustness (3×3, dictionary non-uniqueness vs regime
  stability)
- R5 Burgers within-PINN boundary (external validity)

Also: release receipts recorded (171 tests, grounding OK, 21/21
checksums, environment pinned) at the v3.5 evidence state (7da7c92);
ROADMAP restructured (v4.1 campaign / v4.2+ deferred); stage-14
follow-up superseded by R2.

## [3.5.3] — 2026-09-10 — "H19 Monitor Label Provenance Audit (T3 Closed)"

Stage 19 (H19) executed per v3.5 preregistration. Both registered parts
answered.

- **(a) Label provenance:** the stage-10 RD2D runs (rel L2 1.92-1.94)
  never entered monitor training (subdirectory not scanned by the
  stage-6 pool) — the registered mislabeling concern is structurally
  moot for them. Two pool runs carried artifact labels and were
  excluded from re-training (reaction_diffusion_baseline: init
  transient fs=0, converges 0.0015; gradient_conflict_seed2026: fs=0,
  final 0.0006). Pool 57 -> 55. Collocation starvation (10/10) derives
  no failure step — the monitor never tests collocation failures.
- **(b) Loss-trajectory floor:** logistic on past-only loss channels
  (rel_l2 and gradients excluded), same split/CIs: AUROC **0.786
  [0.615, 0.898]** vs threshold floor 0.468 and conventional 0.875 —
  closes **78.2%** of the gap. The single-threshold floor was a straw
  man; most of the conventional arm's apparent value was the rule,
  not the features. No-SAE-advantage conclusion unchanged (P=0.51).

Acceptance met: label provenance stated explicitly (per-run trace in
the artifact), floor baseline is a trajectory monitor, not a threshold.

## [3.5.2] — 2026-09-10 — "H18 Architecture Boundary: Superposition Within PINNs (H18a)"

Stage 18 (H18) executed per v3.5 preregistration. Outcome: **H18a** —
the within-PINN demonstration the regime-boundary thesis needed.

- Fourier-feature PINN (width 64, n_freq=32, 3 seeds): activation PR
  4.00–5.63 (mean 4.81, PR/W 0.075) — above the width-scaling envelope
  (1.14–2.51), in the FNO's neighborhood (6.8).
- Wherever PR moved, SAE-vs-k-matched-PCA reconstruction ran:
  **SAE beats PCA 3/3 by 25–56×** (mean PCA/SAE 40.7) — the same TopK
  SAE that is null on tanh PINNs. The rank diagnostic predicted exactly
  which runs would flip.
- Depth sweep 2–6 (tanh, 3 seeds each): PR falls monotonically
  2.05 → 1.15 — depth makes representations MORE degenerate; the
  "your PINNs are too simple" attack inverts.
- Tangent rank = 1 everywhere including Fourier: Fourier features
  change covariance rank (energy across frequency channels), not
  manifold dimension — covariance PR is the regime discriminator,
  dissociating from tangent rank as the theory note distinguishes them.

Pre-run machinery fix (recorded before any verdict): `fourier_embed`
existed in the config schema but was dead in `experiments/train.py`
(never passed to MLP) — wired through at f647dca before the run.

## [3.5.1] — 2026-09-10 — "H17 Controller Failure Battery (T2 Closed)"

Stage 17 (H17) executed per v3.5 preregistration. Outcome: **H17a** —
the controller wins its home class, specialized methods win elsewhere.

- Machinery gate: PASS — F1 controller rescue reproduces v3.4 on the
  reference seed 7 (0.421 → 0.000162; all 3 seeds 5.7e-05–1.6e-04).
- F1 boundary starvation: **controller 0.000119** (NTK 0.0044,
  no-action 0.221, GradNorm 3.82 — GradNorm actively harms).
- F2 spectral suppression: **NTK-adaptive 0.00363** (controller 0.335
  with high per-seed variance 0.005/0.995/0.004; no-action 0.509).
- F3 RD-2D pilot: **NTK-adaptive 1.674** — but every arm including the
  oracle ends ~1.96 (best mid-run ≈ 0.70–1.04): the task is unlearned
  at 2500 steps in this regime; no method rescues it.
- The controller never loses catastrophically to no-action on any
  class (worst mean Δ +0.019). Positioning per the pre-written H17a
  language: robustness across unknown failures, not per-failure SOTA.

Machinery incident (fixed BEFORE any verdict was read, `fc77850`): the
v3.4.2 stage-17 driver routed NTK/GradNorm arms through a code path with
no NTK/GradNorm logic (silent no-action clones), hard-coded
failure_class to boundary_starvation, left the preregistered machinery
gate and the aggregation unimplemented, and hand-rolled a second
training loop (the H3 anti-pattern). All five defects fixed pre-run,
smoke-tested; the registered design was not modified.

Also in this release (v3.5 docs sync, `722ef3d`): the H15 direction
error corrected everywhere (best |ρ| 0.578 EXCEEDS random p95 0.494 but
fails Bonferroni p=0.56); the H16 outcome record de-contaminated from
copy-pasted H15 text; ROADMAP restructured (stages 14–17 completed,
H18/H19 planned); paper/main.tex float bug fixed; arXiv package
regenerated (13 pp, H16-complete).

## [3.4.2] — 2026-09-08 — "H16 Operator Causal Asymmetry High-N Resolution (T1 Closed)"

Stage 16 (H16) executed per v3.5 preregistration (committed at ae8dafd
BEFORE the run). Outcome: **H16b** — the preregistered conjunctive rule
fires the null.

- Machinery gate: PASS (planted feature through real operator hook).
- Condition (i) MET: 6/16 features survive Bonferroni + BH-FDR at 20/20
  batch consistency (p = 0.0039 exact binomial floor at n=20).
- Condition (ii) MET: E_T CI [+5.6e-6, +1.6e-5] excludes zero.
- Condition (iii) NOT MET: 0/16 crossover survivors; no feature
  exhibits a consistent amplify-vs-ablate crossover direction.
- Preregistered conjunctive rule fires **H16b** (T1 closed).

Design observation recorded (honest limitation): the binary conflict
label is chronically high-conflict (75–100% of steps), so the binary
label has almost no within-run variation to correlate against — the
discriminative statistic is target-vs-controls, which is what E_T and
the sign tests measure. A continuous-magnitude redesign (or cross-regime
sampling) is registered as future work (H18), NOT run post-hoc.

Recorded conclusion: **H16b — no feature-specific conflict↔activity
bridge at the preregistered bar.** The operator causal-asymmetry thread
T1 is closed; the regime boundary claim rests on the reconstruction
side (PR 6.8, SAE beats PCA 4.7x) alone. Full record:
`runs/operator_highn/operator_highn_report.json`; docs/preregistration.md §H16.

## [3.4.1] — 2026-09-08 — "Lock-In: Review-Response Package + v3.5 Preregistration" 

P0 execution of the forward guide. No research claims changed.

### Added

- `REVIEW_RESPONSE.md` (top-level, reviewer-facing): the 29-issue
  campaign summary + a mandatory **errata table** for every number that
  moved in v3.4 (6.1→0.111, 6/8→2/8, 0.0166→0.0002, 0.859→0.875, the
  E_T sign flip, the SOTA-reading flip) — each with the reason and the
  authoritative section link.
- `docs/external_review_response.md`: append-only policy header; M6
  decline rationale documented so the next reviewer sees it as
  deliberate.
- v3.5 preregistration (H16–H19) committed BEFORE any v3.5 run:
  H16 operator-asymmetry high-n resolution (n≥20 batches, top-16
  features, direction-reversal criterion, both outcomes pre-written);
  H17 controller failure battery (3 reproducible classes, fourier
  trigger removed as a precondition); H18 Fourier-feature-PINN +
  depth-sweep scope probe; H19 monitor label-provenance audit +
  trajectory-loss floor baseline.
- CI stale-marker guard extended: the v3.4 moved numbers may now only
  appear inside explicitly-marked historical/drift contexts.

### Fixed (post-grep triage)

- PCA specificity in RESULTS §5A.1 now matches the artifact (2.58
  [0.87, 5.30] — the earlier "straddles 1" phrasing was wrong; the
  generated table was right).
- Paper Figure-9 caption + operator-table caption: 6/8 → 2/8 (the
  figure itself had been regenerated correctly; the captions hadn't).
- ROADMAP/PROJECT_STATUS stale references re-grounded (0.859 marked
  historical; stage-14 completed entry rewritten to the v3.4 outcome).
- M2 completion: all 5 live `weights_only=False` sites migrated to
  `weights_only=True` (verified safe on committed checkpoints); build/
  artifacts gitignored.
- `trigger_fourier_features` REMOVED from the controller action space
  (H17 precondition): spectral suppression now falls through to the
  bounded lambda rebalance the battery measures. Tests green.
- reproduce-script expectation corrected to the v3.4 verdict class
  ("SAE E_T negative or null" — the old "CI spans zero" check encoded
  the pre-C3 convention; the clean-clone verification pass caught it).
- Full clean-clone reproduction: stages 2–15 re-executed, all verdicts
  reproduced exactly, 16/16 headline checks PASS, artifacts
  re-checksummed (manifest note records the replication).

## [3.4.0] — 2026-09-08 — "External-Review Fix Campaign"

Executed the 29-issue external code review end-to-end (GUIDE.md = review
snapshot; docs/external_review_response.md = full disposition record).
Every fixed bug carries a regression test; every affected stage was
RE-RUN and its drift recorded — no verdict was patched.

### Fixed (critical)

- C1: reproduce-script headline gate is now artifact-derived
  (scripts/generate_expected_headlines.py -> runs/expected_headlines.json;
  CI-derived ranges instead of hard-coded points — the stale 0.872 check
  is structurally impossible now).
- C2: Burgers/AllenCahn exact() now interpolate the FULL space-time
  spectral trajectory (padded cell map, per-PDE pad policy: periodic for
  Burgers' continuous IC, edge-clamp for AllenCahn's aperiodic x³).
  Stage-10 rel L2 was 6.1/6.0 distance-from-zero artifacts -> genuine
  0.111/0.550.
- C3: causal controls are matched-deletion (unrelated control samples
  only ACTIVE features -> 0/88 no-ops with the control_was_noop
  diagnostic; random control ablates instead of injecting), applied to
  the SAE, PCA, and operator batteries.
- C4: doc drift swept (ACTING state removed — action fires on CONFIRMED
  entry; 15 stages; 8 intervention modes; stale §5A.4 forward-link) +
  CI stale-marker guard in check_results_grounded.py.

### Fixed (high/medium) + re-run

- H1: stage-6 gradient monitor arm implemented for real (joins
  gradients.jsonl window cosines); monitor 0.859 -> 0.875.
- H2: ceremonial leakage audit (failure_step=10**9) replaced with the
  true feature-window invariant; 12/12 held-out runs pass; fail-from-init
  runs correctly exempt.
- H3: PINNTrainer seams (get_lambdas / resample_fn / post_step_fn);
  stage 7 now runs THROUGH the trainer — the old loop violated the
  config's resample_every: 0. Config-faithful controller: 0.421 -> 0.0002
  (oracle level); it now beats NTK-adaptive (0.0064) on this failure.
  Both protocol readings recorded.
- H4: trigger_resample wired through the trainer seam; fourier trigger
  documented as signal-only (input-dim-changing restart = future work).
- H5: explicit speed/reaction_rate config fields; loud ambiguity errors.
- M1: SAE post_step() normalisation entry point, windowed dead-mask,
  format_version=2 checkpoints, unit-norm assert-at-save.
- M2: weights_only=True in trainer/SAE loaders (3 legacy sites flagged
  as follow-up). M3: no silent cuda->cpu downgrade; dtype validated.
- M4: no step-0 checkpoint; logging instead of print. M5: steps=0 means
  zero; AllenCahn docstring cleaned; unused failure_label deprecated.
- M7: stiff-RD overflow guarded (float32 exp overflows near 88, not
  350; final eval in float64; stiff configs -> None fallback).
- M6: DECLINED by repository owner — CC BY 4.0 retained.

### Research drift (recorded, verdicts unchanged in kind)

- SAE E_T +0.0020 -> -0.0173 (negative side = representational signature
  against honest controls); PCA -0.0203 -> -0.0131; paired diff now spans
  zero. Nulls HOLD: 0/8 survivors everywhere on PINNs.
- Stage-14 survivors 6/8 -> 2/8 under matched deletion — the suggestive
  operator asymmetry weakens (2/8 vs PINN 0/8) and is re-reported at
  this strength everywhere.
- Monitor 0.859/0.864 -> 0.875/0.877; P(SAE better) 0.53 -> 0.51.
- Controller rescue 0.295->0.0166 -> 0.421->0.0002 (config-faithful);
  controller now beats all three SOTA baselines on this failure; both
  readings recorded with their protocols.
- 171 tests (16 new regression tests for the review fixes).

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
  p = 0.070) marginally exceeds the random-direction p95 = 0.494 but
  fails the Bonferroni bar (p = 0.56).
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
