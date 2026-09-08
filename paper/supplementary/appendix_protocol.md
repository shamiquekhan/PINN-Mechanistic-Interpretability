# Preregistration of Follow-Up Analyses (v3 campaign)

**Registered:** 2026-09-07 (before running stages 8–13 of the pipeline).
**Pre-existing evidence:** the v2.1 hardened negative SAE result (stages 1–7)
was completed and frozen before these analyses were designed; all stage 8–13
hypotheses were written before their experiments were run. Artifacts live in
`runs/` with timestamps.

---

## H1 — PCA causal battery (Stage 8)

**Hypothesis H8a (null).** The leading PCA components fail the 3-control
causal battery exactly as SAE latents did (E_T CI includes zero; 0/8 survive
Bonferroni or BH-FDR), because the low-rank manifold carries no
direction-specific causal signal in ANY basis.

**Hypothesis H8b (linear-alternative).** PCA components beat all three
controls with MC-corrected survivors — linear coordinates of the manifold
are the causal features; SAEs merely fragmented them.

**Design.** Same checkpoints (all boundary-starvation runs), same protocol
(3 controls, per-feature exact sign tests, Bonferroni + BH-FDR across 8
candidates), same scoring code (`compute_causal_score`,
`per_feature_causal_pvalues`) as the SAE battery. Probe control trained on
PCA coefficients with failure labels.

**Analysis plan.** Report E_T CI, specificity CI, sign diagnostic,
MC-corrected survivors, and the paired PCA−SAE difference with bootstrap CI.
Decision rule: the "linear methods are causal" claim (guide Phase 10) is
adopted ONLY under H8b; under H8a the negative result is *generalized*
beyond SAEs to all feature-basis causal interpretability in this benchmark.

**Outcome (recorded after running):** H8a **confirmed**. E_T = −0.0203
[−0.0331, −0.0085] (fresh-campaign values; v3.0 pool had −0.0186
[−0.0301, −0.0092] — same sign, same CI exclusion, verdict unchanged);
0/8 survive; all 88 target deltas positive (representational signature);
probe beats target in 45% of evaluations.

## H2 — Causal abstraction (Stage 9)

**Hypothesis H9a.** Neither PCA nor SAE alignments satisfy the interchange
criterion for the high-level model H_boundary (region identity → PDE loss):
their partial-interchange movement fractions are statistically
indistinguishable from a random orthonormal basis control.

**Hypothesis H9b.** One or both candidate alignments beat the random basis on
every run (run-level consistency gate).

**Design.** Region-level partial interchange interventions: source =
interior collocation batch, donor = boundary-proximal batch; swap the leading
aligned coefficients; measure movement fraction
`(log L_swap − log L_src)/(log L_don − log L_src)`. 16 trials × 11
checkpoints × 3 bases (PCA, SAE decoder top-8, random orthonormal).

**Decision rule.** A basis is a causal abstraction of H_boundary iff its
mean movement beats the random basis on every checkpoint run.

**Outcome:** H9a **confirmed**. PCA mean diff vs random +0.028 (7/11 runs
positive); SAE −0.087 (6/11). No candidate passes on every run; the failure
is basis-independent.

## H3 — Dimensional boundary (Stage 10)

**Hypotheses.** (i) All 2D-input PDEs show local tangent rank exactly 2.
(ii) Covariance PR stays in the 2–3.5 band (no superposition onset from
input dimension alone at these widths). (iii) Time-dependent inputs (t, x)
do not raise PR above the steady 2D band because training explores an
effectively low-parameter family of states.

**Outcome:** All three confirmed. Tangent rank 2 everywhere; PR: AD-2D 2.47,
RD-2D 2.29, Burgers 1.42, Allen-Cahn 1.92. Time-dependent PR is *below*
steady 2D.

## H4 — Operator regime boundary (Stage 11)

**Hypothesis H4a.** FNO block-state representations on a function-space
regression task have PR/W ≳ 0.05 (≥ 3× the PINN level), and the TopK SAE
*beats* k-matched PCA on reconstruction there — the superposition-regime
positive control.

**Hypothesis H4b.** FNO representations remain low-rank (PR/W < 0.05) and
the SAE still loses to PCA — the negative result extends to operator
learning.

**Design.** FNO1d (width 64, 4 spectral layers) on parametric Green's-function
regression (Gaussian-mixture forcing → nonlinear spectral response, 2048
training functions, 64-point grid). Representations = per-(sample, position)
block states (the analogue of token-level residual activations). Same TopK
SAE (expansion 4×, k=8) and k-matched PCA protocol as the PINN battery.

**Decision rule.** Regime boundary declared at the measured PR/W and the
sign of the SAE-vs-PCA reconstruction gap.

**Outcome:** H4a **confirmed**. PR = 6.8 of 64 (ρ = 0.106, ≈3.8× the PINN pooled mean of 1.78);
PCA needs 19 components for 95% energy (vs 2 for PINNs); TopK SAE
reconstruction 0.00070 vs k-matched PCA 0.00334 — **SAE wins 4.7×**.
The regime boundary is measured on both sides.

## H5 — SOTA baselines (Stage 12)

**Hypothesis.** NTK-adaptive weighting (Wang et al. 2021) matches or beats
the closed-loop controller on boundary starvation because instantaneous
reweighting is better suited to a static misweighting failure; GradNorm
(equalization) and RBA (residual attention) do not address the failure and
do not rescue it.

**Outcome:** Confirmed in structure. NTK-adaptive final rel L2 0.0064
(beats controller 0.0166 and oracle 0.0119); GradNorm 2.01 and RBA 2.19
(fail; worse than no-action 0.2953 — active harm). The controller is
positioned as *intermediate*: stronger than no-action, weaker than the
best specialized reweighting, with the monitor-source ablation intact.

## H6 — Statistical hardening (Stage 13)

**Analyses preregistered:** exact sign-test power (MDE) for both causal
batteries; posterior probability of SAE monitor advantage; failure-label
threshold sensitivity 0.02–0.20.

**Outcome:** MDE = 85.7% sign-agreement at 80% power (11 runs/feature) —
the batteries can only detect near-deterministic causal features, and none
approach the threshold. P(SAE monitor better) = 0.53 (no advantage). Label
ordering is stable for boundary/spectral regimes; the `success` regime's
label fraction is threshold-sensitive (0.3 → 0.0 over 0.02–0.20) — already
reflected in its "unlabeled" exclusion from failure-class claims.

---

## H7 — Operator causal battery (Stage 14; registered BEFORE running)

**Background.** Stage 11 (H4a) established the reconstruction-level regime
boundary: FNO block states have PR 6.8/64 (ρ = 0.106) and the same TopK
SAE that loses to k-matched PCA by 12× on PINNs *wins* by 4.7× there. The
remaining question — the difference between "SAEs are appropriate in the
high-rank regime" and "SAEs merely reconstruct better there" — is causal.

**Setup (fixed before running).**
* Operator: FNO1d (W=64, 4 spectral layers, 12 modes) on the stage-11
  nonlinear Green's-function regression task, retrained with the same
  seeds; TopK SAE (expansion 4×, k=8) trained on the last block states
  (per-(sample, position) rows).
* Target readout: function-space regression loss MSE(model(a), u) on
  fixed held-out function batches. Non-target readout: mean output
  Fourier-magnitude statistic (spectral shape).
* Candidates: top-8 SAE latents by total activation.
* Controls (identical semantics to stages 5/8): unrelated latent
  ablation; median-of-5 norm-matched random directions; supervised ridge
  probe in latent space predicting the output's dominant Fourier-mode
  magnitude.
* Replication: 8 independent held-out function batches; per-feature exact
  one-sided paired sign tests; Bonferroni + BH-FDR across the 8
  candidates (the operator analogue of the PINN batteries' run-level
  replication).
* Machinery gate: planted-feature positive control through the exact
  operator hook + battery (one-sided relu readout; must show targeted
  ablation beating all three controls) — if it fails, no verdict is read.
* Interchange: partial interchange of the SAE-aligned subspace between
  donor/source functions (stage-9 criterion), against a random
  orthonormal basis.

**Hypotheses and decision rules.**

- **H14a (regime boundary, causal).** SAE features are causally specific
  in the high-rank regime. *Decision rule:* H14a is confirmed iff
  (i) at least one candidate survives Bonferroni across the 8-batch sign
  tests, (ii) the E_T bootstrap CI excludes zero on the positive side,
  and (iii) the sign diagnostic is mixed (genuine direction-specific
  ablation effects rather than the all-positive representational
  signature). Interpretation: the regime boundary is causal on both
  sides — SAE causal validity tracks measured representation rank.

- **H14b (broader null).** SAE features are not causally specific even in
  the high-rank regime (0/8 survive; or CI spans zero; or all-positive
  sign diagnostic). Interpretation: reconstruction advantage does not
  imply causal validity — the null extends beyond low-rank PINNs, and
  "check superposition" is necessary but not sufficient for SAE causal
  claims.

Either outcome is reported as measured; no post-hoc switching of
thresholds, candidates, batches, or probe targets. If the machinery gate
fails, the stage is re-run only after a code fix that is itself committed
and documented (the gate failing means the *battery* is broken, not that
the hypothesis was tested).

**Outcome (recorded after running):** The preregistered conjunctive rule
fires **H14b** — condition (iii) failed (all 64 target deltas positive),
so H14a is not confirmed. Conditions (i) and (ii) were met, however, and
the nuance is recorded rather than hidden:

- **Machinery gate:** PASS (targeted 0.0229 vs random control 0.0125;
  decoder cosine 0.413 at gate scale).
- **Condition (i) MET:** 6/8 candidates survive Bonferroni (each at 8/8
  batch consistency, p = 0.0039); 6/8 survive BH-FDR. This is the exact
  criterion the PINN batteries failed (0/8 there).
- **Condition (ii) MET:** E_T bootstrap CI = [7.1e-6, +1.6e-5] — excludes
  zero on the positive side (PINN batteries: CI spans zero).
- **Condition (iii) NOT MET:** sign diagnostic 64/64 positive.
- **Comparative context:** target beats all three controls in 48/64
  evaluations (75%), median target/control ratio 1.2× — versus 12/88
  (14%) at ratio 1.00× for the SAE-on-PINN battery and 28/88 (32%) at
  0.98× for the PCA-on-PINN battery. The operator features show
  *substantially more* control-beating specificity than any PINN basis.
- **Weaknesses recorded:** absolute E_T magnitudes are tiny (~1e-5 on a
  near-zero baseline regression loss); the specificity ratio against the
  spectral non-target readout is 0.002 and is cross-unit (MSE vs Fourier
  magnitude), so it is not directly interpretable; and the partial
  interchange does NOT beat a random basis (SAE −1.257 vs random −1.226).

**Post-hoc design observation (flagged as such, not used to overturn the
preregistered verdict):** condition (iii) was motivated by the PINN
batteries, where all-positive deltas accompanied E_T ≈ 0 (generic decode
damage). For an ablation battery on a reconstruction readout, all-positive
target deltas are the *expected* signature whether or not the feature is
causal — information removal always increases the loss — so (iii) is not
diagnostic in this setting; the discriminative statistic is target vs
matched controls, which is what E_T and the sign tests measure. A future
preregistration should replace (iii) with a direction-reversal condition
(e.g., amplify-vs-ablate asymmetry) or drop it.

**Recorded conclusion:** By the preregistered rule: **H14b** — the
strictly-conjunctive boundary claim is not confirmed. By the same
statistical standard the PINN nulls used (MC-corrected survivors + E_T
CI), the operator features ARE causally specific (6/8 survive where PINNs
had 0/8) — a weaker, hedged form of the regime-boundary claim that we
report as a *suggestive asymmetry*: reconstruction advantage in the
high-rank regime is accompanied by weak-but-consistent causal specificity
that the low-rank regime entirely lacks. Full numbers:
`runs/operator_causal/operator_causal_report.json`; RESULTS.md §5A.7.

---



- Stages 8–9 outcomes (H8a, H9a) *generalize* the negative result; they do
  not weaken it and were reported exactly as measured, including the
  4/88→31/88 raw beats-all counts that fail MC correction.
- Stage 11's positive control (H4a) is the paper's regime boundary; its
  SAE-advantage claim was reconstruction-level only when first registered —
  stage 14 (H7) was registered and run to complete the causal side; see
  RESULTS.md §5A.7 for the outcome.
- All artifacts referenced above are the JSON files in `runs/` produced by
  the stage scripts; none were edited by hand.

## H8 — NTK gradient-conflict ↔ SAE feature-activity bridge (Stage 15)

*Registered 2026-09-08, before the stage-15 run (revision-gap audit,
Gap 2: the NTK↔SAE bridge must be constructed, not asserted).*

**Hypothesis H15a (bridge exists).** SAE feature activity on the fixed
probe grid is *feature-specifically* elevated during high-conflict
training steps — steps where the pde-vs-bc gradient cosine is negative,
Wang et al. (2021/2022)'s gradient-pathology criterion — beyond what a
random dictionary of the same geometry exhibits.

**Hypothesis H15b (no specific bridge).** Any apparent feature↔conflict
association is either (i) not feature-specific (driven by global activity
surges that correlate with loss magnitude) or (ii) matched by random
dictionary directions.

**Construction (Option A of the revision guide — the correlational
bridge, defined precisely).** For each logged training step t with both
a gradient record and probe-grid activations:

- conflict(t) = 1[cos(∇θℒ_pde, ∇θℒ_bc)(t) < 0]  (Wang et al. criterion;
  values already logged per step in `runs/<config>/gradients.jsonl`)
- For each SAE feature k: a_k(t) = mean probe-grid activation of feature
  k at step t (SAE encoder applied to the fixed 50-point grid logged in
  `activations.jsonl`; the probe hash is constant per run, so the grid
  is identical across steps)
- Specificity: s_k(t) = a_k(t) − mean_j a_j(t)  (feature-specific lift,
  removing the global activity component)

**Statistics.** Per feature: point-biserial correlation ρ_k between
s_k(t) and conflict(t); per-run significance via exact permutation test
(labels shuffled 1,000×, fixed rng); Bonferroni + BH-FDR across the 8
candidate features (same MC standard as stages 5/8/14); random-dictionary
control: median |ρ| over 32 random orthonormal directions of matching
dimensionality through the identical pipeline.

**Decision rule (conjunctive, all three required for H15a).**
(i) ≥1 candidate feature survives Bonferroni with |ρ_k| ≥ 0.3;
(ii) the surviving feature's |ρ_k| exceeds the 95th percentile of the
random-direction |ρ| distribution;
(iii) the association is not explained by the global-activity covariate:
|ρ_k| on s_k (specificity-filtered) must be ≥ 0.5 × |ρ_k| on a_k (raw).

If any of (i)–(iii) fails, H15b is recorded. Machinery gate: on
synthetic activations with a planted conflict-locked feature (active iff
cosine < 0), the pipeline must recover ρ > 0.7 for the planted feature;
gate failure voids the run and the stage is fixed and re-run before any
hypothesis is tested.

**Interpretation scope (pre-committed).** This is a *correlational*
first pass by design (guide Option A). Confirming H15a would justify —
not constitute — the causal follow-up (amplify/ablate at high-conflict
steps), and Option B (training the SAE on NTK-eigenmode-projected
activations) stays registered as future work either way. A null here
does NOT weaken any existing claim (stages 5/8/9 stand on their own);
it closes the last unconstructed bridge the revision review identified.

**Outcome (recorded after running).** The preregistered conjunctive rule
fires **H15b** — condition (i) failed: no candidate survives Bonferroni
(best max |rho| = 0.578, permutation p = 0.070 uncorrected; Bonferroni
p = 0.56 across 8 features); the best feature only marginally exceeds
the random-direction 95th percentile (0.578 vs 0.494), nowhere near a
consistent bridge.

**Machinery incident, recorded (fixed BEFORE the recorded run was read as
a verdict):** the first implementation joined the activation logs'
duplicated step records, which (a) inflated n past the exact permutation
floor and (b) let runs with only 1–2 non-conflict steps produce
degenerate rho = ±1 single-point artifacts — condition (i) initially
fired with rho = 1.000 "survivors". The fix (step deduplication +
a >=3-steps-in-both-classes balance guard, with skipped runs and reasons
recorded in the artifact) preceded reading any verdict. Post-fix numbers
above are the recorded ones.

**Design observation (honest limitation of this bridge test):** 9 of 11
boundary-starvation runs were skipped by the balance guard because the
pde-vs-bc cosine is negative in 75–100% of logged steps — gradient
conflict in this regime is *chronic*, so a binary conflict label has
almost no within-run variation to correlate against, precisely in the
regime where features were hypothesized causal. The bridge as specified
(Option A, binary label) is structurally low-sensitivity for this
failure mode. A higher-powered redesign would use the continuous
conflict *magnitude* (or cosine *value*) rather than the binary label,
and/or sample runs across regimes (failure + recovery) where the label
actually varies — registered as future work, NOT run post-hoc.

**Recorded conclusion:** H15b — no feature-specific conflict↔activity
bridge at the preregistered bar. This does not weaken stages 5/8/9
(which never asserted a bridge); it closes revision Gap 2 with a null
plus a documented redesign path. Full numbers:
`runs/ntk_bridge/ntk_bridge_report.json`; RESULTS.md §5A.8.

---

# v3.5 Preregistration (H16–H19) — committed BEFORE any v3.5 run

*Registered 2026-09-08 per the forward guide's discipline: hashes for this
section are recorded when committed; outcomes are appended only after the
corresponding stage completes.*

## H16 — Operator causal asymmetry: high-n resolution (Stage 16)

**Motivation (from the drift record).** The v3.4 stage-14 outcome (2/8
survivors at 8/8 batch consistency) sits at the exact sign-test floor for
n=8: p = 0.0039 is the minimum achievable, so the count cannot distinguish
a real boundary effect from two lucky features. This stage resolves T1.

**Design.**
- **H16a (n-raise):** re-run the stage-14 battery with **n ≥ 20 held-out
  function batches per feature** (MDE at 80% power ≈ 72% sign agreement —
  a real effect must be large to survive; a marginal one honestly does
  not), on the **top-16** SAE candidates by activation (MC-corrected
  across 16 tests), with the matched-deletion controls unchanged.
- **H16b (direction-reversal):** replace the retired mixed-sign condition
  with the **crossover criterion** already named in the H7 record: a
  causal feature must show a consistent *amplify-vs-ablate asymmetry* —
  amplification (α = 1.5) moves the target readout oppositely to
  ablation (α = 0), with the crossover direction consistent across
  ≥ 15/20 batches (exact binomial p < 0.05, two-sided).

**Decision rule (pre-written).**
- If ≥1 feature survives Bonferroni at n ≥ 20 AND satisfies the crossover
  criterion → the asymmetry is promoted to **"supported, small"** and §5A.7
  + abstract say so.
- Otherwise → **T1 is closed**: the causal side of the boundary was not
  found even at 2.5× power; the regime claim rests on the reconstruction
  side (PR 6.8, SAE-beats-PCA 4.7×) alone, and the paper says exactly
  that, without hedging.

**Machinery gate (unchanged):** planted-feature positive control through
the real operator hook must pass at the same n; a gate failure voids the
run.

## H17 — Controller generalization: the failure battery (Stage 17)

**Motivation.** v3.4's "controller beats NTK-adaptive" rests on ONE
failure (boundary starvation), one protocol — exactly the fragility H3
exposed. This stage resolves T2.

**Design.** Preregistered evaluation across the reproducible failure
classes of the atlas: **boundary starvation, spectral suppression, and the
2D reaction-diffusion pilot** (the classes whose operational labels
reproduce; gradient-conflict and collocation are excluded per the
existing label-reproducibility record). Same state-machine controller,
same no-oracle protocol, arms = {controller, NTK-adaptive, GradNorm,
no-action} × 3 seeds per class.

**Pre-written expectations (either direction is publishable):**
- H17a: the controller wins or ties on classes where no specialized
  method's trigger condition applies, and loses to the specialized method
  where it does (NTK on static misweighting, GradNorm on
  gradient-imbalance-style failures) → positioning: *"robustness across
  unknown failures, not per-failure SOTA."*
- H17b: the controller wins/ties everywhere tested → positioning:
  *"matches or beats specialized baselines across the reproducible
  failure set"* — still scoped to exactly the classes tested.
- The paper's SOTA paragraph is rewritten to match whichever pattern
  emerges; **no claim beyond the tested set.**

**`trigger_fourier_features` (R2b, pre-decided):** the action is REMOVED
from the controller's action space before this stage runs — a half-wired
action is not carried into a generalization test. (Implementing the
input-dim-changing warm restart is registered as explicit future work,
not a hidden stub.)

## H18 — Scope boundary probe: Fourier-feature PINN + depth (Stage 18)

**Motivation.** The strongest reviewer attack on the null: "width-16–512
tanh MLPs — of course no superposition." The FNO demonstrates the positive
side of the boundary; within-PINN architectural variation is missing.

**Design.** Fixed 1D Poisson task (the width-scaling control), width 64:
- (a) **Fourier-feature PINN** (the model config already carries
  `fourier_embed: true`, n_freq=32 — known to change the NTK/activation
  geometry substantially), 3 seeds;
- (b) **depth sweep** 2→6 hidden layers, 3 seeds each;
- the rank diagnostic (PR/W, tangent rank, PCA-95% components) is
  computed for every run and the SAE-vs-k-matched-PCA reconstruction
  comparison is run wherever PR moves.

**Pre-written outcomes (both strengthen the paper):**
- H18a: the Fourier-feature PINN's activation PR rises toward the
  superposition regime and the SAE's relative advantage over PCA appears
  *within PINNs* → the thesis "the regime boundary, measured, not
  asserted" gains a within-PINN demonstration, with the rank diagnostic
  as *predictor*.
- H18b: PR stays ~2 and the SAE stays null under the geometry-changing
  trick → the null is robust to architectural perturbation that *should*
  have moved the geometry; scope is honestly drawn at
  operator/function-space representations.

## H19 — Monitor label provenance audit (Stage 19)

**Design.**
- (a) RD2D audit: the v3.4 stage-10 RD2D rel L2 of 1.93 either reflects a
  genuinely hard manufactured solution or mislabels runs entering monitor
  training. Trace every RD2D run's label derivation; runs labeled
  "failure" for the wrong reason are excluded from monitor training and
  the audit is recorded in the artifact.
- (b) A loss-only **trajectory** monitor (logistic on past-loss windows,
  same splits/CIs) replaces the single-threshold straw man as the floor
  baseline; if it closes much of the 0.468→0.875 gap, the monitor section
  says so and repositions the conventional arm's value honestly.

**Acceptance:** the monitor section states its label provenance
explicitly and the floor baseline is no longer a threshold rule alone.
