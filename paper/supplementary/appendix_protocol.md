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

**Outcome (recorded after running):** H8a **confirmed**. E_T = −0.0186
[−0.0301, −0.0092]; 0/8 survive; all 88 target deltas positive
(representational signature); probe beats target in 34% of evaluations.

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

**Outcome:** H9a **confirmed**. PCA mean diff vs random −0.030 (4/11 runs
positive); SAE −0.090 (5/11). No candidate passes; the failure is
basis-independent.

## H3 — Dimensional boundary (Stage 10)

**Hypotheses.** (i) All 2D-input PDEs show local tangent rank exactly 2.
(ii) Covariance PR stays in the 2–3.5 band (no superposition onset from
input dimension alone at these widths). (iii) Time-dependent inputs (t, x)
do not raise PR above the steady 2D band because training explores an
effectively low-parameter family of states.

**Outcome:** All three confirmed. Tangent rank 2 everywhere; PR: AD-2D 2.47,
RD-2D 2.29, Burgers 1.63, Allen-Cahn 1.51. Time-dependent PR is *below*
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

**Outcome:** H4a **confirmed**. PR = 6.8 of 64 (ρ = 0.106, 5× the PINN);
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
approach the threshold. P(SAE monitor better) = 0.54 (no advantage). Label
ordering is stable for boundary/spectral regimes; the `success` regime's
label fraction is threshold-sensitive (0.3 → 0.0 over 0.02–0.20) — already
reflected in its "unlabeled" exclusion from failure-class claims.

---

## Interpretive discipline

- Stages 8–9 outcomes (H8a, H9a) *generalize* the negative result; they do
  not weaken it and were reported exactly as measured, including the
  4/88→31/88 raw beats-all counts that fail MC correction.
- Stage 11's positive control (H4a) is the paper's regime boundary; the SAE
  advantage claim is reconstruction-level only (no causal battery was run
  on the FNO — future work, stated explicitly).
- All artifacts referenced above are the JSON files in `runs/` produced by
  the stage scripts; none were edited by hand.
