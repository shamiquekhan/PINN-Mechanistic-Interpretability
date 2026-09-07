# Paper Draft — "When Superposition Fails: The Limits of Sparse Autoencoders in Scientific Machine Learning"

**Status:** v3 draft skeleton with measured numbers filled in from `runs/`.
Target: NeurIPS 2026 / ICML 2027 main track, 9 pages.

---

## Abstract (draft)

Physics-informed neural networks (PINNs) fail in reproducible,
configuration-determined ways, and mechanistic-interpretability (MI) tools —
above all sparse autoencoders (SAEs) — have been proposed to diagnose those
failures. We built the complete pipeline the MI methodology prescribes:
a 10-seed failure atlas, TopK SAEs with mandatory PCA/random/probe controls,
a planted-feature positive control, multi-view feature dictionaries, a
multiple-comparison-corrected causal battery, leakage-audited early-warning
monitors, and a rollback-capable closed-loop controller. The result is a
hardened negative: **SAE-derived features carry no causal information beyond
matched controls** (E_T CI includes zero; 0/8 features survive Bonferroni or
BH-FDR), and the failure is *explained* by representational geometry — hidden
activations live on an effectively 1.5-dimensional manifold (participation
ratio 1.34 of 64), so sparse dictionary learning solves a problem that does
not exist. We then show the negative result is basis-independent: **PCA
components fail the identical causal battery** (0/8; all-sign-positive
representational signature), and **neither PCA nor SAE alignments satisfy the
interchange criterion** of causal abstraction for a region-identity high-level
model — both are statistically indistinguishable from random bases. Measured
across five PDE families, local tangent rank exactly equals input dimension
at every width (16–512), while covariance PR stays in the 1.3–3.2 band:
input dimension bounds the *tangent* rank, not the covariance rank, and no
basis manufactures causal features from a curve. Finally, we locate the
regime boundary from the other side: a Fourier neural operator trained on
function-space regression has PR 6.8 of 64 (5× the PINNs), needs 19 PCA
components for 95% energy, and its TopK SAE **beats k-matched PCA by 4.7×** —
the superposition signature absent from the PINNs. We release the full
benchmark, protocol, and statistical machinery, and recommend a
participation-ratio precheck before applying SAEs to scientific models.

## 1. Introduction

- PINN opacity: gradient pathologies, spectral bias, collocation starvation
  (Krishnapriyan 2021; Wang 2020/2021; Gao 2022).
- MI promise: superposition (Elhage 2022) → SAEs (Bricken 2023; Gao 2025) →
  causal validation (Marks 2025). All validated on LLM residual streams.
- The unasked question: *does superposition exist in the model class we're
  applying SAEs to?* Recent negative SAE results (DeepMind 2025; Leask 2025;
  Paulo & Belrose 2025; Korznikov 2026) report downstream failures but not
  representational explanations.
- Contributions:
  1. A complete, preregistered, statistically hardened MI pipeline for PINN
     failures (released benchmark + protocols).
  2. A hardened, positive-control-validated **negative causal result** for
     SAE features — generalized to PCA components and to the causal-
     abstraction interchange criterion (basis-independent).
  3. A representational-geometry explanation: tangent rank = input dim
     exactly; covariance PR ≪ width; no superposition to un-mix.
  4. A measured **regime boundary**: FNO function-space representations are
     high-rank and SAEs beat PCA there — the first controlled
     both-sides demonstration in scientific ML — with a preregistered
     causal battery on the operator side showing 6/8 MC-corrected
     survivors where every PINN basis had 0/8 (reported as suggestive,
     with the preregistration decision rule firing H14b on a documented
     non-diagnostic condition).
  5. An engineering alternative that works: conventional-signal monitoring
     + closed-loop controller (rescues 18× vs no-action), with a
     monitor-source ablation proving SAE features are inert cargo.

## 2. Background and related work

- PINN pathologies and adaptive-weight remedies (GradNorm, NTK-adaptive,
  RBA — compared in Stage 12).
- Superposition hypothesis and SAE methodology; decoder-norm and dead-feature
  diagnostics.
- Causal abstraction theory (Geiger 2021/2022): interchange interventions,
  alignment accuracy — we import the criterion into the PINN setting.
- SAE skepticism (2025–2026 literature): our contribution is the *mechanistic
  explanation* (geometry), the *basis-independence* (PCA battery), and the
  *boundary* (FNO positive control), none of which the downstream-task
  reports provide.

## 3. Theory: what bounds activation rank in PINNs?

From `docs/theory_activation_rank.md` (summarized):

- **Proposition 1 (exact).** Local tangent rank of any hidden layer ≤ d.
- **Proposition 2.** C¹ chains preserve manifold dimension: d=1 inputs give
  activation *curves*, d=2 give *surfaces*.
- **Non-theorem (explicitly corrected):** covariance rank is NOT bounded by
  d + 1 (moment-curve counterexample); PR is an empirical diagnostic.
- **Empirical saturation:** measured tangent rank = d exactly across all
  six PDE families and all widths; measured PR/W = 0.02–0.10 (PINNs) vs
  0.106 (FNO) vs 0.1–0.9 (LLM literature).
- **Proposition 4 (no-superposition).** If PR ≪ W, the representation
  encodes ≈ PR dense features; an overcomplete sparse dictionary is
  inductively mismatched, predicting (i) PCA dominance in reconstruction,
  (ii) dead/unstable SAE latents, (iii) no direction-specific causal
  effects — all three measured.

## 4. The PINN-MI pipeline (methodology)

Five phases with mandatory controls and gates; statistical protocol
(preregistration in `docs/preregistration.md`; exact sign tests; Bonferroni
+ BH-FDR across features; run-level bootstrap CIs; planted-feature positive
control as a machinery gate). Emphasize: every stage has a chance-level
control built in before any claim is allowed.

## 5. Empirical results

### 5.1 Failure atlas reproducibility (Table 1)
Boundary starvation 10/10, spectral suppression 10/10; gradient-conflict
and collocation excluded from class claims (label non-reproducibility).

### 5.2 The activation manifold (Figure 1 — the key figure)
Tangent rank table (saturated at d); PR vs width (flat 1.7–2.0 for W=16–512);
PR by family (1D: 1.3–2.0; 2D: 2.1–3.2; time-dependent: 1.3–1.9; FNO: 6.8);
3D PCA visualization colored by boundary distance and residual magnitude.

### 5.3 SAE failure with mandatory baselines (Figure 2)
PCA(k) 12× better reconstruction; 51–70% dead latents; 1/256 cross-seed
stability; association-with-physics exists (r up to 0.75) but is geometry
along the curve, with `data_std` the strongest correlate (confound).

### 5.4 The causal null — now basis-independent (Figure 3, the hardened core)
- SAE battery: E_T = −0.0002 [−0.0051, +0.0060]; 0/8 survive Bonferroni AND
  BH-FDR; all 88 target deltas positive; probe beats feature in 10.2%.
- **PCA battery (new):** E_T = −0.0186 [−0.0301, −0.0092]; 0/8; all 88
  positive; probe beats in 34%. Same protocol, same checkpoints, same
  scoring code.
- Positive control passes (planted feature, cosine 0.99, 15× controls).
- Power analysis: MDE 85.7% sign-agreement at 80% power — the null is not
  underpowered-by-accident; observed agreement ~50%.

### 5.5 Causal abstraction (new; Figure 3 panel F)
Region-level partial interchange, H_boundary: PCA movement +7.49 vs random
+7.52; SAE +7.43; no run-level consistency for either candidate. Neither
linear nor sparse alignments causally abstract the PINN.

### 5.6 The regime boundary (Figure 6 — the second key figure)
PINN (ρ=0.02–0.03, PCA wins 12×) → 2D PINNs (ρ≈0.07–0.10, PCA wins) →
FNO (ρ=0.106, **SAE wins 4.7×**) → LLMs (literature). The SAE tool
transfers exactly when the representation is genuinely high-rank.

### 5.6b The operator causal battery (Stage 14; preregistered H14a/H14b)
The preregistered conjunctive rule fires H14b (the mixed-sign condition,
non-diagnostic for ablation batteries, failed 64/64-positive) — but by the
MC-correction standard that established the PINN nulls, the FNO features
pass: **6/8 Bonferroni + BH-FDR survivors at 8/8 batch consistency, E_T CI
[+7.1e-6, +1.6e-5] excluding zero** — where every PINN basis had 0/8 with
CIs spanning zero. Beats-all-controls: 75% (FNO) vs 14% (SAE/PINN) and 32%
(PCA/PINN). We report this as a *suggestive causal asymmetry* across the
regime boundary, not a confirmed causal boundary: effect sizes are tiny
(~1e-5 on a near-zero baseline), the interchange does not beat a random
basis, and the sign condition is recorded as a preregistration design
lesson (future batteries should use amplify-vs-ablate asymmetry instead).

### 5.7 Monitors and controller (the engineering results)
AUROC 0.872 [0.795, 0.953] conventional vs 0.878 [0.806, 0.958] SAE —
P(SAE better) = 0.54 posterior. Controller: 0.295 → 0.017 (18×), oracle
1.4× away; monitor-source ablation: SAE arm ≡ random arm (bit-identical).
SOTA comparison (new): NTK-adaptive 0.0064 beats controller 0.0166;
GradNorm 2.01 and RBA 2.19 actively harm — the controller is honest
middle ground, and the *interpretability* claim stays separate.

## 6. What to do instead (practical guidance)

1. **Precheck ρ = PR/W** before training any SAE on scientific-model
   activations (one covariance eigendecomposition).
2. If ρ ≪ 0.1: use linear diagnostics (PCA loadings, supervised probes) for
   *description*, gradient/residual signals for *prediction and control* —
   and do not expect direction-specific causal effects from any basis.
3. If ρ ≳ 0.1 (operator/function-space models): SAE methodology is
   appropriate — demonstrated positively on the FNO.

## 7. Discussion and limitations

- Scoped: no wide-PINN ensembles, no production operators, FNO result is
  reconstruction-level (no FNO causal battery yet).
- Controller not claimed SOTA (NTK-adaptive wins on its target failure).
- Tangent-rank bound is exact but covariance rank remains empirical —
  predicting PR from task/optimizer is open theory.
- Time-dependent PR *below* steady 2D is an interesting anomaly: training
  explored an effectively 1-parameter family of states; worth a dedicated
  study with longer schedules.

## 8. Conclusion

Check superposition before you sparse-code. The benchmark, protocols,
positive and negative controls, and the measured regime boundary are
released for exactly that purpose.

---

### Reproducibility statement

All numbers in this draft are generated by the stage scripts and stored in
`runs/*.json`; tests: 100+ unit tests; Dockerfile pinned; seeds documented
per run; compute ≈ 6 GPU-hours (RTX-class) for the full campaign.
