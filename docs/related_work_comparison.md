# Related-Work Comparison and Novelty Decision Record

**Status:** source-level comparison for the next research phase
**Date:** October 25, 2026
**Scope:** PhysSAE, SAE evaluation/stability literature, and PINN numerical-analysis context

This document is a design input for future confirmatory work. It does **not**
rewrite the frozen publication campaign, re-score existing artifacts, or claim
that the current repository has independently reproduced every result in the
papers below.

## Executive conclusion

The closest novelty risk is not “SAEs applied to PINNs”: PhysSAE already makes
that contribution. The defensible unresolved question is narrower:

> **When does physical alignment of a PINN representation predict causal
> specificity under a prespecified intervention, and how does that relationship
> depend on representation geometry and feature stability?**

The two projects are not currently in direct numerical contradiction. PhysSAE
evaluates alignment, direct hidden-state functional intervention, and spatial
localization. This repository additionally evaluates reconstruction against
PCA, target-versus-control effect magnitude, causal-abstraction interchange,
dose response, checkpoint dependence, and geometry. Those are related but not
identical estimands.

The highest-value next experiment is therefore a **same-checkpoint,
same-layer, same-dictionary head-to-head** that reports both evidence ladders:

1. PhysSAE-style alignment and spatial-footprint localization.
2. This repository's target-versus-control causal-specificity estimand.

The result should be interpreted as a dissociation or convergence of criteria,
not as a contest in which one paper's metric is declared universally correct.

## Sources and verification level

| Source | What was inspected | Verification status |
|---|---|---|
| Patil, Eshwar, and Honnavar, *PhysSAE: Mechanistic Interpretability with Sparse Autoencoders*, arXiv:2609.07061v2 | Abstract, Sections 2–5, Tables 1–8, Appendix E, intervention equations, control and feature-consistency procedures | Full paper text inspected; public code was not available in the version inspected—the paper says code/checkpoints will be released upon acceptance |
| Karvonen et al., *SAEBench*, PMLR 267 (2025) | Official abstract and evaluation framing | Official proceedings page inspected; this is language-model evidence, not a PINN replication |
| Paulo and Belrose, *Sparse Autoencoders Trained on the Same Data Learn Different Features*, ICLR 2026 | Full conference PDF: methods, matching definition, seed experiments, TopK/ReLU comparison | Full conference paper text inspected; domain is LLMs, so transfer to PINNs remains an open question |
| De Ryck and Mishra, *Numerical analysis of physics-informed neural networks and related models*, Acta Numerica 33 (2024) | Official abstract and numerical-analysis scope | Official journal page/PDF inspected; used only for PINN error and training context |
| This repository, especially `docs/physSAE_reconciliation.md` and `runs/r1_physSAE/` | Current implementation, prior reconciliation, and committed matched-comparison artifacts | Source and artifact review; no new model run performed for this document |

Primary external references:

- [PhysSAE arXiv record](https://arxiv.org/abs/2609.07061)
- [SAEBench proceedings](https://proceedings.mlr.press/v267/karvonen25a.html)
- [Same-data SAE stability paper](https://proceedings.iclr.cc/paper_files/paper/2026/hash/3c1fe56b043848b211030c202764c6a7-Abstract-Conference.html)
- [Acta Numerica PINN review](https://doi.org/10.1017/S0962492923000089)

## 1. Required side-by-side comparison

| Dimension | This project | PhysSAE | Required action before a new confirmatory claim |
|---|---|---|---|
| **Scientific question** | Separates representation geometry, reconstruction quality, physical association, causal specificity, causal abstraction, and stability. The current central thesis is that reconstruction advantage does not imply direction-specific causal mechanism. | Asks whether PINN penultimate-layer SAE atoms align with independent physical concepts and have a localized functional effect in the output field. | State one shared hypothesis—whether alignment predicts a specified causal property—and list the non-shared hypotheses separately. |
| **Feature selection** | Stage 5 uses the leading candidate features from the committed physics-feature dictionary; later batteries use top-activity or registered candidate sets depending on the experiment. `sae/features.py` computes multi-view correlations and cross-seed matching, but selection and confirmation must remain split in a new study. | For each reference concept, selects top-aligned atoms using correlations computed on the held-out evaluation split; the paper also evaluates bilateral positive/negative pairs and matched random atoms. | Reproduce the exact selection rule on a selection split, then evaluate selected atoms only on a fresh confirmatory split. Record whether selection is by correlation, activation, concept, or activity. |
| **Activation layer** | The main SAE/PCA causal battery historically targets `model.acts[layer_index=1]` for the default PINN, with explicit reconstruction-only and natural baselines. Some reconciliation work separately re-extracts penultimate activations. | Uses the penultimate post-activation. Because the output head is linear, the direct intervention has an especially simple readout path. | Use the same penultimate layer for the head-to-head comparison, and report the existing middle-layer result as a separate experiment rather than combining layers. |
| **SAE representation** | TopK SAE is the main current dictionary (`k=8`, commonly 4× expansion); legacy ReLU+L1 and other bases appear in reconciliation and boundary studies. | ReLU+L1 SAE, typically 4× overcomplete (`D=512` for width 128), z-scored activations, unit-norm decoder columns, and a reported operating sparsity around the stated range. | Match PhysSAE's ReLU+L1 configuration exactly for the direct comparison, then report TopK as a separately registered representation family. |
| **Intervention location** | The ordinary SAE hook encodes a hidden activation, modifies the latent, and decodes an intervened reconstruction back into the network. This tests the downstream effect of the SAE-mediated representation. | Directly subtracts `α z_k d_k` from the original frozen hidden state and bypasses the SAE decoder. | Treat these as different intervention estimands. A direct hidden-state intervention must be implemented and reported alongside the decoder-mediated hook. |
| **Target intervention** | Typical target is latent/component ablation (`z_k ← 0`) or amplification, with loss deltas decomposed into projection, incremental intervention, and total effects. | Atom ablation is evaluated at multiple `α` values; the direct hidden-state effect is exactly proportional to the atom activation and output-readout projection under the linear output head. | Freeze α values and report both output-field effects and physical-loss effects. Do not call a decoder-mediated ablation a direct hidden-state ablation. |
| **Controls** | Activity-matched component deletion, random active-component deletion, norm-matched random-direction injection, supervised probe injection, and PCA controls are now represented as distinct operator families. The frozen Stage 8 artifact used activity-matched component control, random-direction injection, and probe injection. | Ten matched random atoms selected by activation magnitude and decoder-norm similarity, with the same direct hidden-state intervention applied to controls. | Keep within-dictionary matched-atom controls separate from across-basis random-direction controls. Report selected indices, operator kind, dose, seed, and no-op status. |
| **Outcome metric** | Main causal metric is target effect relative to the strongest control (`E_T`), with specificity, exact sign tests, multiplicity correction, dose response, and interchange tests. Reconstruction and physical alignment are separate outcomes. | Main causal/localization evidence is spatial concentration of the output intervention footprint (`ESF80`), compared with PCA/ICA and matched random controls. Alignment uses Pearson correlation with independent reference concepts. | Report ESF80 and `E_T` side by side. Neither metric should be used as a proxy for the other. Define the primary outcome before feature selection. |
| **Physical reference** | Uses manufactured/reference PDE quantities and multiple metric views; the repository explicitly flags run-type and data-standard-deviation confounds. | Uses independent analytic or high-resolution numerical reference solutions, not the PINN output, to define concept fields. | Preserve independent reference fields, document interpolation and derivative construction, and prohibit PINN-derived concept labels in confirmatory alignment. |
| **Statistical inference** | Uses paired sign tests across repeated evaluations, Bonferroni and BH-FDR across candidates, bootstrap intervals, and separate hierarchical analysis for checkpoint clustering. The new audit calibrates the actual max-control statistic under synthetic cases. | Reports permutation nulls for alignment and matched-control comparisons for ESF80; the paper does not provide a directly comparable hierarchical model for all feature-level outcomes. | State the independent unit for every analysis. Add run-level clustering and Monte Carlo uncertainty to any new head-to-head result. Do not treat grid points as independent replications. |
| **Feature stability** | Existing code matches SAE decoder directions across seeds and reports non-unique dictionaries; the R4/R1 artifacts distinguish feature identity from regime-level conclusions. | Uses Hungarian matching across three SAE seeds and reports low matched cosine while arguing that ESF80 is more stable than individual atom identity. | Measure both individual-feature stability and functional/subspace stability. Do not interpret low feature cosine as absence of a stable representation without a subspace analysis. |
| **PDE coverage** | Includes steady 1D/2D PDEs, time-dependent PDEs, Fourier-feature PINNs, and an FNO positive-control regime; the current main causal nulls and geometry claims are scoped by experiment. | Six 1+1D PDE families: heat, Burgers, Allen–Cahn, easy/hard convection, and Schrödinger, spanning converged, partially converged, and failed PINNs. | Use an overlapping subset for direct comparison and keep unmatched PDEs as external-validity evidence. Record solver quality and training schedule because they alter representation geometry. |
| **Main conclusion** | Alignment, reconstruction, and causal specificity can diverge; geometry identifies reconstruction regimes but is not a continuous predictor of causal advantage. | Reports strong physical alignment and more concentrated direct intervention footprints for top-aligned atoms, while explicitly distinguishing alignment, functional relevance, and causal evidence. | Formulate the paper around the relationship among these levels. Do not claim that either study falsifies the other until the estimands are aligned. |

## 2. What the full PhysSAE methods establish

The methods comparison is more specific than an abstract-level summary:

- PINNs use five hidden layers of width 128 with `tanh`, Xavier initialization,
  Adam for 8,000 iterations, and up to 300 L-BFGS function evaluations. The
  loss uses residual, boundary, and initial-condition terms with the reported
  weighting. Activations are collected on a dense 20,000-point space-time grid.
- The SAE is trained on standardized penultimate activations with an 80/20
  activation split. Concepts come from analytic or high-resolution numerical
  reference solutions independent of the PINN.
- Direct ablation subtracts `α z_k d_k` from the original hidden state rather
  than decoding an intervened SAE reconstruction. This makes the output effect
  analytically tied to `z_k` and the output projection of `d_k`.
- The negative control is not a generic random vector. It selects ten atoms
  close to the target in mean activation and decoder norm, then applies the
  identical direct intervention to each control.
- The paper evaluates ESF-style spatial concentration, bilateral concept
  regression, dose-response values, and cross-SAE-seed Hungarian matching. It
  reports low atom-level cosine stability but more stable localization metrics.

These details matter because a positive ESF80 result can coexist with a null
target-versus-control loss-specificity result. The former asks whether the
effect footprint is spatially concentrated; the latter asks whether the target
direction changes a selected physical loss more than appropriate controls.

## 3. What is genuinely comparable today?

The repository already contains a substantial reconciliation in
[`docs/physSAE_reconciliation.md`](physSAE_reconciliation.md), including an
R1 artifact that matches the PhysSAE architecture, penultimate layer,
normalization, reference fields, alignment, ESF80, and direct intervention
specification where feasible. That work should be treated as the existing
matched-comparison foundation, not repeated as an unplanned exploratory run.

The current comparison supports these bounded statements:

1. Both projects treat alignment as an association with an independently
   defined physical/reference quantity, not as proof that a feature is a
   physical variable.
2. PhysSAE's direct hidden-state intervention and this project's ordinary SAE
   hook are not the same operation. The existing R1 work is therefore essential
   for any claim that the two evidence families have been compared fairly.
3. Low atom-level seed stability is not unique to either project. It should be
   reported as a feature-identifiability result and separated from
   function-level or subspace-level stability.
4. The current repository's geometry result is a regime result: participation
   ratio helps separate settings where SAE reconstruction beats rank-matched
   PCA, but current R10 evidence does not support treating it as a continuous
   predictor of the effect size.
5. A negative `E_T` result does not refute physical alignment or spatial
   localization. It rejects a stronger, specific causal interpretation under
   the registered intervention and control family.

## 4. Broader literature implications

### SAEBench: proxy metrics are not enough

SAEBench evaluates SAEs across eight metrics spanning interpretability,
disentanglement, and practical tasks, and reports that improvements in proxy
metrics do not reliably transfer to practical performance. The direct lesson
for this project is methodological: reconstruction error, sparsity, alignment,
localization, and causal specificity must remain separate columns in the
evidence table. A single composite “interpretability score” would obscure the
scientific disagreement this project is trying to measure.

### Same-data SAE stability: feature identity is a hypothesis

The ICLR 2026 study trains SAEs on the same model and data with different
initialization seeds and finds substantial seed dependence, including stronger
dependence for TopK than for some ReLU+L1 settings. It is an LLM study, not a
PINN result, so it does not establish that PINN dictionaries behave the same
way. It does establish that feature-level reproducibility must be measured
rather than assumed.

The repository already has the right starting point: decoder matching and
regime-level stability. The next analysis should add activation similarity,
held-out physical alignment, and subspace/function-level causal stability.

### PINN numerical analysis: representation claims depend on solver context

The Acta Numerica review organizes PINN error into approximation,
generalization, and training components and emphasizes PDE-, regularity-, and
stability-dependent behavior, with training error as a key bottleneck. For this
project, representation geometry cannot be interpreted independently of the
training regime, reference solver, PDE difficulty, and held-out physical error.
Every geometry claim should therefore record solver quality and training
configuration alongside PR, tangent rank, and reconstruction outcomes.

## 5. Proposed research question and estimands

This is the recommended working question for the next preregistration; it is
not yet a new registered hypothesis:

> **When does physical alignment of a PINN representation predict causal
> specificity under controlled interventions, and how does that relationship
> depend on representation geometry and feature stability?**

Keep the following quantities separate:

| Quantity | Prespecified question | Candidate measurement |
|---|---|---|
| Physical alignment | Does a feature correspond to an independently defined reference quantity? | Held-out correlation plus a permutation/random-direction null |
| Reconstruction | How well does the representation compress hidden activations? | Held-out SAE error versus rank-matched PCA |
| Causal effect | Does intervention change the selected physical readout? | `Δ_k = L_k - L_nat` or the explicitly selected reconstruction-only contrast |
| Causal specificity | Does the target effect exceed the registered controls? | A frozen target-versus-control contrast, with its null calibrated for the actual control maximum |
| Feature stability | Does an individual feature or equivalent subspace recur? | Cross-seed matching, activation similarity, and subspace alignment |
| Geometry | What activation structure accompanies the outcome? | PR/covariance spectrum and tangent-rank diagnostics, reported separately |

The confirmatory protocol must specify whether the primary readout is PDE
residual, boundary/initial-condition loss, held-out solution error, output-field
localization, or a vector of outcomes. It must also specify whether the baseline
is natural or reconstruction-only and whether direct hidden-state and
decoder-mediated interventions are separate arms.

## 6. Novelty options and decision rule

| Candidate contribution | Existing foundation | Essential new evidence | Main failure risk |
|---|---|---|---|
| **A. Calibrated causal-validity framework** | Intervention engine, matched controls, positive controls, hierarchical analysis, and the PhysSAE reconciliation | A preregistered same-checkpoint comparison showing when alignment/localization and target-specific causal effect agree or diverge | The result may be a criterion difference rather than a new scientific finding unless the divergence changes interpretation |
| **B. Geometry-dependent limits of SAE interpretability** | PR/tangent-rank work, Fourier/PINN boundary, PCA comparisons, R10 | Held-out, independent-run validation that geometry predicts a regime boundary and clarifies which interpretability level changes | PR may be a useful gate but not a general predictor; the claim must stay bounded |
| **C. Feature-level versus subspace-level causal stability** | R4 seed robustness and non-unique dictionary findings | Cross-SAE and cross-PINN seed study with matched features, subspaces, held-out alignment, and causal effects at both levels | Individual features may be unstable and subspace effects may also fail to reproduce |

Recommended decision rule:

1. Use **A** as the primary paper framing because it directly addresses the
   PhysSAE overlap through an estimand comparison.
2. Use **B** as the supporting explanation for why reconstruction advantages
   appear in some regimes and not others.
3. Promote **C** to a co-equal contribution only if subspace-level stability
   reproduces on held-out PDE settings and changes the causal interpretation.

## 7. Required protocol before confirmatory training

Before launching a new model batch, freeze the following in a new
preregistration:

- one primary hypothesis and two supporting hypotheses;
- the primary causal estimand and smallest scientifically meaningful effect;
- the independent unit and hierarchy (`PDE family → PINN seed → SAE seed →
  checkpoint → evaluation point`);
- selection, calibration, and confirmatory splits;
- penultimate-layer direct intervention and decoder-mediated intervention as
  separate arms;
- within-dictionary matched-atom controls versus random-direction injection;
- the exact dose grid and no-op handling;
- held-out reference fields and evaluation points;
- Type I error, multiplicity correction, confidence-interval, and clustering
  procedures;
- power targets based on independent runs rather than grid-point counts;
- explicit exploratory, historical, and confirmatory labels.

The current audit's synthetic calibration is a method check, not a substitute
for this prospective model-based replication. No new model run should be
promoted to confirmatory evidence until the above choices are frozen.

## 8. Claim-to-evidence mapping for the next paper

| Claim family | Minimum evidence | Current status |
|---|---|---|
| Alignment is present | Independent reference fields, held-out evaluation, permutation/null comparison | Existing evidence; retain cautious terminology |
| Alignment is not sufficient for causal specificity | Same selected features, same held-out inputs, calibrated target/control estimand | Strong project direction; direct cross-paper comparison is the next decisive test |
| Geometry identifies reconstruction regimes | Independent runs, rank-matched PCA, held-out activations, cluster-aware uncertainty | Existing regime evidence; continuous prediction is not supported |
| Features are reproducible | Multiple SAE and PINN seeds, feature matching plus subspace/function metrics | Partially established; subspace-level confirmatory test remains open |
| Conclusions generalize | At least one PDE family held out from protocol formulation | Must be declared before the next run |

## 9. Decision record

**Do now:** complete the literature comparison, freeze the estimand and
independent unit, and extend the calibration plan to correlated controls,
checkpoint clusters, selection leakage, and realistic effect sizes.

**Do not do yet:** add unrelated architectures, rewrite historical artifacts,
or select a new test because it gives a more favorable causal verdict.

**Primary paper direction:** a calibrated comparison of physical alignment,
reconstruction, spatial/function-level intervention, and target-specific causal
effect, with geometry and feature stability as explanatory/supporting analyses.

This document is a planning and novelty record. The proposed choices have now
been converted into the prospective protocol in
[`preregistration_alignment_specificity_2026-10-25.md`](preregistration_alignment_specificity_2026-10-25.md),
which must be committed before the corresponding model runs.
