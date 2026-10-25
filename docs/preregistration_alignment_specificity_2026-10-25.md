# Preregistration: Alignment-to-Causal-Specificity Head-to-Head

**Protocol version:** 1.0
**Registered:** October 25, 2026, before any new confirmatory training or
model-based scoring
**Status:** design lock for the next research phase

This preregistration converts the related-work comparison in
[`related_work_comparison.md`](related_work_comparison.md) into a prospective
protocol. It does **not** rewrite the frozen publication campaign, re-score
historical R1/R2/R6 artifacts, or promote the existing synthetic calibration
to model-based evidence.

The primary question is deliberately narrower than “are SAE features
interpretable?”:

> **Does physical alignment measured before causal evaluation predict
> target-specific causal effect under a prespecified intervention, and does
> that relationship depend on the intervention operator?**

The unit of replication is a trained PINN checkpoint, not an evaluation grid
point, feature, or collocation batch.

---

## 1. Hypotheses and decision rules

### H1 — Primary: alignment predicts causal specificity

For each independent PINN checkpoint, rank a fixed candidate set by
selection-split physical alignment. Measure each candidate's causal contrast on
independent confirmatory batches. The checkpoint-level Spearman association
between those two quantities is the primary observation.

Define the confirmatory causal contrast for candidate `k` and batch `b` as:

```text
D[k,b] = |Δ_target[k,b]| -
         max(|Δ_matched[k,b]|,
             |Δ_random[k,b]|,
             |Δ_probe[k,b]|)
```

The candidate-level primary causal-contrast summary is:

```text
D_bar[k] = mean_b D[k,b]
```

The signed project statistic is retained separately for compatibility with
existing reports:

```text
E_T_signed[k] = mean_b(Δ_target[k,b] -
                       max(Δ_matched[k,b],
                           Δ_random[k,b],
                           Δ_probe[k,b]))
```

`D_bar` is the primary H1 quantity. `E_T_signed` is reported as a signed
secondary diagnostic and is never substituted for `D_bar` after the results
are observed.

The primary alignment score is the maximum absolute Pearson correlation with
the pre-registered independent-reference concept panel on the **selection
split**:

```text
A[k] = max_c |corr(z_k, reference_concept_c)|
```

For each checkpoint, compute `rho_checkpoint = Spearman(A[k], D_bar[k])`
across the fixed candidate set. The primary estimand is the mean checkpoint
association on the Fisher-z scale, back-transformed for reporting:

```text
theta_primary = mean_checkpoint(atanh(rho_checkpoint))
```

**Directional hypothesis:** `theta_primary > 0`.
**Smallest meaningful association:** back-transformed `rho = 0.30`.

H1 is supported only if all conditions hold:

1. The one-sided checkpoint-level randomization test is `p <= 0.05`.
2. The lower bound of the 95% checkpoint-cluster bootstrap interval for
   `theta_primary` is above `atanh(0.30)`.
3. The machinery gate passes for the direct intervention arm.
4. The result is present without using confirmatory outcomes for feature,
   dose, control, or checkpoint selection.

A result with a positive point estimate but an interval crossing the registered
`rho = 0.30` threshold is reported as directional evidence, not support for
the primary claim. A null does not establish that no causal features exist.

### H2 — Supporting: alignment predicts spatial localization

Using the same selection-split alignment score and the same fixed candidates,
test whether higher `A[k]` predicts a larger PhysSAE-style localization
advantage on the confirmatory split:

```text
L[k] = mean(ESF80_matched_controls[k]) - ESF80_target[k]
```

Positive `L` means that the target footprint is more concentrated than its
matched controls. H2 uses the same checkpoint-level association procedure as
H1. H2 is secondary and cannot replace a failed H1.

### H3 — Supporting: intervention location changes the estimand

On the same penultimate activation site and the same frozen dictionary, compare
two explicitly separate intervention arms:

1. **Direct hidden-state arm:**
   `h_cf = h + (s - 1) * z_k * d_k`, with the trained activation
   standardization mapped back into raw hidden-state units. Here `s` is the
   latent scale: `s=1` is natural and `s=0` is full ablation.
2. **Decoder-mediated arm:** encode the hidden state, modify `z_k`, decode,
   and replace the hidden activation through the forward hook.

H3 tests the two-sided interaction between alignment score and intervention
arm. It asks whether the alignment-to-effect slope differs by operator; it does
not assume which arm will be larger. The arms are never pooled into one causal
statistic.

---

## 2. Scope and substrates

### Primary substrate

The primary confirmatory comparison uses the PhysSAE-overlapping PINN setup so
that the direct hidden-state operation and independent reference fields are
defined before any result is read:

- PDE families: Burgers and Allen–Cahn.
- PINN: five hidden layers, width 128, `tanh`, Xavier initialization.
- Training: Adam for 8,000 steps followed by up to 300 L-BFGS evaluations.
- Loss weights: `w_BC = w_IC = 100`, `w_F = 1`.
- Activation site: the penultimate post-activation immediately before the
  linear output layer.
- Reference fields: analytic or high-resolution numerical solutions generated
  independently of the PINN.
- Evaluation grid: the fixed `200 x 100` space-time grid used by the matched
  PhysSAE comparison, with its exact ordering and hash recorded in the
  artifact.

The primary dictionary is a ReLU+L1 SAE with `D=512`, `lambda=0.02`,
unit-normalized decoder columns, and three SAE seeds per PINN checkpoint.

Secondary basis arms are TopK SAE, rank-matched PCA, and matched random
directions. They use the same activation bank, split, candidate-slot count,
and confirmatory batches. Secondary arms cannot be used to retroactively
replace the primary basis.

Existing frozen H18 Fourier-feature and tanh checkpoints may be used for
software smoke tests or an explicitly labelled pilot. They are not part of the
primary confirmatory estimand in this registration.

### What is not being tested

- No new controller, monitor, FNO, DeepONet, or unrelated architecture branch.
- No re-scoring of frozen Stage 5, Stage 8, R1, R2, R6, or operator artifacts.
- No claim that alignment is a physical mechanism.
- No claim that a null under this protocol proves the absence of causal
  features in other PDEs, architectures, layers, or training regimes.

---

## 3. Independent units and sample-size rule

The hierarchy is fixed before data collection:

```text
PDE family
  -> independent PINN seed / trained checkpoint
    -> SAE seed
      -> candidate slot
        -> confirmatory evaluation batch
          -> grid points
```

- The **independent unit** for H1–H3 is the trained PINN checkpoint.
- PDE family is a prespecified blocking factor.
- SAE seeds and candidate features are nested repeated measurements, not
  independent replications.
- Evaluation grid points and collocation points are never counted as separate
  samples for power or inferential degrees of freedom.

### Pilot and confirmatory sample size

The pilot is a variance-estimation exercise and has no confirmatory verdict.
It uses three fresh PINN seeds per primary PDE family and the full protocol
except for the final confirmatory decision. Pilot results may determine the
number of independent confirmatory PINN seeds only through the rule below.

Before any confirmatory outcome is inspected:

1. Estimate the checkpoint-level variance of Fisher-z `rho_checkpoint` from
   the pilot.
2. Compute power for `rho = 0.30`, one-sided `alpha = 0.05`, using the exact
   registered checkpoint-level randomization test.
3. Choose the smallest `N` in `{6, 8, 10, 12}` PINN seeds **per PDE family**
   that reaches 80% simulated power.
4. If no candidate reaches 80% power, run the maximum `N=12` and report the
   achieved power and the resulting limitation.

The chosen `N`, pilot seed list, simulation seed, estimated variance, and power
table must be written to the preregistered protocol artifact before any
confirmatory checkpoint is scored. No confirmatory seed may be added or
removed based on its outcome.

Each confirmatory checkpoint uses exactly three SAE seeds. Results are first
aggregated across SAE seeds within checkpoint; SAE seed rows are also retained
for a nested-sensitivity analysis.

---

## 4. Data, dictionary, and analysis splits

All split assignments are generated once from a deterministic hash of
`{pde_family, pinn_seed, grid_index}` and stored in the run manifest. A later
rerun must use the stored assignments rather than regenerate them.

### Activation bank split

The fixed `200 x 100` grid is partitioned into:

- **Dictionary-train split: 60%** — fit standardization statistics and train
  the SAE or fit the linear basis.
- **Selection split: 20%** — compute `A[k]`, rank the fixed candidate set,
  and train the supervised probe control.
- **Confirmatory split: 20%** — report independent alignment and ESF80
  validation; it is never used to select candidates, controls, doses, or
  checkpoints.

Standardization means and standard deviations are fit on the dictionary-train
split only. Any direct hidden-state intervention must transform a dictionary
direction from standardized coordinates into raw hidden-state coordinates using
those stored scale parameters. A missing or inconsistent scale record voids
that arm rather than silently changing the operator.

### Candidate selection

1. Fit the dictionary using only the dictionary-train split.
2. Form the candidate pool as the top 16 components/features by mean absolute
   activation on the dictionary-train split.
3. Compute the selection-split alignment score `A[k]` for every member of that
   fixed pool.
4. Evaluate all 16 candidate slots confirmatorily; do not keep only favorable
   candidates.

The concept panel is fixed before scoring:

- solution amplitude;
- spatial-gradient magnitude / shock or interface indicator, where defined;
- temporal-derivative magnitude, where defined;
- the signed solution field.

Coordinates, model outputs, and PINN-derived residual labels are not treated as
physical concepts for the primary alignment claim. The exact panel and any
PDE-specific unavailable entries are recorded in the manifest.

### Causal batch split

The causal measurement uses independently generated collocation batches:

- **Calibration batches:** used only for machinery checks, variance estimates,
  and dose confirmation; no primary verdict.
- **Confirmatory batches:** fixed before scoring and shared exactly across all
  target and control arms.

The confirmatory batch seed list, number of points per batch, PDE sampler, and
evaluation-point tensors are stored in the run manifest. If a batch is invalid,
the reason is recorded and the batch is not silently replaced.

---

## 5. Intervention and control protocol

### Primary dose

The primary causal comparison uses ablation, `s = 0`, with the natural
state as the direct-arm baseline. The decoder-mediated arm uses the
reconstruction-only baseline so that SAE insertion error is not mislabeled as
intervention effect. Natural, reconstruction-only, intervention, projection,
and incremental effects are all recorded separately.

The registered secondary dose curve is:

```text
s in {-2, -1, 0, 0.5, 1.5, 2}
```

The dose curve is descriptive/supporting evidence. It cannot change the
primary dose or the H1 decision rule after confirmatory outcomes are observed.

### Control families

Controls are retained as separate columns and separate audit records:

1. **Activity-matched non-target component:** the non-target feature with the
   closest mean absolute activation on the dictionary-train split.
2. **Norm-matched random direction:** a seeded random direction orthogonal to
   the target direction, with the registered magnitude-matching rule.
3. **Supervised probe direction:** a linear probe trained only on the selection
   split using a representation-independent physical readout selected before
   fitting. It is a control, not the definition of the target concept.

The primary contrast uses the maximum absolute control effect, matching the
audited target-versus-control statistic:

```text
D = |Δ_target| - max(|Δ_matched|, |Δ_random|, |Δ_probe|)
```

For compatibility with existing project reports, the signed contrast is also
written as `E_T_signed`; it is not the primary H1 estimand.

The sign-test diagnostic uses the absolute-effect contrast `D` defined in H1,
because the target-versus-control question is whether the target perturbation
exceeds the strongest control regardless of the readout's sign. Signed effects
remain available for dose-response and directionality analyses.

Every row must record the requested mode, executed operator, target index,
control index, random seed, direction digest and norm, perturbation norms, and
no-op flags using the existing intervention provenance schema.

---

## 6. Statistical analysis plan

### Primary test

For each checkpoint and SAE seed:

1. Compute `A[k]` on the selection split for all 16 candidate slots.
2. Compute `D_bar[k]` from the confirmatory batches.
3. Compute Spearman `rho_checkpoint` across the 16 candidate slots.
4. Aggregate the three SAE seeds within checkpoint using Fisher-z averaging.
5. Test the mean checkpoint Fisher-z association with a one-sided sign-preserving
   randomization over independent checkpoints, blocked by PDE family.

The primary confidence interval is a checkpoint-cluster bootstrap. Candidate,
SAE-seed, and batch resampling inside a checkpoint is a sensitivity analysis,
not the primary interval.

### Secondary tests

- H2 repeats the primary analysis with `L[k]` instead of `D_bar[k]`.
- H3 fits a prespecified interaction model with intervention arm and selection
  alignment. The arm interaction is two-sided.
- Basis comparisons report paired checkpoint differences between ReLU+L1,
  TopK, PCA, and random-direction arms, with Holm correction across the two
  secondary hypotheses and registered basis contrasts.
- Raw `D_bar`, signed `E_T_signed`, target effects, every control effect,
  ESF80, reconstruction error, held-out alignment, and signed dose curves are
  reported separately.

### Invalid and missing rows

- A no-op control is not converted to zero and not silently dropped; it is
  recorded and excluded from that row's inferential contrast with a reason.
- A checkpoint with fewer than 8 non-tied candidate contrasts is flagged as
  association-invalid and is not replaced.
- If an entire checkpoint fails a machinery or data-integrity gate, the arm is
  void for that checkpoint. The failure is part of the result.
- No post-hoc candidate, control, dose, grid, or checkpoint substitution is
  permitted.

---

## 7. Machinery gates and preregistered artifacts

Gates run before any confirmatory verdict is read:

1. Penultimate extraction exactly reproduces the linear output readout.
2. Standardization and raw-coordinate conversion round-trip within the fixed
   numerical tolerance.
3. The planted-feature positive control passes through the direct operator and
   the decoder-mediated operator using the registered control families.
4. ESF80 concentrated/uniform sanity checks pass.
5. A shared evaluation-point test confirms that target and every control use
   identical physical points and baseline state.
6. All rows contain intervention provenance and no-op status.

If a gate fails, the affected arm is void. A code fix must be committed and
documented before rerunning the gate; no confirmatory verdict may be read from
the failed run.

Expected future artifacts:

- `runs/alignment_specificity_preregistered/pilot_report.json`
- `runs/alignment_specificity_preregistered/power_plan.json`
- `runs/alignment_specificity_preregistered/confirmatory_report.json`
- `runs/alignment_specificity_preregistered/manifest.json`
- `runs/alignment_specificity_preregistered/raw_rows.jsonl`

The report must distinguish `pilot`, `calibration`, `confirmatory`,
`historical`, and `exploratory` rows. A JSON report without the raw-row hash,
manifest, code revision, configuration, and executed checks is not considered
reproducible evidence.

---

## 8. Interpretation and claim ladder

### If H1 is supported

The strongest permitted claim is:

> Under the registered matched PINN setup, selection-split physical alignment
> predicts target-specific causal specificity for the registered intervention
> and control family.

This does not establish that the selected features are physical mechanisms or
that alignment is sufficient in other architectures, layers, or PDE regimes.

### If H1 is not supported

The permitted claim is:

> Under this protocol, selection-split physical alignment did not predict the
> registered target-versus-control causal contrast at the preregistered
> effect threshold.

This does not refute alignment, reconstruction, spatial localization, or the
existence of causal features outside the tested regime.

### For H2 and H3

H2 can establish a localization relationship without establishing causal
specificity. H3 can establish that intervention semantics matter without
identifying which operator is scientifically preferable. These evidence levels
must remain separate in the paper and in generated tables.

---

## 9. Implementation order before any confirmatory run

1. Add deterministic three-way activation splitting and manifest hashes.
2. Add the standardization-aware direct penultimate operator.
3. Add the decoder-mediated penultimate arm with explicit reconstruction-only
   baseline accounting.
4. Add the checkpoint-clustered primary analysis and the pilot power planner.
5. Add tests for split leakage, shared evaluation points, control semantics,
   coordinate conversion, no-op handling, and checkpoint-level resampling.
6. Run only machinery smoke tests and the preregistered pilot.
7. Freeze `power_plan.json` before reading or scoring confirmatory outcomes.
8. Train and score the confirmatory checkpoint cohort.

No new large training campaign begins before steps 1–5 are implemented,
reviewed, and checked. Historical artifacts remain untouched throughout.

---

## 10. Registration decision

This document is the protocol boundary for the next phase. The project will
not use a future positive or negative result to select a more favorable metric,
control family, PDE subset, candidate set, or intervention definition. Any
change to the primary estimand, independent unit, split structure, dose, or
control semantics requires a new dated preregistration and a clearly labelled
protocol deviation.
