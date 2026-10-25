# Research audit follow-up

**Implemented:** October 25, 2026

This note records the implementation slice from the research-grade audit and
the follow-up regeneration completed on October 25, 2026. The fresh artifacts
are current local CPU validation outputs; historical review documents retain
their original values where they explicitly describe an earlier campaign.

## Implemented changes

- SAE and PCA measurements now expose three losses on the same evaluation
  points: raw model, reconstruction-only, and targeted intervention.
- Results include the exact projection, incremental intervention, and total
  deltas. The baseline-selected `delta_*` fields remain available for callers.
- Both measurement APIs accept shared `evaluation_points` tensors and explicit
  sampling seeds.
- `random_direction` is now a genuinely random latent/coefficient-space
  perturbation with a per-row norm matched to the target dose.
- `matched_deletion` is the explicit random active non-target ablation control.
- `unrelated_control` / `unrelated_component` use deterministic
  activity-matched non-target controls rather than the same seeded selector as
  the random control.
- SAE/PCA battery rows now record `candidate_slot` and comparisons join by
  `(run, candidate_slot)`. Missing or duplicate keys fail loudly.
- Hook cleanup and model/SAE train/eval state restoration are exception-safe.
- Invalid dtype configuration is reported before an unavailable CUDA device
  masks the error.

## Regeneration status

The following protocol-dependent artifacts were regenerated after the code
changes:

- `runs/causal_intervention_results.json`
- `runs/positive_control.json`
- `runs/pca_causal_results.json`
- `runs/r2_fourier_causal/`
- `runs/r6_dose_response/`
- `runs/equivalence_test/`
- `runs/r7plus_equivalence/`
- `runs/architecture_boundary/`
- `runs/operator_causal/`

The dependent checksums, manifest metadata, expected-headline ledger, and
current validation record were refreshed after those experiments completed.
Historical review documents may retain old values, but living result claims
must identify which protocol produced them. The current artifact ledger is
[`docs/fresh_pipeline_rerun_2026-10-25.md`](fresh_pipeline_rerun_2026-10-25.md).

## Verification in this checkout

The focused intervention, PCA, external-review, dose-response, and regression
tests pass after the patch. The full suite and results-integrity check were
rerun after artifact regeneration; the exact current artifact values and
verification commands are recorded in
[`docs/fresh_pipeline_rerun_2026-10-25.md`](fresh_pipeline_rerun_2026-10-25.md).
