# External-Review Fix Campaign Record (v3.4)

**Date:** 2026-09-08
**Input:** the external code review (29 issues; `GUIDE.md` is the review
snapshot; this file is the execution record).
**Discipline:** every C1–C4/H1–H5/M1–M7 issue was verified in source before
patching; every fixed bug carries a regression test; every stage whose
machinery changed was **re-run** and the number drift recorded here — no
verdict was patched, only regenerated.

---

## 1. Issue dispositions

| # | Disposition | Notes |
|---|---|---|
| C1 | **FIXED** | Reproduce-script expectations are now artifact-derived ranges (`scripts/generate_expected_headlines.py` → `runs/expected_headlines.json`); the stale hard-coded 0.872 check replaced with within-own-CI + margin checks |
| C2 | **FIXED + RE-RUN (stage 10)** | Full space-time spectral trajectories with bilinear interpolation; periodic-pad (Burgers, continuous IC) vs edge-pad (Allen-Cahn, aperiodic x³ IC) documented per PDE; stage-10 rel L2 was 6.1/6.0 (distance-from-zero artifacts) → **0.111/0.550** (genuine space-time error) |
| C3 | **FIXED + RE-RUNS (stages 5/8/14)** | `unrelated_control` samples only ACTIVE features (no-op impossible; `control_was_noop` diagnostic emitted, 0/88 no-ops); `random_direction` is now matched **deletion** (was noise injection); the same fix applied to the PCA and operator batteries |
| C4 | **FIXED** | ACTING state removed from README (action fires on CONFIRMED entry — honest); stage lists regenerated to 15; "7 intervention modes" → 8; stale §5A.4 "future work" sentence forward-links §5A.7; CI stale-marker guard added to `check_results_grounded.py` |
| H1 | **FIXED + RE-RUN (stage 6)** | Shadowed `_build_dataset` removed; `_augment_with_gradients` now really joins `gradients.jsonl` (mean/std/last window cosines), row-aligned to the dataset builder (the first version of this fix had a row-alignment bug caught by its own regression test) |
| H2 | **FIXED** | The ceremonial `failure_step=10**9` audit replaced with the real invariant: no feature window may contain post-failure records. Runs failing from initialization (fs ≤ first window) are exempted — no pre-failure segment exists to leak. `leakage_audit`'s example-level step rule was mis-designed for this shape; the invariant is checked directly in stage 6 (12/12 held-out runs pass) |
| H3 | **FIXED + RE-RUN (stage 7)** | Trainer gained `get_lambdas` / `resample_fn` / `post_step_fn` seams; weight-affecting `intervention_fn` moved before the forward pass; stage 7 now runs through `PINNTrainer` (one loop, not two). The controller observes the step's real loss/rel L2 and its decision applies to the next step |
| H4 | **WIRED (resample) / DOCUMENTED (fourier)** | `trigger_resample` is consumed through the trainer's `resample_fn` seam; `trigger_fourier_features` is documented as signal-only (a correct implementation requires an input-dim-changing warm restart — registered as future work, not silently no-oped) |
| H5 | **FIXED** | `PDEConfig` gains explicit `speed` / `reaction_rate`; conflicting alias + legacy `source` fails loudly |
| H6–H7 | n/a (not in this review) | — |
| M1 | **FIXED** | `post_step()` normalisation entry point; `dead_feature_mask` window semantics implemented; `format_version=2` in new checkpoints; unit-norm assertion at save when `decoder_normalize` is set |
| M2 | **FIXED** | `weights_only=True` in the trainer/SAE loaders; remaining experiment-script loads centralised (see `experiments/*.py` — three legacy `weights_only=False` sites remain for legacy checkpoint manifests and are flagged in the review-response doc as the migration follow-up) |
| M3 | **FIXED** | Silent cuda→cpu downgrade now raises; dtype validated with a clear error |
| M4 | **FIXED** | No step-0 checkpoint; per-step prints → `logging` |
| M5 | **FIXED** | `steps=0` means zero steps (was config-default); AllenCahn docstring cleaned (drafting text removed, periodic-solver-vs-Dirichlet note reconciled); `build_trajectory_features`' unused `failure_label` made optional |
| M6 | **DECLINED** | Repository owner elected to retain CC BY 4.0 for the whole repository (code included). The reviewer's OSI-license recommendation is recorded here for provenance |
| M7 | **FIXED** | Stiff-RD overflow guarded (float32 exp overflows near 88, not 350 — the guard is `r·max|x| > 80`) and the final array evaluated in float64; stiff configs return `None` (caller falls back) |

## 2. Stage re-runs and verdict drift

Machinery changed ⇒ numbers regenerated. Qualitative verdicts that survive:

| Stage | Verdict | Before → After |
|---|---|---|
| 5 (SAE battery) | **Null holds, harder controls** | E_T +0.0020 [−0.0034, +0.0084] → **−0.0173 [−0.0329, −0.0037]**; survivors 0/8 → **0/8**; sign 88/88 positive unchanged; beats-all 12/88 → **9/88**; positive control 68× → **9.2×** (matched deletion is a harder control — margin narrowing is expected and honest) |
| 8 (PCA battery) | **Null holds** | E_T −0.0203 → **−0.0131 [−0.0268, −0.0006]**; survivors 0/8 → **0/8**; paired PCA−SAE diff −0.0223 [excluded 0] → **+0.0042 [−0.0139, +0.0232]** (spans zero — the two bases are statistically indistinguishable under matched deletion) |
| 6 (monitors) | **No SAE advantage** | Conventional 0.859 → **0.875 [0.814, 0.960]**; SAE+conv 0.864 → **0.877 [0.817, 0.960]** — the gradient features are now REAL (the stub contributed nothing before); CIs overlap; P(SAE better) = **0.51** |
| 7 (controller) | **Rescue holds; now config-faithful** | Controller 0.0166 → **0.00016**; no-action 0.2953 → **0.4208**; oracle 0.0119 → **0.00023**. Root cause of the shift: the old hand-rolled loop resampled collocation EVERY step, violating the config's `resample_every: 0`; the trainer obeys the config. SAE≡random bit-identical (unchanged) |
| 12 (SOTA) | **Controller now BEATS NTK-adaptive** | NTK-adaptive 0.0064 (unchanged) now loses to the controller (0.0002) on the config-faithful protocol. The v3.3 claim "NTK beats the controller" was an artifact of the non-config-faithful loop; both statements are recorded, current protocol is authoritative |
| 14 (operator causal) | **H14b stands; suggestive asymmetry WEAKENS** | Survivors 6/8 → **2/8** (two features at 8/8 batch consistency still survive); beats-all 75% → **66%** (42/64); E_T CI [7.1e-6, 1.6e-5] → **[5.6e-6, 1.6e-5]**; interchange still does not beat random. The "suggestive asymmetry" is retained but explicitly weaker; machinery gate PASS (targeted 1.6× random now, was 1.8×) |
| 10 (dimensional boundary) | **Geometry unchanged; validation fixed** | Tangent rank 2 everywhere (unchanged); PR bands unchanged (1.42–2.47); rel L2 Burgers 6.1 → **0.111**, Allen-Cahn 6.0 → **0.550** — the old values measured distance-from-zero, not error |
| 13 (hardening) | **Refreshed on new batteries** | MDE 85.7% @ 80% power (unchanged); P(SAE better) = **0.51** |
| 15 (NTK bridge) | **H15b unchanged** | No machinery touched (reads gradients.jsonl only); re-run confirmed |

## 3. The three headline interpretation changes

1. **The PINN causal null is now measured against honest controls.** The
   E_T sign flip (−0.0173) is the matched-deletion signature: ablating the
   target feature moves the loss LESS than ablating another active latent —
   the representational (not causal) encoding conclusion is *strengthened*.
2. **The controller's stage-7 numbers were never config-faithful.** The
   hand-rolled loop resampled collocation every step; the trainer obeys
   `resample_every: 0`. The corrected protocol *strengthens* the controller
   (0.0002 vs NTK's 0.0064) and inflates the rescue factor (2605×) — the
   "controller not optimization SOTA" positioning is revised: on the
   config-faithful protocol it beats all three SOTA baselines on this
   failure. Both the old and new claims remain in the record with their
   protocols.
3. **The operator suggestive asymmetry is weaker than v3.3 recorded.**
   Under matched-deletion controls only 2/8 features survive (was 6/8).
   The regime boundary's reconstruction side is untouched; its causal side
   is now "2/8 survivors vs 0/8 everywhere on PINNs" — still an asymmetry,
   further from a confirmed boundary.

## 4. Self-verification notes (recorded, not hidden)

- The first version of the H1 fix had a row-alignment bug (emitted
  `n−history_window` rows vs the builder's `n−history_window−horizon`)
  that would have silently zeroed the new gradient features — caught by its
  own regression test before any stage re-run.
- The first version of the C2 fix used an `(n_x−1)` space map that is
  wrong for endpoint=False spectral grids (up to 0.09 IC error near
  boundaries) and a periodic wraparound pad that is wrong for AllenCahn's
  aperiodic x³ IC — caught by the IC-match regression test; corrected to
  the padded cell map with per-PDE pad policy.
- `leakage_audit`'s shipped step rule cannot pass on honest data of this
  shape (post-failure rows legitimately carry y=0 because their future no
  longer contains the failure) — which is *why* the original author passed
  `10**9`. The H2 fix checks the real feature-window invariant instead of
  salvaging the mis-designed helper.
- M2 note: three `weights_only=False` loads remain in experiment scripts
  that read legacy checkpoint manifests; migrating those checkpoints is
  registered follow-up work, flagged here rather than silently ignored.
