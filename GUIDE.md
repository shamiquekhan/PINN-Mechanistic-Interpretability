# External Code Review & Fix Guide (v3.2)

**Repo:** `shamiquekhan/PINN-Mechanistic-Interpretability` (v3.2)
**Scope of review:** `pinn/`, `sae/`, `interventions/`, `monitoring/`, `controller/`, `experiments/run_pipeline.py`, `scripts/reproduce_main_results.sh`, `RESULTS.md`, `README.md` (read in full); other modules reviewed at API-surface level.
**Verdict up front:** unusually rigorous research engineering (preregistration, positive controls, MC corrections, honest negatives) with a handful of real correctness/documentation defects — none overturn the headline results, but three of them can silently mislead a reader or break CI.

> **Execution log for this guide lives in `docs/external_review_response.md`**
> (per-issue status: verified / fixed / re-run / drifted-number tables).
> Historical snapshot of the review; superseded numbers may appear below.

---

## 0. Issue summary

| # | Severity | Issue | File |
|---|----------|-------|------|
| C1 | **Critical** | Stale hard-coded headline expectation (AUROC 0.872) contradicts fresh artifact (0.859) → reproduce script self-check fails | `scripts/reproduce_main_results.sh` |
| C2 | **Critical** | `Burgers1D.exact` / `AllenCahn1D.exact` return **0 for all t < tT** while `validation_grid` spans the full (t,x) domain → stage-10 rel L2 (6.1 / 6.0) measures distance-from-zero, not solution error | `pinn/pdes.py` |
| C3 | **High** | `unrelated_control` on a TopK SAE (L0 = 8/256) zeroes a feature that is ~97% likely to be already inactive → control is often a no-op; `random_direction` *adds* noise instead of *removing* a matched quantity → asymmetric controls | `interventions/engine.py` |
| C4 | **High** | Documentation drift: stage count (13/14/15), missing ACTING state, "7 intervention modes" vs 8, stale "future work" sentence in RESULTS §5A.4 | README / `engine.py` / `RESULTS.md` |
| H1 | High | Dead code in stage 6: `_build_dataset` defined twice; `_augment_with_gradients` is a no-op stub with a misleading docstring | `experiments/run_pipeline.py` |
| H2 | High | Vacuous `leakage_audit` call in stage 6 (`failure_step=10**9`) — always passes, gives false assurance | `experiments/run_pipeline.py` |
| H3 | Medium | `PINNTrainer.intervention_fn` runs *after* loss computation; controller-λ integration path unclear; stage 7 re-implements the loop instead of reusing the trainer | `pinn/trainer.py` |
| H4 | Medium | Controller actions `trigger_resample` / `trigger_fourier_features` change only the event string — no mechanism wires them to the trainer | `controller/state_machine.py` |
| H5 | Medium | `make_pde` overloads `cfg.source` as advection speed / reaction rate / amplitude | `pinn/pdes.py` |
| M1 | Medium | SAE bookkeeping: `_normalise_decoder` never called by the model, `dead_feature_mask(window)` ignores `window`, `update_feature_use` is caller-dependent | `sae/model.py` |
| M2 | Medium | `torch.load(..., weights_only=False)` in three places — unsafe for untrusted checkpoints, and pinned to legacy behavior | `pinn/trainer.py`, `sae/model.py`, `experiments/run_pipeline.py` |
| M3 | Low | Silent device downgrade; unchecked `getattr(torch, dtype)` | `pinn/trainer.py` |
| M4 | Low | Checkpoint saved at step 0; `print(rec)` per log step | `pinn/trainer.py` |
| M5 | Low | `steps = steps or tcfg.steps` falsy-zero bug; AllenCahn docstring contains leftover drafting text | `pinn/trainer.py`, `pinn/pdes.py` |
| M6 | Low | License CC BY 4.0 for code (no patent grant, not OSI-recommended for software) | `LICENSE` |
| M7 | Low | `ReactionDiffusion1D.exact` overflow/degenerate guards (numpy `exp` overflow, `det` threshold) | `pinn/pdes.py` |

---

## C1 — Reproduce script checks a number that no longer exists (CI-breaking)

`RESULTS.md` (fresh campaign, 2026-09-07) reports the conventional monitor at **AUROC 0.859 [0.780, 0.952]** and SAE-augmented at **0.864**. But the reproduce script hard-codes `abs(auroc - 0.872) < 0.01`, so `|0.859 − 0.872| = 0.013 > 0.01` → the script fails its own headline gate on the committed artifacts.

**Fix:** derive expectations from a single source of truth instead of hard-coding. Point expectations should become ranges tied to recorded CIs, generated into `runs/expected_headlines.json` alongside the artifacts, so the expectation can never drift from the artifact again. Replace point expectations with ranges tied to CIs wherever possible.

## C2 — Burgers / Allen-Cahn validation metric is measuring the wrong thing

Both time-dependent PDEs compute a *final-time-only* spectral reference, then zero-mask everything with `t < tT` — while `validation_grid` spans the full (t,x) product grid and the trainer computes rel L2 over all points. The reported 6.1 / 6.0 numbers are dominated by distance-from-zero at non-final times, not network error; these values feed `derive_failure_step`.

**Fix (preferred):** store the full space-time spectral trajectory and bilinearly interpolate `exact()` in (t, x); add a test asserting `exact()` is nonzero at intermediate t on a mid-domain x.

## C3 — Asymmetric / degenerate causal controls for TopK SAEs

(a) `unrelated_control` usually zeroes an already-inactive latent → literal no-op. (b) `random_direction` *adds* noise while the target *ablates* → arms not matched in kind.

**Fix:** sample the unrelated control from *active* features only (retry loop); make `random_direction` a *matched deletion* (ablate a random active non-target feature). Emit a per-evaluation `control_was_noop` diagnostic. **Re-run stage 5** — control distributions change, so E_T / specificity / survivors must be regenerated, not patched. The null will very likely survive; record drift per the project's own discipline.

## C4 — Documentation drift

- README controller diagram shows an `ACTING` state that does not exist in `ControllerState` (action fires on CONFIRMED entry).
- Stage-count references must regenerate from `main()`'s want-list (15 stages).
- `interventions/engine.py` docstring says "7 intervention modes"; `_validate_mode` accepts 8.
- RESULTS §5A.4 still says the operator causal battery "is future work" one section before §5A.7 reports it.
- Add a CI grep guard so "stages 1–13" / "7 intervention modes" cannot silently reappear.

## H1 — Stage 6 dead code

`_build_dataset` defined twice (first shadowed); `_augment_with_gradients` is a no-op stub yet the monitor is described as using gradients. Implement the gradient-cosine join (the data is already logged in `gradients.jsonl`) and re-run stage 6, or rename the arm honestly.

## H2 — Vacuous leakage audit

`leakage_audit(..., failure_step=10**9)` can never fail. Run the audit per held-out run with that run's real derived `failure_step` and assert all pass; add a test proving the audit fails on a deliberately leaked window.

## H3 — Trainer intervention hook placement & controller integration

`intervention_fn` fires after loss computation (λ updates land a step late; weight mutation would use a stale loss); stage 7 hand-rolls its own loop. Give the trainer a `get_lambdas(step)` seam; move weight-affecting interventions before the forward pass; route the controller demo through the trainer so the published numbers exercise the documented integration.

## H4 — Controller actions that do nothing

`trigger_resample` / `trigger_fourier_features` return unchanged lambdas and nothing consumes the event — indistinguishable from `no_action`. Wire them through the intervention seam (resample collocation on trigger; document Fourier as a warm-restart) or raise `NotImplementedError` — honest scaffolding beats silent no-ops.

## H5 — `make_pde` field overloading

`cfg.source` doubles as Poisson source / advection speed / reaction rate / amplitude. Add explicit optional fields (`speed`, `reaction_rate`, `forcing`) and fail loudly on ambiguity.

## M1 — SAE bookkeeping

Call `_normalise_decoder` from a `post_step()` the trainer must invoke (assert-at-save when the flag is set); implement the rolling window in `dead_feature_mask` or drop the parameter; accumulate `update_feature_use` in `forward`; store `format_version` in new checkpoints and refuse to guess legacy mode below v2.

## M2 — Unsafe checkpoint loading

Centralize `torch.load` with `weights_only=True` plus a safe-globals whitelist (one loader, one audit point).

## M3 — Silent device downgrade; unchecked dtype

Refuse to silently downgrade `cuda`→`cpu` (raise); validate `dtype` with a clear error.

## M4 — Noise & I/O

Skip the step-0 checkpoint (`step > 0 and ...`); route per-step prints through `logging` with a `--quiet` path.

## M5 — Small correctness nits

`steps = steps or tcfg.steps` (falsy-zero bug) → `steps if steps is not None else tcfg.steps`; clean the AllenCahn drafting text and reconcile the periodicity/Dirichlet note; drop or use `build_trajectory_features`' unused `failure_label` arg.

## M6 — License

CC BY 4.0 is a content license. Re-license the **code** under MIT (keep CC BY 4.0 for paper/figures/docs); update badges + CITATION.cff.

## M7 — RD1D numerics

Guard `math.exp` overflow for stiff configs (`r·max|x| > 350` → return None → caller falls back); add a stiff-config test against a fine FD solve.

---

## Verification checklist

1. `pytest tests/` — all tests pass; new tests for C2 (exact nonzero at mid t), C3 (noop diagnostic), H2 (audit fails on leaked window).
2. `bash scripts/run_smoke_test.sh` — CPU sanity.
3. `bash scripts/reproduce_main_results.sh` — all headline checks pass against committed artifacts.
4. Re-run stages 5/6 (+8/10/13/14 where the fixed code is used) and confirm qualitative verdicts unchanged; add drift-table entries per the project's own discipline.
5. Grep sweep for stale markers.

## What to keep (do not "fix" these)

- Preregistration + positive-control + MC-correction discipline — the strongest part of the repo.
- The honest-negative framing (controller not SOTA; SAE features inert cargo; H14b recorded with the design lesson) — verbatim.
- The `check_results_grounded.py` CI gate and checksum manifest — extend expectations there.
- Run-level splits and run-level bootstrap CIs in stage 6 — methodologically correct.
