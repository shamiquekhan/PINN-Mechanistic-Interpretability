# REVIEW RESPONSE — External Code Review of v3.2 (29 issues)

**Status:** fully executed 2026-09-08 (campaign commit `b3d2940`, v3.4.0)
**The full issue-by-issue disposition record, including self-verification
notes and the drift table:**
[`docs/external_review_response.md`](docs/external_review_response.md)
*(append-only — every subsequent review response extends that file).*
**The original review snapshot:** [`GUIDE.md`](GUIDE.md)

This page is the reviewer-facing summary: what was reviewed, what changed,
which numbers moved, and where each verification lives.

---

## Why this document exists

An external review of the v3.2 codebase found 29 issues: 4 critical, 5
high, 7 medium, plus process/licensing notes. The repo's response followed
one rule throughout — **verify in source, patch the machinery, add a
regression test, re-run the affected stage, record the drift** — so the
review itself became evidence for the paper's methodology claims. The
reproducibility statement of the manuscript cites this campaign.

## What the reviewer concluded, and what survived

The reviewer's own verdict: *"unusually honest research repo … none
[of the defects] overturn the headline results, but three of them can
silently mislead a reader or break CI."* After the campaign: all three
critical issues are fixed with re-runs; every headline verdict **survives
in kind** (the PINN causal null holds under harder controls), and three
numbers moved materially — all recorded in the
[errata](#errata-numbers-that-moved-in-v34) below.

## The three critical fixes

| # | Issue | Fix | Re-run consequence |
|---|---|---|---|
| C1 | Reproduce-script gate hard-coded a stale AUROC (0.872 vs artifact 0.859) — CI broke on its own data | Expectations are now **generated from the artifacts** (`scripts/generate_expected_headlines.py` → `runs/expected_headlines.json`); CI-range checks | The gate cannot silently contradict the artifacts again |
| C2 | Burgers/Allen-Cahn `exact()` returned the reference **only at final time** — stage-10 rel L2 (6.1/6.0) measured distance-from-zero, not error | Full space-time spectral trajectories + bilinear interpolation (per-PDE pad policy) | Time-dependent rel L2: **6.1 → 0.111, 6.0 → 0.550** — the PINNs were training fine all along |
| C3 | Causal controls: `unrelated_control` was ~97% likely to no-op on the dead-latent dictionary; `random_direction` **injected** noise while the target **ablated** (unmatched in kind) | **Matched-deletion controls** in all three batteries (SAE, PCA, operator) + a `control_was_noop` diagnostic (0/88 no-ops post-fix) | PINN nulls **hold, harder**: 0/8 survivors on both bases, E_T now strictly negative (the representational signature); stage-14 survivors 6/8 → **2/8** |

## High/medium items (one-line each)

- **H1** stage-6 gradient monitor arm was a no-op stub → implemented for real (joins `gradients.jsonl`); monitor AUROC 0.859 → 0.875
- **H2** leakage audit was ceremonial (`failure_step=10**9` can never fail) → the real feature-window invariant, 12/12 held-out runs pass
- **H3** controller demo hand-rolled a second training loop that violated its own `resample_every: 0` → demo runs through `PINNTrainer` seams; config-faithful controller reaches oracle level
- **H4** `trigger_resample` wired through the trainer seam; `trigger_fourier_features` documented signal-only (implementation = registered follow-up)
- **H5** explicit `speed`/`reaction_rate` config fields with loud ambiguity errors
- **M1–M5, M7** SAE bookkeeping (`post_step()`, format_version=2 checkpoints, save-time invariant assertion), safe `weights_only=True` loaders (3 legacy sites flagged as follow-up), no silent cuda→cpu downgrade, logging hygiene, `steps=0` semantics, stiff-RD overflow guard (float32 `exp` overflows near 88, not 350)

## Declined

- **M6 (re-license to MIT for code):** the repository owner elected to retain
  **CC BY 4.0** for the repository, including the code. The reviewer's
  OSI-license rationale is recorded in the disposition file so the next
  reviewer sees the decision was deliberate, not missed.

## Self-verification (recorded, not hidden)

Three first-pass fixes by the responder had bugs that their **own regression
tests caught before any verdict was read**: the H1 row-alignment bug, the
C2 grid-map error (endpoint=False compression + wrong pad policy for the
aperiodic Allen-Cahn IC), and the H2 audit-semantics confusion (the shipped
`leakage_audit` rule cannot pass on honest data of this shape — which is
why the original author passed `10**9`). Details in
`docs/external_review_response.md` §4.

## Verification receipts (v3.4.0)

- **171 tests** pass (16 new regression tests pinning the review fixes)
- `scripts/check_results_grounded.py` exit 0 — every registered claim is
  grounded in a committed artifact; the stale-marker guard now also enforces
  that v3.4 moved numbers appear only inside marked historical contexts
- Checksums 18/18 (`data/checksums.sha256`, manifest v3.4.0)
- Figures 9/9 and tables 7/7 regenerate from the post-fix artifacts
- The paper builds clean (13 pages) with the corrected numbers

---

## Errata: numbers that moved in v3.4

If you read an earlier version of the paper, README, or RESULTS, these are
the corrections. Full context in the linked sections.

| Quantity | v3.2/v3.3 value | v3.4 value | Why | Where |
|---|---|---|---|---|
| Burgers 1D rel L2 (stage 10) | 6.1 | **0.111** | old metric was distance-from-zero (C2) | RESULTS §5A.3 |
| Allen-Cahn 1D rel L2 (stage 10) | 6.0 | **0.550** | same (C2) | RESULTS §5A.3 |
| Operator-battery survivors (stage 14) | 6/8 | **2/8** | matched-deletion controls (C3); the suggestive asymmetry is re-reported at this weaker strength | RESULTS §5A.7 |
| Operator beats-all | 48/64 (75%) | **42/64 (66%)** | same (C3) | RESULTS §5A.7 |
| SAE E_T (stage 5) | +0.0020 [−0.0034, +0.0084] | **−0.0173 [−0.0329, −0.0037]** | honest controls flip the sign to the representational signature; **null holds** | RESULTS §4.1 |
| PCA E_T (stage 8) | −0.0203 | **−0.0131 [−0.0268, −0.0006]** | same (C3); paired PCA−SAE diff now spans zero | RESULTS §5A.1 |
| Positive-control margin | 68× random control | **9.2×** | matched deletion is a harder control — the narrowing is the fix working | RESULTS §4.4 |
| Monitor AUROC (conventional) | 0.859 | **0.875 [0.814, 0.960]** | gradient arm now real (H1) | RESULTS §6 |
| Monitor AUROC (SAE+conv) | 0.864 | **0.877** | same | RESULTS §6 |
| Controller final (stage 7) | 0.0166 | **0.00016** | config-faithful protocol (H3); old loop resampled collocation every step against the config | RESULTS §7 |
| No-action final | 0.2953 | **0.4208** | same (H3) | RESULTS §7 |
| Rescue factor | 18× | **oracle-level** (0.0002 ≈ 0.00023) | same (H3) | RESULTS §7 |
| SOTA comparison | "NTK-adaptive beats controller" | **controller beats NTK-adaptive on the config-faithful protocol**; both readings recorded | the old reading was measured under the config-violating loop | RESULTS §5A.5 |
| P(SAE monitor better) | 0.53 | **0.51** | re-run | RESULTS §5A.6 |

**Nothing on this page patches a verdict.** Every row above is a machinery
fix followed by a re-run; the drift table in
`docs/external_review_response.md` §2 is the authoritative record.
