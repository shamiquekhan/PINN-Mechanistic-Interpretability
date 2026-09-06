# Contributing

Thank you for considering contributions to this research framework.

## Project context

This is a **scientific research repository**: the code exists to produce
and validate the claims in [RESULTS.md](RESULTS.md). Two kinds of change
have different standards:

1. **Research-affecting changes** (anything that alters a number in
   `runs/`): experiment code, PDE definitions, scoring/statistics,
   protocol changes.
2. **Infrastructure changes** (docs, CI, packaging, figures, scripts).

## Ground rules

* **Reproducibility is load-bearing.** A research-affecting PR must state
  which stages it touches and re-run them; artifacts must be regenerated
  by the pipeline, never hand-edited.
* **Preregistration discipline.** If you add a new experiment that tests a
  hypothesis, write the hypothesis + decision rule down *before* running it
  (extend `docs/preregistration.md` in the same PR, and record the outcome
  after).
* **No hand-entered numbers** in figures/tables/paper: they must read from
  `runs/` artifacts (see `figures/_common.py` and
  `scripts/generate_tables.py` for the pattern).
* **Tests.** New modules need unit tests; the suite must pass:
  `python -m pytest tests/ -q` (139 tests at time of writing). CI runs the
  CPU path on Python 3.10/3.12.
* **Claim ladder.** Result summaries must respect the claim-strength
  language in `RESULTS.md` §8 (e.g., "supported with scope",
  "rejected (hardened negative)") — do not strengthen a claim beyond its
  evidence.

## Workflow

1. Fork, create a feature branch.
2. Make changes; add/adjust tests; update affected docs
   (RESULTS.md, experiment_matrix.md, CHANGELOG.md).
3. Run locally:
   ```bash
   python -m pytest tests/ -q
   python -m compileall -q analysis controller experiments interventions monitoring operators pinn pinn_logging sae tests
   bash scripts/run_smoke_test.sh        # if training-related code changed
   bash scripts/generate_figures.sh      # if figures/artifacts changed
   bash scripts/generate_tables.sh
   ```
4. Open a PR describing: what changed, which stages are affected, what
   numbers moved (before/after), and any preregistration updates.

## Style

* Type hints in new modules (`from __future__ import annotations`).
* Docstrings explaining *why* (protocol rationale), not just *what*.
* Comments only where the code cannot speak (protocol invariants,
  corrections of earlier designs).

## Reporting problems with the science

If you believe a result in RESULTS.md is wrong, open an issue titled
`Result challenge: <claim>` describing the artifact, the expected value,
and the reproduction attempt. These are treated as high priority — negative
results live or die by their rigor.
