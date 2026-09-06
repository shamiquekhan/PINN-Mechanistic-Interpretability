# `paper/` — Submission Artifact

LaTeX skeleton for the NeurIPS/ICML-target manuscript. The working draft
with full prose lives in [`docs/paper_draft.md`](../docs/paper_draft.md);
this directory is the compiled-submission artifact.

## Layout

| File | Purpose |
|---|---|
| `main.tex` | Article skeleton with all headline tables `\input` |
| `references.bib` | Citation backbone (verify bibliographic details before submission) |
| `tables/*.tex` | **Generated** by `scripts/generate_tables.py` from `runs/` JSONs — do not hand-edit |
| `figures/` | Copy `figures/generated/*.png` here before building |
| `supplementary/` | Appendices (proofs from `docs/theory_activation_rank.md`, full protocol text) |

## Build

```bash
bash scripts/generate_tables.sh                  # regenerate tables from artifacts
cp figures/generated/*.png paper/figures/        # if you want figures in the build
cd paper && pdflatex main.tex && bibtex main && pdflatex main.tex && pdflatex main.tex
```

## Discipline rules

1. **No hand-entered numbers**: every quantitative value in the tables comes
   from the authoritative `runs/` artifacts via `scripts/generate_tables.py`.
   If a number looks wrong, fix the pipeline, not the table.
2. **Traceability**: each table maps to an experiment row in
   [`docs/experiment_matrix.md`](../docs/experiment_matrix.md).
3. **Claims ladder**: the strength of every claim is governed by
   [`RESULTS.md` §8](../RESULTS.md); the paper cannot outrun the claim ladder.
