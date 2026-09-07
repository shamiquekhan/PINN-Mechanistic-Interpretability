# `paper/` — Submission Artifact

Full LaTeX manuscript for the NeurIPS/ICML-target paper (v3.2.0: complete
prose, 9 figures, 7 artifact-generated tables, 12-page compiled PDF).
The narrative outline lives in [`docs/paper_draft.md`](../docs/paper_draft.md);
this directory is the compiled-submission artifact.

## Layout

| File | Purpose |
|---|---|
| `main.tex` | Full manuscript (all sections in prose; tables `\input`) |
| `references.bib` | Citation backbone (verify bibliographic details before submission) |
| `main.pdf` | Reference build (12 pages) |
| `tables/*.tex` | **Generated** by `scripts/generate_tables.py` from `runs/` JSONs — do not hand-edit |
| `figures/` | Copies of `figures/generated/*.png` (refreshed by `scripts/generate_figures.sh`) |
| `supplementary/` | Appendices (proofs from `docs/theory_activation_rank.md`, full protocol text) |
| `arxiv_package/` | Self-contained submission bundle (README = submission checklist) |

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
