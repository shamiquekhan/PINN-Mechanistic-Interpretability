# Reproducibility Checklist

Fresh-environment verification protocol. Check items in order; every item
maps to a command and an expected outcome (details in
[reproducibility.md](reproducibility.md)).

## Environment

- [ ] Fresh venv/conda environment created (Python ≥ 3.10)
- [ ] `pip install -r requirements.lock` succeeds (or Docker image builds)
- [ ] `python -c "import torch; print(torch.__version__, torch.cuda.is_available())"` reports the expected backend
- [ ] GPU path only: `CUBLAS_WORKSPACE_CONFIG=:4096:8` exported (silences cuBLAS determinism warnings)

## Unit tests

- [ ] `python -m pytest tests/ -q` → **171 passed** (~15 s, CPU or GPU)
- [ ] `python -m compileall -q analysis controller experiments interventions monitoring operators pinn pinn_logging sae tests` → no output (clean compile)

## Smoke tests

- [ ] `bash scripts/run_smoke_test.sh` completes (< 5 min CPU)
  - short boundary-starvation training run logs activations
  - effective-rank analysis returns nonzero participation ratio

## Campaign (stages 2–15 reproduce from committed `runs/` stage-1 artifacts; stage 1 retrains from scratch)

- [ ] `--stages 2`: failure atlas — boundary_starvation labels 10/10
- [ ] `--stages 3`: SAEs — 3 TopK replicas trained; PCA/random baselines logged
- [ ] `--stages 4`: feature dictionary — random-direction control beats real features only via real>random gate
- [ ] `--stages 5`: causal battery — E_T CI spans zero; 0/8 survive Bonferroni and BH-FDR; **positive control `pipeline_pass=true`**
- [ ] `--stages 6`: monitors — conventional AUROC ≈ 0.859 [0.780, 0.952]
- [ ] `--stages 7`: controller — final ≈ 0.0166 vs no-action 0.2953; SAE arm bit-identical to random arm
- [ ] `--stages 8`: PCA causal battery — 0/8 survive; sign diagnostic 88/88 positive
- [ ] `--stages 9`: causal abstraction — `beats_random_every_run=false` for PCA and SAE
- [ ] `--stages 10`: dimensional boundary — tangent rank = input dim in all families
- [ ] `--stages 11`: operator boundary — `sae_beats_k_matched_pca=true`, PR ≈ 6.8
- [ ] `--stages 12`: SOTA — NTK-adaptive ≈ 0.0064 < controller 0.0166
- [ ] `--stages 13`: hardening — MDE 85.7%, P(SAE monitor better) ≈ 0.53
- [ ] `--stages 14`: operator causal battery — machinery gate passes; 6/8 Bonferroni survivors; preregistered rule fires H14b (see RESULTS.md §5A.7)
- [ ] `--stages 15`: NTK bridge — machinery gate passes; `decision_rule.recorded=H15b`; 0/8 Bonferroni survivors; 9/11 runs skipped by the label-balance guard (chronic conflict)

## Figures and tables

- [ ] `bash scripts/generate_figures.sh` → 9 PNGs in `figures/generated/` without error
- [ ] `bash scripts/generate_tables.sh` → LaTeX tables in `paper/tables/` without error
- [ ] Spot-check: Figure 4 (causal null) E_T value matches `runs/causal_intervention_results.json` `causal_strength_ci.mean`

## Cross-checks (traceability)

- [ ] Pick any claim in RESULTS.md §5A; find its experiment row in [experiment_matrix.md](experiment_matrix.md); re-run that row's command; artifact field matches the claimed number within the documented nondeterminism tolerance (§3 of reproducibility.md).
- [ ] Verify no artifact under `runs/` was hand-edited: regenerate its stage and diff.

## Release hygiene

- [ ] `git status` clean; `CHANGELOG.md` updated; `CITATION.cff` version/date/commit current
- [ ] Docker image builds from the committed Dockerfile
- [ ] Zenodo/archival DOI added to CITATION.cff (when a release is tagged)
