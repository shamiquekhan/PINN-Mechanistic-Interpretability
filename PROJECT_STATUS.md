# Project Status

**Last verified:** 2026-09-07 (post-stage-14; every item below re-verified)
**Verification method:** commands executed end-to-end on this machine; every
"verified" item lists the exact command and the artifact it produced. A
machine check (`scripts/check_results_grounded.py`, wired into CI via
`.github/workflows/results_check.yml`) enforces that the results-bearing
docs cite only artifact-grounded numbers.

---

## Verified and reproducible (run end-to-end, artifacts committed)

Each item: what ran → where the output lives.

| Item | Command | Artifact | Last-run receipt |
|---|---|---|---|
| Full 13-stage campaign, fresh training | `python -m experiments.run_pipeline` (stages 1–13) | `runs/*.json`, `runs/<config>/` | 2026-09-07 fresh rerun; 18/19 headline checks reproduced (drift table: `docs/fresh_campaign_record.md`) |
| Failure atlas (10 seeds × 5 regimes) | `--stages 2` | `runs/failure_atlas/seed_matrix_stats.json` | boundary 10/10, spectral 10/10 labels |
| SAE causal battery + positive control | `--stages 5` | `runs/causal_intervention_results.json`, `runs/positive_control.json` | E_T = +0.0020 [−0.0034, +0.0084]; 0/8 Bonferroni & BH; control PASS (cosine 0.989) |
| PCA causal battery | `--stages 8` | `runs/pca_causal_results.json` | E_T = −0.0203 [−0.0331, −0.0085]; 0/8 |
| Causal abstraction (interchange) | `--stages 9` | `runs/causal_abstraction_results.json` | PCA/SAE both fail beats-random-every-run |
| Dimensional boundary (2D + time-dep.) | `--stages 10` | `runs/dimensional_boundary_expanded/` | tangent rank = input dim in all families |
| FNO regime boundary | `--stages 11` | `runs/operator_boundary/operator_boundary_report.json` | PR 6.80/64; SAE beats k-matched PCA 4.7× |
| SOTA baselines | `--stages 12` | `runs/sota_baselines/sota_baseline_report.json` | NTK-adaptive 0.0064 < controller 0.0166; GradNorm 2.01 / RBA 2.19 (harm) |
| Statistical hardening | `--stages 13` | `runs/statistical_hardening/analysis_report.json` | MDE 85.7% @ 80% power; P(SAE monitor better) 0.53 |
| Monitors + controller + monitor-source ablation | `--stages 6,7` | `runs/monitor_report.json`, `runs/controller_demo/` | AUROC 0.859/0.864; rescue 0.2953→0.0166; SAE arm bit-identical to random |
| Unit tests | `python -m pytest tests/ -q` | — | **146 passed** (2026-09-07, CPU-only) |
| Smoke test | `bash scripts/run_smoke_test.sh` | transient `runs/ci_smoke/` (cleaned) | PASSED (tests + 60-step train + geometry) |
| Figures | `bash scripts/generate_figures.sh` | `figures/generated/*.png` | 8/8 generated |
| Tables | `bash scripts/generate_tables.sh` | `paper/tables/*.tex` | 6/6 generated from artifacts |
| Artifact integrity | `sha256sum -c data/checksums.sha256` | `data/manifest.json` (v3.1.0) | 16/16 OK |
| Results-doc grounding | `python scripts/check_results_grounded.py` | — | OK (exit 0) |

## Completed since last verification (stage 14 — preregistered, run, recorded)

- **Stage 14 — operator causal battery** (`--stages 14` →
  `runs/operator_causal/operator_causal_report.json`): hypotheses
  H14a/H14b + decision rules were committed and pushed (`f6eedf9`)
  BEFORE the run. **Recorded outcome:** machinery gate PASS; preregistered
  conjunctive rule fires **H14b** (mixed-sign condition failed, 64/64
  positive); conditions (i) 6/8 Bonferroni + BH-FDR survivors at 8/8
  batch consistency and (ii) E_T CI [+7.1e-6, +1.6e-5] excluding zero
  were met — 75% beats-all-controls vs 14–32% for the PINN batteries.
  Reported as a **suggestive causal asymmetry**, NOT a confirmed causal
  boundary; the sign condition is documented as a non-diagnostic design
  lesson. Full record: docs/preregistration.md §H7, RESULTS.md §5A.7,
  CHANGELOG.md 3.2.0.

## Explicitly not yet started (honest backlog)

- OSF / AsPredicted external archival of `docs/preregistration.md`
  (repo-internal preregistration only, so far).
- Human-expert validation study (blinded PCA-vs-SAE identification).
- Wider function-space suite (Darcy-style operators, DeepONet, wider FNOs).
- Power upgrade for the causal batteries (44 runs/feature).
- arXiv preprint (blocked on stage-14 outcome + paper polish).

## Known discrepancies (documented, not hidden)

- The pooled 1D-suite mean participation ratio moved 1.34 → 1.78 in the
  fresh rerun. Explanation + drift table:
  [`docs/fresh_campaign_record.md`](docs/fresh_campaign_record.md) §3 —
  the old value was biased by a degenerate reaction-diffusion baseline
  (exact solution u ≡ 0) that also silently failed to retrain due to a
  stage-1 logging bug. Both bugs are fixed; all artifacts on `main` are
  from the corrected pipeline.
- Monitor AUROCs moved 0.872/0.878 → 0.859/0.864 (run-split level noise;
  conclusion unchanged: no SAE advantage).

## How to re-verify any row of the table above

```bash
git clone https://github.com/shamiquekhan/PINN-Mechanistic-Interpretability
cd PINN-Mechanistic-Interpretability
python -m pytest tests/ -q                              # 146 tests
python scripts/check_results_grounded.py                # doc/artifact grounding
sha256sum -c data/checksums.sha256                      # artifact integrity
python -m experiments.run_pipeline --stages 5           # re-run any stage
```

Full environment + per-stage wall times: [`docs/reproducibility.md`](docs/reproducibility.md).
