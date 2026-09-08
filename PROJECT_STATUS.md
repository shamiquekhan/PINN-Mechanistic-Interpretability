# Project Status

**Last verified:** 2026-09-08 (post-external-review-fix campaign v3.4; every item below re-verified)
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
| Full 15-stage campaign, fresh training | `python -m experiments.run_pipeline` (stages 1–15) | `runs/*.json`, `runs/<config>/` | 2026-09-08 post-fix campaign; verdict drift recorded in `docs/external_review_response.md` |
| Failure atlas (10 seeds × 5 regimes) | `--stages 2` | `runs/failure_atlas/seed_matrix_stats.json` | boundary 10/10, spectral 10/10 labels |
| SAE causal battery + positive control | `--stages 5` | `runs/causal_intervention_results.json`, `runs/positive_control.json` | E_T = −0.0173 [−0.0329, −0.0037] (matched-deletion controls, 0 no-ops); 0/8 Bonferroni & BH; control PASS (cosine 0.989, 9.2×) |
| PCA causal battery | `--stages 8` | `runs/pca_causal_results.json` | E_T = −0.0131 [−0.0268, −0.0006]; 0/8; paired PCA−SAE spans zero |
| Causal abstraction (interchange) | `--stages 9` | `runs/causal_abstraction_results.json` | PCA/SAE both fail beats-random-every-run |
| Dimensional boundary (2D + time-dep.) | `--stages 10` | `runs/dimensional_boundary_expanded/` | tangent rank = input dim in all families; time-dep rel L2 now genuine space-time error (0.111/0.550, was 6.1/6.0 artifacts) |
| FNO regime boundary | `--stages 11` | `runs/operator_boundary/operator_boundary_report.json` | PR 6.80/64; SAE beats k-matched PCA 4.7× |
| SOTA baselines | `--stages 12` | `runs/sota_baselines/sota_baseline_report.json` | Config-faithful: controller 0.0002 < NTK 0.0064 (NTK beat the pre-fix controller 0.0166 — see drift record); GradNorm 2.01 / RBA 2.19 (harm) |
| Statistical hardening | `--stages 13` | `runs/statistical_hardening/analysis_report.json` | MDE 85.7% @ 80% power; P(SAE monitor better) 0.51 |
| NTK conflict↔SAE bridge | `--stages 15` | `runs/ntk_bridge/ntk_bridge_report.json` | H15b: 0/8 survive; best \|ρ\| 0.578 < random p95 0.494; machinery gate PASS |
| Monitors + controller + monitor-source ablation | `--stages 6,7` | `runs/monitor_report.json`, `runs/controller_demo/` | AUROC 0.875/0.877 (gradient arm real); rescue 0.421→0.0002 (config-faithful); SAE arm bit-identical to random |
| Unit tests | `python -m pytest tests/ -q` | — | **155 passed** (2026-09-08, CPU-only) |
| Smoke test | `bash scripts/run_smoke_test.sh` | transient `runs/ci_smoke/` (cleaned) | PASSED (tests + 60-step train + geometry) |
| Figures | `bash scripts/generate_figures.sh` | `figures/generated/*.png` | 9/9 generated |
| Tables | `bash scripts/generate_tables.sh` | `paper/tables/*.tex` | 7/7 generated from artifacts |
| Artifact integrity | `sha256sum -c data/checksums.sha256` | `data/manifest.json` (v3.4.0) | 18/18 OK |
| Results-doc grounding | `python scripts/check_results_grounded.py` | — | OK (exit 0) |

## Completed since last verification (stage 14 — preregistered, run, recorded)

- **Stage 14 — operator causal battery** (`--stages 14` →
  `runs/operator_causal/operator_causal_report.json`): preregistered
  (H14a/H14b at `f6eedf9`), executed, recorded H14b. **Post-external-review
  (v3.4) outcome under matched-deletion controls:** machinery gate PASS;
  2/8 Bonferroni + BH-FDR survivors (was 6/8 under the pre-fix asymmetric
  controls — see docs/external_review_response.md §2), each at 8/8 batch
  consistency; E_T CI [+5.6e-6, +1.6e-5] positive; beats-all 66%.
  Reported as a **suggestive, weakened asymmetry** (2/8 vs 0/8 on PINNs),
  NOT a confirmed boundary. Thread now pending the H16 high-n resolution
  (docs/preregistration.md §H8/H16). Full record: RESULTS.md §5A.7.

## Explicitly not yet started (honest backlog)

- OSF / AsPredicted external archival of `docs/preregistration.md`
  (repo-internal preregistration only, so far). **Bundle prepared:** the
  upload-ready files + manifest + provenance table live in
  `docs/osf_upload/`; only the account-holder upload step remains.
- arXiv submission: **package prepared and compiled** at
  `paper/arxiv_package/` (main.tex, references.bib, 9 figures, 7 tables,
  12-page reference PDF; build recipe + submission checklist in its
  README). Remaining: verify bib details against originals, then upload
  (requires author account).
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
- Monitor AUROCs moved (v3.0 published) → 0.859/0.864 on the fresh split (run-level noise;
  conclusion unchanged: no SAE advantage).

## How to re-verify any row of the table above

```bash
git clone https://github.com/shamiquekhan/PINN-Mechanistic-Interpretability
cd PINN-Mechanistic-Interpretability
python -m pytest tests/ -q                              # 155 tests
python scripts/check_results_grounded.py                # doc/artifact grounding
sha256sum -c data/checksums.sha256                      # artifact integrity
python -m experiments.run_pipeline --stages 5           # re-run any stage
```

Full environment + per-stage wall times: [`docs/reproducibility.md`](docs/reproducibility.md).
