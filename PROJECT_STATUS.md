# Project Status

**Last verified:** 2026-09-10 (post-external-review-fix campaign v3.4 + v3.5 docs/paper sync; every item below re-verified)
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
| NTK conflict↔SAE bridge | `--stages 15` | `runs/ntk_bridge/ntk_bridge_report.json` | H15b: 0/8 survive; best \|ρ\| 0.578 exceeds random p95 0.494 but fails Bonferroni (p=0.56) → no bridge at the registered bar; machinery gate PASS |
| Controller failure battery (H17) | `python -m experiments.controller_failure_battery` | `runs/controller_failure_battery/controller_failure_battery_report.json` | Machinery gate PASS (F1 seed-7 rescue 0.421 → 0.000162); H17a: controller wins F1 (0.00012), NTK-adaptive wins F2 (0.0036) & F3 (1.67); controller never catastrophic (worst mean Δ vs no-action +0.019); GradNorm harms F1 (3.82) |
| Architecture boundary probe (H18) | `python -m experiments.architecture_boundary` | `runs/architecture_boundary/architecture_boundary_report.json` | H18a: Fourier PINN PR 4.0–5.6 (PR/W 0.075) → SAE beats k-matched PCA 3/3 by 25–56× (mean 40.7); depth sweep PR falls 2.05 → 1.15; tangent rank 1 everywhere (covariance/tangent dissociation) |
| Monitor label provenance audit (H19) | `python -m experiments.monitor_label_audit` | `runs/monitor_audit/monitor_audit_report.json` | Label trace: RD2D stage-10 runs never entered training; 2 artifact-labeled pool runs excluded (57→55); loss-trajectory floor AUROC 0.786 [0.615, 0.898] closes 78.2% of the threshold→conventional gap |
| R1 PhysSAE head-to-head (v4.1) | `python -m experiments.r1_physSAE_head_to_head` | `runs/r1_physSAE/r1_report.json` | Gates 3/3 PASS; R1b: alignment generic (random 0.85–0.96), ESF80 concentration replicates but generic (SAE/random 2.0–2.3×), E_T null for every basis (random 0/8 floor on all 6 runs) |
| R3 Fourier frequency sweep (v4.1) | `python -m experiments.r3_frequency_sweep` | `runs/r3_frequency_sweep/r3_report.json` | Gate PASS; 18 runs; PR saturates at the width ceiling (~4.8, ρ 0.075) while the SAE/PCA ratio keeps climbing (TopK 6→41×, ReLU+L1 24→137×) past the plateau — geometry gates, embedding richness drives; tangent 1 everywhere; Spearman +0.87/+0.88 |
| R4 SAE-seed robustness (v4.1) | `python -m experiments.r4_sae_seed_robustness` | `runs/r4_sae_seed_robustness/r4_report.json` | Cross-SAE-seed cosine 0.41/0.41 (non-unique dictionaries, replicating PhysSAE §4.7) with 18/18 stable SAE>PCA verdicts — the dissociation is measured: the regime, not the dictionary, is the unit of claim |
| R5 Burgers within-PINN boundary (v4.1) | `python -m experiments.r5_burgers_boundary` | `runs/r5_burgers_boundary/r5_report.json` | **R5a** — the boundary replicates on a time-dependent family: Fourier (t,x) PR 14.0–15.8 (ρ 0.25, above FNO's 6.8), SAE beats PCA 31–38× all seeds; tanh twin stays at PR ~1.8; tangent 1; rel L2 comparable (architectural, not convergence) |
| R2 causal battery on the Fourier PINN (v4.1) | `python -m experiments.r2_fourier_causal` | `runs/r2_fourier_causal/r2_report.json` | Gate PASS (planted cosine 0.989, 20/20); **R2b both arms**: Fourier E_T −15.89 [−17.46, −14.27] (4/45 Bonf negative-side, 0/45 crossover) despite PR 4.0–5.6 and 25–56× reconstruction; tanh control same null at −0.058. Superposition necessary but not sufficient |
| Monitors + controller + monitor-source ablation | `--stages 6,7` | `runs/monitor_report.json`, `runs/controller_demo/` | AUROC 0.875/0.877 (gradient arm real); rescue 0.421→0.0002 (config-faithful); SAE arm bit-identical to random |
| Unit tests | `python -m pytest tests/ -q` | — | **171 passed, 1 skipped** (2026-09-10, CPU-only; 155 at the 2026-09-08 fresh campaign, +16 from the v3.4.2 stage-17 framework) |
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

## Completed since last verification (stages 16–17 — preregistered, run, recorded)

- **Stage 16 — operator causal asymmetry high-n resolution (H16):** see
  RESULTS.md claim 5; preregistered (H16a/H16b at `ae8dafd`), executed,
  recorded H16b — 6/16 Bonferroni survivors, 0/16 direction-reversal;
  thread T1 closed; the boundary claim rests on the reconstruction side.
- **Stage 17 — controller failure battery (H17):** preregistered (v3.5
  section, `docs/preregistration.md` §H17), executed 2026-09-10,
  recorded **H17a** — the controller wins its home class (boundary
  starvation 0.00012, beats NTK 0.0044) and loses to NTK-adaptive on
  spectral suppression (0.335 vs 0.0036) and RD-2D (1.98 vs 1.67),
  never catastrophically below no-action. Machinery gate PASS. A
  v3.4.2-driver machinery incident (5 defects, incl. fake NTK/GradNorm
  arms and a hard-coded failure_class) was fixed at `fc77850` BEFORE
  the run and before any verdict was read — recorded in §H17 and
  RESULTS.md §5A.9. Full numbers:
  `runs/controller_failure_battery/controller_failure_battery_report.json`.
- **Stage 18 — architecture boundary probe (H18):** preregistered
  (v3.5 section, `docs/preregistration.md` §H18), executed 2026-09-10,
  recorded **H18a** — the Fourier-feature PINN (width 64, n_freq=32)
  reaches activation PR 4.0–5.6 (above the width-scaling envelope
  1.14–2.51, FNO neighborhood), and there the same TopK SAE that is
  null on tanh PINNs beats k-matched PCA **25–56×** (mean 40.7, 3/3
  seeds). Depth sweep: PR falls monotonically 2.05 → 1.15 (depth does
  not rescue superposition). Tangent rank = 1 everywhere — covariance
  PR, not manifold dimension, is the regime discriminator. Pre-run
  machinery fix: the dead `fourier_embed` wiring in
  `experiments/train.py` was fixed at `f647dca` before the run. Full
  numbers: `runs/architecture_boundary/architecture_boundary_report.json`;
  RESULTS.md §5A.10.
- **Stage 19 — monitor label provenance audit (H19):** preregistered
  (v3.5 section, `docs/preregistration.md` §H19), executed 2026-09-10,
  recorded — label provenance traced per-run (the stage-10 RD2D runs
  never entered monitor training; 2 artifact-labeled pool runs excluded,
  57→55; collocation derives no failure step — never tested), and the
  loss-trajectory floor (AUROC 0.786 [0.615, 0.898]) closes 78.2% of
  the threshold→conventional gap: the threshold floor was a straw man,
  the conventional arm's marginal value is thin but real. Full numbers:
  `runs/monitor_audit/monitor_audit_report.json`; RESULTS.md §5A.11.

## Explicitly not yet started (honest backlog)

- OSF / AsPredicted external archival of `docs/preregistration.md`
  (repo-internal preregistration only, so far). **Bundle prepared:** the
  upload-ready files + manifest + provenance table live in
  `docs/osf_upload/`; only the account-holder upload step remains.
- arXiv submission: **package prepared and compiled** at
  `paper/arxiv_package/` (main.tex, references.bib, 10 figures, 7 tables,
  17-page reference PDF; build recipe + submission checklist in its
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
python -m pytest tests/ -q                              # 171 tests
python scripts/check_results_grounded.py                # doc/artifact grounding
sha256sum -c data/checksums.sha256                      # artifact integrity
python -m experiments.run_pipeline --stages 5           # re-run any stage
```

Full environment + per-stage wall times: [`docs/reproducibility.md`](docs/reproducibility.md).
