# Reproducibility Guide

This document specifies everything needed to reproduce the results in
[RESULTS.md](../RESULTS.md): the reference environment, deterministic
settings, exact commands, expected outputs and runtimes, and the known
sources of nondeterminism.

---

## 1. Reference environment (results as reported)

| Component | Version used for published artifacts |
|---|---|
| OS | Linux 7.0.0-22-generic (x86_64) |
| Python | 3.13.9 (conda) |
| PyTorch | 2.6.0+cu124 |
| CUDA build | 12.4 (nvcc V12.4.131) |
| GPU driver | 595.71.05 |
| GPU | NVIDIA GeForce GTX 1650 (4 GB) |
| NumPy | 2.3.5 |
| scikit-learn | 1.7.2 |
| Pydantic | 2.13.4 |
| PyYAML | 6.0.3 |
| Matplotlib | 3.10.6 |

Fully pinned versions: [requirements.lock](../requirements.lock) (generated
with `pip freeze` in the reference environment). Minimum versions for
development: [requirements.txt](../requirements.txt). The
[Dockerfile](../Dockerfile) pins the CUDA runtime image
(`pytorch/pytorch:2.6.0-cuda12.4-cudnn9-runtime`).

**GPU memory:** the full campaign runs on a 4 GB GPU. Peak observed usage
is < 2 GB (largest model: 512-wide Poisson; largest SAE input: 65,000×64).
CPU-only execution works for the entire unit-test suite (171 tests) and all
analysis stages; only stage-1 training is slower (see §6).

## 2. Installation

```bash
# Option A: exact reproduction (recommended)
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.lock

# Option B: Docker (CUDA runtime)
docker build -t pinn-mi .
docker run --gpus all -v $(pwd)/runs:/workspace/runs pinn-mi python -m pytest tests/ -q

# Option C: development (minimum versions)
pip install -r requirements.txt
```

## 3. Deterministic settings

* Every training run sets `torch.manual_seed(seed)` /
  `torch.cuda.manual_seed_all(seed)` via `pinn/reproducibility.py::set_seed`
  with `deterministic=True`.
* Collocation sampling uses dedicated `torch.Generator` objects seeded
  per call (`seed=` arguments), independent of global RNG state.
* The SAE positive control isolates itself from ambient global RNG state
  (see `experiments/positive_control.py`).
* Bootstrap CI resampling uses `numpy.random.default_rng(42)` (fixed).
* Seeds per run are recorded in each run's `config.json` and inside every
  checkpoint manifest.

**Known nondeterminism** (documented, not eliminated):

1. CUDA floating-point reduction order can differ across GPU
   architectures/driver versions — small (~1e-6) trajectory deviations on
   different hardware. Set `CUBLAS_WORKSPACE_CONFIG=:4096:8` to silence
   cuBLAS determinism warnings; PyTorch's deterministic-algorithms mode is
   NOT globally enabled because it slows the SAE sweeps and changes some
   CUDA kernels (the positive control temporarily disables it for SAE
   stability and restores it afterwards).
2. Matplotlib rendering (PNG byte-level output) can differ across
   versions; figure *data* is identical.
3. Logistic-regression probe/monitor training (scikit-learn) uses an LBFGS
   solver with its own internal tolerance — AUROC values are stable to
   ~1e-3 across library versions.

## 4. Exact reproduction commands

Unit tests first, then the full 15-stage campaign:

```bash
python -m pytest tests/ -q                 # 171 tests, ~15 s CPU
python -m experiments.run_pipeline          # all 15 stages
```

Or stage-by-stage (each stage reads its predecessors' artifacts from
`runs/`; stages are resumable and order-independent except where noted):

| Stage | Command | Produces | Wall time (GTX 1650) |
|---|---|---|---|
| 1 | `python -m experiments.run_pipeline --stages 1` | trained runs in `runs/<config>/` | ~2.5 h (7 configs × 2000–10000 steps) |
| 2 | `--stages 2` | `runs/failure_atlas/` | < 1 min |
| 3 | `--stages 3` | `runs/sae_models_v2/` | ~20 min |
| 4 | `--stages 4` | `runs/feature_dictionary/` | ~2 min |
| 5 | `--stages 5` | `runs/causal_intervention_results.json`, `runs/positive_control.json` | ~25 min |
| 6 | `--stages 6` | `runs/monitor_report.json` | ~3 min |
| 7 | `--stages 7` | `runs/controller_demo/` | ~15 min |
| 8 | `--stages 8` | `runs/pca_causal_results.json` | ~8 min |
| 9 | `--stages 9` | `runs/causal_abstraction_results.json` | ~6 min |
| 10 | `--stages 10` | `runs/dimensional_boundary_expanded/` | ~25 min (trains 12 PINNs) |
| 11 | `--stages 11` | `runs/operator_boundary/` | ~10 min (FNO 4000 steps + SAE) |
| 12 | `--stages 12` | `runs/sota_baselines/` | ~12 min |
| 13 | `--stages 13` | `runs/statistical_hardening/` | < 1 min |

Total compute budget of the published campaign: **≈ 6 GPU-hours**
(RTX-class 4 GB; estimate, not metered — no CO2 estimate is reported
because runtimes were spread across interactive sessions).

Helper wrappers (deterministic ordering + artifact verification):

```bash
scripts/run_smoke_test.sh              # fast CPU sanity (< 5 min)
scripts/reproduce_main_results.sh      # stages 2–15 (uses existing stage-1 runs)
scripts/generate_figures.sh            # all publication figures from runs/
scripts/generate_tables.sh             # all publication tables from runs/
```

## 5. Expected outputs (headline numbers to verify against)

After the campaign completes, these JSON fields must match RESULTS.md
(allowing only the §3-nondeterminism tolerance):

| Check | File | Field | Expected |
|---|---|---|---|
| Failure atlas reproduces | `runs/failure_atlas/seed_matrix_stats.json` | `boundary_starvation.label_counts` | 10/10 boundary_starvation |
| Positive control passes | `runs/positive_control.json` | `pipeline_pass` | `true` |
| SAE causal null | `runs/causal_intervention_results.json` | `causal_strength_ci` | mean ≈ +0.0020, CI spans 0 |
| SAE MC correction | `runs/causal_intervention_results.json` | `multiple_comparisons.bonferroni_n_survivors` | 0 |
| PCA causal null | `runs/pca_causal_results.json` | `multiple_comparisons.bonferroni_n_survivors` | 0 |
| Causal abstraction null | `runs/causal_abstraction_results.json` | `verdict.pca.beats_random_every_run` | `false` |
| Geometry | `runs/effective_rank_analysis/effective_rank_report.json` | `summary.mean_participation_ratio` | ≈ 1.78 |
| FNO regime boundary | `runs/operator_boundary/operator_boundary_report.json` | `verdict.sae_beats_k_matched_pca` | `true` |
| Monitor | `runs/monitor_report.json` | `conventional_logistic.auroc` | ≈ 0.859 |
| Controller rescue | `runs/controller_demo/controller_comparison.json` | `verdict.controller_final` | ≈ 0.0166 |
| SOTA: NTK beats controller | `runs/sota_baselines/sota_baseline_report.json` | `baselines.NTK-adaptive.final_rel_l2` | ≈ 0.0064 |

The reproducibility checklist for a fresh environment is maintained at
[docs/reproducibility_checklist.md](reproducibility_checklist.md).

## 6. CPU-only execution

All 171 unit tests pass CPU-only (`CUDA_VISIBLE_DEVICES="" python -m
pytest tests/ -q`, ~12 s). Analysis stages (2–13) that only read `runs/`
artifacts are CPU-only by construction. Stage 1 training and stages 7/10/11
retraining arms run on CPU at roughly 5–10× slower wall time. The GitHub CI
workflow (`.github/workflows/tests.yml`) runs the CPU path, including a
60-step training + effective-rank smoke test.

## 7. Random seeds

| Experiment block | Seeds |
|---|---|
| Seed matrix (all 5 regimes) | 7, 42, 123, 555, 777, 888, 999, 2024, 2025, 2026 |
| Failure-config single runs (stage 1 legacy) | per `configs/*.yaml` (`run.seed`) |
| Width scaling | seed 0 |
| 2D Poisson pilot | 7, 42, 123, 2024, 2025 |
| Dimensional-boundary expansion (v3) | 7, 42, 123 |
| FNO operator boundary | seed 0 (train), 999 (test split) |
| SAE replicas | 0, 1, 2 |
| Bootstrap CIs | 42 (numpy default_rng) |
| Monitor run split | numpy default_rng(0) |

## 8. Compute and provenance

* All artifacts in `runs/` were generated by the stage scripts above; none
  were hand-edited. Preregistered hypotheses and decision rules:
  [docs/preregistration.md](preregistration.md).
* The experiment inventory mapping every claim to its command, artifact,
  and paper figure is at [docs/experiment_matrix.md](experiment_matrix.md).
* Raw-data provenance and licensing for the benchmark assets:
  [docs/data_card.md](data_card.md) and [data/README.md](../data/README.md).
