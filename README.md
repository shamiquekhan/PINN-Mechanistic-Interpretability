# Mechanistic Interpretability for Physics-Informed Neural Networks (PINNs)

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch CUDA](https://img.shields.io/badge/PyTorch-CUDA-orange.svg)](https://pytorch.org/)
[![License: CC BY 4.0](https://img.shields.io/badge/License-CC_BY_4.0-lightgrey.svg)](https://creativecommons.org/licenses/by/4.0/)
[![Tests: 171/171 Passed](https://img.shields.io/badge/Tests-171%2F171%20Passed-brightgreen.svg)](tests/)
[![CI](https://github.com/shamiquekhan/PINN-Mechanistic-Interpretability/actions/workflows/tests.yml/badge.svg)](https://github.com/shamiquekhan/PINN-Mechanistic-Interpretability/actions/workflows/tests.yml)

A GPU-accelerated research framework for analyzing, monitoring, and intervening on optimization failure modes in Physics-Informed Neural Networks (PINNs) via Sparse Autoencoders (SAEs), causal counterfactuals, causal-abstraction interchange interventions, early-warning monitors, and closed-loop adaptive control — plus a Fourier Neural Operator (FNO) positive control that locates the regime where SAE methodology does work.

---

## Key Features

- **GPU-Native Core Architecture (`pinn/`)**: Reusable `PINNTrainer`, named layer activations, parameter count utilities, Xavier/Kaiming initializations, and Random Fourier Feature Embeddings ($x \mapsto [\sin(Bx), \cos(Bx)]$).
- **Unified PDE Specification Engine (`pinn/pdes.py`)**: **Poisson 1D**, **Advection 1D**, **Reaction-Diffusion 1D**, **Poisson 2D**, **Advection-Diffusion 2D**, **Reaction-Diffusion 2D**, **viscous Burgers (t,x)**, and **Allen-Cahn (t,x)** — with manufactured solutions, stable integrating-factor reference solvers, enforced initial conditions, and boundary residual computations.
- **Neural Operator Module (`operators/`)**: Dependency-free **FNO1d** (spectral convolutions, named block states) and a parametric **Green's-function regression** dataset — the high-rank positive control for the regime-boundary experiment.
- **Sparse Autoencoder Feature Discovery (`sae/`)**: TopK and legacy ReLU+L1 SAEs with activation normalization, decoder unit-norm enforcement, dead-feature tracking, run-level splits, PCA/random baselines, and cross-seed feature matching.
- **Leakage-Free Causal Interventions (`interventions/`)**: Forward-hook intervention engine with targeted, unrelated, norm-matched random, and supervised probe controls; quantitative Causal Strength ($E_T$) and Specificity ($S_{\text{spec}}$) metrics; **PCA causal battery** (`pca_battery.py`) mirroring the SAE protocol component-for-component; and **partial interchange interventions** (`interchange.py`) implementing the causal-abstraction criterion of Geiger et al. 2021/2022.
- **Early-Warning Failure Monitoring (`monitoring/`)**: Past-only sliding window trajectory feature extractors with strict `leakage_audit` verification, threshold degradation alarms, and logistic early-warning classifiers.
- **Closed-Loop Adaptive PINN Controller (`controller/`)**: Deterministic state-machine controller ($\text{IDLE} \rightarrow \text{WARNING} \rightarrow \text{CONFIRMED} \rightarrow \text{COOLDOWN} \rightarrow \text{IDLE}$; the bounded action fires on CONFIRMED entry) with 4 corrective action handlers (lambda rebalance and gradnorm rebalance act on the loss; the resample/fourier triggers are signals consumed by the training loop), rollback, and a measured monitor-source ablation.
- **Failure Atlas & Physics-Feature Dictionary (`analysis/`)**: Automated taxonomy indexing, annotated physics latent features, effective rank / participation ratio / local tangent rank analysis, statistical hardening (power analysis, Bayesian posteriors, threshold sensitivity), and activation-manifold visualizations.
- **SOTA Optimization Baselines (`experiments/sota_baselines.py`)**: GradNorm, NTK-adaptive weighting, and RBA residual attention for comparison against the controller.

## Current Evidence (v4.1)

The thesis: **when** sparse feature representations support mechanistic interpretation is a measurable function of representational geometry — and the boundary runs *inside* the PINN family, not between PINNs and operators. All numbers are read from committed artifacts (regenerate with `scripts/reproduce_main_results.sh`; machine-checked by `scripts/check_results_grounded.py` in CI):

1. **SAE causal criterion null (stage 5, honest controls):** no tested feature satisfied the preregistered direction-specific causal criterion — $E_T$ = −0.0173, 95% CI [−0.0329, −0.0037] — ablating a candidate feature moves the loss *less* than ablating another active latent (matched-deletion controls); 0/8 features survive Bonferroni or BH-FDR; planted-feature positive control passes (cosine 0.989, 9.2× its matched control).
2. **PCA causal criterion null (stage 8):** the identical battery on PCA components — 0/8 survive, paired PCA−SAE difference spans zero. No feature basis, linear or sparse, satisfies the direction-specific causal criterion in the low-rank regime.
3. **Causal abstraction null (stage 9):** neither PCA nor SAE alignments beat a random basis in partial interchange interventions for region identity.
4. **Geometry (stages 3/10 + H18):** local tangent rank exactly equals input dimension across all eight PDE families and widths 16–512 (the provable bound, saturated); covariance PR 1.14–2.51 — and **depth lowers it further** (2.05 → 1.15, depths 2–6). PINN activations are not in superposition.
5. **The boundary is WITHIN PINNs (stage 18, H18a — the centerpiece):** a Fourier-feature PINN (width 64, n_freq=32) on the same 1D Poisson task reaches activation PR 4.0–5.6 (FNO neighborhood: 6.8) and there **the same TopK SAE beats k-matched PCA 25–56×** (3/3 seeds). Tangent rank stays 1 everywhere — covariance rank is the regime discriminator, dissociated from manifold dimension. The preregistered rank diagnostic predicted which runs would flip.
6. **Operator regime (stages 11/14/16):** FNO PR 6.8, SAE beats k-matched PCA 4.7× (reconstruction); the operator causal-asymmetry thread is closed at H16b (6/16 Bonferroni, 0/16 direction-reversal) — the boundary's causal side rests on R2.
7. **PhysSAE reconciliation (v4.1, R1 — R1b):** on matched-architecture checkpoints (5×128, their training spec, 3 seeds), PhysSAE-style evidence replicates but is *generic*: concept alignment appears for every basis including random directions (max |r| 0.85–0.96); SAE ablation footprints are 2.0–2.5× more concentrated than PCA/ICA — *and equally more concentrated than random directions*; no basis satisfied our preregistered direction-specific causal criterion (random at the 0/8 chance floor on all 6 runs, so the battery is calibrated). The causal criteria dissociate; both programs independently discovered the rank-collapse-with-convergence regularity. The side-by-side basis × criteria matrix is `runs/r1_physSAE/reconciliation_table.json` (regenerate with `python scripts/r1_reconciliation_table.py`). See `docs/physSAE_reconciliation.md`.
8. **Dose-response null (v4.2, R6):** across the full intervention grid α ∈ {−2…2} with dose-matched closest-activity controls, **no feature in any high-rank regime shows a signed causal channel** (Fourier SAE/PCA 0/24; FNO 0/8): the response is dose-symmetric — the reconstructive signature — and the controls match it; the planted-feature gate (whose own quadratic-channel correction is part of the record) separates at 67× where real features reach 1.5–3.6×. The two-point criteria's null is confirmed at curve level and the reason the crossover test returned null is now measured: the curve has no signed shape to detect.
9. **Engineering, honestly bounded (stages 6/7/12 + H17/H19):** conventional monitor AUROC 0.875 — but the honest loss-trajectory floor reaches 0.786 (H19: much of the apparent margin was the rule, not the features); controller rescues boundary starvation to oracle level, **wins its home class** (0.00012 vs NTK 0.0044), never degrades catastrophically across the three-failure battery, and loses to NTK-adaptive elsewhere (H17: robustness, not per-failure SOTA). SAE features are inert cargo throughout.

See [RESULTS.md](RESULTS.md) for the complete evidence, preregistered hypotheses (`docs/preregistration.md`), and limitations. For what is *verified end-to-end right now* versus in-flight, see [PROJECT_STATUS.md](PROJECT_STATUS.md). For the executed 29-issue external review — every fix, re-run, and the errata for numbers that moved — see [REVIEW_RESPONSE.md](REVIEW_RESPONSE.md).

---

## Framework Architecture

```
                               ┌─────────────────────────┐
                               │   PINN Trainer (CUDA)   │
                               └────────────┬────────────┘
                                            │ Forward Pass
                                            ▼
┌──────────────────────────────────────────────────────────────────────────────────────┐
│                                 Intervention Engine                                  │
│  ┌────────────────────┐      ┌─────────────────────────┐     ┌────────────────────┐  │
│  │   Layer Activation │ ───► │  Sparse Autoencoder     │ ──► │ Modified Activation│  │
│  │   h_l = σ(W_l x)   │      │  z = ReLU(W_e h + b_e)  │     │ h' = W_d z' + b_d  │  │
│  └────────────────────┘      └─────────────────────────┘     └────────────────────┘  │
└───────────────────────────────────────────┬──────────────────────────────────────────┘
                                            │ Log Metrics & Trajectories
                                            ▼
                               ┌─────────────────────────┐
                               │  Past-Only Monitor      │
                               │  (Leakage Audit Passed) │
                               └────────────┬────────────┘
                                            │ Alarm Score
                                            ▼
                               ┌─────────────────────────┐
                               │  Closed-Loop Controller │
                               │  State Machine          │
                               └────────────┬────────────┘
                                            │ Bounded Corrective Action
                                            ▼
                               ┌─────────────────────────┐
                               │  Adaptive PINN Rescue   │
                               └─────────────────────────┘
```

---

## Experimental Qualification Summary

All core modules have been validated across a matrix of baseline and failure-inducing configurations on CUDA GPU (fresh-campaign values from `runs/<config>/metrics.jsonl`; 10-seed statistics with CIs in `runs/failure_atlas/seed_matrix_stats.json`):

| Configuration | PDE Family | Key Parameters | Rel $L_2$ Error | Status / Outcome |
| :--- | :--- | :--- | :--- | :--- |
| **`poisson_baseline`** | Poisson 1D | Standard weights, Xavier init | **0.00069** | **Converged** Clean solution |
| **`advection_baseline`** | Advection 1D | Standard weights | **0.0011** | **Converged** |
| **`reaction_diffusion_baseline`** | Reaction-Diffusion 1D | $\mu{=}1, f{=}1, \varepsilon{=}0.01$ | **0.0015** | **Converged** (stiff BL) |
| **`failure_boundary_starvation`** | Poisson 1D | $\lambda_{\text{pde}}=100, \lambda_{\text{bc}}=0.01$ | **0.70** (seed matrix: 2.31 ± 1.59) | **Failed** BC error dominates |
| **`failure_gradient_conflict`** | Poisson 1D | High LR ($\eta = 0.1$) | **1.0** (labels not seed-reproducible → excluded from class claims) | **Failed** Severe gradient oscillation |
| **`failure_spectral_suppression`** | Poisson 1D | Narrow net, source $= 50$ | **0.108** (seed matrix: 0.395 ± 0.114) | **Failed** High-frequency aliasing |
| **`failure_collocation_starvation`**| Poisson 1D | $N_i = 8$ interior points | **0.036** | **Stalled** Insufficient sampling |

---

## Quick Start

### 1. Installation

```bash
git clone https://github.com/shamiquekhan/PINN-Mechanistic-Interpretability.git
cd PINN-Mechanistic-Interpretability
pip install -r requirements.txt
```

### 2. Run Test Suite

```bash
pytest tests/ -v
```
*(The suite currently contains 171 tests. CUDA determinism warnings may appear on systems without the documented cuBLAS workspace setting.)*

### 3. Launch End-to-End Master Research Pipeline

```bash
python -m experiments.run_pipeline              # all 15 stages
python -m experiments.run_pipeline --stages 5   # any subset, e.g. the causal battery
```

Reproducibility wrappers (see [docs/reproducibility.md](docs/reproducibility.md)):

```bash
bash scripts/run_smoke_test.sh           # CPU sanity: tests + short train + geometry (< 5 min)
bash scripts/reproduce_main_results.sh   # stages 2–15 from committed artifacts + headline verification
bash scripts/generate_figures.sh         # all publication figures from runs/ JSONs
bash scripts/generate_tables.sh          # LaTeX tables into paper/tables/
```
This automatically executes:
1. Training PINNs across all baseline & failure configurations.
2. Indexing the Failure Atlas (`runs/failure_atlas/failure_atlas_index.json`).
3. Training TopK Sparse Autoencoders with PCA, random, and seed-replica baselines (`runs/sae_models_v2/`).
4. Building the Physics-Feature Dictionary (`runs/feature_dictionary/physics_feature_dictionary.md`).
5. Computing Causal Intervention & Specificity scores + planted-feature positive control.
6. Training & evaluating Early-Warning Monitors.
7. Demonstrating Closed-Loop Controller rescue + monitor-source ablation.
8. **PCA causal battery** — head-to-head vs SAE (`runs/pca_causal_results.json`).
9. **Causal-abstraction partial interchanges** — PCA/SAE/random alignments (`runs/causal_abstraction_results.json`).
10. **Dimensional boundary** — 2D + time-dependent PDE geometry (`runs/dimensional_boundary_expanded/`).
11. **Operator regime boundary** — FNO on Green's-function regression (`runs/operator_boundary/`).
12. **SOTA baselines** — GradNorm / NTK-adaptive / RBA vs controller (`runs/sota_baselines/`).
13. **Statistical hardening** — power, Bayesian posterior, threshold sensitivity (`runs/statistical_hardening/`).
14. **Operator causal battery** — the 3-control causal protocol + interchange on FNO block states, completing the regime-boundary claim at reconstruction level (`runs/operator_causal/`). **Stage 16 (H16)** closes the causal-asymmetry thread: high-n battery (n=20, direction-reversal criterion) fires H16b — 6/16 Bonferroni survivors, 0/16 direction-reversal survivors (`runs/operator_highn/`).
15. **NTK conflict↔SAE bridge** — the preregistered correlational bridge between Wang-et-al. gradient conflict and feature-specific SAE activity (revision Gap 2; recorded null) (`runs/ntk_bridge/`).
16. **Operator causal asymmetry, high n** — the direction-reversal battery on FNO block states; H16b closed thread T1 (`runs/operator_highn/`).
17. **Controller failure battery** — H17a: robustness across three failure classes, not per-failure SOTA (`runs/controller_failure_battery/`).
18. **Architecture boundary** — H18a: Fourier PINN PR 4.0–5.6, SAE beats PCA 25–56× — superposition *within PINNs* (`runs/architecture_boundary/`).
19. **Monitor label provenance audit** — H19: honest trajectory floor 0.786; threshold floor was a straw man (`runs/monitor_audit/`).
20. **R1 — PhysSAE head-to-head** — preregistered reconciliation on matched checkpoints: their evidence is generic (R1b) (`runs/r1_physSAE/`).
21. **R2 — causal battery on the Fourier PINN** — the third arrow does not close; superposition necessary but not sufficient (R2b) (`runs/r2_fourier_causal/`).
22. **R3 — frequency sweep** — the dose-response curve: rank saturates, compression keeps climbing (`runs/r3_frequency_sweep/`).
23. **R4 — SAE-seed robustness** — dictionaries non-unique (cosine 0.41), verdicts stable 18/18 (`runs/r4_sae_seed_robustness/`).
24. **R5 — Burgers boundary** — R5a: the boundary replicates on a time-dependent family (`runs/r5_burgers_boundary/`).
25. **R6 — intervention dose-response** — the causal criteria at curve level: no signed channel in any regime; dose-symmetric reconstructive signature, controls matched (`runs/r6_dose_response/`).

Stages 1–15 run through `experiments/run_pipeline.py`; stages 16–19 and R1–R5 have dedicated preregistered drivers (see the experiments tree below).

---

## Repository Structure

```
.
├── pinn/                     # GPU-Native PINN Core
│   ├── model.py              # MLP, FourierEmbedding, named layers
│   ├── pdes.py               # Poisson 1D/2D, Advection, RD 1D, AD-2D, RD-2D, Burgers, Allen-Cahn
│   ├── trainer.py            # Reusable PINNTrainer with hooks & per-loss gradients
│   ├── config.py             # Validated Pydantic v2 configuration schemas
│   ├── activations.py        # ActivationLogger & probe point recorder
│   ├── diagnostics.py        # Residual & spectral diagnostic utilities
│   └── gradients.py          # Gradient norm & cosine similarity trackers
├── operators/                # Neural Operator Module (v3)
│   └── fno.py                # FNO1d (spectral conv) + Green's-function dataset
├── sae/                      # Sparse Autoencoder Module
│   ├── model.py              # Overcomplete SparseAutoencoder with L1 & dead-feature tracking
│   ├── dataset.py            # Run-partitioned ActivationDataset
│   ├── train.py              # Training loop, PCA baseline, & hyperparameter sweep
│   └── features.py           # Feature statistics, metric association, & Hungarian matching
├── interventions/            # Causal Intervention Engine
│   ├── engine.py             # SAEInterventionHook implementing 8 causal modes
│   ├── causal.py             # Counterfactual runner, causal scoring (E_T, S_spec), MC corrections
│   ├── pca_battery.py        # PCA-component causal battery (mirrors SAE protocol)
│   ├── interchange.py        # Partial interchange interventions (causal abstraction)
│   └── causal_abstraction.py # CausalModel container + alignment primitives
├── monitoring/               # Early-Warning Failure Monitoring
│   ├── features.py           # Past-only sliding window trajectory features & leakage audit
│   └── models.py             # ThresholdMonitor & LogisticMonitor early warning classifiers
├── controller/               # Closed-Loop Adaptive PINN Controller
│   ├── state_machine.py      # Deterministic state machine controller
│   └── actions.py            # 4 corrective action handlers (BC reweight, GradNorm, Resample, Fourier)
├── analysis/                 # Diagnostic & Reporting Tools
│   ├── activation_manifold.py # Tangent-rank and PCA manifold diagnostics
│   ├── effective_rank.py     # Participation-ratio / stable-rank analysis
│   ├── pca_interpretability.py # PCA basis + component interventions
│   ├── statistical_hardening.py # Power analysis, Bayesian posterior, threshold sensitivity
│   ├── failure_atlas.py      # Quantitative Failure Atlas aggregator & indexer
│   ├── feature_dictionary.py # Physics-Feature Dictionary markdown reporter
│   └── evaluate.py           # Evaluation script for completed runs
├── configs/                  # Experiment Config Files (YAML)
├── experiments/              # Execution Scripts
│   ├── train.py              # Single experiment trainer
│   ├── qualification.py      # Seed matrix qualification gate runner
│   ├── width_scaling.py      # PR/tangent-rank width follow-up
│   ├── dimensional_boundary_expanded.py # 2D suite + time-dependent geometry (v3)
│   ├── operator_boundary.py  # FNO regime-boundary experiment (v3)
│   ├── sota_baselines.py     # GradNorm / NTK-adaptive / RBA baselines (v3)
│   ├── pca_causal.py         # PCA causal battery driver (v3)
│   ├── causal_abstraction.py # Interchange battery driver (v3)
│   ├── positive_control.py   # Planted-feature pipeline sanity
│   ├── operator_causal.py    # Operator causal battery (stage 14)
│   ├── ntk_bridge.py        # NTK conflict ↔ SAE activity bridge (stage 15)
│   ├── operator_highn.py    # High-n direction-reversal battery (stage 16 / H16)
│   ├── controller_failure_battery.py # Controller generalization battery (stage 17 / H17)
│   ├── architecture_boundary.py # Fourier/depth rank probe (stage 18 / H18)
│   ├── monitor_label_audit.py # Monitor label provenance audit (stage 19 / H19)
│   ├── r1_physSAE_head_to_head.py # PhysSAE reconciliation head-to-head (R1)
│   ├── r2_fourier_causal.py # Causal battery on the Fourier PINN (R2)
│   ├── r3_frequency_sweep.py # Fourier frequency dose-response (R3)
│   ├── r4_sae_seed_robustness.py # SAE-seed robustness, Hungarian matching (R4)
│   ├── r5_burgers_boundary.py # Within-PINN boundary on Burgers (R5)
│   └── run_pipeline.py       # Master end-to-end research campaign (stages 1–15)
├── tests/                    # Comprehensive Unit Test Suite (171 tests)
│   └── unit/
├── docs/
│   ├── theory_activation_rank.md # Tangent-rank bound + corrected covariance discussion
│   ├── preregistration.md    # Preregistered hypotheses & outcomes (H1–H19, R1–R5)
│   ├── physSAE_reconciliation.md # Concurrent-work reconciliation + the level ladder
│   ├── final_claim_ladder_v4.md # The binding claim wording + evidence chains
│   ├── paper_draft.md        # Historical v3 draft skeleton (superseded by paper/)
│   └── data_card.md          # Benchmark data card
├── figures/                  # Publication figure scripts (read runs/ JSONs only)
│   ├── _common.py            #   shared artifact-loading + style helpers
│   ├── figure_01..10_*.py    #   one script per paper figure
│   └── generated/            #   PNG output (git-ignored)
├── scripts/                  # User-facing reproducibility commands
│   ├── setup_env.sh          #   pinned venv setup (requirements.lock)
│   ├── run_smoke_test.sh     #   CPU smoke: tests + 60-step train + geometry
│   ├── run_full_pipeline.sh  #   full 15-stage campaign (~6 GPU-hours)
│   ├── reproduce_main_results.sh  # stages 2–15 + headline verification
│   ├── generate_figures.sh   #   all publication figures
│   └── generate_tables.{sh,py}  # LaTeX tables from artifacts
├── paper/                    # Submission artifact (LaTeX)
│   ├── main.tex, references.bib, README.md
│   ├── tables/               #   GENERATED — do not hand-edit
│   ├── figures/, supplementary/
├── data/
│   ├── README.md, manifest.json, checksums.sha256   # artifact provenance + SHA256
├── .github/workflows/        # CI: tests.yml (CPU 3.10/3.12 + smoke), lint.yml
├── Dockerfile                # Pinned reproducible environment
├── LICENSE                   # CC BY 4.0
├── CITATION.cff              # Machine-readable citation metadata
├── CHANGELOG.md              # Research + code evolution (v1 → v4)
├── ROADMAP.md                # Completed / Planned / Future
├── CONTRIBUTING.md, SECURITY.md, CODE_OF_CONDUCT.md
├── ARCHITECTURE.md           # In-depth architectural design specifications
├── DOCUMENTATION.md          # Comprehensive API & pipeline documentation
├── requirements.txt          # Minimum Python dependencies
└── requirements.lock         # Fully pinned reference environment
```

### Key documentation

| Document | Purpose |
|---|---|
| [RESULTS.md](RESULTS.md) | The evidence: every claim with numbers, CIs, and limitations |
| [docs/preregistration.md](docs/preregistration.md) | All hypotheses + decision rules (H1–H19, R1–R5), written before running |
| [docs/physSAE_reconciliation.md](docs/physSAE_reconciliation.md) | Concurrent-work reconciliation, confound decomposition, level ladder |
| [docs/final_claim_ladder_v4.md](docs/final_claim_ladder_v4.md) | Binding claim wording + evidence chains + the L1–L5 interpretability ladder |
| [docs/experiment_matrix.md](docs/experiment_matrix.md) | Claim → hypothesis → command → artifact → figure map |
| [docs/reproducibility.md](docs/reproducibility.md) | Environment, seeds, nondeterminism, expected outputs |
| [docs/reproducibility_checklist.md](docs/reproducibility_checklist.md) | Fresh-environment verification protocol |
| [docs/theory_activation_rank.md](docs/theory_activation_rank.md) | Provable tangent-rank bound + corrected covariance discussion |
| [docs/failure_taxonomy.md](docs/failure_taxonomy.md) | Operational failure-mode definitions |
| [docs/glossary.md](docs/glossary.md) | Terminology used throughout |
| [docs/data_card.md](docs/data_card.md) | Benchmark provenance, intended use, limitations |
| [docs/paper_draft.md](docs/paper_draft.md) | Working manuscript draft |
| [runs/README.md](runs/README.md) | Artifact layout: raw vs derived vs authoritative |

---

## License

This project is licensed under the **Creative Commons Attribution 4.0 International License (CC BY 4.0)** — see the [LICENSE](LICENSE) file for details. Citation metadata: [CITATION.cff](CITATION.cff).
