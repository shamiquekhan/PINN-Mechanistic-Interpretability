# Mechanistic Interpretability for Physics-Informed Neural Networks (PINNs)

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch CUDA](https://img.shields.io/badge/PyTorch-CUDA-orange.svg)](https://pytorch.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Tests: 94/94 Passed](https://img.shields.io/badge/Tests-94%2F94%20Passed-brightgreen.svg)](tests/)

A GPU-accelerated research framework for analyzing, monitoring, and intervening on optimization failure modes in Physics-Informed Neural Networks (PINNs) via Sparse Autoencoders (SAEs), causal counterfactuals, early-warning monitors, and closed-loop adaptive control.

---

## Key Features

- **GPU-Native Core Architecture (`pinn/`)**: Reusable `PINNTrainer`, named layer activations, parameter count utilities, Xavier/Kaiming initializations, and Random Fourier Feature Embeddings ($x \mapsto [\sin(Bx), \cos(Bx)]$).
- **Unified PDE Specification Engine (`pinn/pdes.py`)**: Unified interface supporting **Poisson 1D**, **Advection 1D**, and **Reaction-Diffusion 1D** equations with exact solutions, validation grids, and boundary residual computations.
- **Sparse Autoencoder Feature Discovery (`sae/`)**: TopK and legacy ReLU+L1 SAEs with activation normalization, decoder unit-norm enforcement, dead-feature tracking, run-level splits, PCA/random baselines, and cross-seed feature matching.
- **Leakage-Free Causal Interventions (`interventions/`)**: Forward-hook intervention engine supporting targeted, unrelated, norm-matched random, and supervised probe controls with quantitative Causal Strength ($E_T$) and Specificity ($S_{\text{spec}}$) metrics.
- **Early-Warning Failure Monitoring (`monitoring/`)**: Past-only sliding window trajectory feature extractors with strict `leakage_audit` verification, threshold degradation alarms, and logistic early-warning classifiers.
- **Closed-Loop Adaptive PINN Controller (`controller/`)**: Deterministic state-machine controller ($\text{IDLE} \rightarrow \text{WARNING} \rightarrow \text{CONFIRMED} \rightarrow \text{ACTING} \rightarrow \text{COOLDOWN} \rightarrow \text{IDLE}$) executing 4 bounded corrective action handlers ($\lambda_{\text{bc}}$ reweighting, GradNorm loss balancing, high-residual resampling, and Fourier feature injection).
- **Failure Atlas & Physics-Feature Dictionary (`analysis/`)**: Automated taxonomy indexing experimental failure modes, extracting annotated physics latent features, and measuring effective rank / participation ratio before SAE training.

## Current Evidence

The current benchmark is a scoped negative mechanistic result, not a claim that SAEs never work on scientific networks. Hidden activations in this 1D PINN suite have participation ratio about 1.34 of 64 dimensions, so k-matched PCA outperforms the SAE and no candidate feature survives corrected causal testing. A planted-feature positive control passes, showing that the intervention pipeline can recover a causal feature when one exists.

The conventional and SAE-augmented monitors are evaluable on the final held-out split (AUROC 0.872 and 0.878; run-level 95% CIs [0.795, 0.953] and [0.806, 0.958]). Their intervals overlap, so the benchmark does not establish an SAE advantage. The controller rescue result is reported separately as an engineering result. Gradient-conflict and collocation-starvation were stress-tested but excluded from final failure-class claims because their operational labels did not reproduce cleanly. See [RESULTS.md](RESULTS.md) for the complete evidence and limitations.

A width-scaling follow-up over widths 16–512 on the fixed 1D Poisson task found PR `1.74–2.00` and local tangent rank `1` at every width. This is evidence about this task family, not a universal rank bound; higher-dimensional and operator-learning settings remain future boundary experiments.

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

All core modules have been validated across a matrix of baseline and failure-inducing configurations on CUDA GPU:

| Configuration | PDE Family | Key Parameters | Rel $L_2$ Error | Status / Outcome |
| :--- | :--- | :--- | :--- | :--- |
| **`success_baseline`** | Poisson 1D | Standard weights, Xavier init | **$2.74 \times 10^{-4}$** | **Converged** Clean solution |
| **`boundary_starvation`** | Poisson 1D | $\lambda_{\text{pde}}=100, \lambda_{\text{bc}}=0.01$ | **0.278957** | **Failed** BC error dominates |
| **`gradient_conflict`** | Poisson 1D | High LR ($\eta = 0.1$) | **0.410831** | **Failed** Severe gradient oscillation |
| **`spectral_suppression`** | Poisson 1D | Narrow net, source $= 50$ | **0.265287** | **Failed** High-frequency aliasing |
| **`collocation_starvation`**| Poisson 1D | $N_i = 8$ interior points | **0.042475** | **Stalled** Insufficient sampling |

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
*(The suite currently contains 94 tests. CUDA determinism warnings may appear on systems without the documented cuBLAS workspace setting.)*

### 3. Launch End-to-End Master Research Pipeline

```bash
python -m experiments.run_pipeline
```
This automatically executes:
1. Training PINNs across all baseline & failure configurations.
2. Indexing the Failure Atlas (`runs/failure_atlas/failure_atlas_index.json`).
3. Training TopK Sparse Autoencoders with PCA, random, and seed-replica baselines (`runs/sae_models_v2/`).
4. Building the Physics-Feature Dictionary (`runs/feature_dictionary/physics_feature_dictionary.md`).
5. Computing Causal Intervention & Specificity scores (`runs/causal_intervention_results.json`).
6. Training & evaluating Early-Warning Monitors.
7. Demonstrating Closed-Loop Controller rescue on boundary-starvation failure trajectories.

---

## Repository Structure

```
.
├── pinn/                     # GPU-Native PINN Core
│   ├── model.py              # MLP, FourierEmbedding, named layers
│   ├── pdes.py               # Poisson, Advection, Reaction-Diffusion interfaces
│   ├── trainer.py            # Reusable PINNTrainer with hooks & per-loss gradients
│   ├── config.py             # Validated Pydantic v2 configuration schemas
│   ├── activations.py        # ActivationLogger & probe point recorder
│   ├── diagnostics.py        # Residual & spectral diagnostic utilities
│   └── gradients.py          # Gradient norm & cosine similarity trackers
├── sae/                      # Sparse Autoencoder Module
│   ├── model.py              # Overcomplete SparseAutoencoder with L1 & dead-feature tracking
│   ├── dataset.py            # Run-partitioned ActivationDataset
│   ├── train.py              # Training loop, PCA baseline, & hyperparameter sweep
│   └── features.py           # Feature statistics, metric association, & Hungarian matching
├── interventions/            # Causal Intervention Engine
│   ├── engine.py             # SAEInterventionHook implementing 7 causal modes
│   └── causal.py             # Counterfactual experiment runner & causal scoring (E_T, S_spec)
├── monitoring/               # Early-Warning Failure Monitoring
│   ├── features.py           # Past-only sliding window trajectory features & leakage audit
│   └── models.py             # ThresholdMonitor & LogisticMonitor early warning classifiers
├── controller/               # Closed-Loop Adaptive PINN Controller
│   ├── state_machine.py      # Deterministic state machine controller
│   └── actions.py            # 4 corrective action handlers (BC reweight, GradNorm, Resample, Fourier)
├── analysis/                 # Diagnostic & Reporting Tools
│   ├── activation_manifold.py # Tangent-rank and PCA manifold diagnostics
│   ├── failure_atlas.py      # Quantitative Failure Atlas aggregator & indexer
│   ├── feature_dictionary.py # Physics-Feature Dictionary markdown reporter
│   └── evaluate.py           # Evaluation script for completed runs
├── configs/                  # Experiment Config Files (YAML)
│   ├── poisson_baseline.yaml
│   ├── advection_1d_baseline.yaml
│   ├── reaction_diffusion_1d_baseline.yaml
│   ├── failure_boundary_starvation.yaml
│   ├── failure_gradient_conflict.yaml
│   ├── failure_spectral_suppression.yaml
│   └── failure_collocation_starvation.yaml
├── experiments/              # Execution Scripts
│   ├── train.py              # Single experiment trainer
│   ├── qualification.py      # Seed matrix qualification gate runner
│   ├── width_scaling.py      # PR/tangent-rank width follow-up
│   └── run_pipeline.py       # Master end-to-end research campaign execution script
├── tests/                    # Comprehensive Unit Test Suite (94 tests)
│   └── unit/
├── ARCHITECTURE.md           # In-depth architectural design specifications
├── DOCUMENTATION.md          # Comprehensive API & pipeline documentation
└── requirements.txt          # Python dependencies
```

---

## License

This project is licensed under the MIT License — see the [LICENSE](LICENSE) file for details.
