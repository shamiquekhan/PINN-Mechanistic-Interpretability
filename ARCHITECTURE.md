# System Architecture & Design Specifications

This document outlines the detailed system architecture, module boundaries, state-machine transitions, and design constraints of the **PINN Mechanistic Interpretability** framework.

---

## 1. Modular Subsystem Design

```
+---------------------------------------------------------------------------------+
|                                 pinn/ Core Module                               |
|  - MLP Model & Fourier Embedding     - BasePDE Interfaces (Poisson/Advection/RD)|
|  - PINNTrainer Execution Engine      - ActivationLogger & Per-Loss Gradients    |
+---------------------------------------------------------------------------------+
                                         |
                                         v
+---------------------------------------------------------------------------------+
|                                  sae/ SAE Module                                |
|  - SparseAutoencoder (ReLU + L1 + L2) - Run-Partitioned ActivationDataset       |
|  - Dead Feature Tracker              - Cross-Seed Hungarian Feature Matcher     |
+---------------------------------------------------------------------------------+
                                         |
                                         v
+---------------------------------------------------------------------------------+
|                            interventions/ Engine Module                         |
|  - SAEInterventionHook (7 Modes)     - Inference-Time Counterfactuals         |
|  - Detached Weight Autograd Graph    - Causal Scoring (E_T & S_spec)            |
+---------------------------------------------------------------------------------+
                                         |
                                         v
+---------------------------------------------------------------------------------+
|                            monitoring/ Monitoring Module                        |
|  - Past-Only Window Feature Builder  - Leakage Audit Auditor                    |
|  - ThresholdMonitor Alarm Detector    - Logistic Early-Warning Classifier      |
+---------------------------------------------------------------------------------+
                                         |
                                         v
+---------------------------------------------------------------------------------+
|                            controller/ Closed-Loop Module                       |
|  - Deterministic State Machine       - Bounded Corrective Action Handlers       |
|  - Rollback & Degradation Tracker     - Dynamic PINN Rescue Loop                 |
+---------------------------------------------------------------------------------+
```

---

## 2. Design Guarantees & Constraints

### A. Zero Temporal Leakage
All monitoring features and trajectory statistics strictly enforce past-only information windows.
For any prediction or alarm step $t$, the feature matrix $X_t$ uses only metric history from steps $[t - H, t)$.
The `leakage_audit` utility programmatically verifies that no future metrics or failure labels infect past trajectory representations.

### B. Autograd Graph Maintenance
The intervention engine (`interventions/engine.py`) uses a forward hook `SAEInterventionHook` that intercepts intermediate hidden activations:
$$h' = W_d \left( \text{Mode}\left( \text{Encoder}(h) \right) \right) + b_d$$
To prevent PyTorch `RuntimeError: Trying to backward through the graph a second time` during gradient descent optimization, all SAE weights ($W_e, b_e, W_d, b_d$) are explicitly frozen and detached from autograd gradients.

### C. Deterministic Reproducibility
All stochastic ops (collocation sampling, weight initializations, SAE training split shuffles) support explicit PyTorch generators (`torch.Generator`) and environment flags for deterministic CuBLAS execution (`CUBLAS_WORKSPACE_CONFIG=:4096:8`).

---

## 3. Closed-Loop Controller State Transitions

The closed-loop adaptive PINN controller implements a 5-state deterministic state machine:

```
       ┌───────────┐
       │   IDLE    │ ◄──────────────────────────────────────────────────────┐
       └─────┬─────┘                                                        │
             │ monitor_score > warning_thresh                               │
             ▼                                                              │
       ┌───────────┐                                                        │
       │  WARNING  │ ─── (score drops below threshold) ──► Reset Window     │
       └─────┬─────┘                                                        │
             │ confirmed for N consecutive steps                            │
             ▼                                                              │
       ┌───────────┐                                                        │
       │ CONFIRMED │                                                        │
       └─────┬─────┘                                                        │
             │ Select bounded corrective action                             │
             ▼                                                              │
       ┌───────────┐                                                        │
       │  ACTING   │ ─── Apply Action (e.g. BC Reweight / GradNorm)         │
       └─────┬─────┘                                                        │
             │ Action applied                                               │
             ▼                                                              │
       ┌───────────┐                                                        │
       │ COOLDOWN  │ ─── (after cooldown_steps) ────────────────────────────┘
       └───────────┘
```

### Bounded Corrective Actions
1. **`increase_lambda_bc`**: Multiplies boundary weight $\lambda_{\text{bc}}$ by 2.0 (up to a ceiling of $100.0$).
2. **`gradnorm_rebalance`**: Equalizes PDE and boundary loss gradient magnitudes:
   $$\lambda_{\text{bc}} \leftarrow \lambda_{\text{pde}} \cdot \frac{\|\nabla_\theta L_{\text{pde}}\|}{\|\nabla_\theta L_{\text{bc}}\|+\epsilon}$$
3. **`resample_high_residual`**: Resamples collocation points focused in spatial areas exceeding residual percentiles.
4. **`inject_fourier_features`**: Dynamically enables Fourier feature preprocessing embedding to resolve high-frequency spectral failure.
