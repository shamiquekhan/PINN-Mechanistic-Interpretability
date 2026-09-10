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
|  - SAEInterventionHook (8 Modes)     - Inference-Time Counterfactuals         |
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

The closed-loop adaptive PINN controller implements a 4-state deterministic state machine (`IDLE → WARNING → CONFIRMED → COOLDOWN → IDLE`; the bounded action fires **on entry into CONFIRMED**, when the confirmation count reaches `confirmation_steps` — there is no separate ACTING state):

```
       ┌───────────┐
       │   IDLE    │ ◄──────────────────────────────────────────────────────┐
       └─────┬─────┘                                                        │
             │ monitor_score >= alarm_threshold                              │
             ▼                                                              │
       ┌───────────┐                                                        │
       │  WARNING  │ ─── (score drops below threshold) ──► Reset Window     │
       └─────┬─────┘                                                        │
             │ confirmed for confirmation_steps consecutive steps           │
             ▼                                                              │
       ┌───────────────┐   bounded action applied on CONFIRMED entry        │
       │  CONFIRMED    │ ── (e.g. λ_bc rebalance, capped at max_lambda_bc)  │
       └─────┬─────────┘                                                        │
             ▼                                                              │
       ┌───────────┐                                                        │
       │ COOLDOWN  │ ── (rollback check: post-action L2 must beat            │
       └─────┬─────┘     pre-action × tolerance for degradation_patience,   │
             │            else REAL rollback restores pre-action λs)        │
             │ (after cooldown_steps) ──────────────────────────────────────┘
```

### Bounded Corrective Actions (per failure class)
1. **`increase_lambda_bc`** (boundary starvation, and — via the same bounded rebalance — spectral suppression since the v3.5 preregistration): multiplies $\lambda_{\text{bc}}$ by `bc_rebalance_factor` (4.0, capped at `max_lambda_bc`).
2. **`gradnorm_rebalance`** (gradient conflict): equalizes the two loss weights toward half their sum.
3. **`trigger_resample`** (collocation starvation): a TRIGGER event consumed by the training loop through the `PINNTrainer.train(resample_fn=...)` seam (the H3/H4 wiring — not a silent no-op).
4. **`trigger_fourier_features`**: REMOVED from the action space by the v3.5 preregistration (R2b pre-decision) — a half-wired input-dim-changing warm restart is not carried into generalization tests; implementing it is registered future work.
