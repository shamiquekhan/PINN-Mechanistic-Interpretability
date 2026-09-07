# Complete Framework Documentation & User Guide

This document provides a reproducible API guide and step-by-step user manual for running experiments, training TopK SAEs, building feature dictionaries, running controlled interventions, evaluating monitors, and activating closed-loop adaptive control. The current evidence and limitations are summarized in [RESULTS.md](RESULTS.md).

---

## 1. Running PINN Training Experiments

Training a PINN model uses the `experiments.train` entry point with a validated YAML configuration file:

```bash
python -m experiments.train \
  --config configs/poisson_baseline.yaml \
  --steps 5000 \
  --log-gradients \
  --log-activations \
  --act-save-raw \
  --log-diagnostics
```

### Command-Line Arguments
- `--config`: Path to YAML experiment configuration file (required).
- `--steps`: Total optimization steps (overrides config if specified).
- `--log-gradients`: Enables tracking of per-loss component gradient norms and cosine similarity.
- `--log-activations`: Registers forward hooks to sample and log intermediate layer activations.
- `--act-save-raw`: Saves raw activation vectors to `activations.jsonl` for SAE training.
- `--log-diagnostics`: Computes spatial residual profiles and error statistics.

The first dimensional-boundary prototype is a manufactured 2D Poisson problem:

```bash
python -m experiments.train \
    --config configs/poisson_2d_boundary.yaml \
    --steps 1000 \
    --log-activations \
    --act-save-raw
```

It uses `input_dim: 2`, rectangular `domain_y`, two-dimensional interior samples, perimeter boundary points, and a fixed 20x20 probe grid. The 2D diagnostics path is intentionally limited to training, validation, and activation geometry; the existing 1D spatial diagnostics remain unchanged.

---

## 2. Qualification Gate Suite

The qualification gate evaluates baseline success and failure modes across multiple random seeds:

```bash
python -m experiments.qualification
```

This runs a 10-seed matrix ($[7, 42, 123, 2024, 2025, 2026, 999, 888, 777, 555]$) across:
1. `success_baseline`
2. `boundary_starvation`
3. `gradient_conflict`
4. `spectral_suppression`
5. `collocation_starvation`

The run uses activation, gradient, and diagnostic logging so the failure atlas can verify operational labels. Output results and relative $L_2$ error statistics are written to `runs/qualification/qualification_results.json`. Boundary-starvation and spectral-suppression reproduce cleanly; gradient-conflict and collocation-starvation are retained as stress tests but excluded from final failure-class claims.

---

## 3. Sparse Autoencoder (SAE) Training & Sweeps

Train TopK Sparse Autoencoders on logged activation datasets using `sae.train`:

```python
from sae.train import sweep_sae
from pathlib import Path
import torch

run_dirs = [Path("runs/poisson_baseline"), Path("runs/advection_1d_baseline")]
device = torch.device("cuda")

sae_models = sweep_sae(
    run_dirs=run_dirs,
    layer_name="layers.1",
    expansion_values=[2, 4],
    sparsity_values=[1e-3, 1e-2],
    steps=500,
    batch_size=128,
    learning_rate=1e-3,
    device=device,
    out_dir=Path("runs/sae_models_v2")
)
```

For the locked benchmark pipeline, use `python -m experiments.run_pipeline --stages 3`; it trains TopK replicas and records PCA and frozen-random baselines. Run `python -m analysis.effective_rank --runs runs` before interpreting SAE results to measure the activation participation ratio.

---

## 4. Building the Physics-Feature Dictionary

Construct the Physics-Feature Dictionary from trained SAE weights and logged trajectories:

```python
from sae.features import (
    compute_feature_stats,
    associate_features_with_metrics,
    match_features_across_saes,
    build_feature_dictionary,
    save_feature_dictionary
)
from sae.model import SparseAutoencoder
from sae.dataset import ActivationDataset
from analysis.feature_dictionary import render_feature_dictionary_markdown
from pathlib import Path
import torch

device = torch.device("cuda")
sae = SparseAutoencoder.load("runs/sae_models/exp2_sp1e-03/sae.pt", device)
ds = ActivationDataset([Path("runs/poisson_baseline")], "layers.1", normalise=True)

stats = compute_feature_stats(sae, ds.data, device)
metric_arrays = {"data_std": ds.data.cpu().numpy().std(axis=1)}
corrs = associate_features_with_metrics(sae, ds.data, metric_arrays, device)

dictionary = build_feature_dictionary(
    feature_stats=stats,
    correlations=corrs,
    families=[],
    layer_name="layers.1",
    sae_version="v1.0"
)

save_feature_dictionary(dictionary, Path("runs/feature_dictionary/physics_feature_dictionary.json"))
render_feature_dictionary_markdown(dictionary, Path("runs/feature_dictionary/physics_feature_dictionary.md"))
```

---

## 5. Causal Interventions & Counterfactual Evaluation

Run inference-time causal interventions (`ablate`, `amplify`, `unrelated_control`, `random_direction`, and supervised `probe_direction` controls):

```python
from interventions.causal import run_inference_interventions
from sae.model import SparseAutoencoder
from pinn.model import MLP
from pinn.pdes import Poisson1D
import torch

device = torch.device("cuda")
pde = Poisson1D()
model = MLP(1, 1, [64, 64, 64], "tanh").to(device)
sae = SparseAutoencoder.load("runs/sae_models/exp2_sp1e-03/sae.pt", device)

scores = run_inference_interventions(
    model=model,
    sae=sae,
    pde=pde,
    device=device,
    dtype=torch.float32,
    candidate_features=[0, 1, 2, 3],
    target_loss="pde",
    layer_index=1
)
print("Causal scores:", scores)
```

---

## 6. Closed-Loop Adaptive PINN Controller

Attach the controller to a PINN training loop:

```python
from controller.state_machine import PINNController, ControllerConfig
from monitoring.models import ThresholdMonitor

ctrl_cfg = ControllerConfig(alarm_threshold=0.5, confirmation_steps=2, cooldown_steps=100)
controller = PINNController(ctrl_cfg, out_dir=Path("runs/controller_demo"))
monitor = ThresholdMonitor(plateau_window=10, degradation_factor=2.0)

# Inside PINN training loop:
score = monitor.update(total_loss_val, rel_l2_val)
lambda_pde, lambda_bc, event = controller.step(
    step=step,
    monitor_score=score,
    rel_l2=rel_l2_val,
    lambda_pde=lambda_pde,
    lambda_bc=lambda_bc,
    failure_class="boundary_starvation"
)
```

---

## 7. Master Research Campaign Execution

To run all 15 stages sequentially in a single automated pipeline:

```bash
python -m experiments.run_pipeline
```

Individual stages are also available for reproducible reruns:

```bash
python -m experiments.run_pipeline --stages 2   # failure atlas and seed statistics
python -m experiments.run_pipeline --stages 3   # SAE, PCA, and random baselines
python -m experiments.run_pipeline --stages 4   # feature dictionary
python -m experiments.run_pipeline --stages 5   # causal battery and positive control
python -m experiments.run_pipeline --stages 6   # monitor AUROC/AUPRC and CIs
python -m experiments.run_pipeline --stages 7   # controller and source ablation
python -m experiments.run_pipeline --stages 8   # PCA causal battery (head-to-head vs SAE)
python -m experiments.run_pipeline --stages 9   # causal abstraction (partial interchange)
python -m experiments.run_pipeline --stages 10  # 2D suite + time-dependent geometry
python -m experiments.run_pipeline --stages 11  # FNO operator regime boundary
python -m experiments.run_pipeline --stages 12  # SOTA baselines vs controller
python -m experiments.run_pipeline --stages 13  # statistical hardening
```

## 8. Verification

Run the project checks with the same interpreter used for installation:

```bash
python -m compileall -q analysis controller experiments interventions monitoring pinn pinn_logging sae tests
python -m pytest -q
```

The current baseline is 98 passing tests. On CUDA, set `CUBLAS_WORKSPACE_CONFIG=:4096:8` before deterministic runs when exact cuBLAS reproducibility is required.
