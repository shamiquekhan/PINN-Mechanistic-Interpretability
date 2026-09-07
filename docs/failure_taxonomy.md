# Failure Taxonomy

Operational definitions of the PINN failure modes in the failure atlas
(`analysis/failure_atlas.py`, stage 2). "Operational" means every label is
assigned by a measurable trigger on the logged trajectory — no labels are
hand-assigned. Reproducibility across seeds is a *property of the regime*
and is reported separately (10-seed statistics in
`runs/failure_atlas/seed_matrix_stats.json`).

```
Failure
├── boundary_starvation       (reproducible: 10/10)
├── spectral_suppression      (reproducible: 10/10)
├── gradient_conflict         (label NOT reproducible → excluded from class claims)
└── collocation_starvation    (mostly does not fail → excluded from class claims)
```

---

## boundary_starvation

| Field | Definition |
|---|---|
| **Trigger (config)** | λ_pde = 100, λ_bc = 0.01 (`configs/failure_boundary_starvation.yaml`) |
| **Observable** | loss_bc plateaus ≫ 0 while loss_pde → 0; final rel L2 ≈ 2.31 ± 1.59 (CI [1.31, 3.40]) |
| **Expected trajectory** | interior residual satisfied, Dirichlet conditions violated; solution drifts to the unforced PDE family |
| **Physics interpretation** | the boundary term is weighted out of the gradient signal: the collocation objective admits solutions outside the BC-compatible family |
| **Detection** | loss_bc stagnation + loss_pde/loss_bc ratio explosion (monitors pick this up: conventional AUROC 0.859) |
| **Intervention** | λ_bc rebalancing (controller action; NTK-adaptive weighting does this best: 0.0064 final) |
| **Role in this study** | the causal-battery evaluation class — the regime where SAE/PCA features were hypothesized to encode the failure |

## spectral_suppression

| Field | Definition |
|---|---|
| **Trigger (config)** | narrow net W=16, source=50 (`configs/failure_spectral_suppression.yaml`) |
| **Observable** | error spectrum dominated by high-frequency power; final rel L2 ≈ 0.395 ± 0.114 |
| **Expected trajectory** | the network fits the smooth part of the solution first (spectral bias) and aliases the sharp part |
| **Physics interpretation** | frequency-domain under-capacity: tanh MLPs learn low modes first; large source amplitude makes high modes load-bearing |
| **Detection** | `high_freq_power` in the diagnostics spectrum (dictionary view #2, r = 0.24 with best SAE latents — representational only) |
| **Intervention** | Fourier-feature injection (controller action) |
| **Role in this study** | second reproducible regime; confirms the geometry result is not Poisson-specific |

## gradient_conflict *(excluded from class claims)*

| Field | Definition |
|---|---|
| **Trigger (config)** | learning rate η = 0.1 (`configs/failure_gradient_conflict.yaml`) |
| **Observable** | oscillating losses; pde-vs-BC gradient cosine swings negative |
| **Physics interpretation** | the two loss terms pull shared parameters in opposite directions; large steps amplify the conflict |
| **Detection** | gradient cosine (`grad_cosine` dictionary view, r = 0.75 with best latent — the strongest association, still representational) |
| **Intervention** | GradNorm-style equalization (controller action) |
| **Exclusion reason** | across 10 seeds: 8 spectral / 1 unlabeled / 1 success — the *label* does not reproduce; runs remain in the atlas and in monitor training data, but no failure-class claim is made |

## collocation_starvation *(excluded from class claims)*

| Field | Definition |
|---|---|
| **Trigger (config)** | N_interior = 4, boundary bias 0.8 (`configs/failure_collocation_starvation.yaml`) |
| **Observable** | expected under-sampling failure; measured final rel L2 ≈ 0.0098 ± 0.0116 — the configured stress does *not* reliably break the solution |
| **Physics interpretation** | with heavy boundary weighting the BC term itself constrains the solution enough to compensate |
| **Exclusion reason** | 7 success / 3 unlabeled across seeds; kept as a negative-control configuration |
| **Role in this study** | demonstrates the qualification gate catches configurations whose *intended* failure does not materialize |

---

## Label assignment rules (operational)

An atlas entry receives a class label when the logged trajectory satisfies
the measurable signature of that class (see `analysis/failure_atlas.py`
for the exact predicates). Two discipline rules used throughout:

1. **No label survives in the claim ladder unless it reproduces across the
   10-seed matrix** (boundary starvation and spectral suppression qualify;
   the other two do not).
2. **Success-threshold sensitivity is reported** (stage 13): failure
   fractions for the two reproducible regimes are invariant across
   thresholds 0.02–0.20; the borderline `success` regime is
   threshold-sensitive and therefore carries no class-level claim.
