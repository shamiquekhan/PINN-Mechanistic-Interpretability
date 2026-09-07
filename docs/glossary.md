# Glossary

Working definitions for terms used across this repository. Where a term
has a specific operational meaning here (vs its broader literature use),
the operational definition is given.

## Models and tasks

**PINN** — Physics-Informed Neural Network: an MLP trained to satisfy a PDE
residual at collocation points plus boundary/initial-condition terms,
rather than fitting data alone.

**PDE suite** — the eight families in `pinn/pdes.py`: Poisson 1D,
Advection 1D, Reaction-Diffusion 1D, Poisson 2D, Advection-Diffusion 2D,
Reaction-Diffusion 2D, viscous Burgers (inputs (t, x)), Allen-Cahn
(inputs (t, x)).

**FNO** — Fourier Neural Operator (Li et al. 2021); our dependency-free
1D implementation is `operators/fno.py`. Hidden *block states* are
per-(sample, grid-position) representations — the analogue of token-level
residual-stream activations in LLMs.

**Function-space regression** — the FNO task: input is a random
Gaussian-mixture forcing function, output its nonlinear spectral response
(Green's-function family). Inputs are genuinely high-dimensional objects
(infinite-dimensional function space), unlike PINN point inputs.

**Boundary starvation** — a PINN failure regime induced by λ_pde=100,
λ_bc=0.01: the boundary term is effectively ignored and the solution
violates its Dirichlet conditions (10/10 reproducible).

**Spectral suppression** — failure regime (narrow net W=16, source=50):
the solution captures low frequencies and aliases the high-frequency
content (10/10 reproducible).

**Gradient conflict / collocation starvation** — stress regimes kept in
the atlas but *excluded from failure-class claims* because their
operational labels do not reproduce across seeds.

## Representational geometry

**Participation ratio (PR)** — effective dimensionality of the activation
covariance C: PR = (Σλᵢ)² / Σλᵢ². PR ≈ number of dimensions carrying
non-negligible variance. Measured in
`analysis/effective_rank.py::participation_ratio`.

**Stable rank** — ‖C‖²_F / ‖C‖²₂, a second effective-rank diagnostic
(spectrally robust).

**Superposition ratio (ρ)** — PR / W (participation ratio divided by layer
width). This repository's operational regime parameter: PINN layers
measure ρ = 0.02–0.10; the FNO block states measure ρ = 0.106; LLM
residual streams are reported in the literature at ρ ≳ 0.1–0.9.

**Local tangent rank** — the rank of the Jacobian ∂h/∂x of the
input→activation map (estimated by least-squares local Jacobians on
ordered probe grids; `analysis/activation_manifold.py::local_tangent_rank`).
Provable bound: ≤ input dimension d. Measured: exactly d in every PDE
family and at every width — the bound is saturated.

**Superposition** — encoding more effective features than available
dimensions via sparse, nearly-orthogonal directions (Elhage et al. 2022).
Requires ρ not ≪ 1; absent here in every PINN layer measured.

**Low-rank regime** — the measured PINN condition: a smooth, effectively
1.5-dimensional activation curve (1D tasks) through a 64-dimensional
space; nothing to "un-mix."

## Interpretability machinery

**SAE** — Sparse Autoencoder: overcomplete (4×) dictionary trained to
reconstruct activations with sparse latents.

**TopK SAE** — SAE variant keeping only the k largest latent activations
per sample (k=8 here); the primary SAE of this study.

**Dead feature** — SAE latent that never activates on the evaluation
split. PINN SAEs: 51–70% dead (the dictionary has no real features to
find); measured differently on the FNO control (inactivity on test split).

**k-matched PCA** — PCA restricted to the same active budget (k=8) as the
TopK SAE — the fair linear baseline. Beats the PINN SAE 12× on
reconstruction; *loses* to the SAE 4.7× on FNO representations (the
regime boundary).

**Probe direction (control)** — a supervised logistic-regression direction
on latents/coefficients predicting the failure label; the AXBench-style
"simple linear baseline" control. If the probe matches a discovered
feature's causal effect, the feature adds nothing over supervision.

**Planted-feature positive control** — synthetic activations with a known
sparse causal feature run through the *identical* SAE + hook + intervention
+ scoring battery; PASSES (cosine 0.99, targeted Δ 15× controls),
proving the null is attributable to the PINN activations, not the
machinery.

## Causal testing

**E_T (causal strength)** — target effect minus the max of the control
effects: E_T = ΔL_T^target − max(ΔL_T^unrelated, ΔL_T^random,
ΔL_T^probe). E_T ≤ 0: the intervention does nothing beyond what any
matched perturbation does.

**S_spec (specificity)** — |target effect| / Σ|non-target effects|; ~1
means the intervention shifts only the target loss.

**Three-control protocol** — every causal claim must beat (1) an unrelated
feature/component, (2) a norm-matched random direction (per-row magnitude
matched), and (3) the supervised probe direction.

**Sign diagnostic** — if ablating *any* candidate increases the target
loss (all-positive deltas), the "features" are representational (ablation
damages reconstruction generically), not causal. Observed: 88/88 positive
for both SAE features and PCA components.

**Multiple-comparison corrections** — per-feature exact one-sided paired
sign tests across runs, then Bonferroni (α/8) and BH-FDR (q=0.05) across
the 8 candidates. Survivors: 0/8 in both batteries.

**Power / MDE** — minimum-detectable-effect: with 11 runs per feature, the
sign test needs ≥85.7% sign-agreement for 80% power at α=0.05.

**Causal abstraction** — (Geiger et al. 2021/2022) a high-level causal
model H is an abstraction of the network N if interventions on H
correspond, through an alignment map, to interventions on N.

**Interchange intervention** — replace a (part of a) hidden state from
input A with input B's, and test whether the output moves as H predicts.
Partial interchange: swap only the leading coefficients in a chosen basis.

**Movement fraction** — (log L_swap − log L_src)/(log L_don − log L_src);
≈1 means the swapped subspace fully carries the variable under test; the
causal-abstraction gate requires beating the random basis on every run.

## Monitoring and control

**Monitor (loss-only / conventional / SAE-augmented)** — early-warning
classifiers on past-only trajectory features; leakage-audited, evaluated
with run-level bootstrap CIs. Conventional AUROC 0.859; SAE adds nothing
(P(SAE better) = 0.53).

**Monitor-source ablation** — controller experiment where only the alarm
source differs (SAE features vs matched random projection): the rescue
trajectories are bit-identical, so SAE features are inert cargo.

**Controller** — deterministic state machine (IDLE→WARNING→CONFIRMED→
COOLDOWN with rollback) applying bounded corrective actions (λ_bc
rebalancing, GradNorm-style equalization, resampling, Fourier injection).

**SOTA baselines** — GradNorm (gradient-norm equalization), NTK-adaptive
weighting (λ ratio from gradient/NTK statistics), RBA (per-point residual
attention). On boundary starvation: NTK-adaptive 0.0064 beats the
controller 0.0166; GradNorm/RBA actively harm.

**Run-level split** — train/test splits at the whole-run level (never
per-sample), preventing trajectory leakage between monitor training and
evaluation.

## Misc

**rel L2** — relative L2 error ‖u_pred − u_exact‖/‖u_exact‖ on the
validation grid (computed on the exact solution's support for
time-dependent PDEs).

**Artifact** — a JSON/JSONL/checkpoint file in `runs/` produced by a
pipeline stage; the authoritative sources for figures/tables are listed in
`data/manifest.json`.
