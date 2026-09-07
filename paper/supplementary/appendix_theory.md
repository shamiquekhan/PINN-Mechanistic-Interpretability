# Theoretical Analysis: Activation Rank in Low-Dimensional-Input PINNs

**Status:** rigorous bound (tangent rank) + empirical diagnostic (covariance rank).
**Scope note:** this document deliberately separates what can be *proved* from
what can only be *measured*. An earlier draft of this plan claimed a covariance
effective-rank bound of `d + 1` via Whitney embedding; that claim is **false**
(a curved 1D manifold can have full affine span, so the covariance of points on
it is not rank-bounded by `d`). The correct statements are below.

---

## 1. Setup

Let `f_θ : R^d → R` be a PINN with smooth activation `σ ∈ C^∞` (tanh, GELU
smoothing, etc.). Write `h_ℓ(x) ∈ R^W` for the post-activation output of
hidden layer `ℓ` on input `x ∈ Ω ⊂ R^d`, and let `N` collocation/probe points
`x_1, …, x_N ∈ Ω` define the activation matrix
`A_ℓ = [h_ℓ(x_1); …; h_ℓ(x_N)] ∈ R^{N×W}`.

Two different "rank" objects appear in the literature and in our measurements;
they must not be conflated:

1. **Local tangent rank** — the rank of the Jacobian `J(x) = ∂h_ℓ/∂x (x) ∈
   R^{W×d}`, or empirically the rank of the matrix of local differences of
   activations along the input manifold.
2. **Covariance effective rank** — the participation ratio
   `PR = (Σ λ_i)² / Σ λ_i²` of the empirical covariance
   `C = (A − Ā)ᵀ(A − Ā)/(N−1)`.

## 2. Rigorous bound: local tangent rank ≤ d

**Proposition 1.** *If `h_ℓ` is differentiable at `x`, then
`rank J(x) ≤ d`.*

*Proof.* `J(x)` has `d` columns; rank cannot exceed column count. ∎

**Proposition 2 (chain of smooth maps).** *If `f_θ` uses `C^1` activations
and affine layers, then `h_ℓ = φ_ℓ ∘ … ∘ φ_1` with each `φ_i` a `C^1` map
between Euclidean spaces, and the image `h_ℓ(Ω')` of any `d`-dimensional
submanifold `Ω' ⊆ Ω` is contained in a `d`-dimensional `C^1` immersed
submanifold of `R^W` wherever the immersion has constant rank (by the
constant-rank level-set theorem; at rank-deficient points the image dimension
only drops).*

Consequently, for `d = 1` the activation states lie on a **curve** through
`R^W`; for `d = 2`, on a **surface**. This is the precise and provable
statement behind "a 1D-input PINN produces a 1-parameter family of activation
vectors."

**Empirical validation** (`analysis/activation_manifold.py::local_tangent_rank`,
with ordered probe coordinates and a least-squares local Jacobian):

| Input dim | PDE suite | Measured tangent rank |
|:---:|---|:---:|
| d=1 | Poisson/Advection/RD (W = 16…512) | **1** at every width |
| d=2 | Poisson2D | **2** |
| d=2 | AdvectionDiffusion2D, ReactionDiffusion2D | **2** |
| d=2 (t,x) | Burgers1D, AllenCahn1D | **2** |

The tangent-rank bound is exactly saturated in every benchmark — no hidden
layer manufactures more input-space directions than the domain has.

## 3. What cannot be proved: covariance rank

A smooth curve in `R^W` generically has full affine span (e.g., the moment
curve `(t, t², …, t^W)` has full-rank covariance). Therefore **no bound of the
form `PR ≤ d + 1` holds in general**, and Whitney embedding does not provide
one. The observed covariance PR ≈ 1.14–2.51 (1D suite, mean 1.78), ≈ 2.3–2.5 (steady 2D tasks) is
an *empirical property of trained PINN representations on this benchmark*:
training on a low-dimensional objective produces activation curves with
rapidly decaying singular spectra.

What we *can* say:

**Proposition 3 (finite-sample ceiling).** `rank(C) ≤ min(W, N − 1)`; the
measured `PR ≤ rank(C)` is bounded by the sample count whenever `N ≤ W`.

**Proposition 4 (no superposition hypothesis).** Superposition (Elhage et al.
2022) requires more *effective features* than *available dimensions*. If the
activation covariance has PR ≪ W (measured: PR/W = 0.02–0.05 across the PDE
suite), the network is encoding `~PR` effective features in `W` neurons with
dense (non-sparse) coefficients. An overcomplete sparse dictionary (the SAE
inductive bias) is mismatched to this regime: there are no superposed features
to un-mix, so the SAE's k-sparse reconstruction is strictly dominated by
PCA's optimal dense rank-k reconstruction — exactly the 12× reconstruction
gap and the 0/8 causal survivors we measure.

## 4. NTK connection

For an MLP with smooth activations, the NTK on a `d`-dimensional input
manifold inherits a spectral decay governed by the input dimension: kernel
eigenvalues decay polynomially with degree `d`-dependent rates (Bietti & Bach
2021; Geifman et al. 2020 show dimension-dependent spectral decay of NTK/CK
kernels). A low-rank NTK gram matrix on the collocation set implies gradient
updates concentrate on a low-dimensional subspace, which is consistent with —
though not a proof of — the observed low covariance rank of the trained
representation. We report this as consistency, not implication.

## 5. Regime boundary statement (the paper's claim)

Let `ρ = PR/W` (superposition ratio). Across this study:

| Representation | PR | W | ρ | SAE vs k-matched PCA | Verdict |
|---|---:|---:|---:|---|---|
| 1D PINNs (Poisson/Adv/RD) | 1.34–2.0 | 64 | 0.02–0.03 | PCA wins 12× | low-rank regime |
| 2D PINNs (Poisson/AD/RD) | 2.1–3.2 | 32 | 0.07–0.10 | PCA wins (2D pilot) | low-rank regime |
| Time-dependent PINNs (t,x) | 1.3–1.9 | 64 | 0.02–0.03 | (geometry only) | low-rank regime |
| FNO block states (Green's) | **6.8** | 64 | **0.106** | **SAE wins 4.7×** | superposition onset |
| LLM residual streams (lit.) | ~10³ | ~10³–10⁴ | ~0.1–0.9 | SAEs standard | superposition regime |

**Claim (empirical, scoped).** SAE-based interpretability transfers from LLMs
to scientific ML *only when the representation is genuinely high-rank*
(ρ ≳ 0.1 in our measurements). Below that, sparse dictionary learning
solves a nonexistent problem; linear (PCA/probe) methods are the correct
tools, and *causal* interpretability of any feature basis fails the
interchange criterion (Stage 9).

## 6. Implications for the guide's original theorem sketch

The original sketch ("effective rank ≤ d+1 by Whitney") is retained in the
repo history as a cautionary note: Whitney's theorem gives an *upper bound on
the embedding dimension needed to represent the manifold*, i.e., that a
d-manifold embeds in R^{2d+1} — it says nothing about the affine span of a
sampled point cloud, which is what covariance rank measures. Our final framing
avoids this overclaim and states only the two defensible results: the exact
tangent-rank bound (Prop. 1–2) and the empirical regime table (§5).
