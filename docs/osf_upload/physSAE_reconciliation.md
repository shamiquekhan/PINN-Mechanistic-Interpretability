# PhysSAE Reconciliation — Concurrent Work Analysis and Registered Follow-Up

**Written:** 2026-09-11, after the v3.5 campaign (H16–H19) closed at `7da7c92`.
**Concurrent work:** Patil, Eshwar R.A., Honnavar (2026), *"PhysSAE: Mechanistic
Interpretability with Sparse Autoencoders"*, arXiv:2609.07061v1 (submitted
2026-09-07). CC BY 4.0.
**Status of this document:** literature reconciliation + preregistration of the
v4.1 follow-up (R1–R5). No R-experiment has run; all hypotheses below are
pre-written, per the project's standing discipline.

---

## 1. What PhysSAE reports

Setup (their §2–3, Appendix E): 6 PDE families (heat, Burgers, Allen–Cahn,
convection easy β=1, convection hard β=30, Schrödinger — all 1+1D
space-time, domain [−1,1]×[0,1] or [−5,5]×[0,1]); PINNs 5 hidden layers ×
width 128, tanh, Adam 8000 + L-BFGS 300; 3 PINN seeds × 3 SAE seeds; SAEs
trained on **penultimate-layer** activations, z-scored, ReLU codes + L1
(λ=0.02), D=512 (4× overcomplete), unit-norm decoder columns, L0 ≈ 77–141
(15–28% of D); evaluation on a held-out 20% grid split; concepts defined
from **independent reference solutions** (analytic or pseudo-spectral),
never from the PINN.

Claims (their §4):
- **Alignment:** SAE atoms correlate with physical observables — max
  top-1 |r| per PDE 0.675–0.943 (mean of maxima 0.872; best: conv. hard
  0.951, Schrödinger 0.920, heat 0.890); permutation Z > 10 everywhere.
- **Spatial localization (their causal claim):** ablating a top-aligned
  atom changes the output field with a more spatially concentrated
  footprint than the same intervention on PCA/ICA components — ESF80
  SAE 0.165–0.476 vs PCA 0.546–0.745 (1.2–4.2× "speedup", mean 2.3×).
- **Negative controls:** top-aligned atoms vs matched-activation random
  atoms — ESF80 advantage mean +0.032 (Table 4: −0.025, +0.005, +0.043,
  +0.049, +0.125, −0.013).
- **Bilateral pairs:** two atoms (positive/negative halves) improve
  concept regression ΔR² = +0.030 to +0.146 (mean +0.086) over single
  atoms; random pairs decrease R² by up to 0.60.
- **Feature consistency:** cross-SAE-seed Hungarian-matched cosine
  0.331–0.371 mean (only 1–2% of atoms > 0.7) — dictionaries are
  non-unique, but their ESF80 is stable across seeds.
- **§4.8 Representational collapse:** *failed* PINNs (Allen–Cahn L2
  ≈ 0.93, conv. hard ≈ 0.95) have **higher-rank, more diffuse** hidden
  covariance; *well-trained* PINNs (heat 0.025, Schrödinger 0.119) have
  **lower-rank, more structured** representations. Alignment (not
  sparsity) degrades with failure — except conv. hard, which retains the
  highest alignment (0.951, characteristic coordinate x−30t).

## 2. What we report (v3.5 evidence state)

- Tested tanh PINNs (widths 16–512, depths 2–6, steady 1D/2D and
  time-dependent families): covariance participation ratio 1.14–2.51;
  tangent rank = input dimension everywhere; PR **falls** with depth
  (2.05 → 1.15, H18).
- SAE (TopK, k=8, 4× expansion) **and** k-matched PCA both fail the
  preregistered causal battery on those activations: 0/8 survivors on
  either basis, matched-deletion controls, planted-feature positive
  control passing; causal-abstraction interchange null (stages 5/8/9).
- FNO (function-space): PR 6.8 (PR/W 0.106) — SAE beats k-matched PCA
  reconstruction 4.7× (stage 11); operator causal battery closed at H16b
  (6/16 Bonferroni, 0/16 direction-reversal) — the causal side of the
  boundary was **not** found; the boundary claim rests on reconstruction.
- **H18 (within-PINN boundary):** Fourier-feature PINN (width 64,
  n_freq=32) on the same 1D Poisson task reaches PR 4.0–5.6 (PR/W 0.075)
  and there the same TopK SAE beats k-matched PCA **25–56×** on
  reconstruction (3/3 seeds); tanh depth sweep stays in the low-rank
  envelope; tangent rank stays 1 in every run (covariance/tangent
  dissociation).
- Strong directional associations DO exist in our activations (stage 4:
  r = 0.12–0.29 beating random 7–57×; a 0.93 data-std association
  identified as a run-type confound) — we classify alignment as
  insufficient for causal claims, not as absent.

## 3. The apparent contradiction, precisely stated

| Question | PhysSAE | This project |
|---|---|---|
| Do SAE features correlate with physical observables in PINNs? | Yes (max \|r\| 0.69–0.95) | Yes where tested (r 0.12–0.29 clean; 0.93 confounded) — never disputed |
| Does SAE ablation change the output? | Yes, with concentrated footprint (ESF80 vs PCA/ICA) | Yes trivially (any active direction moves the readout) — never disputed |
| Is the effect **specific** to the feature vs matched controls? | Mixed: mean advantage +0.032; negative on heat/Schrödinger (the well-trained PINNs) | **No** in the tested low-rank tanh regime: 0/8 survive matched-deletion + MC correction on SAE and PCA alike |
| Does SAE beat PCA at **reconstruction**? | (not the headline; recon MSE reported in ablations only) | Only in the higher-rank regimes: 25–56× (Fourier PINN, H18), 4.7× (FNO); ≈1× (tanh PINNs, paired CI spans zero) |
| Does training rank correlate with convergence? | Well-trained → lower rank (§4.8) | Converged tanh PINNs → low PR (1.14–2.51) |

**There is no direct numerical contradiction.** The studies ask different
causal questions, on different layers, with different dictionaries, on
partially different PDE regimes. Five confounds must be separated before
any head-to-head claim is possible (§4). One genuine convergence exists
(§5).

## 4. Confound decomposition (what R1 must control)

1. **Layer.** PhysSAE: penultimate layer (the linear-readout bottleneck;
   Δu = W_out Δh exactly). Ours: `layers.1` — for the default 3-hidden-layer
   nets the **middle** layer, one nonlinear block before the readout. Our
   causal battery propagates through the remaining network; theirs through a
   linear map. Penultimate-layer ablations have a structurally easier causal
   path.
2. **Dictionary.** PhysSAE: ReLU+L1, D=512, L0≈150 (~29% dense). Ours: TopK,
   k=8, D=256 (3% dense). A ~10× code-density difference changes what
   "sparse decomposition" even means; our H18 reconstruction result is
   TopK-specific until tested otherwise.
3. **Causal criterion.** Theirs: **spatial concentration** of the ablation
   footprint (ESF80). Ours: **effect-magnitude specificity** of the target
   direction vs matched-deletion controls (E_T, sign tests, MC correction).
   These are different properties. Critically, at the penultimate layer
   their intervention yields Δu(x,t) = α·z_k(x,t)·(W_out d_k) — the causal
   effect field is **exactly proportional to the atom's own activation
   field**. ESF80 of the intervention is therefore ESF80 of the code z_k
   itself, an activation-geometry statistic; it does not compare the
   *magnitude* of the target effect against what other directions do
   (except partially, via their random-atom controls).
4. **PDE/domain regime.** PhysSAE: all 1+1D time-dependent families
   (input dim 2). Ours: the causal nulls are on width-16–512 steady 1D/2D
   and short-schedule time-dependent families; our H18 rank contrast is on
   steady 1D Poisson. Their regime (rich (x,t) input structure, L-BFGS
   polish, w_BC=w_IC=100) may sit at a different point on the rank axis.
5. **SAE-evidence chain.** PhysSAE's controls are matched-activation random
   *atoms* (within-dictionary); ours are matched-norm random *directions*
   in activation space plus PCA and other active latents (across-basis).
   A within-dictionary null and an across-basis null can both be true.

## 5. The convergence point (both studies agree)

PhysSAE §4.8: *failed* PINNs are higher-rank/diffuse; *well-trained* PINNs
are lower-rank/structured. Our program: converged tanh PINNs sit at
PR 1.14–2.51 (H18 depth sweep shows PR falling as training structure
deepens). These are the **same regularity discovered from two directions**:
successful PINN optimization collapses activation covariance onto a
low-rank structured core. Neither study contradicts the other here; if
anything, PhysSAE's failed-PINN observation (higher rank when training
fails) and our Fourier observation (higher rank when the input embedding
is enriched) jointly suggest rank tracks *effective input richness under
convergence*, not architecture alone. This reframes both papers: the
scientific question is **when** — and the two studies supply complementary
sides of the boundary.

## 6. Terminology map (for the paper's related-work section)

PhysSAE's own discussion (§5) is more careful than its abstract: they
distinguish *alignment* (correlation), *functional relevance* (intervention
changes output), and *causal evidence* (structured intervention effect),
and explicitly disclaim "atoms are physical variables." Our level ladder
maps cleanly:

| PhysSAE term | Our level | Our verdict in low-rank tanh PINNs |
|---|---|---|
| Alignment (r, permutation Z) | L1 association | Present (both studies) |
| (bilateral ΔR²) | L1–L2 (predictivity) | Untested at their scale |
| Functional relevance (Δu ≠ 0) | L3 (trivially satisfied by any active direction) | Not treated as evidence |
| Spatial localization (ESF80) | L4 (effect-shape selectivity) | **The open question — R1** |
| — | L3-specificity vs matched deletion (E_T) | Null (0/8, both bases) |
| — | L5 direction-reversal (amplify/ablate) | Null on operators (H16b: 0/16) |

## 7. Internal-consistency notes on PhysSAE v1 (recorded factually)

For a v1 preprint, three things a replication must resolve:
1. The abstract's negative-control range "ESF80 advantage 0.04–0.44" does
   not match Table 4 (−0.025 to +0.125, mean +0.032).
2. The §2.8 sign convention ("a *negative* advantage means the top-aligned
   atom produces a more concentrated intervention... the result we
   expect") is contradicted by §4.4/Figure 4 ("Blue = atom more localised
   than random (positive result)"). Under the §2.8 convention, heat
   (−0.025) and Schrödinger (−0.013) — the two best-trained PINNs — are
   the *positive* results and the failed PINNs are not; under the §4.4
   convention the reverse. The replication must pre-register which
   convention it uses.
3. The mean top-5 alignment is promised in §2.6 but never reported.

These do not diminish the work's value; they make the head-to-head
experiment *more* necessary, and our replication will state every such
choice in advance.

## 8. Registered follow-up (v4.1 — R1–R5, pre-written, not yet run)

All outcomes below are publishable; none is preferred. Registered in
`docs/preregistration.md` §R1–R5 at the commit that first carries this
document. Machinery gates: the planted-feature positive control must pass
through every new hook (penultimate-layer extraction, ReLU+L1 SAE, ESF80
computation) before any verdict is read.

### R1 — PhysSAE head-to-head on frozen checkpoints (highest priority)

**Design.** One pipeline, both evaluation families, same frozen PINN
checkpoints (no retraining):
- **Substrates:** (i) our time-dependent families that overlap PhysSAE's
  benchmark (Burgers — partially converged; Allen–Cahn — failed ~0.93,
  matching their failure regime), retrained to their architecture spec
  (5×128 tanh, Adam+L-BFGS, w_BC=w_IC=100, 3 seeds) so the regime is
  matched, with **penultimate-layer** activation logging; (ii) our H18
  checkpoints (tanh depth-3 vs Fourier n_freq=32, 1D Poisson) with
  penultimate-layer re-extraction, as the within-task rank contrast.
- **Dictionaries per checkpoint:** ReLU+L1 SAE (D=512, λ=0.02, unit-norm
  decoder — PhysSAE spec) × 3 SAE seeds; TopK SAE (k=8, exp=4 — ours);
  PCA; ICA (FastICA, matched count); random matched directions.
- **Evaluations per dictionary:** (a) PhysSAE-style: concept-field
  alignment + permutation null, ESF80 localization, their negative
  controls (pre-registered sign convention: advantage =
  ESF80_random − ESF80_top, so **positive = top-aligned atom more
  concentrated**); (b) ours: matched-deletion E_T battery with MC
  correction; (c) reconstruction vs k-matched PCA.
- **Concept fields** from our independent reference solvers (same sources
  as stage 2/10), never from the PINN.

**Pre-written outcomes.**
- R1a: PhysSAE-style alignment and ESF80 advantage replicate on the
  matched-architecture Burgers/Allen–Cahn PINNs **and** our E_T battery
  also finds specificity there → the low-rank null is regime-bound to our
  earlier architectures/tasks; both papers' claims merge into a boundary
  story with the boundary placed by measurement.
- R1b: PhysSAE-style metrics replicate (alignment + ESF80 vs PCA) but our
  E_T specificity battery is null on the same checkpoints → the two
  "causal" criteria dissociate: spatial concentration of an exactly-linear
  ablation footprint is achievable without direction-specific effect
  magnitude. The interpretability level ladder becomes the paper's central
  methodological contribution, and both results stand at different levels.
- R1c: neither replicates on matched-architecture retraining → the
  difference is training/architectural regime, not methodology; the rank
  diagnostic arbitrates.
- R1d: PhysSAE-style metrics replicate only on the Fourier/high-rank
  checkpoints → strong regime-boundary confirmation; the reconciliation
  completes in our favor with PhysSAE as the high-rank positive example.

### R2 — Causal battery on the Fourier PINN (the missing third arrow)

The H18 chain is rank ↑ → SAE-reconstruction advantage; the causal arrow is
untested within PINNs. Run the preregistered matched-deletion E_T battery +
direction-reversal criterion (H16 machinery) on the Fourier-PINN
dictionaries (both ReLU+L1 and TopK). Pre-written: if reconstruction
advantage comes with causal specificity → the boundary covers
*interpretability*, not just compression; if not → "superposition is
necessary but not sufficient" becomes the headline and is itself the
reconciliation with PhysSAE (their evidence lives below the specificity
bar).

### R3 — Fourier frequency sweep (dose-response of the boundary)

n_freq ∈ {2, 4, 8, 16, 32, 64} on the fixed 1D Poisson task, 3 seeds,
measuring PR, PR/W, PCA-95, SAE-vs-PCA reconstruction (TopK + ReLU+L1),
and (where PR moves) the R2 causal battery. Pre-written: either the
SAE/PCA ratio rises smoothly with ρ = PR/W (quantitative phase diagram —
the strongest outcome), or it thresholds (a critical ρ, reported as such),
or it is non-monotone (recorded as the geometry being insufficient alone).

### R4 — SAE-seed robustness on the core Fourier condition

3 PINN seeds × 3 SAE seeds × {TopK, ReLU+L1} on the n_freq=32 condition;
Hungarian-matched cross-seed cosine (their §2.10 protocol) alongside our
metrics. Pre-written: dictionaries may be non-unique (their cosine ≈ 0.35)
while the regime-level verdicts (reconstruction advantage, E_T) are stable
— dissociating dictionary identity from regime conclusions, as their §5
already argues for ESF80.

### R5 — One additional PDE for the within-PINN boundary

tanh vs Fourier on one time-dependent family we already instrument
(Burgers, preferred: it spans partial convergence and has a shock concept
matching PhysSAE's localized-concept panel). 3 seeds. Pre-written both
ways: boundary replicates (external validity) or does not (scope honestly
drawn at steady tasks + operator regime).

### Explicitly deferred (v4.2+)

Wider FNO suite, Darcy/DeepONet, 4× causal-battery power upgrade (the
user-facing rationale: methodological mismatch is currently a larger
threat than statistical power — resolve R1/R2 first, then spend compute),
2D/3D domains, human-expert validation, real-physics domain study.

## 9. Implications for the v4.0 paper (binding)

1. PhysSAE **must** be cited as concurrent work, in a dedicated
   related-work subsection ("Concurrent SAE-based interpretation of
   PINNs"), with the §3–§6 comparison above summarized.
2. The paper's framing shifts from "SAEs fail on PINNs" (never our claim,
   but readable that way) to the registered thesis: *when* sparse feature
   representations support which levels of mechanistic interpretation, as
   a function of measurable representational geometry.
3. No claim may assert or imply that PhysSAE is wrong; where their v1 has
   internal tensions (§7), the paper may note them factually with
   citations, and must state our preregistered choices where conventions
   differ.
4. H17/H19 remain supporting contributions (controller generalization;
   monitor floor), unchanged by PhysSAE.
5. The v4.0 title moves to the "when" form (e.g., *"When Sparse Features
   Become Mechanistic: Representation Geometry and Causal
   Interpretability in Physics-Informed Neural Networks"*), per the
   standing note that the regime boundary — now measured within PINNs
   (H18) — is the contribution PhysSAE's concurrent result sharpens rather
   than threatens.
