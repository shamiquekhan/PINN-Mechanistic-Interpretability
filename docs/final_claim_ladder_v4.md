# Final Claim Ladder (v4.1 evidence state)

**Written:** 2026-09-11, after H16–H19 + R1–R2 (R3 in flight). The
binding wording of every major claim, each with its evidence chain
(artifact → checksum → registered protocol). This document is the
reference for the paper, README, and any release communication: claims
must not be strengthened beyond this wording.

## Level ladder (the paper's methodological spine)

- **L1 Association:** a feature correlates with an observable.
- **L2 Predictivity:** a feature (or pair) predicts an observable
  beyond chance.
- **L3 Effect-magnitude specificity:** intervening on the feature
  changes a target readout MORE than matched-deletion controls
  (our E_T criterion, MC-corrected).
- **L4 Effect-shape selectivity:** the intervention's spatial/temporal
  footprint is more structured than controls (PhysSAE's ESF80).
- **L5 Mechanistic consistency:** amplification reverses the ablation
  effect with consistent direction (our crossover criterion).

Evidence at a level does not imply the levels above it.

## Binding wording rule (all docs, paper, and release communication)

The negative causal results are stated as:

> **"No tested feature satisfied the preregistered direction-specific
> causal criterion."**

Never as "no causal effect exists" or "features have no causal
effect": the intervention batteries measure *nonzero* effects (all
ablation deltas are positive; E_T CIs exclude zero on the negative
side — the representational signature); what is absent is the
**specific causal signature the preregistered criteria define**
(target-above-matched-deletion, direction reversal, dose-response
shape). The shorthand "causal null" is permitted ONLY where the full
sentence (or its citation) appears in the same document, and never in
an abstract, title, or claim of record.

## Claim 1 — Geometry of conventional PINNs (strong)

The tested tanh PINNs (widths 16–512, depths 2–6, eight PDE families,
steady and time-dependent) occupy a low-effective-covariance-rank
activation regime: PR 1.14–2.51, falling monotonically with depth
(2.05 → 1.15); local tangent rank exactly equals input dimension
everywhere (the provable bound, saturated). PR is *effective covariance
dimensionality*, not manifold dimension — the two dissociate (Claim 4).

Evidence: stage-3/10 artifacts; H18 depth arm (RESULTS §5A.10).

## Claim 2 — The causal null, low-rank regime (strong)

In that regime, SAE features carry no effect-magnitude causal
information: E_T = −0.0173 [−0.0329, −0.0037] under matched-deletion
controls; 0/8 survive Bonferroni or BH-FDR; the planted-feature
positive control passes (cosine 0.989, 9.2× its control), so the null
is attributable to the activations, not the pipeline. Directional
associations exist (L1) — that is not disputed and never was.

Evidence: stage 5 (`runs/causal_intervention_results.json`).

## Claim 3 — Basis-independence (strong)

The identical battery on PCA components: 0/8 survive; paired PCA−SAE
difference spans zero; no candidate alignment (PCA or SAE) beats a
random basis in causal-abstraction interchange. No feature basis,
linear or sparse, carries L3 specificity in these activations.

Evidence: stages 8/9.

## Claim 4 — The boundary is within PINNs (strong; the centerpiece)

A Fourier-feature PINN (width 64, n_freq=32) on the same 1D Poisson
task crosses into a higher covariance-rank regime (PR 4.0–5.6, PR/W
0.075; the FNO sits at 6.8/0.106) — and there the SAME TopK SAE that
is null on tanh PINNs beats k-matched PCA by 25–56× (3/3 seeds) in
reconstruction. Tangent rank stays 1 everywhere: Fourier features
raise covariance rank, not manifold dimension — the two measures
dissociate, and covariance PR is the regime discriminator. The
preregistered rank diagnostic (PR > 3.0, calibrated on the measured
width-scaling envelope) predicted which runs would flip. Depth does
NOT rescue the regime — it lowers PR further.

Scope: one task, one embedding family, one width, three seeds. PR > 3
is this protocol's operational bar, not a universal constant.

Evidence: H18 (`runs/architecture_boundary/architecture_boundary_report.json`);
figure 10; R3 (dose-response, in flight).

## Claim 5 — The causal arrow does not close (strong; completes the chain — now at curve level, R6)

Even in the high-rank Fourier regime — with the 25–56× reconstruction
advantage — no tested feature satisfied the preregistered
L3/L5 criteria: E_T = −15.89 [−17.46, −14.27]
(target ablations move the residual LESS than matched-deletion
controls); 4/45 Bonferroni survivors, all negative-side; 0/45 satisfy
the L5 crossover criterion; the planted control passes at the same n.
The chain rank → reconstruction advantage → causal specificity has its
first two arrows measured and the third consistently absent across
every regime evaluated (stages 5/8; H16b operators; R2b Fourier).

**R6 (dose-response) closes the shape question the two-point criteria
could not ask:** across α ∈ {−2, −1, 0, 0.5, 1.5, 2} with
dose-matched closest-activity controls, 0 features in any high-rank
regime (Fourier SAE/PCA 0/24+0/24; FNO 0/8) show a signed causal
channel; the response is dose-symmetric (the reconstructive
signature) at same-sign fraction 1.00, and the control curves match
it. The machinery gate's own correction is part of the finding: the
planted readout is a *quadratic* channel — a causal-by-construction
feature has same-sign endpoints — so same-sign endpoints alone are
not reconstructive evidence; the control-matched curve is the
discriminator, and nowhere in the tested regimes does a target curve
separate from its control the way the planted curve does (Fourier
ctrl-flat margins 1.5–3.6× vs the planted 67×). Scope annotation: two
shape-only R6a verdicts exist on inert-scale tanh-PCA components
(max |Δ| ≈ 1e-6 vs the arm's O(1) effects) — the class is shape-only
by registration; no signed channel co-occurs with the high-rank regime
at any magnitude.

**In the tested models, entering the higher-effective-rank regime was
necessary for the observed SAE-over-PCA reconstruction advantage, but
that advantage was not sufficient for direction-specific causal
interpretability.**

Evidence: R2 (`runs/r2_fourier_causal/r2_report.json`);
R6 (`runs/r6_dose_response/r6_report.json`).

## Claim 6 — PhysSAE reconciliation (strong; measured head-to-head)

On matched-architecture checkpoints (their spec: 5×128, Adam+L-BFGS,
w_BC=w_IC=100, 3 seeds), their evidence family replicates but is
generic: concept alignment appears for every basis including random
directions (max |r| 0.85–0.96, Z > 16); SAE ablation footprints are
2.0–2.5× more ESF80-concentrated than PCA/ICA — and equally more
concentrated than random unit directions (2.0–2.3×). The L4 statistic
is a property of any selective code at the penultimate layer (their
Δu = α·z_k·(W_out d_k) is exactly proportional to the code), not of
discovered physics; the between-basis random control is the missing
arm. No basis satisfied our L3 criterion on the same checkpoints (random sits at the 0/8 chance floor, so the battery is calibrated).
Both programs independently discovered the rank-collapse regularity
(their §4.8: failed PINNs higher-rank; our sweeps: converged PINNs
lower-rank). Their v1 internal tensions (abstract vs Table 4 numbers;
§2.8/§4.4 sign contradiction; unreported top-5) are recorded
factually; our replication preregistered its conventions.

Scope caveats: our matched Burgers converged (0.010–0.016) where theirs
plateaued (0.207); our Allen–Cahn is the ε=0.05 family, not ε=10⁻⁴.

Evidence: R1 (`runs/r1_physSAE/r1_report.json`);
`docs/physSAE_reconciliation.md`.

## Claim 7 — Controller (supporting; honestly bounded)

The closed-loop controller rescues boundary starvation to oracle level
(0.421 → 0.0002, config-faithful protocol), WINS its home failure
class (0.000119 vs NTK-adaptive 0.0044, 3 seeds), never degrades
catastrophically across the three-failure battery (worst mean Δ vs
no-action +0.019), and loses to NTK-adaptive elsewhere (spectral
0.335 vs 0.0036 with high per-seed variance; RD-2D 1.98 vs 1.67 —
where EVERY method including the oracle fails: the task is unlearned at
2500 steps). GradNorm actively harms (3.82 on starvation). The
registered reading: robustness (never-catastrophic + home-class
dominance), NOT per-failure SOTA. SAE features are inert cargo
(monitor-source ablation: bit-identical trajectories).

Evidence: stage 7/12; H17 (`runs/controller_failure_battery/`).

## Claim 8 — Monitoring (supporting; floor corrected)

The conventional monitor discriminates held-out failures (AUROC 0.875
[0.814, 0.960]) — but an honest loss-trajectory floor reaches 0.786
[0.615, 0.898], closing 78.2% of the threshold→conventional gap: most
of the apparent margin was the rule (threshold vs trajectory), not the
features. No SAE advantage (P = 0.51). Label provenance is audited
per-run: the stage-10 RD-2D runs never entered training; 2
init-transient artifact labels excluded; collocation derives no
failure step (never tested).

Evidence: stage 6; H19 (`runs/monitor_audit/monitor_audit_report.json`).

## The one-line takeaways

- **For practitioners:** compute PR/W before training any SAE on a
  scientific model; low ρ ⇒ expect ≈PCA, do not expect L3 causal
  specificity from any basis; high ρ ⇒ expect SAE reconstruction
  advantage (L0), still not L3 (R2b).
- **For the literature:** alignment and footprint concentration (L1/L4)
  are achievable by generic bases; claims of SAE-specific mechanism
  require L3/L5 evidence with matched-deletion controls — which no
  regime tested here provides.
