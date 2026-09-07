# Revision-Gap Audit — "Mechanistic Interpretability for PINNs" Review

**Date:** 2026-09-08
**Input:** a four-gap revision guide addressed to the original proposal.
**Purpose:** determine, artifact by artifact, which of the four gaps the
executed v3.2 campaign already closes, and which remain genuinely open.
Historical note: the proposal reviewed here predates the campaign; the
campaign itself was shaped by the same concerns (mandatory baselines,
preconditions, causal controls), so the audit below is largely a
verification exercise — with one real exception (Gap 2).

---

## Gap 1 — Superposition precondition: **CLOSED (and then some)**

The guide's "Phase 0" (sparsity logging, SAE-vs-PCA, pre-registered
falsification rule) is precisely what stages 3/10/11 executed, at larger
scale than requested:

| Guide's Phase-0 element | Campaign status | Evidence |
|---|---|---|
| Train small PINN with known failure mode | done, 10 seeds × 5 regimes + 8 families | `runs/failure_atlas/seed_matrix_stats.json` |
| Activation sparsity / histograms per layer | done (activation statistics + spatial channels logged every 100 steps) | `runs/<config>/activations.jsonl` |
| SAE with PCA baseline of same rank | done — **PCA(k=8) beats the SAE ≈50×** (fresh scale) | `runs/sae_models_v2/sae_stage3_summary.json` |
| L0 vs neuron count | done — TopK k=8 forces L0=8; dead latents 47–62% show the dictionary is *not* finding structure | stage-3 history `mean_l0`, `dead_feature_frac` |
| Pre-registered falsification rule | done — this is the participation-ratio precheck the paper recommends | `docs/theory_activation_rank.md`, RESULTS §5 |

The deeper version of the guide's concern — "a PINN layer is dense,
smooth, low-feature-count; superposition shouldn't apply" — is not just
acknowledged but is **the paper's central finding**, now measured on both
sides of the boundary:

- PINN activations: PR 1.78 of 64 (ρ=0.028), tangent rank = input
  dimension exactly across all 8 families and widths 16–512
  (`runs/effective_rank_analysis/`, `runs/width_scaling/`,
  `runs/dimensional_boundary_expanded/`).
- The superposition regime *does* exist in scientific ML: FNO block
  states PR 6.8 of 64 (ρ=0.106), where the same SAE beats k-matched PCA
  4.7× (`runs/operator_boundary/`), with a suggestive (not confirmed)
  causal asymmetry from stage 14
  (`runs/operator_causal/operator_causal_report.json`).

The guide's "if the test comes back negative, reframe as a
negative/boundary result" instruction: **that is the published framing.**

## Gap 2 — NTK↔SAE bridge: **GENUINELY OPEN → now preregistered as Stage 15**

The criticism is correct: no campaign stage ever constructed the
mathematical object connecting NTK eigenstructure (function/parameter
space) to SAE features (activation space). The campaign measured
gradient-pathology *statistics* (per-term norms, pde-vs-bc cosine) and
SAE feature *effects*, but never the bridge between them.

All ingredients exist and are committed:

- Per-step gradient norms and pde-vs-bc cosines:
  `runs/<config>/gradients.jsonl` (steps 0–999, every 100; conflict
  scores observed range 0.0–0.72 on boundary starvation).
- Fixed deterministic probe-grid activations (same coordinates every
  log step; single probe hash): `runs/<config>/activations.jsonl`,
  `raw` = 50×64 per step.
- Trained SAE + decoder directions: `runs/sae_models_v2/`.
- The Wang et al. (2022) NTK quantities are already computed in stage
  12's NTK-adaptive baseline (`experiments/sota_baselines.py`).

**Preregistered closure (Option A of the guide — the cheap correlational
bridge first):** hypothesis pair **H15a/H15b** in
`docs/preregistration.md` §H8, decision rules committed before the run
(stage-14 discipline). Concretely: do high-conflict training steps
(negative pde-vs-bc gradient cosine — Wang et al.'s pathology criterion)
coincide with elevated activity of specific SAE features on the fixed
probe grid, beyond what a random dictionary exhibits? Stage 15
(`experiments/ntk_bridge.py`) measures exactly this, with a
feature-specificity gate so the result cannot be driven by global
activity surges. Option B (NTK-eigenmode-projected activations) is
registered as future work.

## Gap 3 — Spectral-bias / F-Principle baseline: **MOSTLY CLOSED; citations now added**

The guide's demand — "benchmark the SAE against the Fourier/NTK
diagnostic it is implicitly competing with" — is satisfied in structure:

| Guide's element | Campaign status | Evidence |
|---|---|---|
| Fourier decomposition of hidden-unit responses | done — the diagnostics spectrum is a logged run artifact (`fourier_spectrum.png`, `high_freq_power` dictionary view) | `runs/<config>/`, stage 4 |
| NTK eigenspectrum analysis | done — stage 12 computes NTK ratios per loss term; stage 5's probe control is a supervised spectral readout | `runs/sota_baselines/`, `interventions/causal.py::train_failure_probe` |
| Head-to-head: does SAE add value beyond the spectral baseline? | done — the probe control (a linear spectral direction) matches or beats SAE features in the causal battery (fresh: probe beats target in 8% SAE / 45% PCA of evaluations); stage 14's non-target readout is literally the dominant Fourier-mode magnitude | `runs/causal_intervention_results.json`, `runs/operator_causal/` |
| "If SAE only recovers what Fourier analysis shows, say so" | done — this is the reported representational-claim verdict (associations exist, are geometry along the curve, not causal) | RESULTS §8 claim ladder |

**Gap-closure actions taken (this revision):** the missing citations the
guide lists are now in `paper/references.bib` and cited in the paper's
background section — Xu et al. (F-Principle), Rahaman et al. (spectral
bias), McClenny & Braga-Neto (self-adaptive PINNs), Wang et al. (causal
training), and the Chen et al. toy-model phase-transition analysis
(Gap-1 theory support). The Kim et al. ROM-autoencoder reference
differentiates the gap claim in related work.

## Gap 4 — Controller vs adaptive-training literature: **CLOSED**

Stage 12 is the guide's differentiation table, executed:

| Guide's method | Campaign comparison (same failure, same budget) |
|---|---|
| Wang et al. gradient-stat reweighting (GradNorm-style) | 2.0134 — actively harmful |
| Wang et al. NTK-based reweighting | **0.0064 — beats the controller (0.0166) and oracle (0.0119)** |
| McClenny & Braga-Neto self-adaptive | not run (cite-and-differentiate; point-wise soft attention is orthogonal to failure detection) |
| RAR/RAD residual resampling | controller's bounded action set includes resampling; not head-to-head run |
| Anagnostopoulos et al. residual-based attention (RBA) | 2.1918 — actively harmful |

And the guide's sharpest point — "the controller must react to a
causally validated internal feature, or it's just magnitude tracking" —
was answered *against* ourselves by the monitor-source ablation: the
SAE-feature monitor and a matched random-feature monitor drive
**bit-identical** rescue trajectories (0.0236 both), i.e. the controller
runs on conventional signals and the SAE features are inert cargo
(`runs/controller_demo/controller_comparison.json`). The paper states
plainly that the controller is failure-agnostic machinery, not
optimization SOTA, and keeps the engineering claims structurally
separate from the interpretability claims — exactly the "Phases 1–3
paper, Phase 4 separate" restructure the guide prescribes (already the
repo's structure: stages 1–11 interpretability, stages 6/7/12 engineering
with the ablation firewall).

---

## Summary verdict

| Gap | Status after this revision |
|---|---|
| 1 — Superposition precondition | Closed by the executed campaign; the negative outcome is the published result, with the boundary measured on the FNO side |
| 2 — NTK↔SAE bridge | **Open at review time → preregistered H15a/H15b, stage 15, committed before the run** (this revision) |
| 3 — Spectral-bias baselines | Closed in structure by probe controls + Fourier readouts; missing citations now added |
| 4 — Controller positioning | Closed by stage 12 + the monitor-source ablation; claims firewalled |

The one experiment this review adds to the campaign is stage 15. Its
preregistration, code, and (after the run) its recorded outcome live in
the standard places: `docs/preregistration.md` §H8,
`experiments/ntk_bridge.py`, `runs/ntk_bridge/`.
