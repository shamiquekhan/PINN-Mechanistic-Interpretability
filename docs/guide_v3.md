> **HISTORICAL (v3 planning document).** Superseded by the v3.5/v4.1
> preregistration sections in [preregistration.md](preregistration.md) and
> the evidence in [RESULTS.md](../RESULTS.md). Retained as the provenance
> record of the v3 revision's design decisions.

# Guide & Implementation Plan v3
## Mechanistic Interpretability for PINN Optimization Failures

*Revision history: v1 (original 48-week backlog) → v2 (20-week gate-driven plan; mandatory PCA/random/probe controls; TopK SAEs; controller-regression fix) → **v3 (adds Phase 5.5 "negative-result hardening" as a locked pipeline stage with acceptance criteria, plus the effective-rank diagnostic that upgrades ambiguous nulls into explained ones).***

---

## 0. What v3 adds and why

The v2 revision anticipated that SAE features might fail causal validation and committed to reporting a negative result honestly. The v2 experimental run delivered exactly that — and the v3 revision exists because a *negative* result requires **more** evidentiary support than a positive one before it is publishable. A reviewer's attack surface on a null is larger: maybe the machinery is broken, maybe the statistics are naive, maybe the data was never in the regime the method targets. v3 closes all three attack surfaces explicitly.

The central v3 principle, matching the SAE-skepticism literature it cites:

> **A negative result may only be reported at the strength of its strongest possible counterexplanation, and each counterexplanation must be measured away.**

Three counterexplanations, three measurements:

| Counterexplanation ("the null is fake because...") | v3 measurement that closes it | Status in this repo |
| :--- | :--- | :--- |
| "...the intervention/scoring machinery is broken" | **Planted-feature positive control**: synthetic activations with known causal structure through the exact pipeline | ✅ implemented (`experiments/positive_control.py`), PASSES |
| "...the statistics are naive (88 tests, no correction)" | **Bonferroni + BH-FDR** on per-feature paired sign tests | ✅ implemented (`interventions/causal.py`), 0/8 survive |
| "...the method was applied to data where its precondition fails" | **Effective-rank / participation-ratio analysis**: superposition requires features > neurons; measure whether that precondition holds | ✅ implemented (`analysis/effective_rank.py`), PR ≈ 1.8 of 64 → precondition fails, null *explained* |

Additionally v3 converts two previously-asserted claims into measurements:
- "The controller is driven by conventional signals" → **monitor-source ablation** (SAE-feature vs matched random-feature vs conventional monitors driving identical controllers).
- "Monitors are near chance" → **run-level bootstrap CIs** on AUROC (the run is the independent unit, not the example).

---

## 1. Phase 5.5 — Negative-Result Hardening (locked, with acceptance criteria)

Positioned immediately after the causal-validation gate (Phase 5). It runs **regardless of whether Phase 5 passed**, because the same hardening protects weak positives from being chance artifacts.

### 5.5.1 Multiple-comparison corrections (mandatory)

Every feature-level causal test must be corrected across the number of features tested:

- Per-feature test: one-sided **paired sign test** across evaluation runs (each run contributes one paired observation |target| − max|controls|).
- Corrections: **Bonferroni** (α/m) and **BH-FDR** (q=0.05), both reported.
- Report the **chance expectation** (α·m effective hits) next to the survivor count.

**Acceptance criteria:**
- [x] Corrections computed and reported in the results artifact
- [x] The paper states the corrected survivor count in its abstract-level claim (e.g., "0/8 features survived")
- [x] If ANY feature survives correction: it is flagged for independent validation on fresh runs before any causal claim is made (two-stage discovery/validation design — never use the same trajectories to choose and claim)

### 5.5.2 Planted-feature positive control (mandatory)

Before any null is reported, the exact SAE + hook + intervention + scoring battery must recover a **known** causal feature:

- Synthetic world: `n_features > n_dim` (superposition regime), k-sparse latents, fixed random dictionary; planted linear readout along one dictionary direction (linear readout so orthogonal perturbations — the random control — move the loss ~1/√d of the ablation footprint *by construction*, not by luck).
- Run the REAL pipeline: train the real TopK SAE on the bank, hook the real engine, apply all three controls, score with the real `compute_causal_score`.
- Random control: median over ≥5 direction draws.

**Acceptance criteria:**
- [x] SAE recovers the planted direction (decoder cosine > 0.9)
- [x] Targeted ablation effect > all three controls with E_T > 0, on a majority of seeds
- [x] Artifact + test exist (`runs/positive_control.json`; suite test runs it in a clean subprocess)
- [x] **If the positive control FAILS, the null result is quarantined**: the pipeline must be fixed and re-run before any PINN claim (positive or negative) is reported. The null is attributable to the activations only when the machinery demonstrably works.

### 5.5.3 Effective-rank / superposition-precondition analysis (mandatory)

SAEs presuppose superposition: more features than neurons. Measure whether the target network's activations satisfy that precondition, turning ambiguous nulls into explained ones:

- **Participation ratio** of activation covariance: PR = (Σλ)²/Σλ² per run.
- Stable rank, PCA components for 95/99/99.9% energy.
- Compare PR to layer width. Report `superposition_ratio = PR / width`.

**Acceptance criteria:**
- [x] Computed per run and aggregated (success + failure regimes)
- [x] If ratio << 1 (this repo: 0.02): report the null as **precondition-failure-explained** — "sparse dictionary learning solves a problem that does not exist in this representation" — with the measurement front-and-center in the discussion. This is a *stronger* contribution than an unexplained null.
- [x] If ratio ≳ 1 and the null persists: that is a much more surprising result; escalate to a follow-up investigation before publication (the null is no longer mechanically explained).

### 5.5.4 Claim-separation audit (mandatory for the writeup)

- [x] Mechanistic claim (SAE causality) and engineering claim (controller rescue) are stated in separate sections with separate evidence tables; the controller result must never be phrased as validating the interpretability pipeline.
- [x] Monitor-source ablation quantifies what drives the controller: SAE-feature monitor vs **matched-dimensionality random-feature monitor** vs conventional monitor, identical controller/actions. Identical-or-worse SAE arm ⇒ "inert cargo, measured not asserted."
- [x] Uncertainty on every headline number: run-level bootstrap CIs for monitor AUROC; feature-level CIs for E_T/S_spec; seed-level CIs for failure-regime L².

### 5.5.5 What is NOT permitted in hardening

- **No re-tuning until the gate flips.** Trying more layers/widths/seeds/layers until something survives correction is the exact failure mode the SAE-skepticism literature warns about. Any expansion of the search after seeing the null must be preregistered as a *follow-up* with fresh discovery/validation splits.
- **No reframing the controller's success as an SAE win** (see 5.5.4).
- **No burying the precondition analysis** if it shows the method was applied outside its regime — that finding IS the contribution.

---

## 2. v3 claim ladder (final, with current evidence status)

1. **Representational** — *explained-null*: raw associations beat random controls 7–57× (fresh pool) but PR ≈ 1.8/64 shows they are geometry along a low-rank curve, not superposed features.
2. **Causal** — *rejected, hardened*: E_T CI [−0.003, +0.008]; 0/8 survive Bonferroni or BH-FDR; sign-consistent with representational encoding; probe ≥ SAE in 8% of evaluations; positive control passes so the machinery is exonerated.
3. **Predictive** — *supported with scope*: conventional and SAE-augmented monitor AUROCs are 0.859/0.864 on the current held-out failure mixture, while the loss-only threshold is near chance; overlapping CIs do not establish an SAE advantage.
4. **Prevention** — *supported for controller machinery only*: 0.295→0.017 rescue (18× over no-action, 1.4× off oracle), with the measured ablation showing SAE features are inert in the loop.

**Title-level claim the evidence supports:** SAE-derived features provide no causal information beyond matched controls for PINN failure diagnosis in this benchmark; effective-rank analysis explains why (no superposition); a conventional-signal closed-loop controller reduces failure rate independently; consistent with Korznikov et al. 2026, Leask et al. 2025, Wu et al. 2025.

**Discussion's real contribution:** the participation-ratio precondition test as a cheap, general *go/no-go diagnostic* for whether sparse-autoencoder interpretability is even applicable to a given scientific network — before investing in dictionary learning. Test PR/dim first; if it's ≪ 1, the SAE program is solving a nonexistent problem and linear tools are the ceiling.

**Theoretical caution:** low-dimensional input implies a low-dimensional local tangent space, not a universal covariance-rank bound. Curved one-dimensional activation manifolds can have covariance rank greater than two. Treat PR, stable rank, PCA energy, and local tangent rank as empirical measurements; do not present `PR ≤ d + 1` as a theorem without additional assumptions.

The first width-scaling follow-up is available at `experiments/width_scaling.py`:

```bash
python -m experiments.width_scaling --device cuda --steps 1000
```

It writes per-width activation logs and `width_scaling_report.json` under `runs/width_scaling/`. The report compares covariance PR with local tangent rank while holding the 1D Poisson task fixed.

The completed six-width pilot (16 through 512, 1,000 steps, seed 0) found PR `1.74–2.00` and local tangent rank `1` at every width. This supports the scoped hypothesis that width alone does not create superposition for this fixed 1D task; it is not a universal theorem and should be followed by higher-dimensional or operator-learning experiments.

A manufactured 2D Poisson prototype is now available at `configs/poisson_2d_boundary.yaml`. A five-seed pilot is complete: PR `2.89–3.17` (mean `3.03`) and local tangent rank `2` at every seed, versus PR `1.74–2.00` and tangent rank `1` in the 1D width study. This is dimensional-boundary evidence, not yet a 2D SAE or causal battery.

**Limitations (mandatory, scoped):** no ruling out SAEs on wide PINNs / ensembles / operator learners (FNOs, DeepONets) where effective rank plausibly exceeds width; monitor results are specific to this held-out failure mixture and window protocol; gradient-conflict and collocation-starvation were stress-tested but excluded from final failure-class claims because their operational labels did not reproduce cleanly.

---

## 3. Reproduction map (v3 artifacts)

| Claim | Artifact | Command |
| :--- | :--- | :--- |
| 10-seed failure regimes | `runs/failure_atlas/seed_matrix_stats.json` | `run_pipeline --stages 2` |
| SAE vs PCA/random baselines | `runs/sae_models_v2/sae_stage3_summary.json` | `--stages 3` |
| Multi-view associations + random control | `runs/feature_dictionary/` | `--stages 4` |
| Causal battery + MC corrections + sign/probe diagnostics | `runs/causal_intervention_results.json` | `--stages 5` |
| Planted-feature positive control | `runs/positive_control.json` | `experiments.positive_control` |
| Superposition precondition | `runs/effective_rank_analysis/` | `analysis.effective_rank` |
| Monitor suite + run-level CIs | `runs/monitor_report.json` | `--stages 6` |
| Controller rescue + monitor-source ablation | `runs/controller_demo/controller_comparison.json` | `--stages 7` |
| PCA causal battery / causal abstraction / operator boundary / operator causal / SOTA / hardening | `runs/pca_causal_results.json`, `runs/causal_abstraction_results.json`, `runs/operator_boundary/`, `runs/operator_causal/`, `runs/sota_baselines/`, `runs/statistical_hardening/` | `run_pipeline --stages 8,9,11,12,13,14` |
| Full test suite (171 tests at the v3 planning time) | — | `pytest tests/` |

---

## 4. References (unchanged from v2; re-verify before submission)

Wang, Yu & Perdikaris 2020 (NTK) · Momentum & spectral bias 2022 (arXiv:2206.14862) · Gao et al. 2024 (TopK) · Rajamanoharan et al. 2024 (JumpReLU/Gated) · Bussmann et al. 2024 (Batch-TopK) · **Korznikov et al. 2026 (SAE sanity checks vs random baselines)** · **Heap et al. 2025 (SAEs interpret random transformers)** · **Leask et al. 2025 (no canonical units)** · **Wu et al. 2025 (AXBench: simple baselines beat SAE steering)** · Kantamneni et al. 2025 (sparse probing) · **Qi & Earls 2026 (SAE steering of neural quantum states — the positive-domain precedent)** · Gao et al. 2022 (failure-informed adaptive sampling) · Bricken et al. 2023 (monosemanticity).

*The 2025–2026 SAE-skepticism debate is active; check for rebuttals/follow-ups before final submission, as recommended in v2.*
