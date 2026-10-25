# Fresh Pipeline Validation Record — 2026-10-25

## Scope

The complete local pipeline was rerun on the CPU-only checkout with:

```bash
python -m experiments.run_pipeline --stages all
```

The registered follow-up experiments affected by the intervention-protocol
changes were also rerun:

```bash
python -m experiments.architecture_boundary --steps 2000
python -m experiments.r2_fourier_causal
python -m experiments.r6_dose_response
python -m experiments.r7plus_equivalence
python scripts/equivalence_test.py
```

These are validation reruns of the current implementation, not a new
preregistered publication campaign. The historical review and publication
records retain their original values where they explicitly identify an earlier
campaign. This document is the current artifact ledger for the regenerated
outputs.

## Verification receipts

- Stages 1–15 completed; the pipeline log ends with `=== PIPELINE COMPLETE ===`.
- `python3 -m pytest -q` — 201 passed.
- `python scripts/check_results_grounded.py` — OK.
- `sha256sum -c data/checksums.sha256` — all listed artifacts verified.
- `data/manifest.json` byte counts and SHA256 values were refreshed.
- `runs/expected_headlines.json` was regenerated from the current artifacts.

## Current artifact ledger

| Area | Current artifact result | Interpretation |
|---|---|---|
| Stage 3 SAE reconstruction | SAE beats the random baseline; SAE does not beat k-matched PCA | The current low-rank reconstruction output does not support an SAE advantage over PCA. |
| Stage 5 SAE causal battery | $E_T$ mean `−8.07e−5`, 95% CI `[−0.00316, +0.00545]`; 0/8 Bonferroni and 0/8 BH-FDR survivors | Current rerun remains a causal null; the CI spans zero. |
| Stage 8 PCA causal battery | $E_T$ mean `−0.0102`, 95% CI `[−0.0641, +0.0408]`; 0/8 Bonferroni and 0/8 BH-FDR survivors | Current rerun remains a corrected causal null. |
| Stage 9 causal abstraction | PCA and SAE do not beat the random basis on the current run | No current interchange evidence for the tested region identity. |
| Stage 6 monitor suite | Conventional AUROC `0.8442`; SAE+conventional `0.8476`; loss-only threshold `0.5168`; Bayesian P(SAE better) `0.518` | The current rerun does not support an SAE monitor advantage. |
| Stage 7 controller | Controller final `0.000117`; no-action `0.02799`; oracle `0.000765`; SAE and random monitor trajectories remain identical | The rescue is attributable to controller machinery and instability-correlated signal, not mechanistic SAE features. |
| Stage 10 dimensional boundary | Current family mean PRs: AD2D `2.52`, RD2D `2.30`, Burgers `1.45`, Allen–Cahn `2.03`; tangent rank 2 in each family | These remain pilot outputs; the current CPU rerun is not a new external-validity claim. |
| Stage 11 operator boundary | FNO PR `6.39`; SAE beats k-matched PCA in the current report | Reconstruction separation remains present in the operator artifact. |
| Stage 12 SOTA baselines | Controller `0.000117`, NTK-adaptive `0.02681`, GradNorm `4.2066`, RBA `2.2676` on the current comparison | The current single-run comparison is configuration-specific and does not replace the broader H17 battery. |
| Stage 13 statistical hardening | MDE target remains 80%; monitor posterior P(SAE better) `0.518`; threshold ordering is not stable | The current statistical artifact does not support an SAE monitoring advantage. |
| Stage 14 operator causal battery | 3/8 Bonferroni and 3/8 BH-FDR survivors; positive $E_T$ CI `[1.38e−5, 4.12e−5]`; 31/64 evaluations beat all controls | The current artifact still records H14b: the preregistered broader-null verdict, not a confirmed causal mechanism. |
| Stage 15 NTK↔SAE bridge | H15b; no feature-specific bridge at the registered bar | The current bridge artifact remains null. |
| R2 Fourier causal follow-up | Fourier $E_T$ `−30.51` with CI `[−33.22, −27.88]`, 0/45 Bonferroni survivors, 0 crossovers; tanh control R2b | The higher-rank reconstruction regime still does not establish direction-specific causal specificity. |
| R6 dose response | Fourier SAE/PCA and FNO arms have 0 R6a; tanh-PCA has 2 R6a shape-only findings | The signed mechanistic criterion remains null in the high-rank arms; inert-scale tanh-PCA findings remain scoped as shape-only. |
| R7 equivalence | R7b; current checkpoint-matched comparison remains inconclusive at $\delta = 0.01$ | Formal equivalence is not established by the current 8-pair test. |
| R7+ replication | 21 combined checkpoints; R7+c; cluster 90% CI `[−0.03235, −0.01374]`; TOST `p = 0.9793` | The current registered replication establishes non-equivalence at the stated margin; this supersedes the earlier R7+ intermediate result. |

## Documentation rule

The current rerun changes several numerical outputs from the frozen v4.3
publication record, including the stage-14 count (`2/8` → `3/8`) and the R7+
outcome (`R7+b` → `R7+c`). Those historical values remain in their original
records only when explicitly labeled as historical. Living summaries and
verification metadata point to this ledger and the current JSON artifacts.

## Authoritative files

- `runs/operator_causal/operator_causal_report.json`
- `runs/architecture_boundary/architecture_boundary_report.json`
- `runs/r2_fourier_causal/r2_report.json`
- `runs/r6_dose_response/r6_report.json`
- `runs/equivalence_test/equivalence_report.json`
- `runs/r7plus_equivalence/r7plus_report.json`
- `runs/expected_headlines.json`
- `data/manifest.json`
- `data/checksums.sha256`
