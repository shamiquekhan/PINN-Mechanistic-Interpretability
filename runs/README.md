# `runs/` — Experimental Artifacts

This directory holds all empirical evidence for [RESULTS.md](../RESULTS.md).
Nothing here is hand-edited; every file is produced by a stage of
`experiments/run_pipeline.py` (or one of the standalone runners listed in
[docs/experiment_matrix.md](../docs/experiment_matrix.md)).

## Layout

```
runs/
├── <config_name>/                 # RAW: per-run training outputs (stage 1 / qualification)
│   ├── config.json                #   frozen manifest (seed, PDE, model, weights)
│   ├── metrics.jsonl              #   per-step losses + rel L2
│   ├── gradients.jsonl           #   per-loss gradient norms/cosines
│   ├── activations.jsonl          #   probe-point activations (raw vectors when logged)
│   ├── diagnostics.jsonl         #   residual/spectral diagnostics
│   ├── checkpoint_*.pt            #   model+optimizer snapshots
│   └── *.png                      #   per-run debug plots
├── <config>_seed<N>/              # RAW: seed-matrix runs (10 seeds × 5 regimes)
├── sae_models_v2/                 # DERIVED: TopK SAE replicas + baselines (stage 3)
├── feature_dictionary/            # DERIVED: multi-view dictionary + random control (stage 4)
├── effective_rank_analysis/       # DERIVED: participation-ratio report
├── width_scaling/                 # DERIVED: W=16–512 geometry study
├── dimensional_boundary_2d/       # DERIVED: 2D Poisson pilot (+ its SAE pilot)
├── dimensional_boundary_expanded/ # DERIVED: v3 2D suite + time-dependent (stage 10)
├── operator_boundary/             # DERIVED: FNO + SAE checkpoints + report (stage 11)
├── sota_baselines/                # DERIVED: GradNorm/NTK/RBA trajectories (stage 12)
├── statistical_hardening/         # DERIVED: power/Bayesian/sensitivity (stage 13)
├── controller_demo/               # DERIVED: 5-arm controller comparison (stage 7)
├── failure_atlas/                 # AGGREGATE: atlas index + seed-matrix stats (stage 2)
├── qualification/                 # RAW configs emitted by the qualification runner
├── causal_intervention_results.json      # AGGREGATE: SAE causal battery (stage 5) — AUTHORITATIVE
├── pca_causal_results.json               # AGGREGATE: PCA causal battery (stage 8) — AUTHORITATIVE
├── causal_abstraction_results.json       # AGGREGATE: interchange battery (stage 9) — AUTHORITATIVE
├── monitor_report.json                   # AGGREGATE: monitor AUROCs + CIs (stage 6) — AUTHORITATIVE
├── positive_control.json                 # AGGREGATE: planted-feature gate (stage 5) — AUTHORITATIVE
└── monitor/controller per-run JSONL logs under the demo dirs
```

## Authoritative artifacts for the paper

The six JSON files listed as AUTHORITATIVE above plus
`controller_demo/controller_comparison.json`,
`operator_boundary/operator_boundary_report.json`,
`sota_baselines/sota_baseline_report.json`,
`statistical_hardening/analysis_report.json`,
`effective_rank_analysis/effective_rank_report.json`, and
`failure_atlas/seed_matrix_stats.json` are the exact sources read by
`figures/*.py` and `scripts/generate_tables.sh`. If a number in
RESULTS.md and one of these files disagree, **the JSON is correct and
RESULTS.md must be fixed**.

## Raw vs derived

* **RAW** = produced directly by training (never regenerated from other
  runs/ files; the source of truth for per-step behavior).
* **DERIVED** = computed from RAW logs (deterministic; regenerable with the
  documented stage command).
* **AGGREGATE** = the single authoritative summary of one claim; what the
  figures/tables read.

## Stage → output map

See the stage table in [docs/reproducibility.md §4](../docs/reproducibility.md).

## Storage notes

* Total size ≈ 470 MB, dominated by per-run checkpoints and raw activation
  logs. Figures/tables regenerate from the JSONs, so a slim checkout for
  re-analysis only needs the AGGREGATE/DERIVED JSON files.
* No file exceeds GitHub's 100 MB limit; large binaries are intentionally
  not used anywhere in the pipeline.
* `runs/ci_smoke/` and other `*-test*` directories are transient (CI / debug)
  and safe to delete.
