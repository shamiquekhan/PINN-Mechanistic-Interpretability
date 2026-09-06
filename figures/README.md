# Figure Generation — Publication Figures from `runs/` Artifacts

Every figure is generated **directly from the authoritative JSON artifacts**
in `runs/` (see `runs/README.md` and `data/manifest.json`). No numbers are
hand-entered in any script.

## Generate

```bash
bash scripts/generate_figures.sh        # all figures
python figures/figure_03_sae_vs_pca.py # a single figure
```

Output lands in `figures/generated/` (git-ignored; regenerate on demand).
Figure *data* is deterministic across platforms; PNG bytes may differ
slightly across matplotlib versions (documented in
`docs/reproducibility.md` §3).

## Figure inventory

| Script | Paper figure | Content | Source artifact |
|---|---|---|---|
| `figure_01_failure_atlas.py` | Fig. 1 | Failure-atlas reproducibility: per-regime rel-L2 distributions with 95% CIs (10 seeds) | `runs/failure_atlas/seed_matrix_stats.json` |
| `figure_02_activation_geometry.py` | Fig. 2 | The activation manifold: per-run PR distribution (1D suite), PR vs width, PR by PDE family with tangent ranks, 3D PCA manifold visualization | `effective_rank_report.json`, `width_scaling_report.json`, `dimensional_boundary_expanded_report.json`, `advection_baseline/activations.jsonl` |
| `figure_03_sae_vs_pca.py` | Fig. 3 | SAE failure modes: reconstruction bars (SAE vs PCA-k vs PCA-full vs random), dead-feature fractions, random-baseline association gaps | `sae_stage3_summary.json`, `feature_dictionary/random_baseline_report.json` |
| `figure_04_causal_null.py` | Fig. 4 | The basis-independent causal null: SAE + PCA E_T bootstrap distributions, per-feature sign-test p-values, MC-correction survivors, positive-control panel | `causal_intervention_results.json`, `pca_causal_results.json`, `positive_control.json` |
| `figure_05_causal_abstraction.py` | Fig. 5 | Interchange movement fractions: PCA vs SAE vs random per run (causal-abstraction null) | `causal_abstraction_results.json` |
| `figure_06_regime_boundary.py` | Fig. 6 | The regime boundary: PR/W vs SAE-vs-PCA reconstruction ratio across PINN families and the FNO | `operator_boundary_report.json` + PINN references |
| `figure_07_monitor.py` | Fig. 7 | Monitor AUROCs with run-level bootstrap CIs | `monitor_report.json` |
| `figure_08_controller_sota.py` | Fig. 8 | Controller rescue trajectories + SOTA baseline comparison | `controller_demo/controller_comparison.json`, `sota_baselines/sota_baseline_report.json` |

Each script is standalone (imports only numpy/matplotlib/torch where needed)
and exits non-zero on missing/corrupt artifacts.
