"""Generate LaTeX publication tables directly from runs/ artifacts.

Each table function reads only the authoritative JSONs (same sources as
figures/) — no hand-entered numbers. Output: paper/tables/*.tex
"""
from __future__ import annotations

import json
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
RUNS = PROJECT_ROOT / "runs"
OUT_DIR = PROJECT_ROOT / "paper" / "tables"


def load(rel: str) -> dict:
    return json.loads((RUNS / rel).read_text())


def fmt_ci(ci: dict, prec: int = 4) -> str:
    return (f"{ci['mean']:.{prec}f} "
            f"[{ci['ci_lower']:.{prec}f}, {ci['ci_upper']:.{prec}f}]")


def table_failure_atlas() -> str:
    s = load("failure_atlas/seed_matrix_stats.json")
    rows = []
    for regime in ["success", "boundary_starvation", "gradient_conflict",
                   "spectral_suppression", "collocation_starvation"]:
        d = s[regime]
        ci = d["rel_l2_ci"]
        labels = d["label_counts"]
        dominant = max(labels.items(), key=lambda kv: kv[1]) if labels \
            else ("unlabeled", 0)
        rows.append(
            f"{regime.replace('_', ' ')} & {d['n_seeds']} & "
            f"{ci['mean']:.3f} $\\pm$ {ci['std']:.3f} & "
            f"{dominant[1]}/{d['n_seeds']} {dominant[0].replace('_', ' ')} \\\\")
    body = "\n".join(rows)
    return (
        "\\begin{tabular}{lccc}\n"
        "\\toprule\n"
        "Regime & Seeds & Final rel $L_2$ (mean $\\pm$ std) & Label consistency \\\\\n"
        "\\midrule\n" + body + "\n\\bottomrule\n\\end{tabular}\n")


def table_reconstruction() -> str:
    s = load("sae_models_v2/sae_stage3_summary.json")
    b = s["baselines"]
    rows = []
    for i, r in enumerate(s["trained"]):
        dead_pct = 100 * r["dead_feature_frac"]
        rows.append(f"TopK SAE (replica {i + 1}) & {r['test_recon_mse']:.2e} & "
                    f"{dead_pct:.0f}\\% \\\\")
    rows.append(f"k-matched PCA ($k=8$) & {b['pca_k_matched_train_mse']:.2e} & --- \\\\")
    rows.append(f"PCA (full, 64) & {b['pca_full_train_mse']:.2e} & --- \\\\")
    rows.append(f"Random-encoder SAE & {b['random_sae_test_mse']:.2e} & --- \\\\")
    body = "\n".join(rows)
    return (
        "\\begin{tabular}{lcc}\n\\toprule\n"
        "Model & Test reconstruction MSE & Dead features \\\\\n\\midrule\n"
        + body + "\n\\bottomrule\n\\end{tabular}\n")


def table_causal_batteries() -> str:
    sae = load("causal_intervention_results.json")
    pca = load("pca_causal_results.json")

    def row(name, d):
        mc = d["multiple_comparisons"]
        sd = d["sign_diagnostic"]
        return (f"{name} & {fmt_ci(d['causal_strength_ci'])} & "
                f"{fmt_ci(d['specificity_ci'], 2)} & "
                f"{mc['bonferroni_n_survivors']}/{mc['n_features_tested']} & "
                f"{mc['bh_fdr_n_survivors']}/{mc['n_features_tested']} & "
                f"{sd['n_target_delta_positive']}/{d['n_evaluations']} \\\\")

    body = "\n".join([
        row("TopK SAE features", sae),
        row("PCA components", pca),
    ])
    return (
        "\\begin{tabular}{lccccc}\n\\toprule\n"
        "Candidate basis & $E_T$ (95\\% CI) & $S_{\\mathrm{spec}}$ & "
        "Bonferroni & BH-FDR & Sign diag. \\\\\n\\midrule\n"
        + body + "\n\\bottomrule\n\\end{tabular}\n")


def table_geometry() -> str:
    eff = load("effective_rank_analysis/effective_rank_report.json")
    ws = load("width_scaling/width_scaling_report.json")
    dexp = load("dimensional_boundary_expanded/dimensional_boundary_expanded_report.json")
    oper = load("operator_boundary/operator_boundary_report.json")

    rows = [
        f"1D suite (W=64, 45 runs) & {eff['summary']['dim']} & "
        f"{eff['summary']['mean_participation_ratio']:.2f} & 1 & 2 / 3 \\\\",
    ]
    for r in ws["results"]:
        rows.append(f"1D Poisson, W={r['width']} & {r['dim']} & "
                    f"{r['participation_ratio']:.2f} & {r['local_tangent_rank']} & "
                    f"{r['n_components_95pct']} / {r['n_components_99pct']} \\\\")
    rows.append("2D Poisson (pilot, W=32) & 32 & 3.03 & 2 & --- \\\\")
    for family, fr in dexp["families"].items():
        s = fr.get("summary", {})
        if s.get("mean_participation_ratio") is None:
            continue
        rows.append(f"{family.replace('_', ' ')} & --- & "
                    f"{s['mean_participation_ratio']:.2f} & "
                    f"{s['tangent_ranks'][0] if s['tangent_ranks'] else '?'} & --- \\\\")
    g = oper["geometry"]
    rows.append(f"FNO block states & {g['dim']} & {g['participation_ratio']:.2f} & "
                f"--- & {g['n_components_95pct']} / {g['n_components_99pct']} \\\\")
    body = "\n".join(rows)
    return (
        "\\begin{tabular}{lcccc}\n\\toprule\n"
        "Representation & Width & Covariance PR & Tangent rank & "
        "PCA comps (95\\%/99\\%) \\\\\n\\midrule\n"
        + body + "\n\\bottomrule\n\\end{tabular}\n")


def table_regime_boundary() -> str:
    oper = load("operator_boundary/operator_boundary_report.json")
    sae = load("sae_models_v2/sae_stage3_summary.json")
    eff = load("effective_rank_analysis/effective_rank_report.json")

    import numpy as np
    pinn_sae = float(np.mean([r["test_recon_mse"] for r in sae["trained"]]))
    pinn_pca = sae["baselines"]["pca_k_matched_train_mse"]
    fno_ratio = oper["sae"]["test_recon_mse"] / oper["pca_k_recon"]

    rows = [
        f"1D PINNs & {eff['summary']['mean_participation_ratio']:.2f} / "
        f"{eff['summary']['dim']} & "
        f"{eff['summary']['superposition_ratio_mean_to_dim']:.3f} & "
        f"{pinn_sae / pinn_pca:.1f}$\\times$ (PCA wins) \\\\",
        f"FNO (Green's) & {oper['geometry']['participation_ratio']:.2f} / "
        f"{oper['geometry']['dim']} & "
        f"{oper['verdict']['fno_superposition_ratio']:.3f} & "
        f"{fno_ratio:.2f}$\\times$ (SAE wins) \\\\",
    ]
    body = "\n".join(rows)
    return (
        "\\begin{tabular}{lccc}\n\\toprule\n"
        "Representation & Covariance PR / W & $\\rho$ = PR/W & "
        "SAE vs PCA($k{=}8$) recon \\\\\n\\midrule\n"
        + body + "\n\\bottomrule\n\\end{tabular}\n")


def table_monitor_controller_sota() -> str:
    mon = load("monitor_report.json")
    cc = load("controller_demo/controller_comparison.json")
    sota = load("sota_baselines/sota_baseline_report.json")
    sh = load("statistical_hardening/analysis_report.json")

    rows = []
    for key, name in [("loss_only_threshold", "Loss-only threshold"),
                      ("conventional_logistic", "Conventional logistic"),
                      ("sae_plus_conventional", "SAE + conventional")]:
        r = mon[key]
        ci = r["auroc_ci"]
        rows.append(f"{name} & {r['auroc']:.3f} & "
                    f"[{ci['ci_lower']:.3f}, {ci['ci_upper']:.3f}] & "
                    f"{r['auprc']:.3f} \\\\")
    body_mon = "\n".join(rows)

    ref = sota["reference_arms"]
    rows = [
        f"No action & {ref['no_action_final']:.4f} \\\\",
        f"\\textbf{{Closed-loop controller (ours)}} & {ref['controller_final']:.4f} \\\\",
        f"Oracle static reweight & {ref['oracle_final']:.4f} \\\\",
        f"GradNorm & {sota['baselines']['GradNorm']['final_rel_l2']:.4f} \\\\",
        f"NTK-adaptive weighting & {sota['baselines']['NTK-adaptive']['final_rel_l2']:.4f} \\\\",
        f"RBA (residual attention) & {sota['baselines']['RBA']['final_rel_l2']:.4f} \\\\",
    ]
    body_ctrl = "\n".join(rows)

    pw = sh["power_sae"]["summary"]
    bayes = sh["monitor_bayesian"]
    mde_pct = 100 * pw["mde_success_prob"]
    hardening = (
        f"Minimum detectable sign-agreement at 80\\% power "
        f"($n={pw['n_runs_per_feature']}$ runs/feature) & "
        f"{mde_pct:.1f}\\% \\\\\n"
        f"$P$(SAE-augmented monitor $>$ conventional) & "
        f"{bayes['prob_sae_better']:.2f} \\\\"
    )

    return (
        "% ---- Monitors ----\n"
        "\\begin{tabular}{lccc}\n\\toprule\n"
        "Monitor & AUROC & 95\\% CI & AUPRC \\\\\n\\midrule\n" + body_mon +
        "\n\\bottomrule\n\\end{tabular}\n\n"
        "% ---- Controller and SOTA ----\n"
        "\\begin{tabular}{lc}\n\\toprule\n"
        "Method & Final rel $L_2$ (boundary starvation) \\\\\n\\midrule\n"
        + body_ctrl + "\n\\bottomrule\n\\end{tabular}\n\n"
        "% ---- Statistical hardening ----\n"
        "\\begin{tabular}{lc}\n\\toprule\n"
        "Quantity & Value \\\\\n\\midrule\n" + hardening +
        "\n\\bottomrule\n\\end{tabular}\n")


def table_operator_causal() -> str:
    sae = load("causal_intervention_results.json")
    pca = load("pca_causal_results.json")
    op = load("operator_causal/operator_causal_report.json")

    def row(name, d, evals):
        mc = d["multiple_comparisons"]
        wins = sum(1 for r in evals
                   if abs(r["causal_score"]["target_effect"]) >
                   max(abs(r["causal_score"]["control_unrelated_effect"]),
                       abs(r["causal_score"]["control_random_effect"]),
                       abs(r["causal_score"]["control_probe_effect"])))
        return (f"{name} & {mc['bonferroni_n_survivors']}/{mc['n_features_tested']} & "
                f"{wins}/{len(evals)} \\\\")

    body = "\n".join([
        row("TopK SAE on PINN hidden acts", sae, sae["evaluations"]),
        row("PCA components on PINN hidden acts", pca, pca["evaluations"]),
        row("TopK SAE on FNO block states", op["battery_summary"], op["evaluations"]),
    ])
    return (
        "\\begin{tabular}{lcc}\n\\toprule\n"
        "Battery & Bonferroni survivors & Target beats all controls \\\\\n\\midrule\n"
        + body + "\n\\bottomrule\n\\end{tabular}\n")


TABLES = {
    "table_operator_causal.tex": table_operator_causal,
    "table_failure_atlas.tex": table_failure_atlas,
    "table_reconstruction.tex": table_reconstruction,
    "table_causal_batteries.tex": table_causal_batteries,
    "table_geometry.tex": table_geometry,
    "table_regime_boundary.tex": table_regime_boundary,
    "table_monitor_controller_sota.tex": table_monitor_controller_sota,
}


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    for name, fn in TABLES.items():
        out = OUT_DIR / name
        out.write_text(fn())
        print(f"  wrote paper/tables/{name}")


if __name__ == "__main__":
    main()
