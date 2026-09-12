#!/usr/bin/env python
"""
Run 10-seed matrix for qualification gate and benchmark reporting.
Evaluates per-seed distributions, means, standard deviations, and 95% CIs on GPU.
"""
import sys
import subprocess
import json
import numpy as np
from pathlib import Path
import yaml
import os
import copy


def compute_bootstrap_ci(data: list, n_boot: int = 1000, ci: float = 0.95) -> dict:
    """Compute mean, std, and bootstrap percentile confidence interval."""
    arr = np.array(data, dtype=np.float64)
    if len(arr) == 0:
        return {"mean": 0.0, "std": 0.0, "ci_lower": 0.0, "ci_upper": 0.0}
    if len(arr) == 1:
        return {"mean": float(arr[0]), "std": 0.0, "ci_lower": float(arr[0]), "ci_upper": float(arr[0])}

    rng = np.random.default_rng(42)
    boot_means = [np.mean(rng.choice(arr, size=len(arr), replace=True)) for _ in range(n_boot)]
    alpha = (1.0 - ci) / 2.0
    lower = float(np.percentile(boot_means, 100 * alpha))
    upper = float(np.percentile(boot_means, 100 * (1.0 - alpha)))

    return {
        "mean": float(np.mean(arr)),
        "std": float(np.std(arr)),
        "ci_lower": lower,
        "ci_upper": upper,
    }


def run_experiment(config_path: str, seed: int, steps: int, output_dir: str, extra_args: list = None):
    """Run a single training experiment using current python interpreter."""
    cmd = [
        sys.executable, '-m', 'experiments.train',
        '--config', config_path,
        '--steps', str(steps),
    ]
    if extra_args:
        cmd.extend(extra_args)

    env = os.environ.copy()
    env['PYTHONPATH'] = '.'
    result = subprocess.run(cmd, capture_output=True, text=True, cwd=str(Path(__file__).resolve().parent.parent), env=env)
    return result.returncode == 0, result.stdout, result.stderr


def evaluate_run(run_dir: Path) -> dict:
    """Evaluate a completed run and return metrics."""
    summary_path = run_dir / 'run_summary.json'
    if summary_path.exists():
        with open(summary_path) as f:
            return json.load(f)
    return {}


def create_failure_configs():
    """Create configurations that induce different failure modes."""
    base_config = {
        'run': {
            'name': 'poisson_baseline',
            'output_dir': 'runs',
            'seed': 7,
            'deterministic': True,
            'device': 'cuda',
            'dtype': 'float32'
        },
        'pde': {
            'name': 'poisson_1d',
            'domain': [-1.0, 1.0],
            'source': 1.0,
            'forcing': 0.0,
            'diffusion': 0.01,
            'boundary_values': [0.0, 0.0],
            'validation_points': 1001
        },
        'model': {
            'input_dim': 1,
            'output_dim': 1,
            'hidden_layers': [64, 64, 64],
            'activation': 'tanh',
            'init': 'xavier'
        },
        'training': {
            'optimizer': 'adam',
            'learning_rate': 0.001,
            'steps': 5000,
            'interior_points': 256,
            'boundary_points': 2,
            'log_every': 100,
            'checkpoint_every': 1000,
            'lambda_pde': 1.0,
            'lambda_bc': 1.0,
            'resample_every': 0
        },
        'logging': {
            'save_activations': True,
            'activation_layers': [1],
            'save_pointwise_residuals': True,
            'log_gradients': True,
            'grad_log_every': 100,
            'log_diagnostics': True,
            'diag_log_every': 100
        }
    }

    configs = {}

    configs['success'] = copy.deepcopy(base_config)
    configs['success']['run']['name'] = 'success_baseline'

    configs['boundary_starvation'] = copy.deepcopy(base_config)
    configs['boundary_starvation']['run']['name'] = 'boundary_starvation'
    configs['boundary_starvation']['training']['lambda_pde'] = 100.0
    configs['boundary_starvation']['training']['lambda_bc'] = 0.01

    configs['gradient_conflict'] = copy.deepcopy(base_config)
    configs['gradient_conflict']['run']['name'] = 'gradient_conflict'
    configs['gradient_conflict']['run']['failure_label'] = 'gradient_conflict'
    configs['gradient_conflict']['training']['learning_rate'] = 0.1

    configs['spectral_suppression'] = copy.deepcopy(base_config)
    configs['spectral_suppression']['run']['name'] = 'spectral_suppression'
    configs['spectral_suppression']['run']['failure_label'] = 'spectral_suppression'
    configs['spectral_suppression']['model']['hidden_layers'] = [16, 16, 16, 16, 16]
    configs['spectral_suppression']['pde']['source'] = 50.0

    configs['collocation_starvation'] = copy.deepcopy(base_config)
    configs['collocation_starvation']['run']['name'] = 'collocation_starvation'
    configs['collocation_starvation']['run']['failure_label'] = 'collocation_starvation'
    configs['collocation_starvation']['training']['interior_points'] = 4
    configs['collocation_starvation']['training']['spatial_bias'] = 0.8

    return configs


def main():
    output_base = Path('runs/qualification')
    output_base.mkdir(parents=True, exist_ok=True)

    configs = create_failure_configs()
    seeds = [7, 42, 123, 2024, 2025, 2026, 999, 888, 777, 555]
    steps = 1000

    results = {}

    for config_name, config in configs.items():
        print(f"\n=== Testing {config_name} across {len(seeds)} seeds ===")
        results[config_name] = {"seed_runs": {}, "summary_stats": {}}
        l2_errors = []

        for seed in seeds:
            config['run']['seed'] = seed
            run_name = f"{config_name}_seed{seed}"
            config['run']['name'] = run_name

            config_path = output_base / f'{run_name}.yaml'
            with open(config_path, 'w') as f:
                yaml.dump(config, f)

            print(f"  Running seed {seed}...")
            success, stdout, stderr = run_experiment(
                str(config_path), seed, steps, str(output_base),
                extra_args=[
                    '--log-activations', '--act-save-raw',
                    '--log-gradients', '--log-diagnostics',
                ]
            )

            if success:
                run_dir = Path('runs') / run_name
                eval_env = os.environ.copy()
                eval_env['PYTHONPATH'] = '.'
                subprocess.run(
                    [sys.executable, '-m', 'analysis.evaluate', '--run-dir', str(run_dir)],
                    capture_output=True, text=True,
                    cwd=str(Path(__file__).resolve().parent.parent),
                    env=eval_env
                )

                summary = evaluate_run(run_dir)
                results[config_name]["seed_runs"][f'seed_{seed}'] = summary
                l2 = summary.get('validation_relative_l2')
                if l2 is not None and not np.isnan(l2):
                    l2_errors.append(l2)
                    l2_str = f"{l2:.6f}"
                else:
                    l2_str = "N/A"
                print(f"    L2 error: {l2_str}, Converged: {summary.get('converged', 'N/A')}")
            else:
                print(f"    FAILED: {stderr[:200]}")
                results[config_name]["seed_runs"][f'seed_{seed}'] = {'error': stderr}

        results[config_name]["summary_stats"] = compute_bootstrap_ci(l2_errors)

    with open(output_base / 'qualification_results.json', 'w') as f:
        json.dump(results, f, indent=2)

    print("\n=== QUALIFICATION GATE 10-SEED SUMMARY ===")
    print(f"{'Config':<25} {'Mean L2':<12} {'Std':<10} {'95% CI Lower':<12} {'95% CI Upper':<12}")
    print("-" * 75)

    for config_name, res in results.items():
        stats = res.get("summary_stats", {})
        print(f"{config_name:<25} {stats.get('mean', 0.0):<12.6f} {stats.get('std', 0.0):<10.6f} "
              f"{stats.get('ci_lower', 0.0):<12.6f} {stats.get('ci_upper', 0.0):<12.6f}")


if __name__ == '__main__':
    main()