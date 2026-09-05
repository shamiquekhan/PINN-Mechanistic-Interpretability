#!/usr/bin/env python
"""
Run seed matrix for qualification gate (Week 8).
Tests multiple seeds and failure-inducing configurations on GPU.
"""
import sys
import subprocess
import json
import numpy as np
from pathlib import Path
import yaml
import os
import copy


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
    result = subprocess.run(cmd, capture_output=True, text=True, cwd='/home/shamique/projects/Pinn research/pinn_mechanistic_starter', env=env)
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

    # 1. Success regime (baseline)
    configs['success'] = copy.deepcopy(base_config)
    configs['success']['run']['name'] = 'success_baseline'

    # 2. Boundary starvation - high PDE weight, low BC weight
    configs['boundary_starvation'] = copy.deepcopy(base_config)
    configs['boundary_starvation']['run']['name'] = 'boundary_starvation'
    configs['boundary_starvation']['training']['lambda_pde'] = 100.0
    configs['boundary_starvation']['training']['lambda_bc'] = 0.01

    # 3. Gradient conflict - very high learning rate
    configs['gradient_conflict'] = copy.deepcopy(base_config)
    configs['gradient_conflict']['run']['name'] = 'gradient_conflict'
    configs['gradient_conflict']['training']['learning_rate'] = 0.1

    # 4. Spectral suppression - deep narrow network with high frequency target
    configs['spectral_suppression'] = copy.deepcopy(base_config)
    configs['spectral_suppression']['run']['name'] = 'spectral_suppression'
    configs['spectral_suppression']['model']['hidden_layers'] = [16, 16, 16, 16, 16]
    configs['spectral_suppression']['pde']['source'] = 50.0

    # 5. Collocation starvation - very few interior points
    configs['collocation_starvation'] = copy.deepcopy(base_config)
    configs['collocation_starvation']['run']['name'] = 'collocation_starvation'
    configs['collocation_starvation']['training']['interior_points'] = 8

    return configs


def main():
    output_base = Path('runs/qualification')
    output_base.mkdir(parents=True, exist_ok=True)

    configs = create_failure_configs()
    seeds = [7, 42]
    steps = 1000

    results = {}

    for config_name, config in configs.items():
        print(f"\n=== Testing {config_name} ===")
        results[config_name] = {}

        for seed in seeds:
            config['run']['seed'] = seed
            run_name = f"{config_name}_seed{seed}"
            config['run']['name'] = run_name

            config_path = output_base / f'{run_name}.yaml'
            with open(config_path, 'w') as f:
                yaml.dump(config, f)

            print(f"  Running seed {seed}...")
            success, stdout, stderr = run_experiment(
                str(config_path), seed, steps, str(output_base)
            )

            if success:
                run_dir = Path('runs') / run_name
                eval_env = os.environ.copy()
                eval_env['PYTHONPATH'] = '.'
                subprocess.run(
                    [sys.executable, '-m', 'analysis.evaluate', '--run-dir', str(run_dir)],
                    capture_output=True, text=True,
                    cwd='/home/shamique/projects/Pinn research/pinn_mechanistic_starter',
                    env=eval_env
                )

                summary = evaluate_run(run_dir)
                results[config_name][f'seed_{seed}'] = summary
                l2 = summary.get('validation_relative_l2')
                l2_str = f"{l2:.6f}" if l2 is not None and not np.isnan(l2) else "N/A"
                print(f"    L2 error: {l2_str}, Converged: {summary.get('converged', 'N/A')}")
            else:
                print(f"    FAILED: {stderr[:200]}")
                results[config_name][f'seed_{seed}'] = {'error': stderr}

    with open(output_base / 'qualification_results.json', 'w') as f:
        json.dump(results, f, indent=2)

    print("\n=== QUALIFICATION GATE SUMMARY ===")
    print(f"{'Config':<25} {'Seed':<10} {'Rel L2':<12} {'Converged':<10}")
    print("-" * 60)

    for config_name, seed_results in results.items():
        for seed_key, summary in seed_results.items():
            if 'error' not in summary:
                l2 = summary.get('validation_relative_l2', 0)
                l2_val = l2 if l2 is not None else 0
                print(f"{config_name:<25} {seed_key:<10} "
                      f"{l2_val:<12.6f} "
                      f"{str(summary.get('converged', False)):<10}")
            else:
                print(f"{config_name:<25} {seed_key:<10} {'ERROR':<12} {'FAIL':<10}")

    return results


if __name__ == '__main__':
    main()