import json
import numpy as np
import torch
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from pathlib import Path
from typing import Dict, List, Optional
from pinn.config import ExperimentConfig, load_config
from pinn.pdes import make_pde
from pinn.model import MLP
from pinn.reproducibility import set_seed
from pinn_logging.io import load_jsonl


def load_metrics(metrics_path: Path) -> List[Dict]:
    return load_jsonl(metrics_path)


def load_checkpoint(ckpt_path: Path, device: torch.device, dtype: torch.dtype) -> Dict:
    return torch.load(ckpt_path, map_location=device, weights_only=True)


def rebuild_model_from_config(cfg, device: torch.device, dtype: torch.dtype):
    model = MLP(
        cfg.model.input_dim,
        cfg.model.output_dim,
        cfg.model.hidden_layers,
        cfg.model.activation,
        cfg.model.init,
        cfg.model.fourier_embed,
        cfg.model.fourier_n_freq,
        cfg.model.fourier_scale,
    ).to(device=device, dtype=dtype)
    return model


def evaluate_on_grid(model, pde, device, dtype, n_points=1001):
    x = pde.validation_grid(n_points, device, dtype)
    with torch.no_grad():
        pred = model(x)
    exact = pde.exact(x)
    return x, pred, exact


def compute_error_metrics(pred, exact):
    if exact is None:
        return {'relative_l2': float('nan'), 'linf': float('nan'), 'mse': float('nan')}
    rel_l2 = torch.linalg.vector_norm(pred - exact) / torch.linalg.vector_norm(exact)
    linf = torch.max(torch.abs(pred - exact))
    mse = torch.mean((pred - exact) ** 2)
    return {
        'relative_l2': float(rel_l2),
        'linf': float(linf),
        'mse': float(mse)
    }


def plot_training_curves(metrics: List[Dict], out_dir: Path):
    if not metrics:
        return
    steps = [m['step'] for m in metrics]
    loss = [m['loss'] for m in metrics]
    loss_pde = [m['loss_pde'] for m in metrics]
    loss_bc = [m['loss_bc'] for m in metrics]
    rel_l2 = [m['relative_l2'] for m in metrics]

    fig, axes = plt.subplots(2, 2, figsize=(10, 8))

    axes[0, 0].semilogy(steps, loss, label='Total')
    axes[0, 0].semilogy(steps, loss_pde, label='PDE')
    axes[0, 0].semilogy(steps, loss_bc, label='BC')
    axes[0, 0].set_xlabel('Step')
    axes[0, 0].set_ylabel('Loss')
    axes[0, 0].set_title('Loss Components')
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)

    axes[0, 1].semilogy(steps, rel_l2)
    axes[0, 1].set_xlabel('Step')
    axes[0, 1].set_ylabel('Relative L2 Error')
    axes[0, 1].set_title('Validation Error')
    axes[0, 1].grid(True, alpha=0.3)

    axes[1, 0].plot(steps, loss_pde, label='PDE')
    axes[1, 0].plot(steps, loss_bc, label='BC')
    axes[1, 0].set_xlabel('Step')
    axes[1, 0].set_ylabel('Loss')
    axes[1, 0].set_title('Loss Components (Linear)')
    axes[1, 0].legend()
    axes[1, 0].grid(True, alpha=0.3)

    axes[1, 1].plot(steps, rel_l2)
    axes[1, 1].set_xlabel('Step')
    axes[1, 1].set_ylabel('Relative L2')
    axes[1, 1].set_title('Validation Error (Linear)')
    axes[1, 1].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(out_dir / 'training_curves.png', dpi=150)
    plt.close()


def plot_solution_comparison(x, pred, exact, out_dir: Path):
    if exact is None:
        return
    x_np = x.cpu().numpy().flatten()
    pred_np = pred.cpu().numpy().flatten()
    exact_np = exact.cpu().numpy().flatten()

    fig, axes = plt.subplots(1, 2, figsize=(10, 4))

    axes[0].plot(x_np, exact_np, 'k-', label='Exact', linewidth=2)
    axes[0].plot(x_np, pred_np, 'r--', label='PINN', linewidth=2)
    axes[0].set_xlabel('x')
    axes[0].set_ylabel('u(x)')
    axes[0].set_title('Solution Comparison')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    axes[1].plot(x_np, pred_np - exact_np, 'b-', label='Error', linewidth=1)
    axes[1].set_xlabel('x')
    axes[1].set_ylabel('Error')
    axes[1].set_title('Pointwise Error')
    axes[1].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(out_dir / 'solution_comparison.png', dpi=150)
    plt.close()


def generate_run_summary(run_dir: Path, metrics: List[Dict], final_metrics: Dict, cfg) -> Dict:
    summary = {
        'run_name': cfg.run.name,
        'seed': cfg.run.seed,
        'failure_label': cfg.run.failure_label,
        'final_step': metrics[-1]['step'] if metrics else 0,
        'total_steps': cfg.training.steps,
        'final_loss': metrics[-1]['loss'] if metrics else None,
        'final_loss_pde': metrics[-1]['loss_pde'] if metrics else None,
        'final_loss_bc': metrics[-1]['loss_bc'] if metrics else None,
        'final_relative_l2': metrics[-1]['relative_l2'] if metrics else None,
        'validation_relative_l2': final_metrics['relative_l2'],
        'validation_linf': final_metrics['linf'],
        'validation_mse': final_metrics['mse'],
        'converged': final_metrics['relative_l2'] < 1e-2 if not np.isnan(final_metrics['relative_l2']) else False,
        'pde_config': cfg.pde.model_dump(),
        'model_config': cfg.model.model_dump(),
        'training_config': cfg.training.model_dump(),
    }
    return summary


def main(run_dir: str):
    run_path = Path(run_dir)
    config_path = run_path / 'config.json'

    if not config_path.exists():
        manifest_path = run_path / 'manifest.json'
        if manifest_path.exists():
            with open(manifest_path) as f:
                config_dict = json.load(f).get('config', {})
        else:
            raise FileNotFoundError(f"Config not found at {config_path} or {manifest_path}")
    else:
        with open(config_path) as f:
            config_dict = json.load(f)

    cfg = ExperimentConfig.model_validate(config_dict)

    set_seed(cfg.run.seed, cfg.run.deterministic)
    device = torch.device(cfg.run.device if torch.cuda.is_available() or cfg.run.device == 'cpu' else 'cpu')
    dtype = getattr(torch, cfg.run.dtype)

    pde = make_pde(cfg.pde)
    model = rebuild_model_from_config(cfg, device, dtype)

    checkpoints = sorted(run_path.glob('checkpoint_*.pt'))
    if not checkpoints:
        raise FileNotFoundError("No checkpoints found")

    latest_ckpt = checkpoints[-1]
    ckpt = load_checkpoint(latest_ckpt, device, dtype)
    model.load_state_dict(ckpt['model'])

    metrics_path = run_path / 'metrics.jsonl'
    metrics = load_metrics(metrics_path) if metrics_path.exists() else []

    x, pred, exact = evaluate_on_grid(model, pde, device, dtype, cfg.pde.validation_points)
    final_metrics = compute_error_metrics(pred, exact)

    plot_training_curves(metrics, run_path)
    plot_solution_comparison(x, pred, exact, run_path)

    summary = generate_run_summary(run_path, metrics, final_metrics, cfg)
    with open(run_path / 'run_summary.json', 'w') as f:
        json.dump(summary, f, indent=2)

    print(f"Evaluation complete. Summary saved to {run_path / 'run_summary.json'}")
    print(f"Final relative L2: {final_metrics['relative_l2']:.6f}")
    print(f"Converged: {summary['converged']}")


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--run-dir', required=True, help='Path to run directory')
    args = parser.parse_args()
    main(args.run_dir)