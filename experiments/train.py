import argparse, json
from pathlib import Path
import torch
from pinn.config import load_config
from pinn.reproducibility import set_seed
from pinn.pdes import Poisson1D
from pinn.model import MLP
from pinn.gradients import compute_per_loss_gradients, log_gradient_stats
from pinn.activations import ActivationLogger, create_probe_points
from pinn.diagnostics import DiagnosticsLogger, SpatialBinConfig
from pinn_logging.io import append_jsonl, save_checkpoint


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--config', required=True)
    p.add_argument('--steps', type=int)
    p.add_argument('--resume')
    p.add_argument('--log-gradients', action='store_true', default=False)
    p.add_argument('--grad-log-every', type=int, default=100)
    p.add_argument('--log-activations', action='store_true', default=False)
    p.add_argument('--act-log-every', type=int, default=100)
    p.add_argument('--act-save-raw', action='store_true', default=False)
    p.add_argument('--log-diagnostics', action='store_true', default=False)
    p.add_argument('--diag-log-every', type=int, default=100)
    args = p.parse_args()

    cfg = load_config(args.config)
    set_seed(cfg.run.seed, cfg.run.deterministic)
    device = torch.device(cfg.run.device)
    dtype = getattr(torch, cfg.run.dtype)

    pde = Poisson1D(*cfg.pde.domain, cfg.pde.source, *cfg.pde.boundary_values)
    model = MLP(cfg.model.input_dim, cfg.model.output_dim, cfg.model.hidden_layers, cfg.model.activation).to(device=device, dtype=dtype)
    opt = torch.optim.Adam(model.parameters(), lr=cfg.training.learning_rate)

    out = Path(cfg.run.output_dir) / cfg.run.name
    out.mkdir(parents=True, exist_ok=True)
    manifest = cfg.model_dump(mode='json')
    (out / 'config.json').write_text(json.dumps(manifest, indent=2))

    start = 0
    if args.resume:
        ck = torch.load(args.resume, map_location=device)
        model.load_state_dict(ck['model'])
        opt.load_state_dict(ck['optimizer'])
        start = ck['step'] + 1

    steps = args.steps or cfg.training.steps

    probe_config = create_probe_points(cfg)
    act_logger = ActivationLogger(
        model,
        layer_indices=cfg.logging.activation_layers,
        probe_config=probe_config,
        log_every=args.act_log_every,
        save_raw=args.act_save_raw
    ) if args.log_activations else None
    
    bin_config = SpatialBinConfig(
        n_bins=10,
        domain_left=pde.left,
        domain_right=pde.right
    )
    diag_logger = DiagnosticsLogger(model, pde, bin_config, log_every=args.diag_log_every) if args.log_diagnostics else None
    
    if act_logger:
        act_logger.setup(device, dtype)

    for step in range(start, steps):
        x = torch.rand(cfg.training.interior_points, 1, device=device, dtype=dtype) * (pde.right - pde.left) + pde.left
        r = pde.residual(model, x)
        lp = (r ** 2).mean()

        xb = pde.boundary_points(device, dtype)
        ub = model(xb)
        target = torch.tensor([[pde.left_bc], [pde.right_bc]], device=device, dtype=dtype)
        lb = ((ub - target) ** 2).mean()

        loss = cfg.training.lambda_pde * lp + cfg.training.lambda_bc * lb
        opt.zero_grad()
        
        if args.log_gradients and (step % args.grad_log_every == 0 or step == steps - 1):
            loss_dict = {'pde': lp * cfg.training.lambda_pde, 'bc': lb * cfg.training.lambda_bc}
            grads = compute_per_loss_gradients(model, loss_dict)
            grad_stats = log_gradient_stats(grads)
            grad_rec = {'step': step, 'gradient_stats': grad_stats}
            append_jsonl(out / 'gradients.jsonl', grad_rec)
        
        loss.backward()
        opt.step()

        if act_logger is not None:
            act_stats = act_logger.maybe_log(step)
            if act_stats:
                append_jsonl(out / 'activations.jsonl', act_stats)

        if diag_logger is not None:
            probe_acts = act_logger.recorder.get_all_activations() if act_logger and act_logger.recorder else None
            diag_stats = diag_logger.maybe_log(step, collocation_points=x.detach(), probe_activations=probe_acts)
            if diag_stats:
                append_jsonl(out / 'diagnostics.jsonl', diag_stats)

        if step % cfg.training.log_every == 0 or step == steps - 1:
            xv = pde.validation_grid(cfg.pde.validation_points, device, dtype)
            pred = model(xv)
            exact = pde.exact(xv)
            err = torch.linalg.vector_norm(pred - exact) / torch.linalg.vector_norm(exact)
            rec = {
                'step': step,
                'loss': float(loss),
                'loss_pde': float(lp),
                'loss_bc': float(lb),
                'relative_l2': float(err)
            }
            append_jsonl(out / 'metrics.jsonl', rec)
            print(rec)

        if step % cfg.training.checkpoint_every == 0 or step == steps - 1:
            save_checkpoint(out / f'checkpoint_{step:07d}.pt', model, opt, step, manifest)


if __name__ == '__main__':
    main()