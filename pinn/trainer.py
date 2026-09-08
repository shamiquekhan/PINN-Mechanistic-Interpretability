"""
Reusable PINNTrainer class — wraps the training loop so experiments,
qualification scripts, and the closed-loop controller can all share
one authoritative implementation.
"""
from __future__ import annotations
import logging
import time
from pathlib import Path
from typing import Callable, Dict, List, Optional, Tuple

import torch

from pinn.pdes import BasePDE
from pinn.model import MLP
from pinn.gradients import compute_per_loss_gradients, log_gradient_stats
from pinn.activations import ActivationLogger, create_probe_points

logger = logging.getLogger(__name__)
from pinn.diagnostics import DiagnosticsLogger, SpatialBinConfig
from pinn_logging.io import append_jsonl, save_checkpoint


class PINNTrainer:
    """Configurable PINN training loop.

    Parameters
    ----------
    model:          MLP instance (already moved to device/dtype).
    pde:            BasePDE instance.
    optimizer:      Pre-constructed torch optimizer.
    cfg:            Full ExperimentConfig.
    out_dir:        Directory for JSONL logs and checkpoints.
    intervention_fn: Optional callable(model, step) -> None applied after each
                    forward pass but before backward.  Used by the controller.
    """

    def __init__(
        self,
        model: MLP,
        pde: BasePDE,
        optimizer: torch.optim.Optimizer,
        cfg,                          # ExperimentConfig
        out_dir: Path,
        intervention_fn: Optional[Callable] = None,
    ):
        self.model = model
        self.pde = pde
        self.opt = optimizer
        self.cfg = cfg
        self.out_dir = Path(out_dir)
        self.out_dir.mkdir(parents=True, exist_ok=True)
        self.intervention_fn = intervention_fn

        tcfg = cfg.training
        lcfg = cfg.logging
        rcfg = cfg.run

        # M3 fix: refuse to silently downgrade cuda->cpu (a silent 10-100x
        # slowdown is worse than a loud failure) and validate dtype clearly.
        requested = torch.device(rcfg.device)
        if requested.type == "cuda" and not torch.cuda.is_available():
            raise RuntimeError(
                f"config requests device '{requested}' but CUDA is unavailable "
                "on this machine — refusing to silently downgrade to CPU. "
                "Set run.device='cpu' explicitly if CPU training is intended.")
        self.device = requested
        if not hasattr(torch, rcfg.dtype):
            raise ValueError(
                f"unknown dtype '{rcfg.dtype}' in config "
                f"(expected e.g. 'float32', 'float64')")
        self.dtype = getattr(torch, rcfg.dtype)

        # Activation logger (optional)
        self.act_logger: Optional[ActivationLogger] = None
        if lcfg.save_activations:
            probe_cfg = create_probe_points(cfg)
            self.act_logger = ActivationLogger(
                model,
                layer_indices=lcfg.activation_layers,
                probe_config=probe_cfg,
                log_every=tcfg.log_every,
                save_raw=False,
            )
            self.act_logger.setup(self.device, self.dtype)

        # Diagnostics logger (optional)
        self.diag_logger: Optional[DiagnosticsLogger] = None
        if lcfg.log_diagnostics:
            lo, hi = pde.domain
            bin_cfg = SpatialBinConfig(n_bins=10, domain_left=lo, domain_right=hi)
            self.diag_logger = DiagnosticsLogger(model, pde, bin_cfg, log_every=lcfg.diag_log_every)

    # ------------------------------------------------------------------
    # Training loop
    # ------------------------------------------------------------------

    def train(self, start: int = 0, steps: Optional[int] = None,
              get_lambdas: Optional[Callable] = None,
              resample_fn: Optional[Callable] = None,
              post_step_fn: Optional[Callable] = None) -> List[Dict]:
        """Run training from `start` to `steps` (exclusive).

        Returns a list of step records for downstream analysis.

        H3 seams (external review):
          get_lambdas: callable(step) -> (lambda_pde, lambda_bc), applied
              BEFORE loss construction each step so controller lambda
              updates take effect on the step they are decided (the old
              intervention_fn path mutated tcfg after loss construction,
              landing a step late).  Falls back to the config values.
          resample_fn: callable(x_current, step) -> x_new, invoked before
              the forward pass; used by the controller's trigger_resample
              action (H4).
          post_step_fn: callable(model, step, loss, rel_l2) -> None, fired
              after the optimizer step each step. This is the CONTROLLER
              OBSERVATION seam: the controller sees the step's actual loss
              and rel_l2 and decides lambdas that take effect next step
              via get_lambdas — exactly the timing of the original
              stage-7 loop (H3).
        intervention_fn (constructor arg): fires BEFORE the forward pass
        (moved from after loss computation) so weight-affecting
        interventions are not stale.  Lambda-affecting controllers should
        prefer get_lambdas.
        """
        cfg = self.cfg
        tcfg = cfg.training
        lcfg = cfg.logging
        device = self.device
        dtype = self.dtype

        steps = steps if steps is not None else tcfg.steps  # M5: steps=0 must mean 0
        records: List[Dict] = []

        x = self.pde.sample_interior(
            tcfg.interior_points,
            device,
            dtype,
            spatial_bias=getattr(tcfg, "spatial_bias", None),
        )

        for step in range(start, steps):
            t0 = time.perf_counter()

            # ---- Resample collocation if configured (or controller-triggered) ----
            if tcfg.resample_every > 0 and step % tcfg.resample_every == 0:
                x = self.pde.sample_interior(tcfg.interior_points, device, dtype,
                                             spatial_bias=getattr(tcfg, 'spatial_bias', None))
            if resample_fn is not None:
                x = resample_fn(x, step)

            # ---- Intervention BEFORE forward (H3 fix: was after loss ----
            # construction, so weight mutations rode a stale loss) ----
            if self.intervention_fn is not None:
                self.intervention_fn(self.model, step)

            # ---- Per-step lambdas (H3 seam: controller-visible) ----
            if get_lambdas is not None:
                lam_pde, lam_bc = get_lambdas(step)
            else:
                lam_pde, lam_bc = tcfg.lambda_pde, tcfg.lambda_bc

            # ---- PDE residual loss ----
            r = self.pde.residual(self.model, x)
            lp = (r ** 2).mean()

            # ---- Boundary loss ----
            bc_res = self.pde.boundary_residual(self.model)
            lb = bc_res.mean()

            loss = lam_pde * lp + lam_bc * lb

            # ---- Gradient logging ----
            if lcfg.log_gradients and step % lcfg.grad_log_every == 0:
                loss_dict = {
                    "pde": lam_pde * lp,
                    "bc":  lam_bc  * lb,
                }
                grads = compute_per_loss_gradients(self.model, loss_dict)
                grad_stats = log_gradient_stats(grads)
                append_jsonl(self.out_dir / "gradients.jsonl", {"step": step, **grad_stats})

            # ---- Backward + step ----
            self.opt.zero_grad()
            loss.backward()
            self.opt.step()

            # ---- Post-step observation seam (H3: controller decides here) ----
            if post_step_fn is not None:
                xv = self.pde.validation_grid(cfg.pde.validation_points, device, dtype)
                with torch.no_grad():
                    pred = self.model(xv)
                    exact = self.pde.exact(xv)
                if exact is not None:
                    rel_now = float(torch.linalg.vector_norm(pred - exact)
                                    / torch.linalg.vector_norm(exact).clamp(min=1e-12))
                else:
                    rel_now = float("nan")
                post_step_fn(self.model, step, float(loss), rel_now)

            # ---- Activation logging ----
            if self.act_logger is not None:
                act_stats = self.act_logger.maybe_log(step)
                if act_stats:
                    append_jsonl(self.out_dir / "activations.jsonl", act_stats)

            # ---- Diagnostics logging ----
            if self.diag_logger is not None:
                probe_acts = (
                    self.act_logger.recorder.get_all_activations()
                    if self.act_logger and self.act_logger.recorder
                    else None
                )
                diag = self.diag_logger.maybe_log(step, collocation_points=x.detach(), probe_activations=probe_acts)
                if diag:
                    append_jsonl(self.out_dir / "diagnostics.jsonl", diag)

            # ---- Metrics logging ----
            if step % tcfg.log_every == 0 or step == steps - 1:
                xv = self.pde.validation_grid(cfg.pde.validation_points, device, dtype)
                with torch.no_grad():
                    pred = self.model(xv)
                exact = self.pde.exact(xv)
                if exact is not None:
                    err = torch.linalg.vector_norm(pred - exact) / torch.linalg.vector_norm(exact)
                    rel_l2 = float(err)
                else:
                    rel_l2 = float("nan")

                rec = {
                    "step": step,
                    "loss": float(loss),
                    "loss_pde": float(lp),
                    "loss_bc": float(lb),
                    "relative_l2": rel_l2,
                    "wall_time": time.perf_counter() - t0,
                }
                append_jsonl(self.out_dir / "metrics.jsonl", rec)
                records.append(rec)
                logger.info("step=%s loss=%.6f rel_l2=%.6f",
                            step, rec["loss"], rel_l2)

            # ---- Checkpointing (M4: skip the step-0 artifact) ----
            if (step > 0 and step % tcfg.checkpoint_every == 0) or step == steps - 1:
                manifest = cfg.model_dump(mode="json")
                save_checkpoint(
                    self.out_dir / f"checkpoint_{step:07d}.pt",
                    self.model, self.opt, step, manifest,
                )

        return records

    # ------------------------------------------------------------------
    # Resume support
    # ------------------------------------------------------------------

    def load_checkpoint(self, path: Path) -> int:
        """Load model + optimizer state; returns the step to resume from.

        M2 fix: weights_only=True (these are first-party artifacts, so the
        restrictive unpickler is safe here — dict/str/int/float/tensors
        only; the config manifest is a plain dict).
        """
        ck = torch.load(path, map_location=self.device, weights_only=True)
        self.model.load_state_dict(ck["model"])
        self.opt.load_state_dict(ck["optimizer"])
        return ck["step"] + 1

    def __enter__(self):
        return self

    def __exit__(self, *args):
        if self.act_logger and self.act_logger.recorder:
            self.act_logger.recorder.remove_all()
