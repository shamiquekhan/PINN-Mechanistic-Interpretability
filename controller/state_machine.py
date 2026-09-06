"""
Closed-loop controller state machine.

States:  idle → warning → confirmed → cooldown → idle

On each training checkpoint:
  1. Compute monitor score from past-only features.
  2. If score >= threshold → enter warning.
  3. If warning persists for confirmation_steps → confirmed → apply action.
  4. Enter cooldown after each action.
  5. If validation-grid rel L2 degrades relative to the pre-action value
     for `degradation_patience` consecutive ticks → ROLLBACK to the stored
     pre-action lambdas (a real restore, not a no-op).

Every event records the current rel_l2 so rescue/recovery claims are always
verifiable from controller_events.jsonl alone.
"""
from __future__ import annotations
from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Dict, List, Optional, Tuple
import json
from pathlib import Path

from pinn_logging.io import append_jsonl


class ControllerState(Enum):
    IDLE      = auto()
    WARNING   = auto()
    CONFIRMED = auto()
    COOLDOWN  = auto()


@dataclass
class ControllerConfig:
    alarm_threshold:         float = 0.6
    confirmation_steps:      int   = 2
    cooldown_steps:         int   = 200
    max_interventions:      int   = 10
    rollback_on_degradation: bool = True
    degradation_patience:   int   = 50
    degradation_tolerance:   float = 1.10   # post-action best L2 must beat
                                           # pre-action L2 * this factor
    bc_rebalance_factor:    float = 4.0     # multiply λ_bc by this per action
    max_lambda_bc:          float = 1000.0
    verbose:                bool = True


@dataclass
class ControllerEvent:
    step:          int
    state:         str
    action:        Optional[str]  = None
    score:         float          = 0.0
    rel_l2:        float          = 0.0        # ← metric under management
    pre_lambda_pde: Optional[float] = None
    pre_lambda_bc:  Optional[float] = None
    post_lambda_pde: Optional[float] = None
    post_lambda_bc:  Optional[float] = None
    rollback:      bool           = False
    note:          str            = ""


class PINNController:
    """Deterministic state-machine controller for adaptive PINN training.

    Call `controller.step(step, monitor_score, rel_l2, lambda_pde, lambda_bc,
    failure_class)` at each checkpoint.  Returns the (possibly modified)
    (lambda_pde, lambda_bc) and an event log entry.
    """

    def __init__(self, cfg: ControllerConfig, out_dir: Optional[Path] = None):
        self.cfg = cfg
        self.out_dir = Path(out_dir) if out_dir else None
        self.state = ControllerState.IDLE
        self._warning_count = 0
        self._cooldown_remaining = 0
        self._n_interventions = 0

        # ---- Rollback bookkeeping (real pre-action snapshots) ----
        self._pre_action_l2: Optional[float] = None
        self._pre_action_lambdas: Optional[Tuple[float, float]] = None
        self._degradation_count = 0

        # ---- Outcome tracking for honest rescue reporting ----
        self.best_rel_l2: float = float("inf")
        self.best_lambdas: Optional[Tuple[float, float]] = None
        self._last_action_step: Optional[int] = None

        self.event_log: List[ControllerEvent] = []

    # ------------------------------------------------------------------
    # Main step
    # ------------------------------------------------------------------

    def step(
        self,
        step: int,
        monitor_score: float,
        rel_l2: float,
        lambda_pde: float,
        lambda_bc: float,
        failure_class: str = "unknown",
    ) -> Tuple[float, float, ControllerEvent]:
        """Process one controller tick.

        Returns
        -------
        (new_lambda_pde, new_lambda_bc, event)
        """
        cfg = self.cfg
        event = ControllerEvent(
            step=step, state=self.state.name, score=monitor_score,
            rel_l2=rel_l2,
            pre_lambda_pde=lambda_pde, pre_lambda_bc=lambda_bc,
        )

        # ---- Track best-ever state (for summaries / honest rescue claims) ----
        if rel_l2 < self.best_rel_l2:
            self.best_rel_l2 = rel_l2
            self.best_lambdas = (lambda_pde, lambda_bc)

        # ---- Degradation check while an action is in effect ----
        if self.state == ControllerState.COOLDOWN and self._pre_action_l2 is not None:
            if cfg.rollback_on_degradation:
                if rel_l2 > self._pre_action_l2 * cfg.degradation_tolerance:
                    self._degradation_count += 1
                else:
                    self._degradation_count = 0

                if self._degradation_count >= cfg.degradation_patience:
                    # REAL rollback: restore stored pre-action lambdas.
                    lambda_pde, lambda_bc = self._pre_action_lambdas
                    event.rollback = True
                    event.note = (
                        f"Rollback at step {step}: rel_l2={rel_l2:.4f} stayed worse than "
                        f"pre-action {self._pre_action_l2:.4f} for {self._degradation_count} ticks; "
                        f"restored lambdas=({lambda_pde:.3f}, {lambda_bc:.3f})."
                    )
                    self._reset_after_rollback()
                    event.post_lambda_pde = lambda_pde
                    event.post_lambda_bc = lambda_bc
                    self._log(event)
                    return lambda_pde, lambda_bc, event

        # ---- State transitions ----
        if self.state == ControllerState.COOLDOWN:
            self._cooldown_remaining -= 1
            if self._cooldown_remaining <= 0:
                self.state = ControllerState.IDLE
                self._pre_action_l2 = None        # action window closed
                self._degradation_count = 0
            event.state = self.state.name
            event.post_lambda_pde = lambda_pde
            event.post_lambda_bc = lambda_bc
            self._log(event)
            return lambda_pde, lambda_bc, event

        if self.state == ControllerState.IDLE:
            if monitor_score >= cfg.alarm_threshold:
                self.state = ControllerState.WARNING
                self._warning_count = 1
            event.state = self.state.name
            self._log(event)
            return lambda_pde, lambda_bc, event

        if self.state == ControllerState.WARNING:
            if monitor_score >= cfg.alarm_threshold:
                self._warning_count += 1
                if self._warning_count >= cfg.confirmation_steps:
                    self.state = ControllerState.CONFIRMED
            else:
                self.state = ControllerState.IDLE
                self._warning_count = 0

        if self.state == ControllerState.CONFIRMED:
            if self._n_interventions >= cfg.max_interventions:
                self.state = ControllerState.IDLE
                event.note = "Intervention budget exhausted."
            else:
                # Snapshot the pre-action state for rollback.
                self._pre_action_l2 = rel_l2
                self._pre_action_lambdas = (lambda_pde, lambda_bc)
                self._degradation_count = 0

                lambda_pde, lambda_bc, action_name = self._apply_action(
                    failure_class, lambda_pde, lambda_bc
                )
                self._n_interventions += 1
                self._last_action_step = step
                self.state = ControllerState.COOLDOWN
                self._cooldown_remaining = cfg.cooldown_steps
                self._warning_count = 0
                event.action = action_name

        event.state = self.state.name
        event.post_lambda_pde = lambda_pde
        event.post_lambda_bc = lambda_bc
        self._log(event)

        if cfg.verbose and event.action:
            print(f"[Controller step={step}] {event.action} | "
                  f"λ_pde={lambda_pde:.3f} λ_bc={lambda_bc:.3f} | "
                  f"score={monitor_score:.3f} | rel_l2={rel_l2:.4f}")

        return lambda_pde, lambda_bc, event

    def _reset_after_rollback(self):
        self.state = ControllerState.IDLE
        self._warning_count = 0
        self._cooldown_remaining = 0
        self._pre_action_l2 = None
        self._pre_action_lambdas = None
        self._degradation_count = 0

    # ------------------------------------------------------------------
    # Actions (one per failure class, bounded)
    # ------------------------------------------------------------------

    def _apply_action(
        self, failure_class: str, lambda_pde: float, lambda_bc: float
    ) -> Tuple[float, float, str]:
        cfg = self.cfg

        if failure_class in ("boundary_starvation", "unknown"):
            # Aggressive but bounded rebalance: the failure regime is
            # λ_pde=100, λ_bc=0.01.  Doubling λ_bc from 0.01 needs ~13 actions
            # to reach parity — structurally underpowered within any cooldown.
            # Multiply λ_bc by bc_rebalance_factor per action instead, capped.
            new_bc = min(lambda_bc * cfg.bc_rebalance_factor, cfg.max_lambda_bc)
            return lambda_pde, new_bc, f"increase_lambda_bc ({lambda_bc:.4f} -> {new_bc:.4f})"

        elif failure_class == "gradient_conflict":
            # GradNorm-style rebalance toward equal contributions.
            total = lambda_pde + lambda_bc
            new_pde = min(max(total / 2.0, 1e-6), cfg.max_lambda_bc)
            new_bc  = min(max(total / 2.0, 1e-6), cfg.max_lambda_bc)
            return new_pde, new_bc, f"gradnorm_rebalance (pde={new_pde:.4f}, bc={new_bc:.4f})"

        elif failure_class == "collocation_starvation":
            return lambda_pde, lambda_bc, "trigger_resample"

        elif failure_class == "spectral_suppression":
            return lambda_pde, lambda_bc, "trigger_fourier_features"

        else:
            return lambda_pde, lambda_bc, "no_action"

    # ------------------------------------------------------------------
    # Logging
    # ------------------------------------------------------------------

    def _log(self, event: ControllerEvent):
        self.event_log.append(event)
        if self.out_dir is not None:
            append_jsonl(
                self.out_dir / "controller_events.jsonl",
                {k: v for k, v in vars(event).items()},
            )

    def summary(self) -> Dict:
        return {
            "total_steps":         len(self.event_log),
            "n_interventions":     self._n_interventions,
            "n_rollbacks":         sum(1 for e in self.event_log if e.rollback),
            "final_state":         self.state.name,
            "best_rel_l2":         None if self.best_rel_l2 == float("inf") else self.best_rel_l2,
            "best_lambdas":        list(self.best_lambdas) if self.best_lambdas else None,
            "actions":             [e.action for e in self.event_log if e.action],
            "final_rel_l2":        self.event_log[-1].rel_l2 if self.event_log else None,
            "first_rel_l2":        self.event_log[0].rel_l2 if self.event_log else None,
        }