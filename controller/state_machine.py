"""
Closed-loop controller state machine.

States:  idle → warning → confirmed → acting → cooldown → idle

On each training checkpoint:
  1. Compute monitor score from past-only features.
  2. If score > threshold → enter warning.
  3. If warning persists for confirmation_steps → enter confirmed.
  4. If confirmed → apply bounded corrective action, enter cooldown.
  5. After cooldown_steps → return to idle.
  6. If validation error degrades during/after action → rollback.
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
    ACTING    = auto()
    COOLDOWN  = auto()


@dataclass
class ControllerConfig:
    alarm_threshold:       float = 0.6
    confirmation_steps:    int   = 2
    cooldown_steps:        int   = 200
    max_interventions:     int   = 10
    rollback_on_degradation: bool = True
    degradation_patience:  int   = 100
    verbose:               bool  = True


@dataclass
class ControllerEvent:
    step:          int
    state:         str
    action:        Optional[str]  = None
    score:         float          = 0.0
    pre_lambda_pde: Optional[float] = None
    pre_lambda_bc:  Optional[float] = None
    post_lambda_pde: Optional[float] = None
    post_lambda_bc:  Optional[float] = None
    rollback:      bool           = False
    note:          str            = ""


class PINNController:
    """Deterministic state-machine controller for adaptive PINN training.

    Usage
    -----
    Call `controller.step(step, monitor_score, rel_l2, training_config)` at each
    checkpoint.  The controller modifies `training_config` in-place (lambda_pde,
    lambda_bc) and returns the (possibly modified) config and an event log entry.
    """

    def __init__(self, cfg: ControllerConfig, out_dir: Optional[Path] = None):
        self.cfg = cfg
        self.out_dir = Path(out_dir) if out_dir else None
        self.state = ControllerState.IDLE
        self._warning_count = 0
        self._cooldown_remaining = 0
        self._n_interventions = 0
        self._pre_action_l2: Optional[float] = None
        self._post_action_l2_buffer: List[float] = []
        self._action_step: Optional[int] = None
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
        event = ControllerEvent(step=step, state=self.state.name, score=monitor_score,
                                pre_lambda_pde=lambda_pde, pre_lambda_bc=lambda_bc)

        # ---- Track degradation after an action ----
        if self.state in (ControllerState.ACTING, ControllerState.COOLDOWN):
            self._post_action_l2_buffer.append(rel_l2)
            if (cfg.rollback_on_degradation and
                    self._pre_action_l2 is not None and
                    len(self._post_action_l2_buffer) >= cfg.degradation_patience and
                    min(self._post_action_l2_buffer) > self._pre_action_l2 * 1.5):
                lambda_pde = event.pre_lambda_pde  # rollback
                lambda_bc  = event.pre_lambda_bc
                self.state = ControllerState.IDLE
                self._warning_count = 0
                self._cooldown_remaining = 0
                self._post_action_l2_buffer.clear()
                event.rollback = True
                event.note = "Rollback: performance degraded after intervention."
                self._log(event)
                return lambda_pde, lambda_bc, event

        # ---- State machine transitions ----
        if self.state == ControllerState.COOLDOWN:
            self._cooldown_remaining -= 1
            if self._cooldown_remaining <= 0:
                self.state = ControllerState.IDLE
                self._post_action_l2_buffer.clear()
            event.state = self.state.name
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
                self._pre_action_l2 = rel_l2
                lambda_pde, lambda_bc, action_name = self._apply_action(
                    failure_class, lambda_pde, lambda_bc
                )
                self._n_interventions += 1
                self.state = ControllerState.COOLDOWN
                self._cooldown_remaining = cfg.cooldown_steps
                self._warning_count = 0
                event.action = action_name

        event.state = self.state.name
        event.post_lambda_pde = lambda_pde
        event.post_lambda_bc  = lambda_bc
        self._log(event)

        if cfg.verbose and event.action:
            print(f"[Controller step={step}] {event.action} | "
                  f"λ_pde={lambda_pde:.3f} λ_bc={lambda_bc:.3f} | "
                  f"score={monitor_score:.3f}")

        return lambda_pde, lambda_bc, event

    # ------------------------------------------------------------------
    # Actions (one per failure class, bounded)
    # ------------------------------------------------------------------

    def _apply_action(
        self, failure_class: str, lambda_pde: float, lambda_bc: float
    ) -> Tuple[float, float, str]:
        cfg = self.cfg

        if failure_class in ("boundary_starvation", "unknown"):
            # Increase λ_bc (bounded)
            new_bc = min(lambda_bc * 2.0, 100.0)
            return lambda_pde, new_bc, f"increase_lambda_bc ({lambda_bc:.3f} -> {new_bc:.3f})"

        elif failure_class == "gradient_conflict":
            # Rebalance: push λ_pde and λ_bc toward equal ratio
            total = lambda_pde + lambda_bc
            new_pde = min(total / 2.0, 100.0)
            new_bc  = min(total / 2.0, 100.0)
            return new_pde, new_bc, f"gradnorm_rebalance (pde={new_pde:.3f}, bc={new_bc:.3f})"

        elif failure_class == "collocation_starvation":
            # Signal the trainer to resample (trainer reads action from event log)
            return lambda_pde, lambda_bc, "trigger_resample"

        elif failure_class == "spectral_suppression":
            # Signal trainer to add Fourier features (architecture change)
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
            "total_steps":        len(self.event_log),
            "n_interventions":    self._n_interventions,
            "n_rollbacks":        sum(1 for e in self.event_log if e.rollback),
            "final_state":        self.state.name,
            "actions":            [e.action for e in self.event_log if e.action],
        }
