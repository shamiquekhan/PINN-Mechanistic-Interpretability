"""
Typed configuration for the PINN mechanistic interpretability benchmark.
Covers the full pipeline: PINN training, SAE, interventions, monitoring, controller.
"""
from __future__ import annotations
from pathlib import Path
from typing import Literal, Optional, List
import yaml
from pydantic import BaseModel, Field, ConfigDict


# ---------------------------------------------------------------------------
# Run / environment
# ---------------------------------------------------------------------------

class RunConfig(BaseModel):
    name: str
    output_dir: str = "runs"
    seed: int = 0
    deterministic: bool = True
    device: str = "cuda"           # default: GPU
    dtype: Literal["float32", "float64"] = "float32"
    failure_label: Literal[
        "unlabeled",
        "success",
        "gradient_conflict",
        "boundary_starvation",
        "spectral_suppression",
        "collocation_starvation",
    ] = "unlabeled"


# ---------------------------------------------------------------------------
# PDE
# ---------------------------------------------------------------------------

class PDEConfig(BaseModel):
    name: Literal[
        "poisson_1d", "poisson_2d", "advection_1d", "reaction_diffusion_1d",
        "advection_diffusion_2d", "reaction_diffusion_2d",
        "burgers_1d", "allen_cahn_1d",
    ]
    domain: list[float] = Field(min_length=2, max_length=2)
    domain_y: Optional[list[float]] = Field(default=None, min_length=2, max_length=2)
    time_domain: list[float] = Field(default=[0.0, 1.0], min_length=2, max_length=2)
    source: float = 1.0         # primary scalar PDE param (source / speed / reaction)
    forcing: float = 0.0        # secondary forcing for advection / reaction-diffusion
    # H5 (external review): explicit per-family fields so configs cannot
    # silently overload 'source'. When set, they take precedence and must
    # not conflict with the legacy alias.
    speed: Optional[float] = None          # advection_1d / advection_diffusion_2d x-speed
    reaction_rate: Optional[float] = None  # reaction-diffusion families
    diffusion: float = 0.01     # ε for reaction-diffusion (ignored for other PDEs)
    boundary_values: list[float] = Field(min_length=2, max_length=2)
    validation_points: int = Field(default=1001, ge=10)


# ---------------------------------------------------------------------------
# Model
# ---------------------------------------------------------------------------

class ModelConfig(BaseModel):
    input_dim: int = 1
    output_dim: int = 1
    hidden_layers: list[int] = Field(min_length=1)
    activation: Literal["tanh", "relu", "gelu", "silu", "sin"] = "tanh"
    init: Literal["xavier", "kaiming", "default"] = "xavier"
    fourier_embed: bool = False
    fourier_n_freq: int = 32
    fourier_scale: float = 1.0


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------

class TrainingConfig(BaseModel):
    optimizer: Literal["adam", "adamw", "lbfgs"] = "adam"
    learning_rate: float = Field(gt=0)
    steps: int = Field(gt=0)
    interior_points: int = Field(gt=0)
    boundary_points: int = Field(gt=0)
    log_every: int = Field(gt=0)
    checkpoint_every: int = Field(gt=0)
    lambda_pde: float = Field(ge=0)
    lambda_bc: float = Field(ge=0)
    resample_every: int = 0     # 0 = no resampling; >0 = resample every N steps
    spatial_bias: float = Field(default=0.0, ge=0.0, le=1.0)


# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------

class LoggingConfig(BaseModel):
    save_activations: bool = False
    activation_layers: list[int] = [1]
    save_pointwise_residuals: bool = True
    log_gradients: bool = False
    grad_log_every: int = 100
    log_diagnostics: bool = False
    diag_log_every: int = 100


# ---------------------------------------------------------------------------
# SAE (Phase 4)
# ---------------------------------------------------------------------------

class SAEConfig(BaseModel):
    latent_expansion: int = Field(default=4, ge=1)   # multiplier over hidden dim
    sparsity_coeff: float = Field(default=1e-3, ge=0)
    learning_rate: float = Field(default=1e-3, gt=0)
    steps: int = Field(default=5000, gt=0)
    batch_size: int = Field(default=256, gt=0)
    target_layer_index: int = 1    # which hidden layer to train on
    decoder_normalize: bool = True
    dead_feature_window: int = 200


# ---------------------------------------------------------------------------
# Intervention (Phase 6)
# ---------------------------------------------------------------------------

class InterventionConfig(BaseModel):
    amplification_factor: float = 1.5
    n_control_seeds: int = 5
    training_window: int = 200    # steps over which to apply training-time intervention


# ---------------------------------------------------------------------------
# Monitor (Phase 7)
# ---------------------------------------------------------------------------

class MonitorConfig(BaseModel):
    failure_horizon: int = Field(default=500, gt=0)
    history_window: int = Field(default=200, gt=0)
    model_type: Literal["threshold", "logistic"] = "logistic"
    alarm_recall_target: float = 0.8   # tune threshold to this recall on val set


# ---------------------------------------------------------------------------
# Controller (Phase 8)
# ---------------------------------------------------------------------------

class ControllerConfig(BaseModel):
    cooldown_steps: int = 200
    max_interventions: int = 10
    lambda_bc_max: float = 100.0
    lambda_pde_max: float = 100.0
    rollback_on_degradation: bool = True
    degradation_patience: int = 100   # steps to wait before declaring degradation


# ---------------------------------------------------------------------------
# Top-level experiment config
# ---------------------------------------------------------------------------

class ExperimentConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")
    run: RunConfig
    pde: PDEConfig
    model: ModelConfig
    training: TrainingConfig
    logging: LoggingConfig = LoggingConfig()
    sae: Optional[SAEConfig] = None
    intervention: Optional[InterventionConfig] = None
    monitor: Optional[MonitorConfig] = None
    controller: Optional[ControllerConfig] = None


# ---------------------------------------------------------------------------
# Loader
# ---------------------------------------------------------------------------

def load_config(path: str | Path) -> ExperimentConfig:
    with open(path) as f:
        raw = yaml.safe_load(f)
    return ExperimentConfig.model_validate(raw)
