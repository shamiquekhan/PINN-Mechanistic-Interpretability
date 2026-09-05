"""Tests for typed config: loading, validation, unknown-key rejection."""
import pytest
import yaml
import tempfile
from pathlib import Path
from pinn.config import ExperimentConfig, load_config


MINIMAL_CONFIG = {
    "run":      {"name": "test", "device": "cpu", "seed": 0, "deterministic": True, "dtype": "float32"},
    "pde":      {"name": "poisson_1d", "domain": [-1.0, 1.0], "source": 1.0,
                 "forcing": 0.0, "diffusion": 0.01,
                 "boundary_values": [0.0, 0.0], "validation_points": 101},
    "model":    {"input_dim": 1, "output_dim": 1, "hidden_layers": [16, 16], "activation": "tanh",
                 "init": "xavier", "fourier_embed": False, "fourier_n_freq": 32, "fourier_scale": 1.0},
    "training": {"optimizer": "adam", "learning_rate": 1e-3, "steps": 10,
                 "interior_points": 8, "boundary_points": 2,
                 "log_every": 5, "checkpoint_every": 5,
                 "lambda_pde": 1.0, "lambda_bc": 1.0, "resample_every": 0},
}


def _write_yaml(data: dict) -> Path:
    tmp = tempfile.NamedTemporaryFile(suffix=".yaml", delete=False, mode="w")
    yaml.dump(data, tmp)
    tmp.close()
    return Path(tmp.name)


def test_valid_config_loads():
    path = _write_yaml(MINIMAL_CONFIG)
    cfg = load_config(path)
    assert cfg.run.name == "test"
    assert cfg.pde.name == "poisson_1d"
    path.unlink()


def test_unknown_key_rejected():
    bad = dict(MINIMAL_CONFIG)
    bad["unknown_key"] = "value"
    with pytest.raises(Exception):
        ExperimentConfig.model_validate(bad)


def test_negative_lr_rejected():
    bad = {**MINIMAL_CONFIG, "training": {**MINIMAL_CONFIG["training"], "learning_rate": -1e-3}}
    with pytest.raises(Exception):
        ExperimentConfig.model_validate(bad)


def test_all_pde_names_accepted():
    for pde_name, bv_len in [("poisson_1d", 2), ("advection_1d", 2), ("reaction_diffusion_1d", 2)]:
        cfg_dict = dict(MINIMAL_CONFIG)
        cfg_dict["pde"] = {**MINIMAL_CONFIG["pde"], "name": pde_name, "boundary_values": [0.0, 0.0]}
        cfg = ExperimentConfig.model_validate(cfg_dict)
        assert cfg.pde.name == pde_name


def test_failure_label_field():
    cfg_dict = dict(MINIMAL_CONFIG)
    cfg_dict["run"] = {**MINIMAL_CONFIG["run"], "failure_label": "boundary_starvation"}
    cfg = ExperimentConfig.model_validate(cfg_dict)
    assert cfg.run.failure_label == "boundary_starvation"


def test_default_device_is_cuda():
    cfg_dict = {k: v for k, v in MINIMAL_CONFIG.items()}
    cfg_dict["run"] = {k: v for k, v in MINIMAL_CONFIG["run"].items() if k != "device"}
    cfg = ExperimentConfig.model_validate(cfg_dict)
    assert cfg.run.device == "cuda"
