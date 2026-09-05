import torch
from torch import nn
from typing import Dict, List, Optional, Callable
from dataclasses import dataclass, field
import hashlib
import json


@dataclass
class ProbePointConfig:
    coordinates: List[List[float]]
    layer_indices: List[int]
    
    def to_hash(self) -> str:
        data = json.dumps({
            'coordinates': self.coordinates,
            'layer_indices': self.layer_indices
        }, sort_keys=True)
        return hashlib.sha256(data.encode()).hexdigest()[:16]
    
    def to_tensor(self, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
        return torch.tensor(self.coordinates, device=device, dtype=dtype)


class ActivationHook:
    def __init__(self, module: nn.Module, layer_name: str):
        self.module = module
        self.layer_name = layer_name
        self.activations: Optional[torch.Tensor] = None
        self.hook = module.register_forward_hook(self._hook_fn)
    
    def _hook_fn(self, module, input, output):
        self.activations = output.detach()
    
    def get_activations(self) -> Optional[torch.Tensor]:
        return self.activations
    
    def clear(self):
        self.activations = None
    
    def remove(self):
        self.hook.remove()


class ActivationRecorder:
    def __init__(self, model: nn.Module, layer_names: List[str]):
        self.model = model
        self.layer_names = layer_names
        self.hooks: Dict[str, ActivationHook] = {}
        self._register_hooks()
    
    def _register_hooks(self):
        for name in self.layer_names:
            module = dict(self.model.named_modules()).get(name)
            if module is None:
                raise ValueError(f"Layer {name} not found in model")
            self.hooks[name] = ActivationHook(module, name)
    
    def get_activations(self, layer_name: str) -> Optional[torch.Tensor]:
        if layer_name in self.hooks:
            return self.hooks[layer_name].get_activations()
        return None
    
    def get_all_activations(self) -> Dict[str, torch.Tensor]:
        return {name: hook.get_activations() for name, hook in self.hooks.items()}
    
    def clear(self):
        for hook in self.hooks.values():
            hook.clear()
    
    def remove_all(self):
        for hook in self.hooks.values():
            hook.remove()
        self.hooks.clear()
    
    def __enter__(self):
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        self.remove_all()


def get_layer_names(model: nn.Module) -> List[str]:
    return [name for name, module in model.named_modules() 
            if isinstance(module, nn.Linear)]


def create_probe_points(config: 'ExperimentConfig') -> ProbePointConfig:
    from pinn.config import ExperimentConfig
    pde = config.pde
    if pde.name == 'poisson_1d':
        n_probe = 50
        x_probe = torch.linspace(pde.domain[0], pde.domain[1], n_probe).reshape(-1, 1).tolist()
    else:
        x_probe = torch.linspace(pde.domain[0], pde.domain[1], 50).reshape(-1, 1).tolist()
    
    hidden_layers = config.model.hidden_layers
    n_layers = len(hidden_layers)
    layer_indices = [0, n_layers // 2, n_layers - 1] if n_layers > 1 else [0]
    
    return ProbePointConfig(coordinates=x_probe, layer_indices=layer_indices)


def record_activations_at_probes(
    model: nn.Module,
    probe_points: torch.Tensor,
    layer_names: List[str]
) -> Dict[str, torch.Tensor]:
    recorder = ActivationRecorder(model, layer_names)
    with torch.no_grad():
        _ = model(probe_points)
    activations = recorder.get_all_activations()
    recorder.remove_all()
    return activations


def compute_activation_statistics(activations: torch.Tensor) -> Dict[str, float]:
    if activations is None:
        return {}
    act_flat = activations.flatten()
    return {
        'mean': act_flat.mean().item(),
        'std': act_flat.std().item(),
        'max': act_flat.max().item(),
        'min': act_flat.min().item(),
        'sparsity': (act_flat == 0).float().mean().item(),
        'norm': act_flat.norm(2).item() / act_flat.numel() ** 0.5,
    }


def compute_spatial_activation_map(activations: torch.Tensor, probe_coords: torch.Tensor) -> Dict:
    if activations is None or activations.ndim < 2:
        return {}
    
    if activations.ndim == 2:
        n_points, n_channels = activations.shape
    else:
        n_points = activations.shape[0]
        n_channels = activations.shape[1]
    
    spatial_stats = {}
    
    for c in range(min(n_channels, 10)):
        if activations.ndim == 2:
            channel_acts = activations[:, c]
        else:
            channel_acts = activations[:, c].flatten()
        spatial_stats[f'channel_{c}'] = {
            'mean': channel_acts.mean().item(),
            'std': channel_acts.std().item(),
            'max': channel_acts.max().item(),
            'min': channel_acts.min().item(),
        }
    
    return spatial_stats


class ActivationLogger:
    def __init__(
        self,
        model: nn.Module,
        layer_indices: List[int],
        probe_config: ProbePointConfig,
        log_every: int = 100,
        save_raw: bool = False
    ):
        self.model = model
        self.layer_indices = layer_indices
        self.probe_config = probe_config
        self.log_every = log_every
        self.save_raw = save_raw
        self.layer_names = get_layer_names(model)
        self.probe_points = None
        self.recorder: Optional[ActivationRecorder] = None
    
    def setup(self, device: torch.device, dtype: torch.dtype):
        self.probe_points = self.probe_config.to_tensor(device, dtype)
        self.recorder = ActivationRecorder(self.model, [self.layer_names[i] for i in self.layer_indices])
    
    def maybe_log(self, step: int) -> Dict:
        if self.recorder is None:
            return {}
        
        if step % self.log_every != 0:
            return {}
        
        with torch.no_grad():
            _ = self.model(self.probe_points)
        
        activations = self.recorder.get_all_activations()
        self.recorder.clear()
        
        result = {
            'step': step,
            'probe_hash': self.probe_config.to_hash(),
            'activations': {}
        }
        
        for layer_name, acts in activations.items():
            if acts is None:
                continue
            
            stats = compute_activation_statistics(acts)
            spatial = compute_spatial_activation_map(acts, self.probe_points)
            
            layer_data = {
                'statistics': stats,
                'spatial': spatial,
                'shape': list(acts.shape)
            }
            
            if self.save_raw:
                layer_data['raw'] = acts.cpu().float().numpy().tolist()
            
            result['activations'][layer_name] = layer_data
        
        return result
    
    def __enter__(self):
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        if self.recorder:
            self.recorder.remove_all()