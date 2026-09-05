import torch
import numpy as np
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass
from pinn.pdes import Poisson1D


@dataclass
class SpatialBinConfig:
    n_bins: int = 10
    domain_left: float = -1.0
    domain_right: float = 1.0
    
    def get_bin_edges(self) -> np.ndarray:
        return np.linspace(self.domain_left, self.domain_right, self.n_bins + 1)
    
    def assign_bins(self, coords: torch.Tensor) -> np.ndarray:
        coords_np = coords.flatten().cpu().numpy()
        bin_edges = self.get_bin_edges()
        return np.digitize(coords_np, bin_edges) - 1


def compute_fourier_spectrum(signal: torch.Tensor, domain_length: float) -> Tuple[np.ndarray, np.ndarray]:
    signal_np = signal.detach().cpu().numpy().flatten()
    n = len(signal_np)
    fft_vals = np.fft.rfft(signal_np)
    freqs = np.fft.rfftfreq(n, d=domain_length / n)
    magnitude = np.abs(fft_vals)
    return freqs, magnitude


def compute_error_spectrum(pred: torch.Tensor, exact: torch.Tensor, domain_length: float) -> Dict:
    error = pred - exact
    freqs, mag = compute_fourier_spectrum(error, domain_length)
    
    total_power = np.sum(mag ** 2)
    low_freq_cutoff = len(freqs) // 4
    high_freq_power = np.sum(mag[low_freq_cutoff:] ** 2)
    low_freq_power = np.sum(mag[:low_freq_cutoff] ** 2)
    
    return {
        'frequencies': freqs.tolist(),
        'magnitude': mag.tolist(),
        'total_power': float(total_power),
        'low_freq_power': float(low_freq_power),
        'high_freq_power': float(high_freq_power),
        'high_to_low_ratio': float(high_freq_power / (low_freq_power + 1e-10)),
    }


def compute_spatial_residual_bins(
    model: torch.nn.Module,
    pde: Poisson1D,
    device: torch.device,
    dtype: torch.dtype,
    bin_config: SpatialBinConfig
) -> Dict:
    n_test = 1000
    x = pde.validation_grid(n_test, device, dtype)
    x.requires_grad_(True)
    
    with torch.enable_grad():
        u = model(x)
        ux = torch.autograd.grad(u, x, torch.ones_like(u), create_graph=True)[0]
        uxx = torch.autograd.grad(ux, x, torch.ones_like(ux), create_graph=True)[0]
        residual = uxx - pde.source
    
    residual_np = residual.detach().cpu().numpy().flatten()
    x_np = x.detach().cpu().numpy().flatten()
    
    bin_edges = bin_config.get_bin_edges()
    bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
    
    bin_residuals = {i: [] for i in range(bin_config.n_bins)}
    for xi, ri in zip(x_np, residual_np):
        bin_idx = np.digitize(xi, bin_edges) - 1
        if 0 <= bin_idx < bin_config.n_bins:
            bin_residuals[bin_idx].append(abs(ri))
    
    result = {}
    for i in range(bin_config.n_bins):
        if bin_residuals[i]:
            result[f'bin_{i}'] = {
                'center': float(bin_centers[i]),
                'mean_residual': float(np.mean(bin_residuals[i])),
                'max_residual': float(np.max(bin_residuals[i])),
                'count': len(bin_residuals[i]),
            }
        else:
            result[f'bin_{i}'] = {
                'center': float(bin_centers[i]),
                'mean_residual': 0.0,
                'max_residual': 0.0,
                'count': 0,
            }
    
    return result


def compute_spatial_error_bins(
    model: torch.nn.Module,
    pde: Poisson1D,
    device: torch.device,
    dtype: torch.dtype,
    bin_config: SpatialBinConfig
) -> Dict:
    n_test = 1000
    x = pde.validation_grid(n_test, device, dtype)
    with torch.no_grad():
        pred = model(x)
    exact = pde.exact(x)
    error = torch.abs(pred - exact)
    
    error_np = error.cpu().numpy().flatten()
    x_np = x.cpu().numpy().flatten()
    
    bin_edges = bin_config.get_bin_edges()
    bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
    
    bin_errors = {i: [] for i in range(bin_config.n_bins)}
    for xi, ei in zip(x_np, error_np):
        bin_idx = np.digitize(xi, bin_edges) - 1
        if 0 <= bin_idx < bin_config.n_bins:
            bin_errors[bin_idx].append(ei)
    
    result = {}
    for i in range(bin_config.n_bins):
        if bin_errors[i]:
            result[f'bin_{i}'] = {
                'center': float(bin_centers[i]),
                'mean_error': float(np.mean(bin_errors[i])),
                'max_error': float(np.max(bin_errors[i])),
                'count': len(bin_errors[i]),
            }
        else:
            result[f'bin_{i}'] = {
                'center': float(bin_centers[i]),
                'mean_error': 0.0,
                'max_error': 0.0,
                'count': 0,
            }
    
    return result


def compute_collocation_coverage(
    collocation_points: torch.Tensor,
    bin_config: SpatialBinConfig
) -> Dict:
    points_np = collocation_points.flatten().cpu().numpy()
    bin_edges = bin_config.get_bin_edges()
    bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
    
    counts, _ = np.histogram(points_np, bins=bin_edges)
    
    result = {}
    for i in range(bin_config.n_bins):
        result[f'bin_{i}'] = {
            'center': float(bin_centers[i]),
            'count': int(counts[i]),
            'density': float(counts[i] / len(points_np)) if len(points_np) > 0 else 0.0,
        }
    
    total_points = len(points_np)
    empty_bins = np.sum(counts == 0)
    min_count = np.min(counts) if total_points > 0 else 0
    
    result['summary'] = {
        'total_points': int(total_points),
        'empty_bins': int(empty_bins),
        'min_bin_count': int(min_count),
        'mean_bin_count': float(np.mean(counts)),
        'std_bin_count': float(np.std(counts)),
    }
    
    return result


def compute_boundary_distance_profile(
    coords: torch.Tensor,
    domain_left: float,
    domain_right: float
) -> Dict:
    coords_np = coords.flatten().cpu().numpy()
    dist_left = np.abs(coords_np - domain_left)
    dist_right = np.abs(coords_np - domain_right)
    dist_boundary = np.minimum(dist_left, dist_right)
    
    return {
        'mean_distance_to_boundary': float(np.mean(dist_boundary)),
        'min_distance_to_boundary': float(np.min(dist_boundary)),
        'max_distance_to_boundary': float(np.max(dist_boundary)),
        'std_distance_to_boundary': float(np.std(dist_boundary)),
    }


def compute_activation_fourier_spectrum(
    activations: torch.Tensor,
    probe_coords: torch.Tensor,
    domain_length: float
) -> Dict:
    if activations is None or activations.ndim != 2:
        return {}
    
    n_channels = min(activations.shape[1], 10)
    result = {}
    
    for c in range(n_channels):
        channel_signal = activations[:, c]
        freqs, mag = compute_fourier_spectrum(channel_signal, domain_length)
        
        total_power = np.sum(mag ** 2)
        low_freq_cutoff = len(freqs) // 4
        high_freq_power = np.sum(mag[low_freq_cutoff:] ** 2)
        low_freq_power = np.sum(mag[:low_freq_cutoff] ** 2)
        
        result[f'channel_{c}'] = {
            'frequencies': freqs.tolist(),
            'magnitude': mag.tolist(),
            'total_power': float(total_power),
            'high_freq_power': float(high_freq_power),
            'low_freq_power': float(low_freq_power),
            'high_to_low_ratio': float(high_freq_power / (low_freq_power + 1e-10)),
        }
    
    return result


class DiagnosticsLogger:
    def __init__(
        self,
        model: torch.nn.Module,
        pde: Poisson1D,
        bin_config: SpatialBinConfig,
        log_every: int = 100
    ):
        self.model = model
        self.pde = pde
        self.bin_config = bin_config
        self.log_every = log_every
    
    def maybe_log(
        self,
        step: int,
        collocation_points: Optional[torch.Tensor] = None,
        probe_activations: Optional[Dict[str, torch.Tensor]] = None
    ) -> Dict:
        if step % self.log_every != 0:
            return {}
        
        device = next(self.model.parameters()).device
        dtype = next(self.model.parameters()).dtype
        
        n_test = 501
        x = self.pde.validation_grid(n_test, device, dtype)
        
        with torch.no_grad():
            pred = self.model(x)
        exact = self.pde.exact(x)
        
        result = {
            'step': step,
            'error_spectrum': compute_error_spectrum(pred, exact, self.pde.right - self.pde.left),
            'spatial_residuals': compute_spatial_residual_bins(
                self.model, self.pde, device, dtype, self.bin_config
            ),
            'spatial_errors': compute_spatial_error_bins(
                self.model, self.pde, device, dtype, self.bin_config
            ),
        }
        
        if collocation_points is not None:
            result['collocation_coverage'] = compute_collocation_coverage(
                collocation_points, self.bin_config
            )
        
        if probe_activations is not None:
            for layer_name, acts in probe_activations.items():
                if acts is not None:
                    result[f'activation_spectrum_{layer_name}'] = compute_activation_fourier_spectrum(
                        acts, x, self.pde.right - self.pde.left
                    )
        
        return result