import torch
from typing import Dict, List, Tuple
import numpy as np


def compute_per_loss_gradients(model, loss_dict: Dict[str, torch.Tensor], retain_graph: bool = True) -> Dict[str, Dict[str, torch.Tensor]]:
    grads = {}
    for name, loss in loss_dict.items():
        model.zero_grad()
        loss.backward(retain_graph=retain_graph)
        grads[name] = {n: p.grad.clone() if p.grad is not None else torch.zeros_like(p) 
                       for n, p in model.named_parameters()}
    return grads


def compute_gradient_norms(grads: Dict[str, Dict[str, torch.Tensor]]) -> Dict[str, float]:
    norms = {}
    for loss_name, param_grads in grads.items():
        total_norm = 0.0
        for g in param_grads.values():
            total_norm += g.norm(2).item() ** 2
        norms[loss_name] = total_norm ** 0.5
    return norms


def compute_gradient_cosine_similarities(grads: Dict[str, Dict[str, torch.Tensor]]) -> Dict[str, float]:
    loss_names = list(grads.keys())
    cosines = {}
    for i, name_i in enumerate(loss_names):
        for j, name_j in enumerate(loss_names):
            if i >= j:
                continue
            flat_i = torch.cat([g.flatten() for g in grads[name_i].values()])
            flat_j = torch.cat([g.flatten() for g in grads[name_j].values()])
            cos = torch.nn.functional.cosine_similarity(flat_i.unsqueeze(0), flat_j.unsqueeze(0)).item()
            cosines[f"{name_i}_vs_{name_j}"] = cos
    return cosines


def compute_gradient_conflict_score(cosines: Dict[str, float]) -> float:
    if not cosines:
        return 0.0
    negative_cosines = [c for c in cosines.values() if c < 0]
    if not negative_cosines:
        return 0.0
    return abs(min(negative_cosines))


def log_gradient_stats(grads: Dict[str, Dict[str, torch.Tensor]]) -> Dict:
    norms = compute_gradient_norms(grads)
    cosines = compute_gradient_cosine_similarities(grads)
    conflict_score = compute_gradient_conflict_score(cosines)
    
    stats = {
        'gradient_norms': norms,
        'gradient_cosines': cosines,
        'gradient_conflict_score': conflict_score,
    }
    return stats


class GradientLogger:
    def __init__(self, model, log_every: int = 100, reduced: bool = False):
        self.model = model
        self.log_every = log_every
        self.reduced = reduced
        self.step = 0
        
    def maybe_log(self, loss_dict: Dict[str, torch.Tensor], step: int) -> Dict:
        self.step = step
        if step % self.log_every != 0:
            return {}
        
        grads = compute_per_loss_gradients(self.model, loss_dict)
        stats = log_gradient_stats(grads)
        
        if self.reduced:
            for name in grads:
                for param_name in grads[name]:
                    grads[name][param_name] = grads[name][param_name].cpu().half()
        
        return {
            'step': step,
            'gradient_stats': stats,
            'raw_gradients': grads if not self.reduced else None
        }


def get_gradient_vector(grads: Dict[str, torch.Tensor]) -> torch.Tensor:
    return torch.cat([g.flatten() for g in grads.values()])


def compute_per_parameter_gradient_stats(grads: Dict[str, torch.Tensor]) -> Dict:
    stats = {}
    for name, g in grads.items():
        g_flat = g.flatten()
        stats[name] = {
            'mean': g_flat.mean().item(),
            'std': g_flat.std().item(),
            'max': g_flat.max().item(),
            'min': g_flat.min().item(),
            'norm': g_flat.norm(2).item(),
            'num_params': g.numel(),
        }
    return stats