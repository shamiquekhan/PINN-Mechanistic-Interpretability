"""
Activation dataset builder for SAE training.
Builds run-level activation examples from logged JSONL files,
enforcing run-level splits to prevent temporal leakage.
"""
from __future__ import annotations
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader

from pinn_logging.io import load_jsonl


# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------

class ActivationDataset(Dataset):
    """Flat dataset of (activation_vector,) tensors built from run logs.

    Parameters
    ----------
    run_dirs:   List of run directories containing activations.jsonl.
    layer_name: Which layer key to extract from each activation record.
    normalise:  If True, standardise with mean/std computed on this dataset.
    fit_stats:  Pre-computed (mean, std) to apply (overrides computing from data).
    """

    def __init__(
        self,
        run_dirs: List[Path],
        layer_name: str,
        normalise: bool = True,
        fit_stats: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
    ):
        self.layer_name = layer_name
        vectors: List[np.ndarray] = []

        for run_dir in run_dirs:
            log_path = Path(run_dir) / "activations.jsonl"
            if not log_path.exists():
                continue
            records = load_jsonl(log_path)
            for rec in records:
                acts = rec.get("activations", {})
                # Try exact layer_name match, or fuzzy match if layer_name not found
                target_key = layer_name
                if target_key not in acts:
                    # try matching layer index (e.g. '0' -> 'layers.0' or 'net.0')
                    for k in acts:
                        if layer_name in k or k.endswith(f".{layer_name}"):
                            target_key = k
                            break

                layer_data = acts.get(target_key, {})
                raw = layer_data.get("raw")
                if raw is not None:
                    arr = np.array(raw, dtype=np.float32)
                    for row in arr:
                        vectors.append(row)

        if not vectors:
            raise ValueError(
                f"No activation vectors found for layer '{layer_name}' in {[str(d) for d in run_dirs]}. "
                "Make sure you trained with --log-activations --act-save-raw."
            )

        data = np.stack(vectors, axis=0)   # (N, D)
        self.data = torch.tensor(data, dtype=torch.float32)

        if normalise:
            if fit_stats is not None:
                self.mean, self.std = fit_stats
            else:
                self.mean = self.data.mean(dim=0)
                self.std  = self.data.std(dim=0).clamp(min=1e-8)
            self.data = (self.data - self.mean) / self.std
        else:
            self.mean = torch.zeros(self.data.shape[1])
            self.std  = torch.ones(self.data.shape[1])

    def __len__(self) -> int:
        return len(self.data)

    def __getitem__(self, idx: int) -> torch.Tensor:
        return self.data[idx]

    def get_fit_stats(self) -> Tuple[torch.Tensor, torch.Tensor]:
        return self.mean, self.std


# ---------------------------------------------------------------------------
# Split helpers
# ---------------------------------------------------------------------------

def make_run_splits(
    all_run_dirs: List[Path],
    train_frac: float = 0.7,
    val_frac: float = 0.15,
    seed: int = 0,
) -> Tuple[List[Path], List[Path], List[Path]]:
    """Split run directories at the run level (not per activation vector).

    Returns
    -------
    train_dirs, val_dirs, test_dirs
    """
    rng = np.random.default_rng(seed)
    dirs = list(all_run_dirs)
    rng.shuffle(dirs)
    n = len(dirs)
    n_train = max(1, int(n * train_frac))
    n_val   = max(1, int(n * val_frac))
    train_dirs = dirs[:n_train]
    val_dirs   = dirs[n_train:n_train + n_val]
    test_dirs  = dirs[n_train + n_val:]
    return train_dirs, val_dirs, test_dirs


def build_dataloaders(
    train_dirs: List[Path],
    val_dirs: List[Path],
    test_dirs: List[Path],
    layer_name: str,
    batch_size: int = 256,
    num_workers: int = 0,
    device: Optional[torch.device] = None,
) -> Tuple[DataLoader, DataLoader, DataLoader, Tuple]:
    """Build train / val / test DataLoaders with proper normalisation.

    Normalisation stats are fit only on the training set, then applied to
    val/test sets.

    Returns
    -------
    train_loader, val_loader, test_loader, (mean, std)
    """
    train_ds = ActivationDataset(train_dirs, layer_name, normalise=True)
    fit_stats = train_ds.get_fit_stats()

    val_ds  = ActivationDataset(val_dirs,  layer_name, normalise=True, fit_stats=fit_stats) if val_dirs else train_ds
    test_ds = ActivationDataset(test_dirs, layer_name, normalise=True, fit_stats=fit_stats) if test_dirs else train_ds

    kw = dict(batch_size=min(batch_size, len(train_ds)), num_workers=num_workers, pin_memory=(device is not None and device.type == "cuda"))
    return (
        DataLoader(train_ds, shuffle=True,  **kw),
        DataLoader(val_ds,   shuffle=False, **kw),
        DataLoader(test_ds,  shuffle=False, **kw),
        fit_stats,
    )
