"""Deterministic activation splits and frozen activation standardization.

The prospective alignment/specificity protocol keeps split assignment and
coordinate transforms explicit.  This module contains only deterministic,
serializable helpers; it does not train a model or score a scientific result.
"""
from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, Mapping, Optional, Tuple

import numpy as np
import torch


SPLIT_SCHEMA_VERSION = 2
STANDARDIZER_SCHEMA_VERSION = 1
PROTOCOL_MANIFEST_SCHEMA_VERSION = 1


def _digest_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def array_digest(values: np.ndarray | torch.Tensor) -> str:
    """Hash shape, dtype, and C-contiguous values for a stable manifest id."""
    if isinstance(values, torch.Tensor):
        array = values.detach().to(device="cpu").contiguous().numpy()
    else:
        array = np.ascontiguousarray(values)
    header = f"{array.dtype.str}:{array.shape}".encode("utf-8")
    return _digest_bytes(header + b"\0" + array.tobytes())


def _validate_fractions(fractions: Iterable[float]) -> Tuple[float, ...]:
    values = tuple(float(x) for x in fractions)
    if len(values) != 3:
        raise ValueError("fractions must contain train, selection, confirmatory")
    if any(x <= 0.0 for x in values):
        raise ValueError("all split fractions must be positive")
    if not np.isclose(sum(values), 1.0, atol=1e-12):
        raise ValueError("split fractions must sum to 1")
    return values


def _split_counts(n_rows: int, fractions: Tuple[float, ...]) -> Tuple[int, ...]:
    raw = np.asarray(fractions, dtype=float) * n_rows
    counts = np.floor(raw).astype(int)
    remainder = int(n_rows - counts.sum())
    # Largest-remainder allocation is deterministic and preserves the exact
    # requested total for small activation banks as well as large grids.
    order = sorted(range(len(raw)), key=lambda i: (-(raw[i] - counts[i]), i))
    for i in order[:remainder]:
        counts[i] += 1
    if any(count <= 0 for count in counts):
        raise ValueError(
            f"n_rows={n_rows} is too small for fractions {fractions}"
        )
    return tuple(int(x) for x in counts)


@dataclass(frozen=True)
class ActivationSplit:
    """Immutable row-index partition used by the prospective protocol."""

    split_key: str
    n_rows: int
    fractions: Tuple[float, float, float]
    train_idx: np.ndarray
    selection_idx: np.ndarray
    confirmatory_idx: np.ndarray

    def __post_init__(self) -> None:
        if self.n_rows < 3:
            raise ValueError("n_rows must be at least 3")
        arrays = (
            self.train_idx,
            self.selection_idx,
            self.confirmatory_idx,
        )
        for values in arrays:
            if values.ndim != 1:
                raise ValueError("split indices must be one-dimensional")
            if not np.issubdtype(values.dtype, np.integer):
                raise TypeError("split indices must be integer arrays")
            if np.any(values < 0) or np.any(values >= self.n_rows):
                raise ValueError("split index is outside the activation bank")
        all_idx = np.concatenate(arrays)
        if len(all_idx) != self.n_rows:
            raise ValueError("split indices do not cover every row")
        if len(np.unique(all_idx)) != self.n_rows:
            raise ValueError("split indices overlap")

    @property
    def assignment(self) -> np.ndarray:
        """Return an array with labels 0=train, 1=selection, 2=confirmatory."""
        labels = np.full(self.n_rows, -1, dtype=np.int8)
        labels[self.train_idx] = 0
        labels[self.selection_idx] = 1
        labels[self.confirmatory_idx] = 2
        return labels

    def manifest(self) -> Dict[str, Any]:
        assignment = self.assignment
        return {
            "schema_version": SPLIT_SCHEMA_VERSION,
            "split_key": self.split_key,
            "n_rows": self.n_rows,
            "fractions": list(self.fractions),
            "counts": {
                "train": int(len(self.train_idx)),
                "selection": int(len(self.selection_idx)),
                "confirmatory": int(len(self.confirmatory_idx)),
            },
            "assignment_sha256": array_digest(assignment),
            "train_sha256": array_digest(self.train_idx),
            "selection_sha256": array_digest(self.selection_idx),
            "confirmatory_sha256": array_digest(self.confirmatory_idx),
            # Persist the assignment itself so reruns use the registered rows
            # rather than regenerating a split from a key.
            "assignment": assignment.astype(int).tolist(),
        }

    @classmethod
    def from_manifest(cls, payload: Mapping[str, Any]) -> "ActivationSplit":
        """Reconstruct and validate a split from a persisted manifest."""
        if not isinstance(payload, Mapping):
            raise TypeError("split manifest must be a mapping")
        assignment_value = payload.get("assignment")
        if assignment_value is None:
            raise ValueError("split manifest must contain the row assignment")
        assignment = np.asarray(assignment_value, dtype=np.int8)
        if assignment.ndim != 1 or not np.isin(assignment, [0, 1, 2]).all():
            raise ValueError("assignment must contain one-dimensional labels 0, 1, 2")
        n_rows = int(payload.get("n_rows", len(assignment)))
        if len(assignment) != n_rows:
            raise ValueError("assignment length does not match n_rows")
        fractions = tuple(float(x) for x in payload["fractions"])
        split = cls(
            split_key=str(payload["split_key"]),
            n_rows=n_rows,
            fractions=_validate_fractions(fractions),  # type: ignore[arg-type]
            train_idx=np.flatnonzero(assignment == 0).astype(np.int64),
            selection_idx=np.flatnonzero(assignment == 1).astype(np.int64),
            confirmatory_idx=np.flatnonzero(assignment == 2).astype(np.int64),
        )
        actual = split.manifest()
        for key in (
            "assignment_sha256",
            "train_sha256",
            "selection_sha256",
            "confirmatory_sha256",
        ):
            if key in payload and payload[key] != actual[key]:
                raise ValueError(f"split manifest checksum mismatch for {key}")
        if "counts" in payload and payload["counts"] != actual["counts"]:
            raise ValueError("split manifest counts do not match assignment")
        return split


def make_activation_split(
    n_rows: int,
    *,
    split_key: str,
    fractions: Iterable[float] = (0.60, 0.20, 0.20),
) -> ActivationSplit:
    """Create a stable split without relying on Python's hash randomization.

    Each row is ordered by ``sha256(f"{split_key}:{row_index}")``.  The
    resulting indices are returned in ascending row order so indexing a tensor
    remains easy to inspect and serialize.
    """
    if not isinstance(split_key, str) or not split_key:
        raise ValueError("split_key must be a non-empty string")
    if not isinstance(n_rows, int) or n_rows < 3:
        raise ValueError("n_rows must be an integer >= 3")
    fractions_tuple = _validate_fractions(fractions)
    counts = _split_counts(n_rows, fractions_tuple)
    keyed = [
        (hashlib.sha256(f"{split_key}:{i}".encode("utf-8")).digest(), i)
        for i in range(n_rows)
    ]
    order = np.asarray([i for _, i in sorted(keyed)], dtype=np.int64)
    cut1 = counts[0]
    cut2 = cut1 + counts[1]
    train = np.sort(order[:cut1])
    selection = np.sort(order[cut1:cut2])
    confirmatory = np.sort(order[cut2:])
    return ActivationSplit(
        split_key=split_key,
        n_rows=n_rows,
        fractions=fractions_tuple,  # type: ignore[arg-type]
        train_idx=train,
        selection_idx=selection,
        confirmatory_idx=confirmatory,
    )


@dataclass(frozen=True)
class ActivationStandardizer:
    """Train-split-only affine transform for hidden activations."""

    mean: torch.Tensor
    std: torch.Tensor
    eps: float = 1e-6

    def __post_init__(self) -> None:
        if self.mean.ndim != 1 or self.std.ndim != 1:
            raise ValueError("mean and std must be one-dimensional")
        if self.mean.shape != self.std.shape:
            raise ValueError("mean and std must have the same shape")
        if not torch.isfinite(self.mean).all() or not torch.isfinite(self.std).all():
            raise ValueError("mean and std must be finite")
        if self.eps <= 0:
            raise ValueError("eps must be positive")
        if torch.any(self.std <= 0):
            raise ValueError("std must be strictly positive")

    @classmethod
    def fit(cls, activations: torch.Tensor, eps: float = 1e-6) -> "ActivationStandardizer":
        if activations.ndim != 2 or activations.shape[0] < 2:
            raise ValueError("activations must have shape (samples, width) with >=2 rows")
        values = activations.detach().to(dtype=torch.float32)
        mean = values.mean(dim=0)
        std = values.std(dim=0, unbiased=False).clamp_min(eps)
        return cls(mean=mean, std=std, eps=eps)

    @property
    def width(self) -> int:
        return int(self.mean.numel())

    def _parameters_for(self, values: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        if values.shape[-1] != self.width:
            raise ValueError(
                f"activation width {values.shape[-1]} does not match {self.width}"
            )
        mean = self.mean.to(device=values.device, dtype=values.dtype)
        std = self.std.to(device=values.device, dtype=values.dtype)
        return mean, std

    def transform(self, activations: torch.Tensor) -> torch.Tensor:
        mean, std = self._parameters_for(activations)
        return (activations - mean) / std

    def inverse(self, standardized: torch.Tensor) -> torch.Tensor:
        mean, std = self._parameters_for(standardized)
        return standardized * std + mean

    def direction_to_raw(self, direction: torch.Tensor) -> torch.Tensor:
        """Map a direction in standardized coordinates into raw coordinates."""
        if direction.ndim != 1 or direction.numel() != self.width:
            raise ValueError("direction must have shape (activation_width,)")
        std = self.std.to(device=direction.device, dtype=direction.dtype)
        return direction * std

    def direction_to_standardized(self, direction: torch.Tensor) -> torch.Tensor:
        """Map a raw hidden-state direction into standardized coordinates."""
        if direction.ndim != 1 or direction.numel() != self.width:
            raise ValueError("direction must have shape (activation_width,)")
        std = self.std.to(device=direction.device, dtype=direction.dtype)
        return direction / std

    def manifest(self) -> Dict[str, Any]:
        return {
            "schema_version": STANDARDIZER_SCHEMA_VERSION,
            "width": self.width,
            "eps": self.eps,
            "mean": self.mean.detach().cpu().tolist(),
            "std": self.std.detach().cpu().tolist(),
            "mean_sha256": array_digest(self.mean.to(dtype=torch.float64)),
            "std_sha256": array_digest(self.std.to(dtype=torch.float64)),
        }

    @classmethod
    def from_manifest(cls, payload: Mapping[str, Any]) -> "ActivationStandardizer":
        """Reconstruct and validate train-split scale parameters."""
        if not isinstance(payload, Mapping):
            raise TypeError("standardizer manifest must be a mapping")
        mean = torch.as_tensor(payload["mean"], dtype=torch.float32)
        std = torch.as_tensor(payload["std"], dtype=torch.float32)
        standardizer = cls(
            mean=mean,
            std=std,
            eps=float(payload.get("eps", 1e-6)),
        )
        actual = standardizer.manifest()
        for key in ("mean_sha256", "std_sha256"):
            if key in payload and payload[key] != actual[key]:
                raise ValueError(f"standardizer manifest checksum mismatch for {key}")
        if "width" in payload and int(payload["width"]) != standardizer.width:
            raise ValueError("standardizer manifest width does not match parameters")
        return standardizer


def write_activation_protocol_manifest(
    path: str | Path,
    split: ActivationSplit,
    *,
    standardizer: Optional[ActivationStandardizer] = None,
    activation_bank: Optional[np.ndarray | torch.Tensor] = None,
    metadata: Optional[Mapping[str, Any]] = None,
) -> Dict[str, Any]:
    """Write a replayable split/standardization manifest as JSON.

    The activation values themselves are not copied into the manifest.  When
    supplied, only their shape, dtype, and content digest are recorded.
    """
    payload: Dict[str, Any] = {
        "schema_version": PROTOCOL_MANIFEST_SCHEMA_VERSION,
        "split": split.manifest(),
    }
    if standardizer is not None:
        payload["standardizer"] = standardizer.manifest()
    if activation_bank is not None:
        if isinstance(activation_bank, torch.Tensor):
            shape = list(activation_bank.shape)
            dtype = str(activation_bank.detach().cpu().numpy().dtype)
        else:
            array = np.asarray(activation_bank)
            shape = list(array.shape)
            dtype = str(array.dtype)
        payload["activation_bank"] = {
            "shape": shape,
            "dtype": dtype,
            "sha256": array_digest(activation_bank),
        }
    if metadata is not None:
        payload["metadata"] = dict(metadata)
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    return payload


def load_activation_protocol_manifest(
    path: str | Path,
) -> Tuple[Dict[str, Any], ActivationSplit, Optional[ActivationStandardizer]]:
    """Load and validate a replayable activation protocol manifest."""
    payload = json.loads(Path(path).read_text())
    if int(payload.get("schema_version", -1)) != PROTOCOL_MANIFEST_SCHEMA_VERSION:
        raise ValueError("unsupported activation protocol manifest schema")
    split = ActivationSplit.from_manifest(payload["split"])
    standardizer_payload = payload.get("standardizer")
    standardizer = (
        ActivationStandardizer.from_manifest(standardizer_payload)
        if standardizer_payload is not None
        else None
    )
    return payload, split, standardizer
