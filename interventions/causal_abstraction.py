"""Small interchange-intervention primitives for causal-abstraction studies.

The API is deliberately measurement-oriented: it records low-level donor/source
states and outputs, but does not assume that an alignment is valid. Alignment
quality must be evaluated against a preregistered high-level causal model.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable, Dict, Iterable, Mapping, Optional

import torch


@dataclass
class CausalModel:
    """Container for a high-level variable set and structural dependencies."""

    variables: tuple[str, ...]
    dependencies: Mapping[str, tuple[str, ...]] = field(default_factory=dict)
    state: Dict[str, Any] = field(default_factory=dict)

    def __init__(self, variables, dependencies=None, state=None):
        self.variables = tuple(variables)
        self.dependencies = dict(dependencies or {})
        self.state = dict(state or {})
        unknown = set(self.dependencies) - set(self.variables)
        if unknown:
            raise ValueError(f"dependencies contain unknown variables: {sorted(unknown)}")

    def intervene(self, variable: str, value: Any) -> Dict[str, Any]:
        """Return a copied high-level state under ``do(variable=value)``."""
        if variable not in self.variables:
            raise KeyError(variable)
        intervened = dict(self.state)
        intervened[variable] = value
        return intervened


def interchange_intervention(
    model,
    input_source: torch.Tensor,
    input_donor: torch.Tensor,
    layer_index: int,
    alignment_map: Callable[[torch.Tensor], Any],
    causal_model: Optional[CausalModel] = None,
) -> Dict[str, Any]:
    """Replace a hidden state from ``input_donor`` while running source input."""
    model.eval()
    captured: Dict[str, torch.Tensor] = {}

    def capture(_module, _inputs, output):
        captured["state"] = output.detach()

    recorder = model.acts[layer_index].register_forward_hook(capture)
    try:
        with torch.no_grad():
            source_output = model(input_source)
            source_state = captured["state"]
            donor_output = model(input_donor)
            donor_state = captured["state"]
    finally:
        recorder.remove()

    def replace(_module, _inputs, _output):
        return donor_state

    swap_hook = model.acts[layer_index].register_forward_hook(replace)
    try:
        with torch.no_grad():
            swapped_output = model(input_source)
    finally:
        swap_hook.remove()

    result = {
        "layer_index": layer_index,
        "source_output": source_output,
        "donor_output": donor_output,
        "swapped_output": swapped_output,
        "source_state": source_state,
        "donor_state": donor_state,
        "source_alignment": alignment_map(source_state),
        "donor_alignment": alignment_map(donor_state),
    }
    if causal_model is not None:
        result["causal_variables"] = causal_model.variables
        result["causal_dependencies"] = dict(causal_model.dependencies)
    return result


def alignment_agreement(
    records: Iterable[Dict[str, Any]],
    high_level_predictor: Callable[[Any, Any], Any],
) -> float:
    """Compute exact agreement between observed and predicted high-level swaps."""
    records = list(records)
    if not records:
        return float("nan")
    matches = 0
    for record in records:
        observed = record["swapped_output"]
        predicted = high_level_predictor(
            record["source_alignment"], record["donor_alignment"]
        )
        if torch.is_tensor(observed) and torch.is_tensor(predicted):
            matches += int(torch.allclose(observed, predicted))
        else:
            matches += int(observed == predicted)
    return matches / len(records)
