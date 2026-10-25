#!/usr/bin/env python3
"""Build the registered pilot-to-confirmatory power-plan artifact.

The input contains pilot checkpoint-level rows only.  This script does not
train models, inspect confirmatory outcomes, or choose checkpoints.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from analysis.prospective_protocol import pilot_power_plan_report


DEFAULT_OUTPUT = ROOT / "runs" / "alignment_specificity_preregistered" / "power_plan.json"


def _load_pilot_rows(path: Path):
    payload = json.loads(path.read_text())
    if isinstance(payload, list):
        return payload
    if isinstance(payload, dict) and "pilot_rows" in payload:
        return payload["pilot_rows"]
    if isinstance(payload, dict) and "checkpoint_rows" in payload:
        return payload["checkpoint_rows"]
    raise ValueError(
        "pilot JSON must be a list or an object containing pilot_rows/checkpoint_rows"
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--pilot-json", type=Path, required=True)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--simulation-seed", type=int, default=0)
    parser.add_argument("--n-simulations", type=int, default=2_000)
    parser.add_argument("--n-resamples", type=int, default=2_000)
    args = parser.parse_args()

    raw_input = args.pilot_json.read_bytes()
    report = pilot_power_plan_report(
        _load_pilot_rows(args.pilot_json),
        simulation_seed=args.simulation_seed,
        n_simulations=args.n_simulations,
        n_resamples=args.n_resamples,
    )
    report["provenance"] = {
        "pilot_input_sha256": hashlib.sha256(raw_input).hexdigest(),
        "pilot_input_name": args.pilot_json.name,
        "simulation_seed": args.simulation_seed,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    selected = report["power_plan"]["selected_n_per_pde_family"]
    achieved = report["power_plan"]["power_by_n_per_pde_family"][str(selected)]
    print(f"Wrote {args.output}")
    print(f"Selected N per PDE family: {selected}; achieved power: {achieved:.3f}")


if __name__ == "__main__":
    main()
