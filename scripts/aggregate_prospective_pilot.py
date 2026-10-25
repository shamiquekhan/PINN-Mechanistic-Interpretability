#!/usr/bin/env python3
"""Aggregate raw direct-arm pilot rows into checkpoint-level rho rows."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from analysis.prospective_pilot import aggregate_pilot_rows


DEFAULT_OUTPUT = ROOT / "runs" / "alignment_specificity_preregistered" / "pilot_rows.json"


def _load_rows(path: Path):
    payload = json.loads(path.read_text())
    if isinstance(payload, list):
        return payload
    if isinstance(payload, dict) and "checkpoint_rows" in payload:
        return payload["checkpoint_rows"]
    raise ValueError(
        "raw pilot JSON must be a list or an object containing checkpoint_rows"
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--raw-json", type=Path, required=True)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--expected-candidates", type=int, default=16)
    parser.add_argument("--expected-batches", type=int, default=None)
    args = parser.parse_args()

    raw_bytes = args.raw_json.read_bytes()
    rows = aggregate_pilot_rows(
        _load_rows(args.raw_json),
        expected_candidates=args.expected_candidates,
        expected_batch_count=args.expected_batches,
    )
    report = {
        "schema_version": 1,
        "artifact_kind": "prospective_pilot_checkpoint_rows",
        "intervention_arm": "direct_hidden_state",
        "confirmatory_outcomes_used": False,
        "checkpoint_rows": rows,
        "provenance": {
            "raw_input_sha256": hashlib.sha256(raw_bytes).hexdigest(),
            "raw_input_name": args.raw_json.name,
            "expected_candidates": args.expected_candidates,
            "expected_batches": args.expected_batches,
        },
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(f"Wrote {args.output}")
    print(f"Aggregated checkpoints: {len(rows)}")


if __name__ == "__main__":
    main()
