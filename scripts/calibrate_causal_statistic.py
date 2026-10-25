#!/usr/bin/env python3
"""Run the separately preregistered synthetic causal-statistic audit."""
from __future__ import annotations

import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from analysis.causal_calibration import run_calibration_suite


OUT = ROOT / "runs" / "statistical_calibration" / "causal_calibration_report.json"


def main() -> None:
    report = run_calibration_suite()
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(report, indent=2) + "\n")
    print(f"Wrote {OUT}")
    for case in report["cases"]:
        print(
            f"{case['case']}: sign rejection="
            f"{case['sign_test_rejection_rate']:.3f}; "
            f"randomization applicable={case['randomization_test_applicable']}"
        )


if __name__ == "__main__":
    main()
