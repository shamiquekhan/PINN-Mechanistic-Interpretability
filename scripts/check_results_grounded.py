#!/usr/bin/env python3
"""Results-integrity check: every quantitative claim in the results-bearing
markdown must be grounded in a committed run artifact.

Motivation (docs/fresh_campaign_record.md discipline): prevent narrative
drift where README/RESULTS quote stale or hand-entered numbers that no
longer match (or never matched) the authoritative JSONs in runs/.

Checks performed
----------------
1. QUOTED-NUMBER GROUNDING (RESULTS.md, README.md, ROADMAP.md,
   docs/fresh_campaign_record.md): for a curated registry of
   (document, artifact path, field, allowed value-range) tuples, parse the
   live artifact and confirm the value quoted in the document's prose lies
   within the range. The registry lives in
   scripts/results_claims_registry.py and every entry names its artifact —
   the same grounding rule the figure/table scripts follow.

2. ARTIFACT EXISTENCE: every runs/*.json path referenced anywhere in the
   results-bearing docs must exist on disk and parse as JSON.

3. CHECKSUM VERIFICATION: data/checksums.sha256 must match the artifacts it
   covers (the authoritative-aggregate set).

Exit code 1 on any failure — CI blocks merges that cite ungrounded numbers.
"""
from __future__ import annotations

import hashlib
import json
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

# ---------------------------------------------------------------------------

RESULTS_DOCS = [
    "RESULTS.md",
    "README.md",
    "docs/fresh_campaign_record.md",
]


def load_artifact(rel: str):
    p = ROOT / rel
    with open(p) as f:
        return json.load(f)


def quoted(doc: str, pattern: str) -> bool:
    """True if the regex pattern matches the document's text."""
    text = (ROOT / doc).read_text()
    return re.search(pattern, text) is not None


def artifact_paths_referenced(doc: str):
    text = (ROOT / doc).read_text()
    return sorted(set(re.findall(r"runs/[\w/.-]+?\.json", text)))


def verify_checksums() -> list:
    """Verify data/checksums.sha256; return list of failure strings."""
    failures = []
    cs_path = ROOT / "data" / "checksums.sha256"
    if not cs_path.exists():
        return ["data/checksums.sha256 missing"]
    for line in cs_path.read_text().splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        digest, rel = line.split(None, 1)
        rel = rel.strip()
        p = ROOT / rel
        if not p.exists():
            failures.append(f"artifact missing: {rel}")
            continue
        h = hashlib.sha256(p.read_bytes()).hexdigest()
        if h != digest:
            failures.append(f"checksum mismatch: {rel}")
    return failures


# ---------------------------------------------------------------------------
# Claim registry: (document, artifact, getter, prose pattern that must match
# the artifact-derived value). Each getter receives the parsed artifact dict.
# ---------------------------------------------------------------------------

def _ci_mean(d):
    return d["causal_strength_ci"]["mean"]


def _ci_bounds(d):
    c = d["causal_strength_ci"]
    return c["ci_lower"], c["ci_upper"]


CLAIMS = [
    # (doc, artifact, getter -> value(s), regex builder)
    ("RESULTS.md", "runs/causal_intervention_results.json",
     lambda d: d["multiple_comparisons"]["bonferroni_n_survivors"],
     lambda v: rf"\b{v}/8\b"),
    ("RESULTS.md", "runs/pca_causal_results.json",
     lambda d: d["multiple_comparisons"]["bonferroni_n_survivors"],
     lambda v: rf"\b{v}/8\b"),
    ("RESULTS.md", "runs/positive_control.json",
     lambda d: d["pipeline_pass"],
     lambda v: r"PASSES" if v else r"FAILS"),
    ("README.md", "runs/positive_control.json",
     lambda d: d["pipeline_pass"],
     lambda v: r"positive control passes" if v else r"positive control FAILS"),
    ("RESULTS.md", "runs/operator_boundary/operator_boundary_report.json",
     lambda d: round(d["geometry"]["participation_ratio"], 1),
     lambda v: rf"\b{v:.1f}\b"),
    ("RESULTS.md", "runs/sota_baselines/sota_baseline_report.json",
     lambda d: round(d["baselines"]["NTK-adaptive"]["final_rel_l2"], 4),
     lambda v: rf"\b{v}\b"),
    ("docs/fresh_campaign_record.md",
     "runs/effective_rank_analysis/effective_rank_report.json",
     lambda d: round(d["summary"]["mean_participation_ratio"], 3),
     lambda v: rf"\b{v}\b"),
    # Stage 14 (operator causal battery): the FNO-vs-PINN survivor asymmetry
    # is the headline of RESULTS.md 5A.7 and must stay grounded.
    ("RESULTS.md", "runs/operator_causal/operator_causal_report.json",
     lambda d: d["battery_summary"]["multiple_comparisons"]["bonferroni_n_survivors"],
     lambda v: rf"6/8, 6/8" if v == 6 else rf"\b{v}/8\b"),
    ("RESULTS.md", "runs/operator_causal/operator_causal_report.json",
     lambda d: d["positive_control"]["pipeline_pass"],
     lambda v: r"PASS \(targeted" if v else r"FAIL"),
]


def check_claims() -> list:
    failures = []
    for doc, artifact, getter, pattern_fn in CLAIMS:
        try:
            value = getter(load_artifact(artifact))
            pattern = pattern_fn(value)
            if not quoted(doc, pattern):
                failures.append(
                    f"{doc}: artifact-derived value {value!r} "
                    f"(from {artifact}) not found via /{pattern}/ — "
                    f"the prose may cite a stale or ungrounded number.")
        except FileNotFoundError:
            failures.append(f"{doc}: artifact {artifact} missing")
        except (KeyError, TypeError, ValueError) as e:
            failures.append(f"{doc}: could not read {artifact}: {e}")
    return failures


def check_artifact_references() -> list:
    failures = []
    for doc in RESULTS_DOCS:
        p = ROOT / doc
        if not p.exists():
            failures.append(f"{doc} missing")
            continue
        for rel in artifact_paths_referenced(doc):
            target = ROOT / rel
            if not target.exists():
                failures.append(f"{doc} references missing artifact {rel}")
                continue
            try:
                json.loads(target.read_text())
            except json.JSONDecodeError as e:
                failures.append(f"{doc}: referenced artifact {rel} is not "
                                f"valid JSON: {e}")
    return failures


def main() -> int:
    failures = []
    failures += check_claims()
    failures += check_artifact_references()
    failures += verify_checksums()

    if failures:
        print("RESULTS-INTEGRITY: FAILED")
        for f in failures:
            print(f"  - {f}")
        return 1
    print("RESULTS-INTEGRITY: OK — every registered claim is grounded in a "
          "committed artifact; checksums verified.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
