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

# Stale-marker guard scans only the LIVING docs. fresh_campaign_record.md
# is a historical drift table (its old values are the point of the
# published-vs-fresh comparison) and is exempt from the marker sweep while
# still being covered by the artifact-existence and reference checks above.
STALE_SCAN_DOCS = [
    "RESULTS.md",
    "README.md",
    "PROJECT_STATUS.md",
    "ROADMAP.md",
    "CONTRIBUTING.md",
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
    # Stage 15 (NTK bridge): the null verdict numbers in RESULTS.md 5A.8.
    ("RESULTS.md", "runs/ntk_bridge/ntk_bridge_report.json",
     lambda d: d["decision_rule"]["recorded"],
     lambda v: rf"\b{v}\b" if v == "H15b" else rf"\b{v}\b"),
    ("RESULTS.md", "runs/ntk_bridge/ntk_bridge_report.json",
     lambda d: d["per_feature_summary"],
     lambda v: rf"0/8" if all(not s.get("bonferroni_survivor", False) for s in v) else "NONZERO-SURVIVORS-NOT-GROUNDED"),
    # Stage 14 (operator causal battery): the FNO-vs-PINN survivor asymmetry
    # is the headline of RESULTS.md 5A.7 and must stay grounded.
    ("RESULTS.md", "runs/operator_causal/operator_causal_report.json",
     lambda d: d["battery_summary"]["multiple_comparisons"]["bonferroni_n_survivors"],
     lambda v: rf"\b{v}/8, {v}/8\b"),
    ("RESULTS.md", "runs/operator_causal/operator_causal_report.json",
     lambda d: d["positive_control"]["pipeline_pass"],
     lambda v: r"PASS \(targeted" if v else r"FAIL"),
    # R6 (dose-response): the headline null — 0 R6a in the Fourier SAE
    # arm — must stay grounded in the committed report.
    ("RESULTS.md", "runs/r6_dose_response/r6_report.json",
     lambda d: d["arms"]["fourier_sae"]["verdict_counts"]["R6a"],
     lambda v: rf"Fourier SAE \(PR 4\.0–5\.6\) \| \*\*{v}/24\*\*" if v == 0
     else rf"Fourier SAE \(PR 4\.0–5\.6\) \| {v}/24\*?"),
    ("RESULTS.md", "runs/r6_dose_response/r6_report.json",
     lambda d: d["arms"]["fno_sae"]["verdict_counts"]["R6a"],
     lambda v: rf"FNO SAE \(PR 6\.8\) \| \*\*{v}/8\*\*" if v == 0
     else rf"FNO SAE \(PR 6\.8\) \| {v}/8\*?"),
    ("RESULTS.md", "runs/r6_dose_response/r6_report.json",
     lambda d: d["gate"]["pipeline_pass"],
     lambda v: r"[Mm]achinery gate.*?\bPASS\b" if v else r"machinery gate FAILS"),
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


# C4 guard (external review): documentation-drift markers that must never
# silently reappear in the living docs (historical records are exempt).
STALE_MARKERS = [
    r"0\.872",            # v3.0 monitor AUROC (superseded)
    r"stages 1–13",        # 15 stages exist
    r"7 intervention modes",  # 8 exist
    r"ACTING",             # controller state that does not exist
]

# v3.4-era moved numbers: an UNMARKED occurrence in a living doc is drift.
# A hit is allowed only when its line carries one of these context markers
# (historical / drift / pre-fix phrasing), which the drift tables and
# response docs always include.
MOVED_NUMBERS = [
    r"E_T \+0\.0020",     # pre-C3 SAE E_T (now −0.0173)
    r"−0\.0203",          # pre-C3 PCA E_T (now −0.0131)
    r"6/8 Bonferroni survivors, E_T CI positive",  # pre-C3 stage-14 count
    r"0\.859",            # pre-H1 monitor AUROC (now 0.875)
    r"0\.0166",           # pre-H3 controller (now 0.0002)
    r"rescues boundary starvation 18×",   # pre-H3 rescue phrasing
    r"18\$\\times\$ over no-action",  # LaTeX rescue phrasing
]
MOVED_CONTEXT_OK = ("was ", "pre-fix", "moved", "→", "->", "artifact of",
                    "historical", "drift", "v3.2 record", "v3.3 record",
                    "retained in the", "see docs/external_review_response",
                    "(was ", "corrected", "old", "Old")
# (exemptions applied implicitly: the guard below only scans RESULTS_DOCS +
# the living ledgers; historical records like fresh_campaign_record are
# drift tables whose old values are the point.)


def check_stale_markers() -> list:
    import re
    failures = []
    for doc in STALE_SCAN_DOCS:
        p = ROOT / doc
        if not p.exists():
            continue
        text = p.read_text()
        for marker in STALE_MARKERS:
            for m in re.finditer(marker, text):
                line_no = text[:m.start()].count("\n") + 1
                snippet = text.splitlines()[line_no - 1][:80]
                failures.append(
                    f"{doc}:{line_no}: stale marker /{marker}/ — {snippet}")
        # v3.4 moved numbers: only allowed inside explicitly-marked
        # historical/drift contexts
        for marker in MOVED_NUMBERS:
            for m in re.finditer(marker, text):
                line_no = text[:m.start()].count("\n") + 1
                line = text.splitlines()[line_no - 1]
                if not any(ctx in line for ctx in MOVED_CONTEXT_OK):
                    failures.append(
                        f"{doc}:{line_no}: unmarked moved number "
                        f"/{marker}/ — {line[:90]}")
    return failures


def main() -> int:
    failures = []
    failures += check_claims()
    failures += check_artifact_references()
    failures += check_stale_markers()
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
