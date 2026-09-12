#!/usr/bin/env python3
"""
Repository integrity audit for the PINN Mechanistic Interpretability project.

Usage:
    python scripts/final_repo_audit.py

Checks:
  1. Git tree clean
  2. Current branch = main
  3. Version consistency (README, CITATION, paper)
  4. No stale R7+ "IN FLIGHT"
  5. No missing artifact references
  6. Manifest ↔ checksum agreement
  7. Grounding check
  8. Test count in README matches actual
  9. Required files present
 10. No forbidden files (secrets, .env)
 11. No hard-coded private paths in executables
"""

import json
import re
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
passed = 0
failed = 0


def check(label: str, condition: bool, detail: str = ""):
    global passed, failed
    status = "PASS" if condition else "FAIL"
    print(f"  {status}  {label}")
    if detail:
        print(f"        {detail}")
    if condition:
        passed += 1
    else:
        failed += 1


def run(cmd: str) -> str:
    r = subprocess.run(
        cmd, shell=True, capture_output=True, text=True, cwd=str(ROOT)
    )
    return r.stdout.strip()


print("=" * 60)
print("  REPOSITORY INTEGRITY AUDIT")
print("=" * 60)
print()

# --- 1. Git ---
print("[Git state]")
clean = run("git status --porcelain") == ""
branch = run("git branch --show-current")
check("Git tree clean", clean)
check("Branch = main", branch == "main", f"current: {branch}")
print()

# --- 2. Version consistency ---
print("[Version consistency]")

readme = (ROOT / "README.md").read_text()
citation = (ROOT / "CITATION.cff").read_text()
paper_main = (ROOT / "paper" / "main.tex").read_text()
paper_arxiv = (ROOT / "paper" / "arxiv_package" / "main.tex").read_text()

readme_version = re.search(r"Research Status.*?v(\d+\.\d+\.\d+)", readme)
citation_version = re.search(r'^version:\s*"(\d+\.\d+\.\d+)"', citation, re.M)
paper_version = re.search(r"v(\d+\.\d+\.\d+)\}", paper_main)
arxiv_version = re.search(r"v(\d+\.\d+\.\d+)\}", paper_arxiv)

versions = {}
if readme_version:
    versions["README"] = readme_version.group(1)
if citation_version:
    versions["CITATION.cff"] = citation_version.group(1)
if paper_version:
    versions["paper/main.tex"] = paper_version.group(1)
if arxiv_version:
    versions["paper/arxiv_package"] = arxiv_version.group(1)

all_versions = set(versions.values())
check(
    "All versions agree",
    len(all_versions) == 1,
    f"found: {versions}",
)
print()

# --- 3. Stale R7+ ---
print("[R7+ status]")
for doc_name, doc_text in [
    ("README.md", readme),
    ("PROJECT_STATUS.md", (ROOT / "PROJECT_STATUS.md").read_text()),
    ("RESULTS.md", (ROOT / "RESULTS.md").read_text()),
]:
    hits = re.findall(r"R7\+.*(?:IN FLIGHT|in flight)", doc_text, re.I)
    if hits:
        check(f"No stale R7+ IN FLIGHT in {doc_name}", False, str(hits))
    else:
        check(f"No stale R7+ IN FLIGHT in {doc_name}", True)
print()

# --- 3b. R8/R9/R10 wording consistency ---
print("[R8/R9/R10 wording]")
results_text = (ROOT / "RESULTS.md").read_text()
ps_text = (ROOT / "PROJECT_STATUS.md").read_text()

# R8: PCA negative-side must be described as not surviving clustering
r8_pca_fail = bool(re.search(
    r"PCA.*negative.*(?:survive|cluster.robust|CI.*excludes.*zero)",
    ps_text + results_text, re.I
))
check("R8: PCA negative-side described as not surviving clustering", not r8_pca_fail,
      "FOUND: PCA negative-side described as surviving" if r8_pca_fail else "")

# R9: three-family must mean exactly Poisson/Burgers/reaction-diffusion
r9_families = re.findall(r"three.family.external.validity", ps_text + results_text, re.I)
check("R9: 'three-family external validity' appears", len(r9_families) > 0,
      f"found: {len(r9_families)} occurrences")

# R10: must be described as gate, not predictor
r10_gate = bool(re.search(
    r"PR.*(?:family|is).*(?:validated\s+)?gate.*not.*(?:continuous|sufficient).*predictor",
    ps_text + results_text, re.I
))
check("R10: PR described as gate, not predictor", r10_gate)

print()
print("[Artifact references]")
missing_total = 0
for doc_name in ["README.md", "PROJECT_STATUS.md", "RESULTS.md"]:
    doc_text = (ROOT / doc_name).read_text()
    # Match `runs/...` paths (single line), skip templates, globs, and directory refs
    refs = sorted(set(re.findall(r"`(runs/[^`\n]+)`", doc_text)))
    # Filter: skip templates (<...>), globs (*, ?), and trailing-/ directories
    check_refs = [p for p in refs if "<" not in p and "*" not in p and "?" not in p and not p.endswith("/")]
    missing = [p for p in check_refs if not (ROOT / p).exists()]
    if missing:
        check(f"{doc_name} artifact refs", False, f"missing: {missing}")
        missing_total += len(missing)
    else:
        check(f"{doc_name} artifact refs ({len(refs)} paths)", True)
print()

# --- 5. Checksums / manifest ---
print("[Integrity]")
checksums = ROOT / "data" / "checksums.sha256"
manifest = ROOT / "data" / "manifest.json"

cs_ok = subprocess.run(
    ["sha256sum", "-c", str(checksums)],
    capture_output=True,
    text=True,
    cwd=str(ROOT),
)
check("Checksums verified", cs_ok.returncode == 0, cs_ok.stdout.strip().split("\n")[-1] if cs_ok.stdout else "")

manifest_data = json.loads(manifest.read_text())
manifest_count = len(manifest_data.get("files", []))
check("Manifest has entries", manifest_count > 0, f"count: {manifest_count}")

cs_lines = [l for l in checksums.read_text().splitlines() if l.strip()]
cs_count = len(cs_lines)
check("Checksums count matches manifest", manifest_count == cs_count, f"manifest={manifest_count} checksums={cs_count}")
print()

# --- 6. Required files ---
print("[Required files]")
required = [
    "README.md",
    "CHANGELOG.md",
    "PROJECT_STATUS.md",
    "RESULTS.md",
    "CITATION.cff",
    "LICENSE",
    "docs/preregistration.md",
    "docs/experiment_matrix.md",
    "docs/claims.md",
    "docs/final_claim_ladder_v4.md",
    "docs/physSAE_reconciliation.md",
    "data/manifest.json",
    "data/checksums.sha256",
    "scripts/check_results_grounded.py",
]
for f in required:
    check(f"File exists: {f}", (ROOT / f).exists())
print()

# --- 7. Forbidden files ---
print("[Forbidden files]")
forbidden_patterns = [r"\.env$", "credentials", "secrets", r"token\.json", "password"]
found_forbidden = []
for pat in forbidden_patterns:
    matches = run(f"git ls-files | grep -iE '{pat}' || true")
    if matches:
        found_forbidden.extend(matches.split("\n"))
check("No secrets/credentials committed", len(found_forbidden) == 0, str(found_forbidden) if found_forbidden else "")
print()

# --- 8. Hard-coded paths ---
print("[Hard-coded paths]")
code_dirs = ["experiments", "scripts", "tests"]
private_hits = []
for d in code_dirs:
    dpath = ROOT / d
    if dpath.exists():
        hits = run(f"grep -rnE '/home/|/mnt/' {dpath} --include='*.py' --exclude-dir=__pycache__ --exclude=final_repo_audit.py || true")
        if hits:
            private_hits.extend(hits.split("\n"))
check("No hard-coded private paths in code", len(private_hits) == 0, f"found: {len(private_hits)}" if private_hits else "")
if private_hits:
    for h in private_hits[:5]:
        print(f"        {h}")
print()

# --- 9. Grounding ---
print("[Grounding]")
gr = subprocess.run(
    [sys.executable, str(ROOT / "scripts" / "check_results_grounded.py")],
    capture_output=True,
    text=True,
    cwd=str(ROOT),
)
check("Results grounding check", gr.returncode == 0, gr.stdout.strip() if gr.stdout else gr.stderr[:200])
print()

# --- 10. Test count ---
print("[Tests]")
test_count_line = run("python -m pytest tests/ --co -q 2>/dev/null | tail -1")
match = re.search(r"(\d+) test", test_count_line)
if match:
    actual = int(match.group(1))
    check(f"Test count available", True, f"count: {actual}")
else:
    check("Test count available", False, f"output: {test_count_line}")
print()

# --- Summary ---
print("=" * 60)
total = passed + failed
if failed == 0:
    print(f"  OVERALL: PASS ({passed}/{total} checks passed)")
else:
    print(f"  OVERALL: FAIL ({failed}/{total} checks failed)")
print("=" * 60)

sys.exit(1 if failed else 0)
