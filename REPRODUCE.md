# Reproduce the Paper

This document provides copy-pasteable reproduction instructions at three
levels of depth. Every command assumes you have cloned the repository and
are in its root directory.

---

## Level 1 — Verify repository integrity (< 5 min)

No GPU required. Validates that every committed artifact is consistent
with the documented claims.

```bash
# Install dependencies
pip install -r requirements.txt

# Run the full test suite
python -m pytest tests/ -q

# Verify every headline number in RESULTS.md matches its source artifact
python scripts/check_results_grounded.py

# Verify artifact integrity (SHA256 checksums)
sha256sum -c data/checksums.sha256

# Run the comprehensive repository audit (30 checks)
python scripts/final_repo_audit.py
```

Expected output from the audit:

```
OVERALL: PASS (30/30 checks passed)
```

---

## Level 2 — Regenerate headline analysis (< 30 min)

Re-runs the geometry, compression, and causal analyses from committed
activation artifacts. CPU-only for geometry/comparison; GPU optional
for the equivalence test.

```bash
# R8 — hierarchical analysis (CPU, ~2 min)
python scripts/hierarchical_analysis.py

# R10 — geometry predictor analysis (CPU, ~1 min)
python scripts/geometry_predictors.py

# R7 — equivalence test (CPU, ~1 min)
python scripts/equivalence_test.py

# R1 — PhysSAE reconciliation table (CPU, ~2 min)
python scripts/r1_reconciliation_table.py
```

---

## Level 3 — Reproduce specific experiments

Each experiment has a dedicated driver. These require a CUDA GPU and
take 10–60 minutes depending on the experiment.

### Geometry boundary (H18)

```bash
python -m experiments.architecture_boundary
# Output: runs/architecture_boundary/architecture_boundary_report.json
```

### Fourier frequency sweep (R3)

```bash
python -m experiments.r3_frequency_sweep
# Output: runs/r3_frequency_sweep/r3_report.json
```

### Burgers boundary (R5)

```bash
python -m experiments.r5_burgers_boundary
# Output: runs/r5_burgers_boundary/r5_report.json
```

### Reaction–diffusion boundary (R9)

```bash
python -m experiments.r9_rd_boundary
# Output: runs/r9_rd_boundary/r9_report.json
```

### PhysSAE head-to-head (R1)

```bash
python -m experiments.r1_physSAE_head_to_head
# Output: runs/r1_physSAE/r1_report.json
```

### Causal battery on Fourier PINN (R2)

```bash
python -m experiments.r2_fourier_causal
# Output: runs/r2_fourier_causal/r2_report.json
```

### Dose-response (R6)

```bash
python -m experiments.r6_dose_response
# Output: runs/r6_dose_response/r6_report.json
```

### Power-upgraded equivalence replication (R7+)

```bash
python -m experiments.r7plus_equivalence
# Output: runs/r7plus_equivalence/r7plus_report.json
```

### SAE-seed robustness (R4)

```bash
python -m experiments.r4_sae_seed_robustness
# Output: runs/r4_sae_seed_robustness/r4_report.json
```

---

## Level 4 — Full 15-stage campaign (~6 GPU-hours)

Re-runs the entire original pipeline from scratch. This is the most
expensive option and is typically not needed for verification.

```bash
python -m experiments.run_pipeline
# Stages 1–15; outputs in runs/
```

---

## Generating figures and tables

```bash
# All publication figures (from runs/ JSONs only)
bash scripts/generate_figures.sh

# All LaTeX tables
bash scripts/generate_tables.sh
```

---

## Smoke test (CI-safe, < 5 min)

```bash
bash scripts/run_smoke_test.sh
```

Runs: test suite + 60-step training + geometry analysis. Suitable for
continuous integration.

---

## Environment

- **Python:** 3.10+
- **PyTorch:** 2.6+ with CUDA (for Level 3/4)
- **OS:** Linux (tested on Ubuntu 22.04)
- **GPU:** any CUDA-capable GPU (tested on GTX 1650 4GB)
- **Full environment:** see `requirements.txt` (minimum) or
  `requirements.lock` (pinned reference)

The CPU-only verification (Level 1 and parts of Level 2) runs on any
machine with Python 3.10+ and PyTorch installed.
