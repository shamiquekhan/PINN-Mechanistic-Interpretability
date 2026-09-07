# Fresh Full-Campaign Record — 2026-09-07

**Scope:** complete pipeline rerun, stages 1–13, from scratch (fresh
training, fresh SAEs, fresh batteries, fresh analysis). Recorded for
reproducibility: this document is the diff against the previously published
numbers in RESULTS.md and `data/manifest.json` (v3.0.0 → v3.1.0).

**Driver:** `python -m experiments.run_pipeline` (stages 2–13 after stage-1
training of the 7 base configs; 10-seed matrices retained from the
qualification campaign as designed — stage 1 trains the base configs, the
seed matrix is produced by `experiments/qualification.py`).

---

## 1. Issues found and fixed during the rerun

Two latent bugs surfaced because this was the first time stage 1 was
executed end-to-end on the full config list (the committed tree's base
runs predated the pipeline runner):

1. **Stage 1 passed `--log-diagnostics` unconditionally, and
   `experiments/train.py` raised for non-Poisson PDEs.** The advection and
   reaction-diffusion baselines silently failed in stage 1 (the runner
   captures stderr and continues). *Fix:* train.py now degrades gracefully
   (diagnostics stream disabled for unsupported PDEs, with a printed
   notice) instead of crashing the run.
2. **The reaction-diffusion baseline config was degenerate:** μ=1, f=0,
   homogeneous Dirichlet BCs ⇒ exact solution u≡0 ⇒ relative-L2 = ‖pred‖/0.
   The previously committed numbers for that run came from a truncated
   pre-v2 run and were not reproducible under the current PDE mapping.
   *Fix:* `forcing: 1.0` (non-degenerate manufactured solution, exact norm
   ≈ 29.2); fresh run converges to rel_l2 ≈ 0.0015.

## 2. Fresh headline results vs published

Fresh values from `runs/` (regenerated 2026-09-07; checksums in
`data/checksums.sha256`). 18 of 19 machine-verified checks within the
documented tolerance; the single drift is explained below.

| Check | Published (v3.0.0) | Fresh (v3.1.0) | Drift | Verdict |
|---|---:|---:|---:|---|
| Positive control passes | True | True | — | reproduce |
| SAE E_T mean | −0.0002 | +0.0020 | +0.0022 | reproduce (CI spans 0 both times) |
| SAE E_T CI lower | −0.0051 | −0.0034 | +0.0017 | reproduce |
| SAE Bonferroni / BH-FDR survivors | 0/8, 0/8 | 0/8, 0/8 | — | reproduce |
| SAE sign diagnostic | 88/88 positive | 88/88 positive | — | reproduce |
| PCA E_T mean | −0.0186 | −0.0203 | −0.0017 | reproduce |
| PCA CI excludes zero (negative) | True | True | — | reproduce |
| PCA Bonferroni survivors | 0/8 | 0/8 | — | reproduce |
| Abstraction: PCA beats-random-every-run | False | False | — | reproduce |
| Abstraction: SAE beats-random-every-run | False | False | — | reproduce |
| FNO covariance PR | 6.80 | 6.80 | +0.002 | reproduce (bit-stable task) |
| FNO: SAE beats k-matched PCA | True | True | — | reproduce (4.7×) |
| Monitor conventional AUROC | 0.872 | 0.859 | −0.014 | reproduce (split-level noise) |
| Monitor SAE AUROC | 0.878 | 0.864 | −0.014 | reproduce (no SAE advantage either way) |
| Controller final rel L2 | 0.0166 | 0.0166 | 0.0000 | reproduce (exact) |
| No-action final | 0.2953 | 0.2953 | 0.0000 | reproduce (exact) |
| SAE ≡ random monitor (bit-identical) | True | True | — | reproduce |
| NTK-adaptive final | 0.0064 | 0.0064 | 0.0000 | reproduce (exact) |
| MDE (sign-test power) | 85.7% | 85.7% | — | reproduce |
| **Geometry mean PR (1D suite)** | **1.34** | **1.778** | **+0.44** | **explained update (see §3)** |

Fresh per-family geometry (stage 10, tangent rank = input dimension
everywhere — the provable bound saturated):

| Family | Mean PR | Tangent rank |
|---|---:|:---:|
| Advection-Diffusion 2D | 2.47 | 2 |
| Reaction-Diffusion 2D | 2.29 | 2 |
| Burgers (t,x) | 1.42 | 2 |
| Allen-Cahn (t,x) | 1.92 | 2 |

Fresh stage-3/4 auxiliary numbers (scale shift, see §3): TopK SAE test
reconstruction MSE 2.17–2.64 (replicas), k-matched PCA 4.4×10⁻² (PCA
dominance now ≈ 50×, was 12×), dead latents 47–62%, dictionary
associations real≫random on every view (e.g. grad_cosine 0.226 vs 0.013;
data_std 0.93 vs 0.016 — confound flagged as always).

## 3. The one legitimate drift: geometry mean PR 1.34 → 1.78

The pooled 1D-suite mean PR is computed over all logged runs. In the
previously committed tree, the reaction-diffusion baseline was the
degenerate u≡0 configuration: its near-zero-variance activations dragged
the pooled mean down. With the corrected non-degenerate forcing (a
genuine stiff reaction-diffusion solution with boundary layers), that
run's activations are normal-scale and the pooled mean rises to 1.78 —
the honest number for the corrected benchmark. Every claim keyed to PR
is unaffected:

- PR/W is still 0.028 — deep in the low-rank regime (verdict unchanged:
  "LOW-RANK / NOT IN SUPERPOSITION").
- 95% energy still needs ~2.2 components (was 1.83).
- The FNO boundary (PR 6.8) is unchanged, so the regime gap is ~3.8×
  (was ~5×) — same conclusion, same side of the boundary.
- RESULTS.md §5 numbers quoted from the old pool are superseded by the
  fresh report; the claim ladder is untouched.

## 4. Post-rerun verification

- `python -m pytest tests/ -q` → **139 passed**.
- `sha256sum -c data/checksums.sha256` → **16/16 OK** (regenerated for
  v3.1.0).
- Figures regenerated: 8/8 (`scripts/generate_figures.sh`).
- Tables regenerated: 6/6 (`scripts/generate_tables.sh`).
- Machine verification: 18/19 checks within tolerance, the 19th explained
  in §3 (see the drift table above; the same logic drives
  `scripts/reproduce_main_results.sh`, whose expected-value tolerances
  accommodate this drift).

## 5. Command to reproduce this record

```bash
bash scripts/run_full_pipeline.sh          # retrains everything (~6 GPU-h)
bash scripts/reproduce_main_results.sh     # verification block + figures + tables
sha256sum -c data/checksums.sha256          # artifact integrity
```
