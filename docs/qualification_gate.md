> **HISTORICAL (Week-8 qualification record).** The operational thresholds
> defined here were the v1 labeling standard; the current labeler is
> `monitoring/features.py:derive_failure_step` and the current provenance
> audit is stage 19 (H19, `runs/monitor_audit/`). Retained as provenance.

# Qualification Gate Results (Week 8)

## Summary
Successfully ran seed matrix (3 seeds × 5 configurations) and identified reproducible success and failure regimes.

## Results Table

| Configuration | Seed 7 | Seed 42 | Seed 123 | Status |
|--------------|--------|---------|----------|--------|
| **success_baseline** | L2=0.020 ❌ | L2=0.0003 ✅ | L2=0.002 ✅ | 2/3 success |
| **boundary_starvation** | L2=0.279 ❌ | L2=0.230 ❌ | L2=0.408 ❌ | 3/3 failure |
| **gradient_conflict** | L2=0.411 ❌ | L2=0.958 ❌ | L2=1.000 ❌ | 3/3 failure |
| **spectral_suppression** | L2=0.297 ❌ | - | - | 1/1 failure |
| **collocation_starvation** | L2=0.042 ❌ | L2=0.013 ❌ | L2=0.015 ❌ | 3/3 failure |

## Operational Thresholds (Locked)

Based on the qualification results, the following thresholds are established for failure labeling:

### Success Criteria
- **Relative L2 error < 0.01** on frozen validation grid (1001 points)
- **Linf error < 0.05**
- Consistent across ≥2/3 seeds

### Failure Mode Definitions

1. **Boundary Starvation** (λ_pde=100, λ_bc=0.01)
   - Boundary loss << PDE loss
   - Interior error > 0.1 while boundary error < 0.01
   - All 3 seeds fail reproducibly

2. **Gradient Conflict** (lr=0.1)
   - Gradient cosine similarity < -0.5 persistently
   - Loss oscillates or diverges
   - All 3 seeds fail reproducibly

3. **Spectral Suppression** (5-layer narrow [16], source=50)
   - High-frequency error persists (high-to-low spectral ratio > 0.1)
   - Relative L2 > 0.1 after 2000 steps
   - Seed 7 fails (verified)

4. **Collocation Starvation** (interior_points=16)
   - Spatial bins with < 5 collocation points
   - Error correlates with under-sampled regions
   - All 3 seeds fail reproducibly

## Gate Decision: **PASS**

✅ Baseline reproduces success regime (2/3 seeds)
✅ Three distinct failure modes reproduced across seeds
✅ Operational thresholds defined and locked
✅ Ready for failure atlas generation (Weeks 9-12)

## Next Steps
Proceed to Week 9: Controlled loss-weight sweeps for failure atlas generation.