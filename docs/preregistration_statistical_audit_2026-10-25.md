# Prospective Statistical and Control-Semantics Audit

**Registered:** October 25, 2026, before any new audit report is generated.

**Scope:** This audit preserves the frozen publication campaign and its
artifacts unchanged. It does not replace or reinterpret Stage 5, Stage 8, or
the operator-battery results. The audit has two separate parts: intervention
provenance and synthetic calibration of the causal statistic.

## A1 — Operator provenance

Every new intervention measurement must record:

- the requested mode and the operator actually executed;
- target index and selected non-target control index, when applicable;
- random seed;
- selected random/probe direction digest, dimension, and norm;
- per-row perturbation norm summary;
- control no-op and operator no-op flags.

The audit must treat these as distinct control families:

1. activity-matched component deletion/scaling;
2. random active component deletion/scaling;
3. norm-matched random-direction injection;
4. supervised probe-direction injection.

No control family may be described as another family merely because both are
called “random” or both use a matched dose.

## A2 — Causal-statistic calibration

The audited statistic is the one used by the frozen implementation:

```text
d = |target_effect| - max(|control_effects|)
```

The existing one-sided exact sign test remains unchanged for historical
artifacts. The calibration suite evaluates its rejection behavior at
`alpha = 0.05` with 11 paired observations over 2,000 synthetic trials for:

1. a sign-exchangeable symmetric difference null;
2. an exchangeable target/control effect null with the actual max-control
   statistic;
3. a skewed median-zero null;
4. an all-zero tie case;
5. a planted positive alternative.

The sign-flip randomization test is compared only for the first case, where
sign exchangeability is part of the data-generating process. It is not applied
to the max-control exchangeability case because comparing one target draw with
the maximum of three control draws is not sign-exchangeable.

## A3 — Prospective interpretation rules

- The historical campaign is not re-scored by this audit.
- A calibration failure is a property of the test/statistic pairing, not
  evidence for or against a physical mechanism.
- New model runs must use the provenance schema before their results can be
  used in a causal claim.
- A future PCA battery must report matched deletion and random-direction
  injection as separate controls, with separate effect columns and no pooled
  label that implies equivalence.
- Any change to the test or control family requires a new preregistration and
  a new prospective artifact.

## Reproduction

```bash
python scripts/calibrate_causal_statistic.py
```

The generated report is written to
`runs/statistical_calibration/causal_calibration_report.json` and is separate
from the frozen campaign artifacts.
