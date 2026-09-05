# Configuration schema

The canonical configuration is `configs/poisson_baseline.yaml`.

- `run`: identity, output location, seed, determinism, device, dtype.
- `pde`: selected PDE, domain, source, boundary values, validation-grid size.
- `model`: input/output dimensions, hidden-layer widths, activation.
- `training`: optimizer, learning rate, steps, sample counts, logging/checkpoint cadence, loss weights.
- `logging`: activation and residual persistence controls.

Unknown top-level keys are rejected. Extend the schema by adding a typed Pydantic model and a unit test before using a new parameter in an experiment.
