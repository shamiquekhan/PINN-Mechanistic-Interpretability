# Security Policy

## Supported versions

The `main` branch is the only supported version. Releases are tagged from
it (see CHANGELOG.md).

## Reporting a vulnerability

This repository is a research codebase, but it is public and executed by
others. If you find a security-relevant issue — for example, unsafe
deserialization of untrusted checkpoints (`torch.load` is used on
first-party artifacts only, but a malicious `runs/` tree could matter),
injection via config files, or anything exploitable via a crafted
artifact — please report it **privately**:

1. Use GitHub's *Report a vulnerability* option on the Security tab, or
2. Open a security advisory on the repository.

Please do not open a public issue for security reports. We aim to respond
within 72 hours.

## Scope

In scope: the code in this repository as executed on a user-controlled
machine, including `experiments/`, `analysis/`, `figures/`, and
`scripts/`.

Out of scope: the scientific claims (challenge those via the normal issue
tracker — see CONTRIBUTING.md), and any deployment of this research code
in production settings (it is not designed for that).

## Known notes

* `torch.load` calls in `pinn_logging/io.py` and experiment loaders use
  `weights_only=False` on **first-party** checkpoints. Do not run the
  pipeline against checkpoints from untrusted sources.
* Configs are YAML parsed with `yaml.safe_load` (no arbitrary object
  construction).
* No network access is required anywhere in the pipeline; the only
  download is dependency installation.
