"""
Positive control for the causal-intervention pipeline (v2.1 hardening).

Purpose
-------
Before trusting a NEGATIVE causal result on real PINN activations, a skeptic
will ask: "does your SAE + hook + intervention + scoring machinery even work?"
This module answers that with a planted-ground-truth experiment:

  1. Generate synthetic hidden "activations" with KNOWN sparse features in
     superposition (more features than dimensions): a = sum_j z_j d_j.
  2. Define planted READOUTS as functions of the activation vector itself
     (exactly like a PDE loss is a function of the network output), where
     the target readout depends on ONE planted feature k* (through its
     dictionary direction) and the non-target readout on a different one.
  3. Train the real SparseAutoencoder (TopK) on these activations.
  4. Run the real measure_intervention_effect battery (hooks + three
     controls) on the SAE feature that best matches planted k*.
  5. PASS = E_T > 0 and the target effect beats all three controls.

If this passes while the real-PINN battery returns null, the negative result
is attributable to the PINN activations, not to a pipeline bug.
"""
from __future__ import annotations
from typing import Dict, Optional

import numpy as np
import torch

from sae.model import SparseAutoencoder
from interventions.engine import (
    measure_intervention_effect, measure_probe_direction_effect,
)
from interventions.causal import compute_causal_score


DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# ---------------------------------------------------------------------------
# Synthetic ground-truth construction
# ---------------------------------------------------------------------------

class PlantedFeatureWorld:
    """Synthetic activations in superposition with planted causal structure.

    The activation is a = D z + noise, D is a fixed random dictionary with
    MORE features than dimensions (superposition regime), z is k-sparse-ish
    (Bernoulli * uniform magnitudes).

    The planted losses are READOUTS ON THE ACTIVATION VECTOR:
        L_target(a) = mean((a . d_t)^2)   — energy along planted direction d_t
        L_other(a)  = mean((a . d_o)^2)   — energy along planted direction d_o
    where d_t, d_o are the dictionary columns of the planted target/other
    features.  Ablating the SAE feature that encodes d_t therefore removes
    exactly the target energy — a causal effect the real pipeline must find.
    """

    def __init__(
        self,
        n_dim: int = 64,
        n_features: int = 128,
        sparsity: float = 0.1,
        target_feature: int = 3,
        other_feature: int = 7,
        seed: int = 0,
    ):
        self.n_dim = n_dim
        self.n_features = n_features
        self.sparsity = sparsity
        self.target_feature = target_feature
        self.other_feature = other_feature
        g = torch.Generator().manual_seed(seed)

        D = torch.randn(n_dim, n_features, generator=g)
        D = D / D.norm(dim=0, keepdim=True)
        self.dictionary = D  # (n_dim, n_features)
        self._g = g

    def sample_activations(self, n: int, seed: Optional[int] = None) -> torch.Tensor:
        g = torch.Generator().manual_seed(seed) if seed is not None else self._g
        z = (torch.rand(n, self.n_features, generator=g) < self.sparsity).float()
        z = z * torch.rand(n, self.n_features, generator=g)
        a = z @ self.dictionary.T
        return a + 0.05 * torch.randn(n, self.n_dim, generator=g)


# ---------------------------------------------------------------------------
# Synthetic model + PDE stand-ins wired for the real hook machinery
# ---------------------------------------------------------------------------

class _ActHost(torch.nn.Module):
    """Hosts the planted activations; the hook intercepts its output."""
    def __init__(self, acts_bank: torch.Tensor):
        super().__init__()
        self.register_buffer("acts_bank", acts_bank)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x's first column selects rows from the bank (deterministic).
        idx = (x[:, 0].long()) % self.acts_bank.shape[0]
        return self.acts_bank[idx].to(x.device, x.dtype)


class _SyntheticModel(torch.nn.Module):
    def __init__(self, acts: torch.Tensor):
        super().__init__()
        self.acts = torch.nn.ModuleList([_ActHost(acts)])

    def forward(self, x):
        return self.acts[0](x)


class _SyntheticPDE:
    """Planted readout on the (possibly hooked) activation output.

    Design (v2.1, robust to control perturbations):
      L_target(a) = mean( relu(a . d_t) )
    — a LINEAR readout along the planted target direction d_t.

    * Ablating the SAE feature that encodes d_t removes z_t * d_t from the
      decoded activation, so a.d_t collapses toward the SAE residual error:
      a large, directed NEGATIVE shift in L_target.
    * The random-direction control replaces z_t*d_t with a random unit
      direction of matched magnitude.  A random 64-d direction has an
      expected |cos| ~ 1/sqrt(64) ~ 0.125 with d_t, so the readout changes
      by only ~12% of the ablation footprint — controls stay small BY
      CONSTRUCTION, not by luck.
    * The non-target readout is the same construction along a different
      planted direction d_o (for specificity testing).
    """
    name = "planted"
    domain = (-1.0, 1.0)

    def __init__(self, target_dir: torch.Tensor, other_dir: torch.Tensor,
                 probe_x: torch.Tensor):
        self.target_dir = target_dir
        self.other_dir = other_dir
        self.probe_x = probe_x

    def exact(self, x):
        return torch.zeros_like(x)

    def residual(self, model, x):
        out = model(x)                        # (n, n_dim), post-hook
        proj = out @ self.target_dir          # projection along target dir
        # Residual = 1 - relu(proj): ablation drops proj -> residual rises.
        e = 1.0 - torch.relu(proj)
        return e.unsqueeze(1)

    def boundary_residual(self, model):
        out = model(self.probe_x)
        proj = out @ self.other_dir
        e = 1.0 - torch.relu(proj)
        return e ** 2

    def boundary_points(self, device, dtype):
        return self.probe_x.to(device, dtype)

    def validation_grid(self, n, device, dtype):
        return torch.arange(n, device=device, dtype=dtype).unsqueeze(1)

    def sample_interior(self, n, device, dtype, seed=None):
        # Deterministic row-selector inputs.
        if seed is not None:
            g = torch.Generator().manual_seed(seed)
            rows = torch.randint(0, 10_000, (n,), generator=g)
        else:
            rows = torch.arange(n)
        return rows.to(device=device, dtype=dtype).unsqueeze(1) % 10_000


# ---------------------------------------------------------------------------
# Positive-control runner
# ---------------------------------------------------------------------------

def run_positive_control(
    n_dim: int = 64,
    n_features: int = 128,
    sparsity: float = 0.1,
    n_train: int = 8192,
    n_bank: int = 10_000,
    sae_steps: int = 2000,
    topk: int = 8,
    expansion: int = 4,
    seed: int = 0,
) -> Dict:
    """Run the planted-feature positive control through the real pipeline.

    Environment isolation: other test modules enable torch's global
    deterministic-algorithms mode, which changes CUDA kernel selection and
    can destabilise SAE training.  The control seeds all generators and
    restores default algorithm mode for its own duration.
    """
    torch.manual_seed(seed + 100)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed + 100)
    _prev_det = torch.are_deterministic_algorithms_enabled()
    torch.use_deterministic_algorithms(False)
    try:
        return _run_positive_control_impl(
            n_dim=n_dim, n_features=n_features, sparsity=sparsity,
            n_train=n_train, n_bank=n_bank, sae_steps=sae_steps,
            topk=topk, expansion=expansion, seed=seed,
        )
    finally:
        torch.use_deterministic_algorithms(_prev_det)


def _run_positive_control_impl(
    n_dim: int = 64,
    n_features: int = 128,
    sparsity: float = 0.1,
    n_train: int = 8192,
    n_bank: int = 10_000,
    sae_steps: int = 2000,
    topk: int = 8,
    expansion: int = 4,
    seed: int = 0,
) -> Dict:
    world = PlantedFeatureWorld(
        n_dim=n_dim, n_features=n_features, sparsity=sparsity, seed=seed,
    )

    # 1. Fixed activation bank + selector inputs (hooks must see identical
    #    inputs in baseline and intervened passes; only the hook differs).
    acts = world.sample_activations(n_bank, seed=seed + 1)
    model = _SyntheticModel(acts).to(DEVICE)
    target_dir = world.dictionary[:, world.target_feature].to(DEVICE)
    other_dir = world.dictionary[:, world.other_feature].to(DEVICE)
    probe_x = torch.arange(256, device=DEVICE, dtype=torch.float32).unsqueeze(1)
    pde = _SyntheticPDE(target_dir, other_dir, probe_x)

    # 2. Train the REAL TopK SAE on the same bank.
    sae = SparseAutoencoder(
        input_dim=n_dim, latent_expansion=expansion,
        activation_mode="topk", topk=topk, decoder_normalize=True,
    ).to(DEVICE)
    bank = acts.to(DEVICE)
    opt = torch.optim.Adam(sae.parameters(), lr=1e-2)
    # Private, CPU-seeded batch sampler: immune to ambient global RNG state
    # left by preceding tests (which toggle deterministic flags, CUDA
    # generators, and allocator state).
    batch_rng = torch.Generator(device="cpu").manual_seed(seed + 777)
    for step in range(sae_steps):
        idx_cpu = torch.randint(0, n_bank, (256,), generator=batch_rng)
        idx = idx_cpu.to(DEVICE)
        opt.zero_grad()
        loss, _ = sae.loss(bank[idx])
        loss.backward()
        opt.step()
        if sae.decoder_normalize:
            sae._normalise_decoder()
    sae.eval()

    # 3. Match SAE latent to the planted target direction.
    dec_cols = sae.W_d.weight.detach().T  # (latent, n_dim)
    dec_normed = dec_cols / dec_cols.norm(dim=1, keepdim=True).clamp(min=1e-8)
    cosines = dec_normed @ target_dir
    matched = int(torch.argmax(cosines).item())
    decoder_cosine = float(cosines[matched].item())

    # 4. REAL intervention battery (hook-based) on the matched latent.
    BL = "reconstruction_only"   # isolate intervention from insertion error
    targeted = measure_intervention_effect(
        model, sae, pde, DEVICE, torch.float32,
        feature_idx=matched, mode="ablate",
        n_interior=512, layer_index=0, baseline_mode=BL,
    )
    ctrl_unrelated = measure_intervention_effect(
        model, sae, pde, DEVICE, torch.float32,
        feature_idx=matched, mode="unrelated_control",
        n_interior=512, layer_index=0, baseline_mode=BL,
    )
    ctrl_random = None
    random_deltas = []
    for trial in range(5):
        r_t = measure_intervention_effect(
            model, sae, pde, DEVICE, torch.float32,
            feature_idx=matched, mode="random_direction",
            n_interior=512, layer_index=0, baseline_mode=BL,
            random_seed=trial,
        )
        random_deltas.append(r_t["delta_pde"])
    # Use the median random delta (robust to seed variance in the random
    # direction draw) and record the full spread.
    ctrl_random_delta = float(np.median(random_deltas))
    ctrl_random = {"delta_pde": ctrl_random_delta}
    probe_direction = torch.zeros(sae.latent_dim, device=DEVICE)
    probe_direction[matched] = 1.0
    ctrl_probe = measure_probe_direction_effect(
        model, sae, pde, DEVICE, torch.float32,
        feature_idx=matched, probe_direction=probe_direction,
        n_interior=512, layer_index=0, baseline_mode="reconstruction_only",
    )

    score = compute_causal_score(
        target_effect=targeted["delta_pde"],
        control_unrelated_effect=ctrl_unrelated["delta_pde"],
        control_random_effect=ctrl_random["delta_pde"],
        control_probe_effect=ctrl_probe["delta_pde"],
        non_target_effects=[targeted["delta_bc"]],
    )

    beats_all = (
        abs(targeted["delta_pde"]) > abs(ctrl_unrelated["delta_pde"])
        and abs(targeted["delta_pde"]) > abs(ctrl_random["delta_pde"])
        and abs(targeted["delta_pde"]) > abs(ctrl_probe["delta_pde"])
    )
    pipeline_pass = bool(score["causal_strength"] > 0 and beats_all)

    return {
        "pipeline_pass": pipeline_pass,
        "matched_feature": matched,
        "decoder_cosine_to_planted": decoder_cosine,
        "targeted_delta": targeted["delta_pde"],
        "ctrl_unrelated_delta": ctrl_unrelated["delta_pde"],
        "ctrl_random_delta": ctrl_random["delta_pde"],
        "ctrl_random_delta_trials": random_deltas,
        "ctrl_probe_delta": ctrl_probe["delta_pde"],
        "delta_bc_non_target": targeted["delta_bc"],
        "causal_score": score,
        "config": {
            "n_dim": n_dim, "n_features": n_features,
            "sparsity": sparsity, "sae_steps": sae_steps,
            "topk": topk, "expansion": expansion, "seed": seed,
        },
    }


if __name__ == "__main__":
    import argparse, json
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="runs/positive_control.json")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    result = run_positive_control(seed=args.seed)
    with open(args.out, "w") as f:
        json.dump(result, f, indent=2)
    print(json.dumps({k: v for k, v in result.items()
                      if k not in ("causal_score", "config")}, indent=2))
    print("PIPELINE PASS:", result["pipeline_pass"])