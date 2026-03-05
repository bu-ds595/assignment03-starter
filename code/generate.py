"""
Generation script for Assignment 3: Jet Generation Challenge.

This script must be self-contained: it defines the model architecture,
loads trained parameters, generates jets, and saves submission.npz.

Usage:
    python generate.py                  # Full submission: 2000 jets/type
    python generate.py --n-samples 10   # Quick test: 10 jets/type

TODO: Replace SimpleMLPVelocity with your model and update the generation
code as needed (e.g., guidance, number of steps, etc.).
"""

import argparse

import flax.linen as nn
import jax
import jax.numpy as jnp
import jax.random as jr
import numpy as np
from utils import JET_TYPES, N_FEATURES, N_PARTICLES, N_TYPES, load_model, sinusoidal_embedding

N_SAMPLES = 2000  # per jet type


# ── Model definition ──────────────────────────────────────────
# TODO: Replace this with your actual model architecture.
# It must match the architecture used to train model_params.pkl.


class SimpleMLPVelocity(nn.Module):
    hidden_dim: int = 128
    n_layers: int = 2
    n_types: int = N_TYPES
    time_dim: int = 32

    @nn.compact
    def __call__(self, x_t, t, y, mask):
        B = x_t.shape[0]
        te = nn.relu(nn.Dense(self.hidden_dim)(sinusoidal_embedding(t, self.time_dim)))
        y_onehot = jax.nn.one_hot(y, self.n_types + 1)
        x_flat = x_t.reshape(B, -1)
        mask_flat = mask.reshape(B, -1)
        h = jnp.concatenate([x_flat, mask_flat, te, y_onehot], axis=-1)
        for _ in range(self.n_layers):
            h = nn.silu(nn.Dense(self.hidden_dim)(h))
        v = nn.Dense(N_PARTICLES * N_FEATURES)(h)
        v = v.reshape(B, N_PARTICLES, N_FEATURES)
        return v * mask[:, :, None]


# ── Generation ────────────────────────────────────────────────


def sample_jets(model, params, jet_type_idx, masks_ref, key, n_samples=N_SAMPLES, steps=100):
    """Generate jets via Euler integration.

    Feel free to modify, but don't add additional dependencies, keep the function signature the same for grading.
    """
    dt = 1.0 / steps
    k1, k2 = jr.split(key)
    mask_idx = jr.randint(k1, (n_samples,), 0, masks_ref.shape[0])
    masks = jnp.array(masks_ref[np.array(mask_idx)])
    x = jr.normal(k2, (n_samples, N_PARTICLES, N_FEATURES)) * masks[:, :, None]
    y = jnp.full((n_samples,), jet_type_idx, dtype=jnp.int32)
    for i in range(steps):
        t = jnp.full((n_samples,), i * dt)
        v = model.apply(params, x, t, y, masks)
        x = x + dt * v
    return np.array(x * masks[:, :, None]), np.array(masks)


# ── Main ─────────────────────────────────────────────────


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--n-samples", type=int, default=N_SAMPLES, help="Jets per type (default: 2000)")
    args = parser.parse_args()
    n = args.n_samples

    # Load model
    model = SimpleMLPVelocity()  # TODO: Replace with your model
    params = load_model("model_params.pkl")

    # Load training masks (needed to sample realistic particle multiplicities)
    train_npz = np.load("data/train.npz")
    train_masks = {jt: train_npz[f"{jt}_masks"] for jt in JET_TYPES}

    # Generate
    gen_jets, gen_masks = {}, {}
    for i, jt in enumerate(JET_TYPES):
        print(f"Generating {n} {jt} jets...")
        gen_jets[jt], gen_masks[jt] = sample_jets(model, params, i, train_masks[jt], jr.PRNGKey(i), n_samples=n)

    # Save
    save_dict = {}
    for jt in JET_TYPES:
        save_dict[f"{jt}_jets"] = gen_jets[jt]
        save_dict[f"{jt}_masks"] = gen_masks[jt]
    np.savez_compressed("submission.npz", **save_dict)
    print("Saved submission.npz")
    for jt in JET_TYPES:
        print(f"  {jt}: {gen_jets[jt].shape}")


if __name__ == "__main__":
    main()
