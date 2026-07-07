"""Toy 2D jWave sanity run — CPU, no GPU needed.

Reproduces jWave's own documented "Homogeneous Medium" reference example
(https://ucl-bug.github.io/jwave/notebooks/ivp/homogeneous_medium.html):
a circular point source in a constant-sound-speed medium. This is an
exploratory smoke test to see real signals locally — it is NOT a substitute
for Gate 1's GPU-timed reference reproduction (still pending, see
jwave/notebooks/phase1_gate1_reference_repro.ipynb and jwave/LOG.md).
"""

from jax import numpy as jnp
from jax import jit
from matplotlib import pyplot as plt

from jwave import FourierSeries
from jwave.geometry import Domain, Medium, TimeAxis, circ_mask
from jwave.acoustics import simulate_wave_propagation

N, dx = (128, 128), (0.1e-3, 0.1e-3)
domain = Domain(N, dx)

medium = Medium(domain=domain, sound_speed=1500.0)
time_axis = TimeAxis.from_medium(medium, cfl=0.3)

p0 = 1.0 * jnp.expand_dims(circ_mask(N, 4, (80, 60)), -1)
p0 = FourierSeries(p0, domain)


@jit
def compiled_simulator(medium, p0):
    return simulate_wave_propagation(medium, time_axis, p0=p0)


if __name__ == "__main__":
    print("Running toy 2D homogeneous-medium simulation on CPU...")
    pressure = compiled_simulator(medium, p0)
    print("Done. Output shape:", pressure.on_grid.shape)

    field = pressure.on_grid[..., 0]  # (n_steps, Nx, Ny)
    maxval = float(jnp.amax(jnp.abs(field)))

    fig, axes = plt.subplots(1, 4, figsize=(16, 4))
    for ax, t in zip(axes, [0, 100, 200, 300]):
        im = ax.imshow(field[t], cmap="RdBu_r", vmin=-maxval, vmax=maxval,
                        interpolation="nearest")
        ax.set_title(f"t={time_axis.to_array()[t]:.2e}s (step {t})")
        ax.axis("off")
    fig.colorbar(im, ax=axes, shrink=0.8)
    out_path = "results/figures/toy_2d_homogeneous_wavefronts.png"
    import os
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    plt.savefig(out_path, dpi=150)
    print(f"Saved figure to {out_path}")
