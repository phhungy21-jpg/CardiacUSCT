"""Toy 3D jWave sanity run — CPU, no GPU needed.

3D analogue of toy_2d_homogeneous.py: a spherical point source in a
constant-sound-speed medium. Grid kept small (64^3) to stay CPU-tractable;
this is exploratory only, not Gate 1 (still pending on GPU, see
jwave/notebooks/phase1_gate1_reference_repro.ipynb).
"""

import os

from jax import numpy as jnp
from jax import jit
from matplotlib import pyplot as plt

from jwave import FourierSeries
from jwave.geometry import Domain, Medium, TimeAxis, sphere_mask
from jwave.acoustics import simulate_wave_propagation

N, dx = (64, 64, 64), (0.2e-3, 0.2e-3, 0.2e-3)
domain = Domain(N, dx)

medium = Medium(domain=domain, sound_speed=1500.0)
time_axis = TimeAxis.from_medium(medium, cfl=0.3)

p0 = 1.0 * jnp.expand_dims(sphere_mask(N, 3, (32, 32, 32)), -1)
p0 = FourierSeries(p0, domain)


@jit
def compiled_simulator(medium, p0):
    return simulate_wave_propagation(medium, time_axis, p0=p0)


if __name__ == "__main__":
    print("Running toy 3D homogeneous-medium simulation on CPU...")
    pressure = compiled_simulator(medium, p0)
    print("Done. Output shape:", pressure.on_grid.shape)

    field = pressure.on_grid[..., 0]  # (n_steps, Nx, Ny, Nz)
    mid_z = N[2] // 2
    maxval = float(jnp.amax(jnp.abs(field)))

    steps = [0, 40, 80, 120]
    fig, axes = plt.subplots(1, len(steps), figsize=(16, 4))
    for ax, t in zip(axes, steps):
        im = ax.imshow(field[t, :, :, mid_z], cmap="RdBu_r",
                        vmin=-maxval, vmax=maxval, interpolation="nearest")
        ax.set_title(f"t={time_axis.to_array()[t]:.2e}s (step {t}), z-mid slice")
        ax.axis("off")
    fig.colorbar(im, ax=axes, shrink=0.8)

    out_path = "results/figures/toy_3d_homogeneous_wavefronts.png"
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    plt.savefig(out_path, dpi=150)
    print(f"Saved figure to {out_path}")
