"""Toy 2D jWave linear-array transducer source — CPU.

Closer to actual USCT transmit geometry than a single point source: a line
of elements emits a short toneburst. Two transmit modes are compared:
  - plane-wave transmit: all elements fire in sync -> roughly planar
    wavefront with edge diffraction at the array ends.
  - focused transmit: per-element delays chosen so every element's pulse
    arrives at a chosen focal point simultaneously -> converging wavefront.

Homogeneous medium (1500 m/s, water-like) — no tissue heterogeneity here,
this toy is only about the source geometry/delay-law, not tissue physics.
"""

import os

import numpy as np
from jax import numpy as jnp
from jax import jit
from matplotlib import pyplot as plt

from jwave import FourierSeries
from jwave.geometry import Domain, Medium, TimeAxis, Sources
from jwave.acoustics import simulate_wave_propagation

N, dx = (128, 128), (0.1e-3, 0.1e-3)
domain = Domain(N, dx)
c0 = 1500.0
medium = Medium(domain=domain, sound_speed=c0)

# Shorten t_end relative to the default corner-to-corner travel time — we
# only need to see the wavefront cross most of the domain, not bounce off
# the PML-absorbed edges.
base_time_axis = TimeAxis.from_medium(medium, cfl=0.3)
dt = base_time_axis.dt
t_end = 0.75 * base_time_axis.t_end
time_axis = TimeAxis(dt=dt, t_end=t_end)
n_steps = int(np.round(t_end / dt))
t_arr = np.arange(n_steps) * dt

# 16-element line array at x=20, spanning y=24..104.
n_elements = 16
elem_x = np.full(n_elements, 20, dtype=int)
elem_y = np.linspace(24, 104, n_elements).astype(int)

f0 = 2e6      # 2 MHz center frequency
n_cycles = 3  # toneburst length


def toneburst(t, t_delay):
    """Gaussian-windowed toneburst centered at t_delay."""
    tau = t - t_delay
    duration = n_cycles / f0
    window = np.exp(-(tau - duration / 2) ** 2 / (2 * (duration / 4) ** 2))
    return np.sin(2 * np.pi * f0 * tau) * window * (tau >= 0) * (tau <= duration)


def build_signals(delays):
    signals = np.stack([toneburst(t_arr, d) for d in delays])
    return jnp.array(signals)


# --- Mode 1: plane-wave transmit (zero delay for every element) ---
delays_plane = np.zeros(n_elements)
signals_plane = build_signals(delays_plane)
sources_plane = Sources(positions=(list(elem_x), list(elem_y)),
                         signals=signals_plane, dt=dt, domain=domain)

# --- Mode 2: focused transmit, focal point at (100, 64) ---
focus = (100, 64)
dist_to_focus = np.sqrt((elem_x - focus[0]) ** 2 + (elem_y - focus[1]) ** 2) * dx[0]
delays_focus = (dist_to_focus.max() - dist_to_focus) / c0
signals_focus = build_signals(delays_focus)
sources_focus = Sources(positions=(list(elem_x), list(elem_y)),
                         signals=signals_focus, dt=dt, domain=domain)


@jit
def run_plane(medium):
    return simulate_wave_propagation(medium, time_axis, sources=sources_plane)


@jit
def run_focus(medium):
    return simulate_wave_propagation(medium, time_axis, sources=sources_focus)


if __name__ == "__main__":
    print("Running toy 2D plane-wave array transmit on CPU...")
    p_plane = run_plane(medium).on_grid[..., 0]
    print("Running toy 2D focused array transmit on CPU...")
    p_focus = run_focus(medium).on_grid[..., 0]

    steps = [n_steps // 4, n_steps // 2, int(n_steps * 0.75)]
    maxval = max(float(jnp.amax(jnp.abs(p_plane))),
                 float(jnp.amax(jnp.abs(p_focus)))) * 0.6

    fig, axes = plt.subplots(2, len(steps), figsize=(14, 9))
    for row, (label, field) in enumerate([("plane-wave", p_plane),
                                           ("focused", p_focus)]):
        for col, t in enumerate(steps):
            ax = axes[row, col]
            im = ax.imshow(field[t].T, cmap="RdBu_r", vmin=-maxval, vmax=maxval,
                            interpolation="nearest", origin="lower")
            ax.plot(elem_x, elem_y, "g.", markersize=3)
            if label == "focused":
                ax.plot(*focus, "kx", markersize=8)
            ax.set_title(f"{label}, t={t_arr[t]:.2e}s")
            ax.axis("off")
    fig.colorbar(im, ax=axes, shrink=0.7)
    plt.tight_layout()
    out_path = "results/figures/toy_2d_array_source_wavefronts.png"
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    plt.savefig(out_path, dpi=150)
    print(f"Saved figure to {out_path}")
