"""Toy 2D jWave two-tissue reflection/refraction sanity check — CPU.

A step toward Phase 2 realism: introduces a single planar interface between
blood and myocardium and checks that a point source produces the expected
qualitative behavior — partial reflection back toward the source, partial
transmission across the interface with a wavespeed change, and a receiver
time-trace on the source side showing two distinct arrivals (direct +
reflected).

Sound speed / density values are CITED, real tissue values (not invented),
per CLAUDE.md's requirement for Phase 2 tissue-property assignment:

    Blood:          c = 1584 m/s, rho = 1060 kg/m^3
    Cardiac muscle: c = 1576 m/s, rho = 1060 kg/m^3

Source: Mast, T.D. (2000). "Empirical relationships between acoustic
parameters in human soft tissues." Acoustics Research Letters Online,
1(2), 37-42, Table 1 (blood and "Muscle, cardiac" rows) — itself compiling
values from ICRU Report 61 (1998) and Duck, F.A. (1990) "Physical
Properties of Tissue: A Comprehensive Reference Book".

Note the blood/myocardium impedance contrast is very small (identical
density, ~0.5% sound-speed difference -> analytic |R| ~ 0.0025), which is
consistent with the well-known clinical difficulty of resolving the
blood-myocardium (endocardial) border in echocardiography. This run
doubles as an early, quantitative look at whether that weak contrast is
even visible above this simulation's numerical noise floor at this grid
resolution — informative either way, per Gate 2's spirit.
"""

import os

from jax import numpy as jnp
from jax import jit
import numpy as np
from matplotlib import pyplot as plt

from jwave import FourierSeries
from jwave.geometry import Domain, Medium, TimeAxis, circ_mask
from jwave.acoustics import simulate_wave_propagation

N, dx = (128, 128), (0.1e-3, 0.1e-3)
domain = Domain(N, dx)

# Two-medium split: vertical interface at x = 64 grid points.
# Medium 1 (x < 64): blood, c=1584 m/s, rho=1060 kg/m^3
# Medium 2 (x >= 64): cardiac muscle (myocardium), c=1576 m/s, rho=1060 kg/m^3
# Values cited from Mast (2000), Table 1 — see module docstring for source.
interface_x = 64
c1, rho1 = 1584.0, 1060.0
c2, rho2 = 1576.0, 1060.0

x_idx = np.arange(N[0])[:, None] * np.ones((1, N[1]))
sound_speed = np.where(x_idx < interface_x, c1, c2).astype(np.float32)
density = np.where(x_idx < interface_x, rho1, rho2).astype(np.float32)

sound_speed = jnp.expand_dims(jnp.array(sound_speed), -1)
density = jnp.expand_dims(jnp.array(density), -1)

medium = Medium(
    domain=domain,
    sound_speed=FourierSeries(sound_speed, domain),
    density=FourierSeries(density, domain),
)
time_axis = TimeAxis.from_medium(medium, cfl=0.3)

# Source on the medium-1 side, centered in y, offset from the interface.
source_pos = (30, 64)
receiver_pos = (10, 64)  # further back, to catch direct + reflected arrivals

p0 = 1.0 * jnp.expand_dims(circ_mask(N, 3, source_pos), -1)
p0 = FourierSeries(p0, domain)

# Analytic normal-incidence plane-wave reflection coefficient, for reference
# only (the actual source is a cylindrical point source, not a plane wave,
# so this is an order-of-magnitude / sign check, not an exact prediction).
Z1, Z2 = rho1 * c1, rho2 * c2
R_analytic = (Z2 - Z1) / (Z2 + Z1)


@jit
def compiled_simulator(medium, p0):
    return simulate_wave_propagation(medium, time_axis, p0=p0)


if __name__ == "__main__":
    print(f"Analytic normal-incidence reflection coefficient (plane-wave "
          f"reference, not an exact prediction for this point source): "
          f"R = {R_analytic:.4f}")
    print("Running toy 2D two-tissue reflection simulation on CPU...")
    pressure = compiled_simulator(medium, p0)
    field = pressure.on_grid[..., 0]  # (n_steps, Nx, Ny)
    print("Done. Output shape:", field.shape)

    maxval = float(jnp.amax(jnp.abs(field))) * 0.6

    steps = [0, 60, 120, 200]
    fig, axes = plt.subplots(1, len(steps), figsize=(18, 4.5))
    for ax, t in zip(axes, steps):
        im = ax.imshow(field[t].T, cmap="RdBu_r", vmin=-maxval, vmax=maxval,
                        interpolation="nearest", origin="lower")
        ax.axvline(interface_x, color="k", linewidth=0.8, linestyle="--")
        ax.plot(*source_pos, "g+", markersize=8)
        ax.set_title(f"t={time_axis.to_array()[t]:.2e}s (step {t})")
        ax.axis("off")
    fig.colorbar(im, ax=axes, shrink=0.8)
    out_path = "results/figures/toy_2d_blood_myocardium_reflection_wavefronts.png"
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    plt.savefig(out_path, dpi=150)
    print(f"Saved wavefront figure to {out_path}")

    # Receiver time-trace: should show a direct-arrival pulse followed by a
    # smaller reflected-arrival pulse (both negative-going or positive-going
    # consistently, since R here should be small and positive: Z2 > Z1).
    trace = np.array(field[:, receiver_pos[0], receiver_pos[1]])
    t_arr = np.array(time_axis.to_array())

    plt.figure(figsize=(8, 4))
    plt.plot(t_arr, trace)
    plt.xlabel("time (s)")
    plt.ylabel("pressure (a.u.)")
    plt.title(f"Receiver trace at {receiver_pos} (medium-1 side)")
    plt.tight_layout()
    trace_path = "results/figures/toy_2d_blood_myocardium_reflection_trace.png"
    plt.savefig(trace_path, dpi=150)
    print(f"Saved receiver trace figure to {trace_path}")
