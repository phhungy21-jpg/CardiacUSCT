"""Validate attenuation_solver.py against the known analytic absorption law.

PROXY AUDIT (solo-dev stand-in, NOT Gate 2 -- see attenuation_solver.py).

Method: homogeneous medium, point source, several receivers at increasing
distance. Run BOTH the lossless jWave solver and the new attenuating
solver on the IDENTICAL geometry. Taking the ratio (attenuating peak /
lossless peak) at each receiver cancels the common geometric-spreading
factor (1/sqrt(r) in 2D), isolating the pure absorption factor
exp(-alpha*(r-r_ref)) -- directly checkable against the cited attenuation
coefficient, independent of geometric-spreading confounds.
"""

import numpy as np
from jax import numpy as jnp
from jax import jit

from jwave import FourierSeries
from jwave.geometry import Domain, Medium, TimeAxis, circ_mask
from jwave.acoustics import simulate_wave_propagation

import phase2_config as cfg
from attenuation_solver import (simulate_wave_propagation_attenuating,
                                attenuation_field_np_per_m)

N, dx = (300, 300), (cfg.DX_M, cfg.DX_M)
domain = Domain(N, dx)
center = (N[0] // 2, N[1] // 2)

# Homogeneous chest-wall-proxy medium (highest cited attenuation of our
# three tissues -- 0.74 dB/cm@1MHz -- gives the largest, most-checkable
# signal). sound_speed passed as a FourierSeries (not a bare scalar) --
# ongrid_wave_prop_params needs at least one Medium field to be a Field
# subclass to read grid params from, even for a homogeneous medium.
tissue = cfg.CHEST_WALL_PROXY
sound_speed_field = FourierSeries(
    jnp.full((*N, 1), tissue.sound_speed, dtype=jnp.float32), domain)
medium = Medium(domain=domain, sound_speed=sound_speed_field,
                density=tissue.density)

alpha_np_per_m = attenuation_field_np_per_m(tissue.attenuation, cfg.F0_HZ, y=1.0)
print(f"Tissue: {tissue.name}, cited attenuation={tissue.attenuation} dB/cm@1MHz")
print(f"-> alpha at f0={cfg.F0_HZ/1e6}MHz: {alpha_np_per_m:.4f} Np/m "
      f"(assuming y=1 linear-with-frequency, a documented simplification)")

atten_field = FourierSeries(
    jnp.full((*N, 1), alpha_np_per_m, dtype=jnp.float32), domain)

time_axis = TimeAxis.from_medium(medium, cfl=cfg.CFL)
t_end = 0.5 * time_axis.t_end
time_axis = TimeAxis(dt=time_axis.dt, t_end=t_end)

p0 = 1.0 * jnp.expand_dims(circ_mask(N, 3, center), -1)
p0 = FourierSeries(p0, domain)


@jit
def run_lossless(medium, p0):
    return simulate_wave_propagation(medium, time_axis, p0=p0)


@jit
def run_lossy(medium, p0):
    return simulate_wave_propagation_attenuating(
        medium, time_axis, atten_field, cfg.F0_HZ, p0=p0)


if __name__ == "__main__":
    print("Running lossless (baseline) simulation...")
    field_lossless = run_lossless(medium, p0).on_grid[..., 0]
    print("Running attenuating simulation...")
    field_lossy = run_lossy(medium, p0).on_grid[..., 0]

    # Receivers along +x from center at increasing distance.
    distances_cells = [40, 70, 100, 130]
    distances_m = [d * dx[0] for d in distances_cells]

    peaks_lossless = []
    peaks_lossy = []
    for d in distances_cells:
        rx, ry = center[1] + d, center[0]
        trace_ll = field_lossless[:, ry, rx]
        trace_ly = field_lossy[:, ry, rx]
        peaks_lossless.append(float(jnp.max(jnp.abs(trace_ll))))
        peaks_lossy.append(float(jnp.max(jnp.abs(trace_ly))))

    ref_ratio = peaks_lossy[0] / peaks_lossless[0]
    print(f"\n{'dist(mm)':>10} {'lossless':>12} {'lossy':>12} "
          f"{'obs.ratio':>12} {'obs.ratio/ref':>14} {'expected':>10}")
    for i, d_m in enumerate(distances_m):
        ratio = peaks_lossy[i] / peaks_lossless[i]
        rel_ratio = ratio / ref_ratio
        expected = np.exp(-alpha_np_per_m * (d_m - distances_m[0]))
        print(f"{d_m*1e3:>10.2f} {peaks_lossless[i]:>12.6f} "
              f"{peaks_lossy[i]:>12.6f} {ratio:>12.6f} {rel_ratio:>14.4f} "
              f"{expected:>10.4f}")

    print("\n'obs.ratio/ref' should match 'expected' (both relative to the "
          "first receiver) if the implementation matches the analytic "
          "exp(-alpha*distance) absorption law.")
