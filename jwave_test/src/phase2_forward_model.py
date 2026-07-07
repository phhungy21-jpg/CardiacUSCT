"""Phase 2.2 — forward model for a single slice / single transmit.

Builds a synthetic ring phantom (LV blood pool + myocardial wall +
chest-wall-proxy background), using the cited tissue properties and
transducer geometry from phase2_config.py, and runs ONE focused transmit
from an anterior line array toward the LV center.

NOT real cardiac anatomy (per protocol Phase 3's explicit "before touching
real cardiac anatomy" guidance — that's reserved for Phase 4). This is a
reduced-scale synthetic phantom: real LV/wall dimensions are larger and the
full transthoracic chest-wall standoff path is not modeled here (background
tissue abuts the myocardium directly) — both simplifications are flagged
for collaborator review, not hidden.

GATE 2 STATUS: NOT PASSED — see phase2_config.py docstring. This script
produces a documented candidate configuration + a first sanity look, not an
established result.
"""

import os

import numpy as np
from jax import numpy as jnp
from jax import jit
from matplotlib import pyplot as plt

from jwave import FourierSeries
from jwave.geometry import Domain, Medium, TimeAxis, Sources
from jwave.acoustics import simulate_wave_propagation

import phase2_config as cfg

# --- Domain --------------------------------------------------------------
# Reduced-scale demo domain (documented simplification, see module
# docstring): 30mm x 30mm at the Phase-2-specified grid spacing. Kept
# small deliberately — this local machine has ~7.6GB free RAM, and jWave's
# default time-stepper keeps the FULL time-history array in memory
# (n_steps * Nx * Ny * 4 bytes); a 700x700 grid needed ~3.9GB for that
# array alone and hit an OOM. Phase 4's real-scale, real-anatomy runs will
# need GPU / more RAM (per protocol Appendix A) — this demo intentionally
# stays small enough for a CPU laptop.
N = (300, 300)
dx = (cfg.DX_M, cfg.DX_M)
domain = Domain(N, dx)
center = (N[0] // 2, N[1] // 2)

# --- Synthetic ring phantom -----------------------------------------------
# Reduced-scale (not literal physiological size, see module docstring):
# LV cavity radius 6mm, myocardial wall thickness 3mm (outer radius 9mm).
lv_radius_cells = 60      # 6 mm
wall_thickness_cells = 30  # 3 mm
myo_outer_radius_cells = lv_radius_cells + wall_thickness_cells

yy, xx = np.mgrid[0:N[0], 0:N[1]]
dist = np.sqrt((xx - center[1]) ** 2 + (yy - center[0]) ** 2)

label_map = np.zeros(N, dtype=int)  # 0 = background/chest-wall-proxy
label_map[dist < myo_outer_radius_cells] = 2  # myocardium
label_map[dist < lv_radius_cells] = 3         # LV cavity (blood)

sound_speed_map = np.zeros(N, dtype=np.float32)
density_map = np.zeros(N, dtype=np.float32)
for label, tissue in cfg.ACDC_LABEL_TO_TISSUE.items():
    mask = label_map == label
    sound_speed_map[mask] = tissue.sound_speed
    density_map[mask] = tissue.density

sound_speed_map = jnp.expand_dims(jnp.array(sound_speed_map), -1)
density_map = jnp.expand_dims(jnp.array(density_map), -1)

medium = Medium(
    domain=domain,
    sound_speed=FourierSeries(sound_speed_map, domain),
    density=FourierSeries(density_map, domain),
)

# --- Timing: shortened to cover transmit -> LV -> back, not full domain ---
base_time_axis = TimeAxis.from_medium(medium, cfl=cfg.CFL)
dt = base_time_axis.dt
t_end = 0.6 * base_time_axis.t_end
time_axis = TimeAxis(dt=dt, t_end=t_end)
n_steps = int(np.round(t_end / dt))
t_arr = np.arange(n_steps) * dt

# --- Anterior line array, single focused transmit at the LV center -------
# Simplification (flagged): a straight line array stands in for the
# anterior probe footprint (real curvilinear/phased sector geometry
# deferred). Positioned near the top (anterior) edge, inside the PML.
array_y = 30  # 3mm from top edge
array_span_cells = int(N[1] * 0.5)
elem_x = np.linspace(center[1] - array_span_cells // 2,
                     center[1] + array_span_cells // 2,
                     cfg.N_ELEMENTS).astype(int)
elem_y = np.full(cfg.N_ELEMENTS, array_y, dtype=int)

focus = (center[0], center[1])  # LV center
c_ref = cfg.CHEST_WALL_PROXY.sound_speed  # delay law computed in the
                                            # near-field chest-wall-proxy
                                            # medium the array sits in.
dist_to_focus = np.sqrt((elem_x - focus[1]) ** 2 +
                         (elem_y - focus[0]) ** 2) * dx[0]
delays_focus = (dist_to_focus.max() - dist_to_focus) / c_ref


def toneburst(t, t_delay):
    tau = t - t_delay
    duration = cfg.N_CYCLES / cfg.F0_HZ
    window = np.exp(-(tau - duration / 2) ** 2 / (2 * (duration / 4) ** 2))
    return np.sin(2 * np.pi * cfg.F0_HZ * tau) * window * (tau >= 0) * (tau <= duration)


signals = np.stack([toneburst(t_arr, d) for d in delays_focus])
signals = jnp.array(signals)
sources = Sources(positions=(list(elem_x), list(elem_y)), signals=signals,
                   dt=dt, domain=domain)


@jit
def run(medium):
    return simulate_wave_propagation(medium, time_axis, sources=sources)


if __name__ == "__main__":
    print("Phase 2 forward model — synthetic ring phantom, single focused "
          "transmit. GATE 2 NOT PASSED (no collaborator signoff yet).")
    print(f"Domain: {N}, dx={dx[0]*1e3:.3f}mm, f0={cfg.F0_HZ/1e6:.2f}MHz, "
          f"n_steps={n_steps}")

    pressure = run(medium)
    field = pressure.on_grid[..., 0]

    n_nan = int(jnp.sum(jnp.isnan(field)))
    n_inf = int(jnp.sum(jnp.isinf(field)))
    max_abs = float(jnp.amax(jnp.abs(field)))
    print(f"NaN count: {n_nan}, Inf count: {n_inf}, max|p|: {max_abs:.4f}")
    print("Stability check (Gate 2 checklist item):",
          "PASS (no NaN/Inf, bounded amplitude)"
          if n_nan == 0 and n_inf == 0 and np.isfinite(max_abs)
          else "FAIL")

    steps = [n_steps // 5, 2 * n_steps // 5, 3 * n_steps // 5, 4 * n_steps // 5]
    maxval = max_abs * 0.5

    fig, axes = plt.subplots(1, len(steps), figsize=(20, 5.5))
    for ax, t in zip(axes, steps):
        im = ax.imshow(field[t].T, cmap="RdBu_r", vmin=-maxval, vmax=maxval,
                        interpolation="nearest", origin="lower")
        # overlay tissue boundaries for reference
        ax.contour(label_map.T, levels=[1.5, 2.5], colors="k", linewidths=0.5)
        ax.plot(elem_x, elem_y, "g.", markersize=2)
        ax.plot(*focus[::-1], "kx", markersize=6)
        ax.set_title(f"t={t_arr[t]:.2e}s (step {t})")
        ax.axis("off")
    fig.colorbar(im, ax=axes, shrink=0.7)
    plt.tight_layout()
    out_path = "results/figures/phase2_ring_phantom_single_transmit.png"
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    plt.savefig(out_path, dpi=150)
    print(f"Saved figure to {out_path}")
