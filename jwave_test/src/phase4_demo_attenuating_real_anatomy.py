"""Phase 4 technical demonstration — real anatomy + validated attenuation
+ calibrated amplitude, all in one forward model.

STATUS: PENDING ACOUSTIC-PHYSICS SIGNOFF (see labels.py). This is a
capability demonstration following the PROXY_AUDIT.md fixes -- NOT Phase
4.2's real dataset generation (still blocked, see
PHASE2_TO_PHASE3_DIAGNOSTIC_HANDOFF.md "Phase 3->4 hard gate"). It exists
to show the full pipeline (real segmentation -> cited tissue properties ->
validated attenuation -> calibrated amplitude) runs end to end on one real
patient, ready for collaborator review -- not to produce a result used in
any Phase 5 conclusion.
"""

import os

import numpy as np
from jax import numpy as jnp
from jax import jit
from matplotlib import pyplot as plt

from jwave import FourierSeries
from jwave.geometry import Domain, Medium, TimeAxis, Sources

import phase2_config as cfg
import phase4_benchmark as p4
import calibration
import labels
from attenuation_solver import (simulate_wave_propagation_attenuating,
                                attenuation_field_np_per_m)

N = 350  # the one benchmark size with a genuine tissue boundary (see PROXY_AUDIT.md item 6)
dx = (cfg.DX_M, cfg.DX_M)
domain = Domain((N, N), dx)

full_map = p4.load_real_mask_upsampled()
label_map = p4.crop_center(full_map, N)

sound_speed_map = np.zeros((N, N), dtype=np.float32)
density_map = np.zeros((N, N), dtype=np.float32)
atten_map = np.zeros((N, N), dtype=np.float32)
for label, tissue in cfg.ACDC_LABEL_TO_TISSUE.items():
    m = label_map == label
    sound_speed_map[m] = tissue.sound_speed
    density_map[m] = tissue.density
    atten_map[m] = attenuation_field_np_per_m(tissue.attenuation, cfg.F0_HZ, y=1.0)

medium = Medium(
    domain=domain,
    sound_speed=FourierSeries(jnp.expand_dims(jnp.array(sound_speed_map), -1), domain),
    density=FourierSeries(jnp.expand_dims(jnp.array(density_map), -1), domain),
)
atten_field = FourierSeries(jnp.expand_dims(jnp.array(atten_map), -1), domain)

time_axis = TimeAxis.from_medium(medium, cfl=cfg.CFL)
t_end = 0.5 * time_axis.t_end
time_axis = TimeAxis(dt=time_axis.dt, t_end=t_end)
n_steps = int(np.round(t_end / time_axis.dt))
t_arr = np.arange(n_steps) * time_axis.dt

# Calibrated amplitude (item 3): Pa per arbitrary unit, applied to the
# source signal so the output is in physical pressure units.
pa_per_unit = calibration.calibrate_arbitrary_units(cfg.F0_HZ)


def toneburst(t):
    duration = cfg.N_CYCLES / cfg.F0_HZ
    sigma = duration / 6
    window = np.exp(-(t - duration / 2) ** 2 / (2 * sigma ** 2))
    return pa_per_unit * np.sin(2 * np.pi * cfg.F0_HZ * t) * window


src_x, src_y = N // 2 - 5, 10
signal = jnp.array(toneburst(t_arr))[None, :]
sources = Sources(positions=([src_x], [src_y]), signals=signal,
                   dt=time_axis.dt, domain=domain)


@jit
def run(medium):
    return simulate_wave_propagation_attenuating(
        medium, time_axis, atten_field, cfg.F0_HZ, sources=sources)


if __name__ == "__main__":
    print(f"*** {labels.PENDING_SIGNOFF_BANNER} ***")
    print(f"Real anatomy (patient001, N={N}, genuine myocardium/blood "
          f"boundary), validated attenuation, calibrated amplitude "
          f"({pa_per_unit:.0f} Pa/unit, MI={calibration.TARGET_MI}).")

    pressure = run(medium)
    field = pressure.on_grid[..., 0]
    n_nan = int(jnp.sum(jnp.isnan(field)))
    print(f"Stability: NaN count={n_nan}, max|p|={float(jnp.max(jnp.abs(field))):.1f} Pa")

    steps = [n_steps // 4, n_steps // 2, 3 * n_steps // 4]
    maxval = float(jnp.max(jnp.abs(field))) * 0.5
    fig, axes = plt.subplots(1, len(steps), figsize=(15, 5))
    for ax, t in zip(axes, steps):
        im = ax.imshow(field[t].T, cmap="RdBu_r", vmin=-maxval, vmax=maxval,
                        interpolation="nearest", origin="lower")
        ax.contour(label_map.T, levels=[1.5, 2.5], colors="k", linewidths=0.5)
        ax.set_title(f"t={t_arr[t]:.2e}s (step {t})")
        ax.axis("off")
    fig.colorbar(im, ax=axes, shrink=0.7, label="pressure (Pa)")
    plt.tight_layout(rect=[0, 0.08, 1, 1])
    labels.add_banner(fig)
    os.makedirs("results/figures", exist_ok=True)
    out_path = "results/figures/phase4_demo_attenuating_real_anatomy.png"
    plt.savefig(out_path, dpi=150)
    print(f"Saved {out_path}")
