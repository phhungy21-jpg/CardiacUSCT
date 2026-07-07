"""Phase 3 — 4-probe boundary tracking around a bounding square.

Per user request: place 4 probes at the middle of each side of the 2D
square bounding the ring phantom (top/bottom/left/right, each 12mm from
the ring center -- matching the exact distance already validated in
phase3_motion_recovery.py's single top probe). Each probe does a
per-frame, single-pulse transmit/receive (reusing the validated Level 2
reference-tracking detector from phase3_reference_tracking_recovery.py),
giving 4 independent range measurements per cardiac-cycle frame -- one
from each side of the heart model.

For this symmetric (isotropic radial-scaling) toy motion model, all 4
probes should recover the SAME motion magnitude -- a built-in consistency
check that the method generalizes correctly to different probe
orientations, not just the original top-only setup. This also lays the
groundwork for tracking asymmetric real anatomy later, where the 4
probes would genuinely disagree in an informative way.
"""

import numpy as np
from scipy.signal import correlate

from jax import numpy as jnp
from jax import jit
from jwave import FourierSeries
from jwave.geometry import Domain, Medium, TimeAxis, Sources
from jwave.acoustics import simulate_wave_propagation

import phase2_config as cfg
import phase3_config as p3cfg
import labels

from matplotlib import pyplot as plt
import os

c_ref = cfg.CHEST_WALL_PROXY.sound_speed

N = (300, 300)
dx = (cfg.DX_M, cfg.DX_M)
domain = Domain(N, dx)
center = (150, 150)
PROBE_DIST_CELLS = 120  # matches phase3_motion_recovery.py's array_y=30 distance exactly

# Each probe: (probe_row, probe_col, look_axis, look_sign)
# look_axis: 'row' or 'col' -- which coordinate the probe's pitch-catch
# offset and pointing direction vary along.
PROBES = {
    "top":    dict(row=center[0] - PROBE_DIST_CELLS, col=center[1], axis="col", sign=+1),
    "bottom": dict(row=center[0] + PROBE_DIST_CELLS, col=center[1], axis="col", sign=-1),
    "left":   dict(row=center[0], col=center[1] - PROBE_DIST_CELLS, axis="row", sign=+1),
    "right":  dict(row=center[0], col=center[1] + PROBE_DIST_CELLS, axis="row", sign=-1),
}


def build_medium(lv_radius_cells, wall_thickness_cells):
    yy, xx = np.mgrid[0:N[0], 0:N[1]]
    dist = np.sqrt((xx - center[1]) ** 2 + (yy - center[0]) ** 2)
    label_map = np.zeros(N, dtype=int)
    label_map[dist < lv_radius_cells + wall_thickness_cells] = 2
    label_map[dist < lv_radius_cells] = 3
    sound_speed_map = np.zeros(N, dtype=np.float32)
    density_map = np.zeros(N, dtype=np.float32)
    for label, tissue in cfg.ACDC_LABEL_TO_TISSUE.items():
        m = label_map == label
        sound_speed_map[m] = tissue.sound_speed
        density_map[m] = tissue.density
    ssm = jnp.expand_dims(jnp.array(sound_speed_map), -1)
    dm = jnp.expand_dims(jnp.array(density_map), -1)
    return Medium(domain=domain, sound_speed=FourierSeries(ssm, domain),
                  density=FourierSeries(dm, domain))


_dummy_medium = build_medium(p3cfg.LV_RADIUS_ED_CELLS, p3cfg.WALL_THICKNESS_CELLS)
_base_time_axis = TimeAxis.from_medium(_dummy_medium, cfl=cfg.CFL)
dt = _base_time_axis.dt
t_end = 0.35 * _base_time_axis.t_end  # matches phase3_motion_recovery.py's window
time_axis = TimeAxis(dt=dt, t_end=t_end)
n_steps = int(time_axis.Nt)
t_arr = np.arange(n_steps) * dt

DIRECT_EXCLUDE_S = 3.0e-6  # matches phase3_motion_recovery.py, validated there
SEARCH_MARGIN_S = 3.0e-6  # matches phase3_reference_tracking_recovery.py


def toneburst(t):
    duration = cfg.N_CYCLES / cfg.F0_HZ
    sigma = duration / 6
    window = np.exp(-(t - duration / 2) ** 2 / (2 * sigma ** 2))
    return np.sin(2 * np.pi * cfg.F0_HZ * t) * window


_signal_template = jnp.array(toneburst(t_arr))[None, :]


def probe_source_and_receiver(probe):
    """Small pitch-catch pair centered on the probe position, offset
    along the perpendicular (tangential) axis -- same 10-cell spacing
    validated in phase3_motion_recovery.py (src_x=center-5, rcv_x=center+5)."""
    row, col, axis = probe["row"], probe["col"], probe["axis"]
    if axis == "col":  # top/bottom: probe varies along column (row fixed)
        src = (col - 5, row)   # (x=col-like, y=row-like), established field/Sources convention
        rcv = (col + 5, row)
    else:  # left/right: probe varies along row (col fixed)
        src = (col, row - 5)
        rcv = (col, row + 5)
    return src, rcv


def build_sources(probe):
    src, rcv = probe_source_and_receiver(probe)
    sources = Sources(positions=([src[0]], [src[1]]), signals=_signal_template,
                      dt=dt, domain=domain)
    return sources, src, rcv


def simulate_probe_frame(probe, lv_radius_cells):
    medium = build_medium(lv_radius_cells, p3cfg.WALL_THICKNESS_CELLS)
    sources, src, rcv = build_sources(probe)

    @jit
    def run(medium):
        return simulate_wave_propagation(medium, time_axis, sources=sources)

    pressure = run(medium)
    field = pressure.on_grid[..., 0]
    return np.array(field[:, rcv[0], rcv[1]])  # established field[:, x, y] convention


_duration = cfg.N_CYCLES / cfg.F0_HZ
_template_t = np.arange(0, _duration, dt)
_template = toneburst(_template_t)


def find_reference_echo_time(reference_trace):
    mask = t_arr > DIRECT_EXCLUDE_S
    segment = reference_trace[mask]
    corr = np.abs(correlate(segment, _template, mode="valid"))
    threshold = 0.3 * corr.max()
    idx_local = np.argmax(corr > threshold)
    return t_arr[mask][idx_local] + _duration / 2


def track_sequential_range(traces, t_ref_initial):
    tracked = [t_ref_initial]
    for i in range(1, len(traces)):
        prev_t = tracked[-1]
        mask = ((t_arr > max(prev_t - SEARCH_MARGIN_S, DIRECT_EXCLUDE_S)) &
                (t_arr < prev_t + SEARCH_MARGIN_S))
        segment = traces[i][mask]
        if len(segment) < len(_template):
            tracked.append(prev_t)
            continue
        corr = np.abs(correlate(segment, _template, mode="valid"))
        idx_local = np.argmax(corr)
        t_window_start = t_arr[mask][0]
        t_echo = t_window_start + idx_local * dt + _duration / 2
        tracked.append(t_echo)
    return np.array(tracked)


if __name__ == "__main__":
    print(f"*** {labels.PENDING_SIGNOFF_BANNER} ***")
    print(f"*** {labels.TOY_EXACT_GT_CAPTION} ***")
    print("4-probe boundary tracking: top/bottom/left/right, each 12mm "
          "from ring center, per-frame single-pulse transmit/receive.")

    phases = np.linspace(0, 1, p3cfg.N_FRAMES)
    lv_radii = [p3cfg.lv_radius_at_phase(p) for p in phases]
    ground_truth_outer_radius_mm = np.array([
        (r + p3cfg.WALL_THICKNESS_CELLS) * cfg.DX_M * 1e3 for r in lv_radii
    ])
    vertical_dist_mm = PROBE_DIST_CELLS * cfg.DX_M * 1e3  # 12mm, same for all 4 probes

    results = {}
    for side, probe in PROBES.items():
        print(f"\n=== Probe: {side} ===")
        traces = []
        for i, r in enumerate(lv_radii):
            trace = simulate_probe_frame(probe, r)
            traces.append(trace)
            print(f"  frame {i+1}/{len(lv_radii)} done")

        t_ref = find_reference_echo_time(traces[0])
        tracked_times = track_sequential_range(traces, t_ref)
        tracked_range_mm = tracked_times * c_ref / 2 * 1e3
        tracked_outer_radius_mm = vertical_dist_mm - tracked_range_mm
        results[side] = tracked_outer_radius_mm

        rmse = np.sqrt(np.mean((tracked_outer_radius_mm - ground_truth_outer_radius_mm) ** 2))
        print(f"  {side}: RMSE={rmse:.4f}mm")

    fig, ax = plt.subplots(figsize=(9, 5.5))
    ax.plot(phases, ground_truth_outer_radius_mm, "k--", label="ground truth (all sides, by symmetry)", linewidth=2)
    markers = {"top": "o", "bottom": "s", "left": "^", "right": "v"}
    for side, tracked in results.items():
        rmse = np.sqrt(np.mean((tracked - ground_truth_outer_radius_mm) ** 2))
        ax.plot(phases, tracked, markers[side] + "-", label=f"{side} (RMSE={rmse:.3f}mm)")
    ax.set_xlabel("cardiac phase (0=ED, 0.5=ES, 1=ED)")
    ax.set_ylabel("outer myocardial radius (mm)")
    ax.set_title("4-probe boundary tracking around the heart model\n(TOY: exact prescribed ground truth, isotropic motion)")
    ax.legend(fontsize=8)
    plt.tight_layout(rect=[0, 0.06, 1, 1])
    labels.add_banner(fig)
    os.makedirs("results/figures", exist_ok=True)
    plt.savefig("results/figures/phase3_four_probe_tracking.png", dpi=150)
    print("\nSaved results/figures/phase3_four_probe_tracking.png")
