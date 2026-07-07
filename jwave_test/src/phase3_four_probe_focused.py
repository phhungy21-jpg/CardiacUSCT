"""Phase 3 — 4 FOCUSED probes (delay-steered arrays), testing whether
proper focusing resolves the regional-hypokinesis test cleanly.

Per the unfocused-probe result (phase3_four_probe_asymmetric.py): simple
2-element pitch-catch probes did NOT cleanly isolate the hypokinetic
region (left showed near-normal motion, top showed unexpectedly reduced
motion) -- hypothesized cause: wide angular sensitivity picking up
reflections from adjacent regions, not a narrow on-axis beam. This
replaces each probe with an 8-element array, delay-focused toward the
ring center (reusing the validated focusing law from
phase2_forward_model.py / toy_2d_array_source.py), to narrow the
illuminated region and test whether that resolves the mismatch.
"""

import numpy as np
from scipy.signal import correlate

from jax import numpy as jnp
from jax import jit
from jwave.geometry import Sources
from jwave.acoustics import simulate_wave_propagation

import phase2_config as cfg
import phase3_config as p3cfg
import phase3_four_probe_tracking as fpt
import phase3_asymmetric_phantom as ap
import labels

from matplotlib import pyplot as plt
import os

c_ref = fpt.c_ref
N_ELEM = 8
ARRAY_WIDTH_CELLS = 80  # +/-4mm span, same scale as earlier sub-apertures


def build_focused_probe(probe):
    """8-element array centered on the probe position, oriented
    perpendicular to its look direction, delay-focused toward the ring
    center. Returns (Sources, receiver_position)."""
    row, col, axis = probe["row"], probe["col"], probe["axis"]
    half = ARRAY_WIDTH_CELLS // 2
    if axis == "col":  # top/bottom: elements vary along column, fixed row
        elem_cols = np.linspace(col - half, col + half, N_ELEM).astype(int)
        elem_rows = np.full(N_ELEM, row, dtype=int)
        rcv = (col + 5, row)  # (x=col-like, y=row-like)
    else:  # left/right: elements vary along row, fixed column
        elem_rows = np.linspace(row - half, row + half, N_ELEM).astype(int)
        elem_cols = np.full(N_ELEM, col, dtype=int)
        rcv = (col, row + 5)

    dist_to_focus = np.sqrt((elem_cols - fpt.center[1]) ** 2 +
                            (elem_rows - fpt.center[0]) ** 2) * fpt.dx[0]
    delays = (dist_to_focus.max() - dist_to_focus) / c_ref

    def toneburst(t, t_delay):
        tau = t - t_delay
        duration = cfg.N_CYCLES / cfg.F0_HZ
        sigma = duration / 6
        window = np.exp(-(tau - duration / 2) ** 2 / (2 * sigma ** 2))
        return np.sin(2 * np.pi * cfg.F0_HZ * tau) * window

    signals = jnp.array(np.stack([toneburst(fpt.t_arr, d) for d in delays]))
    sources = Sources(positions=(list(elem_cols), list(elem_rows)), signals=signals,
                      dt=fpt.dt, domain=fpt.domain)
    return sources, rcv


def simulate_focused_probe_frame(probe, phase, asymmetric):
    if asymmetric:
        medium, _ = ap.build_medium_asymmetric(phase, p3cfg.WALL_THICKNESS_CELLS)
    else:
        lv_r = p3cfg.lv_radius_at_phase(phase)
        medium = fpt.build_medium(lv_r, p3cfg.WALL_THICKNESS_CELLS)
    sources, rcv = build_focused_probe(probe)

    @jit
    def run(medium):
        return simulate_wave_propagation(medium, fpt.time_axis, sources=sources)

    pressure = run(medium)
    field = pressure.on_grid[..., 0]
    return np.array(field[:, rcv[0], rcv[1]])


if __name__ == "__main__":
    print(f"*** {labels.PENDING_SIGNOFF_BANNER} ***")
    print(f"*** {labels.TOY_EXACT_GT_CAPTION} ***")
    print("4 FOCUSED probes on the asymmetric (regional hypokinesis) phantom.")

    phases = np.linspace(0, 1, p3cfg.N_FRAMES)
    probe_theta = {"top": 0.0, "right": 90.0, "bottom": 180.0, "left": -90.0}
    ground_truth = {}
    for side, theta in probe_theta.items():
        lv_r = np.array([ap.local_lv_radius(theta, p) for p in phases])
        ground_truth[side] = (lv_r + p3cfg.WALL_THICKNESS_CELLS) * cfg.DX_M * 1e3

    vertical_dist_mm = fpt.PROBE_DIST_CELLS * cfg.DX_M * 1e3

    results = {}
    for side, probe in fpt.PROBES.items():
        print(f"\n=== Focused probe: {side} ===")
        traces = []
        for i, p in enumerate(phases):
            trace = simulate_focused_probe_frame(probe, p, asymmetric=True)
            traces.append(trace)
            print(f"  frame {i+1}/{len(phases)} done")

        t_ref = fpt.find_reference_echo_time(traces[0])
        tracked_times = fpt.track_sequential_range(traces, t_ref)
        tracked_range_mm = tracked_times * c_ref / 2 * 1e3
        tracked_outer_radius_mm = vertical_dist_mm - tracked_range_mm
        results[side] = tracked_outer_radius_mm

        rmse = np.sqrt(np.mean((tracked_outer_radius_mm - ground_truth[side]) ** 2))
        contraction_recovered = tracked_outer_radius_mm[0] - tracked_outer_radius_mm.min()
        contraction_expected = ground_truth[side][0] - ground_truth[side].min()
        print(f"  {side}: RMSE={rmse:.4f}mm, contraction recovered={contraction_recovered:.3f}mm "
              f"(expected {contraction_expected:.3f}mm)")

    fig, ax = plt.subplots(figsize=(9, 5.5))
    markers = {"top": "o", "bottom": "s", "left": "^", "right": "v"}
    colors = {"top": "C0", "bottom": "C1", "left": "C2", "right": "C3"}
    for side in fpt.PROBES:
        ax.plot(phases, ground_truth[side], "--", color=colors[side], alpha=0.5,
                label=f"{side} ground truth")
        ax.plot(phases, results[side], markers[side] + "-", color=colors[side],
                label=f"{side} recovered")
    ax.set_xlabel("cardiac phase (0=ED, 0.5=ES, 1=ED)")
    ax.set_ylabel("outer myocardial radius (mm)")
    ax.set_title("4 FOCUSED probes: regional hypokinesis (LEFT reduced to 30%)\n"
                "(TOY: exact prescribed ground truth, asymmetric motion)")
    ax.legend(fontsize=7, ncol=2)
    plt.tight_layout(rect=[0, 0.06, 1, 1])
    labels.add_banner(fig)
    os.makedirs("results/figures", exist_ok=True)
    plt.savefig("results/figures/phase3_four_probe_focused.png", dpi=150)
    print("\nSaved results/figures/phase3_four_probe_focused.png")
