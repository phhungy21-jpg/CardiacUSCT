"""Phase 3 — 4-probe tracking on the asymmetric (regional hypokinesis) phantom.

Same 4 probes and detection method as phase3_four_probe_tracking.py
(validated on the isotropic phantom, run -23), applied to the
asymmetric phantom from phase3_asymmetric_phantom.py. Prediction: top,
right, bottom should track near-normal contraction (regional_factor~1.0
there); left should show markedly REDUCED contraction (regional_factor
=0.3 at its center) -- a real, physically distinct prediction, not just
a symmetry check.
"""

import numpy as np

from jax import jit
from jwave.acoustics import simulate_wave_propagation

import phase2_config as cfg
import phase3_config as p3cfg
import phase3_four_probe_tracking as fpt
import phase3_asymmetric_phantom as ap
import labels

from matplotlib import pyplot as plt
import os

c_ref = fpt.c_ref


def simulate_probe_frame_asymmetric(probe, phase):
    medium, label_map = ap.build_medium_asymmetric(phase, p3cfg.WALL_THICKNESS_CELLS)
    sources, src, rcv = fpt.build_sources(probe)

    @jit
    def run(medium):
        return simulate_wave_propagation(medium, fpt.time_axis, sources=sources)

    pressure = run(medium)
    field = pressure.on_grid[..., 0]
    return np.array(field[:, rcv[0], rcv[1]])


if __name__ == "__main__":
    print(f"*** {labels.PENDING_SIGNOFF_BANNER} ***")
    print(f"*** {labels.TOY_EXACT_GT_CAPTION} ***")
    print("4-probe tracking on ASYMMETRIC (regional hypokinesis) phantom.")
    print(f"Hypokinetic region centered at LEFT probe (theta={ap.HYPOKINETIC_CENTER_DEG}deg), "
          f"factor={ap.HYPOKINETIC_FACTOR}, half-width={ap.HYPOKINETIC_HALFWIDTH_DEG}deg.")

    phases = np.linspace(0, 1, p3cfg.N_FRAMES)

    # Ground truth per probe: each probe's own local outer radius, using
    # its own theta (0/90/180/-90 for top/right/bottom/left).
    probe_theta = {"top": 0.0, "right": 90.0, "bottom": 180.0, "left": -90.0}
    ground_truth = {}
    for side, theta in probe_theta.items():
        lv_r = np.array([ap.local_lv_radius(theta, p) for p in phases])
        ground_truth[side] = (lv_r + p3cfg.WALL_THICKNESS_CELLS) * cfg.DX_M * 1e3

    vertical_dist_mm = fpt.PROBE_DIST_CELLS * cfg.DX_M * 1e3

    results = {}
    for side, probe in fpt.PROBES.items():
        print(f"\n=== Probe: {side} ===")
        traces = []
        for i, p in enumerate(phases):
            trace = simulate_probe_frame_asymmetric(probe, p)
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
    ax.set_title("4-probe tracking: regional hypokinesis (LEFT reduced to 30% contraction)\n"
                "(TOY: exact prescribed ground truth, asymmetric motion)")
    ax.legend(fontsize=7, ncol=2)
    plt.tight_layout(rect=[0, 0.06, 1, 1])
    labels.add_banner(fig)
    os.makedirs("results/figures", exist_ok=True)
    plt.savefig("results/figures/phase3_four_probe_asymmetric.png", dpi=150)
    print("\nSaved results/figures/phase3_four_probe_asymmetric.png")
