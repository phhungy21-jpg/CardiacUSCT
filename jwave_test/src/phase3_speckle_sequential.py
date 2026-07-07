"""Phase 3 speckle tracking, corrected — sequential (frame-to-frame) search.

Per the diagnosis in phase3_speckle_tracking.py / LOG.md run -17: a single
wide search window sized for the FULL cardiac-cycle motion excursion
(~2.53us) is geometrically incompatible with also excluding the two
strong boundary echoes, because the wall itself only spans ~3.8us and the
boundaries move WITH the wall (rigid shift). There is no window that is
both wide enough for the whole-cycle excursion and narrow enough to avoid
the boundaries.

Fix, matching how real speckle-tracking/elastography actually works:
track INCREMENTALLY. The max per-step (frame-to-frame) motion is only
~1.07us -- comfortably within the ~1.9us clearance from mid-wall to each
boundary. Search each step within a narrow window centered on the
PREVIOUS step's tracked position (not a wide window around the original
frame-0 reference), using frame-0's original template as a fixed
reference signature throughout (avoids per-step template drift while
respecting the geometric constraint).
"""

import numpy as np
from scipy.signal import correlate

import phase3_motion_recovery as pmr
import phase3_speckle_tracking as spk
import phase2_config as cfg
import phase3_config as p3cfg
import labels

from matplotlib import pyplot as plt
import os

c_ref = cfg.CHEST_WALL_PROXY.sound_speed

TEMPLATE_HALFWIDTH_S = 0.5e-6   # narrower than the original 1-pulse-duration window
SEARCH_MARGIN_PER_STEP_S = 1.2e-6  # covers max per-step shift (~1.07us) + buffer


def extract_narrow_template(reference_trace, t_center):
    idx_center = int(round(t_center / pmr.dt))
    half_w = int(round(TEMPLATE_HALFWIDTH_S / pmr.dt))
    return reference_trace[max(0, idx_center - half_w):idx_center + half_w]


def track_sequential(traces, t_ref_initial, template):
    """Returns tracked mid-wall round-trip TIMES, one per frame, tracked
    incrementally (each step searches only near the previous step's
    result, not the original fixed reference)."""
    tracked_times = [t_ref_initial]
    for i in range(1, len(traces)):
        prev_t = tracked_times[-1]
        window_mask = ((pmr.t_arr > prev_t - SEARCH_MARGIN_PER_STEP_S) &
                       (pmr.t_arr < prev_t + SEARCH_MARGIN_PER_STEP_S))
        segment = traces[i][window_mask]
        if len(segment) < len(template):
            tracked_times.append(prev_t)  # can't search, hold position
            continue
        corr = np.abs(correlate(segment, template, mode="valid"))
        idx_local = np.argmax(corr)
        t_window_start = pmr.t_arr[window_mask][0]
        t_echo = t_window_start + idx_local * pmr.dt + TEMPLATE_HALFWIDTH_S
        tracked_times.append(t_echo)
    return np.array(tracked_times)


if __name__ == "__main__":
    print(f"*** {labels.PENDING_SIGNOFF_BANNER} ***")
    print(f"*** {labels.TOY_EXACT_GT_CAPTION} ***")
    print("Sequential (frame-to-frame) speckle tracking -- geometrically "
          "correct boundary exclusion, per LOG.md run -17's diagnosis.")

    phases = np.linspace(0, 1, p3cfg.N_FRAMES)
    lv_radii_motion = [p3cfg.lv_radius_at_phase(p) for p in phases]
    ground_truth_mid_wall_radius_mm = np.array([
        pmr.vertical_dist_mm - spk.expected_mid_wall_range_mm(r, p3cfg.WALL_THICKNESS_CELLS)
        for r in lv_radii_motion
    ])
    baseline_pred = np.mean(ground_truth_mid_wall_radius_mm)
    baseline_rmse = np.sqrt(np.mean((ground_truth_mid_wall_radius_mm - baseline_pred) ** 2))

    print("=== Simulating speckle-textured frames (reusing phase3_speckle_tracking) ===")
    traces_speckle = [spk.simulate_frame_speckle(r) for r in lv_radii_motion]

    t_ref = spk.expected_mid_wall_range_mm(p3cfg.LV_RADIUS_ED_CELLS, p3cfg.WALL_THICKNESS_CELLS) * 2 * 1e-3 / c_ref
    template = extract_narrow_template(traces_speckle[0], t_ref)

    # Verify boundary exclusion explicitly before trusting anything.
    t_near = 2 * (pmr.vertical_dist_mm - (p3cfg.LV_RADIUS_ED_CELLS + p3cfg.WALL_THICKNESS_CELLS) * cfg.DX_M * 1e3) * 1e-3 / c_ref
    t_far = 2 * (pmr.vertical_dist_mm - p3cfg.LV_RADIUS_ED_CELLS * cfg.DX_M * 1e3) * 1e-3 / c_ref
    clearance_near = t_ref - TEMPLATE_HALFWIDTH_S - SEARCH_MARGIN_PER_STEP_S - t_near
    clearance_far = t_far - (t_ref + TEMPLATE_HALFWIDTH_S + SEARCH_MARGIN_PER_STEP_S)
    print(f"Boundary clearance check: near={clearance_near*1e6:.3f}us, "
          f"far={clearance_far*1e6:.3f}us (both must be >0)")
    assert clearance_near > 0 and clearance_far > 0, "Window still overlaps a boundary!"

    print("\nNoiseless per-frame check (sequential tracking):")
    tracked_times = track_sequential(traces_speckle, t_ref, template)
    tracked_radius = pmr.vertical_dist_mm - tracked_times * c_ref / 2 * 1e3
    for i in range(len(traces_speckle)):
        print(f"  frame {i}: expected={ground_truth_mid_wall_radius_mm[i]:.3f}mm "
              f"tracked={tracked_radius[i]:.3f}mm")

    reference_peak = np.max(np.abs(traces_speckle[0]))
    N_REALIZATIONS = 20
    print(f"\n(RMSE averaged over {N_REALIZATIONS} independent noise realizations)")
    print(f"\n{'noise':>8} {'sequential speckle tracking (mm)':>34}")
    seq_rmses = []
    for noise_level in p3cfg.NOISE_LEVELS:
        rmses = []
        for trial in range(N_REALIZATIONS):
            rng = np.random.default_rng(1000 * trial + 7)
            noisy_traces = [t + rng.normal(0, noise_level * reference_peak, size=t.shape)
                            for t in traces_speckle]
            tt = track_sequential(noisy_traces, t_ref, template)
            recovered = pmr.vertical_dist_mm - tt * c_ref / 2 * 1e3
            rmses.append(np.sqrt(np.mean((recovered - ground_truth_mid_wall_radius_mm) ** 2)))
        seq_rmses.append(np.mean(rmses))
        print(f"{noise_level:>8} {np.mean(rmses):>34.4f}")

    print(f"\nnaive constant-baseline RMSE: {baseline_rmse:.4f}mm")
    print("For comparison: single-window speckle (contaminated, run -17): "
          "~1.31-1.35mm plateau. Level 2/3 boundary tracking (run -16): "
          "L2~1.01mm, L3~0.86mm.")

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(p3cfg.NOISE_LEVELS, seq_rmses, "o-", color="green",
            label="sequential speckle tracking (boundary-excluded)")
    ax.axhline(baseline_rmse, color="gray", linestyle=":", label="naive constant baseline")
    ax.set_xlabel("noise level (arbitrary, fraction of a fixed reference peak)")
    ax.set_ylabel("RMSE vs. expected mid-wall position (mm)")
    ax.set_title("Sequential speckle tracking, properly boundary-excluded\n(TOY: exact prescribed ground truth)")
    ax.legend(fontsize=9)
    plt.tight_layout(rect=[0, 0.06, 1, 1])
    labels.add_banner(fig)
    os.makedirs("results/figures", exist_ok=True)
    plt.savefig("results/figures/phase3_speckle_sequential.png", dpi=150)
    print("Saved results/figures/phase3_speckle_sequential.png")
