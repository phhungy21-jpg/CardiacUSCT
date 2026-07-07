"""Phase 3 pivot, continued — multi-channel (beamformed) speckle tracking.

Per the single-A-line speckle test (phase3_speckle_tracking.py): tracking
a mid-wall material point via a single receiver underperformed boundary
tracking, but for an EXPECTED reason -- the mid-wall speckle signal is
~670x weaker than the boundary echo (measured), and a single channel
gives no correlation-processing gain to compensate. Real speckle tracking
works by aggregating correlation over many channels/scatterers, not by
having a strong individual echo.

This combines the speckle-textured medium (phase3_speckle_tracking.py)
with the 7-element receive aperture (phase3_beamforming_and_multiangle.py),
but delay-and-sums with a FIXED FOCUS AT THE MID-WALL DEPTH (not the
boundary depth) -- the correct comparison to fairly test whether
multi-channel correlation gain can recover the weak speckle signal's
trackability.
"""

import numpy as np
from scipy.signal import correlate

import phase3_motion_recovery as pmr
import phase3_speckle_tracking as spk
import phase3_beamforming_and_multiangle as bfma
import phase2_config as cfg
import phase3_config as p3cfg
import labels

from matplotlib import pyplot as plt
import os

c_ref = cfg.CHEST_WALL_PROXY.sound_speed


def simulate_frame_speckle_multi_rx(lv_radius_cells):
    medium = spk.build_medium_with_speckle(lv_radius_cells, p3cfg.WALL_THICKNESS_CELLS)
    pressure = pmr.run(medium)
    field = pressure.on_grid[..., 0]
    return {off: np.array(field[:, pmr.center[1] + off, pmr.array_y])
            for off in bfma.RX_OFFSETS_CELLS}


def beamform_at_depth(traces_dict, depth_m, noise_level, reference_peak, rng):
    """Same delay-and-sum as phase3_beamforming_and_multiangle.beamform,
    but noise is scaled to a FIXED reference_peak (not each channel's own
    peak) -- the fairness fix identified in the single-channel test."""
    dt = pmr.dt
    aligned = []
    for off_cells, trace in traces_dict.items():
        noisy = trace + rng.normal(0, noise_level * reference_peak, size=trace.shape)
        offset_m = off_cells * cfg.DX_M
        extra_path_m = np.sqrt(offset_m ** 2 + depth_m ** 2) - depth_m
        extra_time_s = extra_path_m / c_ref
        shift_samples = int(round(extra_time_s / dt))
        shifted = np.roll(noisy, -shift_samples)
        aligned.append(shifted)
    return np.mean(aligned, axis=0)


if __name__ == "__main__":
    print(f"*** {labels.PENDING_SIGNOFF_BANNER} ***")
    print(f"*** {labels.TOY_EXACT_GT_CAPTION} ***")
    print("Multi-channel (beamformed) speckle tracking: correlation gain "
          "across 7 receive channels, focused at mid-wall depth.")

    phases = np.linspace(0, 1, p3cfg.N_FRAMES)
    lv_radii_motion = [p3cfg.lv_radius_at_phase(p) for p in phases]
    ground_truth_mid_wall_radius_mm = np.array([
        pmr.vertical_dist_mm - spk.expected_mid_wall_range_mm(r, p3cfg.WALL_THICKNESS_CELLS)
        for r in lv_radii_motion
    ])
    baseline_pred = np.mean(ground_truth_mid_wall_radius_mm)
    baseline_rmse = np.sqrt(np.mean((ground_truth_mid_wall_radius_mm - baseline_pred) ** 2))

    print("=== Simulating speckle-textured, multi-receiver frames ===")
    multi_rx_speckle = [simulate_frame_speckle_multi_rx(r) for r in lv_radii_motion]

    t_mid_ref = spk.expected_mid_wall_range_mm(p3cfg.LV_RADIUS_ED_CELLS, p3cfg.WALL_THICKNESS_CELLS) * 2 * 1e-3 / c_ref
    depth_m = t_mid_ref * c_ref / 2
    reference_peak = np.max(np.abs(multi_rx_speckle[0][0]))  # on-axis channel, frame 0

    ref_window = spk.extract_reference_window(multi_rx_speckle[0][0], t_mid_ref)
    print(f"Reference mid-wall time: {t_mid_ref:.3e}s, depth={depth_m*1e3:.3f}mm, "
          f"window length: {len(ref_window)} samples")

    # Noiseless sanity check first.
    print("\nNoiseless per-frame check (beamformed speckle tracking):")
    rng_dummy = np.random.default_rng(0)
    for i, traces_dict in enumerate(multi_rx_speckle):
        bf_trace = beamform_at_depth(traces_dict, depth_m, 0.0, reference_peak, rng_dummy)
        r = spk.track_mid_wall_range_mm(bf_trace, ref_window, t_mid_ref, c_ref)
        tracked_radius = pmr.vertical_dist_mm - r
        print(f"  frame {i}: expected={ground_truth_mid_wall_radius_mm[i]:.3f}mm "
              f"tracked={tracked_radius:.3f}mm")

    N_REALIZATIONS = 20
    print(f"\n(RMSE averaged over {N_REALIZATIONS} independent noise realizations)")
    print(f"\n{'noise':>8} {'single-channel speckle (mm)':>28} {'7-ch beamformed speckle (mm)':>30}")
    single_rmses, bf_rmses = [], []
    for noise_level in p3cfg.NOISE_LEVELS:
        rmses_single, rmses_bf = [], []
        for trial in range(N_REALIZATIONS):
            rng_single = np.random.default_rng(1000 * trial + 7)
            rng_bf = np.random.default_rng(1000 * trial + 7)

            recovered_single = []
            for traces_dict in multi_rx_speckle:
                trace = traces_dict[0]  # on-axis channel only
                noisy = trace + rng_single.normal(0, noise_level * reference_peak, size=trace.shape)
                r = spk.track_mid_wall_range_mm(noisy, ref_window, t_mid_ref, c_ref)
                recovered_single.append(pmr.vertical_dist_mm - r)
            recovered_single = np.array(recovered_single)

            recovered_bf = []
            for traces_dict in multi_rx_speckle:
                bf_trace = beamform_at_depth(traces_dict, depth_m, noise_level, reference_peak, rng_bf)
                r = spk.track_mid_wall_range_mm(bf_trace, ref_window, t_mid_ref, c_ref)
                recovered_bf.append(pmr.vertical_dist_mm - r)
            recovered_bf = np.array(recovered_bf)

            rmses_single.append(np.sqrt(np.mean((recovered_single - ground_truth_mid_wall_radius_mm) ** 2)))
            rmses_bf.append(np.sqrt(np.mean((recovered_bf - ground_truth_mid_wall_radius_mm) ** 2)))

        single_rmses.append(np.mean(rmses_single))
        bf_rmses.append(np.mean(rmses_bf))
        print(f"{noise_level:>8} {np.mean(rmses_single):>28.4f} {np.mean(rmses_bf):>30.4f}")

    print(f"\nnaive constant-baseline RMSE: {baseline_rmse:.4f}mm")
    print("For comparison, Level 2/3 boundary tracking (run -16): "
          "L2=1.01/1.01/1.01mm, L3=0.87/0.86/0.86mm at noise 0.02/0.05/0.10")

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(p3cfg.NOISE_LEVELS, single_rmses, "o-", color="purple", label="single-channel speckle")
    ax.plot(p3cfg.NOISE_LEVELS, bf_rmses, "s-", color="darkorange", label="7-channel beamformed speckle")
    ax.axhline(baseline_rmse, color="gray", linestyle=":", label="naive constant baseline")
    ax.set_xlabel("noise level (arbitrary, fraction of a fixed reference peak)")
    ax.set_ylabel("RMSE vs. expected mid-wall position (mm)")
    ax.set_title("Does multi-channel correlation gain rescue speckle tracking?\n(TOY: exact prescribed ground truth)")
    ax.legend(fontsize=9)
    plt.tight_layout(rect=[0, 0.06, 1, 1])
    labels.add_banner(fig)
    os.makedirs("results/figures", exist_ok=True)
    plt.savefig("results/figures/phase3_speckle_beamformed.png", dpi=150)
    print("Saved results/figures/phase3_speckle_beamformed.png")
