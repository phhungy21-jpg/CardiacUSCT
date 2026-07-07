"""Phase 3.2 upgrade — Level 3 (delay-and-sum receive beamforming) and
Level 4 (multi-angle fusion), tested together per user request (test both
now rather than defer to a later calibration phase).

Level 3: instead of one receive element, use a modest 7-element receive
aperture (+/-1.5mm) and delay-and-sum the channels (fixed-focus, using the
Level 2 reference-derived depth) before running Level 2's detector on the
beamformed trace. Coherent summation of N independent noise realizations
with an aligned signal gives a classical sqrt(N) SNR improvement.

Level 4: a second, laterally-offset pitch-catch pair (a mildly oblique
path to the same boundary, not a dramatically different angle) gives an
independent measurement chain (independent noise realization, independent
detection). Fusing the two pairs' Level-2 range estimates by simple
averaging tests whether redundancy helps. NOTE: this is a SIMPLIFIED
multi-angle test -- it reuses the on-axis vertical_dist-minus-range
approximation for the offset pair too, which is only approximately valid
at this lateral offset; a full vector/off-axis geometric correction is a
bigger task, deferred (see PIPELINE_STATUS_AND_ROADMAP.md).
"""

import numpy as np
from scipy.signal import correlate

import phase3_motion_recovery as pmr
import phase3_reference_tracking_recovery as rtr
import phase2_config as cfg
import phase3_config as p3cfg
import labels

from matplotlib import pyplot as plt
import os

c_ref = cfg.CHEST_WALL_PROXY.sound_speed

# --- Level 3: beamforming aperture -----------------------------------
RX_OFFSETS_CELLS = [-15, -10, -5, 0, 5, 10, 15]  # +/-1.5mm, 7 elements


def simulate_frame_multi_rx(lv_radius_cells):
    """Reuses pmr's medium/source setup; extracts traces at multiple
    receiver x-positions from the SAME simulated field (no extra
    simulation cost -- jWave already returns the full spatial field)."""
    medium = pmr.build_medium(lv_radius_cells, p3cfg.WALL_THICKNESS_CELLS)
    pressure = pmr.run(medium)
    field = pressure.on_grid[..., 0]
    traces = {off: np.array(field[:, pmr.center[1] + off, pmr.array_y])
              for off in RX_OFFSETS_CELLS}
    return traces


def beamform(traces_dict, depth_m, noise_level, rng):
    """Delay-and-sum: shift each channel by its geometric extra-path delay
    (fixed-focus at depth_m, the Level-2 reference depth), add independent
    noise per channel, then average."""
    dt = pmr.dt
    aligned = []
    for off_cells, trace in traces_dict.items():
        peak = np.max(np.abs(trace))
        noisy = trace + rng.normal(0, noise_level * peak, size=trace.shape)
        offset_m = off_cells * cfg.DX_M
        extra_path_m = np.sqrt(offset_m ** 2 + depth_m ** 2) - depth_m
        extra_time_s = extra_path_m / c_ref  # one-way; receive-side delay only
        shift_samples = int(round(extra_time_s / dt))
        shifted = np.roll(noisy, -shift_samples)  # bring the delayed echo forward
        aligned.append(shifted)
    return np.mean(aligned, axis=0)


def recover_outer_radius_mm_beamformed(traces_dict, t_ref, depth_m, noise_level, rng):
    bf_trace = beamform(traces_dict, depth_m, noise_level, rng)
    return rtr.recover_outer_radius_mm_reference_tracking(bf_trace, t_ref, c_ref)


# --- Level 4: second, laterally-offset pitch-catch pair -----------------
PAIR_B_OFFSET_CELLS = 40  # ~4mm lateral shift from the original pair


def simulate_frame_pair_b(lv_radius_cells):
    medium = pmr.build_medium(lv_radius_cells, p3cfg.WALL_THICKNESS_CELLS)
    pressure = pmr.run(medium)  # NOTE: reuses pmr's fixed on-axis SOURCE;
                                 # only the receiver differs for pair B here,
                                 # a simplification -- see module docstring.
    field = pressure.on_grid[..., 0]
    rcv_x_b = pmr.rcv_x - PAIR_B_OFFSET_CELLS
    return np.array(field[:, rcv_x_b, pmr.array_y])


if __name__ == "__main__":
    print(f"*** {labels.PENDING_SIGNOFF_BANNER} ***")
    print(f"*** {labels.TOY_EXACT_GT_CAPTION} ***")
    print("Testing Level 3 (beamforming) and Level 4 (multi-angle) together.")

    phases = np.linspace(0, 1, p3cfg.N_FRAMES)
    lv_radii_motion = [p3cfg.lv_radius_at_phase(p) for p in phases]
    ground_truth_outer_radius_mm = np.array([
        (r + p3cfg.WALL_THICKNESS_CELLS) * cfg.DX_M * 1e3 for r in lv_radii_motion
    ])
    baseline_pred = np.mean(ground_truth_outer_radius_mm)
    baseline_rmse = np.sqrt(np.mean((ground_truth_outer_radius_mm - baseline_pred) ** 2))

    print("=== Simulating multi-receiver traces (Level 3) ===")
    multi_rx_traces = [simulate_frame_multi_rx(r) for r in lv_radii_motion]

    print("=== Simulating pair-B traces (Level 4) ===")
    pair_b_traces = [simulate_frame_pair_b(r) for r in lv_radii_motion]

    # Reference calibration (noiseless frame 0 / ED) for both.
    t_ref_a = rtr.find_reference_echo_time(multi_rx_traces[0][0], c_ref)
    depth_m = (t_ref_a * c_ref / 2)
    t_ref_b = rtr.find_reference_echo_time(pair_b_traces[0], c_ref)
    print(f"Pair A reference range: {t_ref_a*c_ref/2*1e3:.3f}mm, "
          f"Pair B reference range: {t_ref_b*c_ref/2*1e3:.3f}mm "
          f"(should be similar if the simplification holds)")

    # Sanity check pair B's noiseless recovery against ground truth BEFORE
    # trusting the fusion result -- per this project's established discipline.
    print("\nPair-B noiseless sanity check (per frame):")
    for i, trace in enumerate(pair_b_traces):
        r = rtr.recover_outer_radius_mm_reference_tracking(trace, t_ref_b, c_ref)
        print(f"  frame {i}: gt={ground_truth_outer_radius_mm[i]:.3f}mm "
              f"pair_B_recovered={r:.3f}mm")

    # Average over multiple independent noise realizations per condition --
    # with only 8 frames, a single noise draw's RMSE has high sampling
    # variance; this makes the Level 2/3/4 comparison statistically
    # meaningful rather than an artifact of one particular noise draw.
    N_REALIZATIONS = 20
    traces_center = [t[0] for t in multi_rx_traces]  # offset 0 == original on-axis receiver

    print(f"\n(each RMSE averaged over {N_REALIZATIONS} independent noise realizations)")
    print(f"\n{'noise':>8} {'L2 single-rx (mm)':>18} {'L3 beamformed (mm)':>20} {'L4 fused A+B (mm)':>18}")
    l2_rmses, l3_rmses, l4_rmses = [], [], []
    for noise_level in p3cfg.NOISE_LEVELS:
        rmses_l2, rmses_l3, rmses_l4 = [], [], []
        for trial in range(N_REALIZATIONS):
            rng_l2 = np.random.default_rng(1000 * trial)
            rng_l3 = np.random.default_rng(1000 * trial)
            rng_l4a = np.random.default_rng(1000 * trial)
            rng_l4b = np.random.default_rng(1000 * trial + 1)

            rec_l2 = []
            for trace in traces_center:
                peak = np.max(np.abs(trace))
                noisy = trace + rng_l2.normal(0, noise_level * peak, size=trace.shape)
                rec_l2.append(rtr.recover_outer_radius_mm_reference_tracking(noisy, t_ref_a, c_ref))
            rec_l2 = np.array(rec_l2)

            rec_l3 = np.array([
                recover_outer_radius_mm_beamformed(traces_dict, t_ref_a, depth_m, noise_level, rng_l3)
                for traces_dict in multi_rx_traces
            ])

            rec_l4a = []
            for trace in traces_center:
                peak = np.max(np.abs(trace))
                noisy = trace + rng_l4a.normal(0, noise_level * peak, size=trace.shape)
                rec_l4a.append(rtr.recover_outer_radius_mm_reference_tracking(noisy, t_ref_a, c_ref))
            rec_l4b = []
            for trace in pair_b_traces:
                peak = np.max(np.abs(trace))
                noisy = trace + rng_l4b.normal(0, noise_level * peak, size=trace.shape)
                rec_l4b.append(rtr.recover_outer_radius_mm_reference_tracking(noisy, t_ref_b, c_ref))
            rec_l4 = (np.array(rec_l4a) + np.array(rec_l4b)) / 2

            rmses_l2.append(np.sqrt(np.mean((rec_l2 - ground_truth_outer_radius_mm) ** 2)))
            rmses_l3.append(np.sqrt(np.mean((rec_l3 - ground_truth_outer_radius_mm) ** 2)))
            rmses_l4.append(np.sqrt(np.mean((rec_l4 - ground_truth_outer_radius_mm) ** 2)))

        rmse_l2, rmse_l3, rmse_l4 = np.mean(rmses_l2), np.mean(rmses_l3), np.mean(rmses_l4)
        l2_rmses.append(rmse_l2); l3_rmses.append(rmse_l3); l4_rmses.append(rmse_l4)
        print(f"{noise_level:>8} {rmse_l2:>18.4f} {rmse_l3:>20.4f} {rmse_l4:>18.4f}")

    print(f"\nnaive constant-baseline RMSE: {baseline_rmse:.4f}mm")

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(p3cfg.NOISE_LEVELS, l2_rmses, "^-", label="Level 2: single-rx ref-tracking")
    ax.plot(p3cfg.NOISE_LEVELS, l3_rmses, "d-", label="Level 3: 7-element beamformed")
    ax.plot(p3cfg.NOISE_LEVELS, l4_rmses, "v-", label="Level 4: 2-pair fused (simplified)")
    ax.axhline(baseline_rmse, color="gray", linestyle=":", label="naive constant baseline")
    ax.set_xlabel("noise level (arbitrary, fraction of trace peak)")
    ax.set_ylabel("RMSE vs. prescribed motion (mm)")
    ax.set_title("Level 3 vs Level 4 vs Level 2\n(TOY: exact prescribed ground truth)")
    ax.legend(fontsize=9)
    plt.tight_layout(rect=[0, 0.06, 1, 1])
    labels.add_banner(fig)
    os.makedirs("results/figures", exist_ok=True)
    plt.savefig("results/figures/phase3_level3_level4_comparison.png", dpi=150)
    print("Saved results/figures/phase3_level3_level4_comparison.png")
