"""Phase 3 — genuine multi-angle vector motion reconstruction.

Per the user's request: "keep one array, but simulate multi-angle
transmissions and multi-channel receive, then fuse into vector motion
estimates." This is a proper test of that idea, distinct from the
earlier Level 4 (phase3_beamforming_and_multiangle.py), which only
averaged two SIMILAR scalar range measurements of (approximately)
different points -- not a real vector reconstruction.

Setup: TWO independently delay-focused sub-apertures, both steered to
converge on the SAME off-axis target point (outer myocardial boundary,
30 degrees off boresight). Each sub-aperture measures range-change along
ITS OWN line-of-sight to that point (a scalar, like every method in this
project so far). With two known, non-parallel look directions u_A, u_B,
the two scalar range-change measurements can be combined by solving a
2x2 linear system for the full 2D displacement vector -- genuine
triangulation, testable against this toy's KNOWN radial displacement
(known magnitude and direction at any point on the ring, since the
motion model is pure radial scaling).

Uses the CLEAN (non-speckle) boundary reflector -- this tests whether
multi-angle triangulation recovers vector motion, a separate question
from the still-open speckle-density question (LOG.md run -17/-18).
"""

import numpy as np
from scipy.signal import correlate

import phase3_motion_recovery as pmr
import phase2_config as cfg
import phase3_config as p3cfg
import labels

from jax import numpy as jnp
from jax import jit
from jwave.geometry import Sources, TimeAxis
from jwave.acoustics import simulate_wave_propagation

# BUG CAUGHT DURING VALIDATION (see chat log): sub-aperture B's path is
# ~9.5mm longer than the on-axis case, giving a round-trip time (~12.6us)
# that exceeds pmr.time_axis's t_end (~9.4us, sized only for the shorter
# on-axis boundary-tracking case) -- caused an empty-array crash when
# searching for the echo. Fixed with a longer, LOCAL time axis (same dt,
# more steps) instead of reusing pmr.time_axis directly.
_dt = pmr.dt
_n_steps_local = int(round(20e-6 / _dt))  # 20us, comfortable margin over ~12.6us
time_axis_local = TimeAxis(dt=_dt, t_end=_n_steps_local * _dt)
t_arr_local = np.arange(_n_steps_local) * _dt

from matplotlib import pyplot as plt
import os

c_ref = cfg.CHEST_WALL_PROXY.sound_speed

THETA_DEG = 30.0
THETA = np.radians(THETA_DEG)

# Sub-aperture geometry: N elements each, centered at a chosen column,
# spanning a modest width -- reusing the delay-focusing law already
# validated in phase2_forward_model.py / toy_2d_array_source.py.
N_ELEM_PER_SUBAPERTURE = 8
SUBAPERTURE_WIDTH_CELLS = 80  # +/-4mm span


def target_position_at_radius(r_boundary_cells):
    row = pmr.center[0] - r_boundary_cells * np.cos(THETA)
    col = pmr.center[1] + r_boundary_cells * np.sin(THETA)
    return row, col


def subaperture_elements(center_col):
    half = SUBAPERTURE_WIDTH_CELLS // 2
    xs = np.linspace(center_col - half, center_col + half, N_ELEM_PER_SUBAPERTURE).astype(int)
    ys = np.full(N_ELEM_PER_SUBAPERTURE, pmr.array_y, dtype=int)
    return xs, ys


def build_focused_sources(center_col, target_row_ref, target_col_ref):
    """Delay-focus a sub-aperture on the REFERENCE (ED) target position --
    fixed focus, same simplification as phase2_forward_model.py."""
    xs, ys = subaperture_elements(center_col)
    dist_to_focus = np.sqrt((xs - target_col_ref) ** 2 + (ys - target_row_ref) ** 2) * cfg.DX_M
    delays = (dist_to_focus.max() - dist_to_focus) / c_ref

    def toneburst(t, t_delay):
        tau = t - t_delay
        duration = cfg.N_CYCLES / cfg.F0_HZ
        sigma = duration / 6
        window = np.exp(-(tau - duration / 2) ** 2 / (2 * sigma ** 2))
        return np.sin(2 * np.pi * cfg.F0_HZ * tau) * window

    signals = jnp.array(np.stack([toneburst(t_arr_local, d) for d in delays]))
    return Sources(positions=(list(xs), list(ys)), signals=signals, dt=_dt, domain=pmr.domain), xs, ys


def simulate_subaperture(lv_radius_cells, sources):
    medium = pmr.build_medium(lv_radius_cells, p3cfg.WALL_THICKNESS_CELLS)

    @jit
    def run(medium):
        return simulate_wave_propagation(medium, time_axis_local, sources=sources)

    pressure = run(medium)
    field = pressure.on_grid[..., 0]
    return field


def expected_round_trip_time(center_col, rcv_col, target_row_ref, target_col_ref):
    """Correct physics for a delay-focused sub-aperture's expected
    round-trip time to the target.

    BUG CAUGHT DURING VALIDATION (see LOG.md): the original approach used
    a naive symmetric approximation (2*dist(receiver,target)/c_ref),
    implicitly assuming transmit and receive are co-located. This is only
    a good approximation when the sub-aperture is roughly symmetric
    around the target (true for sub-aperture A here, close enough for B,
    but badly wrong for an asymmetric aperture like C -- a 30-cell offset
    with an 80-cell-wide 8-element span puts the farthest element ~70
    columns from the target, giving a real transmit-focus convergence
    time very different from the naive guess).

    Correct physics: because delay-focusing sets delay_i =
    (max_dist-dist_i)/c_ref, EVERY element's wavefront arrives at the
    target simultaneously at t = max(dist_i)/c_ref (not any single
    element's own distance/c_ref). The receive path is then a separate
    one-way distance from target to the (single) receiver.
    """
    xs, ys = subaperture_elements(center_col)
    dist_tx = np.sqrt((xs - target_col_ref) ** 2 + (ys - target_row_ref) ** 2) * cfg.DX_M
    tx_time = dist_tx.max() / c_ref
    dist_rx = np.sqrt((rcv_col - target_col_ref) ** 2 +
                      (pmr.array_y - target_row_ref) ** 2) * cfg.DX_M
    rx_time = dist_rx / c_ref
    return tx_time + rx_time


def look_direction(subaperture_center_col, target_row_ref, target_col_ref):
    """Unit vector (row,col) from the sub-aperture's phase center (at the
    array, y=array_y) to the reference target position."""
    d_row = target_row_ref - pmr.array_y
    d_col = target_col_ref - subaperture_center_col
    mag = np.sqrt(d_row ** 2 + d_col ** 2)
    return np.array([d_row, d_col]) / mag


def track_sequential_range(traces_over_frames, t_ref, template_halfwidth_s, search_margin_s):
    """Same sequential (frame-to-frame) approach validated in
    phase3_speckle_sequential.py -- avoids the wide-window/boundary-
    contamination trap for a target that moves over the whole cycle.
    Uses t_arr_local (this script's longer time axis), not pmr.t_arr."""
    template_idx = int(round(t_ref / _dt))
    half_w = int(round(template_halfwidth_s / _dt))
    template = traces_over_frames[0][max(0, template_idx - half_w):template_idx + half_w]

    tracked_times = [t_ref]
    for i in range(1, len(traces_over_frames)):
        prev_t = tracked_times[-1]
        mask = ((t_arr_local > max(prev_t - search_margin_s, pmr.DIRECT_EXCLUDE_S)) &
                (t_arr_local < prev_t + search_margin_s))
        segment = traces_over_frames[i][mask]
        if len(segment) < len(template):
            tracked_times.append(prev_t)
            continue
        corr = np.abs(correlate(segment, template, mode="valid"))
        idx_local = np.argmax(corr)
        t_window_start = t_arr_local[mask][0]
        t_echo = t_window_start + idx_local * _dt + template_halfwidth_s
        tracked_times.append(t_echo)
    return np.array(tracked_times)


if __name__ == "__main__":
    print(f"*** {labels.PENDING_SIGNOFF_BANNER} ***")
    print(f"*** {labels.TOY_EXACT_GT_CAPTION} ***")
    print(f"Multi-angle vector triangulation: target at theta={THETA_DEG} deg off boresight.")

    phases = np.linspace(0, 1, p3cfg.N_FRAMES)
    lv_radii_motion = [p3cfg.lv_radius_at_phase(p) for p in phases]
    r_boundary = np.array([r + p3cfg.WALL_THICKNESS_CELLS for r in lv_radii_motion])

    target_row_ref, target_col_ref = target_position_at_radius(r_boundary[0])
    print(f"Reference (ED) target position: row={target_row_ref:.2f}, col={target_col_ref:.2f}")

    # Sub-aperture A: centered directly above the target (on-axis to it).
    center_col_A = target_col_ref
    # Sub-aperture B: offset substantially to one side.
    center_col_B = target_col_ref - 95

    sources_A, xs_A, ys_A = build_focused_sources(center_col_A, target_row_ref, target_col_ref)
    sources_B, xs_B, ys_B = build_focused_sources(center_col_B, target_row_ref, target_col_ref)

    u_A = look_direction(center_col_A, target_row_ref, target_col_ref)
    u_B = look_direction(center_col_B, target_row_ref, target_col_ref)
    angle_between = np.degrees(np.arccos(np.clip(u_A @ u_B, -1, 1)))
    print(f"u_A={u_A}, u_B={u_B}, angle between look directions: {angle_between:.1f} deg")

    rcv_col_A = int(center_col_A) + 5  # small pitch-catch-style offset from sub-aperture center
    rcv_col_B = int(center_col_B) + 5

    print("=== Simulating sub-aperture A (on-axis to target) ===")
    traces_A = []
    for r in r_boundary:
        field = simulate_subaperture(r - p3cfg.WALL_THICKNESS_CELLS, sources_A)
        traces_A.append(np.array(field[:, rcv_col_A, pmr.array_y]))

    print("=== Simulating sub-aperture B (offset, oblique) ===")
    traces_B = []
    for r in r_boundary:
        field = simulate_subaperture(r - p3cfg.WALL_THICKNESS_CELLS, sources_B)
        traces_B.append(np.array(field[:, rcv_col_B, pmr.array_y]))

    t_ref_A = expected_round_trip_time(center_col_A, rcv_col_A, target_row_ref, target_col_ref)
    t_ref_B = expected_round_trip_time(center_col_B, rcv_col_B, target_row_ref, target_col_ref)
    # Refine against the actual echo peak near this (now physically
    # correct) estimate in the ED (frame 0) trace, not trusted blindly.
    def refine_t_ref(trace, t_guess, margin=1e-6):
        mask = (t_arr_local > t_guess - margin) & (t_arr_local < t_guess + margin)
        idx = np.argmax(np.abs(trace[mask]))
        return t_arr_local[mask][idx]

    t_ref_A = refine_t_ref(traces_A[0], t_ref_A)
    t_ref_B = refine_t_ref(traces_B[0], t_ref_B)
    print(f"Refined t_ref_A={t_ref_A:.3e}s, t_ref_B={t_ref_B:.3e}s")

    template_halfwidth = 0.5e-6
    search_margin = 1.2e-6  # per-step, same reasoning as sequential speckle tracking

    times_A = track_sequential_range(traces_A, t_ref_A, template_halfwidth, search_margin)
    times_B = track_sequential_range(traces_B, t_ref_B, template_halfwidth, search_margin)

    range_A = times_A * c_ref / 2 * 1e3  # mm, one-way from rcv_A's phase center approx
    range_B = times_B * c_ref / 2 * 1e3
    range_change_A = range_A - range_A[0]
    range_change_B = range_B - range_B[0]

    # True displacement (row,col), relative to ED, at each frame -- pure
    # radial scaling, direction (-cos(theta), sin(theta)).
    true_disp_mag_mm = (r_boundary - r_boundary[0]) * cfg.DX_M * 1e3
    true_disp_row_mm = true_disp_mag_mm * (-np.cos(THETA))
    true_disp_col_mm = true_disp_mag_mm * (np.sin(THETA))

    # Recover (d_row, d_col) from range_change_A/B via the 2x2 system:
    # u_A . d = range_change_A ; u_B . d = range_change_B
    M = np.array([u_A, u_B])
    M_inv = np.linalg.inv(M)

    print("\nNoiseless per-frame check (recovered vs true displacement vector, mm):")
    recovered_rows, recovered_cols = [], []
    for i in range(len(phases)):
        rc = np.array([range_change_A[i], range_change_B[i]])
        d = M_inv @ rc
        recovered_rows.append(d[0]); recovered_cols.append(d[1])
        print(f"  frame {i}: true=({true_disp_row_mm[i]:.3f},{true_disp_col_mm[i]:.3f}) "
              f"recovered=({d[0]:.3f},{d[1]:.3f})")

    recovered_rows = np.array(recovered_rows); recovered_cols = np.array(recovered_cols)
    rmse_row = np.sqrt(np.mean((recovered_rows - true_disp_row_mm) ** 2))
    rmse_col = np.sqrt(np.mean((recovered_cols - true_disp_col_mm) ** 2))
    print(f"\nRMSE: row={rmse_row:.4f}mm, col={rmse_col:.4f}mm")

    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    axes[0].plot(phases, true_disp_row_mm, "k--", label="true")
    axes[0].plot(phases, recovered_rows, "o-", label="recovered")
    axes[0].set_title(f"Row (boresight) displacement, RMSE={rmse_row:.3f}mm")
    axes[0].set_xlabel("cardiac phase"); axes[0].legend(fontsize=8)
    axes[1].plot(phases, true_disp_col_mm, "k--", label="true")
    axes[1].plot(phases, recovered_cols, "s-", label="recovered", color="orange")
    axes[1].set_title(f"Col (cross-range) displacement, RMSE={rmse_col:.3f}mm")
    axes[1].set_xlabel("cardiac phase"); axes[1].legend(fontsize=8)
    fig.suptitle(f"Vector triangulation from 2 focused sub-apertures "
                f"(look-direction angle: {angle_between:.1f} deg)\n(TOY: exact prescribed ground truth)")
    plt.tight_layout(rect=[0, 0.08, 1, 0.94])
    labels.add_banner(fig)
    os.makedirs("results/figures", exist_ok=True)
    plt.savefig("results/figures/phase3_vector_triangulation.png", dpi=150)
    print("Saved results/figures/phase3_vector_triangulation.png")

    # --- Noise sweep (same realistic 20-34dB SNR range as runs -14..-16) ---
    # Also tracks each sub-aperture's OWN scalar RMSE (against its own
    # analytically-expected projection) to disentangle whether cross-range
    # noise is dominated by sub-aperture B's oblique echo being weaker, or
    # by sequential-tracking drift affecting both apertures similarly.
    expected_rc_A = true_disp_row_mm * u_A[0] + true_disp_col_mm * u_A[1]
    expected_rc_B = true_disp_row_mm * u_B[0] + true_disp_col_mm * u_B[1]

    reference_peak_A = np.max(np.abs(traces_A[0]))
    reference_peak_B = np.max(np.abs(traces_B[0]))

    N_REALIZATIONS = 20
    print(f"\n(RMSE averaged over {N_REALIZATIONS} independent noise realizations)")
    print(f"\n{'noise':>8} {'row RMSE (mm)':>14} {'col RMSE (mm)':>14} "
          f"{'subA scalar RMSE':>18} {'subB scalar RMSE':>18}")
    row_rmses, col_rmses = [], []
    for noise_level in p3cfg.NOISE_LEVELS:
        row_trials, col_trials, subA_trials, subB_trials = [], [], [], []
        for trial in range(N_REALIZATIONS):
            rng_A = np.random.default_rng(1000 * trial + 11)
            rng_B = np.random.default_rng(1000 * trial + 13)
            noisy_A = [t + rng_A.normal(0, noise_level * reference_peak_A, size=t.shape)
                      for t in traces_A]
            noisy_B = [t + rng_B.normal(0, noise_level * reference_peak_B, size=t.shape)
                      for t in traces_B]

            tA = track_sequential_range(noisy_A, t_ref_A, template_halfwidth, search_margin)
            tB = track_sequential_range(noisy_B, t_ref_B, template_halfwidth, search_margin)
            rA = tA * c_ref / 2 * 1e3 - tA[0] * c_ref / 2 * 1e3
            rB = tB * c_ref / 2 * 1e3 - tB[0] * c_ref / 2 * 1e3

            rec_row, rec_col = [], []
            for i in range(len(phases)):
                d = M_inv @ np.array([rA[i], rB[i]])
                rec_row.append(d[0]); rec_col.append(d[1])
            rec_row = np.array(rec_row); rec_col = np.array(rec_col)

            row_trials.append(np.sqrt(np.mean((rec_row - true_disp_row_mm) ** 2)))
            col_trials.append(np.sqrt(np.mean((rec_col - true_disp_col_mm) ** 2)))
            subA_trials.append(np.sqrt(np.mean((rA - expected_rc_A) ** 2)))
            subB_trials.append(np.sqrt(np.mean((rB - expected_rc_B) ** 2)))

        row_rmses.append(np.mean(row_trials)); col_rmses.append(np.mean(col_trials))
        print(f"{noise_level:>8} {np.mean(row_trials):>14.4f} {np.mean(col_trials):>14.4f} "
              f"{np.mean(subA_trials):>18.4f} {np.mean(subB_trials):>18.4f}")

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(p3cfg.NOISE_LEVELS, row_rmses, "o-", label="row (boresight)")
    ax.plot(p3cfg.NOISE_LEVELS, col_rmses, "s-", label="col (cross-range)")
    ax.set_xlabel("noise level (fraction of reference peak; 0.02/0.05/0.10 = 34/26/20dB)")
    ax.set_ylabel("RMSE vs. true displacement (mm)")
    ax.set_title("Vector triangulation under realistic noise\n(TOY: exact prescribed ground truth)")
    ax.legend(fontsize=9)
    plt.tight_layout(rect=[0, 0.06, 1, 1])
    labels.add_banner(fig)
    plt.savefig("results/figures/phase3_vector_triangulation_noise.png", dpi=150)
    print("Saved results/figures/phase3_vector_triangulation_noise.png")
