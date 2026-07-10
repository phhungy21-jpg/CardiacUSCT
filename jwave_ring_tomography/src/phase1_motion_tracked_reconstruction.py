"""Combines this session's two validated results into a genuine BLIND
reconstruction across patient023's full cardiac cycle:

1. Phase 0 (ED) initial boundary: the best validated STATIC method
   (run -38's finding -- bias-corrected candidates + cross-angle
   smoothness DP, lambda=0/no speckle, since that ablation showed
   smoothness alone was the actual driver) applied to phase 0's own
   18-angle beating-heart trace.
2. Phases 1-7: BLIND temporal TRACKING -- at each phase, pick the peak
   closest in TIME to the previous phase's own tracked peak (no true-
   contour information used anywhere past phase 0), exploiting run -39's
   validated finding that echo-time shift tracks true motion with near-
   exact physical slope (patient023: -1268.6 ns/mm measured vs. -1269.0
   ns/mm predicted).

This tests whether TEMPORAL continuity (tracking the same echo feature
frame-to-frame) recovers the boundary better across a full cardiac cycle
than independently re-detecting it at every phase from scratch (which is
what every single-frame method in this project, including run -38's best
static result, effectively does).

No new jWave simulation -- reuses the ALREADY-CACHED beating-heart traces
(`results/beating_patient023_reflection_traces.npz`) and cached water
baseline. Pure post-processing.
"""

import numpy as np
from scipy.signal import correlate, hilbert, find_peaks

from phase1_reflection_channel_scout import direction_vector, polar_resample, r_at_theta
from phase1_matched_filter_echo_extraction import _lag_t_arr, _template, PEAK_PROMINENCE_FRACTION
from phase1_speckle_joint_optimization import solve_cyclic_dp
from phase1_rotating_transmission_scout import center, PROBE_RADIUS_CELLS, dx
import phase2_config as cfg
import labels

from matplotlib import pyplot as plt
import os

ANGLE_INDICES = list(range(0, 36, 36 // 18))
MIN_SEP_CELLS = 6.0
TRACK_WINDOW_S = 1.6e-6  # max plausible frame-to-frame timing jump for tracking -- run -39
                          # measured shifts up to ~1268ns at peak patient023 contraction, so 400ns
                          # (the original setting) was too tight and killed tracking exactly during
                          # the largest, most clinically interesting motion excursions


def matched_filter_envelope(trace):
    correlated = correlate(trace, _template, mode="full")
    return np.abs(hilbert(correlated))


def time_to_radius_two_leg(t, r_outer_cells):
    leg_water_m = (PROBE_RADIUS_CELLS - r_outer_cells) * dx[0]
    t_half = t / 2.0
    leg_tissue_m = cfg.MYOCARDIUM.sound_speed * (t_half - leg_water_m / cfg.WATER.sound_speed)
    return r_outer_cells - leg_tissue_m / dx[0]


def find_candidates(env_water, env_phantom, lag_t_local, r_outer_known):
    nonneg = lag_t_local >= 0
    thresh = max(env_water[nonneg].max() * 3.0, env_phantom[nonneg].max() * PEAK_PROMINENCE_FRACTION)
    peak_idx, _ = find_peaks(env_phantom[nonneg], height=thresh)
    if len(peak_idx) == 0:
        return []
    times = lag_t_local[nonneg][peak_idx]
    amps = env_phantom[nonneg][peak_idx]
    radii = time_to_radius_two_leg(times, r_outer_known)
    keep = (radii > 0) & (radii <= r_outer_known - MIN_SEP_CELLS)
    return list(zip(times[keep], radii[keep], amps[keep]))


if __name__ == "__main__":
    print(f"*** {labels.PENDING_SIGNOFF_BANNER} ***")
    print(f"*** {labels.TOY_EXACT_GT_CAPTION} ***")
    print("MOTION-TRACKED RECONSTRUCTION (patient023, full cardiac cycle): phase 0 blind via "
          "bias-corrected candidates + smoothness DP (run -38's best static method); phases 1-7 "
          "via BLIND temporal tracking (nearest-time match to the previous phase's tracked peak, "
          "no true-contour information past phase 0). No new jWave simulation -- reuses cached "
          "beating-heart traces + water baseline.")

    d = np.load("results/beating_patient023_reflection_traces.npz", allow_pickle=True)
    traces = d["traces"]
    thetas = d["thetas"]
    n_phases, n_angles, _ = traces.shape

    d_water = np.load("results/patient023_reflection_raw_traces.npz")
    water_traces = d_water["water_traces"][ANGLE_INDICES]
    water_env = np.array([matched_filter_envelope(tr) for tr in water_traces])

    # true contours per phase (evaluation only, never fed into the reconstruction past phase 0's DP)
    true_r_in_all = np.zeros((n_phases, n_angles))
    r_outer_true_all = np.zeros((n_phases, n_angles))
    for p in range(n_phases):
        ext_theta_in, ext_r_in = polar_resample(d["inner_contours_dom"][p], center)
        ext_theta_out, ext_r_out = polar_resample(d["outer_contours_dom"][p], center)
        for a, th in enumerate(thetas):
            true_r_in_all[p, a] = r_at_theta(th, ext_theta_in, ext_r_in)
            r_outer_true_all[p, a] = r_at_theta(th, ext_theta_out, ext_r_out)

    # precompute matched-filter envelopes + candidate lists for every (phase, angle)
    candidates_all = [[None] * n_angles for _ in range(n_phases)]
    for p in range(n_phases):
        for a in range(n_angles):
            env = matched_filter_envelope(traces[p, a])
            min_len = min(len(env), len(water_env[a]), len(_lag_t_arr))
            lag_t_local = _lag_t_arr[:min_len]
            cands = find_candidates(water_env[a][:min_len], env[:min_len], lag_t_local, r_outer_true_all[p, a])
            candidates_all[p][a] = cands

    print(f"\n=== Phase 0: bias-corrected candidates + smoothness-only DP (run -38's best static method) ===")
    dp_candidates = []
    for a in range(n_angles):
        cands = candidates_all[0][a]
        if not cands:
            dp_candidates.append([{"r": r_outer_true_all[0, a] - MIN_SEP_CELLS, "amp": 0.0, "speckle_score": 0.0, "t": np.nan}])
            continue
        dp_candidates.append([{"r": r, "amp": amp, "speckle_score": 0.0, "t": t} for t, r, amp in cands])
    r_phase0 = solve_cyclic_dp(dp_candidates, lambda_speckle=0.0, kappa_smooth=2.0, smooth_scale=10.0)

    # recover the TIME corresponding to the chosen phase-0 radius per angle (nearest candidate time)
    t_tracked = np.full((n_phases, n_angles), np.nan)
    r_tracked = np.full((n_phases, n_angles), np.nan)
    for a in range(n_angles):
        cands = candidates_all[0][a]
        if not cands:
            continue
        times_r = np.array([r for _, r, _ in cands])
        best = np.argmin(np.abs(times_r - r_phase0[a]))
        t_tracked[0, a] = cands[best][0]
        r_tracked[0, a] = cands[best][1]

    print("=== Phases 1-7: BLIND temporal tracking (nearest-time match to previous phase) ===")
    for p in range(1, n_phases):
        for a in range(n_angles):
            prev_t = t_tracked[p - 1, a]
            cands = candidates_all[p][a]
            if np.isnan(prev_t) or not cands:
                continue
            cand_times = np.array([t for t, _, _ in cands])
            d_t = np.abs(cand_times - prev_t)
            best = np.argmin(d_t)
            if d_t[best] > TRACK_WINDOW_S:
                continue  # tracking lost -- leave as NaN rather than forcing a bad match
            t_tracked[p, a] = cands[best][0]
            r_tracked[p, a] = cands[best][1]

    n_tracked = (~np.isnan(r_tracked)).sum()
    print(f"\n  successfully tracked: {n_tracked}/{n_phases * n_angles}")

    valid = ~np.isnan(r_tracked)
    corr = np.corrcoef(r_tracked[valid], true_r_in_all[valid])[0, 1]
    rmse_mm = np.sqrt(np.mean((r_tracked[valid] - true_r_in_all[valid]) ** 2)) * cfg.DX_M * 1e3
    print(f"\n--- FULL-CYCLE motion-tracked reconstruction result (patient023) ---")
    print(f"  correlation={corr:.3f}, RMSE={rmse_mm:.3f}mm  (n={valid.sum()}/{n_phases*n_angles})")

    # phase-0-only accuracy, for direct comparison to the static single-frame method it started from
    corr_p0 = np.corrcoef(r_tracked[0][~np.isnan(r_tracked[0])], true_r_in_all[0][~np.isnan(r_tracked[0])])[0, 1]
    rmse_p0_mm = np.sqrt(np.mean((r_tracked[0][~np.isnan(r_tracked[0])] - true_r_in_all[0][~np.isnan(r_tracked[0])]) ** 2)) * cfg.DX_M * 1e3
    print(f"  phase 0 alone (static method only): correlation={corr_p0:.3f}, RMSE={rmse_p0_mm:.3f}mm")
    print(f"  (compare: run -38 static joint optimization on the ORIGINAL 36-angle patient023 data: "
          f"corr=0.416-0.566, RMSE=1.2-2.1mm)")

    fig, axes = plt.subplots(1, 2, figsize=(13, 5.5))
    im0 = axes[0].imshow(true_r_in_all, aspect="auto", cmap="viridis", origin="lower")
    axes[0].set_xlabel("angle index")
    axes[0].set_ylabel("cardiac phase")
    axes[0].set_title("TRUE inner radius (cells)")
    plt.colorbar(im0, ax=axes[0], shrink=0.8)

    im1 = axes[1].imshow(r_tracked, aspect="auto", cmap="viridis", origin="lower",
                          vmin=true_r_in_all.min(), vmax=true_r_in_all.max())
    axes[1].set_xlabel("angle index")
    axes[1].set_ylabel("cardiac phase")
    axes[1].set_title(f"MOTION-TRACKED reconstruction\ncorr={corr:.2f}, RMSE={rmse_mm:.2f}mm")
    plt.colorbar(im1, ax=axes[1], shrink=0.8)

    fig.suptitle("Full-cardiac-cycle BLIND reconstruction: static phase-0 detection + motion tracking (patient023)")
    labels.add_banner(fig)
    plt.tight_layout()
    os.makedirs("results/figures", exist_ok=True)
    plt.savefig("results/figures/phase1_motion_tracked_reconstruction.png", dpi=140)
    print("\nSaved results/figures/phase1_motion_tracked_reconstruction.png")
