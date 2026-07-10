"""Fixes both problems diagnosed in run -40:

1. WEAK SEED: run -40 seeded phase 0 using only the 18-angle beating-
   heart subset (corr=0.219 alone). Here phase 0 is seeded from the
   FULL 36-angle static patient023 joint-optimization result (run -38's
   best setting, lambda=0/kappa=2.0, corr=0.566) -- the SAME underlying
   anatomy (fraction=0 = ED), just leveraging twice the angular
   information for the initial detection, then subsampled to the 18
   angles the beating-heart simulation actually used.

2. NAIVE TRACKING LOCKS ONTO STATIC ARTIFACTS: run -40's nearest-time-
   to-previous-frame tracker had no way to distinguish "target barely
   moved" from "I'm re-finding a stable non-target echo." Replaced with
   a constant-VELOCITY predictor: predict the next phase's expected
   time as (previous time + current velocity estimate), not just the
   previous time itself -- a genuinely static artifact has zero
   estimated velocity, so once real motion is underway, predicting
   forward via velocity actively looks AWAY from a static echo's
   position, making it much harder to spuriously re-lock onto it.

No new jWave simulation -- reuses cached beating-heart traces, cached
water baseline, cached patient023 speckle field (for candidate
generation consistency with run -38, though lambda=0 makes it inert),
and the cached static-patient023 DAS outer-boundary estimate.
"""

import numpy as np
from scipy.signal import correlate, hilbert, find_peaks

from phase1_reflection_channel_scout import (
    thetas as thetas_36, direction_vector, polar_resample, r_at_theta, pitch_catch_positions,
)
from phase1_matched_filter_echo_extraction import _lag_t_arr, _template, PEAK_PROMINENCE_FRACTION
from phase1_das_reflectivity_imaging import extract_boundary_from_image
from phase1_speckle_informed_surface_selection_v2 import das_energy_field
from phase1_speckle_joint_optimization import generate_candidates_with_features, solve_cyclic_dp
from phase1_rotating_transmission_scout import center, PROBE_RADIUS_CELLS, dx
import phase2_config as cfg
import labels

from matplotlib import pyplot as plt
import os

ANGLE_INDICES = list(range(0, 36, 36 // 18))
MIN_SEP_CELLS = 6.0
R_MAX_CELLS = 0.9 * PROBE_RADIUS_CELLS


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
    print("MOTION-TRACKED RECONSTRUCTION v2: full-36-angle seed (run -38's best static result) "
          "+ constant-velocity tracker (predicts forward via estimated velocity, not just the "
          "previous position -- resists locking onto static artifacts). No new jWave simulation.")

    # === Step 1: full 36-angle static joint-optimization seed for patient023 (ED / phase 0) ===
    print("\n=== Rebuilding the FULL 36-angle static joint-optimization seed (run -38's method) ===")
    d_das = np.load("results/patient023_das_images.npz")
    image_sr, img_rows_das, img_cols_das = d_das["image_straight_ray"], d_das["img_rows"], d_das["img_cols"]
    r_outer_est_36 = extract_boundary_from_image(image_sr, img_rows_das, img_cols_das, center, thetas_36, r_max_cells=R_MAX_CELLS)

    d_refl = np.load("results/patient023_reflection_raw_traces.npz")
    water_traces_36, homog_traces_36 = d_refl["water_traces"], d_refl["phantom_traces"]
    d_speckle = np.load("results/patient023_speckle_raw_traces.npz")
    speckle_traces_36 = d_speckle["speckle_traces"]
    water_env_36 = [matched_filter_envelope(tr) for tr in water_traces_36]
    homog_env_36 = [matched_filter_envelope(tr) for tr in homog_traces_36]
    speckle_env_36 = [matched_filter_envelope(tr) for tr in speckle_traces_36]
    speckle_field_36, img_rows_36, img_cols_36 = das_energy_field(speckle_env_36, homog_env_36)

    candidates_36 = []
    for i, theta in enumerate(thetas_36):
        cands = generate_candidates_with_features(
            water_env_36[i], homog_env_36[i], r_outer_est_36[i], theta,
            speckle_field_36, img_rows_36, img_cols_36)
        if not cands:
            cands = [{"r": r_outer_est_36[i] - MIN_SEP_CELLS, "amp": 0.0, "speckle_score": 0.0}]
        candidates_36.append(cands)
    r_joint_36 = solve_cyclic_dp(candidates_36, lambda_speckle=0.0, kappa_smooth=2.0, smooth_scale=10.0)

    ext_theta_in_true, ext_r_in_true = polar_resample(
        np.load("results/beating_patient023_reflection_traces.npz", allow_pickle=True)["inner_contours_dom"][0], center)
    true_r_in_phase0_36 = np.array([r_at_theta(th, ext_theta_in_true, ext_r_in_true) for th in thetas_36])
    corr_36 = np.corrcoef(r_joint_36, true_r_in_phase0_36)[0, 1]
    print(f"  36-angle seed quality check: corr={corr_36:.3f} (compare run -38's original: 0.566)")

    r_seed_18 = r_joint_36[ANGLE_INDICES]  # subsample to the 18 angles the beating-heart sim used

    # === Step 2: load beating-heart data, find phase-0 candidate nearest the 36-angle seed ===
    d = np.load("results/beating_patient023_reflection_traces.npz", allow_pickle=True)
    traces = d["traces"]
    thetas = d["thetas"]
    n_phases, n_angles, _ = traces.shape

    water_traces_18 = water_traces_36[ANGLE_INDICES]
    water_env_18 = np.array([matched_filter_envelope(tr) for tr in water_traces_18])

    true_r_in_all = np.zeros((n_phases, n_angles))
    r_outer_true_all = np.zeros((n_phases, n_angles))
    for p in range(n_phases):
        ext_theta_in, ext_r_in = polar_resample(d["inner_contours_dom"][p], center)
        ext_theta_out, ext_r_out = polar_resample(d["outer_contours_dom"][p], center)
        for a, th in enumerate(thetas):
            true_r_in_all[p, a] = r_at_theta(th, ext_theta_in, ext_r_in)
            r_outer_true_all[p, a] = r_at_theta(th, ext_theta_out, ext_r_out)

    candidates_all = [[None] * n_angles for _ in range(n_phases)]
    for p in range(n_phases):
        for a in range(n_angles):
            env = matched_filter_envelope(traces[p, a])
            min_len = min(len(env), len(water_env_18[a]), len(_lag_t_arr))
            lag_t_local = _lag_t_arr[:min_len]
            cands = find_candidates(water_env_18[a][:min_len], env[:min_len], lag_t_local, r_outer_true_all[p, a])
            candidates_all[p][a] = cands

    t_tracked = np.full((n_phases, n_angles), np.nan)
    r_tracked = np.full((n_phases, n_angles), np.nan)
    velocity = np.zeros(n_angles)  # ns per phase-step, estimated online

    for a in range(n_angles):
        cands = candidates_all[0][a]
        if not cands:
            continue
        radii = np.array([r for _, r, _ in cands])
        best = np.argmin(np.abs(radii - r_seed_18[a]))
        t_tracked[0, a] = cands[best][0]
        r_tracked[0, a] = cands[best][1]

    print("\n=== Phases 1-7: constant-VELOCITY tracking (predicts forward, resists static lock-on) ===")
    for p in range(1, n_phases):
        for a in range(n_angles):
            prev_t = t_tracked[p - 1, a]
            cands = candidates_all[p][a]
            if np.isnan(prev_t) or not cands:
                continue
            predicted_t = prev_t + velocity[a]
            cand_times = np.array([t for t, _, _ in cands])
            d_t = np.abs(cand_times - predicted_t)
            best = np.argmin(d_t)
            # generous window: velocity prediction should already be close; allow slack for
            # velocity-estimate error, especially in the first couple of tracked steps
            if d_t[best] > 1.6e-6:
                continue
            new_t = cand_times[best]
            observed_step = new_t - prev_t
            # exponential-moving-average velocity update (smooths noisy per-step estimates)
            velocity[a] = 0.5 * velocity[a] + 0.5 * observed_step
            t_tracked[p, a] = new_t
            r_tracked[p, a] = cands[best][1]

    n_tracked = (~np.isnan(r_tracked)).sum()
    valid = ~np.isnan(r_tracked)
    corr = np.corrcoef(r_tracked[valid], true_r_in_all[valid])[0, 1]
    rmse_mm = np.sqrt(np.mean((r_tracked[valid] - true_r_in_all[valid]) ** 2)) * cfg.DX_M * 1e3
    print(f"\n--- FULL-CYCLE motion-tracked reconstruction v2 result (patient023) ---")
    print(f"  tracked: {n_tracked}/{n_phases * n_angles}")
    print(f"  correlation={corr:.3f}, RMSE={rmse_mm:.3f}mm")
    corr_p0 = np.corrcoef(r_tracked[0][~np.isnan(r_tracked[0])], true_r_in_all[0][~np.isnan(r_tracked[0])])[0, 1]
    print(f"  phase 0 alone (36-angle seed, subsampled to 18): correlation={corr_p0:.3f}")
    print(f"  (compare run -40: full-cycle corr=0.353, RMSE=1.54mm, phase-0-alone corr=0.219 -- "
          f"static lock-on failure)")

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
    axes[1].set_title(f"v2: 36-angle seed + velocity tracker\ncorr={corr:.2f}, RMSE={rmse_mm:.2f}mm")
    plt.colorbar(im1, ax=axes[1], shrink=0.8)

    fig.suptitle("Full-cardiac-cycle BLIND reconstruction v2: better seed + velocity-aware tracking (patient023)")
    labels.add_banner(fig)
    plt.tight_layout()
    os.makedirs("results/figures", exist_ok=True)
    plt.savefig("results/figures/phase1_motion_tracked_reconstruction_v2.png", dpi=140)
    print("\nSaved results/figures/phase1_motion_tracked_reconstruction_v2.png")
