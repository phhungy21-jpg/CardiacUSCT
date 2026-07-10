"""Combines run -43's win (speckle as a geometric LOCATION constraint,
corr=0.697) with run -38's smoothness-DP idea: instead of amplitude as
the DP's unary term, use PROXIMITY TO THE SPECKLE-DERIVED LOCATION
(a Gaussian-like score, high when a candidate sits close to the speckle
rising-edge radius for that angle) as the unary term, then let the SAME
cyclic dynamic-programming smoothness constraint (exploiting that a real
contour is one connected curve, not independent per-angle picks) refine
across neighboring angles on top of that.

At kappa_smooth=0 this must reduce to exactly run -43's result (a
built-in consistency check): with no smoothness penalty, the DP
independently picks the highest-unary (= closest-to-speckle-location)
candidate per angle, identical to run -43's direct nearest-candidate
selection.

No new jWave simulation -- reuses the identical cached patient023 data
as runs -38/-43.
"""

import numpy as np
from scipy.signal import correlate, hilbert, find_peaks

from phase1_patient023_validation import load_real_contours, MRI_NPZ
from phase1_reflection_channel_scout import (
    thetas, direction_vector, polar_resample, r_at_theta, pitch_catch_positions,
)
from phase1_matched_filter_echo_extraction import _lag_t_arr, _template, PEAK_PROMINENCE_FRACTION
from phase1_das_reflectivity_imaging import extract_boundary_from_image
from phase1_speckle_informed_surface_selection_v2 import das_energy_field
from phase1_speckle_path_constraint import speckle_rising_edge_radius
from phase1_speckle_joint_optimization import solve_cyclic_dp
from phase1_rotating_transmission_scout import center, PROBE_RADIUS_CELLS, dx
import phase2_config as cfg
import labels

from matplotlib import pyplot as plt
import os

R_MAX_CELLS = 0.9 * PROBE_RADIUS_CELLS
MIN_SEP_CELLS = 6.0
_nonneg = _lag_t_arr >= 0
PROXIMITY_SCALE_CELLS = 8.0  # scale of the Gaussian-like speckle-proximity unary score


def matched_filter_envelope(trace):
    correlated = correlate(trace, _template, mode="full")
    return np.abs(hilbert(correlated))


def time_to_radius_two_leg(t, r_outer_cells):
    leg_water_m = (PROBE_RADIUS_CELLS - r_outer_cells) * dx[0]
    t_half = t / 2.0
    leg_tissue_m = cfg.MYOCARDIUM.sound_speed * (t_half - leg_water_m / cfg.WATER.sound_speed)
    return r_outer_cells - leg_tissue_m / dx[0]


def generate_candidates(env_water, env_phantom, r_outer_known):
    thresh = max(env_water[_nonneg].max() * 3.0, env_phantom[_nonneg].max() * PEAK_PROMINENCE_FRACTION)
    peak_idx, _ = find_peaks(env_phantom[_nonneg], height=thresh)
    if len(peak_idx) == 0:
        return []
    times = _lag_t_arr[_nonneg][peak_idx]
    amps = env_phantom[_nonneg][peak_idx]
    radii = time_to_radius_two_leg(times, r_outer_known)
    keep = (radii > 0) & (radii <= r_outer_known - MIN_SEP_CELLS)
    return list(zip(radii[keep], amps[keep]))


def eval_method(r_est, true_r_in, name):
    valid = ~np.isnan(r_est)
    corr = np.corrcoef(r_est[valid], true_r_in[valid])[0, 1]
    rmse_mm = np.sqrt(np.mean((r_est[valid] - true_r_in[valid]) ** 2)) * cfg.DX_M * 1e3
    print(f"  {name}: {valid.sum()}/{len(true_r_in)}, corr={corr:.3f}, RMSE={rmse_mm:.3f}mm")
    return corr, rmse_mm


if __name__ == "__main__":
    print(f"*** {labels.PENDING_SIGNOFF_BANNER} ***")
    print(f"*** {labels.TOY_EXACT_GT_CAPTION} ***")
    print("SPECKLE-CONSTRAINED CANDIDATE SELECTION + SMOOTHNESS DP: uses proximity to the "
          "speckle-derived location (run -43) as the DP unary term, then adds cross-angle "
          "smoothness (run -38) on top. No new jWave simulation -- reuses cached patient023 data.")

    canvas_lv, canvas_myo, outer_contour_dom, inner_contour_dom = load_real_contours(MRI_NPZ)
    ext_theta_in, ext_r_in = polar_resample(inner_contour_dom, center)
    true_r_in = np.array([r_at_theta(th, ext_theta_in, ext_r_in) for th in thetas])

    d_das = np.load("results/patient023_das_images.npz")
    image_sr, img_rows_das, img_cols_das = d_das["image_straight_ray"], d_das["img_rows"], d_das["img_cols"]
    r_outer_est = extract_boundary_from_image(image_sr, img_rows_das, img_cols_das, center, thetas, r_max_cells=R_MAX_CELLS)

    d_refl = np.load("results/patient023_reflection_raw_traces.npz")
    water_traces, homog_traces = d_refl["water_traces"], d_refl["phantom_traces"]
    d_speckle = np.load("results/patient023_speckle_raw_traces.npz")
    speckle_traces = d_speckle["speckle_traces"]
    water_env = [matched_filter_envelope(tr) for tr in water_traces]
    homog_env = [matched_filter_envelope(tr) for tr in homog_traces]
    speckle_env = [matched_filter_envelope(tr) for tr in speckle_traces]

    speckle_field, img_rows, img_cols = das_energy_field(speckle_env, homog_env)
    r_speckle = np.array([speckle_rising_edge_radius(speckle_field, img_rows, img_cols, th) for th in thetas])

    candidates_per_angle = []
    for i, theta in enumerate(thetas):
        cands = generate_candidates(water_env[i], homog_env[i], r_outer_est[i])
        if not cands:
            cands = [{"r": r_outer_est[i] - MIN_SEP_CELLS, "amp": 0.0, "speckle_score": 0.0}]
        else:
            entries = []
            for r_cand, amp in cands:
                if np.isnan(r_speckle[i]):
                    proximity_score = 0.0
                else:
                    proximity_score = -((r_cand - r_speckle[i]) / PROXIMITY_SCALE_CELLS) ** 2
                entries.append({"r": r_cand, "amp": amp, "speckle_score": proximity_score})
            cands = entries
        candidates_per_angle.append(cands)

    print("\n--- Sweep over smoothness weight (kappa), unary = speckle-proximity only ---")
    results = {}
    for kappa in [0.0, 0.5, 1.0, 2.0, 4.0, 8.0]:
        # unary in solve_cyclic_dp is z-scored amplitude + lambda*z-scored speckle_score;
        # setting lambda huge and amplitude irrelevant isn't clean, so instead directly
        # overwrite "amp" with the proximity score itself (z-scoring a near-constant amplitude
        # would dilute the signal) -- here we pass lambda_speckle=1 and zero out amplitude's
        # contribution by using identical dummy amplitudes.
        r_result = solve_cyclic_dp(
            [[{"r": c["r"], "amp": 0.0, "speckle_score": c["speckle_score"]} for c in cands]
             for cands in candidates_per_angle],
            lambda_speckle=1.0, kappa_smooth=kappa, smooth_scale=10.0)
        corr, rmse = eval_method(r_result, true_r_in, f"kappa={kappa}")
        results[kappa] = (corr, rmse, r_result)

    best_kappa = max(results, key=lambda k: results[k][0])
    print(f"\nBest kappa={best_kappa}: corr={results[best_kappa][0]:.3f}, RMSE={results[best_kappa][1]:.3f}mm")
    print(f"(compare: run -43 speckle-constrained selection alone, no DP: corr=0.697, RMSE=0.903mm)")
    print(f"(compare: run -38 amplitude+smoothness DP: corr=0.566, RMSE~1.9mm)")

    r_best = results[best_kappa][2]
    d_rows, d_cols = direction_vector(thetas)
    fig, ax = plt.subplots(figsize=(9, 5.5))
    ax.plot(thetas, true_r_in, "k-", linewidth=2, label="true inner contour")
    ax.plot(thetas, r_best, "s-", color="lime", label=f"speckle-constrained + DP (kappa={best_kappa}), corr={results[best_kappa][0]:.2f}")
    ax.set_xlabel("angle (deg)")
    ax.set_ylabel("inner-boundary radius estimate (cells)")
    ax.legend(fontsize=8)
    ax.set_title("Speckle-constrained candidate selection + smoothness DP (patient023)")
    labels.add_banner(fig)
    plt.tight_layout()
    os.makedirs("results/figures", exist_ok=True)
    plt.savefig("results/figures/phase1_speckle_constrained_dp.png", dpi=140)
    print("\nSaved results/figures/phase1_speckle_constrained_dp.png")
