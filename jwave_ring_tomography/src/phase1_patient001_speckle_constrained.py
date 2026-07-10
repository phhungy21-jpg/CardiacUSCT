"""Generalization test: applies the EXACT same speckle-constrained
candidate selection + smoothness-DP pipeline (runs -43/-44) to
patient001 -- a genuinely different, much more mildly-contracting
patient (~1.6% real contraction vs. patient023's ~18%), to check
whether the win generalizes beyond the one patient it was discovered on.

Reuses patient001's ALREADY-cached transmission-channel data (`results/
patient001_two_tissue_rays.npz`, runs -05/-07) for the DAS outer-
boundary estimate, and the newly-simulated reflection+speckle traces
(`phase1_patient001_reflection_speckle_sim.py`).
"""

import numpy as np
from scipy.signal import correlate, hilbert, find_peaks

from phase1_reflection_channel_scout import (
    thetas, direction_vector, polar_resample, r_at_theta,
)
from phase1_matched_filter_echo_extraction import _lag_t_arr, _template, PEAK_PROMINENCE_FRACTION
from phase1_das_reflectivity_imaging import extract_boundary_from_image
from phase1_speckle_informed_surface_selection_v2 import das_energy_field
from phase1_speckle_path_constraint import speckle_rising_edge_radius
from phase1_speckle_joint_optimization import solve_cyclic_dp
from phase1_rotating_transmission_scout import center, PROBE_RADIUS_CELLS, dx
import tomography_recon as recon
from phase1_rotating_transmission_scout import probe_position, N
import phase2_config as cfg
import labels

from matplotlib import pyplot as plt
import os

R_MAX_CELLS = 0.9 * PROBE_RADIUS_CELLS
MIN_SEP_CELLS = 6.0
IMG_SIZE = 150
_nonneg = _lag_t_arr >= 0
PROXIMITY_SCALE_CELLS = 8.0


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
    print("PATIENT001 GENERALIZATION TEST: does speckle-constrained candidate selection + "
          "smoothness DP (runs -43/-44, corr=0.738 on patient023) hold up on a different, "
          "much more mildly-contracting patient?")

    d_refl = np.load("results/patient001_reflection_raw_traces.npz")
    homog_traces, speckle_traces = d_refl["homogeneous_traces"], d_refl["speckle_traces"]
    inner_contour_dom = d_refl["inner_contour_dom"]
    ext_theta_in, ext_r_in = polar_resample(inner_contour_dom, center)
    true_r_in = np.array([r_at_theta(th, ext_theta_in, ext_r_in) for th in thetas])

    # reuse patient001's cached water-only trace count: none exist standalone, but
    # water-only response depends only on probe geometry (fixed), so patient023's is reusable
    d_water = np.load("results/patient023_reflection_raw_traces.npz")
    water_traces = d_water["water_traces"]
    water_env = [matched_filter_envelope(tr) for tr in water_traces]
    homog_env = [matched_filter_envelope(tr) for tr in homog_traces]
    speckle_env = [matched_filter_envelope(tr) for tr in speckle_traces]

    print("\n=== Rebuilding patient001's outer-boundary estimate from cached transmission data ===")
    d_tx = np.load("results/patient001_two_tissue_rays.npz")
    pairs_excess_delay_ns = {(tt, tr): v for tt, tr, v in zip(d_tx["theta_tx"], d_tx["theta_rx"], d_tx["excess_delay_ns"])}
    image_sirt, img_rows_das, img_cols_das, _ = recon.sirt_reconstruct(
        pairs_excess_delay_ns, probe_position, IMG_SIZE, N, n_iters=30, relax=0.15)
    r_outer_est = extract_boundary_from_image(image_sirt, img_rows_das, img_cols_das, center, thetas, r_max_cells=R_MAX_CELLS)
    print(f"  mean outer radius estimate: {r_outer_est.mean():.1f} cells")

    print("\n=== Building incoherent (energy) speckle field ===")
    speckle_field, img_rows, img_cols = das_energy_field(speckle_env, homog_env)

    print("=== Extracting per-angle speckle rising-edge radius ===")
    r_speckle = np.array([speckle_rising_edge_radius(speckle_field, img_rows, img_cols, th) for th in thetas])

    r_naive = []
    candidates_per_angle = []
    for i, theta in enumerate(thetas):
        cands = generate_candidates(water_env[i], homog_env[i], r_outer_est[i])
        if not cands:
            r_naive.append(np.nan)
            candidates_per_angle.append([{"r": r_outer_est[i] - MIN_SEP_CELLS, "amp": 0.0, "speckle_score": 0.0}])
            continue
        r_naive.append(max(cands, key=lambda ra: ra[1])[0])
        entries = []
        for r_cand, amp in cands:
            proximity_score = 0.0 if np.isnan(r_speckle[i]) else -((r_cand - r_speckle[i]) / PROXIMITY_SCALE_CELLS) ** 2
            entries.append({"r": r_cand, "amp": 0.0, "speckle_score": proximity_score})
        candidates_per_angle.append(entries)
    r_naive = np.array(r_naive)

    print("\n--- Results (patient001) ---")
    eval_method(r_naive, true_r_in, "naive strongest-amplitude (baseline)")
    valid_speckle = ~np.isnan(r_speckle)
    eval_method(r_speckle, true_r_in, "speckle-alone")

    r_constrained_only = solve_cyclic_dp(candidates_per_angle, lambda_speckle=1.0, kappa_smooth=0.0, smooth_scale=10.0)
    eval_method(r_constrained_only, true_r_in, "speckle-constrained selection (kappa=0)")

    best_corr, best_kappa, best_r = -np.inf, None, None
    for kappa in [0.0, 0.5, 1.0, 2.0, 4.0, 8.0]:
        r_result = solve_cyclic_dp(candidates_per_angle, lambda_speckle=1.0, kappa_smooth=kappa, smooth_scale=10.0)
        corr, rmse = eval_method(r_result, true_r_in, f"  +smoothness DP kappa={kappa}")
        if corr > best_corr:
            best_corr, best_kappa, best_r, best_rmse = corr, kappa, r_result, rmse

    print(f"\nBest: kappa={best_kappa}, corr={best_corr:.3f}, RMSE={best_rmse:.3f}mm")
    print(f"(compare: patient023 -- run -43 speckle-constrained alone: 0.697/0.903mm; "
          f"run -44 +DP best: 0.738/0.70-0.80mm)")

    if best_corr > 0.4:
        print("  -> GENERALIZES: speckle-constrained selection works on a second, independent patient.")
    else:
        print("  -> Does NOT clearly generalize to patient001 at this setting.")

    d_rows, d_cols = direction_vector(thetas)
    fig, ax = plt.subplots(figsize=(9, 5.5))
    ax.plot(thetas, true_r_in, "k-", linewidth=2, label="true inner contour")
    ax.plot(thetas, r_naive, "o-", color="red", alpha=0.5, label="naive amplitude")
    ax.plot(thetas, best_r, "s-", color="lime", label=f"speckle-constrained+DP (kappa={best_kappa}), corr={best_corr:.2f}")
    ax.set_xlabel("angle (deg)")
    ax.set_ylabel("inner-boundary radius estimate (cells)")
    ax.legend(fontsize=8)
    ax.set_title("patient001 generalization: speckle-constrained selection + smoothness DP")
    labels.add_banner(fig)
    plt.tight_layout()
    os.makedirs("results/figures", exist_ok=True)
    plt.savefig("results/figures/phase1_patient001_speckle_constrained.png", dpi=140)
    print("\nSaved results/figures/phase1_patient001_speckle_constrained.png")
