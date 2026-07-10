"""Quick parameter sweep over (lambda_speckle, kappa_smooth) for the
joint optimization (phase1_speckle_joint_optimization.py) -- free to
run since everything is cached, no new jWave simulation. Checks whether
run -38's first choice (lambda=1.0, kappa=0.5, corr=0.416) was a lucky
setting or whether the improvement is robust across nearby settings.
"""

import numpy as np

from phase1_patient023_validation import load_real_contours, MRI_NPZ
from phase1_reflection_channel_scout import thetas, polar_resample, r_at_theta
from phase1_das_reflectivity_imaging import extract_boundary_from_image
from phase1_speckle_informed_surface_selection_v2 import das_energy_field, matched_filter_envelope, R_MAX_CELLS
from phase1_speckle_joint_optimization import generate_candidates_with_features, solve_cyclic_dp
from phase1_rotating_transmission_scout import center
import phase2_config as cfg
import labels

if __name__ == "__main__":
    print(f"*** {labels.PENDING_SIGNOFF_BANNER} ***")
    print("JOINT OPTIMIZATION PARAMETER SWEEP (lambda_speckle, kappa_smooth) -- no new simulation.")

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

    candidates_per_angle = []
    for i, theta in enumerate(thetas):
        cands = generate_candidates_with_features(
            water_env[i], homog_env[i], r_outer_est[i], theta, speckle_field, img_rows, img_cols)
        if not cands:
            cands = [{"r": r_outer_est[i] - 6.0, "amp": 0.0, "speckle_score": 0.0}]
        candidates_per_angle.append(cands)

    print(f"\n{'lambda':>8} {'kappa':>8} {'corr':>8} {'RMSE(mm)':>10} {'bias(mm)':>10}")
    results = []
    for lam in [0.0, 0.5, 1.0, 2.0, 4.0]:
        for kap in [0.1, 0.5, 1.0, 2.0, 4.0]:
            r_joint = solve_cyclic_dp(candidates_per_angle, lam, kap, 10.0)
            corr = np.corrcoef(r_joint, true_r_in)[0, 1]
            rmse_mm = np.sqrt(np.mean((r_joint - true_r_in) ** 2)) * cfg.DX_M * 1e3
            bias_mm = np.mean(r_joint - true_r_in) * cfg.DX_M * 1e3
            results.append((lam, kap, corr, rmse_mm, bias_mm))
            print(f"{lam:8.1f} {kap:8.1f} {corr:8.3f} {rmse_mm:10.3f} {bias_mm:+10.3f}")

    best = max(results, key=lambda r: r[2])
    print(f"\nBest by correlation: lambda={best[0]}, kappa={best[1]} -> corr={best[2]:.3f}, RMSE={best[3]:.3f}mm")
    lam0_best = max([r for r in results if r[0] == 0.0], key=lambda r: r[2])
    print(f"Best with lambda=0 (smoothness+amplitude only, NO speckle term): "
          f"kappa={lam0_best[1]} -> corr={lam0_best[2]:.3f}, RMSE={lam0_best[3]:.3f}mm")
    print("(if the overall best beats the lambda=0 best, speckle is adding real value beyond "
          "smoothness alone; if not, smoothness/DP is doing all the work and speckle isn't helping)")
