"""Implements the user's original proposed integration in full: a JOINT
score (surface-fit evidence + speckle-wall-likelihood evidence),
optimized ACROSS candidates AND across neighboring angles (a smoothness
constraint -- a real anatomical contour doesn't jump wildly from one
angle to the next), rather than either criterion alone or an
independent per-angle max-score pick (runs -36/-37, both of which
showed neither criterion alone reliably identifies the right candidate
among the ~6 plausible ones per angle).

score(theta_i, r) = unary(theta_i, r) - kappa * smoothness_penalty(r, r_prev)
unary(theta_i, r) = z(amplitude) + lambda * z(speckle rising-edge score)

Solved via exact dynamic programming over the CYCLIC sequence of 36
angles (not a simple chain -- the ring wraps around, so neighbor(35)=0):
for each candidate choice at angle 0, run forward DP through angles
1..35, then close the loop back to angle 0's fixed choice, and take the
best result over all angle-0 starting choices. With ~6 candidates per
angle and 36 angles this is exact and cheap (no approximation).

No new jWave simulation -- reuses the SAME bias-corrected candidates and
cached data as run -37 (`phase1_speckle_informed_surface_selection_v2.py`),
pure post-processing, for a direct, apples-to-apples comparison against
both the naive and independent-speckle-rescored results.
"""

import numpy as np
from scipy.signal import correlate, hilbert, find_peaks

from phase1_patient023_validation import load_real_contours, MRI_NPZ
from phase1_reflection_channel_scout import (
    thetas, direction_vector, polar_resample, r_at_theta, pitch_catch_positions,
)
from phase1_matched_filter_echo_extraction import _lag_t_arr, _template, PEAK_PROMINENCE_FRACTION
from phase1_das_reflectivity_imaging import extract_boundary_from_image
from phase1_speckle_informed_surface_selection_v2 import (
    time_to_radius_two_leg, das_energy_field, sample_field, matched_filter_envelope,
    R_MAX_CELLS, MIN_SEP_CELLS, EDGE_MARGIN_CELLS,
)
from phase1_rotating_transmission_scout import center, N, dx, PROBE_RADIUS_CELLS
import phase2_config as cfg
import labels

from matplotlib import pyplot as plt
import os

_nonneg = _lag_t_arr >= 0
LAMBDA_SPECKLE = 1.0     # weight of speckle rising-edge term relative to amplitude
KAPPA_SMOOTH = 0.5       # weight of the cross-angle smoothness penalty
SMOOTH_SCALE_CELLS = 10.0  # a jump of this many cells costs 1 unit (same scale as unary terms)


def generate_candidates_with_features(env_water, env_phantom, r_outer_known, theta, speckle_field, img_rows, img_cols):
    thresh = max(env_water[_nonneg].max() * 3.0, env_phantom[_nonneg].max() * PEAK_PROMINENCE_FRACTION)
    peak_idx, _ = find_peaks(env_phantom[_nonneg], height=thresh)
    if len(peak_idx) == 0:
        return []
    times = _lag_t_arr[_nonneg][peak_idx]
    amps = env_phantom[_nonneg][peak_idx]
    radii = time_to_radius_two_leg(times, r_outer_known)
    keep = (radii > 0) & (radii <= r_outer_known - MIN_SEP_CELLS)
    radii, amps = radii[keep], amps[keep]

    d_row, d_col = direction_vector(theta)
    out = []
    for r_cand, amp in zip(radii, amps):
        row_out = center[0] + (r_cand + EDGE_MARGIN_CELLS) * d_row
        col_out = center[1] + (r_cand + EDGE_MARGIN_CELLS) * d_col
        row_in = center[0] + (r_cand - EDGE_MARGIN_CELLS) * d_row
        col_in = center[1] + (r_cand - EDGE_MARGIN_CELLS) * d_col
        e_out = sample_field(speckle_field, img_rows, img_cols, row_out, col_out)
        e_in = sample_field(speckle_field, img_rows, img_cols, row_in, col_in)
        speckle_score = e_out - e_in
        out.append({"r": r_cand, "amp": amp, "speckle_score": speckle_score})
    return out


def solve_cyclic_dp(candidates_per_angle, lambda_speckle, kappa_smooth, smooth_scale):
    n = len(candidates_per_angle)
    all_amps = np.array([c["amp"] for cands in candidates_per_angle for c in cands])
    all_speck = np.array([c["speckle_score"] for cands in candidates_per_angle for c in cands])
    amp_mu, amp_sd = all_amps.mean(), all_amps.std() + 1e-12
    spk_mu, spk_sd = all_speck.mean(), all_speck.std() + 1e-12

    for cands in candidates_per_angle:
        for c in cands:
            z_amp = (c["amp"] - amp_mu) / amp_sd
            z_spk = (c["speckle_score"] - spk_mu) / spk_sd
            c["unary"] = z_amp + lambda_speckle * z_spk

    def pairwise(r1, r2):
        return kappa_smooth * ((r1 - r2) / smooth_scale) ** 2

    best_overall_cost = np.inf
    best_overall_path = None
    for start_k, start_c in enumerate(candidates_per_angle[0]):
        # cost[i][k] = min cost to reach candidate k at angle i, with angle 0 fixed to start_k
        cost = [dict() for _ in range(n)]
        back = [dict() for _ in range(n)]
        cost[0][start_k] = -start_c["unary"]
        for i in range(1, n):
            for k, c in enumerate(candidates_per_angle[i]):
                best_prev_cost, best_prev_k = np.inf, None
                for pk, pcost in cost[i - 1].items():
                    prev_c = candidates_per_angle[i - 1][pk]
                    total = pcost + pairwise(prev_c["r"], c["r"]) - c["unary"]
                    if total < best_prev_cost:
                        best_prev_cost, best_prev_k = total, pk
                if best_prev_k is not None:
                    cost[i][k] = best_prev_cost
                    back[i][k] = best_prev_k
        # close the cycle: angle (n-1) -> angle 0 (fixed to start_k)
        for k, kcost in cost[n - 1].items():
            c_last = candidates_per_angle[n - 1][k]
            total = kcost + pairwise(c_last["r"], start_c["r"])
            if total < best_overall_cost:
                # reconstruct path: back[1][...] always resolves to start_k, since
                # cost[0] contains only that one candidate by construction
                path = [None] * n
                path[n - 1] = k
                for i in range(n - 1, 0, -1):
                    path[i - 1] = back[i][path[i]]
                best_overall_cost = total
                best_overall_path = path

    r_path = np.array([candidates_per_angle[i][best_overall_path[i]]["r"] for i in range(n)])
    return r_path


if __name__ == "__main__":
    print(f"*** {labels.PENDING_SIGNOFF_BANNER} ***")
    print(f"*** {labels.TOY_EXACT_GT_CAPTION} ***")
    print("SPECKLE JOINT OPTIMIZATION: combines surface-fit (amplitude) + speckle rising-edge "
          "evidence into a single score, jointly optimized across candidates AND neighboring "
          f"angles via exact cyclic DP (lambda={LAMBDA_SPECKLE}, kappa={KAPPA_SMOOTH}). "
          "No new jWave simulation -- reuses all cached data from runs -32/-36/-37.")

    canvas_lv, canvas_myo, outer_contour_dom, inner_contour_dom = load_real_contours(MRI_NPZ)
    ext_theta_out, ext_r_out = polar_resample(outer_contour_dom, center)
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

    print("\n=== Building INCOHERENT (energy) speckle field ===")
    speckle_field, img_rows, img_cols = das_energy_field(speckle_env, homog_env)

    print("=== Generating candidates with BOTH surface-fit and speckle features ===")
    candidates_per_angle = []
    for i, theta in enumerate(thetas):
        cands = generate_candidates_with_features(
            water_env[i], homog_env[i], r_outer_est[i], theta, speckle_field, img_rows, img_cols)
        if not cands:
            cands = [{"r": r_outer_est[i] - MIN_SEP_CELLS, "amp": 0.0, "speckle_score": 0.0}]
        candidates_per_angle.append(cands)
    print(f"  mean candidates per angle: {np.mean([len(c) for c in candidates_per_angle]):.1f}")

    print("\n=== Solving joint optimization (exact cyclic DP) ===")
    r_joint = solve_cyclic_dp(candidates_per_angle, LAMBDA_SPECKLE, KAPPA_SMOOTH, SMOOTH_SCALE_CELLS)

    corr = np.corrcoef(r_joint, true_r_in)[0, 1]
    rmse_mm = np.sqrt(np.mean((r_joint - true_r_in) ** 2)) * cfg.DX_M * 1e3
    bias_mm = np.mean(r_joint - true_r_in) * cfg.DX_M * 1e3
    print(f"\n--- Result: joint-optimized (surface-fit + speckle + smoothness) ---")
    print(f"  correlation={corr:.3f}, RMSE={rmse_mm:.3f}mm, mean bias={bias_mm:+.3f}mm")
    print(f"  (compare: run -37 corrected naive strongest-peak: corr=0.272, RMSE=2.146mm)")
    print(f"  (compare: run -37 corrected speckle-rescored (independent, no smoothness): "
          f"corr=0.098, RMSE=1.740mm)")

    if corr > 0.5:
        print("  -> SUBSTANTIAL improvement: joint optimization with smoothness genuinely helps.")
    elif corr > 0.35:
        print("  -> Modest improvement over both prior methods.")
    else:
        print("  -> still no clear win -- smoothness alone did not resolve the identification problem.")

    d_rows, d_cols = direction_vector(thetas)
    fig, axes = plt.subplots(1, 2, figsize=(13, 5.5))
    im = axes[0].imshow(speckle_field, cmap="hot", origin="upper",
                         extent=[img_cols.min(), img_cols.max(), img_rows.max(), img_rows.min()])
    h_row = [center[0] + r_at_theta(th, ext_theta_out, ext_r_out) * direction_vector(th)[0] for th in thetas]
    h_col = [center[1] + r_at_theta(th, ext_theta_out, ext_r_out) * direction_vector(th)[1] for th in thetas]
    axes[0].plot(h_col + [h_col[0]], h_row + [h_row[0]], "c--", linewidth=1.2, label="true outer")
    hi_row = [center[0] + true_r_in[i] * d_rows[i] for i in range(len(thetas))]
    hi_col = [center[1] + true_r_in[i] * d_cols[i] for i in range(len(thetas))]
    axes[0].plot(hi_col + [hi_col[0]], hi_row + [hi_row[0]], "b--", linewidth=1.2, label="true inner")
    j_row = [center[0] + r_joint[i] * d_rows[i] for i in range(len(thetas))]
    j_col = [center[1] + r_joint[i] * d_cols[i] for i in range(len(thetas))]
    axes[0].plot(j_col + [j_col[0]], j_row + [j_row[0]], "lime", linewidth=1.5, label="joint-optimized estimate")
    axes[0].set_title("Incoherent speckle field + joint-optimized contour")
    axes[0].legend(fontsize=7)
    plt.colorbar(im, ax=axes[0], shrink=0.7)

    axes[1].plot(thetas, true_r_in, "k-", linewidth=2, label="true inner contour")
    axes[1].plot(thetas, r_joint, "o-", color="lime", label=f"joint-optimized (corr={corr:.2f}, RMSE={rmse_mm:.2f}mm)")
    axes[1].set_xlabel("angle (deg)")
    axes[1].set_ylabel("inner-boundary radius estimate (cells)")
    axes[1].set_title("Joint-optimized (surface-fit + speckle + smoothness) vs. truth")
    axes[1].legend(fontsize=8)

    fig.suptitle("Speckle-informed surface selection: JOINT optimization (candidates x angles)")
    labels.add_banner(fig)
    plt.tight_layout()
    os.makedirs("results/figures", exist_ok=True)
    out_fig = "results/figures/phase1_speckle_joint_optimization.png"
    plt.savefig(out_fig, dpi=140)
    print(f"\nSaved {out_fig}")
