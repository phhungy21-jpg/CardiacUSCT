"""Fixes run -36's diagnosed cause: candidate radii were generated via
a naive ALL-WATER round-trip conversion, which systematically overshoots
for any echo whose true path partly crosses faster myocardium (the
mechanism characterized back in run -09, ~+7.8 cells) -- this pushed
BOTH the naive and speckle-rescored candidate clouds outward in the
SAME direction, contaminating the rescoring comparison before speckle
ever got a chance to discriminate between candidates.

Fix: use a proper TWO-LEG conversion (water leg from probe to the
ALREADY-VALIDATED outer boundary, run -32's DAS estimate, RMSE=1.04mm --
then a myocardium leg from the outer boundary inward to the candidate)
instead of assuming the whole round trip is water. This correctly
credits the myocardium leg's faster sound speed, which should pull
candidate radii back toward their true physical locations.

Re-runs the EXACT same naive-vs-speckle-rescored comparison as run -36
on these corrected candidates. No new jWave simulation -- reuses all
three cached trace sets (water/homogeneous run -32, speckle run -36)
plus the cached DAS image (`results/patient023_das_images.npz`) for the
outer-boundary estimate, pure post-processing.
"""

import numpy as np
from scipy.signal import correlate, hilbert, find_peaks

from phase1_patient023_validation import load_real_contours, MRI_NPZ
from phase1_reflection_channel_scout import (
    thetas, direction_vector, polar_resample, r_at_theta, pitch_catch_positions,
)
from phase1_matched_filter_echo_extraction import _lag_t_arr, _template, PEAK_PROMINENCE_FRACTION
from phase1_das_reflectivity_imaging import extract_boundary_from_image
from phase1_rotating_transmission_scout import center, N, dx, PROBE_RADIUS_CELLS
import phase2_config as cfg
import labels

from matplotlib import pyplot as plt
import os

IMG_SIZE = 150
R_MAX_CELLS = 0.9 * PROBE_RADIUS_CELLS
MIN_SEP_CELLS = 6.0
EDGE_MARGIN_CELLS = 4.0
_nonneg = _lag_t_arr >= 0


def matched_filter_envelope(trace):
    correlated = correlate(trace, _template, mode="full")
    return np.abs(hilbert(correlated))


def time_to_radius_two_leg(t, r_outer_cells):
    """CORRECTED conversion (fixes run -36): assumes a WATER leg from the
    probe to the already-known outer boundary, then a MYOCARDIUM leg
    from there inward to the candidate -- not a naive all-water round
    trip. `t` is the matched-filter lag time (seconds, no group-delay
    correction, this project's established matched-filter convention)."""
    leg_water_m = (PROBE_RADIUS_CELLS - r_outer_cells) * dx[0]
    t_half = t / 2.0
    leg_tissue_m = cfg.MYOCARDIUM.sound_speed * (t_half - leg_water_m / cfg.WATER.sound_speed)
    r_candidate = r_outer_cells - leg_tissue_m / dx[0]
    return r_candidate


def generate_inner_candidates_corrected(env_water, env_phantom, r_outer_known):
    thresh = max(env_water[_nonneg].max() * 3.0, env_phantom[_nonneg].max() * PEAK_PROMINENCE_FRACTION)
    peak_idx, _ = find_peaks(env_phantom[_nonneg], height=thresh)
    if len(peak_idx) == 0:
        return []
    times = _lag_t_arr[_nonneg][peak_idx]
    amps = env_phantom[_nonneg][peak_idx]
    radii = time_to_radius_two_leg(times, r_outer_known)
    keep = (radii > 0) & (radii <= r_outer_known - MIN_SEP_CELLS)
    return list(zip(radii[keep], amps[keep]))


def das_energy_field(speckle_env, homog_env, img_size=IMG_SIZE):
    img_rows = np.linspace(0, N[0], img_size)
    img_cols = np.linspace(0, N[1], img_size)
    RR, CC = np.meshgrid(img_rows, img_cols, indexing="ij")
    accumulator = np.zeros((img_size, img_size))
    for i, theta in enumerate(thetas):
        src, rcv = pitch_catch_positions(theta)
        dist_src = np.hypot(RR - src[0], CC - src[1]) * dx[0]
        dist_rcv = np.hypot(RR - rcv[0], CC - rcv[1]) * dx[0]
        t_pred = (dist_src + dist_rcv) / cfg.WATER.sound_speed
        excess_env = np.clip(speckle_env[i] - homog_env[i], 0, None)
        sampled = np.interp(t_pred.ravel(), _lag_t_arr, excess_env, left=0.0, right=0.0)
        accumulator += (sampled.reshape(img_size, img_size)) ** 2
    return accumulator, img_rows, img_cols


def sample_field(field, img_rows, img_cols, row, col):
    ri = np.argmin(np.abs(img_rows - row))
    ci = np.argmin(np.abs(img_cols - col))
    return field[ri, ci]


if __name__ == "__main__":
    print(f"*** {labels.PENDING_SIGNOFF_BANNER} ***")
    print(f"*** {labels.TOY_EXACT_GT_CAPTION} ***")
    print("SPECKLE-INFORMED SURFACE SELECTION, v2 (BIAS-CORRECTED): fixes run -36's diagnosed "
          "all-water conversion bias with a proper two-leg (water+myocardium) time-to-radius "
          "conversion, using the already-validated outer boundary (run -32, RMSE=1.04mm). "
          "No new jWave simulation -- reuses all cached traces + the cached DAS image.")

    canvas_lv, canvas_myo, outer_contour_dom, inner_contour_dom = load_real_contours(MRI_NPZ)
    ext_theta_out, ext_r_out = polar_resample(outer_contour_dom, center)
    ext_theta_in, ext_r_in = polar_resample(inner_contour_dom, center)
    true_r_in = np.array([r_at_theta(th, ext_theta_in, ext_r_in) for th in thetas])

    d_das = np.load("results/patient023_das_images.npz")
    image_sr, img_rows_das, img_cols_das = d_das["image_straight_ray"], d_das["img_rows"], d_das["img_cols"]
    r_outer_est = extract_boundary_from_image(image_sr, img_rows_das, img_cols_das, center, thetas, r_max_cells=R_MAX_CELLS)
    print(f"  outer boundary re-used from run -32's validated DAS estimate "
          f"(mean={r_outer_est.mean():.1f} cells, matches true mean={ext_r_out.mean():.1f} cells)")

    d_refl = np.load("results/patient023_reflection_raw_traces.npz")
    water_traces, homog_traces = d_refl["water_traces"], d_refl["phantom_traces"]
    d_speckle = np.load("results/patient023_speckle_raw_traces.npz")
    speckle_traces = d_speckle["speckle_traces"]

    water_env = [matched_filter_envelope(tr) for tr in water_traces]
    homog_env = [matched_filter_envelope(tr) for tr in homog_traces]
    speckle_env = [matched_filter_envelope(tr) for tr in speckle_traces]

    print("\n=== Building INCOHERENT (energy) speckle field for patient023 ===")
    speckle_field, img_rows, img_cols = das_energy_field(speckle_env, homog_env)

    print("=== Generating BIAS-CORRECTED multi-candidate inner-boundary peaks per angle ===")
    d_rows, d_cols = direction_vector(thetas)
    naive_r, speckle_r, n_candidates_per_angle = [], [], []
    for i, theta in enumerate(thetas):
        candidates = generate_inner_candidates_corrected(water_env[i], homog_env[i], r_outer_est[i])
        n_candidates_per_angle.append(len(candidates))
        if not candidates:
            naive_r.append(np.nan)
            speckle_r.append(np.nan)
            continue
        r_naive = max(candidates, key=lambda ra: ra[1])[0]
        naive_r.append(r_naive)

        best_r, best_score = None, -np.inf
        for r_cand, amp in candidates:
            row_out = center[0] + (r_cand + EDGE_MARGIN_CELLS) * d_rows[i]
            col_out = center[1] + (r_cand + EDGE_MARGIN_CELLS) * d_cols[i]
            row_in = center[0] + (r_cand - EDGE_MARGIN_CELLS) * d_rows[i]
            col_in = center[1] + (r_cand - EDGE_MARGIN_CELLS) * d_cols[i]
            e_out = sample_field(speckle_field, img_rows, img_cols, row_out, col_out)
            e_in = sample_field(speckle_field, img_rows, img_cols, row_in, col_in)
            score = e_out - e_in
            if score > best_score:
                best_score, best_r = score, r_cand
        speckle_r.append(best_r)

    naive_r, speckle_r = np.array(naive_r), np.array(speckle_r)
    print(f"  mean candidates per angle: {np.mean(n_candidates_per_angle):.1f}")

    def eval_method(r_est, name):
        valid = ~np.isnan(r_est) & (r_est > 0)
        if valid.sum() < 2:
            print(f"  {name}: too few valid detections to evaluate")
            return None
        corr = np.corrcoef(r_est[valid], true_r_in[valid])[0, 1]
        rmse_mm = np.sqrt(np.mean(((r_est[valid] - true_r_in[valid]) * cfg.DX_M * 1e3) ** 2))
        bias_mm = np.mean((r_est[valid] - true_r_in[valid]) * cfg.DX_M * 1e3)
        print(f"  {name}: {valid.sum()}/{len(thetas)} detected, "
              f"correlation={corr:.3f}, RMSE={rmse_mm:.3f}mm, mean bias={bias_mm:+.3f}mm")
        return corr, rmse_mm

    print("\n--- Result: BIAS-CORRECTED naive vs. speckle-rescored inner-boundary selection ---")
    naive_result = eval_method(naive_r, "corrected naive strongest-peak")
    speckle_result = eval_method(speckle_r, "corrected speckle-rescored")

    print(f"\n  (compare: run -36's UNCORRECTED naive=0.262 corr/1.839mm RMSE, "
          f"speckle-rescored=0.106 corr/1.409mm RMSE)")
    if speckle_result and naive_result and speckle_result[0] > naive_result[0] + 0.1:
        print("  -> with the bias fixed, speckle rescoring NOW IMPROVES over naive amplitude ranking.")
    elif speckle_result and naive_result and speckle_result[0] < naive_result[0] - 0.1:
        print("  -> still WORSE than naive even after the bias fix -- a genuine, separate limitation.")
    else:
        print("  -> no clear improvement either way, even with the bias corrected.")

    fig, axes = plt.subplots(1, 2, figsize=(13, 5.5))
    im = axes[0].imshow(speckle_field, cmap="hot", origin="upper",
                         extent=[img_cols.min(), img_cols.max(), img_rows.max(), img_rows.min()])
    h_row = [center[0] + r_at_theta(th, ext_theta_out, ext_r_out) * direction_vector(th)[0] for th in thetas]
    h_col = [center[1] + r_at_theta(th, ext_theta_out, ext_r_out) * direction_vector(th)[1] for th in thetas]
    axes[0].plot(h_col + [h_col[0]], h_row + [h_row[0]], "c--", linewidth=1.2, label="true outer")
    hi_row = [center[0] + r_at_theta(th, ext_theta_in, ext_r_in) * direction_vector(th)[0] for th in thetas]
    hi_col = [center[1] + r_at_theta(th, ext_theta_in, ext_r_in) * direction_vector(th)[1] for th in thetas]
    axes[0].plot(hi_col + [hi_col[0]], hi_row + [hi_row[0]], "b--", linewidth=1.2, label="true inner")
    axes[0].set_title("Incoherent speckle field, patient023")
    axes[0].legend(fontsize=7)
    plt.colorbar(im, ax=axes[0], shrink=0.7)

    valid_n = ~np.isnan(naive_r)
    valid_s = ~np.isnan(speckle_r)
    axes[1].plot(thetas, true_r_in, "k-", linewidth=2, label="true inner contour")
    axes[1].plot(thetas[valid_n], naive_r[valid_n], "o", color="red", alpha=0.6, label="corrected naive strongest-peak")
    axes[1].plot(thetas[valid_s], speckle_r[valid_s], "s", color="lime", alpha=0.8, label="corrected speckle-rescored")
    axes[1].set_xlabel("angle (deg)")
    axes[1].set_ylabel("inner-boundary radius estimate (cells)")
    axes[1].set_title("Per-angle inner-boundary candidate (bias-corrected conversion)")
    axes[1].legend(fontsize=8)

    fig.suptitle("Speckle-informed surface selection v2: bias-corrected candidates")
    labels.add_banner(fig)
    plt.tight_layout()
    os.makedirs("results/figures", exist_ok=True)
    out_fig = "results/figures/phase1_speckle_informed_surface_selection_v2.png"
    plt.savefig(out_fig, dpi=140)
    print(f"\nSaved {out_fig}")
