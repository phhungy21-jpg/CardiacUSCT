"""Fixes a real bug found by inspecting phase1_das_patient023.py's own
figure: the "outermost sufficiently-prominent peak = outer wall"
heuristic grabbed NEAR-PROBE SIDELOBE artifacts (the bright rosette
pattern visible between the true anatomy and the probe ring itself,
same artifact family as the off-center heart test's sidelobe arcs, just
closer to the probe here) instead of the real outer wall, and
mislabeled the TRUE outer-wall peak as "inner" -- visually obvious once
checked: the "DAS inner" markers sat almost exactly on the TRUE OUTER
contour, not the true inner one.

Fix: constrain the boundary search to r_max_cells=108 (0.9x the probe
radius, 120 cells) -- a generic physical constraint (near-probe
self-artifacts cluster closest to the ring itself), NOT informed by
patient023's own specific true contour, so the search stays genuinely
blind.

No new jWave simulation -- reuses the ALREADY-SIMULATED reflection raw
traces (`results/patient023_reflection_raw_traces.npz`) and transmission
rays (`results/patient023_transmission_rays.npz`) from the just-completed
run, pure post-processing.
"""

import numpy as np
from scipy.signal import correlate, hilbert

from phase1_patient023_validation import MRI_NPZ, load_real_contours
from phase1_reflection_channel_scout import (
    thetas, pitch_catch_positions, direction_vector, polar_resample, r_at_theta,
)
from phase1_matched_filter_echo_extraction import _lag_t_arr, _template
from phase1_das_reflectivity_imaging import das_straight_ray_image
from phase1_das_patient023 import extract_two_boundaries_from_image
from phase1_bent_ray_correction import nearest_grid_index, fmm_travel_time_field, _threshold_and_clean
from phase1_rotating_transmission_scout import probe_position, dx, N, center, PROBE_RADIUS_CELLS
import tomography_recon as recon
import phase2_config as cfg
import labels

from matplotlib import pyplot as plt
import os

IMG_SIZE = 150
THRESHOLD_FRACTION = 0.3
R_MAX_CELLS = 0.9 * PROBE_RADIUS_CELLS  # generic physical constraint, not patient-specific


def matched_filter_envelope(trace):
    correlated = correlate(trace, _template, mode="full")
    return np.abs(hilbert(correlated))


if __name__ == "__main__":
    print(f"*** {labels.PENDING_SIGNOFF_BANNER} ***")
    print("DAS PATIENT023 BOUNDARY-EXTRACTION FIX: re-derives both boundaries with r_max_cells "
          f"={R_MAX_CELLS:.0f} (was 140), a generic physical constraint that excludes near-probe "
          "sidelobe artifacts. No new jWave simulation -- reuses cached raw traces.")

    canvas_lv, canvas_myo, outer_contour_dom, inner_contour_dom = load_real_contours(MRI_NPZ)
    ext_theta_out, ext_r_out = polar_resample(outer_contour_dom, center)
    ext_theta_in, ext_r_in = polar_resample(inner_contour_dom, center)
    true_r_out = np.array([r_at_theta(th, ext_theta_out, ext_r_out) for th in thetas])
    true_r_in = np.array([r_at_theta(th, ext_theta_in, ext_r_in) for th in thetas])
    print(f"  true outer radius range: {true_r_out.min():.1f}-{true_r_out.max():.1f} cells "
          f"(R_MAX_CELLS={R_MAX_CELLS:.0f}, probe radius={PROBE_RADIUS_CELLS})")

    d_refl = np.load("results/patient023_reflection_raw_traces.npz")
    water_traces, phantom_traces = d_refl["water_traces"], d_refl["phantom_traces"]
    water_env = [matched_filter_envelope(tr) for tr in water_traces]
    phantom_env = [matched_filter_envelope(tr) for tr in phantom_traces]

    print("\n=== Rebuilding straight-ray DAS image (cached traces, no new sim) ===")
    image_sr, img_rows, img_cols = das_straight_ray_image(phantom_env, water_env, img_size=IMG_SIZE)

    print("=== Rebuilding bent-ray DAS image (cached traces + cached transmission rays) ===")
    d_tx = np.load("results/patient023_transmission_rays.npz")
    pairs_excess_delay_ns = {(tt, tr): v for tt, tr, v in zip(d_tx["theta_tx"], d_tx["theta_rx"], d_tx["excess_delay_ns"])}
    image_sirt, _, _, _ = recon.sirt_reconstruct(pairs_excess_delay_ns, probe_position, IMG_SIZE, N, n_iters=30, relax=0.15)
    is_tissue = _threshold_and_clean(image_sirt, THRESHOLD_FRACTION)
    sound_speed_grid = np.where(is_tissue, cfg.MYOCARDIUM.sound_speed, cfg.WATER.sound_speed).astype(np.float64)
    cell_size_m = ((img_rows[1] - img_rows[0]) * dx[0], (img_cols[1] - img_cols[0]) * dx[0])
    accumulator_br = np.zeros((IMG_SIZE, IMG_SIZE))
    for i, theta in enumerate(thetas):
        src, rcv = pitch_catch_positions(theta)
        sr, sc = nearest_grid_index(src[0], src[1], img_rows, img_cols)
        rr_, rc_ = nearest_grid_index(rcv[0], rcv[1], img_rows, img_cols)
        t_pred = (fmm_travel_time_field(sound_speed_grid, sr, sc, cell_size_m)
                  + fmm_travel_time_field(sound_speed_grid, rr_, rc_, cell_size_m))
        excess_env = np.clip(phantom_env[i] - water_env[i], 0, None)
        sampled = np.interp(t_pred.ravel(), _lag_t_arr, excess_env, left=0.0, right=0.0)
        accumulator_br += sampled.reshape(IMG_SIZE, IMG_SIZE)

    np.savez("results/patient023_das_images.npz", image_straight_ray=image_sr, image_bent_ray=accumulator_br,
             img_rows=img_rows, img_cols=img_cols)
    print("  saved results/patient023_das_images.npz (for reuse)")

    d_rows, d_cols = direction_vector(thetas)
    results = {}
    fig, axes = plt.subplots(1, 2, figsize=(13, 5.5))
    for ax, tag, image in [(axes[0], "straight-ray", image_sr), (axes[1], "bent-ray", accumulator_br)]:
        r_out, r_in = extract_two_boundaries_from_image(
            image, img_rows, img_cols, center, thetas, r_max_cells=R_MAX_CELLS)
        outer_valid, inner_valid = ~np.isnan(r_out), ~np.isnan(r_in)
        outer_rmse = np.sqrt(np.mean(((r_out[outer_valid] - true_r_out[outer_valid]) * cfg.DX_M * 1e3) ** 2))
        print(f"\n--- {tag} DAS result (fixed) ---")
        print(f"  outer boundary: {outer_valid.sum()}/{len(thetas)} detected, RMSE={outer_rmse:.4f}mm")
        if inner_valid.sum() > 0:
            inner_rmse = np.sqrt(np.mean(((r_in[inner_valid] - true_r_in[inner_valid]) * cfg.DX_M * 1e3) ** 2))
            print(f"  INNER boundary: {inner_valid.sum()}/{len(thetas)} detected, RMSE={inner_rmse:.4f}mm")
        else:
            inner_rmse = float("nan")
            print(f"  INNER boundary: 0/{len(thetas)} detected")
        results[tag] = (r_out, r_in, outer_valid, inner_valid, outer_rmse, inner_rmse)

        im = ax.imshow(image, cmap="hot", origin="upper",
                        extent=[img_cols.min(), img_cols.max(), img_rows.max(), img_rows.min()])
        h_row = [center[0] + r * d for r, d in zip(true_r_out, d_rows)]
        h_col = [center[1] + r * d for r, d in zip(true_r_out, d_cols)]
        ax.plot(h_col + [h_col[0]], h_row + [h_row[0]], "c--", linewidth=1.2, label="true outer")
        hi_row = [center[0] + r * d for r, d in zip(true_r_in, d_rows)]
        hi_col = [center[1] + r * d for r, d in zip(true_r_in, d_cols)]
        ax.plot(hi_col + [hi_col[0]], hi_row + [hi_row[0]], "b--", linewidth=1.2, label="true inner")
        ax.scatter([center[1] + r * d_cols[i] for i, r in enumerate(r_out) if outer_valid[i]],
                   [center[0] + r * d_rows[i] for i, r in enumerate(r_out) if outer_valid[i]],
                   c="lime", marker="s", s=15, edgecolor="k", linewidth=0.3, label="DAS outer", zorder=5)
        ax.scatter([center[1] + r * d_cols[i] for i, r in enumerate(r_in) if inner_valid[i]],
                   [center[0] + r * d_rows[i] for i, r in enumerate(r_in) if inner_valid[i]],
                   c="yellow", marker="o", s=15, edgecolor="k", linewidth=0.3, label="DAS inner", zorder=5)
        ax.set_title(f"{tag} DAS, fixed r_max={R_MAX_CELLS:.0f}\n"
                     f"outer RMSE={outer_rmse:.2f}mm, inner {inner_valid.sum()}/36 det., RMSE={inner_rmse:.2f}mm")
        ax.legend(fontsize=6, loc="upper right")

    fig.suptitle("DAS reflectivity imaging, patient023 (FIXED boundary extraction, r_max constrained)")
    labels.add_banner(fig)
    plt.tight_layout()
    os.makedirs("results/figures", exist_ok=True)
    out_fig = "results/figures/phase1_das_patient023_fixed.png"
    plt.savefig(out_fig, dpi=140)
    print(f"\nSaved {out_fig}")
