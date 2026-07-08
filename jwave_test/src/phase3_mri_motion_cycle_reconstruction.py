"""Phase 3 — acoustic reconstruction of the REAL, registration-derived
motion cycle (`phase3_mri_motion_cycle_prep.py`), the first test in this
thread where ground truth is imperfect real motion, not an exactly
prescribed toy/real-shape value.

DESIGN (per the prep script's docstring): the scale-factor fit is
against the FIXED ED real-shape template (r(theta) from the smoothed
ED contour, `phase3_mri_irregular_ring_prep.py`/run -47's output) --
the SAME template at every phase, deliberately NOT re-derived per
frame. This tests whether a single scale parameter against a fixed
shape can track genuinely non-uniform real wall motion, rather than
trivially confirming a per-frame-refit shape. Each frame's own
(currently-true) centroid is used as that frame's ray-sweep origin --
consistent with this thread's established "each boundary uses its own
known center" convention (runs -46/-48), not blind joint fitting.

GROUND-TRUTH FLOOR: unlike every previous test in this thread, "true"
motion here is Phase I's registration output (mean_dice=0.784 for this
patient, below the pilot's own 0.80 Gate-3 threshold) -- see
`labels.GT_FLOOR_CAPTION`, applied throughout.
"""

import sys
import glob

import numpy as np
from scipy.interpolate import RegularGridInterpolator

from jax import numpy as jnp
from jwave import FourierSeries
from jwave.geometry import Medium

import phase2_config as cfg
from phase3_backprojection_shape_fit_triangle import (
    capture_all_pairs, build_medium_homogeneous, direction_vector,
    _SRC, _RCV, t_arr, c_ref, _ENVELOPE_GROUP_DELAY_S,
    img_rows, img_cols, center, dx, N, domain, labels,
)
from phase3_ring_curvature_weighted_fit import pair_weight_at_R
from phase3_mri_irregular_ring_reconstruction import (
    _polar_resample, r_at_theta, build_medium_real_contour, build_search_grid,
    fit_scale_curvature_weighted, N_ANGLES, _THETAS, SCALE_GRID, GUARD_BAND_CELLS,
)

from matplotlib import pyplot as plt
import os

PATIENT_ID = sys.argv[1] if len(sys.argv) > 1 else "patient001"
_static_matches = glob.glob(f"results/mri_irregular_ring_{PATIENT_ID}_slice*.npz")
_motion_matches = glob.glob(f"results/mri_motion_cycle_{PATIENT_ID}_slice*.npz")
if not _static_matches or not _motion_matches:
    raise FileNotFoundError(f"missing prep npz for {PATIENT_ID} -- run the prep scripts first")
MRI_STATIC_NPZ = _static_matches[0]  # ED template (run -47/-48)
MOTION_NPZ = _motion_matches[0]  # 8-phase real motion (this thread)

if __name__ == "__main__":
    print(f"*** {labels.PENDING_SIGNOFF_BANNER} ***")
    print(f"*** {labels.GT_FLOOR_CAPTION} ***")
    print(f"Acoustic reconstruction of the REAL, registration-derived 8-phase "
          f"motion cycle (ACDC {PATIENT_ID}). Fixed-ED-template "
          f"scale-factor fit, curvature-weighted + guard-band (run -46/-48's "
          f"validated method), testing whether a single scale parameter "
          f"tracks genuinely non-uniform real wall motion.")

    d_static = np.load(MRI_STATIC_NPZ)
    d_motion = np.load(MOTION_NPZ, allow_pickle=True)

    myo_frames = d_motion["myo_frames"]
    lv_frames = d_motion["lv_frames"]
    ring_frames = d_motion["ring_frames"]
    outer_contours_true = d_motion["outer_contours"]
    inner_contours_true = d_motion["inner_contours"]
    true_inner_radius_mm = d_motion["true_inner_radius_mm"]
    true_outer_radius_mm = d_motion["true_outer_radius_mm"]
    phases = d_motion["phases"]
    fractions = d_motion["fractions"]
    N_PHASES = len(phases)
    print(f"  registration quality: mean_dice={float(d_motion['mean_dice']):.3f} "
          f"(myo={float(d_motion['myo_dice']):.3f}, lv={float(d_motion['lv_dice']):.3f})")

    # Fixed ED template (run -47/-48): same placement offset, same r(theta)
    # template used as the scale-fit's shape family at EVERY phase.
    ring_ed = d_static["ring_mask"].astype(bool)
    lv_ed = d_static["lv_mask"].astype(bool)
    ys, xs = np.where(ring_ed)
    ring_centroid_native_ed = (ys.mean(), xs.mean())
    lys, lxs = np.where(lv_ed)
    lv_centroid_native_ed = (lys.mean(), lxs.mean())
    offset_row = int(round(center[0] - ring_centroid_native_ed[0]))
    offset_col = int(round(center[1] - ring_centroid_native_ed[1]))
    print(f"  fixed placement offset (from ED ring centroid): ({offset_row}, {offset_col})")

    outer_contour_ed_dom = d_static["outer_contour"] + np.array([offset_row, offset_col])
    inner_contour_ed_dom = d_static["inner_contour"] + np.array([offset_row, offset_col])
    lv_centroid_ed_dom = (lv_centroid_native_ed[0] + offset_row, lv_centroid_native_ed[1] + offset_col)
    ring_centroid_ed_dom = (ring_centroid_native_ed[0] + offset_row, ring_centroid_native_ed[1] + offset_col)
    ext_theta_in, ext_r_in = _polar_resample(inner_contour_ed_dom, lv_centroid_ed_dom)
    ext_theta_out, ext_r_out = _polar_resample(outer_contour_ed_dom, ring_centroid_ed_dom)
    ed_mean_r_in = ext_r_in.mean()
    ed_mean_r_out = ext_r_out.mean()
    print(f"  fixed ED template: mean inner radius={ed_mean_r_in:.1f} cells, mean outer radius={ed_mean_r_out:.1f} cells")

    # Use MAX (not mean) contour radius -- a real, non-circular contour
    # extends further at some angles than its mean, and using the mean
    # here under-sizes the grid relative to the static-shape script's
    # (correct) convention, silently risking clipping at the sweep's
    # largest candidate scales even when the mean-based check "passes".
    needed_extent = max(ext_r_out.max(), ext_r_in.max()) * SCALE_GRID.max() + 15.0
    if needed_extent > (img_rows.max() - center[0]):
        img_rows_g, img_cols_g = build_search_grid(needed_extent)
        print(f"  default search grid too small for this patient's anatomy (needs +/-{needed_extent:.0f} cells) "
              f"-- using a wider {len(img_rows_g)}x{len(img_cols_g)} grid instead")
    else:
        img_rows_g, img_cols_g = img_rows, img_cols

    print("\n=== Simulating homogeneous reference (shared across all phases) ===")
    pairs_ref = capture_all_pairs(build_medium_homogeneous())

    rows_native, cols_native = np.mgrid[0:ring_frames.shape[1], 0:ring_frames.shape[2]]
    rows_dom, cols_dom = rows_native + offset_row, cols_native + offset_col
    valid = (rows_dom >= 0) & (rows_dom < N[0]) & (cols_dom >= 0) & (cols_dom < N[1])

    fitted_inner_scale, fitted_outer_scale = [], []
    fitted_inner_r_mm, fitted_outer_r_mm = [], []
    frame_centroids_lv, frame_centroids_ring = [], []
    accumulators = []

    for i in range(N_PHASES):
        print(f"\n=== Phase {i+1}/{N_PHASES} (frac={fractions[i]:.2f}) ===")
        myo_i, lv_i, ring_i = myo_frames[i], lv_frames[i], ring_frames[i]

        canvas_myo = np.zeros(N, dtype=bool)
        canvas_lv = np.zeros(N, dtype=bool)
        canvas_myo[rows_dom[valid], cols_dom[valid]] = myo_i[valid]
        canvas_lv[rows_dom[valid], cols_dom[valid]] = lv_i[valid]
        label_map = np.zeros(N, dtype=int)
        label_map[canvas_myo] = 2
        label_map[canvas_lv] = 3

        lys_i, lxs_i = np.where(lv_i)
        ys_i, xs_i = np.where(ring_i)
        lv_centroid_dom = (lys_i.mean() + offset_row, lxs_i.mean() + offset_col)
        ring_centroid_dom = (ys_i.mean() + offset_row, xs_i.mean() + offset_col)
        frame_centroids_lv.append(lv_centroid_dom)
        frame_centroids_ring.append(ring_centroid_dom)

        medium = build_medium_real_contour(label_map)
        pairs_real = capture_all_pairs(medium)

        s_in, _ = fit_scale_curvature_weighted(pairs_real, ext_theta_in, ext_r_in, SCALE_GRID, lv_centroid_dom, img_rows_g, img_cols_g)
        fitted_inner_mean_radius = s_in * ed_mean_r_in
        scale_grid_guarded = SCALE_GRID[np.abs(SCALE_GRID * ed_mean_r_out - fitted_inner_mean_radius) > GUARD_BAND_CELLS]
        s_out, _ = fit_scale_curvature_weighted(pairs_real, ext_theta_out, ext_r_out, scale_grid_guarded, ring_centroid_dom, img_rows_g, img_cols_g)

        fitted_inner_scale.append(s_in)
        fitted_outer_scale.append(s_out)
        fitted_inner_r_mm.append(s_in * ed_mean_r_in * dx[0] * 1e3)
        fitted_outer_r_mm.append(s_out * ed_mean_r_out * dx[0] * 1e3)

        RR_full, CC_full = np.meshgrid(img_rows_g, img_cols_g, indexing="ij")
        accumulator = np.zeros(RR_full.shape)
        for (tx, rx), envelope in pairs_real.items():
            src, rcv = _SRC[tx], _RCV[rx]
            dist_tx = np.sqrt((CC_full - src[0]) ** 2 + (RR_full - src[1]) ** 2) * dx[0]
            dist_rx = np.sqrt((CC_full - rcv[0]) ** 2 + (RR_full - rcv[1]) ** 2) * dx[0]
            t_total = (dist_tx + dist_rx) / c_ref + _ENVELOPE_GROUP_DELAY_S
            accumulator += np.interp(t_total, t_arr, envelope, left=0, right=0)
        accumulator_ref = np.zeros(RR_full.shape)
        for (tx, rx), envelope in pairs_ref.items():
            src, rcv = _SRC[tx], _RCV[rx]
            dist_tx = np.sqrt((CC_full - src[0]) ** 2 + (RR_full - src[1]) ** 2) * dx[0]
            dist_rx = np.sqrt((CC_full - rcv[0]) ** 2 + (RR_full - rcv[1]) ** 2) * dx[0]
            t_total = (dist_tx + dist_rx) / c_ref + _ENVELOPE_GROUP_DELAY_S
            accumulator_ref += np.interp(t_total, t_arr, envelope, left=0, right=0)
        accumulators.append(accumulator - accumulator_ref)

        in_err_mm = abs(fitted_inner_r_mm[-1] - true_inner_radius_mm[i])
        out_err_mm = abs(fitted_outer_r_mm[-1] - true_outer_radius_mm[i])
        print(f"  inner: true_r={true_inner_radius_mm[i]:.2f}mm fitted_r={fitted_inner_r_mm[-1]:.2f}mm err={in_err_mm:.2f}mm")
        print(f"  outer: true_r={true_outer_radius_mm[i]:.2f}mm fitted_r={fitted_outer_r_mm[-1]:.2f}mm err={out_err_mm:.2f}mm")

    inner_errs = np.abs(np.array(fitted_inner_r_mm) - true_inner_radius_mm)
    outer_errs = np.abs(np.array(fitted_outer_r_mm) - true_outer_radius_mm)
    print(f"\n--- RMSE across {N_PHASES} real-motion phases ---")
    print(f"  inner boundary RMSE={np.sqrt(np.mean(inner_errs**2)):.4f}mm  (per-phase: {np.round(inner_errs,2).tolist()})")
    print(f"  outer boundary RMSE={np.sqrt(np.mean(outer_errs**2)):.4f}mm  (per-phase: {np.round(outer_errs,2).tolist()})")
    print(f"  (recall registration floor: median ~1 voxel/1.5mm, up to ~3-4.5mm for high-contraction patients)")
    print(f"  true inner-radius range this cycle: {true_inner_radius_mm.min():.2f}-{true_inner_radius_mm.max():.2f}mm "
          f"(span={true_inner_radius_mm.max()-true_inner_radius_mm.min():.2f}mm -- the actual signal being tracked)")

    # --- Figure: 8-frame filmstrip, true (real, non-uniform) vs fitted (scaled fixed ED template) contour ---
    n_cols = 4
    n_rows = int(np.ceil(N_PHASES / n_cols))
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(3.4 * n_cols, 3.6 * n_rows))
    axes = np.array(axes).reshape(-1)
    vmax = max(np.abs(a).max() for a in accumulators)
    for i, ax in enumerate(axes[:N_PHASES]):
        ax.imshow(np.abs(accumulators[i]), cmap="hot", vmin=0, vmax=vmax, origin="upper",
                  extent=[img_cols_g.min(), img_cols_g.max(), img_rows_g.max(), img_rows_g.min()])
        true_in = inner_contours_true[i] + np.array([offset_row, offset_col])
        true_out = outer_contours_true[i] + np.array([offset_row, offset_col])
        ax.plot(true_in[:, 1], true_in[:, 0], "c--", linewidth=1, label="true inner")
        ax.plot(true_out[:, 1], true_out[:, 0], "c-", linewidth=1, label="true outer")

        s_in, s_out = fitted_inner_scale[i], fitted_outer_scale[i]
        lv_c, ring_c = frame_centroids_lv[i], frame_centroids_ring[i]
        fit_in_pts = np.array([
            (lv_c[0] + r_at_theta(th, ext_theta_in, ext_r_in) * s_in * direction_vector(th)[0],
             lv_c[1] + r_at_theta(th, ext_theta_in, ext_r_in) * s_in * direction_vector(th)[1]) for th in _THETAS])
        fit_out_pts = np.array([
            (ring_c[0] + r_at_theta(th, ext_theta_out, ext_r_out) * s_out * direction_vector(th)[0],
             ring_c[1] + r_at_theta(th, ext_theta_out, ext_r_out) * s_out * direction_vector(th)[1]) for th in _THETAS])
        ax.plot(fit_in_pts[:, 1], fit_in_pts[:, 0], "g:", linewidth=1.5, label="fitted inner")
        ax.plot(fit_out_pts[:, 1], fit_out_pts[:, 0], "g-.", linewidth=1.5, label="fitted outer")

        ax.set_title(f"phase={phases[i]:.2f}, frac={fractions[i]:.2f}\n"
                     f"in_err={inner_errs[i]:.2f}mm out_err={outer_errs[i]:.2f}mm", fontsize=8)
        ax.axis("off")
        if i == 0:
            ax.legend(fontsize=6, loc="lower left")
    for ax in axes[N_PHASES:]:
        ax.axis("off")
    fig.suptitle(f"Real registration-derived motion cycle (ACDC {PATIENT_ID})\n"
                 "cyan = true (non-uniform real deformation), green = fitted (scale x FIXED ED template)\n"
                 "single scale parameter vs. genuinely non-uniform real wall motion", fontsize=10)
    plt.tight_layout(rect=[0, 0.03, 1, 0.87])
    labels.add_banner(fig, caption=labels.GT_FLOOR_CAPTION)
    os.makedirs("results/figures", exist_ok=True)
    out_fig = f"results/figures/phase3_mri_motion_cycle_reconstruction_{PATIENT_ID}.png"
    plt.savefig(out_fig, dpi=130)
    print(f"\nSaved {out_fig}")
