"""Smoke test (isolated, no existing file modified): does the new
LOCAL-MAXIMUM-ONLY selection rule (`select_best_local_peak`, run -57,
currently only in `phase3_mri_8probe_test.py`) change or harm the
ALREADY-VALIDATED 4-probe results for patient001 and the synthetic
ring phantom, before this fix is ever considered for the official
pipeline?

Per user: "upload a fork to github first, and run those smoke tests"
-- this is the second of the two gaps flagged before porting: confirm
the new selection rule reproduces (not regresses) already-logged
numbers on cases outside the one it was designed for (patient023).

Method: reuse the EXISTING, UNMODIFIED `fit_scale_curvature_weighted`
from `phase3_mri_irregular_ring_reconstruction.py` (returns raw scores,
still using naive global argmax internally for its own reported
answer) to get the SAME score curves used in the already-logged runs,
then apply `select_best_local_peak` (copied here, not imported, to
keep this test isolated and not create a cross-dependency on the
experimental branch's file) to those same scores and compare.
"""

import numpy as np
from scipy.signal import find_peaks

from phase3_mri_irregular_ring_reconstruction import (
    _polar_resample, build_medium_real_contour, fit_scale_curvature_weighted,
    SCALE_GRID, GUARD_BAND_CELLS,
)
from phase3_backprojection_shape_fit_triangle import (
    capture_all_pairs, build_medium_homogeneous, center, dx, N, labels,
)
from phase3_ring_curvature_weighted_fit import fit_circle_radius_curvature_weighted, INNER_R_GRID, OUTER_R_GRID
from phase3_ring_phantom_shapefit import build_medium_ring
import phase3_config as p3cfg


def select_best_local_peak(scale_grid, scores, step_tol=1.5):
    """Copy of run -57's selector (see phase3_mri_8probe_test.py for
    the full docstring/rationale) -- duplicated here deliberately to
    keep this smoke test isolated from the experimental branch's file."""
    diffs = np.diff(scale_grid)
    step = np.median(diffs)
    gap_idx = np.where(diffs > step * step_tol)[0]
    boundaries = [0] + [i + 1 for i in gap_idx] + [len(scale_grid)]
    all_peak_positions, all_peak_scores = [], []
    for lo, hi in zip(boundaries[:-1], boundaries[1:]):
        seg_scores = scores[lo:hi]
        peak_idx, _ = find_peaks(seg_scores)
        for p in peak_idx:
            all_peak_positions.append(lo + p)
            all_peak_scores.append(seg_scores[p])
    if not all_peak_positions:
        best_idx = int(np.argmax(scores))
        return scale_grid[best_idx], False, 0.0
    order = np.argsort(all_peak_scores)[::-1]
    best_pos = all_peak_positions[order[0]]
    confidence = all_peak_scores[order[0]] / all_peak_scores[order[1]] if len(order) > 1 else np.inf
    return scale_grid[best_pos], True, confidence


if __name__ == "__main__":
    print(f"*** {labels.PENDING_SIGNOFF_BANNER} ***")
    print("SMOKE TEST: does local-max-only selection change patient001's real-shape "
          "result (already validated, run -55 corrected numbers) or the synthetic "
          "ring phantom's result (run -45)? Neither case was designed with the tail "
          "artifact in mind -- this checks for silent regressions before considering "
          "porting the fix to the official pipeline.")

    # --- Part 1: patient001 real-shape static reconstruction ---
    print("\n=== Part 1: patient001 real-shape (run -48/-55) ===")
    d = np.load("results/mri_irregular_ring_patient001_slice4.npz")
    lv_mask = d["lv_mask"].astype(bool)
    myo_mask = d["myo_mask"].astype(bool)
    ring_mask = d["ring_mask"].astype(bool)
    outer_contour = d["outer_contour"]
    inner_contour = d["inner_contour"]

    ys, xs = np.where(ring_mask)
    ring_centroid_native = (ys.mean(), xs.mean())
    lv_ys, lv_xs = np.where(lv_mask)
    lv_centroid_native = (lv_ys.mean(), lv_xs.mean())
    offset_row = int(round(center[0] - ring_centroid_native[0]))
    offset_col = int(round(center[1] - ring_centroid_native[1]))

    rows_native, cols_native = np.mgrid[0:myo_mask.shape[0], 0:myo_mask.shape[1]]
    rows_dom, cols_dom = rows_native + offset_row, cols_native + offset_col
    valid = (rows_dom >= 0) & (rows_dom < N[0]) & (cols_dom >= 0) & (cols_dom < N[1])
    canvas_myo = np.zeros(N, dtype=bool)
    canvas_lv = np.zeros(N, dtype=bool)
    canvas_myo[rows_dom[valid], cols_dom[valid]] = myo_mask[valid]
    canvas_lv[rows_dom[valid], cols_dom[valid]] = lv_mask[valid]
    label_map = np.zeros(N, dtype=int)
    label_map[canvas_myo] = 2
    label_map[canvas_lv] = 3

    lv_centroid_dom = (lv_centroid_native[0] + offset_row, lv_centroid_native[1] + offset_col)
    ring_centroid_dom = (ring_centroid_native[0] + offset_row, ring_centroid_native[1] + offset_col)
    inner_contour_dom = inner_contour + np.array([offset_row, offset_col])
    outer_contour_dom = outer_contour + np.array([offset_row, offset_col])
    ext_theta_in, ext_r_in = _polar_resample(inner_contour_dom, lv_centroid_dom)
    ext_theta_out, ext_r_out = _polar_resample(outer_contour_dom, ring_centroid_dom)

    from phase3_mri_irregular_ring_reconstruction import img_rows as default_rows, img_cols as default_cols
    medium = build_medium_real_contour(label_map)
    pairs_real = capture_all_pairs(medium)
    pairs_ref = capture_all_pairs(build_medium_homogeneous())

    s_in_argmax, scores_in = fit_scale_curvature_weighted(pairs_real, ext_theta_in, ext_r_in, SCALE_GRID, lv_centroid_dom, default_rows, default_cols)
    s_in_localmax, in_is_peak, in_conf = select_best_local_peak(SCALE_GRID, scores_in)

    fitted_inner_mean_radius = s_in_argmax * ext_r_in.mean()
    scale_grid_guarded = SCALE_GRID[np.abs(SCALE_GRID * ext_r_out.mean() - fitted_inner_mean_radius) > GUARD_BAND_CELLS]
    s_out_argmax, scores_out = fit_scale_curvature_weighted(pairs_real, ext_theta_out, ext_r_out, scale_grid_guarded, ring_centroid_dom, default_rows, default_cols)
    s_out_localmax, out_is_peak, out_conf = select_best_local_peak(scale_grid_guarded, scores_out)

    print(f"  inner: argmax={s_in_argmax:.3f} (run -55: 0.995) | local-max={s_in_localmax:.3f} (conf={in_conf:.2f}, is_peak={in_is_peak})")
    print(f"  outer: argmax={s_out_argmax:.3f} (run -55: 1.035) | local-max={s_out_localmax:.3f} (conf={out_conf:.2f}, is_peak={out_is_peak})")
    in_err_argmax = abs(s_in_argmax - 1.0) * ext_r_in.mean() * dx[0] * 1e3
    in_err_localmax = abs(s_in_localmax - 1.0) * ext_r_in.mean() * dx[0] * 1e3
    out_err_argmax = abs(s_out_argmax - 1.0) * ext_r_out.mean() * dx[0] * 1e3
    out_err_localmax = abs(s_out_localmax - 1.0) * ext_r_out.mean() * dx[0] * 1e3
    print(f"  inner error: argmax={in_err_argmax:.3f}mm vs local-max={in_err_localmax:.3f}mm")
    print(f"  outer error: argmax={out_err_argmax:.3f}mm vs local-max={out_err_localmax:.3f}mm")

    # --- Part 2: synthetic ring phantom (run -45's 2 frames: ED strong-signal, ES-adjacent weak-signal) ---
    print("\n=== Part 2: synthetic ring phantom (run -45's ED + ES-adjacent frames) ===")
    for inner_R_true, label in [(60.0, "ED frame"), (41.0, "ES-adjacent frame")]:
        outer_R_true = inner_R_true + p3cfg.WALL_THICKNESS_CELLS
        pairs_tri = capture_all_pairs(build_medium_ring(inner_R_true))
        pairs_ref2 = capture_all_pairs(build_medium_homogeneous())

        fitted_inner_argmax, scores_syn_in, conf_inner = fit_circle_radius_curvature_weighted(pairs_tri, INNER_R_GRID)
        fitted_inner_localmax, inner_is_peak_syn, inner_conf_syn = select_best_local_peak(INNER_R_GRID, scores_syn_in)

        outer_grid_guarded = OUTER_R_GRID[np.abs(OUTER_R_GRID - fitted_inner_argmax) > 8.0]
        fitted_outer_argmax, scores_syn_out, conf_outer = fit_circle_radius_curvature_weighted(pairs_tri, outer_grid_guarded)
        fitted_outer_localmax, outer_is_peak_syn, outer_conf_syn = select_best_local_peak(outer_grid_guarded, scores_syn_out)

        print(f"\n  {label}: inner_true={inner_R_true}, outer_true={outer_R_true}")
        print(f"    inner: argmax={fitted_inner_argmax:.1f} | local-max={fitted_inner_localmax:.1f} (is_peak={inner_is_peak_syn})")
        print(f"    outer: argmax={fitted_outer_argmax:.1f} | local-max={fitted_outer_localmax:.1f} (is_peak={outer_is_peak_syn}, conf={outer_conf_syn:.2f})")
        print(f"    inner err: argmax={abs(fitted_inner_argmax-inner_R_true)*dx[0]*1e3:.3f}mm vs "
              f"local-max={abs(fitted_inner_localmax-inner_R_true)*dx[0]*1e3:.3f}mm")
        print(f"    outer err: argmax={abs(fitted_outer_argmax-outer_R_true)*dx[0]*1e3:.3f}mm vs "
              f"local-max={abs(fitted_outer_localmax-outer_R_true)*dx[0]*1e3:.3f}mm")

    print("\n=== Smoke test complete ===")
