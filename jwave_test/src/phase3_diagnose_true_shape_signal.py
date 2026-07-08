"""Diagnostic: does patient023's phase 2 (frac=0.61) inner boundary have
ANY usable acoustic signal, if we fit against its OWN TRUE shape
(instead of the fixed ED template)?

This isolates "is the problem shape-representation (fixable by adding
deformation modes to the template)" from "is the problem an absence of
usable signal at this contraction level (not fixable by any shape
model)". If a clean peak appears at scale=1.0 when fit against the true
shape, deformation modes are worth building. If the curve is STILL
featureless/wrong even against the true shape, the limitation is more
fundamental than template shape.
"""

import numpy as np
from scipy.signal import find_peaks
from scipy.interpolate import RegularGridInterpolator

from phase3_mri_8probe_test import (
    _SRC, _RCV, capture_all_pairs, build_medium_homogeneous, build_medium_real_contour,
    build_search_grid, direction_vector, _polar_resample, r_at_theta, center, dx, N, c_ref,
    t_arr, _ENVELOPE_GROUP_DELAY_S, pair_weight_at_R, labels,
)

from matplotlib import pyplot as plt
import os

PATIENT_ID = "patient023"
PHASE_IDX = 2  # frac=0.61, the failing frame

SCALE_GRID_NARROW = np.arange(0.5, 1.55, 0.01)  # widened after finding the first narrow window (0.85-1.16) was cut off at its own left edge

if __name__ == "__main__":
    print(f"*** {labels.PENDING_SIGNOFF_BANNER} ***")
    print(f"Testing: does phase {PHASE_IDX} (frac=0.61) have usable acoustic signal "
          f"when fit against its OWN TRUE shape, instead of the fixed ED template?")

    d_static = np.load(f"results/mri_irregular_ring_{PATIENT_ID}_slice4.npz")
    d_motion = np.load(f"results/mri_motion_cycle_{PATIENT_ID}_slice4.npz", allow_pickle=True)

    ring_ed = d_static["ring_mask"].astype(bool)
    ys, xs = np.where(ring_ed)
    ring_centroid_native_ed = (ys.mean(), xs.mean())
    offset_row = int(round(center[0] - ring_centroid_native_ed[0]))
    offset_col = int(round(center[1] - ring_centroid_native_ed[1]))

    myo_i = d_motion["myo_frames"][PHASE_IDX]
    lv_i = d_motion["lv_frames"][PHASE_IDX]
    ring_i = d_motion["ring_frames"][PHASE_IDX]
    true_inner_radius_mm = d_motion["true_inner_radius_mm"][PHASE_IDX]
    inner_contour_true = d_motion["inner_contours"][PHASE_IDX]

    rows_native, cols_native = np.mgrid[0:ring_i.shape[0], 0:ring_i.shape[1]]
    rows_dom, cols_dom = rows_native + offset_row, cols_native + offset_col
    valid = (rows_dom >= 0) & (rows_dom < N[0]) & (cols_dom >= 0) & (cols_dom < N[1])
    canvas_myo = np.zeros(N, dtype=bool)
    canvas_lv = np.zeros(N, dtype=bool)
    canvas_myo[rows_dom[valid], cols_dom[valid]] = myo_i[valid]
    canvas_lv[rows_dom[valid], cols_dom[valid]] = lv_i[valid]
    label_map = np.zeros(N, dtype=int)
    label_map[canvas_myo] = 2
    label_map[canvas_lv] = 3

    lys_i, lxs_i = np.where(lv_i)
    lv_centroid_dom = (lys_i.mean() + offset_row, lxs_i.mean() + offset_col)

    # THE TRUE SHAPE at this phase, as the template (not the ED template)
    inner_contour_true_dom = inner_contour_true + np.array([offset_row, offset_col])
    ext_theta_true, ext_r_true = _polar_resample(inner_contour_true_dom, lv_centroid_dom)
    print(f"  true inner mean radius this phase: {ext_r_true.mean():.1f} cells ({true_inner_radius_mm:.2f}mm)")
    print(f"  fitting a scale sweep against the TRUE shape itself -- true scale should be ~1.0 by construction")

    needed_extent = ext_r_true.max() * SCALE_GRID_NARROW.max() + 15.0
    img_rows_g, img_cols_g = build_search_grid(needed_extent)

    medium = build_medium_real_contour(label_map)
    print("\n=== Simulating this phase's phantom (8 transmits) ===")
    pairs_real = capture_all_pairs(medium)
    pairs_ref = capture_all_pairs(build_medium_homogeneous())

    def score_curve(pairs, scale_grid, origin):
        RR, CC = np.meshgrid(img_rows_g, img_cols_g, indexing="ij")
        per_pair_grids = {}
        for (tx, rx), envelope in pairs.items():
            src, rcv = _SRC[tx], _RCV[rx]
            dist_tx = np.sqrt((CC - src[0]) ** 2 + (RR - src[1]) ** 2) * dx[0]
            dist_rx = np.sqrt((CC - rcv[0]) ** 2 + (RR - rcv[1]) ** 2) * dx[0]
            t_total = (dist_tx + dist_rx) / c_ref + _ENVELOPE_GROUP_DELAY_S
            per_pair_grids[(tx, rx)] = np.interp(t_total, t_arr, envelope, left=0, right=0)
        interpolators = {
            key: RegularGridInterpolator((img_rows_g, img_cols_g), np.abs(grid), bounds_error=False, fill_value=0.0)
            for key, grid in per_pair_grids.items()
        }
        N_ANGLES = 144
        thetas = np.linspace(0, 360, N_ANGLES, endpoint=False)
        scores = np.zeros(len(scale_grid))
        for i, s in enumerate(scale_grid):
            d_vals = r_at_theta(thetas, ext_theta_true, ext_r_true) * s
            d_rows, d_cols = direction_vector(thetas)
            pts = np.stack([origin[0] + d_vals * d_rows, origin[1] + d_vals * d_cols], axis=1)
            total = 0.0
            for (tx, rx), interp in interpolators.items():
                w = pair_weight_at_R(tx, rx, np.mean(d_vals))
                total += w * interp(pts).sum()
            scores[i] = total
        return scores

    scores_real = score_curve(pairs_real, SCALE_GRID_NARROW, lv_centroid_dom)
    scores_ref = score_curve(pairs_ref, SCALE_GRID_NARROW, lv_centroid_dom)

    peak_idx, _ = find_peaks(scores_real)
    print(f"\n  genuine local maxima at scales: {SCALE_GRID_NARROW[peak_idx]}")
    if len(peak_idx):
        print(f"  their heights (normalized): {scores_real[peak_idx]/scores_real.max()}")
        prom = (scores_real[peak_idx].max() - scores_real.min()) / (scores_real.max() - scores_real.min() + 1e-12)
        print(f"  best peak prominence: {prom:.2f}")
    else:
        print("  NO genuine local maxima found anywhere in this range either.")

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(SCALE_GRID_NARROW, scores_real / scores_real.max(), label="real phantom (fit vs TRUE shape)")
    ax.plot(SCALE_GRID_NARROW, scores_ref / scores_real.max(), label="homogeneous control", linestyle="--", alpha=0.6)
    ax.axvline(1.0, color="k", linestyle="--", label="true scale=1.0 (true shape used as template)")
    for p in peak_idx:
        ax.plot(SCALE_GRID_NARROW[p], scores_real[p] / scores_real.max(), "ro")
    ax.set_xlabel("candidate scale s (relative to TRUE shape)")
    ax.set_ylabel("normalized score")
    ax.set_title(f"{PATIENT_ID} phase {PHASE_IDX} (frac=0.61): fit vs. TRUE shape template")
    ax.legend(fontsize=8)
    os.makedirs("results/figures", exist_ok=True)
    out_fig = f"results/figures/phase3_diagnose_true_shape_signal_{PATIENT_ID}.png"
    plt.tight_layout()
    plt.savefig(out_fig, dpi=130)
    print(f"\nSaved {out_fig}")
