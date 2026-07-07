"""Phase 3 — diagnostic: where does run -32's ~8-cell (~0.8mm) systematic
R-undershoot come from?

Run -32's global shape-fit (fit_triangle_radius, sweeping R and summing
accumulator energy along ALL 72 angles of each candidate's predicted
boundary) consistently underestimated the true R by ~7-8.5 cells in
every frame -- small, uniform, flagged as an honest residual but not
root-caused. Two candidate explanations:
(a) the bias is already present in each individual angle's own,
    independently-best echo location (a signal/geometry effect, e.g.
    related to how backprojection combines pairs differently for
    non-probe-aligned directions), or
(b) the bias is an artifact of TYING all 72 angles to one shared R via
    least-squares-like joint optimization -- i.e., the 4 cardinal
    angles (already validated accurate to <0.2mm via independent local
    peak search in run -29) might be getting dragged off their own
    correct answer by the other 68, mostly non-probe-aligned, angles.

This reuses the exact same capture/backprojection machinery (no new
simulation design, one frame -- the ED frame, R=60, same case examined
throughout this thread) and asks two direct, cheap questions:
1. For EACH of the 72 angles independently, what distance maximizes
   that ray's own accumulator profile (ignoring every other angle)? How
   does that compare to what the formula predicts at the KNOWN true R?
   This isolates per-angle bias directly, with no joint-fit coupling.
2. Does restricting the global fit to ONLY the 4 cardinal angles (which
   independent per-axis search already showed to be accurate) reproduce
   run -29's near-zero bias, while adding the other 68 angles back in
   reproduces run -32's ~8-cell undershoot? If so, that identifies the
   non-cardinal angles' diffuse/weak signal as the culprit dragging the
   joint fit, not a flaw in the individual cardinal measurements.
"""

import numpy as np

from phase3_backprojection_shape_fit_triangle import (
    capture_all_pairs, build_medium_triangle, build_medium_homogeneous,
    backproject, direction_vector, ray_triangle_distance, triangle_vertices,
    img_rows, img_cols, center, dx, p3cfg, labels,
)
from scipy.interpolate import RegularGridInterpolator
from matplotlib import pyplot as plt
import os

if __name__ == "__main__":
    print(f"*** {labels.PENDING_SIGNOFF_BANNER} ***")
    print(f"*** {labels.TOY_EXACT_GT_CAPTION} ***")
    print("Diagnostic: isolating the source of run -32's ~8-cell global "
          "shape-fit R-undershoot -- per-angle independent bias vs. "
          "joint-fit coupling artifact. Single ED frame (R=60), reusing "
          "the exact validated capture/backprojection machinery, no new "
          "simulation design.")

    R_true = float(p3cfg.LV_RADIUS_ED_CELLS)  # 60 cells
    print(f"\n=== Capturing pairs: R={R_true} cells (ED) + homogeneous reference ===")
    pairs_ref = capture_all_pairs(build_medium_homogeneous())
    pairs_tri = capture_all_pairs(build_medium_triangle(R_true))
    accumulator_ref = backproject(pairs_ref)
    accumulator_tri = backproject(pairs_tri)
    accumulator_clean = accumulator_tri - accumulator_ref

    interp = RegularGridInterpolator((img_rows, img_cols), np.abs(accumulator_clean),
                                      bounds_error=False, fill_value=0.0)

    N_ANGLES = 72
    thetas = np.linspace(0, 360, N_ANGLES, endpoint=False)
    d_grid = np.arange(5.0, 90.0, 0.25)

    # --- Question 1: per-angle independent bias ---
    print("\n--- Per-angle independent peak distance vs. formula-at-true-R ---")
    independent_d = np.zeros(N_ANGLES)
    expected_d = np.zeros(N_ANGLES)
    bias = np.zeros(N_ANGLES)
    for i, th in enumerate(thetas):
        d_row, d_col = direction_vector(th)
        pts = np.array([(center[0] + d * d_row, center[1] + d * d_col) for d in d_grid])
        vals = interp(pts)
        independent_d[i] = d_grid[np.argmax(vals)]
        expected_d[i] = ray_triangle_distance(th, R_true)
        bias[i] = independent_d[i] - expected_d[i]

    cardinal_idx = [np.argmin(np.abs(thetas - a)) for a in (0, 90, 180, 270)]
    cardinal_names = ["top(0)", "right(90)", "bottom(180)", "left(270)"]
    print("  cardinal angles (validated accurate in run -29):")
    for name, idx in zip(cardinal_names, cardinal_idx):
        print(f"    {name}: independent={independent_d[idx]:.2f}, expected={expected_d[idx]:.2f}, "
              f"bias={bias[idx]:.2f} cells ({bias[idx]*dx[0]*1e3:.2f}mm)")

    non_cardinal_mask = np.ones(N_ANGLES, dtype=bool)
    non_cardinal_mask[cardinal_idx] = False
    print(f"\n  cardinal angles mean bias: {bias[cardinal_idx].mean():.2f} cells "
          f"({bias[cardinal_idx].mean()*dx[0]*1e3:.2f}mm)")
    print(f"  non-cardinal angles mean bias: {bias[non_cardinal_mask].mean():.2f} cells "
          f"({bias[non_cardinal_mask].mean()*dx[0]*1e3:.2f}mm)")
    print(f"  non-cardinal angles bias std: {bias[non_cardinal_mask].std():.2f} cells")
    print(f"  overall (all 72) mean bias: {bias.mean():.2f} cells ({bias.mean()*dx[0]*1e3:.2f}mm)")

    # --- Verification: does the ORIGINAL track_four_directions-style
    # discrete grid-mask search (no interpolation, position-masked half-
    # plane) reproduce run -29's reported left=33.6 for this exact same
    # accumulator? If it does, but my new interpolated free-distance
    # search along the same ray gives something very different, the
    # discrepancy is a search-methodology artifact, not new physics. ---
    col_idx = np.argmin(np.abs(img_cols - center[1]))
    row_idx = np.argmin(np.abs(img_rows - center[0]))
    vert_profile = np.abs(accumulator_clean[:, col_idx])
    horiz_profile = np.abs(accumulator_clean[row_idx, :])
    top_mask = img_rows < center[0]
    bottom_mask = img_rows > center[0]
    left_mask = img_cols < center[1]
    right_mask = img_cols > center[1]
    orig_top = center[0] - img_rows[top_mask][np.argmax(vert_profile[top_mask])]
    orig_bottom = img_rows[bottom_mask][np.argmax(vert_profile[bottom_mask])] - center[0]
    orig_left = center[1] - img_cols[left_mask][np.argmax(horiz_profile[left_mask])]
    orig_right = img_cols[right_mask][np.argmax(horiz_profile[right_mask])] - center[1]
    print("\n--- Verification: original track_four_directions-style search on the SAME accumulator ---")
    print(f"  top={orig_top:.2f}, bottom={orig_bottom:.2f}, left={orig_left:.2f}, right={orig_right:.2f}")
    print(f"  (run -29 reported for this exact R=60 frame: top=31.8, bottom=31.8(!), left=33.6, right=30.0)")
    print(f"  (compare to this diagnostic's new independent free-distance search: "
          f"top={independent_d[cardinal_idx[0]]:.2f}, bottom={independent_d[cardinal_idx[2]]:.2f}, "
          f"left={independent_d[cardinal_idx[3]]:.2f}, right={independent_d[cardinal_idx[1]]:.2f})")

    # --- Are there TWO competing peaks along the left ray, with the
    # coarse native grid (100 pts, ~1.8-cell spacing) happening to
    # under-sample the taller one? Print both candidate heights at fine
    # (0.25-cell) resolution to check directly. ---
    d_row_left, d_col_left = direction_vector(270)
    d_fine = np.arange(5.0, 90.0, 0.05)
    pts_left = np.array([(center[0] + d * d_row_left, center[1] + d * d_col_left) for d in d_fine])
    vals_left = interp(pts_left)
    print("\n--- Left-ray profile: checking for a second, taller competing peak ---")
    near_true = np.abs(d_fine - 33.64) < 3
    near_ghost = np.abs(d_fine - 59.25) < 3
    print(f"  peak height near d=33.6 (original/track_four_directions answer): {vals_left[near_true].max():.6f}")
    print(f"  peak height near d=59.25 (this diagnostic's free-search answer): {vals_left[near_ghost].max():.6f}")
    print(f"  global max over full profile: {vals_left.max():.6f} at d={d_fine[np.argmax(vals_left)]:.2f}")
    # native-grid sample values nearest each candidate, to check for aliasing
    native_near_true_idx = np.argmin(np.abs(img_cols - (center[1] - 33.64)))
    native_near_ghost_idx = np.argmin(np.abs(img_cols - (center[1] - 59.25)))
    print(f"  NATIVE grid (100pt) value nearest d=33.6: {horiz_profile[native_near_true_idx]:.6f} "
          f"(at col={img_cols[native_near_true_idx]:.2f}, i.e. d={center[1]-img_cols[native_near_true_idx]:.2f})")
    print(f"  NATIVE grid (100pt) value nearest d=59.25: {horiz_profile[native_near_ghost_idx]:.6f} "
          f"(at col={img_cols[native_near_ghost_idx]:.2f}, i.e. d={center[1]-img_cols[native_near_ghost_idx]:.2f})")

    # --- Question 2: cardinal-only fit vs full-72 fit ---
    R_GRID = np.arange(25.0, 75.0, 0.25)

    def score_for_angles(angles):
        scores = np.zeros(len(R_GRID))
        for i, R in enumerate(R_GRID):
            pts = []
            for th in angles:
                d = ray_triangle_distance(th, R)
                d_row, d_col = direction_vector(th)
                pts.append((center[0] + d * d_row, center[1] + d * d_col))
            scores[i] = interp(np.array(pts)).sum()
        return scores

    scores_full = score_for_angles(thetas)
    R_fit_full = R_GRID[np.argmax(scores_full)]

    scores_cardinal = score_for_angles(np.array([0, 90, 180, 270]))
    R_fit_cardinal = R_GRID[np.argmax(scores_cardinal)]

    print(f"\n--- Global fit comparison ---")
    print(f"  true R = {R_true:.1f} cells")
    print(f"  fitted R (all 72 angles)      = {R_fit_full:.1f} cells, "
          f"error={abs(R_fit_full-R_true)*dx[0]*1e3:.2f}mm")
    print(f"  fitted R (4 cardinal angles)  = {R_fit_cardinal:.1f} cells, "
          f"error={abs(R_fit_cardinal-R_true)*dx[0]*1e3:.2f}mm")

    # --- Visual: bias vs angle (polar-style line plot) ---
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))
    axes[0].plot(thetas, bias, "b-")
    axes[0].scatter(thetas[cardinal_idx], bias[cardinal_idx], color="red", zorder=5, label="cardinal angles")
    axes[0].axhline(0, color="k", linewidth=0.8)
    axes[0].set_xlabel("angle theta (deg, 0=top/vertex, 90=right, 180=bottom, 270=left)")
    axes[0].set_ylabel("independent peak bias (cells)")
    axes[0].set_title("Per-angle independent bias (ED frame, R=60)")
    axes[0].legend()

    axes[1].plot(R_GRID, scores_full / scores_full.max(), label="all 72 angles")
    axes[1].plot(R_GRID, scores_cardinal / scores_cardinal.max(), label="4 cardinal angles only")
    axes[1].axvline(R_true, color="k", linestyle="--", label=f"true R={R_true:.0f}")
    axes[1].set_xlabel("candidate R (cells)")
    axes[1].set_ylabel("normalized score")
    axes[1].set_title("Global fit score curve: full vs. cardinal-only")
    axes[1].legend()

    labels.add_banner(fig)
    plt.tight_layout()
    os.makedirs("results/figures", exist_ok=True)
    plt.savefig("results/figures/phase3_shape_fit_bias_diagnostic.png", dpi=130)
    print("\nSaved results/figures/phase3_shape_fit_bias_diagnostic.png")
