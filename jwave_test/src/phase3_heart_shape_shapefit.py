"""Phase 3 — generalizing the GLOBAL shape-fit principle (run -32) to
the non-convex heart phantom, per the run -37 finding that pair-
exclusion patches (run -34's `backproject_no_adjacent`) do not
generalize past the triangle's 3-corner case.

Run -32 showed that sweeping a known parametric shape family and
scoring each candidate by integrating accumulator energy along its
ENTIRE predicted boundary (not picking independent local peaks) is
naturally robust to localized ghosts, because a single bad sector gets
diluted by the many OTHER, unaffected angles voting correctly -- this
worked even on the plain NAIVE accumulator, no pair exclusion needed.
Run -37's heart-shape test never tried this: it only tested naive vs.
the triangle-tuned pair-exclusion heuristic, both of which failed. This
script closes that gap: the heart's shape family is fully known (same
10-vertex polygon as run -37, one free parameter R, same ED/ES
schedule) -- so the SAME 1-parameter global template-match approach
used for the triangle applies directly, generalized to a non-convex
polygon via a proper multi-edge ray intersection (collecting ALL
crossings and keeping the nearest one, since concave boundaries can in
principle have more than one).

Uses the known true center (SHIFTED_CENTER from run -37) as the ray
origin -- this tests whether the SHAPE-FIT PRINCIPLE handles a concave,
4-corner boundary, without also solving the (separate, harder) joint
position+size fitting problem in the same step.
"""

import numpy as np
from scipy.interpolate import RegularGridInterpolator

from phase3_backprojection_shape_fit_triangle import (
    capture_all_pairs, build_medium_homogeneous, backproject,
    backproject_no_adjacent, direction_vector, img_rows, img_cols,
    center, dx, p3cfg, labels,
)
from phase3_heart_shape_offcenter_test import (
    heart_vertices, build_medium_heart, HEART_UNIT_VERTICES,
    OFFSET, SHIFTED_CENTER,
)

from matplotlib import pyplot as plt
import os


def _ray_segment_intersection(origin, d_row, d_col, p1, p2):
    ax, ay = origin
    bx, by = p1
    ex, ey = p2[0] - p1[0], p2[1] - p1[1]
    denom = d_row * ey - d_col * ex
    if abs(denom) < 1e-9:
        return None
    t = ((bx - ax) * ey - (by - ay) * ex) / denom
    s = ((bx - ax) * d_col - (by - ay) * d_row) / denom
    if t > 0 and 0 <= s <= 1:
        return t
    return None


def ray_heart_distance(theta_deg, R, origin=SHIFTED_CENTER):
    """Generalizes ray_triangle_distance to the 10-vertex (possibly
    concave) heart polygon: collect ALL valid edge crossings along the
    ray, return the NEAREST one (the first boundary the ray exits
    through from an interior origin) -- correct for both convex and
    concave polygons, unlike assuming exactly one crossing exists."""
    verts = heart_vertices(R)
    d_row, d_col = direction_vector(theta_deg)
    n = len(verts)
    candidates = []
    for i in range(n):
        p1, p2 = verts[i], verts[(i + 1) % n]
        t = _ray_segment_intersection(origin, d_row, d_col, p1, p2)
        if t is not None:
            candidates.append(t)
    if not candidates:
        raise ValueError(f"no valid intersection at theta={theta_deg}, R={R}")
    return min(candidates)


# Sanity check: the notch (theta=0, straight up from origin) should sit
# at the notch's own dy=0.40 -- verify before trusting the general
# function, established practice this thread.
for _R_check in (40.0, 50.0, 60.0):
    _d = ray_heart_distance(0, _R_check)
    assert abs(_d - 0.40 * _R_check) < 1e-6, f"notch check failed: {_d} vs {0.40*_R_check}"
    _d_down = ray_heart_distance(180, _R_check)
    assert abs(_d_down - 1.00 * _R_check) < 1e-6, f"bottom-tip check failed: {_d_down} vs {_R_check}"

N_ANGLES = 144  # finer than the triangle's 72, given the more complex 10-vertex shape
_THETAS = np.linspace(0, 360, N_ANGLES, endpoint=False)


def fit_heart_radius(accumulator, R_grid):
    interp = RegularGridInterpolator((img_rows, img_cols), np.abs(accumulator),
                                      bounds_error=False, fill_value=0.0)
    scores = np.zeros(len(R_grid))
    for i, R in enumerate(R_grid):
        pts = []
        for th in _THETAS:
            d = ray_heart_distance(th, R)
            d_row, d_col = direction_vector(th)
            pts.append((SHIFTED_CENTER[0] + d * d_row, SHIFTED_CENTER[1] + d * d_col))
        scores[i] = interp(np.array(pts)).sum()
    best_R = R_grid[np.argmax(scores)]
    return best_R, scores


R_GRID = np.arange(25.0, 75.0, 0.25)


if __name__ == "__main__":
    print(f"*** {labels.PENDING_SIGNOFF_BANNER} ***")
    print(f"*** {labels.TOY_EXACT_GT_CAPTION} ***")
    print("Generalizing the GLOBAL shape-fit (run -32) to the non-convex, "
          "off-center heart phantom (run -37), which pair-exclusion "
          "patching failed on. Same 1-parameter (R) template-match "
          "principle: sweep candidate R, integrate accumulator energy "
          "along the ENTIRE predicted 10-vertex boundary (144 angles), "
          "on both naive and no-adjacent-pairs accumulators.")

    print("\n=== Control: homogeneous medium (no heart) ===")
    pairs_ref = capture_all_pairs(build_medium_homogeneous())
    accumulator_ref_naive = backproject(pairs_ref)
    accumulator_ref_noadj = backproject_no_adjacent(pairs_ref)
    ref_R_naive, _ = fit_heart_radius(accumulator_ref_naive, R_GRID)
    ref_R_noadj, _ = fit_heart_radius(accumulator_ref_noadj, R_GRID)
    print(f"  homogeneous-medium fitted R: naive={ref_R_naive:.1f}, no-adjacent={ref_R_noadj:.1f} cells "
          f"(should be meaningless/low-confidence)")

    N_FRAMES_MOVIE = 8
    phases = np.linspace(0, 1, N_FRAMES_MOVIE)
    radii_cells = [p3cfg.lv_radius_at_phase(p) for p in phases]

    frames_naive = []
    frames_noadj = []
    all_fitted_R_naive = []
    all_fitted_R_noadj = []
    ed_scores_naive = None
    ed_scores_noadj = None
    for i, R in enumerate(radii_cells):
        print(f"=== Frame {i+1}/{N_FRAMES_MOVIE} (R={R:.1f} cells = {R*dx[0]*1e3:.2f}mm) ===")
        pairs = capture_all_pairs(build_medium_heart(R))

        accumulator_clean_naive = backproject(pairs) - accumulator_ref_naive
        frames_naive.append(accumulator_clean_naive)
        fitted_R_naive, scores_naive = fit_heart_radius(accumulator_clean_naive, R_GRID)
        all_fitted_R_naive.append(fitted_R_naive)

        accumulator_clean_noadj = backproject_no_adjacent(pairs) - accumulator_ref_noadj
        frames_noadj.append(accumulator_clean_noadj)
        fitted_R_noadj, scores_noadj = fit_heart_radius(accumulator_clean_noadj, R_GRID)
        all_fitted_R_noadj.append(fitted_R_noadj)

        if i == 0:
            ed_scores_naive = scores_naive
            ed_scores_noadj = scores_noadj

        err_naive_mm = abs(fitted_R_naive - R) * dx[0] * 1e3
        err_noadj_mm = abs(fitted_R_noadj - R) * dx[0] * 1e3
        print(f"  true R={R:.1f} | naive shape-fit R={fitted_R_naive:.1f} err={err_naive_mm:.2f}mm "
              f"| no-adj shape-fit R={fitted_R_noadj:.1f} err={err_noadj_mm:.2f}mm")

    print("\n--- Global R-fit RMSE across all frames ---")
    errs_naive = np.array([abs(all_fitted_R_naive[i] - radii_cells[i]) * dx[0] * 1e3 for i in range(N_FRAMES_MOVIE)])
    errs_noadj = np.array([abs(all_fitted_R_noadj[i] - radii_cells[i]) * dx[0] * 1e3 for i in range(N_FRAMES_MOVIE)])
    print(f"  naive shape-fit       R RMSE={np.sqrt(np.mean(errs_naive**2)):.4f}mm  (per-frame: {np.round(errs_naive,2).tolist()})")
    print(f"  no-adjacent shape-fit R RMSE={np.sqrt(np.mean(errs_noadj**2)):.4f}mm  (per-frame: {np.round(errs_noadj,2).tolist()})")
    print(f"\n  (for reference, run -37's naive/no-adjacent axis-based readouts did not recover the boundary at all)")

    # --- Figure 1: score(R) vs R, ED frame, naive vs no-adjacent ---
    fig1, ax1 = plt.subplots(figsize=(7, 4.5))
    ax1.plot(R_GRID, ed_scores_naive / ed_scores_naive.max(), label="naive, all 16 pairs")
    ax1.plot(R_GRID, ed_scores_noadj / ed_scores_noadj.max(), label="no-adjacent-pairs")
    ax1.axvline(radii_cells[0], color="k", linestyle="--", label=f"true R={radii_cells[0]:.1f}")
    ax1.axvline(all_fitted_R_naive[0], color="C0", linestyle=":", label=f"naive fit={all_fitted_R_naive[0]:.1f}")
    ax1.axvline(all_fitted_R_noadj[0], color="C1", linestyle=":", label=f"no-adj fit={all_fitted_R_noadj[0]:.1f}")
    ax1.set_xlabel("candidate R (cells)")
    ax1.set_ylabel("normalized total boundary energy")
    ax1.set_title("ED frame: heart shape-fit score vs. candidate R")
    ax1.legend(fontsize=8)
    labels.add_banner(fig1)
    plt.tight_layout()
    os.makedirs("results/figures", exist_ok=True)
    plt.savefig("results/figures/phase3_heart_shape_shapefit_score_curve.png", dpi=130)
    print("\nSaved results/figures/phase3_heart_shape_shapefit_score_curve.png")

    # --- Figure 2: full 8-frame filmstrip, no-adjacent-pairs variant, true+fitted heart overlaid ---
    n_cols = 4
    n_rows = int(np.ceil(N_FRAMES_MOVIE / n_cols))
    fig2, axes2 = plt.subplots(n_rows, n_cols, figsize=(3.2 * n_cols, 3.4 * n_rows))
    axes2 = np.array(axes2).reshape(-1)
    vmax = max(np.abs(f).max() for f in frames_noadj)
    for i, (ax, frame, R) in enumerate(zip(axes2, frames_noadj, radii_cells)):
        ax.imshow(np.abs(frame), cmap="hot", vmin=0, vmax=vmax, origin="upper",
                  extent=[img_cols.min(), img_cols.max(), img_rows.max(), img_rows.min()])
        verts = heart_vertices(R)
        h_row = [v[0] for v in verts] + [verts[0][0]]
        h_col = [v[1] for v in verts] + [verts[0][1]]
        ax.plot(h_col, h_row, "c--", linewidth=1, alpha=0.7, label="true")

        fitted_R = all_fitted_R_noadj[i]
        fverts = heart_vertices(fitted_R)
        f_row = [v[0] for v in fverts] + [fverts[0][0]]
        f_col = [v[1] for v in fverts] + [fverts[0][1]]
        ax.plot(f_col, f_row, "g:", linewidth=1.5, alpha=0.9, label="fitted")

        err_mm = abs(fitted_R - R) * dx[0] * 1e3
        ax.set_title(f"phase={phases[i]:.2f}, R_true={R*dx[0]*1e3:.2f}mm\nerr={err_mm:.2f}mm", fontsize=8)
        ax.axis("off")
    for ax in axes2[len(frames_noadj):]:
        ax.axis("off")
    fig2.suptitle("Heart shape-fit (no-adjacent-pairs accumulator)\n"
                "cyan dashed = true, green dotted = globally-fitted heart (1-param R sweep)\n"
                "(TOY: exact prescribed ground truth; known center, R only)", fontsize=10)
    plt.tight_layout(rect=[0, 0.02, 1, 0.86])
    labels.add_banner(fig2)
    plt.savefig("results/figures/phase3_heart_shape_shapefit_filmstrip.png", dpi=130)
    print("\nSaved results/figures/phase3_heart_shape_shapefit_filmstrip.png")

    # --- Figure 3: same filmstrip, NAIVE accumulator (the actually-good result) ---
    fig3, axes3 = plt.subplots(n_rows, n_cols, figsize=(3.2 * n_cols, 3.4 * n_rows))
    axes3 = np.array(axes3).reshape(-1)
    vmax3 = max(np.abs(f).max() for f in frames_naive)
    for i, (ax, frame, R) in enumerate(zip(axes3, frames_naive, radii_cells)):
        ax.imshow(np.abs(frame), cmap="hot", vmin=0, vmax=vmax3, origin="upper",
                  extent=[img_cols.min(), img_cols.max(), img_rows.max(), img_rows.min()])
        verts = heart_vertices(R)
        h_row = [v[0] for v in verts] + [verts[0][0]]
        h_col = [v[1] for v in verts] + [verts[0][1]]
        ax.plot(h_col, h_row, "c--", linewidth=1, alpha=0.7, label="true")

        fitted_R = all_fitted_R_naive[i]
        fverts = heart_vertices(fitted_R)
        f_row = [v[0] for v in fverts] + [fverts[0][0]]
        f_col = [v[1] for v in fverts] + [fverts[0][1]]
        ax.plot(f_col, f_row, "g:", linewidth=1.5, alpha=0.9, label="fitted")

        err_mm = abs(fitted_R - R) * dx[0] * 1e3
        ax.set_title(f"phase={phases[i]:.2f}, R_true={R*dx[0]*1e3:.2f}mm\nerr={err_mm:.2f}mm", fontsize=8)
        ax.axis("off")
    for ax in axes3[len(frames_naive):]:
        ax.axis("off")
    fig3.suptitle("Heart shape-fit (NAIVE accumulator, all 16 pairs, no exclusion)\n"
                "cyan dashed = true, green dotted = globally-fitted heart (1-param R sweep)\n"
                "(TOY: exact prescribed ground truth; known center, R only)", fontsize=10)
    plt.tight_layout(rect=[0, 0.02, 1, 0.86])
    labels.add_banner(fig3)
    plt.savefig("results/figures/phase3_heart_shape_shapefit_filmstrip_naive.png", dpi=130)
    print("\nSaved results/figures/phase3_heart_shape_shapefit_filmstrip_naive.png")
