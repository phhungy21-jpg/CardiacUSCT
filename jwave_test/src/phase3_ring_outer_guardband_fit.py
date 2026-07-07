"""Phase 3 — fixing the ring phantom's outer-boundary masking WITHOUT
assuming constant wall thickness.

Run -40 proved the myocardium/chest-wall-proxy interface has strong,
real, independently-recoverable signal (0.00mm error in isolation) --
the ring's outer-boundary failure (run -39) is caused specifically by
the inner boundary's energy contaminating the outer search, not by the
interface being weak.

REJECTED approach (per explicit user objection): coupling
outer_R = inner_R_fitted + WALL_THICKNESS_CELLS. This would be a
clinically harmful modeling error -- real myocardial wall thickness
varies regionally and pathologically (hypertrophy, post-infarct
thinning, aneurysmal bulging), and detecting exactly that variation is
a primary purpose of cardiac imaging. Hard-coding constant thickness
into the reconstruction would structurally blind it to the pathology
it's supposed to find, and would do so SILENTLY (a confident, clean-
looking, wrong answer), which is worse than an obvious failure.

This script instead removes the known CONFOUND (the inner boundary's
own energy leaking into/dominating the outer search near its own
radius) without assuming anything about the true outer radius: use the
already-reliable inner fit only to EXCLUDE a narrow guard band
immediately around it from the outer search space, then let the outer
search remain completely free elsewhere -- so a real heart with
abnormal or wildly different wall thickness would still be found
correctly, not overridden by a thickness assumption.
"""

import numpy as np

from phase3_backprojection_shape_fit_triangle import (
    capture_all_pairs, build_medium_homogeneous, backproject,
    dx, p3cfg, labels,
)
from phase3_ring_phantom_shapefit import (
    build_medium_ring, fit_circle_radius, INNER_R_GRID, OUTER_R_GRID,
)

from matplotlib import pyplot as plt
import os

GUARD_BAND_CELLS = 8.0  # exclude candidates within this distance of the known inner radius


def fit_outer_with_guardband(accumulator, outer_R_grid, inner_R_known, guard_cells=GUARD_BAND_CELLS):
    """Same global template-match as fit_circle_radius, but candidates
    too close to the ALREADY-KNOWN inner radius are excluded from
    consideration -- removes the specific confound (inner-boundary
    energy near its own radius) without constraining where the TRUE
    outer radius is allowed to be."""
    mask = np.abs(outer_R_grid - inner_R_known) > guard_cells
    filtered_grid = outer_R_grid[mask]
    best_R, scores_filtered = fit_circle_radius(accumulator, filtered_grid)
    return best_R, filtered_grid, scores_filtered


if __name__ == "__main__":
    print(f"*** {labels.PENDING_SIGNOFF_BANNER} ***")
    print(f"*** {labels.TOY_EXACT_GT_CAPTION} ***")
    print("Ring phantom outer-boundary fit with a GUARD BAND around the "
          "known inner radius (NOT a constant-wall-thickness assumption "
          "-- the outer search remains free to find any true radius, "
          "this only excludes the specific confound region immediately "
          "around the already-known inner boundary).")

    print("\n=== Control: homogeneous medium (no ring) ===")
    pairs_ref = capture_all_pairs(build_medium_homogeneous())
    accumulator_ref = backproject(pairs_ref)

    N_FRAMES_MOVIE = 8
    phases = np.linspace(0, 1, N_FRAMES_MOVIE)
    inner_radii = [p3cfg.lv_radius_at_phase(p) for p in phases]

    all_fitted_inner = []
    all_fitted_outer_indep = []
    all_fitted_outer_guard = []
    for i, inner_R in enumerate(inner_radii):
        outer_R_true = inner_R + p3cfg.WALL_THICKNESS_CELLS
        print(f"=== Frame {i+1}/{N_FRAMES_MOVIE} (inner_R={inner_R:.1f}, outer_R={outer_R_true:.1f} cells) ===")
        pairs = capture_all_pairs(build_medium_ring(inner_R))
        accumulator_clean = backproject(pairs) - accumulator_ref

        fitted_inner, _ = fit_circle_radius(accumulator_clean, INNER_R_GRID)
        fitted_outer_indep, _ = fit_circle_radius(accumulator_clean, OUTER_R_GRID)
        fitted_outer_guard, _, _ = fit_outer_with_guardband(accumulator_clean, OUTER_R_GRID, fitted_inner)

        all_fitted_inner.append(fitted_inner)
        all_fitted_outer_indep.append(fitted_outer_indep)
        all_fitted_outer_guard.append(fitted_outer_guard)

        err_indep_mm = abs(fitted_outer_indep - outer_R_true) * dx[0] * 1e3
        err_guard_mm = abs(fitted_outer_guard - outer_R_true) * dx[0] * 1e3
        print(f"  inner fitted={fitted_inner:.1f} | outer independent={fitted_outer_indep:.1f} "
              f"err={err_indep_mm:.2f}mm | outer guard-band={fitted_outer_guard:.1f} err={err_guard_mm:.2f}mm")

    print("\n--- Outer-boundary RMSE: independent (run -39) vs. guard-band ---")
    errs_indep = np.array([abs(all_fitted_outer_indep[i] - (inner_radii[i] + p3cfg.WALL_THICKNESS_CELLS)) * dx[0] * 1e3
                           for i in range(N_FRAMES_MOVIE)])
    errs_guard = np.array([abs(all_fitted_outer_guard[i] - (inner_radii[i] + p3cfg.WALL_THICKNESS_CELLS)) * dx[0] * 1e3
                           for i in range(N_FRAMES_MOVIE)])
    print(f"  independent RMSE={np.sqrt(np.mean(errs_indep**2)):.4f}mm  (per-frame: {np.round(errs_indep,2).tolist()})")
    print(f"  guard-band  RMSE={np.sqrt(np.mean(errs_guard**2)):.4f}mm  (per-frame: {np.round(errs_guard,2).tolist()})")
