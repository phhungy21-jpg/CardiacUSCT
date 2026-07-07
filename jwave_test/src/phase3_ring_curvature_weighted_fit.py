"""Phase 3 — curvature-aware weighted backprojection for the ring
phantom's outer boundary, replacing run -43's rejected fixed pair-
subset rules with a physically-motivated, per-candidate-radius weight.

Per run -44's confirmed root cause: a pair's reliability for a given
candidate boundary radius R depends on R's own curvature (how much a
circle of that radius diverges its reflected energy across baseline
angle) -- NOT a fixed pair category. Measured amplitude ratios
(baseline-category amplitude / that same radius's own monostatic
amplitude), from run -44's isolated single-boundary phantoms:
  R=41 (inner): cross/mono=0.136, antipodal/mono=0.045
  R=71 (outer): cross/mono=0.000, antipodal/mono=0.000
This script builds a SIMPLE LINEAR interpolation/extrapolation between
these two measured points as the weight model -- explicitly a first
approximation (only 2 radii measured), not a fully-derived physical
divergence formula. Clipped to [0,1] outside the measured range (weight
can't be negative; can't exceed 1).

For each candidate R in the sweep, a pair's contribution is multiplied
by weight(baseline_category(pair), R) before summing -- so at a SMALL
candidate R, cross/antipodal pairs keep substantial weight (preserving
the redundancy that kept the inner fit robust, avoiding run -43's
monostatic-only collapse); at a LARGE candidate R, cross/antipodal
pairs are naturally down-weighted (removing exactly the votes that
caused the outer boundary to lock onto the inner boundary's real
signal). This is tested on the SAME two frames as run -43's ablation
(ED strong-signal, ES-adjacent weak-signal) for direct, apples-to-
apples comparison against every previously-tried approach.
"""

import numpy as np
from scipy.interpolate import RegularGridInterpolator
from scipy.signal import find_peaks

from phase3_backprojection_shape_fit_triangle import (
    capture_all_pairs, build_medium_homogeneous, direction_vector,
    _SRC, _RCV, t_arr, c_ref, _ENVELOPE_GROUP_DELAY_S,
    img_rows, img_cols, center, dx, p3cfg, labels,
)
from phase3_ring_phantom_shapefit import build_medium_ring, INNER_R_GRID, OUTER_R_GRID

N_ANGLES = 72
_THETAS = np.linspace(0, 360, N_ANGLES, endpoint=False)

MONOSTATIC = {("top", "top"), ("bottom", "bottom"), ("left", "left"), ("right", "right")}
ANTIPODAL = {("bottom", "top"), ("top", "bottom"), ("left", "right"), ("right", "left")}
# everything else (8 pairs) is "cross"

# Measured calibration points from run -44 (isolated single-boundary
# amplitude ratios, baseline-category amplitude / monostatic amplitude
# at that SAME radius):
_CAL_R = np.array([41.0, 71.0])
_CAL_CROSS = np.array([0.136, 0.000])
_CAL_ANTIPODAL = np.array([0.045, 0.000])


def _linear_weight(R, cal_r, cal_w):
    """Linear interpolation between the 2 measured points, clipped to
    [0,1] outside the measured range."""
    slope = (cal_w[1] - cal_w[0]) / (cal_r[1] - cal_r[0])
    w = cal_w[0] + slope * (R - cal_r[0])
    return float(np.clip(w, 0.0, 1.0))


def pair_baseline_category(tx, rx):
    if (tx, rx) in MONOSTATIC:
        return "monostatic"
    if (tx, rx) in ANTIPODAL:
        return "antipodal"
    return "cross"


def pair_weight_at_R(tx, rx, R):
    cat = pair_baseline_category(tx, rx)
    if cat == "monostatic":
        return 1.0
    if cat == "cross":
        return _linear_weight(R, _CAL_R, _CAL_CROSS)
    return _linear_weight(R, _CAL_R, _CAL_ANTIPODAL)


def fit_circle_radius_curvature_weighted(pairs, R_grid, origin=center):
    """Same global template-match principle, but each pair's
    contribution at candidate R is scaled by pair_weight_at_R(R) BEFORE
    summing -- unlike a fixed pair-subset, this weight varies per
    candidate R within a single sweep."""
    RR, CC = np.meshgrid(img_rows, img_cols, indexing="ij")
    # Precompute each pair's raw (unweighted) contribution grid once.
    per_pair_grids = {}
    for (tx, rx), envelope in pairs.items():
        src = _SRC[tx]
        rcv = _RCV[rx]
        dist_tx = np.sqrt((CC - src[0]) ** 2 + (RR - src[1]) ** 2) * dx[0]
        dist_rx = np.sqrt((CC - rcv[0]) ** 2 + (RR - rcv[1]) ** 2) * dx[0]
        t_total = (dist_tx + dist_rx) / c_ref + _ENVELOPE_GROUP_DELAY_S
        per_pair_grids[(tx, rx)] = np.interp(t_total, t_arr, envelope, left=0, right=0)

    scores = np.zeros(len(R_grid))
    # For scoring we need each pair's contribution SPECIFICALLY at points
    # on the candidate circle (not the whole grid), so build an
    # interpolator per pair once, then weight per candidate R.
    interpolators = {
        key: RegularGridInterpolator((img_rows, img_cols), np.abs(grid), bounds_error=False, fill_value=0.0)
        for key, grid in per_pair_grids.items()
    }
    for i, R in enumerate(R_grid):
        pts = np.array([(origin[0] + R * direction_vector(th)[0], origin[1] + R * direction_vector(th)[1])
                         for th in _THETAS])
        total = 0.0
        for (tx, rx), interp in interpolators.items():
            w = pair_weight_at_R(tx, rx, R)
            total += w * interp(pts).sum()
        scores[i] = total

    best_idx = np.argmax(scores)
    best_R = R_grid[best_idx]
    peak_idx, _ = find_peaks(scores)
    if len(peak_idx) >= 2:
        sorted_peaks = np.sort(scores[peak_idx])[::-1]
        confidence = sorted_peaks[0] / (sorted_peaks[1] + 1e-12)
    else:
        confidence = np.inf
    return best_R, scores, confidence


GUARD_BAND_CELLS = 8.0  # exclude outer-grid candidates within this distance of the already-fitted inner radius


if __name__ == "__main__":
    print(f"*** {labels.PENDING_SIGNOFF_BANNER} ***")
    print(f"*** {labels.TOY_EXACT_GT_CAPTION} ***")
    print("Curvature-aware weighted shape-fit, COMBINED with a guard-band "
          "exclusion around the already-fitted inner radius (run -41's "
          "idea, now combined with run -44's curvature weighting instead "
          "of applied alone). Tested at the same 2 frames as run -43's "
          "ablation for direct comparison against every prior approach.")

    for inner_R_true, label in [(60.0, "ED frame"), (41.0, "ES-adjacent frame")]:
        outer_R_true = inner_R_true + p3cfg.WALL_THICKNESS_CELLS
        print(f"\n{'='*70}\n=== {label}: inner_R_true={inner_R_true}, outer_R_true={outer_R_true} ===\n{'='*70}")
        pairs_tri = capture_all_pairs(build_medium_ring(inner_R_true))
        pairs_ref = capture_all_pairs(build_medium_homogeneous())

        fitted_inner, _, conf_inner = fit_circle_radius_curvature_weighted(pairs_tri, INNER_R_GRID)
        fitted_inner_ref, _, _ = fit_circle_radius_curvature_weighted(pairs_ref, INNER_R_GRID)

        # No guard band (as before)
        fitted_outer_plain, _, conf_outer_plain = fit_circle_radius_curvature_weighted(pairs_tri, OUTER_R_GRID)

        # WITH guard band around the already-fitted inner radius
        outer_grid_guarded = OUTER_R_GRID[np.abs(OUTER_R_GRID - fitted_inner) > GUARD_BAND_CELLS]
        fitted_outer_guarded, _, conf_outer_guarded = fit_circle_radius_curvature_weighted(pairs_tri, outer_grid_guarded)
        fitted_outer_ref, _, _ = fit_circle_radius_curvature_weighted(pairs_ref, OUTER_R_GRID)

        inner_err_mm = abs(fitted_inner - inner_R_true) * dx[0] * 1e3
        outer_err_plain_mm = abs(fitted_outer_plain - outer_R_true) * dx[0] * 1e3
        outer_err_guarded_mm = abs(fitted_outer_guarded - outer_R_true) * dx[0] * 1e3
        locked_plain = abs(fitted_outer_plain - fitted_inner) < 3.0
        locked_guarded = abs(fitted_outer_guarded - fitted_inner) < 3.0

        print(f"\n[curvature-weighted] inner: fitted={fitted_inner:.1f} true={inner_R_true:.1f} "
              f"err={inner_err_mm:.2f}mm conf={conf_inner:.2f}")
        print(f"[curvature-weighted, no guard band] outer: fitted={fitted_outer_plain:.1f} true={outer_R_true:.1f} "
              f"err={outer_err_plain_mm:.2f}mm conf={conf_outer_plain:.2f} locked={locked_plain}")
        print(f"[curvature-weighted + guard band] outer: fitted={fitted_outer_guarded:.1f} true={outer_R_true:.1f} "
              f"err={outer_err_guarded_mm:.2f}mm conf={conf_outer_guarded:.2f} locked={locked_guarded}")
        print(f"  homogeneous-medium control: inner fit={fitted_inner_ref:.1f}, outer fit={fitted_outer_ref:.1f} "
              f"(should be meaningless/low-confidence)")
