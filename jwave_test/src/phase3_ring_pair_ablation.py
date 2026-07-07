"""Phase 3 — pair-CLASS ablation for the ring phantom's outer boundary
(NOT anatomical cancellation). Per explicit user direction after run
-42's mechanism diagnosis and the "should I proceed" discussion: any
fix here must be a pure instrumentation/geometry decision (which tx/rx
pairs to trust), never an anatomical assumption (rejected in run -41)
-- and must be checked for whether it collapses the reconstruction
(loses redundancy/generalizability) rather than just improving one
number on one idealized, perfectly circular, centered phantom.

Five pair-class configurations, all derived from run -42's own
per-pair evidence table (not guessed):
  A. all 16 pairs (baseline, = run -39/-41's "independent" result)
  B. monostatic only: top->top, bottom->bottom, left->left, right->right
     (4 pairs; run -42 showed these are the ONLY pairs with ratio<1,
     i.e. the only pairs that correctly favor the true outer boundary)
  C. remove antipodal pairs only (exclude bottom<->top, left<->right;
     keep monostatic + all 8 cross pairs = 12 pairs) -- antipodal pairs
     were run -42's single worst offender (ratio 1.59)
  D. remove ALL pairs run -42's table flagged as favoring the false
     peak (ratio > 1 at R=55 vs R=71) -- every non-monostatic pair
     showed ratio > 1, so this is numerically identical to B; kept as a
     distinct, explicit configuration to directly verify that
     "monostatic-only" is what the per-pair evidence itself recommends,
     not just a convenient guess
  E. monostatic + the 2 LEAST-biased cross pairs by run -42's own
     numbers (right->top, bottom->left, ratio 1.19 -- the closest to
     neutral of the 8 cross pairs) = 6 pairs -- tests whether a LITTLE
     extra angular diversity can be added back without reintroducing
     much bias

For each configuration, measures: inner RMSE, outer RMSE, score-curve
confidence (ratio of the best candidate's score to the next-best local
competing peak -- a low ratio means multiple candidates are nearly
tied, a fragile/unreliable fit even if the argmax happens to be
correct), whether the outer fit locks onto the inner fit's own value,
and whether the inner fit's own (already-excellent) accuracy collapses
under any reduced pair set -- the key generalization/safety checks per
the user's explicit concern about reconstruction collapse.

Tested at TWO points to start checking generalization rather than
over-fitting to one case: the ring's ED frame (inner_R=60, large/strong
signal) and the ES-adjacent frame (inner_R=41, where run -39/-42's R~55
anomaly was originally found, weaker signal). A true cross-SHAPE
generalization check (triangle/heart-cartoon) is flagged as a further
follow-on, not done in this pass, given compute budget.
"""

import numpy as np
from scipy.interpolate import RegularGridInterpolator
from scipy.signal import find_peaks

from phase3_backprojection_shape_fit_triangle import (
    capture_all_pairs, build_medium_homogeneous, direction_vector,
    img_rows, img_cols, center, dx, p3cfg, labels,
)
from phase3_ring_phantom_shapefit import build_medium_ring, INNER_R_GRID, OUTER_R_GRID

N_ANGLES = 72
_THETAS = np.linspace(0, 360, N_ANGLES, endpoint=False)

MONOSTATIC = {("top", "top"), ("bottom", "bottom"), ("left", "left"), ("right", "right")}
ANTIPODAL = {("bottom", "top"), ("top", "bottom"), ("left", "right"), ("right", "left")}
LEAST_BIASED_CROSS = {("right", "top"), ("bottom", "left")}  # ratio 1.19 in run -42's table

CONFIGS = {
    "A_all16": None,  # None = no filter, use all pairs
    "B_monostatic_only": MONOSTATIC,
    "C_remove_antipodal": set(MONOSTATIC) | (set() ),  # placeholder, built properly below
    "D_remove_run42_implicated": MONOSTATIC,  # numerically same as B, see docstring
    "E_monostatic_plus_least_biased": MONOSTATIC | LEAST_BIASED_CROSS,
}


def backproject_filtered(pairs, allowed_pairs=None):
    """allowed_pairs=None means use ALL pairs (config A). Otherwise only
    sum pairs whose (tx,rx) key is in allowed_pairs."""
    from phase3_backprojection_shape_fit_triangle import _SRC, _RCV, t_arr, c_ref, _ENVELOPE_GROUP_DELAY_S
    RR, CC = np.meshgrid(img_rows, img_cols, indexing="ij")
    accumulator = np.zeros(RR.shape)
    for (tx, rx), envelope in pairs.items():
        if allowed_pairs is not None and (tx, rx) not in allowed_pairs:
            continue
        src = _SRC[tx]
        rcv = _RCV[rx]
        dist_tx = np.sqrt((CC - src[0]) ** 2 + (RR - src[1]) ** 2) * dx[0]
        dist_rx = np.sqrt((CC - rcv[0]) ** 2 + (RR - rcv[1]) ** 2) * dx[0]
        t_total = (dist_tx + dist_rx) / c_ref + _ENVELOPE_GROUP_DELAY_S
        accumulator += np.interp(t_total, t_arr, envelope, left=0, right=0)
    return accumulator


def fit_circle_radius_with_confidence(accumulator, R_grid, origin=center):
    interp = RegularGridInterpolator((img_rows, img_cols), np.abs(accumulator),
                                      bounds_error=False, fill_value=0.0)
    scores = np.zeros(len(R_grid))
    for i, R in enumerate(R_grid):
        pts = []
        for th in _THETAS:
            d_row, d_col = direction_vector(th)
            pts.append((origin[0] + R * d_row, origin[1] + R * d_col))
        scores[i] = interp(np.array(pts)).sum()
    best_idx = np.argmax(scores)
    best_R = R_grid[best_idx]

    # confidence = best score / next-best LOCAL peak's score (excluding
    # points immediately adjacent to the best one)
    peak_idx, _ = find_peaks(scores)
    if len(peak_idx) >= 2:
        peak_scores = scores[peak_idx]
        sorted_peaks = np.sort(peak_scores)[::-1]
        confidence = sorted_peaks[0] / (sorted_peaks[1] + 1e-12)
    else:
        confidence = np.inf  # only one local peak at all -- maximally confident
    return best_R, scores, confidence


def run_ablation_at(inner_R_true, label):
    outer_R_true = inner_R_true + p3cfg.WALL_THICKNESS_CELLS
    print(f"\n{'='*70}\n=== {label}: inner_R_true={inner_R_true}, outer_R_true={outer_R_true} ===\n{'='*70}")

    print("Capturing pairs (ring + homogeneous reference)...")
    pairs_tri = capture_all_pairs(build_medium_ring(inner_R_true))
    pairs_ref = capture_all_pairs(build_medium_homogeneous())

    # Build the real config C set now that MONOSTATIC/ANTIPODAL are defined.
    all_pairs = set(pairs_tri.keys())
    configs = {
        "A_all16": None,
        "B_monostatic_only": MONOSTATIC,
        "C_remove_antipodal": all_pairs - ANTIPODAL,
        "D_remove_run42_implicated": MONOSTATIC,
        "E_monostatic_plus_least_biased": MONOSTATIC | LEAST_BIASED_CROSS,
    }

    results = {}
    for name, allowed in configs.items():
        acc_tri = backproject_filtered(pairs_tri, allowed)
        acc_ref = backproject_filtered(pairs_ref, allowed)
        acc_clean = acc_tri - acc_ref

        fitted_inner, _, conf_inner = fit_circle_radius_with_confidence(acc_clean, INNER_R_GRID)
        fitted_outer, _, conf_outer = fit_circle_radius_with_confidence(acc_clean, OUTER_R_GRID)

        inner_err_mm = abs(fitted_inner - inner_R_true) * dx[0] * 1e3
        outer_err_mm = abs(fitted_outer - outer_R_true) * dx[0] * 1e3
        locked_to_inner = abs(fitted_outer - fitted_inner) < 3.0  # cells

        n_pairs_used = len(allowed) if allowed is not None else len(all_pairs)
        results[name] = dict(fitted_inner=fitted_inner, fitted_outer=fitted_outer,
                              inner_err_mm=inner_err_mm, outer_err_mm=outer_err_mm,
                              conf_inner=conf_inner, conf_outer=conf_outer,
                              locked_to_inner=locked_to_inner, n_pairs=n_pairs_used)
        print(f"\n[{name}] ({n_pairs_used} pairs)")
        print(f"  inner: fitted={fitted_inner:.1f} true={inner_R_true:.1f} err={inner_err_mm:.2f}mm conf={conf_inner:.2f}")
        print(f"  outer: fitted={fitted_outer:.1f} true={outer_R_true:.1f} err={outer_err_mm:.2f}mm conf={conf_outer:.2f}")
        print(f"  outer locked to inner fit? {locked_to_inner}")

    return results


if __name__ == "__main__":
    print(f"*** {labels.PENDING_SIGNOFF_BANNER} ***")
    print(f"*** {labels.TOY_EXACT_GT_CAPTION} ***")
    print("Pair-CLASS ablation (not anatomical cancellation) for the ring "
          "phantom's outer boundary. Testing 5 pair-subset configurations "
          "at 2 frames (ED inner_R=60, and the R~55-anomaly inner_R=41 "
          "case) to check both accuracy AND whether reduced pair sets "
          "cause reconstruction collapse (loss of confidence/redundancy), "
          "per the explicit concern that any fix must generalize to real "
          "physiological/asymmetric anatomy, not just this idealized "
          "circular, centered phantom.")

    all_results = {}
    all_results["ED_inner60"] = run_ablation_at(60.0, "ED frame")
    all_results["ES_inner41"] = run_ablation_at(41.0, "ES-adjacent frame (R~55 anomaly case)")

    print(f"\n\n{'='*70}\n=== SUMMARY ===\n{'='*70}")
    for frame_label, results in all_results.items():
        print(f"\n--- {frame_label} ---")
        print(f"{'config':<32}{'n_pairs':>8}{'inner_err':>11}{'outer_err':>11}{'in_conf':>9}{'out_conf':>9}{'locked':>8}")
        for name, r in results.items():
            print(f"{name:<32}{r['n_pairs']:>8}{r['inner_err_mm']:>10.2f}mm{r['outer_err_mm']:>10.2f}mm"
                  f"{r['conf_inner']:>9.2f}{r['conf_outer']:>9.2f}{str(r['locked_to_inner']):>8}")
