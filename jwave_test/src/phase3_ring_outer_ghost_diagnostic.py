"""Phase 3 — diagnostic: which tx/rx pair(s) cause the ring phantom's
R~55 false peak (runs -39/-41), and is it explained by genuine internal
reverberation (multiple bouncing between the inner and outer
boundaries), a cross-pair ghost (like the triangle's confirmed
mechanism), or something else?

Per user's question: does the outer-boundary failure come from
multiple/double bouncing between the two close boundaries (inner
41 cells, outer 71 cells, wall_thickness=30 cells = 3mm apart)? jWave
solves the full wave PDE, so the raw captured trace ALREADY contains
every order of reverberation automatically (transmit through outer,
reflect off inner, transmit back through outer; reflect internally off
outer again, back to inner, out again; etc.) -- this is real physics,
not a simulation gap. The open question is whether the RECONSTRUCTION's
naive single-bounce-per-candidate-point model is misattributing one of
these genuine multi-bounce echoes to the wrong candidate radius, the
same class of problem already confirmed for the triangle's corner
ghosts (run -36).

Method: (1) identify which of the 16 tx/rx pairs actually dominate the
accumulator's energy at R~55 (same pair-ablation technique that found
the triangle's ghosts); (2) for the dominant pair(s), compute BOTH
candidate explanations' predicted arrival times using the pair's own
exact geometry (not a monostatic approximation): single-bounce off the
TRUE inner radius, single-bounce off the TRUE outer radius, and 1st/2nd
-order internal reverberation (extra round trips through the wall
thickness); (3) compare all of these against the pair's ACTUAL recorded
peaks to see which mechanism (if any) explains the anomaly.
"""

import numpy as np
from scipy.signal import find_peaks

from phase3_backprojection_shape_fit_triangle import (
    capture_all_pairs, build_medium_homogeneous, backproject,
    _SRC, _RCV, t_arr, c_ref, dx, img_rows, img_cols, center, p3cfg,
    labels, _ENVELOPE_GROUP_DELAY_S, DIRECT_EXCLUDE_MARGIN_S,
)
from phase3_ring_phantom_shapefit import build_medium_ring, fit_circle_radius, OUTER_R_GRID

INNER_R = 41.0  # frame 4/5's true inner radius -- the case where R~55 appeared
WALL_THICKNESS = p3cfg.WALL_THICKNESS_CELLS
OUTER_R = INNER_R + WALL_THICKNESS  # 71.0
FALSE_R = 55.0  # the observed anomaly


def per_pair_value_at_R(pairs, R, n_angles=72):
    """For each pair, sum its own (uncombined) backprojected contribution
    around the circle of radius R -- same principle as
    phase3_backprojection_pair_diagnostic.py, generalized to a full
    circular boundary instead of one point."""
    thetas = np.linspace(0, 360, n_angles, endpoint=False)
    results = {}
    for (tx, rx), envelope in pairs.items():
        src = _SRC[tx]
        rcv = _RCV[rx]
        total = 0.0
        for th in thetas:
            theta = np.deg2rad(th)
            d_row, d_col = -np.cos(theta), np.sin(theta)
            row = center[0] + R * d_row
            col = center[1] + R * d_col
            dist_tx = np.hypot(col - src[0], row - src[1]) * dx[0]
            dist_rx = np.hypot(col - rcv[0], row - rcv[1]) * dx[0]
            t_total = (dist_tx + dist_rx) / c_ref + _ENVELOPE_GROUP_DELAY_S
            total += np.interp(t_total, t_arr, envelope, left=0, right=0)
        results[(tx, rx)] = total
    return results


if __name__ == "__main__":
    print(f"*** {labels.PENDING_SIGNOFF_BANNER} ***")
    print(f"*** {labels.TOY_EXACT_GT_CAPTION} ***")
    print(f"Ring outer-boundary ghost diagnostic: inner_R={INNER_R}, "
          f"outer_R={OUTER_R}, investigating the R~{FALSE_R} false peak. "
          f"Testing reverberation vs. cross-pair ghost mechanisms.")

    print(f"\n=== Capturing all 16 pairs: inner_R={INNER_R} + homogeneous reference ===")
    pairs_tri = capture_all_pairs(build_medium_ring(INNER_R))
    pairs_ref = capture_all_pairs(build_medium_homogeneous())

    # Step 1: which pairs dominate at the false R~55 location vs the true outer R=71?
    vals_false = per_pair_value_at_R(pairs_tri, FALSE_R)
    refs_false = per_pair_value_at_R(pairs_ref, FALSE_R)
    vals_true_outer = per_pair_value_at_R(pairs_tri, OUTER_R)
    refs_true_outer = per_pair_value_at_R(pairs_ref, OUTER_R)

    print(f"\n{'tx->rx':<16}{'val@R=55(false)':>16}{'val@R=71(true outer)':>22}{'ratio':>10}")
    results = []
    for key in vals_false:
        v_false = abs(vals_false[key] - refs_false[key])
        v_true = abs(vals_true_outer[key] - refs_true_outer[key])
        ratio = v_false / (v_true + 1e-12)
        results.append((f"{key[0]}->{key[1]}", v_false, v_true, ratio))
    results.sort(key=lambda r: -r[1])
    for name, vf, vt, ratio in results:
        print(f"{name:<16}{vf:>16.5f}{vt:>22.5f}{ratio:>10.2f}")

    top3 = results[:3]
    total_false = sum(r[1] for r in results)
    top3_share = sum(r[1] for r in top3) / (total_false + 1e-12)
    print(f"\nTop-3 pairs' share of energy at false R={FALSE_R}: {top3_share*100:.1f}%")

    # Step 2: for the dominant NON-ANTIPODAL pair, test reverberation
    # hypotheses directly against its actual recorded trace. (Antipodal
    # pairs like bottom->top are geometrically degenerate for the simple
    # monostatic-direction approximation used below -- their src/rcv
    # midpoint sits exactly at the domain center, exactly the
    # degenerate-locus property first found in run -29 -- so they are
    # skipped here in favor of the next-ranked well-conditioned pair.)
    ANTIPODAL_PAIRS_LIST = [("bottom", "top"), ("top", "bottom"), ("right", "left"), ("left", "right")]
    ANTIPODAL_PAIRS = set(ANTIPODAL_PAIRS_LIST)
    non_antipodal = [r for r in results if (r[0].split("->")[0], r[0].split("->")[1]) not in ANTIPODAL_PAIRS]
    dominant_tx, dominant_rx = non_antipodal[0][0].split("->")
    print(f"\n=== Testing hypotheses for dominant NON-ANTIPODAL pair: {dominant_tx}->{dominant_rx} ===")
    print(f"    (skipped antipodal pairs -- degenerate midpoint-at-center for the "
          f"simplified monostatic-direction hypothesis test below)")
    src = _SRC[dominant_tx]
    rcv = _RCV[dominant_rx]
    env_tri = pairs_tri[(dominant_tx, dominant_rx)]
    env_ref = pairs_ref[(dominant_tx, dominant_rx)]
    env_clean = env_tri - env_ref

    direct_time = np.hypot(src[0] - rcv[0], src[1] - rcv[1]) * dx[0] / c_ref
    mask = np.abs(t_arr - direct_time) >= DIRECT_EXCLUDE_MARGIN_S
    env_masked = np.where(mask, np.abs(env_clean), 0.0)
    peak_idx, _ = find_peaks(env_masked, height=env_masked.max() * 0.15)
    peak_times = t_arr[peak_idx]
    peak_heights = env_masked[peak_idx]
    order = np.argsort(-peak_heights)
    print(f"\nActual peaks in {dominant_tx}->{dominant_rx} cleaned trace (>=15% of max):")
    for i in order[:8]:
        print(f"  t={peak_times[i]*1e6:.3f}us, height={peak_heights[i]:.6f}, "
              f"round-trip path={peak_times[i]*c_ref*1e3:.2f}mm")

    # Approximate specular point for a CIRCLE boundary (near-monostatic
    # given src/rcv are only 10 cells apart): use the point on the circle
    # closest to the src-rcv midpoint's radial line as the specular point
    # approximation for hypothesis time prediction (adequate since src/rcv
    # separation (10 cells) is small vs. probe distance to center (120 cells)).
    mid = ((src[1] + rcv[1]) / 2, (src[0] + rcv[0]) / 2)  # (row, col) approx
    mid_dist_from_center = np.hypot(mid[1] - center[1], mid[0] - center[0])
    direction = ((mid[0] - center[0]) / mid_dist_from_center, (mid[1] - center[1]) / mid_dist_from_center)

    def predicted_time_for_radius(R):
        point = (center[0] + R * direction[0], center[1] + R * direction[1])
        dist_src = np.hypot(point[1] - src[0], point[0] - src[1]) * dx[0]
        dist_rcv = np.hypot(point[1] - rcv[0], point[0] - rcv[1]) * dx[0]
        return (dist_src + dist_rcv) / c_ref + _ENVELOPE_GROUP_DELAY_S

    t_inner = predicted_time_for_radius(INNER_R)
    t_outer = predicted_time_for_radius(OUTER_R)
    extra_reverb_time = 2 * (2 * WALL_THICKNESS * dx[0]) / c_ref  # one extra internal round trip
    t_reverb1 = t_outer + extra_reverb_time
    t_reverb2 = t_outer + 2 * extra_reverb_time
    t_false_hypothesis = predicted_time_for_radius(FALSE_R)

    print(f"\nHypothesis predicted times (approx. specular point on this pair's own axis):")
    print(f"  single-bounce inner (R={INNER_R}):  t={t_inner*1e6:.3f}us")
    print(f"  single-bounce outer (R={OUTER_R}):  t={t_outer*1e6:.3f}us")
    print(f"  1st-order reverberation (extra internal round trip): t={t_reverb1*1e6:.3f}us")
    print(f"  2nd-order reverberation: t={t_reverb2*1e6:.3f}us")
    print(f"  naive single-bounce AT THE FALSE R={FALSE_R}: t={t_false_hypothesis*1e6:.3f}us")

    print("\n--- Matching actual peaks to hypotheses (within 0.3us) ---")
    hypotheses = {
        "inner (true)": t_inner, "outer (true)": t_outer,
        "reverb-1": t_reverb1, "reverb-2": t_reverb2,
        "false-R=55 naive": t_false_hypothesis,
    }
    for i in order[:8]:
        pt = peak_times[i]
        matches = [name for name, ht in hypotheses.items() if abs(pt - ht) < 0.3e-6]
        print(f"  peak t={pt*1e6:.3f}us (height={peak_heights[i]:.6f}) -> matches: {matches if matches else 'NONE'}")

    # --- Step 3: proper OFF-AXIS test for the dominant ANTIPODAL pair
    # (bottom->top), the actual main contributor to the false R=55 peak.
    # The on-axis (theta=0/180) locus is degenerate (constant regardless
    # of R, per run -29) -- but at theta=90 (perpendicular, e.g. the
    # "right"/"left" position on the circle), the geometry is
    # non-degenerate and well-defined, and this angle is exactly what
    # was implicitly driving the antipodal pair's R-dependent energy in
    # the per_pair_value_at_R sum (only 2 of 72 angles are degenerate;
    # the rest, including this one, vary normally with R). ---
    print(f"\n=== Off-axis test for the dominant ANTIPODAL pair: {ANTIPODAL_PAIRS_LIST[0][0]}->{ANTIPODAL_PAIRS_LIST[0][1]} ===")
    anti_tx, anti_rx = ANTIPODAL_PAIRS_LIST[0]
    src_a = _SRC[anti_tx]
    rcv_a = _RCV[anti_rx]
    env_tri_a = pairs_tri[(anti_tx, anti_rx)]
    env_ref_a = pairs_ref[(anti_tx, anti_rx)]
    env_clean_a = env_tri_a - env_ref_a

    direct_time_a = np.hypot(src_a[0] - rcv_a[0], src_a[1] - rcv_a[1]) * dx[0] / c_ref
    mask_a = np.abs(t_arr - direct_time_a) >= DIRECT_EXCLUDE_MARGIN_S
    env_masked_a = np.where(mask_a, np.abs(env_clean_a), 0.0)
    peak_idx_a, _ = find_peaks(env_masked_a, height=env_masked_a.max() * 0.15)
    peak_times_a = t_arr[peak_idx_a]
    peak_heights_a = env_masked_a[peak_idx_a]
    order_a = np.argsort(-peak_heights_a)
    print(f"\nActual peaks in {anti_tx}->{anti_rx} cleaned trace (>=15% of max):")
    for i in order_a[:8]:
        print(f"  t={peak_times_a[i]*1e6:.3f}us, height={peak_heights_a[i]:.6f}")

    def predicted_time_offaxis(R, theta_deg):
        th = np.deg2rad(theta_deg)
        d_row, d_col = -np.cos(th), np.sin(th)
        point_row = center[0] + R * d_row
        point_col = center[1] + R * d_col
        dist_src = np.hypot(point_col - src_a[0], point_row - src_a[1]) * dx[0]
        dist_rcv = np.hypot(point_col - rcv_a[0], point_row - rcv_a[1]) * dx[0]
        return (dist_src + dist_rcv) / c_ref + _ENVELOPE_GROUP_DELAY_S

    THETA_TEST = 90.0  # perpendicular to the bottom-top axis -- non-degenerate
    t_inner_a = predicted_time_offaxis(INNER_R, THETA_TEST)
    t_outer_a = predicted_time_offaxis(OUTER_R, THETA_TEST)
    t_false_a = predicted_time_offaxis(FALSE_R, THETA_TEST)
    t_reverb1_a = t_outer_a + extra_reverb_time
    t_reverb2_a = t_outer_a + 2 * extra_reverb_time

    print(f"\nOff-axis (theta={THETA_TEST}) hypothesis predicted times for {anti_tx}->{anti_rx}:")
    print(f"  single-bounce inner (R={INNER_R}):  t={t_inner_a*1e6:.3f}us")
    print(f"  single-bounce outer (R={OUTER_R}):  t={t_outer_a*1e6:.3f}us")
    print(f"  false R={FALSE_R} naive:            t={t_false_a*1e6:.3f}us")
    print(f"  1st-order reverberation:            t={t_reverb1_a*1e6:.3f}us")
    print(f"  2nd-order reverberation:            t={t_reverb2_a*1e6:.3f}us")

    hypotheses_a = {
        "inner (true)": t_inner_a, "outer (true)": t_outer_a,
        "false-R=55 naive": t_false_a, "reverb-1": t_reverb1_a, "reverb-2": t_reverb2_a,
    }
    print("\n--- Matching actual antipodal-pair peaks to off-axis hypotheses (within 0.3us) ---")
    for i in order_a[:8]:
        pt = peak_times_a[i]
        matches = [name for name, ht in hypotheses_a.items() if abs(pt - ht) < 0.3e-6]
        print(f"  peak t={pt*1e6:.3f}us (height={peak_heights_a[i]:.6f}) -> matches: {matches if matches else 'NONE'}")
