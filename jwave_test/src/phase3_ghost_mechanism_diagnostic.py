"""Phase 3 — diagnostic: is the `left->top` ghost caused by (1) genuine
bistatic specular reflection from an unexpected point on the true
boundary, or (2) genuine corner diffraction from the known vertex? Both
are real physics the capture step would correctly record; the
reconstruction's naive travel-time-only matching (no specular-
consistency check, no diffraction model) would misattribute either one
to the wrong candidate point. This directly tests which mechanism
explains the RAW captured trace, without going through backprojection
at all -- the cleanest way to separate "the physics being captured is
fine, reconstruction misinterprets it" from "something is wrong in the
capture/simulation itself."

Per user's own reasoning: since the ghost pattern persists under
translation (run -35), it must depend on relative probe/target
geometry, not absolute domain position -- consistent with either
mechanism below, and inconsistent with a translation-dependent
simulation bug (PML asymmetry, grid indexing, etc.).

All geometry below is done explicitly in (row, col) coordinates to
avoid the (x,y)=(col,row) Sources-convention mixups that have bitten
this project before -- _SRC/_RCV are converted to (row,col) once, up
front, and everything downstream stays in that one convention.

Method: capture the `left->top` pair's raw envelope trace (cleaned via
homogeneous-medium subtraction) for the ED frame (R=60, centered case,
matching phase3_left_ghost_diagnostic.py). Find where it ACTUALLY peaks
in time. Compute two candidate predicted arrival times:
  (1) best specular point: sweep many points along the TRUE triangle
      boundary, compute the law-of-reflection "defect" (how far the
      mirror-reflected incident ray deviates from the actual outgoing
      direction to the receiver) at each, find the minimum-defect point,
      and its predicted round-trip time.
  (2) vertex diffraction: predicted time assuming energy radiates from
      the KNOWN vertex position (exact, since this is a toy with known
      ground truth), (dist(src,vertex)+dist(vertex,rcv))/c + group delay.
Compare both predictions directly to the actual observed peak time.
"""

import numpy as np
from scipy.signal import find_peaks

from phase3_backprojection_shape_fit_triangle import (
    capture_all_pairs, build_medium_triangle, build_medium_homogeneous,
    triangle_vertices, _SRC, _RCV, t_arr, c_ref, dx, p3cfg, labels,
    _ENVELOPE_GROUP_DELAY_S, DIRECT_EXCLUDE_MARGIN_S,
)

TX, RX = "left", "top"


def unit(v):
    v = np.asarray(v, dtype=float)
    n = np.linalg.norm(v)
    return v / n if n > 1e-12 else v


if __name__ == "__main__":
    print(f"*** {labels.PENDING_SIGNOFF_BANNER} ***")
    print(f"*** {labels.TOY_EXACT_GT_CAPTION} ***")
    print(f"Ghost mechanism diagnostic: testing whether the '{TX}->{RX}' "
          f"pair's false-peak energy (run -34's confirmed LEFT ghost) is "
          f"explained by genuine bistatic specular reflection from an "
          f"unexpected boundary point, or genuine corner diffraction from "
          f"the known vertex -- both real physics the reconstruction's "
          f"naive travel-time model misattributes, as opposed to a bug in "
          f"capture/simulation itself.")

    R = p3cfg.LV_RADIUS_ED_CELLS  # 60 cells, ED, centered case (matches run -34)
    print(f"\n=== Capturing '{TX}' transmit, all receivers, R={R} + homogeneous reference ===")
    pairs_tri = capture_all_pairs(build_medium_triangle(R))
    pairs_ref = capture_all_pairs(build_medium_homogeneous())

    env_tri = pairs_tri[(TX, RX)]
    env_ref = pairs_ref[(TX, RX)]
    env_clean = env_tri - env_ref

    # _SRC/_RCV are (x, y) = (col, row) per the established Sources
    # convention -- convert ONCE to (row, col) and stay in that
    # convention for everything below.
    src_xy = _SRC[TX]
    rcv_xy = _RCV[RX]
    src = (src_xy[1], src_xy[0])  # (row, col)
    rcv = (rcv_xy[1], rcv_xy[0])  # (row, col)

    direct_time = np.hypot(src[0] - rcv[0], src[1] - rcv[1]) * dx[0] / c_ref

    mask = np.abs(t_arr - direct_time) >= DIRECT_EXCLUDE_MARGIN_S
    env_masked = np.where(mask, np.abs(env_clean), 0.0)
    peak_idx, _ = find_peaks(env_masked, height=env_masked.max() * 0.3)
    peak_times = t_arr[peak_idx]
    peak_heights = env_masked[peak_idx]
    order = np.argsort(-peak_heights)
    print(f"\nActual peaks in '{TX}->{RX}' cleaned trace (>=30% of max, direct-arrival excluded):")
    for i in order[:5]:
        dist_equiv = peak_times[i] * c_ref
        print(f"  t={peak_times[i]*1e6:.3f}us, height={peak_heights[i]:.6f}, "
              f"round-trip path={dist_equiv*1e3:.2f}mm")

    top_true_peak_time = peak_times[order[0]]

    # --- Mechanism 2: vertex diffraction (exact, known vertex) ---
    top_v, botleft_v, botright_v = triangle_vertices(R)  # (row, col) tuples
    dist_src_vertex = np.hypot(src[0] - top_v[0], src[1] - top_v[1]) * dx[0]
    dist_vertex_rcv = np.hypot(rcv[0] - top_v[0], rcv[1] - top_v[1]) * dx[0]
    t_vertex = (dist_src_vertex + dist_vertex_rcv) / c_ref + _ENVELOPE_GROUP_DELAY_S
    print(f"\nMechanism 2 (vertex diffraction) predicted time: {t_vertex*1e6:.3f}us")
    print(f"  difference from actual top peak: {(top_true_peak_time - t_vertex)*1e6:.3f}us "
          f"({(top_true_peak_time - t_vertex) * c_ref * 1e3:.2f}mm path-equivalent)")

    # --- Mechanism 1: best specular point on the TRUE boundary ---
    edges = [(top_v, botleft_v), (botleft_v, botright_v), (botright_v, top_v)]
    centroid = ((top_v[0] + botleft_v[0] + botright_v[0]) / 3,
                (top_v[1] + botleft_v[1] + botright_v[1]) / 3)

    best_defect = np.inf
    best_point = None
    best_time = None
    N_SAMPLES_PER_EDGE = 400
    for p1, p2 in edges:
        edge_vec = np.array([p2[0] - p1[0], p2[1] - p1[1]])
        edge_len = np.linalg.norm(edge_vec)
        normal = np.array([-edge_vec[1], edge_vec[0]]) / edge_len  # perpendicular, (row,col)
        mid = ((p1[0] + p2[0]) / 2, (p1[1] + p2[1]) / 2)
        to_centroid = np.array([centroid[0] - mid[0], centroid[1] - mid[1]])
        if np.dot(normal, to_centroid) > 0:
            normal = -normal  # ensure OUTWARD-pointing normal

        for s in np.linspace(0.02, 0.98, N_SAMPLES_PER_EDGE):
            point = (p1[0] + s * edge_vec[0], p1[1] + s * edge_vec[1])  # (row, col)
            d_in = unit((point[0] - src[0], point[1] - src[1]))
            d_out_actual = unit((rcv[0] - point[0], rcv[1] - point[1]))
            d_reflected = d_in - 2 * np.dot(d_in, normal) * normal
            defect = 1 - np.dot(d_reflected, d_out_actual)
            if defect < best_defect:
                best_defect = defect
                best_point = point
                dist_src = np.hypot(point[0] - src[0], point[1] - src[1]) * dx[0]
                dist_rcv = np.hypot(rcv[0] - point[0], rcv[1] - point[1]) * dx[0]
                best_time = (dist_src + dist_rcv) / c_ref + _ENVELOPE_GROUP_DELAY_S

    print(f"\nMechanism 1 (best specular boundary point) found at (row,col)={best_point}, "
          f"defect={best_defect:.4f} (0=perfect specular match, 2=worst)")
    print(f"  predicted time: {best_time*1e6:.3f}us")
    print(f"  difference from actual top peak: {(top_true_peak_time - best_time)*1e6:.3f}us "
          f"({(top_true_peak_time - best_time) * c_ref * 1e3:.2f}mm path-equivalent)")

    print("\n--- Verdict ---")
    err2_mm = abs(top_true_peak_time - t_vertex) * c_ref * 1e3
    err1_mm = abs(top_true_peak_time - best_time) * c_ref * 1e3
    print(f"  mechanism 1 (specular, defect={best_defect:.4f}) time-match error: {err1_mm:.2f}mm equiv")
    print(f"  mechanism 2 (vertex diffraction) time-match error: {err2_mm:.2f}mm equiv")
    if err1_mm < err2_mm and best_defect < 0.05:
        print("  -> Mechanism 1 (real specular reflection from an unexpected boundary point) "
              "is the better match.")
    elif err2_mm < err1_mm:
        print("  -> Mechanism 2 (real corner diffraction from the known vertex) is the better match.")
    else:
        print("  -> Inconclusive with this method -- neither mechanism cleanly explains the peak.")
