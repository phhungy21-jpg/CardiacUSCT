"""Phase 3 — pure-geometry test of the curvature hypothesis for why the
ring phantom's inner boundary is accurate but the outer is not.

Per user question: "what makes the inner ring accurate but not the
outer one?" Hypothesis (checked against cited tissue impedances first:
the OUTER interface's reflection coefficient (0.0035) is actually
LARGER than the INNER's (0.0025), so intrinsic signal strength is not
the explanation -- run -40 also already proved the outer interface is
perfectly recoverable in isolation): a SMALLER circle (inner boundary)
is more sharply curved, so -- like a small convex mirror -- it scatters
specular reflections across a WIDE range of bistatic angles, making it
visible to many different tx/rx pair geometries. A LARGER circle
(outer boundary) is flatter locally, so it mostly reflects straight
back toward whichever probe is looking directly at it -- visible
mainly to monostatic pairs, invisible to most bistatic pairs. This
would directly explain run -42's finding (cross/antipodal pairs' real
energy matches the INNER boundary's prediction, not the outer's) and
run -43's finding (only monostatic pairs correctly favor the outer
boundary).

This is a pure-geometry test, no new simulation: for each of several
tx/rx pairs (monostatic, cross, antipodal) and each of the two true
radii (inner=41, outer=71), sweep every point on that circle and find
the SMALLEST "specular defect" (same law-of-reflection mismatch metric
validated in run -36's triangle-vertex mechanism diagnosis) achievable
anywhere on the circle. If the outer circle's best achievable defect is
much worse (larger) than the inner circle's for bistatic pairs, that
directly confirms the curvature hypothesis.
"""

import numpy as np

from phase3_backprojection_shape_fit_triangle import _SRC, _RCV, center

INNER_R = 41.0
OUTER_R = 71.0

TEST_PAIRS = [
    ("top", "top", "monostatic"),
    ("top", "right", "cross (adjacent, 90deg)"),
    ("bottom", "top", "antipodal (180deg)"),
]


def unit(v):
    v = np.asarray(v, dtype=float)
    n = np.linalg.norm(v)
    return v / n if n > 1e-12 else v


def best_specular_defect(src_xy, rcv_xy, R, n_samples=2000):
    """src_xy/rcv_xy are (x,y)=(col,row) per the established Sources
    convention; converted to (row,col) once for consistency with the
    circle's own (row,col) parametrization."""
    src = (src_xy[1], src_xy[0])  # (row, col)
    rcv = (rcv_xy[1], rcv_xy[0])
    thetas = np.linspace(0, 2 * np.pi, n_samples, endpoint=False)
    best_defect = np.inf
    best_theta = None
    for th in thetas:
        point = (center[0] + R * np.cos(th), center[1] + R * np.sin(th))
        normal = np.array([np.cos(th), np.sin(th)])  # outward radial normal for a circle
        d_in = unit((point[0] - src[0], point[1] - src[1]))
        d_out_actual = unit((rcv[0] - point[0], rcv[1] - point[1]))
        d_reflected = d_in - 2 * np.dot(d_in, normal) * normal
        defect = 1 - np.dot(d_reflected, d_out_actual)
        if defect < best_defect:
            best_defect = defect
            best_theta = np.degrees(th)
    return best_defect, best_theta


if __name__ == "__main__":
    print("Curvature hypothesis test: best achievable specular-reflection")
    print("match (0=perfect, 2=worst) anywhere on the INNER vs OUTER circle,")
    print("for monostatic / cross / antipodal pair geometries.\n")
    print(f"{'pair type':<26}{'tx->rx':<14}{'inner R=41 defect':>20}{'outer R=71 defect':>20}")
    for tx, rx, label in TEST_PAIRS:
        src = _SRC[tx]
        rcv = _RCV[rx]
        defect_inner, theta_inner = best_specular_defect(src, rcv, INNER_R)
        defect_outer, theta_outer = best_specular_defect(src, rcv, OUTER_R)
        print(f"{label:<26}{tx+'->'+rx:<14}{defect_inner:>20.5f}{defect_outer:>20.5f}")

    print("\nLower defect = a valid (or near-valid) specular reflection point exists")
    print("on that circle for that pair. If outer defects are much larger than inner")
    print("defects specifically for cross/antipodal (not monostatic) pairs, that")
    print("confirms the curvature hypothesis: smaller circles are visible to wide-")
    print("baseline bistatic pairs, larger circles are not.")
