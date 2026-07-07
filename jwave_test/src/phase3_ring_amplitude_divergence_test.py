"""Phase 3 — testing the REFINED curvature hypothesis: does a smaller
reflecting circle deliver more amplitude to wide-baseline (cross/
antipodal) bistatic pairs than a larger circle does, at its own
predicted specular time?

Run -[curvature diagnostic] refuted "does a specular point exist" as
the explanation (a valid specular point always exists on a full circle
for any pair, regardless of radius -- unlike the triangle's finite
edges). The remaining, physically well-grounded candidate: a convex
reflecting circle's curvature controls how much its reflected energy
DIVERGES across receiver angle (a standard convex-mirror effect) --
smaller (inner-boundary-like) circles should spread reflected energy
across a WIDER range of bistatic angles, larger (outer-boundary-like)
circles should keep it more concentrated near the monostatic direction.

This isolates each boundary ALONE (myocardium disk in chest-wall-proxy
background, no competing boundary -- same construction validated in
run -40 for R=90) at BOTH the ring's true inner (R=41) and outer (R=71)
radii, captures all 16 pairs for each, and directly compares each
pair's REAL recorded amplitude at its own predicted specular time,
grouped by baseline angle (monostatic=0deg, cross=90deg,
antipodal=180deg). If outer (R=71) amplitude falls off much faster with
baseline angle than inner (R=41) does, that confirms the divergence
hypothesis using real simulated physics, not just geometry.
"""

import numpy as np

from phase3_backprojection_shape_fit_triangle import (
    capture_all_pairs, build_medium_homogeneous, _SRC, _RCV, t_arr,
    c_ref, dx, _ENVELOPE_GROUP_DELAY_S, labels,
)
from phase3_outer_boundary_diagnostic import build_medium_myocardium_disk

RADII = {"inner (R=41)": 41.0, "outer (R=71)": 71.0}
PAIRS_BY_BASELINE = {
    "monostatic (0deg)": [("top", "top"), ("bottom", "bottom"), ("left", "left"), ("right", "right")],
    "cross (90deg)": [("top", "right"), ("top", "left"), ("bottom", "right"), ("bottom", "left"),
                       ("right", "top"), ("right", "bottom"), ("left", "top"), ("left", "bottom")],
    "antipodal (180deg)": [("top", "bottom"), ("bottom", "top"), ("left", "right"), ("right", "left")],
}


def predicted_time(tx_name, rx_name, R):
    src = _SRC[tx_name]
    rcv = _RCV[rx_name]
    # nearest point on the circle of radius R to this pair's own axis --
    # reuse the exact per-pair specular search already validated in
    # phase3_ring_curvature_diagnostic.py, but we only need the TIME here,
    # so just take distance-to-circle along the pair's own bisector as an
    # adequate approximation for a near-monostatic-ish reference; for
    # generality we instead directly search all angles for the true
    # minimum round-trip time (the actual arrival, whatever it is).
    thetas = np.linspace(0, 2 * np.pi, 720, endpoint=False)
    center_row, center_col = 150, 150
    best_t = None
    best_dist_sum = np.inf
    for th in thetas:
        row = center_row + R * np.cos(th)
        col = center_col + R * np.sin(th)
        d_tx = np.hypot(col - src[0], row - src[1]) * dx[0]
        d_rx = np.hypot(col - rcv[0], row - rcv[1]) * dx[0]
        if d_tx + d_rx < best_dist_sum:
            best_dist_sum = d_tx + d_rx
            best_t = (d_tx + d_rx) / c_ref + _ENVELOPE_GROUP_DELAY_S
    return best_t


if __name__ == "__main__":
    print(f"*** {labels.PENDING_SIGNOFF_BANNER} ***")
    print(f"*** {labels.TOY_EXACT_GT_CAPTION} ***")
    print("Amplitude-divergence test: for an ISOLATED single boundary "
          "(no competing boundary), does a smaller circle deliver more "
          "real amplitude to wide-baseline (cross/antipodal) pairs than "
          "a larger circle does, at its own predicted nearest-point "
          "specular time?")

    pairs_ref = capture_all_pairs(build_medium_homogeneous())

    results = {}
    for r_label, R in RADII.items():
        print(f"\n=== Capturing isolated myocardium disk: {r_label} ===")
        pairs_disk = capture_all_pairs(build_medium_myocardium_disk(R))
        amp_by_baseline = {}
        for baseline_label, pair_list in PAIRS_BY_BASELINE.items():
            amps = []
            for tx, rx in pair_list:
                env_clean = pairs_disk[(tx, rx)] - pairs_ref[(tx, rx)]
                t_pred = predicted_time(tx, rx, R)
                amp = abs(np.interp(t_pred, t_arr, env_clean))
                amps.append(amp)
            amp_by_baseline[baseline_label] = np.mean(amps)
        results[r_label] = amp_by_baseline

    print(f"\n{'baseline category':<22}{'inner (R=41) amp':>18}{'outer (R=71) amp':>18}{'outer/inner ratio':>18}")
    for baseline_label in PAIRS_BY_BASELINE:
        a_in = results["inner (R=41)"][baseline_label]
        a_out = results["outer (R=71)"][baseline_label]
        print(f"{baseline_label:<22}{a_in:>18.6f}{a_out:>18.6f}{a_out/(a_in+1e-12):>18.3f}")

    print("\nIf the outer/inner ratio drops sharply from monostatic -> cross -> antipodal")
    print("(outer amplitude falling off faster with baseline angle than inner does),")
    print("that confirms the divergence hypothesis: larger circles concentrate reflected")
    print("energy near the monostatic direction more than smaller circles do.")
