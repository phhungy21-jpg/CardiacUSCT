"""Retests the bent-ray/eikonal correction (run -28, patient001 single
boundary, +48.6% improvement) on REAL anatomy: patient023 (strong ~45%
contraction), same patient jwave_test used as its hardest real case and
the same patient this project's reflection channel already validated
(runs -21 through -26).

Per user, choosing to retest bent-ray correction on "off-center heart or
patient023": this script does the patient023 side. Runs a FRESH
transmission-only simulation (this project has only ever run patient023
through the REFLECTION channel/pitch-catch so far -- no transmission
data exists yet) and applies the same straight-ray vs. bent-ray
comparison used on patient001's synthetic single-tissue phantom, using
the real two-tissue (myocardium+blood) contour. The bent-ray sound-speed
estimate uses a single guessed tissue speed (MYOCARDIUM's, since the
straight-ray SIRT image can't distinguish myocardium from blood any more
than it can distinguish anything else -- it just says "tissue" vs.
"water") -- same simplification as the other two tests, for a fair
three-way comparison.
"""

import numpy as np

from phase1_patient023_validation import (
    load_real_contours, build_medium_two_tissue, build_medium_water_only, MRI_NPZ, thetas,
)
from phase1_transmission_tomography_reconstruction import simulate_transmit_all_receivers
from phase1_bent_ray_correction import evaluate_bent_ray_correction
import phase2_config as cfg
import labels

import os

if __name__ == "__main__":
    print(f"*** {labels.PENDING_SIGNOFF_BANNER} ***")
    print(f"*** {labels.TOY_EXACT_GT_CAPTION} ***")
    print("BENT-RAY CORRECTION RETEST: patient023 real irregular two-tissue anatomy. Does the "
          "scikit-fmm eikonal fix (run -28, +48.6% on patient001's centered single boundary) "
          "generalize to real, irregular, two-tissue anatomy?")
    print("  compute estimate: 36 angles x 2 media (transmission only) = 72 forward sims "
          "-- ~15-20 minutes based on prior-run precedent")

    canvas_lv, canvas_myo, outer_contour_dom, inner_contour_dom = load_real_contours(MRI_NPZ)
    medium_water = build_medium_water_only()
    medium_phantom = build_medium_two_tissue(canvas_lv, canvas_myo)

    print("\n=== TRANSMISSION channel: water-only control, all 36 angles, all receivers ===")
    water_arrivals = {th: simulate_transmit_all_receivers(medium_water, th) for th in thetas}
    print("=== TRANSMISSION channel: patient023 two-tissue phantom, all 36 angles, all receivers ===")
    phantom_arrivals = {th: simulate_transmit_all_receivers(medium_phantom, th) for th in thetas}

    pairs_excess_delay_ns = {}
    for theta_tx in thetas:
        for theta_rx, t_water in water_arrivals[theta_tx].items():
            if theta_rx not in phantom_arrivals[theta_tx]:
                continue
            t_phantom = phantom_arrivals[theta_tx][theta_rx]
            pairs_excess_delay_ns[(theta_tx, theta_rx)] = (t_phantom - t_water) * 1e9

    print(f"  {len(pairs_excess_delay_ns)} transmission ray paths captured")
    os.makedirs("results", exist_ok=True)
    tt = np.array([k[0] for k in pairs_excess_delay_ns.keys()])
    tr = np.array([k[1] for k in pairs_excess_delay_ns.keys()])
    ed = np.array(list(pairs_excess_delay_ns.values()))
    np.savez("results/patient023_transmission_rays.npz", theta_tx=tt, theta_rx=tr, excess_delay_ns=ed)
    print("  saved results/patient023_transmission_rays.npz")

    result = evaluate_bent_ray_correction(pairs_excess_delay_ns, cfg.MYOCARDIUM.sound_speed, tag="patient023")
    print(f"\n--- Comparison across phantoms ---")
    print(f"  patient001 (centered, single boundary, synthetic): +48.6% improvement")
    print(f"  patient023 (real, irregular, two-tissue):           {result['improvement_pct']:+.1f}% improvement")
