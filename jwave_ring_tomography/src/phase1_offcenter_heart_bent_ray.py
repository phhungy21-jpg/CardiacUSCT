"""Retests the bent-ray/eikonal correction (run -28, patient001 single
boundary, +48.6% improvement) on HARDER anatomy: the exact off-center
concave heart phantom that broke jwave_test's sparse-probe blind
reconstruction (runs -70/-72/-73), already re-tested through this
project's channels in run -27 (reflection RMSE=1.3047mm).

Per user, choosing to retest bent-ray correction on "off-center heart or
patient023": this script does the off-center heart side. Runs a FRESH
transmission-only simulation (no cached rays exist for this phantom --
run -27 only ran reflection channel raw traces for accuracy, plus a
transmission SIRT image used just for visualization, not saved to disk)
and applies the same straight-ray vs. bent-ray comparison used on
patient001, to see whether the +48.6% improvement generalizes to an
irregular, off-center, concave boundary shape (the exact case designed
to be hard).
"""

import numpy as np

from phase1_offcenter_heart_blind_test import build_medium_heart, build_medium_water_only, HEART_R
from phase1_reflection_channel_scout import thetas
from phase1_transmission_tomography_reconstruction import simulate_transmit_all_receivers
from phase1_bent_ray_correction import evaluate_bent_ray_correction
import phase2_config as cfg
import labels

import os

if __name__ == "__main__":
    print(f"*** {labels.PENDING_SIGNOFF_BANNER} ***")
    print(f"*** {labels.TOY_EXACT_GT_CAPTION} ***")
    print("BENT-RAY CORRECTION RETEST: off-center concave heart phantom (same exact shape as "
          "run -27, HEART_R=50, offset (10,-15)). Does the scikit-fmm eikonal fix (run -28, "
          "+48.6% on patient001's centered single boundary) generalize to an off-center, "
          "concave, irregular boundary -- the harder case this whole project exists to test?")
    print("  compute estimate: 36 angles x 2 media (transmission only) = 72 forward sims "
          "-- ~15-20 minutes based on prior-run precedent")

    medium_water = build_medium_water_only()
    medium_phantom = build_medium_heart(HEART_R)

    print("\n=== TRANSMISSION channel: water-only control, all 36 angles, all receivers ===")
    water_arrivals = {th: simulate_transmit_all_receivers(medium_water, th) for th in thetas}
    print("=== TRANSMISSION channel: heart phantom, all 36 angles, all receivers ===")
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
    np.savez("results/offcenter_heart_rays.npz", theta_tx=tt, theta_rx=tr, excess_delay_ns=ed)
    print("  saved results/offcenter_heart_rays.npz")

    result = evaluate_bent_ray_correction(pairs_excess_delay_ns, cfg.MYOCARDIUM.sound_speed, tag="offcenter_heart")
    print(f"\n--- Comparison to patient001 (run -28) ---")
    print(f"  patient001 (centered, single boundary): +48.6% improvement")
    print(f"  off-center concave heart:                {result['improvement_pct']:+.1f}% improvement")
