"""Generalization test for run -43's win (speckle as a geometric path
constraint, corr=0.697 on patient023) -- per user, "proceed with both":
does the same speckle-constrained candidate-selection approach work on
a SECOND, independent patient (patient001, much milder ~1.6% real
contraction vs. patient023's ~18%, a genuinely different anatomy)?

Runs the two reflection-channel simulations needed (homogeneous
two-tissue phantom + speckle-injected myocardium variant, 36 angles
each = 72 forward sims) -- patient001 has NEVER had reflection-channel
data simulated in this project before (only transmission, runs -05/-06/
-07). Water-only baseline is REUSED from the already-cached patient023
run (geometry-only, patient-independent, no new simulation needed for
that half).
"""

import numpy as np

from phase1_patient023_validation import load_real_contours, build_medium_two_tissue
from phase1_speckle_patient023_sim import build_medium_speckle_two_tissue
from phase1_das_reflectivity_imaging import simulate_pitch_catch_raw_at
from phase1_reflection_channel_scout import thetas
import labels

import os

PATIENT_ID = "patient001"
MRI_NPZ = f"../jwave_test/results/mri_irregular_ring_{PATIENT_ID}_slice4.npz"

if __name__ == "__main__":
    print(f"*** {labels.PENDING_SIGNOFF_BANNER} ***")
    print(f"*** {labels.TOY_EXACT_GT_CAPTION} ***")
    print(f"PATIENT001 REFLECTION+SPECKLE SIMULATION: generalization test for run -43's "
          "speckle-constrained candidate selection win. Homogeneous two-tissue phantom + "
          "speckle-injected myocardium variant, 36 angles each. Water-only baseline reused "
          "from cache (geometry-only, patient-independent).")
    print("  compute estimate: 36 angles x 2 media (homogeneous, speckle) = 72 forward sims "
          "-- ~15-20 minutes based on prior-run precedent")

    canvas_lv, canvas_myo, outer_contour_dom, inner_contour_dom = load_real_contours(MRI_NPZ)
    medium_homogeneous = build_medium_two_tissue(canvas_lv, canvas_myo)
    medium_speckle = build_medium_speckle_two_tissue(canvas_lv, canvas_myo)

    print("\n=== Simulating patient001 HOMOGENEOUS two-tissue phantom, pitch-catch at 36 angles ===")
    homog_traces = [simulate_pitch_catch_raw_at(medium_homogeneous, th) for th in thetas]
    print("=== Simulating patient001 SPECKLE two-tissue phantom, pitch-catch at 36 angles ===")
    speckle_traces = [simulate_pitch_catch_raw_at(medium_speckle, th) for th in thetas]

    os.makedirs("results", exist_ok=True)
    np.savez("results/patient001_reflection_raw_traces.npz",
             thetas=thetas, homogeneous_traces=np.array(homog_traces), speckle_traces=np.array(speckle_traces),
             outer_contour_dom=outer_contour_dom, inner_contour_dom=inner_contour_dom)
    print("  saved results/patient001_reflection_raw_traces.npz")
