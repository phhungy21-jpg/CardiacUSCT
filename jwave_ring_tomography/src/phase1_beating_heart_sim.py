"""Reverse-engineering experiment, per user: take a REAL BEATING ACDC
cross section, "blast it with US (forward projecting), record it," for
later use training a simple CNN and testing whether it generalizes to
an "unleaked" (held-out) cross section.

Simulates the REFLECTION channel (pitch-catch) at all 8 phases of
patient001's cardiac motion cycle (TRAINING patient) and all 8 phases of
patient023's cycle (TEST patient -- genuinely never touched during
training; patient023 also has much stronger real contraction, ~18% vs
patient001's ~1.6%, an intentionally harder generalization test).

Compute-saving reuse: the water-only baseline trace only depends on
probe geometry (fixed across this whole project, never the phantom), so
the ALREADY-cached 36-angle water-only traces
(`results/patient023_reflection_raw_traces.npz`) are reused directly --
subsampled to 18 angles (every other of the original 36) to control
compute for this already-large 16-phase batch, no water-only
resimulation needed at all.
"""

import numpy as np

from phase1_beating_heart_medium import load_motion_cycle, load_phase_contours, build_medium_two_tissue
from phase1_das_reflectivity_imaging import simulate_pitch_catch_raw_at
from phase1_reflection_channel_scout import thetas as thetas_full
import labels

import os

N_ANGLE_SUBSET = 18
ANGLE_INDICES = list(range(0, 36, 36 // N_ANGLE_SUBSET))  # every other of the 36
THETAS = thetas_full[ANGLE_INDICES]

PATIENTS = ["patient001", "patient023"]

if __name__ == "__main__":
    print(f"*** {labels.PENDING_SIGNOFF_BANNER} ***")
    print(f"*** {labels.TOY_EXACT_GT_CAPTION} ***")
    print("BEATING HEART SIMULATION: reflection channel (pitch-catch) at all 8 cardiac-cycle "
          f"phases for patient001 (TRAIN) and patient023 (TEST, held out entirely), "
          f"{N_ANGLE_SUBSET} angles each (subsampled from the usual 36 to control compute). "
          "Water-only baseline REUSED from cache -- no resimulation of that half.")
    n_new_sims = len(PATIENTS) * 8 * N_ANGLE_SUBSET
    print(f"  compute estimate: {len(PATIENTS)} patients x 8 phases x {N_ANGLE_SUBSET} angles "
          f"= {n_new_sims} forward sims -- ~{n_new_sims/72*17.5:.0f} minutes based on the "
          "72-sim-per-~17.5-min precedent used throughout this project")

    os.makedirs("results", exist_ok=True)
    for patient_id in PATIENTS:
        d = load_motion_cycle(patient_id)
        n_phases = d["myo_frames"].shape[0]
        print(f"\n=== {patient_id}: {n_phases} phases, fractions={d['fractions']} ===")

        traces_list = []
        inner_contours_dom = []
        outer_contours_dom = []
        for phase_idx in range(n_phases):
            canvas_lv, canvas_myo, outer_dom, inner_dom = load_phase_contours(d, phase_idx)
            medium = build_medium_two_tissue(canvas_lv, canvas_myo)
            print(f"  phase {phase_idx} (fraction={d['fractions'][phase_idx]:.3f}): simulating {N_ANGLE_SUBSET} angles...")
            phase_traces = [simulate_pitch_catch_raw_at(medium, th) for th in THETAS]
            traces_list.append(np.array(phase_traces))
            inner_contours_dom.append(inner_dom)
            outer_contours_dom.append(outer_dom)

        traces_arr = np.stack(traces_list, axis=0)  # (n_phases, N_ANGLE_SUBSET, n_samples)
        out_path = f"results/beating_{patient_id}_reflection_traces.npz"
        np.savez(out_path, traces=traces_arr, thetas=THETAS, fractions=d["fractions"],
                 inner_contours_dom=np.array(inner_contours_dom, dtype=object),
                 outer_contours_dom=np.array(outer_contours_dom, dtype=object))
        print(f"  saved {out_path}  (traces shape: {traces_arr.shape})")
