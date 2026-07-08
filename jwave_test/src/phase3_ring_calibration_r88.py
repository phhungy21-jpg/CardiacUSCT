"""Phase 3 — extending the curvature-weight calibration (run -44) with
a THIRD measurement point at R=88 cells, the radius patient023's real
outer (epicardial) boundary sits at after toy-rescaling
(`phase3_mri_irregular_ring_prep.py patient023`).

Per user: "log and calibrate and log" -- after the wide-probe-standoff
test (`phase3_mri_wide_probe_standoff_test.py`) confirmed patient023's
noisy outer-boundary fit is NOT a probe-standoff artifact (identical
result at 46-cell and 92-cell standoff), the remaining explanation is
that `pair_weight_at_R`'s linear model was only ever measured at R=41
and R=71 (run -44) -- R=88 requires extrapolation, which the model
clips to 0 (same as R=71), discarding whatever real cross/antipodal
signal actually exists at that radius. This measures it directly,
exact same method as run -44 (isolated myocardium disk, no competing
boundary, standard probe geometry -- already shown standoff-invariant),
rather than assuming the R=71 value still applies.
"""

import numpy as np

from phase3_backprojection_shape_fit_triangle import (
    capture_all_pairs, build_medium_homogeneous, _SRC, _RCV, t_arr,
    c_ref, dx, _ENVELOPE_GROUP_DELAY_S, labels,
)
from phase3_outer_boundary_diagnostic import build_medium_myocardium_disk
from phase3_ring_amplitude_divergence_test import PAIRS_BY_BASELINE, predicted_time

R_NEW = 88.0

if __name__ == "__main__":
    print(f"*** {labels.PENDING_SIGNOFF_BANNER} ***")
    print(f"*** {labels.TOY_EXACT_GT_CAPTION} ***")
    print(f"Calibration measurement at R={R_NEW} cells (patient023's real outer "
          f"boundary radius) -- isolated myocardium disk, same method as run -44's "
          f"R=41/71 calibration points, standard probe geometry (already shown "
          f"standoff-invariant by the wide-probe-standoff test).")

    pairs_ref = capture_all_pairs(build_medium_homogeneous())
    print(f"\n=== Capturing isolated myocardium disk: R={R_NEW} ===")
    pairs_disk = capture_all_pairs(build_medium_myocardium_disk(R_NEW))

    amp_by_baseline = {}
    for baseline_label, pair_list in PAIRS_BY_BASELINE.items():
        amps = []
        for tx, rx in pair_list:
            env_clean = pairs_disk[(tx, rx)] - pairs_ref[(tx, rx)]
            t_pred = predicted_time(tx, rx, R_NEW)
            amp = abs(np.interp(t_pred, t_arr, env_clean))
            amps.append(amp)
        amp_by_baseline[baseline_label] = np.mean(amps)

    mono = amp_by_baseline["monostatic (0deg)"]
    cross = amp_by_baseline["cross (90deg)"]
    antipodal = amp_by_baseline["antipodal (180deg)"]
    cross_ratio = cross / (mono + 1e-12)
    antipodal_ratio = antipodal / (mono + 1e-12)

    print(f"\n{'baseline category':<22}{'amplitude':>14}")
    for k, v in amp_by_baseline.items():
        print(f"{k:<22}{v:>14.6f}")

    print(f"\nR={R_NEW}: cross/mono ratio={cross_ratio:.4f}, antipodal/mono ratio={antipodal_ratio:.4f}")
    print(f"\nFor comparison, run -44's existing calibration points:")
    print(f"  R=41 (inner): cross/mono=0.136, antipodal/mono=0.045")
    print(f"  R=71 (outer): cross/mono=0.000, antipodal/mono=0.000")
    print(f"\nUpdate phase3_ring_curvature_weighted_fit.py's _CAL_R/_CAL_CROSS/_CAL_ANTIPODAL "
          f"with this new point: R={R_NEW}, cross={cross_ratio:.4f}, antipodal={antipodal_ratio:.4f}")
