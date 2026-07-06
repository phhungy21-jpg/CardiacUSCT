"""Test mask-guided (Option A) registration on the worst-3 patients found
in the full-cohort Demons run, plus the 3 originally-good smoke-test
patients, to check: does it fix LV Dice, and do the substitute validations
(diffeomorphism, intensity residual) still look sane?"""

import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from registration import (  # noqa: E402
    dice_per_label,
    diffeomorphism_check,
    register_ed_to_es_demons,
    register_ed_to_es_mask_guided,
    surface_distance_per_label,
    warp,
    warped_intensity_residual,
)

PROC_DIR = Path("data/processed/ACDC")
TEST_PATIENTS = ["patient142", "patient104", "patient029", "patient001", "patient002", "patient101"]
LABEL_NAMES = {1: "RV", 2: "myo", 3: "LV"}


def main() -> None:
    for pid in TEST_PATIENTS:
        d = np.load(PROC_DIR / f"{pid}.npz")
        spacing = tuple(d["spacing"])
        ed_arr, es_arr = d["ed_frame"], d["es_frame"]
        ed_mask, es_mask = d["ed_mask"], d["es_mask"]

        t0 = time.time()
        transform_old = register_ed_to_es_demons(ed_arr, es_arr, spacing)
        old_dice = dice_per_label(warp(ed_mask, spacing, es_mask, transform_old, True), es_mask)
        old_elapsed = time.time() - t0

        t0 = time.time()
        transform_new = register_ed_to_es_mask_guided(
            ed_arr, es_arr, ed_mask, es_mask, spacing, label_weights={1: 1.0, 2: 1.0, 3: 2.0}
        )
        new_elapsed = time.time() - t0
        warped_new = warp(ed_mask, spacing, es_mask, transform_new, True)
        new_dice = dice_per_label(warped_new, es_mask)
        new_dist = surface_distance_per_label(warped_new, es_mask, spacing)
        diffeo = diffeomorphism_check(transform_new, es_mask, spacing)
        residual = warped_intensity_residual(ed_arr, es_arr, spacing, transform_new)

        old_s = ", ".join(f"{LABEL_NAMES[k]}={v:.3f}" for k, v in sorted(old_dice.items()))
        new_s = ", ".join(f"{LABEL_NAMES[k]}={v:.3f}" for k, v in sorted(new_dice.items()))
        dist_s = ", ".join(f"{LABEL_NAMES[k]}={v:.2f}mm" for k, v in sorted(new_dist.items()))
        print(f"{pid}:")
        print(f"  old (intensity-only)  ({old_elapsed:.1f}s) Dice: {old_s}")
        print(f"  new (mask-guided)     ({new_elapsed:.1f}s) Dice: {new_s}  |  SurfDist: {dist_s}")
        print(f"  diffeomorphism: frac_nonpositive_jac={diffeo['frac_nonpositive_jacobian']:.4f} "
              f"min_jac={diffeo['min_jacobian']:.3f} max_jac={diffeo['max_jacobian']:.3f}")
        print(f"  warped-intensity residual (mean abs diff, z-scored units): {residual:.3f}")
        print()


if __name__ == "__main__":
    main()
