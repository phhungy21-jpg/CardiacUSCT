"""Diagnostic: is the RV/myocardium Dice gap a fixable registration-quality
problem, or a Dice-is-unfair-to-thin-structures problem? Compares B-spline
(mean-squares) vs. diffeomorphic Demons, and Dice vs. average surface
distance, on the same 3 smoke-test patients."""

import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from registration import (  # noqa: E402
    dice_per_label,
    register_ed_to_es,
    register_ed_to_es_demons,
    surface_distance_per_label,
    warp,
)

PROC_DIR = Path("data/processed/ACDC")
LABEL_NAMES = {1: "RV", 2: "myo", 3: "LV"}


def report(pid: str, method: str, dice: dict, dist: dict, elapsed: float) -> None:
    dice_s = ", ".join(f"{LABEL_NAMES[k]}={v:.3f}" for k, v in sorted(dice.items()))
    dist_s = ", ".join(f"{LABEL_NAMES[k]}={v:.2f}mm" for k, v in sorted(dist.items()))
    print(f"{pid} [{method}] ({elapsed:.1f}s)  Dice: {dice_s}  |  AvgSurfDist: {dist_s}")


def main() -> None:
    for pid in ["patient001", "patient002", "patient101"]:
        d = np.load(PROC_DIR / f"{pid}.npz")
        spacing = tuple(d["spacing"])

        t0 = time.time()
        transform_bspline = register_ed_to_es(d["ed_frame"], d["es_frame"], spacing)
        elapsed_bspline = time.time() - t0
        warped_bspline = warp(d["ed_mask"], spacing, d["es_mask"], transform_bspline, is_mask=True)
        dice_bspline = dice_per_label(warped_bspline, d["es_mask"])
        dist_bspline = surface_distance_per_label(warped_bspline, d["es_mask"], spacing)
        report(pid, "bspline", dice_bspline, dist_bspline, elapsed_bspline)

        t0 = time.time()
        transform_demons = register_ed_to_es_demons(d["ed_frame"], d["es_frame"], spacing)
        elapsed_demons = time.time() - t0
        warped_demons = warp(d["ed_mask"], spacing, d["es_mask"], transform_demons, is_mask=True)
        dice_demons = dice_per_label(warped_demons, d["es_mask"])
        dist_demons = surface_distance_per_label(warped_demons, d["es_mask"], spacing)
        report(pid, "demons ", dice_demons, dist_demons, elapsed_demons)
        print()


if __name__ == "__main__":
    main()
