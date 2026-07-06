"""Quick timing/correctness check on a few patients before the full Phase 3 run."""

import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from registration import dice_per_label, register_ed_to_es, warp  # noqa: E402

PROC_DIR = Path("data/processed/ACDC")


def main() -> None:
    test_patients = ["patient001", "patient002", "patient101"]
    for pid in test_patients:
        d = np.load(PROC_DIR / f"{pid}.npz")
        spacing = tuple(d["spacing"])
        t0 = time.time()
        transform = register_ed_to_es(d["ed_frame"], d["es_frame"], spacing)
        elapsed = time.time() - t0

        warped_mask = warp(d["ed_mask"], spacing, d["es_mask"], transform, is_mask=True)
        dice = dice_per_label(warped_mask, d["es_mask"])
        print(f"{pid}: registration took {elapsed:.1f}s, per-label Dice = {dice}, mean = {np.mean(list(dice.values())):.3f}")


if __name__ == "__main__":
    main()
