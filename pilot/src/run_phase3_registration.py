"""Phase 3 driver — ED->ES registration for every ACDC patient.

Saves, per patient, into data/processed/ACDC_reg/patientXXX.npz:
  - displacement_field: (Z, Y, X, 3) dense field, ED->ES, mm
  - warped_ed_mask, dice per label (for the Gate 3 check)
This is the ground-truth motion Phase 4 will project into synthetic
Doppler signals. M&Ms is not touched (Phase 6 only).
"""

import csv
import sys
import time
from pathlib import Path

import numpy as np
import SimpleITK as sitk

sys.path.insert(0, str(Path(__file__).resolve().parent))
from registration import (  # noqa: E402
    _to_sitk,
    dice_per_label,
    register_ed_to_es_demons,
    surface_distance_per_label,
    warp,
)

PROC_DIR = Path("data/processed/ACDC")
OUT_DIR = Path("data/processed/ACDC_reg")
DICE_THRESHOLD = 0.80  # pre-registered, see LOG.md run 2026-07-06-04


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    files = sorted(PROC_DIR.glob("patient*.npz"))
    rows = []
    failures = []

    for i, f in enumerate(files):
        pid = f.stem
        d = np.load(f)
        spacing = tuple(d["spacing"])
        t0 = time.time()
        try:
            transform = register_ed_to_es_demons(d["ed_frame"], d["es_frame"], spacing)
            elapsed = time.time() - t0

            warped_mask = warp(d["ed_mask"], spacing, d["es_mask"], transform, is_mask=True)
            dice = dice_per_label(warped_mask, d["es_mask"])
            dist = surface_distance_per_label(warped_mask, d["es_mask"], spacing)
            mean_dice = float(np.mean(list(dice.values())))

            es_ref = _to_sitk(d["es_mask"], spacing)
            disp_filter = sitk.TransformToDisplacementFieldFilter()
            disp_filter.SetReferenceImage(es_ref)
            disp_field = sitk.GetArrayFromImage(disp_filter.Execute(transform))  # (Z, Y, X, 3)

            np.savez_compressed(
                OUT_DIR / f"{pid}.npz",
                displacement_field=disp_field.astype(np.float32),
                warped_ed_mask=warped_mask.astype(np.uint8),
                dice_rv=dice.get(1, np.nan),
                dice_myo=dice.get(2, np.nan),
                dice_lv=dice.get(3, np.nan),
                mean_dice=mean_dice,
                dist_rv=dist.get(1, np.nan),
                dist_myo=dist.get(2, np.nan),
                dist_lv=dist.get(3, np.nan),
                spacing=spacing,
                split=str(d["split"]),
            )
            rows.append({
                "patient_id": pid, "split": str(d["split"]), "seconds": round(elapsed, 1),
                "dice_rv": dice.get(1, np.nan), "dice_myo": dice.get(2, np.nan), "dice_lv": dice.get(3, np.nan),
                "mean_dice": mean_dice, "gate_pass": mean_dice >= DICE_THRESHOLD,
                "dist_rv_mm": dist.get(1, np.nan), "dist_myo_mm": dist.get(2, np.nan), "dist_lv_mm": dist.get(3, np.nan),
            })
        except Exception as e:
            failures.append((pid, str(e)))

        if (i + 1) % 10 == 0:
            print(f"  ... {i + 1}/{len(files)} done", flush=True)

    csv_path = Path("results/phase3_dice.csv")
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    with open(csv_path, "w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    mean_dices = [r["mean_dice"] for r in rows]
    n_pass = sum(r["gate_pass"] for r in rows)
    print(f"\nRegistered {len(rows)}/{len(files)} patients ({len(failures)} failures)")
    print(f"Mean Dice across cohort: {np.mean(mean_dices):.3f} +/- {np.std(mean_dices):.3f}")
    print(f"Patients passing Dice >= {DICE_THRESHOLD}: {n_pass}/{len(rows)}")
    print(f"Per-patient results written to {csv_path}")
    for pid, err in failures:
        print(f"  ! {pid}: {err}")


if __name__ == "__main__":
    main()
