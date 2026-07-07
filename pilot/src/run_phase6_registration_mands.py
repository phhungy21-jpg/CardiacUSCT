"""Phase 3-equivalent for M&Ms — same frozen method as ACDC (intensity-only
diffeomorphic Demons, src/registration.py::register_ed_to_es_demons), used
ONLY to produce the ground-truth displacement field needed to evaluate the
ACDC-trained model's predictions. This is not training or tuning on M&Ms."""

import csv
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from registration import dice_per_label, register_ed_to_es_demons, surface_distance_per_label, warp  # noqa: E402

PROC_DIR = Path("data/processed/MandMs")
OUT_DIR = Path("data/processed/MandMs_reg")


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    files = sorted(PROC_DIR.glob("*.npz"))
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

            import SimpleITK as sitk
            es_ref = sitk.GetImageFromArray(d["es_mask"])
            es_ref.SetSpacing(spacing)
            disp_filter = sitk.TransformToDisplacementFieldFilter()
            disp_filter.SetReferenceImage(es_ref)
            disp_field = sitk.GetArrayFromImage(disp_filter.Execute(transform))

            np.savez_compressed(
                OUT_DIR / f"{pid}.npz",
                displacement_field=disp_field.astype(np.float32),
                dice_rv=dice.get(1, np.nan), dice_myo=dice.get(2, np.nan), dice_lv=dice.get(3, np.nan),
                mean_dice=mean_dice, spacing=spacing,
                split=str(d["split"]), vendor=str(d["vendor"]),
            )
            rows.append({
                "patient_id": pid, "vendor": str(d["vendor"]), "seconds": round(elapsed, 1),
                "dice_rv": dice.get(1, np.nan), "dice_myo": dice.get(2, np.nan), "dice_lv": dice.get(3, np.nan),
                "mean_dice": mean_dice,
                "dist_rv_mm": dist.get(1, np.nan), "dist_myo_mm": dist.get(2, np.nan), "dist_lv_mm": dist.get(3, np.nan),
            })
        except Exception as e:
            failures.append((pid, str(e)))

        if (i + 1) % 25 == 0:
            print(f"  ... {i + 1}/{len(files)} done", flush=True)

    csv_path = Path("results/phase6_mands_dice.csv")
    with open(csv_path, "w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    mean_dices = [r["mean_dice"] for r in rows]
    print(f"\nRegistered {len(rows)}/{len(files)} M&Ms patients ({len(failures)} failures)")
    print(f"Mean Dice: {np.mean(mean_dices):.3f} +/- {np.std(mean_dices):.3f}")
    for pid, err in failures:
        print(f"  ! {pid}: {err}")


if __name__ == "__main__":
    main()
