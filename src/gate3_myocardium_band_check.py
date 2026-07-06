"""Is the myocardium's Dice drop uniform, or concentrated away from the LV
endocardial surface? Splits the true ES myocardium into an "endocardial
band" (near the LV cavity, tightly coupled to the now-accurate LV fit) and
the rest (epicardial surface / septal-RV side), and computes Dice for each
band separately, using the mask-guided run's saved warped masks."""

from pathlib import Path

import numpy as np
import SimpleITK as sitk

PROC_DIR = Path("data/processed/ACDC")
REG_DIR = Path("data/processed/ACDC_reg")  # currently holds the mask-guided (LV-only) run's output

LV_LABEL = 3
MYO_LABEL = 2
ENDO_BAND_MM = 3.0  # ~2 voxels at 1.5625mm in-plane spacing


def band_dice(true_myo: np.ndarray, warped_myo: np.ndarray, band_mask: np.ndarray) -> float:
    t = np.logical_and(true_myo, band_mask)
    w = np.logical_and(warped_myo, band_mask)
    denom = t.sum() + w.sum()
    if denom == 0:
        return float("nan")
    return float(2 * np.logical_and(t, w).sum() / denom)


def main() -> None:
    files = sorted(PROC_DIR.glob("patient*.npz"))
    endo_dices, other_dices, whole_myo_dices = [], [], []

    for f in files:
        pid = f.stem
        proc = np.load(f)
        reg = np.load(REG_DIR / f"{pid}.npz")
        spacing = tuple(proc["spacing"])

        es_mask = proc["es_mask"]
        warped_mask = reg["warped_ed_mask"]

        lv_bin = (es_mask == LV_LABEL).astype(np.uint8)
        lv_img = sitk.GetImageFromArray(lv_bin)
        lv_img.SetSpacing(spacing)
        dist_to_lv = sitk.GetArrayFromImage(
            sitk.SignedMaurerDistanceMap(lv_img, insideIsPositive=False, squaredDistance=False, useImageSpacing=True)
        )

        true_myo = es_mask == MYO_LABEL
        warped_myo = warped_mask == MYO_LABEL
        endo_band = np.logical_and(true_myo, dist_to_lv <= ENDO_BAND_MM)
        other_band = np.logical_and(true_myo, dist_to_lv > ENDO_BAND_MM)

        endo_dices.append(band_dice(true_myo, warped_myo, endo_band))
        other_dices.append(band_dice(true_myo, warped_myo, other_band))

        denom = true_myo.sum() + warped_myo.sum()
        whole_myo_dices.append(float(2 * np.logical_and(true_myo, warped_myo).sum() / denom) if denom > 0 else float("nan"))

    endo_dices = np.array(endo_dices)
    other_dices = np.array(other_dices)
    whole_myo_dices = np.array(whole_myo_dices)

    print(f"n={len(files)} patients")
    print(f"Whole myocardium Dice:        mean={np.nanmean(whole_myo_dices):.3f} std={np.nanstd(whole_myo_dices):.3f}")
    print(f"Endocardial band (<= {ENDO_BAND_MM}mm from LV) Dice: mean={np.nanmean(endo_dices):.3f} std={np.nanstd(endo_dices):.3f}")
    print(f"Rest of myocardium (epicardial/septal-RV side) Dice: mean={np.nanmean(other_dices):.3f} std={np.nanstd(other_dices):.3f}")
    print(f"Patients with endo band Dice >= 0.80: {(endo_dices >= 0.8).mean():.2%}")
    print(f"Patients with 'other' band Dice >= 0.80: {(other_dices >= 0.8).mean():.2%}")


if __name__ == "__main__":
    main()
