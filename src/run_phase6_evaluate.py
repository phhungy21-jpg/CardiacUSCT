"""Phase 6 — the actual result. ONE clean evaluation of the frozen Phase 5
model (trained on ACDC only) on the held-out M&Ms cohort. No tuning here —
whatever comes out is reported as-is."""

import csv
import sys
from pathlib import Path

import numpy as np
import SimpleITK as sitk
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
from phase5_model import SmallMotionCNN  # noqa: E402
from registration import dice_per_label, warp  # noqa: E402
from run_phase5_cv import endpoint_error_mm  # noqa: E402
from seed import set_all_seeds  # noqa: E402

DOPPLER_DIR = Path("data/processed/MandMs_doppler")
PROC_DIR = Path("data/processed/MandMs")
MODEL_PATH = Path("results/phase5_final_model.pt")


def warped_dice_for_patient(pred_xy, ed_mask, es_mask, spacing) -> dict:
    z, h, w = pred_xy.shape[:3]
    disp_field = np.zeros((z, h, w, 3), dtype=np.float64)
    disp_field[..., 2] = pred_xy[..., 0]
    disp_field[..., 1] = pred_xy[..., 1]
    vec_img = sitk.GetImageFromArray(disp_field, isVector=True)
    vec_img.SetSpacing(spacing)
    transform = sitk.DisplacementFieldTransform(vec_img)
    warped_mask = warp(ed_mask, spacing, es_mask, transform, is_mask=True)
    return dice_per_label(warped_mask, es_mask)


def main() -> None:
    set_all_seeds(42)
    model = SmallMotionCNN()
    model.load_state_dict(torch.load(MODEL_PATH))
    model.eval()

    files = sorted(DOPPLER_DIR.glob("*.npz"))
    rows = []

    for f in files:
        pid = f.stem
        dop = np.load(f)
        proc = np.load(PROC_DIR / f"{pid}.npz")

        proj = torch.tensor(dop["proj_noisy"], dtype=torch.float32).permute(1, 0, 2, 3)
        with torch.no_grad():
            pred = model(proj).permute(0, 2, 3, 1).numpy()

        true_xy = dop["target_xy"]
        heart_mask = (proc["es_mask"] > 0).astype(np.float32)
        spacing = tuple(dop["spacing"])

        err = endpoint_error_mm(pred, true_xy, heart_mask)
        dice = warped_dice_for_patient(pred, proc["ed_mask"], proc["es_mask"], spacing)

        rows.append({
            "patient_id": pid, "vendor": str(dop["vendor"]),
            "endpoint_err_mm": err, "mean_dice": float(np.mean(list(dice.values()))),
            "dice_rv": dice.get(1, np.nan), "dice_myo": dice.get(2, np.nan), "dice_lv": dice.get(3, np.nan),
        })

    csv_path = Path("results/phase6_evaluation.csv")
    with open(csv_path, "w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    errs = np.array([r["endpoint_err_mm"] for r in rows])
    dices = np.array([r["mean_dice"] for r in rows])

    print(f"=== Phase 6: M&Ms evaluation, {len(rows)} patients ===")
    print(f"Endpoint error (mm): {errs.mean():.3f} +/- {errs.std():.3f} (median {np.median(errs):.3f})")
    print(f"Warped-mask mean Dice: {dices.mean():.3f} +/- {dices.std():.3f}")
    print(f"\n=== Side-by-side with ACDC in-distribution (Phase 5 CV) ===")
    print(f"ACDC:  endpoint error 0.849 +/- 0.227 mm, Dice 0.664 +/- 0.104 (n=150)")
    print(f"M&Ms:  endpoint error {errs.mean():.3f} +/- {errs.std():.3f} mm, Dice {dices.mean():.3f} +/- {dices.std():.3f} (n={len(rows)})")
    print(f"Gap (endpoint error): {errs.mean() - 0.849:+.3f} mm ({(errs.mean()/0.849 - 1)*100:+.1f}%)")
    print(f"Gap (Dice): {dices.mean() - 0.664:+.3f}")

    print("\nBy vendor:")
    vendors = {}
    for r in rows:
        vendors.setdefault(r["vendor"], []).append(r["endpoint_err_mm"])
    for v, vals in sorted(vendors.items()):
        vals = np.array(vals)
        print(f"  Vendor {v}: n={len(vals)}, endpoint error {vals.mean():.3f} +/- {vals.std():.3f} mm")

    print(f"\nResults written to {csv_path}")


if __name__ == "__main__":
    main()
