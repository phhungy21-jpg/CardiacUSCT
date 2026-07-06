"""Gate 1 cross-cohort half (deferred from Phase 1) — now that M&Ms is
available: load one M&Ms patient, overlay ED+ES masks, confirm alignment,
and record spacing vs. ACDC."""

import csv
from pathlib import Path

import matplotlib.pyplot as plt
import nibabel as nib
import numpy as np

MANDMS_ROOT = Path("data/MandMs")
CSV_PATH = MANDMS_ROOT / "211230_M&Ms_Dataset_information_diagnosis_opendataset.csv"
OUT_DIR = Path("results/gate1_mands")


def find_patient_file(code: str) -> Path:
    for split in ("Training", "Validation", "Testing"):
        p = MANDMS_ROOT / split / f"{code}_sa.nii.gz"
        if p.exists():
            return p, split
    raise FileNotFoundError(code)


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    rows = list(csv.DictReader(open(CSV_PATH, encoding="utf-8")))
    row = next(r for r in rows if r["External code"] == "A1K2P5")
    code = row["External code"]
    ed_frame, es_frame = int(row["ED"]), int(row["ES"])

    img_path, split = find_patient_file(code)
    gt_path = img_path.parent / f"{code}_sa_gt.nii.gz"

    img = nib.load(img_path)
    gt = nib.load(gt_path)
    print(f"Patient {code} ({split}): full shape {img.shape}, spacing {img.header.get_zooms()}")
    print(f"ED frame={ed_frame}, ES frame={es_frame}")
    print(f"ACDC spacing mode was (1.5625, 1.5625, 10.0) mm for comparison")

    img_data = img.get_fdata()
    gt_data = gt.get_fdata()

    for label, frame in (("ED", ed_frame), ("ES", es_frame)):
        vol = img_data[..., frame]
        mask = gt_data[..., frame]
        mid_z = vol.shape[2] // 2
        fig, ax = plt.subplots(figsize=(5, 5))
        ax.imshow(vol[:, :, mid_z], cmap="gray")
        ax.imshow(np.ma.masked_where(mask[:, :, mid_z] == 0, mask[:, :, mid_z]), cmap="jet", alpha=0.4)
        ax.set_title(f"M&Ms {code} {label} (frame{frame}, slice {mid_z})")
        ax.axis("off")
        out_path = OUT_DIR / f"{code}_{label}_overlay.png"
        fig.savefig(out_path, dpi=150, bbox_inches="tight")
        plt.close(fig)
        print(f"Saved {out_path}")


if __name__ == "__main__":
    main()
