"""Protocol Gate 1 (ACDC pre-registration check).

Load one ACDC patient, confirm image/mask alignment voxel-for-voxel at ED
and ES, note the labelled-vs-total frame count, and save overlay figures to
results/ for visual inspection. Run before any registration work.
"""

import configparser
from pathlib import Path

import matplotlib.pyplot as plt
import nibabel as nib
import numpy as np

PATIENT_DIR = Path("data/ACDC/ACDC/database/training/patient001")
OUT_DIR = Path("results/gate1")


def load_info(patient_dir: Path) -> dict:
    cfg_text = "[info]\n" + (patient_dir / "Info.cfg").read_text()
    parser = configparser.ConfigParser()
    parser.read_string(cfg_text)
    return dict(parser["info"])


def overlay(volume: np.ndarray, mask: np.ndarray, slice_idx: int, title: str, out_path: Path) -> None:
    fig, ax = plt.subplots(figsize=(5, 5))
    ax.imshow(volume[:, :, slice_idx], cmap="gray")
    ax.imshow(np.ma.masked_where(mask[:, :, slice_idx] == 0, mask[:, :, slice_idx]), cmap="jet", alpha=0.4)
    ax.set_title(title)
    ax.axis("off")
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    info = load_info(PATIENT_DIR)
    ed_frame = int(info["ed"])
    es_frame = int(info["es"])
    nb_frame = int(info["nbframe"])
    patient_id = PATIENT_DIR.name

    print(f"Patient: {patient_id}")
    print(f"Total frames in cine sequence: {nb_frame}")
    print(f"Labelled frames: ED={ed_frame}, ES={es_frame} (2 of {nb_frame} — {nb_frame - 2} unlabelled)")

    for label, frame in (("ED", ed_frame), ("ES", es_frame)):
        vol_path = PATIENT_DIR / f"{patient_id}_frame{frame:02d}.nii.gz"
        mask_path = PATIENT_DIR / f"{patient_id}_frame{frame:02d}_gt.nii.gz"

        vol_img = nib.load(vol_path)
        mask_img = nib.load(mask_path)

        vol = vol_img.get_fdata()
        mask = mask_img.get_fdata()

        assert vol.shape == mask.shape, f"{label}: volume/mask shape mismatch {vol.shape} vs {mask.shape}"
        assert vol_img.header.get_zooms() == mask_img.header.get_zooms(), f"{label}: spacing mismatch"
        assert np.allclose(vol_img.affine, mask_img.affine), f"{label}: affine (orientation) mismatch"

        print(f"{label} frame{frame:02d}: shape={vol.shape}, spacing(mm)={vol_img.header.get_zooms()}")
        print(f"{label}: image/mask share shape, spacing, and affine — voxel-for-voxel aligned")

        mid_slice = vol.shape[2] // 2
        out_path = OUT_DIR / f"{patient_id}_{label}_overlay.png"
        overlay(vol, mask, mid_slice, f"{patient_id} {label} (frame{frame:02d}, slice {mid_slice})", out_path)
        print(f"Saved overlay: {out_path}")


if __name__ == "__main__":
    main()
