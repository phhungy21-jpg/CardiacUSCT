"""Phase 2-equivalent preprocessing for M&Ms, reusing the exact frozen
canonical representation from `preprocessing.py` (same TARGET_SPACING,
TARGET_ORIENTATION, OUT_SIZE) so ACDC and M&Ms land in an identical
representation — required for Gate 2's cross-cohort check and for the
model (trained on ACDC's representation) to be applicable at all.

Two M&Ms-specific handling differences from ACDC:
- Single 4D file per patient (cine + GT), not separate per-frame files —
  ED/ES frames are extracted by index, read from the metadata CSV (indices
  are NOT assumed to be 0/some fixed frame — confirmed they vary per
  patient, e.g. ED=33, ES=11 for one patient).
- Label remap: M&Ms uses 1=LV, 2=myocardium, 3=RV — the OPPOSITE of ACDC's
  1=RV, 2=myo, 3=LV (labels 1 and 3 swapped). Confirmed by geometry (see
  LOG.md run 2026-07-06-18) before writing this code. Remapped here so all
  downstream per-label code (registration Dice, LV-centered probe geometry)
  means the same anatomical structure in both cohorts.
"""

import csv
import sys
from pathlib import Path

import numpy as np
import SimpleITK as sitk

sys.path.insert(0, str(Path(__file__).resolve().parent))
from preprocessing import OUT_SIZE, TARGET_SPACING, _crop_or_pad, _orient_and_resample, _zscore  # noqa: E402

MANDMS_ROOT = Path("data/MandMs")
CSV_PATH = MANDMS_ROOT / "211230_M&Ms_Dataset_information_diagnosis_opendataset.csv"
LABEL_REMAP = {1: 3, 2: 2, 3: 1}  # M&Ms (LV,myo,RV) -> ACDC convention (RV,myo,LV)


def remap_labels(mask: np.ndarray) -> np.ndarray:
    out = np.zeros_like(mask)
    for src, dst in LABEL_REMAP.items():
        out[np.round(mask) == src] = dst
    return out


def load_metadata() -> list:
    return list(csv.DictReader(open(CSV_PATH, encoding="utf-8")))


def find_patient_file(code: str) -> tuple:
    for split in ("Training", "Validation", "Testing"):
        p = MANDMS_ROOT / split / f"{code}_sa.nii.gz"
        if p.exists():
            return p, split
    raise FileNotFoundError(code)


def extract_3d_frame(img4d: sitk.Image, frame_idx: int) -> sitk.Image:
    size = list(img4d.GetSize())
    size[3] = 0
    index = [0, 0, 0, frame_idx]
    return sitk.Extract(img4d, size, index)


def resolve_ed_es_indices(gt4d: sitk.Image, ed_idx: int, es_idx: int, code: str) -> tuple:
    """Some M&Ms CSV rows have a placeholder ED=ES=0 instead of real frame
    indices (confirmed for 25 patients, all with CSV ED=ES=0 while the GT
    volume's actual nonzero frames are elsewhere — see LOG.md). Falls back
    to detecting the two nonzero-mask frames directly and assigning ED/ES
    by LV cavity size (ED = larger/relaxed cavity, ES = smaller/contracted)
    when the CSV value doesn't correspond to any labelled frame."""
    n_frames = gt4d.GetSize()[3]

    def frame_nonzero(t):
        f = extract_3d_frame(gt4d, t)
        return np.count_nonzero(sitk.GetArrayFromImage(f))

    if frame_nonzero(ed_idx) > 0 and frame_nonzero(es_idx) > 0 and ed_idx != es_idx:
        return ed_idx, es_idx

    nonzero_frames = [t for t in range(n_frames) if frame_nonzero(t) > 0]
    if len(nonzero_frames) != 2:
        raise ValueError(f"{code}: expected exactly 2 labelled frames as fallback, found {len(nonzero_frames)}")

    lv_label_raw = 1  # M&Ms convention pre-remap
    counts = []
    for t in nonzero_frames:
        f = sitk.GetArrayFromImage(extract_3d_frame(gt4d, t))
        counts.append(np.count_nonzero(np.round(f) == lv_label_raw))
    ed = nonzero_frames[0] if counts[0] > counts[1] else nonzero_frames[1]
    es = nonzero_frames[1] if counts[0] > counts[1] else nonzero_frames[0]
    return ed, es


def preprocess_mands_patient(row: dict) -> dict:
    code = row["External code"]
    img_path, split = find_patient_file(code)
    gt_path = img_path.parent / f"{code}_sa_gt.nii.gz"

    img4d = sitk.ReadImage(str(img_path))
    gt4d = sitk.ReadImage(str(gt_path))

    ed_idx, es_idx = resolve_ed_es_indices(gt4d, int(row["ED"]), int(row["ES"]), code)

    ed_img_raw = extract_3d_frame(img4d, ed_idx)
    es_img_raw = extract_3d_frame(img4d, es_idx)
    ed_mask_raw = extract_3d_frame(gt4d, ed_idx)
    es_mask_raw = extract_3d_frame(gt4d, es_idx)

    ed_img = _orient_and_resample(ed_img_raw, is_mask=False)
    es_img = _orient_and_resample(es_img_raw, is_mask=False)
    ed_mask = _orient_and_resample(ed_mask_raw, is_mask=True)
    es_mask = _orient_and_resample(es_mask_raw, is_mask=True)

    ed_mask_arr = remap_labels(sitk.GetArrayFromImage(ed_mask))
    es_mask_arr = remap_labels(sitk.GetArrayFromImage(es_mask))

    # Matches ACDC's convention exactly (preprocessing.py:preprocess_patient): center on the
    # whole-heart mask (any nonzero label), not LV specifically — required for Gate 2 parity.
    ys, xs = np.nonzero(ed_mask_arr.sum(axis=0))
    if len(xs) == 0:
        raise ValueError(f"{code}: ED mask empty after resample")
    center_xy = (xs.mean(), ys.mean())

    ed_arr = _crop_or_pad(sitk.GetArrayFromImage(ed_img), center_xy, OUT_SIZE)
    es_arr = _crop_or_pad(sitk.GetArrayFromImage(es_img), center_xy, OUT_SIZE)
    ed_mask_arr = _crop_or_pad(ed_mask_arr, center_xy, OUT_SIZE)
    es_mask_arr = _crop_or_pad(es_mask_arr, center_xy, OUT_SIZE)

    ed_arr = _zscore(ed_arr).astype(np.float32)
    es_arr = _zscore(es_arr).astype(np.float32)

    if np.isnan(ed_arr).any() or np.isnan(es_arr).any():
        raise ValueError(f"{code}: NaNs after normalization")

    return {
        "patient_id": code,
        "ed_idx": ed_idx,
        "es_idx": es_idx,
        "ed_frame": ed_arr,
        "es_frame": es_arr,
        "ed_mask": ed_mask_arr.astype(np.uint8),
        "es_mask": es_mask_arr.astype(np.uint8),
        "spacing": TARGET_SPACING,
        "split": split,
        "vendor": row["Vendor"],
        "centre": row["Centre"],
        "pathology": row["Pathology"],
    }
