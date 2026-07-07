"""Phase 2 — canonical preprocessing shared by ACDC and (later) M&Ms.

One function maps a patient's raw NIfTI cine + ED/ES masks into a uniform
representation: fixed spacing, fixed orientation, fixed in-plane matrix
size, per-frame intensity normalization. All frames of a patient are
resampled onto the *same* physical grid so frame-to-frame correspondence
(needed for Phase 3 registration) is preserved.
"""

import configparser
from pathlib import Path

import numpy as np
import SimpleITK as sitk

TARGET_SPACING = (1.5625, 1.5625, 10.0)  # mm; mode of the ACDC cohort (see LOG.md run 2026-07-06-02)
TARGET_ORIENTATION = "LPS"
OUT_SIZE = (128, 128)  # in-plane (x, y)


def load_info(patient_dir: Path) -> dict:
    cfg_text = "[info]\n" + (patient_dir / "Info.cfg").read_text()
    parser = configparser.ConfigParser()
    parser.read_string(cfg_text)
    return dict(parser["info"])


def _orient_and_resample(image: sitk.Image, is_mask: bool) -> sitk.Image:
    image = sitk.DICOMOrient(image, TARGET_ORIENTATION)
    orig_spacing = image.GetSpacing()
    orig_size = image.GetSize()
    new_size = [
        max(1, int(round(orig_size[i] * orig_spacing[i] / TARGET_SPACING[i])))
        for i in range(3)
    ]
    resampler = sitk.ResampleImageFilter()
    resampler.SetOutputSpacing(TARGET_SPACING)
    resampler.SetSize(new_size)
    resampler.SetOutputOrigin(image.GetOrigin())
    resampler.SetOutputDirection(image.GetDirection())
    resampler.SetInterpolator(sitk.sitkNearestNeighbor if is_mask else sitk.sitkLinear)
    resampler.SetDefaultPixelValue(0)
    return resampler.Execute(image)


def _crop_or_pad(arr: np.ndarray, center_xy: tuple, out_size: tuple) -> np.ndarray:
    """arr is (Z, Y, X). Crop/pad the last two axes to out_size, centered at center_xy=(x,y)."""
    z, y, x = arr.shape
    out_y, out_x = out_size[1], out_size[0]
    cx, cy = center_xy

    y0 = int(round(cy - out_y / 2))
    x0 = int(round(cx - out_x / 2))

    result = np.zeros((z, out_y, out_x), dtype=arr.dtype)

    src_y0, src_y1 = max(0, y0), min(y, y0 + out_y)
    src_x0, src_x1 = max(0, x0), min(x, x0 + out_x)
    dst_y0, dst_x0 = src_y0 - y0, src_x0 - x0

    if src_y1 > src_y0 and src_x1 > src_x0:
        result[:, dst_y0:dst_y0 + (src_y1 - src_y0), dst_x0:dst_x0 + (src_x1 - src_x0)] = (
            arr[:, src_y0:src_y1, src_x0:src_x1]
        )
    return result


def _zscore(vol: np.ndarray) -> np.ndarray:
    mean, std = vol.mean(), vol.std()
    return (vol - mean) / std if std > 1e-8 else vol - mean


def preprocess_patient(patient_dir: Path) -> dict:
    """Returns a dict of processed arrays/metadata for one patient, or raises on failure."""
    pid = patient_dir.name
    info = load_info(patient_dir)
    ed_idx, es_idx = int(info["ed"]), int(info["es"])

    ed_img_raw = sitk.ReadImage(str(patient_dir / f"{pid}_frame{ed_idx:02d}.nii.gz"))
    ed_mask_raw = sitk.ReadImage(str(patient_dir / f"{pid}_frame{ed_idx:02d}_gt.nii.gz"))
    es_img_raw = sitk.ReadImage(str(patient_dir / f"{pid}_frame{es_idx:02d}.nii.gz"))
    es_mask_raw = sitk.ReadImage(str(patient_dir / f"{pid}_frame{es_idx:02d}_gt.nii.gz"))

    ed_img = _orient_and_resample(ed_img_raw, is_mask=False)
    ed_mask = _orient_and_resample(ed_mask_raw, is_mask=True)
    es_img = _orient_and_resample(es_img_raw, is_mask=False)
    es_mask = _orient_and_resample(es_mask_raw, is_mask=True)

    if ed_img.GetSize() != ed_mask.GetSize() or ed_img.GetSpacing() != ed_mask.GetSpacing():
        raise ValueError(f"{pid}: ED image/mask grid mismatch after resample")
    if es_img.GetSize() != es_mask.GetSize():
        raise ValueError(f"{pid}: ES image/mask grid mismatch after resample")

    ed_mask_arr = sitk.GetArrayFromImage(ed_mask)  # (Z, Y, X)
    ys, xs = np.nonzero(ed_mask_arr.sum(axis=0))
    if len(xs) == 0:
        raise ValueError(f"{pid}: ED mask is empty after resample")
    center_xy = (xs.mean(), ys.mean())

    ed_arr = _crop_or_pad(sitk.GetArrayFromImage(ed_img), center_xy, OUT_SIZE)
    es_arr = _crop_or_pad(sitk.GetArrayFromImage(es_img), center_xy, OUT_SIZE)
    ed_mask_arr = _crop_or_pad(ed_mask_arr, center_xy, OUT_SIZE)
    es_mask_arr = _crop_or_pad(sitk.GetArrayFromImage(es_mask), center_xy, OUT_SIZE)

    ed_arr = _zscore(ed_arr).astype(np.float32)
    es_arr = _zscore(es_arr).astype(np.float32)

    if np.isnan(ed_arr).any() or np.isnan(es_arr).any():
        raise ValueError(f"{pid}: NaNs after normalization")

    return {
        "patient_id": pid,
        "ed_idx": ed_idx,
        "es_idx": es_idx,
        "ed_frame": ed_arr,
        "es_frame": es_arr,
        "ed_mask": ed_mask_arr.astype(np.uint8),
        "es_mask": es_mask_arr.astype(np.uint8),
        "spacing": TARGET_SPACING,
        "crop_center_xy": center_xy,
    }
