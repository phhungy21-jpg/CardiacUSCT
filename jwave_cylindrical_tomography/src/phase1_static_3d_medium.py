"""Phase 1: builds a genuine 3D acoustic medium from patient023's REAL,
single-timestamp (ED) 3D segmentation -- a 9-slice stack
(`pilot/data/processed/ACDC/patient023.npz`), RV cavity / myocardium /
LV cavity labels (NOT a full anatomic 4-chamber view; ACDC has no atria
segmentation -- same 3-tissue convention as `jwave_ring_tomography`
throughout).

The real, non-trivial problem this solves: ACDC's native spacing is
STRONGLY ANISOTROPIC (1.5625mm in-plane, 10mm slice thickness) --
resampling onto a uniform acoustic grid (dx TBD, pending the GPU timing
sweep, run -03) needs care. Per this repo's own standing rule (CLAUDE.md):
"Masks resample with nearest-neighbor, never linear/cubic -- protocol
pitfall, silently corrupts labels if violated" -- respected here via
`scipy.ndimage.zoom(..., order=0)`.

This does NOT yet build a jWave Medium object or run any simulation --
that needs the final dx decision from run -03's GPU timing sweep. This
validates the resampling pipeline in isolation first, at a placeholder
dx, so the (separate, real) question of "does the anatomy survive
resampling sensibly" is checked before any compute-heavy step.
"""

import numpy as np
from scipy import ndimage

PATIENT023_NPZ = "../pilot/data/processed/ACDC/patient023.npz"


def load_patient023_ed_volume():
    """CAUGHT AND FIXED: the stored `spacing` array is in (row, col,
    slice) order, but `ed_mask`'s array axes are (slice, row, col) --
    using the stored order directly gives a physically absurd extent
    (14mm total height, 1280mm in-plane). Verified by checking BOTH
    possible correspondences against physical plausibility for a
    cardiac short-axis stack: reversing the spacing array to (slice,
    row, col) = (10.0, 1.5625, 1.5625) gives 90mm x 200mm x 200mm --
    sensible; the as-stored order does not."""
    d = np.load(PATIENT023_NPZ, allow_pickle=True)
    mask = d["ed_mask"]
    native_spacing_mm = np.array(d["spacing"])[::-1]  # reorder to match mask's (slice,row,col) axes
    return mask, native_spacing_mm


def resample_mask_isotropic(mask, native_spacing_mm, target_dx_mm):
    """Resamples a labeled mask (integer labels, NOT continuous data) onto
    an isotropic grid at `target_dx_mm`, using NEAREST-NEIGHBOR
    interpolation only (order=0) -- linear/cubic would blend adjacent
    labels into meaningless intermediate values, silently corrupting the
    segmentation (this repo's own standing pitfall warning)."""
    zoom_factors = native_spacing_mm / target_dx_mm
    resampled = ndimage.zoom(mask, zoom_factors, order=0, mode="nearest")
    return resampled


if __name__ == "__main__":
    print("PHASE 1 STATIC 3D MEDIUM: resampling patient023's real 9-slice ED volume "
          "(anisotropic 1.5625x1.5625x10mm) onto an isotropic grid.")

    mask, native_spacing = load_patient023_ed_volume()
    assert 60 < mask.shape[0] * native_spacing[0] < 150, \
        "through-plane extent looks physically implausible for a cardiac stack -- check spacing axis order again"
    print(f"  native volume shape: {mask.shape}, native spacing: {native_spacing} mm")
    print(f"  native physical extent: {np.array(mask.shape) * native_spacing} mm")

    for target_dx_mm in [0.2, 0.5, 1.0]:
        resampled = resample_mask_isotropic(mask, native_spacing, target_dx_mm)
        labels_before = {int(l): int((mask == l).sum()) for l in np.unique(mask)}
        labels_after = {int(l): int((resampled == l).sum()) for l in np.unique(resampled)}
        # sanity: resampled voxel COUNT ratio per label should roughly match the volume
        # ratio (native_voxel_volume / target_voxel_volume), not be wildly off
        native_vox_vol = np.prod(native_spacing)
        target_vox_vol = target_dx_mm ** 3
        expected_ratio = native_vox_vol / target_vox_vol
        print(f"\n  target_dx={target_dx_mm}mm -> resampled shape={resampled.shape} "
              f"({resampled.size:,} voxels, expected_voxel_ratio~{expected_ratio:.1f}x)")
        for label, name in [(1, "RV"), (2, "MYO"), (3, "LV")]:
            before_vol_mm3 = labels_before.get(label, 0) * native_vox_vol
            after_vol_mm3 = labels_after.get(label, 0) * target_vox_vol
            pct_diff = (after_vol_mm3 - before_vol_mm3) / before_vol_mm3 * 100 if before_vol_mm3 else float("nan")
            print(f"    {name}: native volume={before_vol_mm3:.0f}mm3, "
                  f"resampled volume={after_vol_mm3:.0f}mm3 ({pct_diff:+.1f}%)")
