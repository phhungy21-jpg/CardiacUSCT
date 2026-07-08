"""Phase 3 — data preparation for the MRI-derived irregular myocardial
ring stress test (escalation from the smooth eccentric ring, run -46).

Per user: "maybe reconstruct the irregular ring from one of the mri, a
sound escalation from smooth eccentric off center rings." Real anatomy
source: ACDC patient001 (already used once before in this project for
the Phase 4.1 benchmark, `phase4_benchmark.py` -- known to have a
genuine, non-trivial myocardium/LV boundary at N=350, per run -11's
staircasing check).

Loads the real ED-frame segmentation (`pilot/data/processed/ACDC/
patient001.npz`), picks a representative mid-ventricular slice (LV
cross-section near its maximum area, avoiding basal/apical slices where
the ring is incomplete or absent), isolates myocardium (label 2) + LV
cavity (label 3) -- RV (label 1) is excluded for this first test to keep
the phantom a direct 2-tissue-boundary analog of the ring tests already
validated (runs -39 through -46), not a 3-chamber model.

RESCALING STRATEGY (deliberate choice, not incidental): rather than
simulating at real anatomical scale (which the earlier run -09 CPU
benchmark showed is infeasible locally at dx=0.1mm, and would also
invalidate run -44's curvature-weighting calibration, measured at
specific toy-scale radii 41/71 cells), the real contour's SHAPE is
preserved exactly (isotropic rescale only, no distortion) while its
SIZE is rescaled to match this thread's established toy scale (target
LV radius ~60 cells / 6mm at ED, matching `phase3_config.py`'s existing
schedule) -- isolating the "does real anatomical IRREGULARITY break the
method" question from "does a completely different physical scale
change the physics," which would need its own separate recalibration.

Resampling is nearest-neighbor only throughout (CLAUDE.md's explicit
rule for mask data), combining the native-pixel-to-acoustic-grid zoom
and the toy-rescale zoom into a single `scipy.ndimage.zoom(..., order=0)`
call.

SMOOTHING (added per user request -- "stick with 1 first ... apply
smoothing, mimicking true tissue-like irregularities"): the raw
nearest-neighbor-upsampled mask has visible pixel-block staircasing
inherited from the native 1.5625mm/px segmentation -- at this zoom
factor (~2.5x) that staircase is finer than real tissue texture but
coarser than acoustically-meaningless single-grid-cell noise, so left
untreated it would partly test the reconstruction method against a
sampling-grid artifact rather than genuine anatomical curvature
variation. A light Gaussian smoothing (sigma tied to the native pixel
size in upsampled-grid units, i.e. sigma = zoom_factor / 2, so the
smoothing kernel spans about one native pixel) is applied to the float
mask post-zoom, then re-thresholded at 0.5 -- the same precedent as
run -11's `PROXY_AUDIT.md` staircasing check (sigma=2 cells / 0.2mm
Gaussian smoothing of a tissue map, 0.59% relative L2 wavefield
difference, judged acceptable). This is a smoothing POST-PROCESS step
applied after nearest-neighbor resampling, not a replacement of it --
CLAUDE.md's nearest-neighbor rule governs how label masks are resampled
across grids, which still happens via `order=0` zoom; smoothing the
final binary boundary afterward is a distinct, deliberate step to
approximate a real (non-jagged) tissue boundary. Both raw and smoothed
contours are kept and saved, so the acoustic phantom build (next step)
can use the smoothed version while the raw version remains available
for comparison/diagnostics.
"""

import sys

import numpy as np
from scipy import ndimage
from skimage import measure

from matplotlib import pyplot as plt
import os

PATIENT_ID = sys.argv[1] if len(sys.argv) > 1 else "patient001"
PATIENT_NPZ = f"../pilot/data/processed/ACDC/{PATIENT_ID}.npz"
DX_M = 0.1e-3  # acoustic grid spacing, matches phase2_config.DX_M
TARGET_LV_RADIUS_CELLS = 60.0  # matches phase3_config.LV_RADIUS_ED_CELLS


def pick_midventricular_slice(mask):
    """Slice with LV cavity area nearest its own maximum across the
    volume -- avoids basal (RV-dominated) and apical (tiny/absent LV)
    slices where the ring boundary is incomplete."""
    lv_areas = [(mask[z] == 3).sum() for z in range(mask.shape[0])]
    return int(np.argmax(lv_areas)), lv_areas


if __name__ == "__main__":
    print(f"Loading real ACDC {PATIENT_ID} ED-frame segmentation...")
    d = np.load(PATIENT_NPZ)
    mask3d = d["ed_mask"]
    spacing = d["spacing"]  # (row_mm, col_mm, slice_mm)
    print(f"  volume shape={mask3d.shape}, in-plane spacing={spacing[0]:.4f}mm/px")

    z, lv_areas = pick_midventricular_slice(mask3d)
    print(f"  per-slice LV pixel counts: {lv_areas}")
    print(f"  selected slice {z} (LV area={lv_areas[z]} px)")

    slice2d = mask3d[z]
    myo_mask = (slice2d == 2)
    lv_mask = (slice2d == 3)
    ring_mask = myo_mask | lv_mask  # combined myocardium + LV cavity, RV excluded

    # Real-world equivalent LV radius (from area), native units.
    lv_area_mm2 = lv_mask.sum() * (spacing[0] * spacing[1])
    real_lv_radius_mm = np.sqrt(lv_area_mm2 / np.pi)
    print(f"  real LV area={lv_area_mm2:.1f}mm^2, equivalent radius={real_lv_radius_mm:.2f}mm")

    # Combined zoom: native-pixel-to-acoustic-grid AND toy-scale rescale,
    # in one nearest-neighbor resample (no intermediate interpolation).
    target_lv_radius_mm = TARGET_LV_RADIUS_CELLS * DX_M * 1e3
    zoom_factor = (spacing[0] / (DX_M * 1e3)) * (target_lv_radius_mm / real_lv_radius_mm)
    print(f"  target LV radius={target_lv_radius_mm:.2f}mm, combined zoom factor={zoom_factor:.3f}x")

    ring_upsampled = ndimage.zoom(ring_mask.astype(np.uint8), zoom_factor, order=0)
    lv_upsampled = ndimage.zoom(lv_mask.astype(np.uint8), zoom_factor, order=0)
    myo_upsampled = ndimage.zoom((slice2d == 2).astype(np.uint8), zoom_factor, order=0)
    print(f"  upsampled shape: {ring_upsampled.shape}")

    # Smoothing post-process (see docstring): remove native-pixel staircase
    # while preserving genuine anatomical curvature, mimicking a real
    # (non-jagged) tissue boundary rather than a blocky pixel silhouette.
    smooth_sigma = zoom_factor / 2.0
    ring_smooth = ndimage.gaussian_filter(ring_upsampled.astype(float), sigma=smooth_sigma) >= 0.5
    lv_smooth = ndimage.gaussian_filter(lv_upsampled.astype(float), sigma=smooth_sigma) >= 0.5
    myo_smooth = ndimage.gaussian_filter(myo_upsampled.astype(float), sigma=smooth_sigma) >= 0.5
    print(f"  smoothing: sigma={smooth_sigma:.2f} cells (~1 native pixel)")

    # Extract true boundary contours (for later use as the known-ground-
    # truth shape in the acoustic-reconstruction fit) -- outer = myo+LV
    # combined boundary, inner = LV cavity boundary alone. Raw (staircased)
    # contours kept for comparison; smoothed contours are the ones intended
    # for the acoustic phantom build.
    outer_contours_raw = measure.find_contours(ring_upsampled, 0.5)
    inner_contours_raw = measure.find_contours(lv_upsampled, 0.5)
    outer_contour_raw = max(outer_contours_raw, key=len)
    inner_contour_raw = max(inner_contours_raw, key=len)

    outer_contours = measure.find_contours(ring_smooth.astype(np.uint8), 0.5)
    inner_contours = measure.find_contours(lv_smooth.astype(np.uint8), 0.5)
    outer_contour = max(outer_contours, key=len)
    inner_contour = max(inner_contours, key=len)
    print(f"  smoothed outer contour: {len(outer_contour)} points, inner contour: {len(inner_contour)} points")

    # Verify the rescaled LV radius lands close to the target (sanity check).
    # Checked on the smoothed mask since that's what feeds the acoustic build.
    lv_area_upsampled_cells = lv_smooth.sum()
    achieved_radius_cells = np.sqrt(lv_area_upsampled_cells / np.pi)
    print(f"  achieved LV radius after rescale+smoothing: {achieved_radius_cells:.1f} cells "
          f"(target {TARGET_LV_RADIUS_CELLS:.1f}, error {abs(achieved_radius_cells-TARGET_LV_RADIUS_CELLS):.1f} cells)")

    out_npz = f"results/mri_irregular_ring_{PATIENT_ID}_slice{z}.npz"
    np.savez(out_npz,
             ring_mask=ring_smooth, lv_mask=lv_smooth, myo_mask=myo_smooth,
             ring_mask_raw=ring_upsampled, lv_mask_raw=lv_upsampled, myo_mask_raw=myo_upsampled,
             outer_contour=outer_contour, inner_contour=inner_contour,
             outer_contour_raw=outer_contour_raw, inner_contour_raw=inner_contour_raw,
             smooth_sigma=smooth_sigma,
             zoom_factor=zoom_factor, real_lv_radius_mm=real_lv_radius_mm,
             achieved_radius_cells=achieved_radius_cells)
    print(f"Saved {out_npz}")

    # Visualize before committing to any acoustic simulation.
    fig, axes = plt.subplots(1, 3, figsize=(16, 5.5))
    axes[0].imshow(slice2d, cmap="viridis", origin="upper")
    axes[0].set_title(f"Native ACDC {PATIENT_ID} slice {z}\n(1=RV, 2=myo, 3=LV), {spacing[0]:.2f}mm/px", fontsize=10)
    axes[0].axis("off")

    axes[1].imshow(np.abs(ring_upsampled), cmap="hot", origin="upper")
    axes[1].plot(inner_contour_raw[:, 1], inner_contour_raw[:, 0], "c-", linewidth=1.0, label="LV cavity (inner)")
    axes[1].plot(outer_contour_raw[:, 1], outer_contour_raw[:, 0], "lime", linewidth=1.0, label="epicardium (outer)")
    axes[1].set_title(f"Raw rescale (zoom={zoom_factor:.2f}x, order=0)\n"
                       f"nearest-neighbor staircase visible", fontsize=10)
    axes[1].legend(fontsize=8)
    axes[1].axis("off")

    axes[2].imshow(np.abs(ring_smooth), cmap="hot", origin="upper")
    axes[2].plot(inner_contour[:, 1], inner_contour[:, 0], "c-", linewidth=1.5, label="LV cavity (inner)")
    axes[2].plot(outer_contour[:, 1], outer_contour[:, 0], "lime", linewidth=1.5, label="epicardium (outer)")
    axes[2].set_title(f"Smoothed (sigma={smooth_sigma:.2f} cells)\n"
                       f"LV radius: real={real_lv_radius_mm:.1f}mm -> toy={achieved_radius_cells*DX_M*1e3:.2f}mm", fontsize=10)
    axes[2].legend(fontsize=8)
    axes[2].axis("off")

    plt.tight_layout()
    os.makedirs("results/figures", exist_ok=True)
    out_fig = f"results/figures/phase3_mri_irregular_ring_prep_{PATIENT_ID}.png"
    plt.savefig(out_fig, dpi=140)
    print(f"Saved {out_fig}")
