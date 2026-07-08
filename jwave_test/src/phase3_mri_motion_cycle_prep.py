"""Phase 3 — data preparation for a REAL, registration-derived motion
cycle on the MRI-derived irregular ring (escalation from runs -47/-48's
single static real-shape frame).

Per user: after confirming "8 consecutive MRI SLICES" would be the
wrong axis (that's base->apex anatomy at one instant, not motion over
time -- ACDC's ground-truth segmentation only exists at 2 real
timepoints, ED and ES, per the Phase I pilot's established floor), the
agreed approach is to INTERPOLATE between the two real ED/ES contours
using the Phase I registration-derived ED->ES displacement field
(`pilot/data/processed/ACDC_reg/patient001.npz`), applied at fractional
strength across 8 phases (same half-cosine ED->ES->ED schedule as the
synthetic toy, `phase3_config.lv_radius_at_phase`), instead of a
synthetic formula.

DISPLACEMENT FIELD CONVENTION (verified against `pilot/src/registration.py`
before trusting it, not assumed): `displacement_field[z]` is a (Y,X,3)
field defined ON THE ES GRID, in mm, order (dz,dy,dx)
(`pilot/src/doppler.py`'s established convention). For each ES-grid
point p, displacement(p) = ED_location - ES_location, i.e.
warped_ED(p) = ED_mask(p + displacement(p)) ~= ES_mask(p) -- this is
exactly what `pilot/src/run_phase3_registration.py` computed and
Dice/surface-distance-validated already (Gate 3, Phase I). Applying a
FRACTION f of this same displacement (f=0 -> ED, f=1 -> ES) via
`scipy.ndimage.map_coordinates` (order=0, i.e. nearest-neighbor, per
CLAUDE.md's mask-resampling rule) gives an intermediate-phase mask.
**Self-consistency check before trusting this** (this project's
standing discipline): the f=1 result is compared directly against the
already Dice-validated `warped_ed_mask` field saved in the same npz --
if they don't closely match, the sign/convention was gotten wrong and
must be fixed before proceeding.

GROUND-TRUTH CAVEAT (must be carried into every result from this point
on, per `labels.GT_FLOOR_CAPTION`): unlike every phantom test so far in
this thread (circle, triangle, heart-cartoon, ring, and even the static
real-shape test in run -48), the "true" motion here is NOT exactly
known -- it is Phase I's registration output, itself imperfect
(patient001: mean_dice=0.784, myocardium dice=0.789, slightly below the
pilot's own pre-registered 0.80 Gate-3 threshold; LV dice=0.926, surface
distances 0.20mm (LV) / 0.53mm (myocardium)). Any acoustic-recovery
error reported against this motion cannot be smaller than this floor
without it being a coincidence, not a demonstration of superior
accuracy.

TEMPLATE-VS-TRUTH DESIGN (the actual point of this escalation): the
acoustic reconstruction fits a single SCALE FACTOR against the FIXED
ED real-shape template (`phase3_mri_irregular_ring_prep.py`'s r(theta),
unchanged across phases) -- deliberately NOT re-fit to each phase's own
true (non-uniformly deformed) shape, which would trivially "recover"
by construction. This tests the real, useful question: does real
cardiac motion resemble a uniform contraction/relaxation of a fixed
shape well enough for a single scale parameter to track it adequately,
or does genuine non-uniform wall motion break that approximation? Both
outcomes are informative; this script does not presuppose which.
"""

import numpy as np
import sys

from scipy import ndimage
from skimage import measure

from matplotlib import pyplot as plt
import os

PATIENT_ID = sys.argv[1] if len(sys.argv) > 1 else "patient001"
PATIENT_NPZ = f"../pilot/data/processed/ACDC/{PATIENT_ID}.npz"
REG_NPZ = f"../pilot/data/processed/ACDC_reg/{PATIENT_ID}.npz"
DX_M = 0.1e-3
TARGET_LV_RADIUS_CELLS = 60.0
N_PHASES = 8


def contraction_fraction(phase: float) -> float:
    """0 at phase=0 (ED), 1 at phase=0.5 (full ES), back to 0 at phase=1 --
    same half-cosine SHAPE as phase3_config.lv_radius_at_phase, applied
    here to the registration displacement's fractional strength instead
    of a synthetic radius."""
    return 0.5 * (1 - np.cos(2 * np.pi * phase))


def warp_mask_fraction(mask, dy_px, dx_px, frac):
    """Sample `mask` (defined on the ED grid) at (row + frac*dy_px,
    col + frac*dx_px) -- i.e. the ES-grid displacement field pulled back
    by a fraction -- via nearest-neighbor (order=0, mask-safe)."""
    rows, cols = np.mgrid[0:mask.shape[0], 0:mask.shape[1]].astype(np.float64)
    coords = np.stack([rows + frac * dy_px, cols + frac * dx_px])
    return ndimage.map_coordinates(mask.astype(np.float64), coords, order=0, mode="nearest") >= 0.5


if __name__ == "__main__":
    print(f"Loading real ACDC {PATIENT_ID} segmentation + registration-derived displacement field...")
    d_seg = np.load(PATIENT_NPZ)
    d_reg = np.load(REG_NPZ)
    mask3d = d_seg["ed_mask"]
    spacing = d_seg["spacing"]  # (row_mm, col_mm, slice_mm)
    disp_field = d_reg["displacement_field"]  # (Z, Y, X, 3) mm, (dz, dy, dx), ED->ES, defined on ES grid
    print(f"  registration quality (Gate 3, Phase I): mean_dice={float(d_reg['mean_dice']):.3f}, "
          f"myo_dice={float(d_reg['dice_myo']):.3f}, lv_dice={float(d_reg['dice_lv']):.3f}, "
          f"myo_surf_dist={float(d_reg['dist_myo']):.2f}mm, lv_surf_dist={float(d_reg['dist_lv']):.2f}mm")
    if float(d_reg["mean_dice"]) < 0.80:
        print("  NOTE: mean_dice is below the pilot's own pre-registered 0.80 Gate-3 threshold -- "
              "this patient's registration is usable but imperfect; carry this caveat forward.")

    lv_areas_ed = [(mask3d[z] == 3).sum() for z in range(mask3d.shape[0])]
    SLICE_Z = int(np.argmax(lv_areas_ed))  # same mid-ventricular convention as run -47's prep script
    print(f"  selected slice {SLICE_Z} (max LV area, {lv_areas_ed[SLICE_Z]} px)")

    slice2d = mask3d[SLICE_Z]
    myo_ed = (slice2d == 2)
    lv_ed = (slice2d == 3)

    dy_mm = disp_field[SLICE_Z, ..., 1]
    dx_mm = disp_field[SLICE_Z, ..., 2]
    dz_mm = disp_field[SLICE_Z, ..., 0]
    dy_px = dy_mm / spacing[0]
    dx_px = dx_mm / spacing[1]
    print(f"  in-plane displacement magnitude at this slice: mean={np.sqrt(dy_mm**2+dx_mm**2).mean():.2f}mm, "
          f"max={np.sqrt(dy_mm**2+dx_mm**2).max():.2f}mm "
          f"(through-plane dz mean={np.abs(dz_mm).mean():.2f}mm, NOT modeled -- 2D phantom, flagged limitation)")

    # Self-consistency check (per project discipline: verify before trusting)
    warped_at_f1 = warp_mask_fraction(slice2d, dy_px, dx_px, 1.0)
    official_warped = d_reg["warped_ed_mask"][SLICE_Z] > 0  # any nonzero label
    agreement = (warped_at_f1 == official_warped).mean()
    print(f"  cross-check: this script's f=1.0 warp vs. pilot's own Dice-validated warped_ed_mask: "
          f"{agreement*100:.1f}% pixel agreement (should be high, e.g. >95%, if convention is correct)")
    assert agreement > 0.90, "displacement-field convention mismatch -- do not proceed without fixing this"

    phases = np.linspace(0, 1, N_PHASES)
    fractions = [contraction_fraction(p) for p in phases]

    # Fixed rescale zoom factor, computed ONCE from the ED frame (run -47's
    # value) -- reused for every phase so all 8 frames share the same
    # physical scale (a genuinely different physical scale per frame would
    # not be physically meaningful and would also invalidate the fixed
    # ED-derived r(theta) template's units).
    lv_area_mm2 = lv_ed.sum() * (spacing[0] * spacing[1])
    real_lv_radius_mm = np.sqrt(lv_area_mm2 / np.pi)
    target_lv_radius_mm = TARGET_LV_RADIUS_CELLS * DX_M * 1e3
    zoom_factor = (spacing[0] / (DX_M * 1e3)) * (target_lv_radius_mm / real_lv_radius_mm)
    smooth_sigma = zoom_factor / 2.0
    print(f"  fixed zoom_factor={zoom_factor:.3f}x (from ED frame, run -47), smooth_sigma={smooth_sigma:.2f} cells")

    myo_frames, lv_frames, ring_frames = [], [], []
    outer_contours, inner_contours = [], []
    true_inner_radius_mm, true_outer_radius_mm = [], []

    for i, (phase, frac) in enumerate(zip(phases, fractions)):
        myo_native = warp_mask_fraction(myo_ed, dy_px, dx_px, frac)
        lv_native = warp_mask_fraction(lv_ed, dy_px, dx_px, frac)
        ring_native = myo_native | lv_native

        ring_up = ndimage.zoom(ring_native.astype(np.uint8), zoom_factor, order=0)
        lv_up = ndimage.zoom(lv_native.astype(np.uint8), zoom_factor, order=0)
        myo_up = ndimage.zoom(myo_native.astype(np.uint8), zoom_factor, order=0)

        ring_smooth = ndimage.gaussian_filter(ring_up.astype(float), sigma=smooth_sigma) >= 0.5
        lv_smooth = ndimage.gaussian_filter(lv_up.astype(float), sigma=smooth_sigma) >= 0.5
        myo_smooth = ndimage.gaussian_filter(myo_up.astype(float), sigma=smooth_sigma) >= 0.5

        outer_c = max(measure.find_contours(ring_smooth.astype(np.uint8), 0.5), key=len)
        inner_c = max(measure.find_contours(lv_smooth.astype(np.uint8), 0.5), key=len)

        # Mean radius from the BOUNDARY (contour points), not all filled
        # interior pixels -- a filled disk's mean pixel-to-centroid distance
        # is ~(2/3)R, not R; must match run -47/48's contour-based convention.
        ys, xs = np.where(ring_smooth)
        lys, lxs = np.where(lv_smooth)
        ring_centroid = (ys.mean(), xs.mean())
        lv_centroid = (lys.mean(), lxs.mean())
        outer_mean_r_cells = np.sqrt((outer_c[:, 0] - ring_centroid[0])**2 + (outer_c[:, 1] - ring_centroid[1])**2).mean()
        inner_mean_r_cells = np.sqrt((inner_c[:, 0] - lv_centroid[0])**2 + (inner_c[:, 1] - lv_centroid[1])**2).mean()

        myo_frames.append(myo_smooth)
        lv_frames.append(lv_smooth)
        ring_frames.append(ring_smooth)
        outer_contours.append(outer_c)
        inner_contours.append(inner_c)
        true_inner_radius_mm.append(inner_mean_r_cells * DX_M * 1e3)
        true_outer_radius_mm.append(outer_mean_r_cells * DX_M * 1e3)

        print(f"  phase {i+1}/{N_PHASES} (frac={frac:.2f}): "
              f"inner mean-r={true_inner_radius_mm[-1]:.2f}mm, outer mean-r={true_outer_radius_mm[-1]:.2f}mm")

    out_npz = f"results/mri_motion_cycle_{PATIENT_ID}_slice{SLICE_Z}.npz"
    np.savez(out_npz,
             myo_frames=np.array(myo_frames), lv_frames=np.array(lv_frames), ring_frames=np.array(ring_frames),
             outer_contours=np.array(outer_contours, dtype=object), inner_contours=np.array(inner_contours, dtype=object),
             true_inner_radius_mm=np.array(true_inner_radius_mm), true_outer_radius_mm=np.array(true_outer_radius_mm),
             fractions=np.array(fractions), phases=phases, zoom_factor=zoom_factor, smooth_sigma=smooth_sigma,
             mean_dice=float(d_reg["mean_dice"]), myo_dice=float(d_reg["dice_myo"]), lv_dice=float(d_reg["dice_lv"]))
    print(f"Saved {out_npz}")

    # Visualize: 8-frame filmstrip, sanity check before running the (expensive) acoustic sim
    n_cols = 4
    n_rows = int(np.ceil(N_PHASES / n_cols))
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(3.2 * n_cols, 3.4 * n_rows))
    axes = np.array(axes).reshape(-1)
    for i, ax in enumerate(axes[:N_PHASES]):
        ax.imshow(np.abs(ring_frames[i]), cmap="hot", origin="upper")
        ax.plot(inner_contours[i][:, 1], inner_contours[i][:, 0], "c-", linewidth=1)
        ax.plot(outer_contours[i][:, 1], outer_contours[i][:, 0], "lime", linewidth=1)
        ax.set_title(f"phase={phases[i]:.2f}, frac={fractions[i]:.2f}\n"
                     f"in_r={true_inner_radius_mm[i]:.2f}mm out_r={true_outer_radius_mm[i]:.2f}mm", fontsize=8)
        ax.axis("off")
    for ax in axes[N_PHASES:]:
        ax.axis("off")
    fig.suptitle(f"Real registration-derived motion cycle (ACDC {PATIENT_ID}, slice {SLICE_Z})\n"
                 f"mean_dice={float(d_reg['mean_dice']):.3f} (Phase I Gate 3 floor -- imperfect, not exact ground truth)",
                 fontsize=10)
    plt.tight_layout(rect=[0, 0.02, 1, 0.90])
    os.makedirs("results/figures", exist_ok=True)
    out_fig = f"results/figures/phase3_mri_motion_cycle_prep_filmstrip_{PATIENT_ID}.png"
    plt.savefig(out_fig, dpi=130)
    print(f"Saved {out_fig}")
