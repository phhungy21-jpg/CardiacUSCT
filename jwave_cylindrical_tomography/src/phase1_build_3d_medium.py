"""Builds the actual cropped, resampled 3D heterogeneous medium for
patient023's static (ED) volume at this project's design point (run
-05): N=150, domain=130mm, dx=0.867mm, f0=0.8MHz.

Crops to a bounding box around the real tissue (bbox measured in run
-05: 80x90.6x85.9mm) plus a water-bath margin, rather than the full
200x200mm MRI field of view (which would be far larger than needed).
Resamples via nearest-neighbor only (mask data, per this repo's
standing rule) onto the target isotropic grid.

Tissue properties reused UNCHANGED from `jwave_ring_tomography/src/
phase2_config.py` (verified against Mast 2000's primary source this
session) -- sound speed/density are frequency-independent in this
simple model, so lowering F0_HZ for this project does not require new
tissue-property citations, only a new attenuation-frequency-scaling
check later (attenuation IS frequency-dependent, not addressed yet).
"""

import numpy as np
from scipy import ndimage

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "jwave_ring_tomography", "src"))
import phase2_config as cfg

PATIENT023_NPZ = "../pilot/data/processed/ACDC/patient023.npz"

N_GRID = 150
DOMAIN_MM = 130.0
DX_MM = DOMAIN_MM / N_GRID
DX_M = DX_MM * 1e-3
F0_HZ = 0.8e6  # this project's own design point (run -05), NOT jwave_ring_tomography's 2.5MHz


def load_and_crop_patient023(margin_mm=20.0):
    d = np.load(PATIENT023_NPZ, allow_pickle=True)
    mask = d["ed_mask"]
    native_spacing_mm = np.array(d["spacing"])[::-1]  # (slice,row,col), see run -04's fix

    tissue = mask > 0
    coords = np.argwhere(tissue)
    mins, maxs = coords.min(axis=0), coords.max(axis=0)
    # add margin (in voxels, per-axis, since spacing is anisotropic)
    margin_vox = np.ceil(margin_mm / native_spacing_mm).astype(int)
    mins = np.maximum(mins - margin_vox, 0)
    maxs = np.minimum(maxs + margin_vox, np.array(mask.shape) - 1)
    cropped = mask[mins[0]:maxs[0]+1, mins[1]:maxs[1]+1, mins[2]:maxs[2]+1]
    return cropped, native_spacing_mm


def resample_to_target(cropped_mask, native_spacing_mm, target_n=N_GRID):
    """Resamples the cropped mask onto an N x N x N isotropic grid at
    DX_MM, nearest-neighbor only (label data)."""
    zoom_factors = native_spacing_mm / DX_MM
    resampled = ndimage.zoom(cropped_mask, zoom_factors, order=0, mode="nearest")
    # pad/crop to exactly N_GRID^3 (resampled shape from zoom won't be exactly N^3 in general)
    out = np.zeros((target_n, target_n, target_n), dtype=resampled.dtype)
    src_shape = np.array(resampled.shape)
    tgt_shape = np.array(out.shape)
    # center the resampled volume within the target grid
    start_src = np.maximum((src_shape - tgt_shape) // 2, 0)
    start_tgt = np.maximum((tgt_shape - src_shape) // 2, 0)
    copy_shape = np.minimum(src_shape - start_src, tgt_shape - start_tgt)
    out[start_tgt[0]:start_tgt[0]+copy_shape[0],
        start_tgt[1]:start_tgt[1]+copy_shape[1],
        start_tgt[2]:start_tgt[2]+copy_shape[2]] = \
        resampled[start_src[0]:start_src[0]+copy_shape[0],
                  start_src[1]:start_src[1]+copy_shape[1],
                  start_src[2]:start_src[2]+copy_shape[2]]
    return out


def build_sound_speed_density_maps(resampled_mask):
    sound_speed = np.where(resampled_mask == 2, cfg.MYOCARDIUM.sound_speed,
                   np.where((resampled_mask == 1) | (resampled_mask == 3), cfg.BLOOD.sound_speed,
                            cfg.WATER.sound_speed)).astype(np.float32)
    density = np.where(resampled_mask == 2, cfg.MYOCARDIUM.density,
               np.where((resampled_mask == 1) | (resampled_mask == 3), cfg.BLOOD.density,
                        cfg.WATER.density)).astype(np.float32)
    return sound_speed, density


if __name__ == "__main__":
    print(f"BUILD 3D MEDIUM: N={N_GRID}, domain={DOMAIN_MM}mm, dx={DX_MM:.3f}mm, f0={F0_HZ/1e6}MHz")
    cropped, native_spacing = load_and_crop_patient023()
    print(f"  cropped native shape: {cropped.shape}, native spacing: {native_spacing}")
    print(f"  cropped physical extent: {np.array(cropped.shape) * native_spacing} mm")

    resampled = resample_to_target(cropped, native_spacing)
    print(f"  resampled shape: {resampled.shape} ({resampled.size:,} voxels)")
    for label, name in [(0, "water"), (1, "RV/blood"), (2, "MYO"), (3, "LV/blood")]:
        frac = (resampled == label).mean() * 100
        print(f"    {name}: {frac:.1f}% of volume")

    sound_speed, density = build_sound_speed_density_maps(resampled)
    print(f"  sound_speed range: [{sound_speed.min():.1f}, {sound_speed.max():.1f}] m/s")
    print(f"  density range: [{density.min():.1f}, {density.max():.1f}] kg/m3")

    np.savez("results/patient023_3d_medium_N150.npz",
             sound_speed=sound_speed, density=density, resampled_mask=resampled,
             dx_m=DX_M, n_grid=N_GRID, f0_hz=F0_HZ)
    print("\nSaved results/patient023_3d_medium_N150.npz (for Colab GPU memory-verification test)")
