"""Evaluate the first 3D DAS reconstruction (`phase1_das_3d_reconstruction.py`)
against patient023's real ground-truth anatomy: rebuilds the SAME crop+
resample pipeline used to build the simulated medium (`phase1_cylindrical_
multistatic_scan.py`'s `load_and_crop`/`resample_to_target`, at N_GRID=150),
downsamples it onto the reconstruction's own coarser 60^3 image grid, and
checks whether high-accumulator-value voxels line up with the true tissue
boundary better than chance -- the same kind of quantitative check this
repo's 2D work always used before trusting a DAS result.

No new simulation -- ground truth comes from the already-available ACDC
mask, reconstruction comes from the already-saved accumulator.
"""

import numpy as np
from scipy import ndimage

PATIENT023_NPZ = r"D:\Cardiac USCT\pilot\data\processed\ACDC\patient023.npz"
RECON_NPZ = r"D:\Cardiac USCT\jwave_cylindrical_tomography\results\das_3d_reconstruction_patient023.npz"

N_GRID = 150
DOMAIN_MM = 130.0
DX_MM = DOMAIN_MM / N_GRID
MARGIN_MM = 20.0


def load_and_crop(npz_path, margin_mm=MARGIN_MM):
    d = np.load(npz_path, allow_pickle=True)
    mask = d["ed_mask"]
    native_spacing_mm = np.array(d["spacing"])[::-1]
    tissue = mask > 0
    coords = np.argwhere(tissue)
    mins, maxs = coords.min(axis=0), coords.max(axis=0)
    margin_vox = np.ceil(margin_mm / native_spacing_mm).astype(int)
    mins = np.maximum(mins - margin_vox, 0)
    maxs = np.minimum(maxs + margin_vox, np.array(mask.shape) - 1)
    cropped = mask[mins[0]:maxs[0]+1, mins[1]:maxs[1]+1, mins[2]:maxs[2]+1]
    return cropped, native_spacing_mm


def resample_to_target(cropped_mask, native_spacing_mm, target_n=N_GRID):
    zoom_factors = native_spacing_mm / DX_MM
    resampled = ndimage.zoom(cropped_mask, zoom_factors, order=0, mode="nearest")
    out = np.zeros((target_n, target_n, target_n), dtype=resampled.dtype)
    src_shape, tgt_shape = np.array(resampled.shape), np.array(out.shape)
    start_src = np.maximum((src_shape - tgt_shape) // 2, 0)
    start_tgt = np.maximum((tgt_shape - src_shape) // 2, 0)
    copy_shape = np.minimum(src_shape - start_src, tgt_shape - start_tgt)
    out[start_tgt[0]:start_tgt[0]+copy_shape[0], start_tgt[1]:start_tgt[1]+copy_shape[1], start_tgt[2]:start_tgt[2]+copy_shape[2]] = \
        resampled[start_src[0]:start_src[0]+copy_shape[0], start_src[1]:start_src[1]+copy_shape[1], start_src[2]:start_src[2]+copy_shape[2]]
    return out


if __name__ == "__main__":
    recon = np.load(RECON_NPZ)
    accumulator = recon["accumulator"]
    img_coords = recon["img_coords"]
    img_size = accumulator.shape[0]
    print(f"Loaded reconstruction: accumulator shape {accumulator.shape}")

    cropped, native_spacing = load_and_crop(PATIENT023_NPZ)
    ground_truth_150 = resample_to_target(cropped, native_spacing)
    print(f"Rebuilt ground-truth N=150 volume: tissue label counts {np.unique(ground_truth_150, return_counts=True)}")

    # sample the fine (N=150) ground truth at the SAME coarse img_coords grid
    # points the reconstruction used, so the two arrays are directly comparable voxel-for-voxel
    idx = np.clip(np.round(img_coords).astype(int), 0, N_GRID - 1)
    RR, CC, ZZ = np.meshgrid(idx, idx, idx, indexing="ij")
    gt_coarse = ground_truth_150[RR, CC, ZZ]
    is_tissue = gt_coarse > 0

    # true boundary: tissue voxels adjacent to a non-tissue voxel (surface shell only)
    eroded = ndimage.binary_erosion(is_tissue)
    boundary = is_tissue & ~eroded
    print(f"Coarse grid ({img_size}^3): {is_tissue.sum()} tissue voxels, {boundary.sum()} boundary-shell voxels")

    # Check 1: does the accumulator's energy concentrate at/near the true boundary,
    # compared to a random / interior / exterior baseline?
    acc_flat = accumulator.ravel()
    boundary_flat = boundary.ravel()
    interior_flat = (is_tissue & eroded).ravel()
    exterior_flat = (~is_tissue).ravel()

    print("\n=== Mean accumulator value by region ===")
    print(f"  boundary shell voxels: {acc_flat[boundary_flat].mean():.5g}  (n={boundary_flat.sum()})")
    print(f"  tissue interior voxels: {acc_flat[interior_flat].mean():.5g}  (n={interior_flat.sum()})")
    print(f"  exterior (water) voxels: {acc_flat[exterior_flat].mean():.5g}  (n={exterior_flat.sum()})")

    # Check 2: of the TOP-K highest-accumulator voxels, what fraction land within
    # 1 coarse voxel of the true boundary shell (vs. what a random voxel selection would get)?
    k = boundary_flat.sum()  # compare top-K to the true boundary's own voxel count
    top_k_idx = np.argsort(acc_flat)[-k:]
    dist_to_boundary = ndimage.distance_transform_edt(~boundary)
    dist_flat = dist_to_boundary.ravel()
    mean_dist_topk = dist_flat[top_k_idx].mean()
    mean_dist_random = dist_flat.mean()
    print(f"\n=== Top-{k} accumulator voxels vs. distance-to-true-boundary (in coarse voxels) ===")
    print(f"  mean distance-to-boundary, top-K accumulator voxels: {mean_dist_topk:.3f}")
    print(f"  mean distance-to-boundary, ALL voxels (baseline):     {mean_dist_random:.3f}")
    print(f"  ratio (< 1.0 means top-K voxels are CLOSER to the true boundary than chance): "
          f"{mean_dist_topk / mean_dist_random:.3f}")

    np.savez_compressed("results/das_3d_evaluation_patient023.npz",
                         gt_coarse=gt_coarse, boundary=boundary, accumulator=accumulator)
    print("\nSaved results/das_3d_evaluation_patient023.npz")
