"""Extracts an actual 3D boundary SURFACE from the DAS accumulator
(run -10/-11) and measures RMSE against ground truth, the 3D analog of
`jwave_ring_tomography`'s per-angle boundary-radius RMSE convention
(run -31: 0.15mm idealized-phantom / run -32: 1.04mm real-anatomy,
patient023, straight-ray DAS).

Method (mirrors the 2D convention, generalized to a full sphere of
directions instead of 36 planar angles):
  1. Ray-casting ORIGIN = the true tissue centroid (a known-geometry
     assumption, same role as 2D's known phantom/probe center -- NOT
     ground-truth-derived boundary information, so this doesn't leak
     the answer).
  2. ~1000 evenly-distributed 3D directions (Fibonacci sphere).
  3. Per direction: sample the accumulator via trilinear interpolation
     along the ray, restricted to [R_MIN, R_MAX] (R_MAX set safely
     BELOW the probe-ring-radius artifact identified in run -11, so
     that known artifact cannot be picked as a false boundary) -- take
     the argmax as the predicted boundary radius.
  4. Ground truth radius per direction: trilinear-interpolate the
     (float-cast) tissue mask along the same ray, find the outermost
     0.5-crossing (mask -> water transition) -- the true boundary
     distance along that direction.
  5. RMSE across all directions with a valid ground-truth crossing.

Pure post-processing on already-saved arrays -- no new simulation.
"""

import numpy as np
from scipy.interpolate import RegularGridInterpolator
from scipy.signal import find_peaks

EVAL_NPZ = r"D:\Cardiac USCT\jwave_cylindrical_tomography\results\das_3d_evaluation_patient023.npz"
RECON_NPZ = r"D:\Cardiac USCT\jwave_cylindrical_tomography\results\das_3d_reconstruction_patient023.npz"

DX_MM = 130.0 / 150.0          # fine-grid mm/unit (this project's design point)
PROBE_RADIUS_MM = 56.0
R_MIN_MM = 10.0                 # dead zone near center, avoids interior speckle false-peaks
R_MAX_MM = 50.0                 # safely below the 56mm probe-ring artifact (run -11)
N_DIRECTIONS = 1000
N_RADIAL_SAMPLES = 400


def fibonacci_sphere(n):
    """n evenly-distributed unit direction vectors."""
    i = np.arange(n)
    phi = np.arccos(1 - 2 * (i + 0.5) / n)
    golden = np.pi * (1 + 5 ** 0.5)
    theta = golden * i
    x = np.sin(phi) * np.cos(theta)
    y = np.sin(phi) * np.sin(theta)
    z = np.cos(phi)
    return np.stack([x, y, z], axis=1)


if __name__ == "__main__":
    d_eval = np.load(EVAL_NPZ)
    accumulator = d_eval["accumulator"]
    gt_coarse = d_eval["gt_coarse"].astype(np.float64)
    d_recon = np.load(RECON_NPZ)
    img_coords = d_recon["img_coords"]  # fine-grid-unit coordinates, shared axis for all 3 dims

    is_tissue = gt_coarse > 0
    RR, CC, ZZ = np.meshgrid(img_coords, img_coords, img_coords, indexing="ij")
    centroid_u = np.array([RR[is_tissue].mean(), CC[is_tissue].mean(), ZZ[is_tissue].mean()])
    print(f"Ray origin (true tissue centroid, fine-grid units): {centroid_u}")
    print(f"  = {centroid_u * DX_MM} mm from domain corner")

    acc_interp = RegularGridInterpolator((img_coords, img_coords, img_coords), accumulator,
                                          bounds_error=False, fill_value=0.0)
    gt_interp = RegularGridInterpolator((img_coords, img_coords, img_coords), gt_coarse,
                                         bounds_error=False, fill_value=0.0)

    directions = fibonacci_sphere(N_DIRECTIONS)
    r_samples_mm = np.linspace(R_MIN_MM, R_MAX_MM, N_RADIAL_SAMPLES)
    r_samples_u = r_samples_mm / DX_MM

    r_pred_mm = np.full(N_DIRECTIONS, np.nan)
    r_true_mm = np.full(N_DIRECTIONS, np.nan)

    # full-range ground-truth search (0 to a bit beyond R_MAX) to find the true outer crossing,
    # independent of the artifact-avoidance restriction applied to the RECONSTRUCTION's own search
    r_gt_search_mm = np.linspace(0.0, R_MAX_MM + 10.0, N_RADIAL_SAMPLES)
    r_gt_search_u = r_gt_search_mm / DX_MM

    n_no_local_peak = 0
    for i, d in enumerate(directions):
        pts_pred = centroid_u + np.outer(r_samples_u, d)
        acc_vals = acc_interp(pts_pred)
        # require a GENUINE local maximum (surrounded by lower values on both sides) --
        # otherwise argmax silently picks the search window's edge whenever the signal is
        # still monotonically rising toward it (e.g. trending toward the probe-ring artifact
        # just beyond R_MAX), which is not a real detected boundary at all.
        peaks, props = find_peaks(acc_vals, prominence=acc_vals.max() * 0.05)
        if len(peaks) > 0:
            best = peaks[np.argmax(props["prominences"])]
            r_pred_mm[i] = r_samples_mm[best]
        else:
            n_no_local_peak += 1
            r_pred_mm[i] = np.nan

        pts_gt = centroid_u + np.outer(r_gt_search_u, d)
        gt_vals = gt_interp(pts_gt)
        above = np.where(gt_vals >= 0.5)[0]
        if len(above) > 0:
            r_true_mm[i] = r_gt_search_mm[above[-1]]  # outermost crossing = true boundary

    print(f"\n{n_no_local_peak}/{N_DIRECTIONS} directions had NO genuine local peak within "
          f"[{R_MIN_MM},{R_MAX_MM}]mm (excluded, not clipped to the window edge)")
    valid = ~np.isnan(r_true_mm) & ~np.isnan(r_pred_mm)
    print(f"{valid.sum()}/{N_DIRECTIONS} directions had BOTH a valid ground-truth crossing "
          f"and a genuine predicted peak")

    errors_mm = r_pred_mm[valid] - r_true_mm[valid]
    rmse_mm = np.sqrt(np.mean(errors_mm ** 2))
    bias_mm = errors_mm.mean()
    print(f"\n=== 3D boundary RMSE (patient023, real anatomy) ===")
    print(f"  RMSE = {rmse_mm:.3f} mm")
    print(f"  mean bias = {bias_mm:.3f} mm")
    print(f"  mean |error| = {np.abs(errors_mm).mean():.3f} mm")
    print(f"  median |error| = {np.median(np.abs(errors_mm)):.3f} mm")

    print(f"\n=== For comparison ===")
    print(f"  jwave_ring_tomography run -31 (2D, IDEALIZED synthetic phantom, bent-ray DAS): RMSE=0.147mm")
    print(f"  jwave_ring_tomography run -32 (2D, REAL patient023 anatomy, straight-ray DAS): RMSE=1.04mm")
    print(f"  this run (3D, REAL patient023 anatomy, straight-ray DAS):                     RMSE={rmse_mm:.2f}mm")

    np.savez_compressed("results/das_3d_boundary_extraction_patient023.npz",
                         directions=directions, r_pred_mm=r_pred_mm, r_true_mm=r_true_mm,
                         centroid_u=centroid_u, rmse_mm=rmse_mm)
    print("\nSaved results/das_3d_boundary_extraction_patient023.npz")
