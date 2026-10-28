"""Rebuilds the boundary extraction (run -14) on a regular latitude/
longitude direction grid instead of a Fibonacci sphere, SPECIFICALLY so
the result has natural quad-mesh connectivity -- enabling a real shaded
surface render instead of a disconnected point cloud (the previous
viewer's "hideous"/noisy look was largely a consequence of rendering
raw scattered points where a continuous surface was the honest
comparison).

Same underlying method as run -14 (ray-cast from the true tissue
centroid, `scipy.signal.find_peaks` for a genuine local maximum in the
accumulator, ground truth via the outermost 0.5-crossing of the
tissue mask) -- only the direction sampling changes, from unstructured
(Fibonacci) to structured (theta x phi grid), which is what makes mesh
faces possible. RMSE should land in the same ballpark as run -14, not
a new result in its own right.
"""

import numpy as np
from scipy.interpolate import RegularGridInterpolator
from scipy.signal import find_peaks

EVAL_NPZ = r"D:\Cardiac USCT\jwave_cylindrical_tomography\results\das_3d_evaluation_patient023.npz"
RECON_NPZ = r"D:\Cardiac USCT\jwave_cylindrical_tomography\results\das_3d_reconstruction_patient023.npz"

DX_MM = 130.0 / 150.0
R_MIN_MM = 10.0
R_MAX_MM = 50.0
N_RADIAL_SAMPLES = 400
N_THETA = 96   # azimuth, periodic (wraps around)
N_PHI = 48     # elevation, open interval (avoids exact pole singularities)


if __name__ == "__main__":
    d_eval = np.load(EVAL_NPZ)
    accumulator = d_eval["accumulator"]
    gt_coarse = d_eval["gt_coarse"].astype(np.float64)
    d_recon = np.load(RECON_NPZ)
    img_coords = d_recon["img_coords"]

    is_tissue = gt_coarse > 0
    RR, CC, ZZ = np.meshgrid(img_coords, img_coords, img_coords, indexing="ij")
    centroid_u = np.array([RR[is_tissue].mean(), CC[is_tissue].mean(), ZZ[is_tissue].mean()])
    centroid_mm = centroid_u * DX_MM

    acc_interp = RegularGridInterpolator((img_coords, img_coords, img_coords), accumulator,
                                          bounds_error=False, fill_value=0.0)
    gt_interp = RegularGridInterpolator((img_coords, img_coords, img_coords), gt_coarse,
                                         bounds_error=False, fill_value=0.0)

    thetas = np.linspace(0, 2 * np.pi, N_THETA, endpoint=False)
    phis = np.linspace(-np.pi / 2 * 0.94, np.pi / 2 * 0.94, N_PHI)  # avoid exact poles

    directions = np.zeros((N_PHI, N_THETA, 3))
    for j, phi in enumerate(phis):
        for i, theta in enumerate(thetas):
            directions[j, i] = [np.cos(phi) * np.cos(theta), np.cos(phi) * np.sin(theta), np.sin(phi)]

    r_samples_mm = np.linspace(R_MIN_MM, R_MAX_MM, N_RADIAL_SAMPLES)
    r_samples_u = r_samples_mm / DX_MM
    r_gt_search_mm = np.linspace(0.0, R_MAX_MM + 10.0, N_RADIAL_SAMPLES)
    r_gt_search_u = r_gt_search_mm / DX_MM

    r_pred_mm = np.full((N_PHI, N_THETA), np.nan)
    r_true_mm = np.full((N_PHI, N_THETA), np.nan)

    for j in range(N_PHI):
        for i in range(N_THETA):
            d = directions[j, i]
            pts_pred = centroid_u + np.outer(r_samples_u, d)
            acc_vals = acc_interp(pts_pred)
            peaks, props = find_peaks(acc_vals, prominence=acc_vals.max() * 0.05)
            if len(peaks) > 0:
                best = peaks[np.argmax(props["prominences"])]
                r_pred_mm[j, i] = r_samples_mm[best]

            pts_gt = centroid_u + np.outer(r_gt_search_u, d)
            gt_vals = gt_interp(pts_gt)
            above = np.where(gt_vals >= 0.5)[0]
            if len(above) > 0:
                r_true_mm[j, i] = r_gt_search_mm[above[-1]]
        print(f"  phi row {j+1}/{N_PHI} done")

    valid = ~np.isnan(r_true_mm) & ~np.isnan(r_pred_mm)
    errors_mm = (r_pred_mm - r_true_mm)[valid]
    rmse_mm = np.sqrt(np.mean(errors_mm ** 2))
    print(f"\nvalid={valid.sum()}/{N_PHI*N_THETA}  RMSE={rmse_mm:.3f}mm  "
          f"no-peak={np.isnan(r_pred_mm).sum()}/{N_PHI*N_THETA}")

    np.savez_compressed("results/das_3d_mesh_patient023.npz",
                         directions=directions, r_pred_mm=r_pred_mm, r_true_mm=r_true_mm,
                         centroid_mm=centroid_mm, rmse_mm=rmse_mm, thetas=thetas, phis=phis)
    print("Saved results/das_3d_mesh_patient023.npz")
