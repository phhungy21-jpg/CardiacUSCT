"""Evaluates the single-ring 36-element test scan (run -17) against
ground truth, using the SAME 2D DAS + angular boundary-extraction
method as `phase1_das_perring_2d_reconstruction.py` (run -16) --
directly comparable to that run's 12-element ring-3 result (z=68,
RMSE=15.26mm) and to `jwave_ring_tomography`'s validated 36-element/
ring 2D result (RMSE=1.04mm).
"""

import numpy as np
from scipy.signal import correlate, hilbert, find_peaks
from scipy.interpolate import RegularGridInterpolator

SCAN_NPZ = r"D:\Cardiac USCT\jwave_cylindrical_tomography\single_ring_36_multistatic_scan_patient023.npz"
EVAL_NPZ = r"D:\Cardiac USCT\jwave_cylindrical_tomography\results\das_3d_evaluation_patient023.npz"

DX_MM = 130.0 / 150.0
WATER_SOUND_SPEED = 1520.0
MIN_ELEMENT_SEP_CELLS = 15.0
N_ANGLES = 360
R_MIN_MM, R_MAX_MM = 10.0, 50.0
N_RADIAL_SAMPLES = 400


def build_template(f0_hz, dt, n_cycles=3):
    duration_s = n_cycles / f0_hz * 3
    n = int(duration_s / dt)
    t = np.arange(n) * dt
    center_t = n_cycles / f0_hz
    signal = np.sin(2 * np.pi * f0_hz * t) * np.exp(-((t - center_t) ** 2) / (2 * (n_cycles / 3 / f0_hz) ** 2))
    return signal.astype(np.float32)


def matched_filter_envelope(trace, template):
    correlated = correlate(trace, template, mode="full")
    return np.abs(hilbert(correlated))


if __name__ == "__main__":
    d = np.load(SCAN_NPZ)
    water_traces, phantom_traces = d["water_traces"], d["phantom_traces"]
    positions = d["positions"]
    dt, n_steps, n_grid = float(d["dt"]), int(d["n_steps"]), int(d["n_grid"])
    f0_hz = float(d["f0_hz"])
    z = int(d["ring_z_cell"])
    n_elements = positions.shape[0]

    d_eval = np.load(EVAL_NPZ)
    gt_coarse = d_eval["gt_coarse"]

    template = build_template(f0_hz, dt)
    lag_t = (np.arange(len(template) + n_steps - 1) - (len(template) - 1)) * dt

    excess_energy = {}
    pairs = []
    for tx in range(n_elements):
        for rx in range(n_elements):
            if tx == rx:
                continue
            sep = np.linalg.norm(positions[tx] - positions[rx])
            if sep < MIN_ELEMENT_SEP_CELLS:
                continue
            env_p = matched_filter_envelope(phantom_traces[tx, :, rx], template)
            env_w = matched_filter_envelope(water_traces[tx, :, rx], template)
            excess = np.clip(env_p - env_w, 0, None)
            excess_energy[(tx, rx)] = (excess ** 2).astype(np.float32)
            pairs.append((tx, rx))
    print(f"{len(pairs)} valid pairs (of {n_elements*(n_elements-1)} possible)")

    img_coords = np.arange(n_grid, dtype=np.float64)
    RRi, CCi = np.meshgrid(img_coords, img_coords, indexing="ij")
    accumulator_2d = np.zeros((n_grid, n_grid), dtype=np.float32)
    for (tx, rx) in pairs:
        tx_pos, rx_pos = positions[tx], positions[rx]
        dist_tx = np.sqrt((RRi - tx_pos[0])**2 + (CCi - tx_pos[1])**2 + (z - tx_pos[2])**2) * (DX_MM / 1000.0)
        dist_rx = np.sqrt((RRi - rx_pos[0])**2 + (CCi - rx_pos[1])**2 + (z - rx_pos[2])**2) * (DX_MM / 1000.0)
        t_pred = (dist_tx + dist_rx) / WATER_SOUND_SPEED
        sampled = np.interp(t_pred.ravel(), lag_t, excess_energy[(tx, rx)], left=0.0, right=0.0)
        accumulator_2d += sampled.reshape(n_grid, n_grid)

    gt_slice = gt_coarse[:, :, z].astype(np.float64)
    is_tissue = gt_slice > 0
    print(f"tissue voxels at z={z}: {is_tissue.sum()}")

    rr_idx, cc_idx = np.meshgrid(img_coords, img_coords, indexing="ij")
    centroid_row = rr_idx[is_tissue].mean()
    centroid_col = cc_idx[is_tissue].mean()

    acc_interp = RegularGridInterpolator((img_coords, img_coords), accumulator_2d, bounds_error=False, fill_value=0.0)
    gt_interp = RegularGridInterpolator((img_coords, img_coords), gt_slice, bounds_error=False, fill_value=0.0)

    angles = np.linspace(0, 2 * np.pi, N_ANGLES, endpoint=False)
    r_samples_mm = np.linspace(R_MIN_MM, R_MAX_MM, N_RADIAL_SAMPLES)
    r_samples_u = r_samples_mm / DX_MM
    r_gt_search_mm = np.linspace(0.0, R_MAX_MM + 10.0, N_RADIAL_SAMPLES)
    r_gt_search_u = r_gt_search_mm / DX_MM

    r_pred_mm = np.full(N_ANGLES, np.nan)
    r_true_mm = np.full(N_ANGLES, np.nan)
    for k, a in enumerate(angles):
        drow, dcol = np.cos(a), np.sin(a)
        pts_pred = np.stack([centroid_row + r_samples_u * drow, centroid_col + r_samples_u * dcol], axis=1)
        acc_vals = acc_interp(pts_pred)
        peaks, props = find_peaks(acc_vals, prominence=acc_vals.max() * 0.05 if acc_vals.max() > 0 else 1)
        if len(peaks) > 0:
            best = peaks[np.argmax(props["prominences"])]
            r_pred_mm[k] = r_samples_mm[best]

        pts_gt = np.stack([centroid_row + r_gt_search_u * drow, centroid_col + r_gt_search_u * dcol], axis=1)
        gt_vals = gt_interp(pts_gt)
        above = np.where(gt_vals >= 0.5)[0]
        if len(above) > 0:
            r_true_mm[k] = r_gt_search_mm[above[-1]]

    valid = ~np.isnan(r_true_mm) & ~np.isnan(r_pred_mm)
    no_peak_frac = np.isnan(r_pred_mm).mean()
    errors = r_pred_mm[valid] - r_true_mm[valid]
    rmse = np.sqrt(np.mean(errors ** 2))
    print(f"\n=== Single-ring 36-element result (z={z}) ===")
    print(f"  RMSE={rmse:.3f}mm, no-peak={no_peak_frac:.1%} ({valid.sum()}/{N_ANGLES} valid)")
    print(f"\n=== For comparison ===")
    print(f"  jwave_ring_tomography (2D, 36 elements/ring, real anatomy):    RMSE=1.04mm")
    print(f"  this project, 12 elements/ring, SAME z=68 slice (run -16):    RMSE=15.26mm")
    print(f"  this project, 36 elements/ring, SAME z=68 slice (this run):   RMSE={rmse:.2f}mm")
    print(f"  full 3D multistatic (run -14, all 96 elements, all pairs):   RMSE=10.27mm")

    np.savez_compressed("results/single_ring_36_evaluation_patient023.npz",
                         accumulator_2d=accumulator_2d, gt_slice=gt_slice,
                         r_pred_mm=r_pred_mm, r_true_mm=r_true_mm, angles=angles,
                         centroid_row=centroid_row, centroid_col=centroid_col, rmse_mm=rmse)
    print("\nSaved results/single_ring_36_evaluation_patient023.npz")
