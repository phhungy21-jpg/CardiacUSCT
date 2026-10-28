"""Tests the user's hypothesis directly: restrict to SAME-RING (same-Z)
transmit/receive pairs only, and reconstruct each ring's coronal plane
as an independent 2D problem -- exactly `jwave_ring_tomography`'s
proven method (RMSE=1.04mm on this same patient) -- rather than the
full 3D multistatic combination (which mixes in cross-ring pairs whose
ellipsoids are NOT confined to a shared plane, and only pin down a
point in 3D when enough independent, non-coplanar pairs contribute --
run -14/-15's discussion).

Rationale: two ellipses confined to a shared 2D plane (same-ring
elements, same Z) generically intersect at a small number of DISCRETE
points (Bezout), unlike two 3D ellipsoids (which meet along a curve).
Restricting to same-ring pairs recovers that favorable 2D geometry at
each of the 8 ring heights, at the cost of only 12 elements/ring
(vs. 2D project's 36) -- so this ALSO isolates whether the 3D result's
poor accuracy is caused by (a) genuinely mixing in poorly-conditioned
cross-plane pairs, or (b) simply having fewer elements per ring than
the validated 2D setup.

Pure post-processing on the already-completed raw scan -- no new
simulation.
"""

import numpy as np
from scipy.signal import correlate, hilbert, find_peaks
from scipy.interpolate import RegularGridInterpolator

SCAN_NPZ = r"D:\Cardiac USCT\jwave_cylindrical_tomography\cylindrical_scan_output-20260710T100320Z-2-001\cylindrical_scan_output\cylindrical_multistatic_scan_patient023.npz"
EVAL_NPZ = r"D:\Cardiac USCT\jwave_cylindrical_tomography\results\das_3d_evaluation_patient023.npz"

DX_MM = 130.0 / 150.0
WATER_SOUND_SPEED = 1520.0
MIN_ELEMENT_SEP_CELLS = 15.0
IMG_SIZE_2D = 150   # match the fine simulation grid, same as run -13's full-resolution test
N_ANGLES = 360      # dense angular sampling for boundary extraction, cheap in 2D
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
    dx_m, f0_hz = float(d["dx_m"]), float(d["f0_hz"])

    d_eval = np.load(EVAL_NPZ)
    gt_coarse = d_eval["gt_coarse"]  # full N=150, real ground truth tissue mask

    template = build_template(f0_hz, dt)
    lag_t = (np.arange(len(template) + n_steps - 1) - (len(template) - 1)) * dt

    zs = positions[:, 2]
    unique_zs = sorted(set(zs.tolist()))
    print(f"{len(unique_zs)} rings, 12 elements each")

    per_ring_rmse = []
    per_ring_no_peak_frac = []

    for ring_i, z in enumerate(unique_zs):
        ring_idx = np.where(zs == z)[0]
        ring_pos = positions[ring_idx]

        # matched-filter, background-subtracted, squared envelope for same-ring pairs only
        excess_energy = {}
        pairs = []
        for a_local, tx in enumerate(ring_idx):
            for b_local, rx in enumerate(ring_idx):
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

        # 2D accumulator over the (row, col) plane at this ring's fixed z
        img_coords = np.arange(n_grid, dtype=np.float64)
        RRi, CCi = np.meshgrid(img_coords, img_coords, indexing="ij")
        accumulator_2d = np.zeros((n_grid, n_grid), dtype=np.float32)
        for (tx, rx) in pairs:
            tx_pos, rx_pos = positions[tx], positions[rx]
            dist_tx = np.sqrt((RRi - tx_pos[0])**2 + (CCi - tx_pos[1])**2 + (z - tx_pos[2])**2) * dx_m
            dist_rx = np.sqrt((RRi - rx_pos[0])**2 + (CCi - rx_pos[1])**2 + (z - rx_pos[2])**2) * dx_m
            t_pred = (dist_tx + dist_rx) / WATER_SOUND_SPEED
            sampled = np.interp(t_pred.ravel(), lag_t, excess_energy[(tx, rx)], left=0.0, right=0.0)
            accumulator_2d += sampled.reshape(n_grid, n_grid)

        # ground truth slice at this ring's z (nearest integer index into the N=150 volume)
        z_idx = int(round(z))
        gt_slice = gt_coarse[:, :, z_idx].astype(np.float64)
        is_tissue = gt_slice > 0
        if is_tissue.sum() < 10:
            print(f"  ring {ring_i} (z={z}): negligible tissue in this slice, skipping")
            continue

        rr_idx, cc_idx = np.meshgrid(img_coords, img_coords, indexing="ij")
        centroid_row = rr_idx[is_tissue].mean()
        centroid_col = cc_idx[is_tissue].mean()

        acc_interp = RegularGridInterpolator((img_coords, img_coords), accumulator_2d,
                                              bounds_error=False, fill_value=0.0)
        gt_interp = RegularGridInterpolator((img_coords, img_coords), gt_slice,
                                             bounds_error=False, fill_value=0.0)

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
        if valid.sum() < 5:
            print(f"  ring {ring_i} (z={z}): too few valid angles ({valid.sum()}), skipping RMSE")
            continue
        errors = r_pred_mm[valid] - r_true_mm[valid]
        rmse = np.sqrt(np.mean(errors ** 2))
        per_ring_rmse.append(rmse)
        per_ring_no_peak_frac.append(no_peak_frac)
        print(f"  ring {ring_i} (z={z}): {len(pairs)} pairs, RMSE={rmse:.3f}mm, "
              f"no-peak={no_peak_frac:.1%} ({valid.sum()}/{N_ANGLES} valid)")

    print(f"\n=== Per-ring 2D reconstruction (same-Z pairs only) ===")
    print(f"  mean RMSE across rings: {np.mean(per_ring_rmse):.3f}mm")
    print(f"  mean no-peak fraction: {np.mean(per_ring_no_peak_frac):.1%}")
    print(f"\n=== For comparison ===")
    print(f"  jwave_ring_tomography (2D, 36 elements/ring, real anatomy): RMSE=1.04mm")
    print(f"  this run (per-ring 2D, 12 elements/ring, same patient):    RMSE={np.mean(per_ring_rmse):.2f}mm")
    print(f"  full 3D multistatic (run -14, all 96 elements, all pairs):  RMSE=10.27mm")
