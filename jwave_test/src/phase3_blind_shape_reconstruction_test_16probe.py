"""Phase 3 — GENUINE BLIND shape reconstruction, 16-PROBE geometry
(uniform 22.5-degree circular spacing), extending runs -70/-71/-72.

Loads the self-consistent calibration measured in
`phase3_16probe_calibration.py` (not assumed/reused from the 8-probe
geometry, per run -60's finding that calibration doesn't transfer
across geometries). Tests both the synthetic ring (known circle,
direct comparison to runs -70/-71) and, if that is clean, the same
off-center concave heart phantom (run -72) to see whether doubling the
probe count again narrows the shape-complexity-driven ghost problem
found there.
"""

import sys
import numpy as np
from scipy.interpolate import RegularGridInterpolator

from phase3_mri_16probe_test import (
    _SRC, _RCV, capture_all_pairs, build_medium_homogeneous, build_medium_real_contour,
    build_search_grid, direction_vector, pair_weight_at_R, select_best_local_peak,
    set_calibration, N, center, dx, c_ref, t_arr, _ENVELOPE_GROUP_DELAY_S, labels,
)

from matplotlib import pyplot as plt
import os

N_ANGLES = 144
_THETAS = np.linspace(0, 360, N_ANGLES, endpoint=False)
R_GRID = np.arange(25.0, 100.0, 1.0)
BASELINE_INNER_R = 60.0
WALL_THICKNESS = 30.0

SHAPE = sys.argv[1] if len(sys.argv) > 1 else "circle"


def build_synthetic_ring_label_map(inner_r):
    yy, xx = np.mgrid[0:N[0], 0:N[1]]
    dist = np.sqrt((xx - center[1]) ** 2 + (yy - center[0]) ** 2)
    outer_r = inner_r + WALL_THICKNESS
    label_map = np.zeros(N, dtype=int)
    label_map[dist < outer_r] = 2
    label_map[dist < inner_r] = 3
    return label_map


def load_calibration():
    d = np.load("results/mri_16probe_calibration.npz")
    seps = d["separations"]
    cal = {}
    for sep in seps:
        cal[float(sep)] = d[f"ratio_{sep}"]
    set_calibration(cal)
    print(f"  loaded calibration for separations: {sorted(cal.keys())}")


def blind_radial_scan(pairs, R_grid, img_rows_g, img_cols_g, origin):
    RR, CC = np.meshgrid(img_rows_g, img_cols_g, indexing="ij")
    per_pair_grids = {}
    for (tx, rx), envelope in pairs.items():
        src, rcv = _SRC[tx], _RCV[rx]
        dist_tx = np.sqrt((CC - src[0]) ** 2 + (RR - src[1]) ** 2) * dx[0]
        dist_rx = np.sqrt((CC - rcv[0]) ** 2 + (RR - rcv[1]) ** 2) * dx[0]
        t_total = (dist_tx + dist_rx) / c_ref + _ENVELOPE_GROUP_DELAY_S
        per_pair_grids[(tx, rx)] = np.interp(t_total, t_arr, envelope, left=0, right=0)
    interpolators = {
        key: RegularGridInterpolator((img_rows_g, img_cols_g), np.abs(grid), bounds_error=False, fill_value=0.0)
        for key, grid in per_pair_grids.items()
    }
    results = []
    for th in _THETAS:
        d_row, d_col = direction_vector(th)
        pts = np.array([(origin[0] + R * d_row, origin[1] + R * d_col) for R in R_grid])
        scores = np.zeros(len(R_grid))
        for i, R in enumerate(R_grid):
            total = 0.0
            for (tx, rx), interp in interpolators.items():
                w = pair_weight_at_R(tx, rx, R)
                total += w * interp(pts[i:i+1])[0]
            scores[i] = total
        best_R, is_peak, confidence = select_best_local_peak(R_grid, scores)
        results.append((best_R, is_peak, confidence))
    return results


if __name__ == "__main__":
    print(f"*** {labels.PENDING_SIGNOFF_BANNER} ***")
    print(f"*** {labels.TOY_EXACT_GT_CAPTION} ***")
    print(f"GENUINE BLIND shape reconstruction, 16-PROBE geometry, shape={SHAPE}.")
    load_calibration()

    if SHAPE == "circle":
        origin = center
        pairs_medium = capture_all_pairs(build_medium_real_contour(build_synthetic_ring_label_map(BASELINE_INNER_R)))
        true_r_by_angle = np.full(N_ANGLES, BASELINE_INNER_R)
        img_rows_g = np.linspace(center[0] - 100, center[0] + 100, 100)
        img_cols_g = np.linspace(center[1] - 100, center[1] + 100, 100)
        title_shape = "synthetic ring (known circle)"
    else:
        from phase3_heart_shape_offcenter_test import heart_vertices, build_medium_heart, OFFSET, SHIFTED_CENTER
        from phase3_heart_shape_shapefit import ray_heart_distance
        HEART_R = 50.0
        origin = SHIFTED_CENTER
        pairs_medium = capture_all_pairs(build_medium_heart(HEART_R))
        true_r_by_angle = np.array([ray_heart_distance(th, HEART_R, origin=SHIFTED_CENTER) for th in _THETAS])
        needed_extent = true_r_by_angle.max() * 1.3 + 15.0
        img_rows_g = np.linspace(center[0] - needed_extent, center[0] + needed_extent, 160)
        img_cols_g = np.linspace(center[1] - needed_extent, center[1] + needed_extent, 160)
        title_shape = f"off-center concave heart (R={HEART_R}, offset={OFFSET})"

    print(f"\n=== Simulating {title_shape} (16 transmits) ===")
    print("=== Simulating homogeneous reference ===")
    pairs_ref = capture_all_pairs(build_medium_homogeneous())

    print("\n=== Blind per-angle radial scan (real phantom, 16 probes) ===")
    results_real = blind_radial_scan(pairs_medium, R_GRID, img_rows_g, img_cols_g, origin)
    print("=== Blind per-angle radial scan (homogeneous control) ===")
    results_ref = blind_radial_scan(pairs_ref, R_GRID, img_rows_g, img_cols_g, origin)

    fitted_r = np.array([r[0] for r in results_real])
    is_peak = np.array([r[1] for r in results_real])
    err_mm = np.abs(fitted_r - true_r_by_angle) * dx[0] * 1e3
    fitted_r_ref = np.array([r[0] for r in results_ref])

    print(f"\n--- Result: blind per-angle discovery, {title_shape}, 16 PROBES ---")
    print(f"  RMSE={np.sqrt(np.mean(err_mm**2)):.4f}mm across {N_ANGLES} independently-discovered angles")
    print(f"  genuine local max found at {is_peak.sum()}/{N_ANGLES} angles")
    print(f"  angles with error > 1.0mm: {int((err_mm > 1.0).sum())}/{N_ANGLES}")
    if SHAPE == "circle":
        print(f"  (compare: 4-probe RMSE=1.3816mm run -70; 8-probe RMSE=0.3249mm run -71)")
    else:
        print(f"  (compare: 8-probe RMSE=1.5440mm run -72)")

    RR_full, CC_full = np.meshgrid(img_rows_g, img_cols_g, indexing="ij")
    accumulator = np.zeros(RR_full.shape)
    for (tx, rx), envelope in pairs_medium.items():
        src, rcv = _SRC[tx], _RCV[rx]
        dist_tx = np.sqrt((CC_full - src[0]) ** 2 + (RR_full - src[1]) ** 2) * dx[0]
        dist_rx = np.sqrt((CC_full - rcv[0]) ** 2 + (RR_full - rcv[1]) ** 2) * dx[0]
        t_total = (dist_tx + dist_rx) / c_ref + _ENVELOPE_GROUP_DELAY_S
        accumulator += np.interp(t_total, t_arr, envelope, left=0, right=0)
    accumulator_ref = np.zeros(RR_full.shape)
    for (tx, rx), envelope in pairs_ref.items():
        src, rcv = _SRC[tx], _RCV[rx]
        dist_tx = np.sqrt((CC_full - src[0]) ** 2 + (RR_full - src[1]) ** 2) * dx[0]
        dist_rx = np.sqrt((CC_full - rcv[0]) ** 2 + (RR_full - rcv[1]) ** 2) * dx[0]
        t_total = (dist_tx + dist_rx) / c_ref + _ENVELOPE_GROUP_DELAY_S
        accumulator_ref += np.interp(t_total, t_arr, envelope, left=0, right=0)
    accumulator_clean = accumulator - accumulator_ref

    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    axes[0].imshow(np.abs(accumulator_clean), cmap="hot", origin="upper",
                   extent=[img_cols_g.min(), img_cols_g.max(), img_rows_g.max(), img_rows_g.min()])
    if SHAPE == "circle":
        theta_plot = np.linspace(0, 2 * np.pi, 200)
        axes[0].plot(center[1] + BASELINE_INNER_R * np.sin(theta_plot), center[0] - BASELINE_INNER_R * np.cos(theta_plot),
                     "c--", linewidth=1.5, label="true circle")
    else:
        verts = heart_vertices(HEART_R)
        h_row = [v[0] for v in verts] + [verts[0][0]]
        h_col = [v[1] for v in verts] + [verts[0][1]]
        axes[0].plot(h_col, h_row, "c--", linewidth=1.5, label="true heart boundary")
    d_rows, d_cols = direction_vector(_THETAS)
    blind_pts_row = origin[0] + fitted_r * d_rows
    blind_pts_col = origin[1] + fitted_r * d_cols
    axes[0].plot(np.append(blind_pts_col, blind_pts_col[0]), np.append(blind_pts_row, blind_pts_row[0]),
                 "g-", linewidth=1.5, marker="o", markersize=2, label="BLIND discovered contour (16 probes)")
    axes[0].set_title(f"Blind per-angle reconstruction, 16 probes\nRMSE={np.sqrt(np.mean(err_mm**2)):.3f}mm")
    axes[0].legend(fontsize=8)
    axes[0].axis("off")

    axes[1].plot(np.arange(N_ANGLES) * (360 / N_ANGLES), fitted_r, "g-", label="blind fitted R (16 probes)")
    axes[1].plot(np.arange(N_ANGLES) * (360 / N_ANGLES), true_r_by_angle, "k--", label="true distance")
    axes[1].plot(np.arange(N_ANGLES) * (360 / N_ANGLES), fitted_r_ref, "orange", linestyle=":", label="homogeneous control")
    for probe_angle in np.linspace(0, 360, 16, endpoint=False):
        axes[1].axvline(probe_angle, color="gray", linestyle=":", alpha=0.3)
    axes[1].set_xlabel("angle (deg)")
    axes[1].set_ylabel("discovered / true radius (cells)")
    axes[1].set_title("Per-angle discovered vs. true radius\n(gray = 16 probe directions)")
    axes[1].legend(fontsize=8)

    fig.suptitle(f"GENUINE BLIND shape reconstruction, 16-PROBE geometry, {title_shape}")
    labels.add_banner(fig)
    plt.tight_layout()
    os.makedirs("results/figures", exist_ok=True)
    out_fig = f"results/figures/phase3_blind_shape_reconstruction_test_16probe_{SHAPE}.png"
    plt.savefig(out_fig, dpi=140)
    print(f"\nSaved {out_fig}")
