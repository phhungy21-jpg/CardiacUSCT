"""Phase 3 — GENUINE BLIND shape reconstruction, 8-PROBE geometry, on
an OFF-CENTER, CONCAVE heart-shaped phantom (not just a centered
circle).

Per user: "try 8 probe with an off center heart shape." Escalates
run -71's clean 8-probe blind-circle result to a genuinely harder,
still-synthetic (known ground truth) case: the same 10-vertex
off-center concave heart phantom used in run -37/-38 (which broke
every fixed pair-subset rule and originally motivated the global
shape-fit approach). Reuses that phantom's exact geometry
(`heart_vertices`, `build_medium_heart`, `SHIFTED_CENTER`,
`OFFSET=(10,-15)`) and its own ray-distance ground truth
(`ray_heart_distance`, handles the concave notch's multi-edge
intersection correctly) -- but the BLIND method itself is told NONE of
this: it only knows `SHIFTED_CENTER` (the known origin) and performs
the exact same per-angle independent local-max radius search as runs
-70/-71, with no polygon/vertex information at all.

This tests two things simultaneously: (1) does the 8-probe ghost-cone
narrowing (run -71) generalize from a circle to a genuinely irregular,
off-center, CONCAVE shape; (2) does blind per-angle discovery survive
near the concave notch specifically, where run -36 confirmed real
diffraction occurs (a sharp curvature discontinuity, unlike anywhere
on a smooth circle).
"""

import numpy as np
from scipy.interpolate import RegularGridInterpolator

from phase3_mri_8probe_test import (
    _SRC, _RCV, capture_all_pairs, build_medium_homogeneous,
    direction_vector, pair_weight_at_R, select_best_local_peak,
    center, dx, c_ref, t_arr, _ENVELOPE_GROUP_DELAY_S, labels,
)
from phase3_heart_shape_offcenter_test import (
    heart_vertices, build_medium_heart, OFFSET, SHIFTED_CENTER,
)
from phase3_heart_shape_shapefit import ray_heart_distance

from matplotlib import pyplot as plt
import os

N_ANGLES = 144
_THETAS = np.linspace(0, 360, N_ANGLES, endpoint=False)
R_GRID = np.arange(20.0, 90.0, 1.0)
HEART_R = 50.0  # representative single-frame size, consistent with this thread's ED/ES range (40-60)


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
        best_R, best_score, is_peak, confidence, prominence = select_best_local_peak(R_grid, scores)
        results.append((best_R, is_peak, confidence, prominence))
    return results


if __name__ == "__main__":
    print(f"*** {labels.PENDING_SIGNOFF_BANNER} ***")
    print(f"*** {labels.TOY_EXACT_GT_CAPTION} ***")
    print(f"GENUINE BLIND shape reconstruction, 8-PROBE geometry, OFF-CENTER "
          f"CONCAVE heart phantom (R={HEART_R}, offset={OFFSET}). The blind method "
          f"is given ONLY the known center (SHIFTED_CENTER) -- no polygon/vertex "
          f"information at all. Ground truth (`ray_heart_distance`) used ONLY for "
          f"scoring afterward, not fed to the search.")

    true_r_by_angle = np.array([ray_heart_distance(th, HEART_R, origin=SHIFTED_CENTER) for th in _THETAS])
    print(f"  true distance range across angles: {true_r_by_angle.min():.1f} - {true_r_by_angle.max():.1f} cells "
          f"(a circle would have ZERO spread; this shape's spread reflects genuine irregularity)")

    needed_extent = true_r_by_angle.max() * 1.3 + 15.0
    img_rows_g = np.linspace(center[0] - needed_extent, center[0] + needed_extent, 160)
    img_cols_g = np.linspace(center[1] - needed_extent, center[1] + needed_extent, 160)

    print(f"\n=== Simulating off-center concave heart phantom (R={HEART_R}, 8 probes) ===")
    pairs_heart = capture_all_pairs(build_medium_heart(HEART_R))
    print("=== Simulating homogeneous reference ===")
    pairs_ref = capture_all_pairs(build_medium_homogeneous())

    print("\n=== Blind per-angle radial scan (real phantom, 8 probes) ===")
    results_real = blind_radial_scan(pairs_heart, R_GRID, img_rows_g, img_cols_g, SHIFTED_CENTER)
    print("=== Blind per-angle radial scan (homogeneous control) ===")
    results_ref = blind_radial_scan(pairs_ref, R_GRID, img_rows_g, img_cols_g, SHIFTED_CENTER)

    fitted_r = np.array([r[0] for r in results_real])
    is_peak = np.array([r[1] for r in results_real])
    conf = np.array([r[2] for r in results_real])
    prom = np.array([r[3] for r in results_real])
    err_mm = np.abs(fitted_r - true_r_by_angle) * dx[0] * 1e3
    fitted_r_ref = np.array([r[0] for r in results_ref])

    print(f"\n--- Result: blind per-angle discovery, off-center concave heart, 8 probes ---")
    print(f"  RMSE={np.sqrt(np.mean(err_mm**2)):.4f}mm across {N_ANGLES} independently-discovered angles")
    print(f"  genuine local max found at {is_peak.sum()}/{N_ANGLES} angles")
    print(f"  angles with error > 1.0mm: {int((err_mm > 1.0).sum())}/{N_ANGLES}")
    # Angle 0 is the bottom tip (a sharp CONVEX vertex); the notch sits opposite --
    # per HEART_UNIT_VERTICES, notch is at unit coord (0,0.40), i.e. directly
    # "up" from SHIFTED_CENTER in this thread's convention: theta=180 (down=tip,
    # 0=up -- matches direction_vector(0)=(-1,0)=up). Report the notch region specifically.
    notch_idx = np.argmin(np.abs(_THETAS - 0))
    print(f"  notch region (theta~0deg) error: {err_mm[max(0,notch_idx-3):notch_idx+4]}")
    tip_idx = np.argmin(np.abs(_THETAS - 180))
    print(f"  tip region (theta~180deg) error: {err_mm[max(0,tip_idx-3):tip_idx+4]}")

    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    RR_full, CC_full = np.meshgrid(img_rows_g, img_cols_g, indexing="ij")
    accumulator = np.zeros(RR_full.shape)
    for (tx, rx), envelope in pairs_heart.items():
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

    axes[0].imshow(np.abs(accumulator_clean), cmap="hot", origin="upper",
                   extent=[img_cols_g.min(), img_cols_g.max(), img_rows_g.max(), img_rows_g.min()])
    verts = heart_vertices(HEART_R)
    h_row = [v[0] for v in verts] + [verts[0][0]]
    h_col = [v[1] for v in verts] + [verts[0][1]]
    axes[0].plot(h_col, h_row, "c--", linewidth=1.5, label="true heart boundary")
    d_rows, d_cols = direction_vector(_THETAS)
    blind_pts_row = SHIFTED_CENTER[0] + fitted_r * d_rows
    blind_pts_col = SHIFTED_CENTER[1] + fitted_r * d_cols
    axes[0].plot(np.append(blind_pts_col, blind_pts_col[0]), np.append(blind_pts_row, blind_pts_row[0]),
                 "g-", linewidth=1.5, marker="o", markersize=2, label="BLIND discovered contour (8 probes)")
    axes[0].set_title(f"Blind per-angle reconstruction, off-center concave heart\nRMSE={np.sqrt(np.mean(err_mm**2)):.3f}mm")
    axes[0].legend(fontsize=8)
    axes[0].axis("off")

    axes[1].plot(np.arange(N_ANGLES) * (360 / N_ANGLES), fitted_r, "g-", label="blind fitted R (8 probes)")
    axes[1].plot(np.arange(N_ANGLES) * (360 / N_ANGLES), true_r_by_angle, "k--", label="true distance")
    axes[1].plot(np.arange(N_ANGLES) * (360 / N_ANGLES), fitted_r_ref, "orange", linestyle=":", label="homogeneous control")
    for probe_angle in [0, 45, 90, 135, 180, 225, 270, 315]:
        axes[1].axvline(probe_angle, color="gray", linestyle=":", alpha=0.4)
    axes[1].set_xlabel("angle (deg)")
    axes[1].set_ylabel("discovered / true radius (cells)")
    axes[1].set_title("Per-angle discovered vs. TRUE radius\n(gray = probe directions; theta=0 is the notch side)")
    axes[1].legend(fontsize=8)

    fig.suptitle("GENUINE BLIND shape reconstruction, 8-PROBE, off-center CONCAVE heart phantom\n"
                 "no shape family assumed -- only known center (SHIFTED_CENTER)")
    labels.add_banner(fig)
    plt.tight_layout()
    os.makedirs("results/figures", exist_ok=True)
    out_fig = "results/figures/phase3_blind_shape_reconstruction_test_8probe_heart.png"
    plt.savefig(out_fig, dpi=140)
    print(f"\nSaved {out_fig}")
