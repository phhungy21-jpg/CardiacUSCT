"""Phase 3 — GENUINE BLIND shape reconstruction, 8-PROBE geometry.

Direct follow-on to run -70 (standard 4-probe blind test): per-angle
independent radius discovery on the SAME synthetic ring phantom (known
exact circle, R=60 cells), using the 8-probe geometry + fully self-
consistent calibration (runs -56/-60/-61) instead of the standard
4-probe model. Tests the run -70 prediction directly: since the
4-probe blind failure was diagnosed as ghost-cone corruption specific
to angles FAR from any probe's own viewing axis, adding 4 more probes
(45-degree spacing instead of 90) should narrow those gaps
substantially -- a much stronger, more mechanistic expectation than
the modest improvement more probes gave the earlier scale-only
estimation tests (runs -56/-63).
"""

import numpy as np
from scipy.interpolate import RegularGridInterpolator

from phase3_mri_8probe_test import (
    _SRC, _RCV, capture_all_pairs, build_medium_homogeneous, build_medium_real_contour,
    direction_vector, pair_weight_at_R, select_best_local_peak,
    N, center, dx, c_ref, t_arr, _ENVELOPE_GROUP_DELAY_S, labels,
)

from matplotlib import pyplot as plt
import os

N_ANGLES = 144
_THETAS = np.linspace(0, 360, N_ANGLES, endpoint=False)
R_GRID = np.arange(25.0, 100.0, 1.0)

BASELINE_INNER_R = 60.0
WALL_THICKNESS = 30.0


def build_synthetic_ring_label_map(inner_r):
    yy, xx = np.mgrid[0:N[0], 0:N[1]]
    dist = np.sqrt((xx - center[1]) ** 2 + (yy - center[0]) ** 2)
    outer_r = inner_r + WALL_THICKNESS
    label_map = np.zeros(N, dtype=int)
    label_map[dist < outer_r] = 2
    label_map[dist < inner_r] = 3
    return label_map


def blind_radial_scan(pairs, R_grid, img_rows_g, img_cols_g, origin=center):
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
    print("GENUINE BLIND shape reconstruction, 8-PROBE geometry -- direct follow-on "
          "to run -70's 4-probe test. Same synthetic ring (known exact circle), "
          "same per-angle independent radius discovery, testing whether more probe "
          "angles narrow the ghost-cone corruption found between probes with only 4.")

    img_rows_g = np.linspace(center[0] - 100, center[0] + 100, 100)
    img_cols_g = np.linspace(center[1] - 100, center[1] + 100, 100)

    print(f"\n=== Simulating ring phantom (inner_R={BASELINE_INNER_R}, 8 probes) ===")
    pairs_ring = capture_all_pairs(build_medium_real_contour(build_synthetic_ring_label_map(BASELINE_INNER_R)))
    print("=== Simulating homogeneous reference ===")
    pairs_ref = capture_all_pairs(build_medium_homogeneous())

    print("\n=== Blind per-angle radial scan (real phantom, 8 probes) ===")
    results_real = blind_radial_scan(pairs_ring, R_GRID, img_rows_g, img_cols_g)
    print("=== Blind per-angle radial scan (homogeneous control) ===")
    results_ref = blind_radial_scan(pairs_ref, R_GRID, img_rows_g, img_cols_g)

    fitted_r = np.array([r[0] for r in results_real])
    is_peak = np.array([r[1] for r in results_real])
    conf = np.array([r[2] for r in results_real])
    prom = np.array([r[3] for r in results_real])
    err_mm = np.abs(fitted_r - BASELINE_INNER_R) * dx[0] * 1e3
    fitted_r_ref = np.array([r[0] for r in results_ref])

    print(f"\n--- Result: blind per-angle inner-boundary discovery (8 PROBES) ---")
    print(f"  true R={BASELINE_INNER_R} cells (constant, exact circle)")
    print(f"  fitted R: mean={fitted_r.mean():.2f}, std={fitted_r.std():.2f}, "
          f"min={fitted_r.min():.1f}, max={fitted_r.max():.1f}")
    print(f"  RMSE={np.sqrt(np.mean(err_mm**2)):.4f}mm across {N_ANGLES} independently-discovered angles")
    print(f"  (compare to run -70's 4-probe RMSE=1.3816mm)")
    print(f"  genuine local max found at {is_peak.sum()}/{N_ANGLES} angles")
    print(f"  angles with error > 1.0mm: {int((err_mm > 1.0).sum())}/{N_ANGLES} "
          f"(compare to 4-probe's angular ghost-cone extent)")

    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    RR_full, CC_full = np.meshgrid(img_rows_g, img_cols_g, indexing="ij")
    accumulator = np.zeros(RR_full.shape)
    for (tx, rx), envelope in pairs_ring.items():
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
    theta_plot = np.linspace(0, 2 * np.pi, 200)
    axes[0].plot(center[1] + BASELINE_INNER_R * np.sin(theta_plot), center[0] - BASELINE_INNER_R * np.cos(theta_plot),
                 "c--", linewidth=1.5, label="true circle")
    d_rows, d_cols = direction_vector(_THETAS)
    blind_pts_row = center[0] + fitted_r * d_rows
    blind_pts_col = center[1] + fitted_r * d_cols
    axes[0].plot(np.append(blind_pts_col, blind_pts_col[0]), np.append(blind_pts_row, blind_pts_row[0]),
                 "g-", linewidth=1.5, marker="o", markersize=2, label="BLIND discovered contour (8 probes)")
    axes[0].set_title(f"Blind per-angle reconstruction (8 probes)\nRMSE={np.sqrt(np.mean(err_mm**2)):.3f}mm")
    axes[0].legend(fontsize=8)
    axes[0].axis("off")

    axes[1].plot(np.arange(N_ANGLES) * (360 / N_ANGLES), fitted_r, "g-", label="blind fitted R (8 probes)")
    axes[1].axhline(BASELINE_INNER_R, color="k", linestyle="--", label=f"true R={BASELINE_INNER_R}")
    axes[1].plot(np.arange(N_ANGLES) * (360 / N_ANGLES), fitted_r_ref, "orange", linestyle=":", label="homogeneous control")
    for probe_angle in [0, 45, 90, 135, 180, 225, 270, 315]:
        axes[1].axvline(probe_angle, color="gray", linestyle=":", alpha=0.4)
    axes[1].set_xlabel("angle (deg)")
    axes[1].set_ylabel("discovered radius (cells)")
    axes[1].set_title("Per-angle discovered radius vs. angle\n(gray lines = probe viewing directions)")
    axes[1].legend(fontsize=8)

    fig.suptitle("GENUINE BLIND shape reconstruction, 8-PROBE geometry (synthetic ring, known circle)\n"
                 "direct follow-on to run -70's 4-probe result")
    labels.add_banner(fig)
    plt.tight_layout()
    os.makedirs("results/figures", exist_ok=True)
    out_fig = "results/figures/phase3_blind_shape_reconstruction_test_8probe_ring.png"
    plt.savefig(out_fig, dpi=140)
    print(f"\nSaved {out_fig}")
