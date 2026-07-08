"""Phase 3 — GENUINE BLIND shape reconstruction: per-angle radius
discovery, no shape family assumed.

Per user's correction: every prior "reconstruction" in this thread
(circle/triangle/heart-cartoon/ring/real-MRI-shape/RV) swept a SINGLE
scalar (radius or scale) against an ALREADY-KNOWN shape family or
literal true contour -- the backprojection IMAGE was genuinely blind
(no shape assumed at the pixel level), but the READOUT never was. This
is the first attempt at actually using that blind image to discover an
UNKNOWN shape: for each angle independently, sweep a candidate radius
and find that angle's own local-max radius via the SAME validated,
safety-checked selector (`select_best_local_peak`, runs -57/-65) and
curvature weighting (`pair_weight_at_R`, run -44) already used
everywhere else -- just applied PER ANGLE instead of integrated across
all angles for one global shape hypothesis. The only assumptions kept:
a known center (already used throughout this thread, e.g. run -46's
"each boundary uses its own known center" convention) and a star-convex
topology (one boundary crossing per ray) -- NOT a specific shape.

VALIDATION ORDER (per this project's standing discipline: cheapest,
best-understood case first): test on the SYNTHETIC RING phantom's
inner boundary first (true shape = exact circle, radius known exactly,
standard validated 4-probe geometry) -- if blind per-angle discovery
correctly recovers a flat, constant r(theta) profile close to the true
radius, the method works on the simplest possible case. Only then
escalate to the real, irregular MRI shape where the true contour is
NOT given to the algorithm at all.
"""

import numpy as np
from scipy.interpolate import RegularGridInterpolator

from phase3_backprojection_shape_fit_triangle import (
    capture_all_pairs, build_medium_homogeneous, direction_vector,
    _SRC, _RCV, t_arr, c_ref, _ENVELOPE_GROUP_DELAY_S,
    img_rows, img_cols, center, dx, labels,
)
from phase3_ring_phantom_shapefit import build_medium_ring
from phase3_ring_curvature_weighted_fit import pair_weight_at_R, select_best_local_peak

from matplotlib import pyplot as plt
import os

N_ANGLES = 144
_THETAS = np.linspace(0, 360, N_ANGLES, endpoint=False)


def blind_radial_scan(pairs, R_grid, origin=center):
    """For EACH angle independently: sweep R_grid, score each candidate
    point by the SAME curvature-weighted sum used in every validated
    fit so far, then apply the local-max-only selector to that angle's
    OWN score curve -- no shape assumed, no information shared across
    angles. Returns per-angle (best_R, is_peak, confidence, prominence)
    plus the full score curves for diagnostic plotting."""
    RR, CC = np.meshgrid(img_rows, img_cols, indexing="ij")
    per_pair_grids = {}
    for (tx, rx), envelope in pairs.items():
        src, rcv = _SRC[tx], _RCV[rx]
        dist_tx = np.sqrt((CC - src[0]) ** 2 + (RR - src[1]) ** 2) * dx[0]
        dist_rx = np.sqrt((CC - rcv[0]) ** 2 + (RR - rcv[1]) ** 2) * dx[0]
        t_total = (dist_tx + dist_rx) / c_ref + _ENVELOPE_GROUP_DELAY_S
        per_pair_grids[(tx, rx)] = np.interp(t_total, t_arr, envelope, left=0, right=0)
    interpolators = {
        key: RegularGridInterpolator((img_rows, img_cols), np.abs(grid), bounds_error=False, fill_value=0.0)
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
        # prominence, computed the same way as the shared module's version
        prominence = (scores.max() - scores.min())
        prominence = 0.0 if prominence < 1e-12 else (scores[np.argmin(np.abs(R_grid - best_R))] - scores.min()) / prominence
        results.append((best_R, is_peak, confidence, prominence))
    return results


R_GRID = np.arange(25.0, 100.0, 1.0)

if __name__ == "__main__":
    print(f"*** {labels.PENDING_SIGNOFF_BANNER} ***")
    print(f"*** {labels.TOY_EXACT_GT_CAPTION} ***")
    print("GENUINE BLIND shape reconstruction test: per-angle radius discovery, "
          "NO shape family assumed (only known center + star-convex topology). "
          "Validating on the synthetic ring's inner boundary (true = exact circle) "
          "before escalating to a real, irregular, truly-unknown shape.")

    TRUE_R = 60.0  # ED frame, same convention as every ring test in this thread
    print(f"\n=== Simulating ring phantom (inner_R={TRUE_R}, standard 4-probe) ===")
    pairs_ring = capture_all_pairs(build_medium_ring(TRUE_R))
    print("=== Simulating homogeneous reference ===")
    pairs_ref = capture_all_pairs(build_medium_homogeneous())

    print("\n=== Blind per-angle radial scan (real phantom) ===")
    results_real = blind_radial_scan(pairs_ring, R_GRID)
    print("=== Blind per-angle radial scan (homogeneous control) ===")
    results_ref = blind_radial_scan(pairs_ref, R_GRID)

    fitted_r = np.array([r[0] for r in results_real])
    is_peak = np.array([r[1] for r in results_real])
    conf = np.array([r[2] for r in results_real])
    prom = np.array([r[3] for r in results_real])
    err_mm = np.abs(fitted_r - TRUE_R) * dx[0] * 1e3

    fitted_r_ref = np.array([r[0] for r in results_ref])
    prom_ref = np.array([r[3] for r in results_ref])

    print(f"\n--- Result: blind per-angle inner-boundary discovery ---")
    print(f"  true R={TRUE_R} cells (constant, exact circle)")
    print(f"  fitted R: mean={fitted_r.mean():.2f}, std={fitted_r.std():.2f}, "
          f"min={fitted_r.min():.1f}, max={fitted_r.max():.1f}")
    print(f"  RMSE={np.sqrt(np.mean(err_mm**2)):.4f}mm across {N_ANGLES} independently-discovered angles")
    print(f"  mean prominence (real)={prom.mean():.2f} vs (homogeneous control)={prom_ref.mean():.2f}")
    print(f"  genuine local max found at {is_peak.sum()}/{N_ANGLES} angles")

    # Per-angle table (sampled every 8th angle to keep output manageable)
    print(f"\n  per-angle sample (every 8th angle):")
    for i in range(0, N_ANGLES, 8):
        print(f"    theta={_THETAS[i]:.0f}deg: fitted_R={fitted_r[i]:.1f} err={err_mm[i]:.2f}mm "
              f"is_peak={is_peak[i]} conf={conf[i]:.2f} prom={prom[i]:.2f}")

    # --- Figure: discovered contour (blind) vs true circle, overlaid on accumulator ---
    RR_full, CC_full = np.meshgrid(img_rows, img_cols, indexing="ij")
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

    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    axes[0].imshow(np.abs(accumulator_clean), cmap="hot", origin="upper",
                   extent=[img_cols.min(), img_cols.max(), img_rows.max(), img_rows.min()])
    theta_plot = np.linspace(0, 2 * np.pi, 200)
    axes[0].plot(center[1] + TRUE_R * np.sin(theta_plot), center[0] - TRUE_R * np.cos(theta_plot),
                 "c--", linewidth=1.5, label="true circle")
    d_rows, d_cols = direction_vector(_THETAS)
    blind_pts_row = center[0] + fitted_r * d_rows
    blind_pts_col = center[1] + fitted_r * d_cols
    axes[0].plot(np.append(blind_pts_col, blind_pts_col[0]), np.append(blind_pts_row, blind_pts_row[0]),
                 "g-", linewidth=1.5, marker="o", markersize=2, label="BLIND discovered contour")
    axes[0].set_title(f"Blind per-angle reconstruction\nRMSE={np.sqrt(np.mean(err_mm**2)):.3f}mm")
    axes[0].legend(fontsize=8)
    axes[0].axis("off")

    axes[1].plot(np.arange(N_ANGLES) * (360 / N_ANGLES), fitted_r, "g-", label="blind fitted R (real)")
    axes[1].axhline(TRUE_R, color="k", linestyle="--", label=f"true R={TRUE_R}")
    axes[1].plot(np.arange(N_ANGLES) * (360 / N_ANGLES), fitted_r_ref, "orange", linestyle=":", label="homogeneous control")
    axes[1].set_xlabel("angle (deg)")
    axes[1].set_ylabel("discovered radius (cells)")
    axes[1].set_title("Per-angle discovered radius vs. angle")
    axes[1].legend(fontsize=8)

    fig.suptitle("GENUINE BLIND shape reconstruction test (synthetic ring, known circle)\n"
                 "no shape family assumed -- only known center + per-angle local-max search")
    labels.add_banner(fig)
    plt.tight_layout()
    os.makedirs("results/figures", exist_ok=True)
    out_fig = "results/figures/phase3_blind_shape_reconstruction_test_ring.png"
    plt.savefig(out_fig, dpi=140)
    print(f"\nSaved {out_fig}")
