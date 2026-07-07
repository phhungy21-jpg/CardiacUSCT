"""Phase 3 — the generalization test flagged as essential after run -45:
an OFF-CENTER, ECCENTRIC myocardial ring (inner and outer circle centers
offset from EACH OTHER, creating physiologically-realistic thick/thin
wall regions), full 8-frame cardiac cycle, using the validated
curvature-weighted + guard-band fit (run -45).

Per user: "now run with off center heart phantom, with the 2 ring's
epicenter off (to create thick/thin regions mimicking true LV wall). do
the 8 frame one."

Two distinct offsets, both fixed (not scaling with radius) across the
whole cycle:
  - OUTER_CENTER_OFFSET: the whole phantom is shifted off the domain/
    probe-array center (like run -35's off-center triangle test).
  - INNER_ECCENTRICITY_OFFSET: the inner (LV cavity) circle's center is
    further offset from the outer (epicardial) circle's own center --
    this is what creates the thick/thin wall pattern (a standard way to
    model basal/regional wall asymmetry). Eccentricity magnitude ~7.8
    cells (0.78mm) against a nominal 30-cell (3mm) wall thickness gives
    a ~1.7x thickness ratio between the thick and thin sides -- clearly
    asymmetric, clinically meaningful.

Both circles still follow the existing constant-wall-thickness motion
model (phase3_config.py: inner 60->40->60 cells, outer = inner + 30) --
only their relative POSITION is now eccentric, not their motion pattern.

Requires a WIDER accumulator/search grid than every previous script in
this thread (the offset phantom, at its largest radius, extends further
from the domain center than the default grid safely covers) -- this
script defines its own local grid and a grid-parametrized copy of run
-45's fit function, rather than reusing the imported fixed-grid version.

Known caveat carried forward explicitly: run -41/-45's guard band
compares candidate OUTER radius VALUES to the fitted INNER radius VALUE
as a proxy for spatial overlap. With inner and outer centered at
different points, this numeric comparison is a coarser, less exact
proxy for genuine spatial overlap than in the concentric case -- kept
as-is here (not re-derived) to test the validated method under
imperfect-but-plausible conditions rather than re-engineering it before
even seeing whether it still works.
"""

import numpy as np
from scipy.interpolate import RegularGridInterpolator
from scipy.signal import find_peaks

from jax import numpy as jnp
from jwave import FourierSeries
from jwave.geometry import Medium

from phase3_backprojection_shape_fit_triangle import (
    capture_all_pairs, build_medium_homogeneous, direction_vector,
    _SRC, _RCV, t_arr, c_ref, _ENVELOPE_GROUP_DELAY_S,
    center, dx, N, domain, cfg, p3cfg, labels,
)
from phase3_ring_curvature_weighted_fit import pair_weight_at_R

from matplotlib import pyplot as plt
import os

# --- Eccentric geometry (fixed offsets, do not scale with radius) ---
OUTER_CENTER_OFFSET = (8, 6)     # whole phantom shifted off domain/probe center
INNER_ECC_OFFSET = (6, 5)        # inner circle further offset from outer circle's own center
OUTER_CENTER = (center[0] + OUTER_CENTER_OFFSET[0], center[1] + OUTER_CENTER_OFFSET[1])
INNER_CENTER = (OUTER_CENTER[0] + INNER_ECC_OFFSET[0], OUTER_CENTER[1] + INNER_ECC_OFFSET[1])

WALL_THICKNESS = p3cfg.WALL_THICKNESS_CELLS
ECC_MAGNITUDE = np.hypot(*INNER_ECC_OFFSET)
print(f"Eccentricity magnitude: {ECC_MAGNITUDE:.2f} cells ({ECC_MAGNITUDE*dx[0]*1e3:.2f}mm) "
      f"vs wall thickness {WALL_THICKNESS} cells ({WALL_THICKNESS*dx[0]*1e3:.2f}mm) -- "
      f"thin side ~{(WALL_THICKNESS-ECC_MAGNITUDE)*dx[0]*1e3:.2f}mm, "
      f"thick side ~{(WALL_THICKNESS+ECC_MAGNITUDE)*dx[0]*1e3:.2f}mm")


def build_medium_eccentric_ring(inner_R):
    outer_R = inner_R + WALL_THICKNESS
    yy, xx = np.mgrid[0:N[0], 0:N[1]]
    dist_outer = np.sqrt((xx - OUTER_CENTER[1]) ** 2 + (yy - OUTER_CENTER[0]) ** 2)
    dist_inner = np.sqrt((xx - INNER_CENTER[1]) ** 2 + (yy - INNER_CENTER[0]) ** 2)

    label_map = np.zeros(N, dtype=int)  # default: chest-wall-proxy
    label_map[dist_outer < outer_R] = 2   # myocardium
    label_map[dist_inner < inner_R] = 3   # blood (LV cavity, eccentric)

    sound_speed_map = np.zeros(N, dtype=np.float32)
    density_map = np.zeros(N, dtype=np.float32)
    for label, tissue in cfg.ACDC_LABEL_TO_TISSUE.items():
        m = label_map == label
        sound_speed_map[m] = tissue.sound_speed
        density_map[m] = tissue.density

    ssm = jnp.expand_dims(jnp.array(sound_speed_map), -1)
    dm = jnp.expand_dims(jnp.array(density_map), -1)
    return Medium(domain=domain, sound_speed=FourierSeries(ssm, domain),
                  density=FourierSeries(dm, domain))


# --- Wider local accumulator grid (the default +/-90 grid used elsewhere
# in this thread is too small for this off-center, R-up-to-90 phantom) ---
SEARCH_RADIUS_LOCAL = 100
IMG_N_LOCAL = 140
_grid_lo = center[0] - SEARCH_RADIUS_LOCAL - 10
_grid_hi = center[0] + SEARCH_RADIUS_LOCAL + 10
img_rows_w = np.linspace(_grid_lo, _grid_hi, IMG_N_LOCAL)
img_cols_w = np.linspace(_grid_lo, _grid_hi, IMG_N_LOCAL)

N_ANGLES = 72
_THETAS = np.linspace(0, 360, N_ANGLES, endpoint=False)

INNER_R_GRID = np.arange(25.0, 75.0, 0.5)
OUTER_R_GRID = np.arange(55.0, 120.0, 0.5)
GUARD_BAND_CELLS = 8.0


def backproject_wide(pairs):
    """Plain NAIVE (unweighted) full-image backprojection on the wide
    grid, for VISUALIZATION only -- the curvature-weighted score used
    for fitting is a 1D function of candidate R, not directly a 2D
    image, so a plain image is what we show in the filmstrip."""
    RR, CC = np.meshgrid(img_rows_w, img_cols_w, indexing="ij")
    accumulator = np.zeros(RR.shape)
    for (tx, rx), envelope in pairs.items():
        src = _SRC[tx]
        rcv = _RCV[rx]
        dist_tx = np.sqrt((CC - src[0]) ** 2 + (RR - src[1]) ** 2) * dx[0]
        dist_rx = np.sqrt((CC - rcv[0]) ** 2 + (RR - rcv[1]) ** 2) * dx[0]
        t_total = (dist_tx + dist_rx) / c_ref + _ENVELOPE_GROUP_DELAY_S
        accumulator += np.interp(t_total, t_arr, envelope, left=0, right=0)
    return accumulator


def fit_circle_radius_curvature_weighted_wide(pairs, R_grid, origin):
    """Same principle as run -45's fit_circle_radius_curvature_weighted,
    parametrized on the WIDER local grid instead of the imported
    fixed-size one."""
    interpolators = {}
    for (tx, rx), envelope in pairs.items():
        src = _SRC[tx]
        rcv = _RCV[rx]
        RR, CC = np.meshgrid(img_rows_w, img_cols_w, indexing="ij")
        dist_tx = np.sqrt((CC - src[0]) ** 2 + (RR - src[1]) ** 2) * dx[0]
        dist_rx = np.sqrt((CC - rcv[0]) ** 2 + (RR - rcv[1]) ** 2) * dx[0]
        t_total = (dist_tx + dist_rx) / c_ref + _ENVELOPE_GROUP_DELAY_S
        grid = np.interp(t_total, t_arr, envelope, left=0, right=0)
        interpolators[(tx, rx)] = RegularGridInterpolator(
            (img_rows_w, img_cols_w), np.abs(grid), bounds_error=False, fill_value=0.0)

    scores = np.zeros(len(R_grid))
    for i, R in enumerate(R_grid):
        pts = np.array([(origin[0] + R * direction_vector(th)[0], origin[1] + R * direction_vector(th)[1])
                         for th in _THETAS])
        total = 0.0
        for (tx, rx), interp in interpolators.items():
            w = pair_weight_at_R(tx, rx, R)
            total += w * interp(pts).sum()
        scores[i] = total

    best_idx = np.argmax(scores)
    best_R = R_grid[best_idx]
    peak_idx, _ = find_peaks(scores)
    if len(peak_idx) >= 2:
        sorted_peaks = np.sort(scores[peak_idx])[::-1]
        confidence = sorted_peaks[0] / (sorted_peaks[1] + 1e-12)
    else:
        confidence = np.inf
    return best_R, scores, confidence


if __name__ == "__main__":
    print(f"*** {labels.PENDING_SIGNOFF_BANNER} ***")
    print(f"*** {labels.TOY_EXACT_GT_CAPTION} ***")
    print(f"Eccentric off-center ring phantom, 8-frame cycle, curvature-"
          f"weighted + guard-band fit (run -45's validated method). "
          f"OUTER_CENTER={OUTER_CENTER} (offset {OUTER_CENTER_OFFSET} from "
          f"domain center), INNER_CENTER={INNER_CENTER} (offset "
          f"{INNER_ECC_OFFSET} from outer center).")

    print("\n=== Capturing homogeneous reference (reused across all frames) ===")
    pairs_ref = capture_all_pairs(build_medium_homogeneous())
    accumulator_ref = backproject_wide(pairs_ref)

    N_FRAMES_MOVIE = 8
    phases = np.linspace(0, 1, N_FRAMES_MOVIE)
    inner_radii = [p3cfg.lv_radius_at_phase(p) for p in phases]

    all_fitted_inner = []
    all_fitted_outer = []
    frames = []
    for i, inner_R in enumerate(inner_radii):
        outer_R_true = inner_R + WALL_THICKNESS
        print(f"\n=== Frame {i+1}/{N_FRAMES_MOVIE} (inner_R={inner_R:.1f}, outer_R={outer_R_true:.1f}) ===")
        pairs_tri = capture_all_pairs(build_medium_eccentric_ring(inner_R))

        accumulator_clean = backproject_wide(pairs_tri) - accumulator_ref
        frames.append(accumulator_clean)

        fitted_inner, _, conf_inner = fit_circle_radius_curvature_weighted_wide(
            pairs_tri, INNER_R_GRID, origin=INNER_CENTER)

        outer_grid_guarded = OUTER_R_GRID[np.abs(OUTER_R_GRID - fitted_inner) > GUARD_BAND_CELLS]
        fitted_outer, _, conf_outer = fit_circle_radius_curvature_weighted_wide(
            pairs_tri, outer_grid_guarded, origin=OUTER_CENTER)

        all_fitted_inner.append(fitted_inner)
        all_fitted_outer.append(fitted_outer)

        inner_err_mm = abs(fitted_inner - inner_R) * dx[0] * 1e3
        outer_err_mm = abs(fitted_outer - outer_R_true) * dx[0] * 1e3
        locked = abs(fitted_outer - fitted_inner) < 3.0
        print(f"  inner: fitted={fitted_inner:.1f} true={inner_R:.1f} err={inner_err_mm:.2f}mm conf={conf_inner:.2f}")
        print(f"  outer: fitted={fitted_outer:.1f} true={outer_R_true:.1f} err={outer_err_mm:.2f}mm conf={conf_outer:.2f}")
        print(f"  outer locked to inner? {locked}")

    print("\n--- RMSE across all 8 frames ---")
    inner_errs = np.array([abs(all_fitted_inner[i] - inner_radii[i]) * dx[0] * 1e3 for i in range(N_FRAMES_MOVIE)])
    outer_errs = np.array([abs(all_fitted_outer[i] - (inner_radii[i] + WALL_THICKNESS)) * dx[0] * 1e3
                            for i in range(N_FRAMES_MOVIE)])
    print(f"  inner RMSE={np.sqrt(np.mean(inner_errs**2)):.4f}mm  (per-frame: {np.round(inner_errs,2).tolist()})")
    print(f"  outer RMSE={np.sqrt(np.mean(outer_errs**2)):.4f}mm  (per-frame: {np.round(outer_errs,2).tolist()})")

    # --- Figure: 8-frame filmstrip, true (cyan) + fitted (green) circles,
    # each using its OWN true center (inner vs outer), showing the
    # eccentric thick/thin wall directly. ---
    theta_plot = np.linspace(0, 2 * np.pi, 100)
    n_cols = 4
    n_rows = int(np.ceil(N_FRAMES_MOVIE / n_cols))
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(3.4 * n_cols, 3.6 * n_rows))
    axes = np.array(axes).reshape(-1)
    vmax = max(np.abs(f).max() for f in frames)
    for i, (ax, frame, inner_R) in enumerate(zip(axes, frames, inner_radii)):
        outer_R_true = inner_R + WALL_THICKNESS
        ax.imshow(np.abs(frame), cmap="hot", vmin=0, vmax=vmax, origin="upper",
                  extent=[img_cols_w.min(), img_cols_w.max(), img_rows_w.max(), img_rows_w.min()])
        # true boundaries (each around its OWN true center)
        ax.plot(INNER_CENTER[1] + inner_R * np.cos(theta_plot), INNER_CENTER[0] + inner_R * np.sin(theta_plot),
                "c--", linewidth=1.2, alpha=0.8, label="true inner")
        ax.plot(OUTER_CENTER[1] + outer_R_true * np.cos(theta_plot), OUTER_CENTER[0] + outer_R_true * np.sin(theta_plot),
                "c--", linewidth=1.2, alpha=0.8, label="true outer")
        # fitted boundaries (each around its OWN true center, per the script's design)
        fit_in, fit_out = all_fitted_inner[i], all_fitted_outer[i]
        ax.plot(INNER_CENTER[1] + fit_in * np.cos(theta_plot), INNER_CENTER[0] + fit_in * np.sin(theta_plot),
                "g:", linewidth=1.6, alpha=0.9, label="fitted inner")
        ax.plot(OUTER_CENTER[1] + fit_out * np.cos(theta_plot), OUTER_CENTER[0] + fit_out * np.sin(theta_plot),
                "g:", linewidth=1.6, alpha=0.9, label="fitted outer")
        in_err = abs(fit_in - inner_R) * dx[0] * 1e3
        out_err = abs(fit_out - outer_R_true) * dx[0] * 1e3
        ax.set_title(f"phase={phases[i]:.2f}\nin_err={in_err:.2f}mm out_err={out_err:.2f}mm", fontsize=8)
        ax.axis("off")
    for ax in axes[len(frames):]:
        ax.axis("off")
    fig.suptitle("Eccentric off-center myocardial ring, 8-frame cycle\n"
                "cyan dashed = true boundaries (inner+outer, different centers), green dotted = fitted\n"
                "(TOY: exact prescribed ground truth; curvature-weighted + guard-band fit)", fontsize=10)
    plt.tight_layout(rect=[0, 0.02, 1, 0.86])
    labels.add_banner(fig)
    os.makedirs("results/figures", exist_ok=True)
    plt.savefig("results/figures/phase3_ring_eccentric_offcenter_filmstrip.png", dpi=130)
    print("\nSaved results/figures/phase3_ring_eccentric_offcenter_filmstrip.png")

    # --- Figure: single ED-frame close-up, showing the eccentric wall clearly ---
    fig2, ax2 = plt.subplots(figsize=(6.5, 6.5))
    ed_idx = 0
    inner_R_ed = inner_radii[ed_idx]
    outer_R_ed = inner_R_ed + WALL_THICKNESS
    ax2.imshow(np.abs(frames[ed_idx]), cmap="hot", vmin=0, vmax=vmax, origin="upper",
               extent=[img_cols_w.min(), img_cols_w.max(), img_rows_w.max(), img_rows_w.min()])
    ax2.plot(INNER_CENTER[1] + inner_R_ed * np.cos(theta_plot), INNER_CENTER[0] + inner_R_ed * np.sin(theta_plot),
             "c--", linewidth=1.5, alpha=0.85, label="true inner (LV cavity)")
    ax2.plot(OUTER_CENTER[1] + outer_R_ed * np.cos(theta_plot), OUTER_CENTER[0] + outer_R_ed * np.sin(theta_plot),
             "c--", linewidth=1.5, alpha=0.85, label="true outer (epicardium)")
    ax2.plot(INNER_CENTER[1] + all_fitted_inner[ed_idx] * np.cos(theta_plot),
              INNER_CENTER[0] + all_fitted_inner[ed_idx] * np.sin(theta_plot),
              "g:", linewidth=2.0, alpha=0.95, label="fitted inner")
    ax2.plot(OUTER_CENTER[1] + all_fitted_outer[ed_idx] * np.cos(theta_plot),
              OUTER_CENTER[0] + all_fitted_outer[ed_idx] * np.sin(theta_plot),
              "g:", linewidth=2.0, alpha=0.95, label="fitted outer")
    ax2.plot(INNER_CENTER[1], INNER_CENTER[0], "wx", markersize=10, markeredgewidth=2, label="inner center")
    ax2.plot(OUTER_CENTER[1], OUTER_CENTER[0], "w+", markersize=12, markeredgewidth=2, label="outer center")
    ax2.set_title(f"ED frame close-up: eccentric wall (thin side ~{(WALL_THICKNESS-ECC_MAGNITUDE)*dx[0]*1e3:.2f}mm, "
                  f"thick side ~{(WALL_THICKNESS+ECC_MAGNITUDE)*dx[0]*1e3:.2f}mm)", fontsize=10)
    ax2.legend(fontsize=8, loc="upper left")
    ax2.axis("off")
    labels.add_banner(fig2)
    plt.tight_layout()
    plt.savefig("results/figures/phase3_ring_eccentric_offcenter_ED_closeup.png", dpi=140)
    print("Saved results/figures/phase3_ring_eccentric_offcenter_ED_closeup.png")
