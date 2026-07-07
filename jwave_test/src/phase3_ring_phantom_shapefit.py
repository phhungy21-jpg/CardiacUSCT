"""Phase 3 — the actual "heart phantom": two-boundary myocardial ring
(inner LV cavity / outer epicardial boundary, constant wall thickness),
using the validated naive + global shape-fit pipeline (runs -28, -32,
-38) rather than the single-boundary toy shapes (circle, triangle,
heart-cartoon) used so far in this thread.

Per user: "log and proceed to heart phantom I guess?" after confirming
the concave heart-cartoon shape-fit result (run -38) was very good.
This is the real escalation: TWO concentric reflecting boundaries
(blood/myocardium interface, myocardium/chest-wall-proxy interface),
using this project's own established anatomical model
(`phase3_config.py`: LV_RADIUS_ED_CELLS=60, LV_RADIUS_ES_CELLS=40,
WALL_THICKNESS_CELLS=30, myocardial wall thickness held constant while
only the LV cavity radius contracts) and cited tissue properties
(`phase2_config.py`: BLOOD, MYOCARDIUM, CHEST_WALL_PROXY).

Since both boundaries are simple concentric CIRCLES (not the
triangle/heart's polygon shapes), each one's ray-distance is trivially
constant (= its own radius) at every angle -- no analytic polygon
intersection needed. Fits the inner and outer radius INDEPENDENTLY,
each via the same "sweep candidate R, integrate accumulator energy over
many angles, pick the best" global template-match principle validated
on every shape tried so far this thread. Centered at domain center for
this first pass (matches this project's established ring-phantom
convention elsewhere) -- off-center + two boundaries together is a
natural follow-on, not attempted here.
"""

import numpy as np
from scipy.interpolate import RegularGridInterpolator

from jax import numpy as jnp
from jwave import FourierSeries
from jwave.geometry import Medium

from phase3_backprojection_shape_fit_triangle import (
    capture_all_pairs, build_medium_homogeneous, backproject,
    direction_vector, img_rows, img_cols, center, dx, N, domain, cfg,
    p3cfg, labels,
)

from matplotlib import pyplot as plt
import os


def build_medium_ring(inner_R, wall_thickness=p3cfg.WALL_THICKNESS_CELLS):
    yy, xx = np.mgrid[0:N[0], 0:N[1]]
    dist = np.sqrt((xx - center[1]) ** 2 + (yy - center[0]) ** 2)
    outer_R = inner_R + wall_thickness

    label_map = np.zeros(N, dtype=int)
    label_map[dist < outer_R] = 2   # myocardium (ring)
    label_map[dist < inner_R] = 3   # blood (LV cavity)
    # default (dist >= outer_R) stays label 0 = chest-wall-proxy

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


N_ANGLES = 72
_THETAS = np.linspace(0, 360, N_ANGLES, endpoint=False)


def fit_circle_radius(accumulator, R_grid, origin=center):
    """Same global template-match principle as runs -32/-38, specialized
    to a circle (ray-distance = R at every angle, trivially -- no
    polygon intersection needed)."""
    interp = RegularGridInterpolator((img_rows, img_cols), np.abs(accumulator),
                                      bounds_error=False, fill_value=0.0)
    scores = np.zeros(len(R_grid))
    for i, R in enumerate(R_grid):
        pts = []
        for th in _THETAS:
            d_row, d_col = direction_vector(th)
            pts.append((origin[0] + R * d_row, origin[1] + R * d_col))
        scores[i] = interp(np.array(pts)).sum()
    best_R = R_grid[np.argmax(scores)]
    return best_R, scores


INNER_R_GRID = np.arange(25.0, 75.0, 0.25)
OUTER_R_GRID = np.arange(55.0, 110.0, 0.25)  # inner + wall_thickness(30), with margin


if __name__ == "__main__":
    print(f"*** {labels.PENDING_SIGNOFF_BANNER} ***")
    print(f"*** {labels.TOY_EXACT_GT_CAPTION} ***")
    print("Ring (myocardial) phantom -- the project's actual 'heart "
          "phantom' -- using the validated naive + global shape-fit "
          "pipeline. Two concentric boundaries (inner LV cavity, outer "
          "epicardial), each fit independently via the same template-"
          "match principle used for the circle/triangle/heart-cartoon "
          "shapes earlier in this thread. Centered at domain center "
          "(this project's established ring-phantom convention).")

    print("\n=== Control: homogeneous medium (no ring) ===")
    pairs_ref = capture_all_pairs(build_medium_homogeneous())
    accumulator_ref = backproject(pairs_ref)
    ref_inner_R, _ = fit_circle_radius(accumulator_ref, INNER_R_GRID)
    ref_outer_R, _ = fit_circle_radius(accumulator_ref, OUTER_R_GRID)
    print(f"  homogeneous-medium fitted: inner={ref_inner_R:.1f}, outer={ref_outer_R:.1f} cells "
          f"(should be meaningless/low-confidence)")

    N_FRAMES_MOVIE = 8
    phases = np.linspace(0, 1, N_FRAMES_MOVIE)
    inner_radii = [p3cfg.lv_radius_at_phase(p) for p in phases]

    frames = []
    all_fitted_inner = []
    all_fitted_outer = []
    ed_scores_inner = None
    ed_scores_outer = None
    for i, inner_R in enumerate(inner_radii):
        outer_R_true = inner_R + p3cfg.WALL_THICKNESS_CELLS
        print(f"=== Frame {i+1}/{N_FRAMES_MOVIE} (inner_R={inner_R:.1f}, outer_R={outer_R_true:.1f} cells) ===")
        pairs = capture_all_pairs(build_medium_ring(inner_R))
        accumulator_clean = backproject(pairs) - accumulator_ref
        frames.append(accumulator_clean)

        fitted_inner, scores_inner = fit_circle_radius(accumulator_clean, INNER_R_GRID)
        fitted_outer, scores_outer = fit_circle_radius(accumulator_clean, OUTER_R_GRID)
        all_fitted_inner.append(fitted_inner)
        all_fitted_outer.append(fitted_outer)
        if i == 0:
            ed_scores_inner = scores_inner
            ed_scores_outer = scores_outer

        err_inner_mm = abs(fitted_inner - inner_R) * dx[0] * 1e3
        err_outer_mm = abs(fitted_outer - outer_R_true) * dx[0] * 1e3
        print(f"  inner: true={inner_R:.1f}, fitted={fitted_inner:.1f}, err={err_inner_mm:.2f}mm | "
              f"outer: true={outer_R_true:.1f}, fitted={fitted_outer:.1f}, err={err_outer_mm:.2f}mm")

    print("\n--- RMSE across all frames ---")
    inner_errs = np.array([abs(all_fitted_inner[i] - inner_radii[i]) * dx[0] * 1e3 for i in range(N_FRAMES_MOVIE)])
    outer_errs = np.array([abs(all_fitted_outer[i] - (inner_radii[i] + p3cfg.WALL_THICKNESS_CELLS)) * dx[0] * 1e3
                            for i in range(N_FRAMES_MOVIE)])
    print(f"  inner boundary RMSE={np.sqrt(np.mean(inner_errs**2)):.4f}mm  (per-frame: {np.round(inner_errs,2).tolist()})")
    print(f"  outer boundary RMSE={np.sqrt(np.mean(outer_errs**2)):.4f}mm  (per-frame: {np.round(outer_errs,2).tolist()})")

    # --- Figure 1: score(R) vs R, ED frame, both boundaries ---
    fig1, axes1 = plt.subplots(1, 2, figsize=(11, 4.5))
    axes1[0].plot(INNER_R_GRID, ed_scores_inner / ed_scores_inner.max())
    axes1[0].axvline(inner_radii[0], color="k", linestyle="--", label=f"true={inner_radii[0]:.1f}")
    axes1[0].axvline(all_fitted_inner[0], color="g", linestyle=":", label=f"fitted={all_fitted_inner[0]:.1f}")
    axes1[0].set_title("Inner (LV cavity) boundary")
    axes1[0].set_xlabel("candidate R (cells)")
    axes1[0].legend(fontsize=8)

    axes1[1].plot(OUTER_R_GRID, ed_scores_outer / ed_scores_outer.max())
    outer_true_ed = inner_radii[0] + p3cfg.WALL_THICKNESS_CELLS
    axes1[1].axvline(outer_true_ed, color="k", linestyle="--", label=f"true={outer_true_ed:.1f}")
    axes1[1].axvline(all_fitted_outer[0], color="g", linestyle=":", label=f"fitted={all_fitted_outer[0]:.1f}")
    axes1[1].set_title("Outer (epicardial) boundary")
    axes1[1].set_xlabel("candidate R (cells)")
    axes1[1].legend(fontsize=8)
    fig1.suptitle("ED frame: ring-phantom shape-fit score vs. candidate R", fontsize=11)
    labels.add_banner(fig1)
    plt.tight_layout()
    os.makedirs("results/figures", exist_ok=True)
    plt.savefig("results/figures/phase3_ring_phantom_shapefit_score_curve.png", dpi=130)
    print("\nSaved results/figures/phase3_ring_phantom_shapefit_score_curve.png")

    # --- Figure 2: full 8-frame filmstrip, true + fitted rings overlaid ---
    n_cols = 4
    n_rows = int(np.ceil(N_FRAMES_MOVIE / n_cols))
    fig2, axes2 = plt.subplots(n_rows, n_cols, figsize=(3.2 * n_cols, 3.4 * n_rows))
    axes2 = np.array(axes2).reshape(-1)
    vmax = max(np.abs(f).max() for f in frames)
    theta_plot = np.linspace(0, 2 * np.pi, 100)
    for i, (ax, frame, inner_R) in enumerate(zip(axes2, frames, inner_radii)):
        ax.imshow(np.abs(frame), cmap="hot", vmin=0, vmax=vmax, origin="upper",
                  extent=[img_cols.min(), img_cols.max(), img_rows.max(), img_rows.min()])
        outer_R_true = inner_R + p3cfg.WALL_THICKNESS_CELLS
        for R_true, style in [(inner_R, "c--"), (outer_R_true, "c--")]:
            ax.plot(center[1] + R_true * np.cos(theta_plot), center[0] + R_true * np.sin(theta_plot),
                    style, linewidth=1, alpha=0.7)
        for R_fit, style in [(all_fitted_inner[i], "g:"), (all_fitted_outer[i], "g:")]:
            ax.plot(center[1] + R_fit * np.cos(theta_plot), center[0] + R_fit * np.sin(theta_plot),
                    style, linewidth=1.5, alpha=0.9)
        err_in_mm = abs(all_fitted_inner[i] - inner_R) * dx[0] * 1e3
        err_out_mm = abs(all_fitted_outer[i] - outer_R_true) * dx[0] * 1e3
        ax.set_title(f"phase={phases[i]:.2f}\nin_err={err_in_mm:.2f}mm, out_err={err_out_mm:.2f}mm", fontsize=8)
        ax.axis("off")
    for ax in axes2[len(frames):]:
        ax.axis("off")
    fig2.suptitle("Myocardial ring phantom -- naive + global shape-fit (both boundaries)\n"
                "cyan dashed = true, green dotted = fitted\n"
                "(TOY: exact prescribed ground truth)", fontsize=10)
    plt.tight_layout(rect=[0, 0.02, 1, 0.86])
    labels.add_banner(fig2)
    plt.savefig("results/figures/phase3_ring_phantom_shapefit_filmstrip.png", dpi=130)
    print("\nSaved results/figures/phase3_ring_phantom_shapefit_filmstrip.png")
