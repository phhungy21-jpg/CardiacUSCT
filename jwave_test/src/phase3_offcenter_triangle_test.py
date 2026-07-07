"""Phase 3 — off-center triangle test: does moving the target away from
the domain/array center change the adjacent-pair ghost pattern found in
runs -29/-33/-34, or is it a fixed property of the 4-probe array
regardless of target position?

All prior triangle tests placed the phantom exactly at the domain
center, equidistant from all 4 probes -- a special, maximally symmetric
case. Per user request ("try off-center triangle to see if that reveals
anything"): shift the SAME equilateral triangle away from center by a
modest offset (still comfortably inside the accumulator search grid),
keep the exact same 4-probe geometry and multistatic machinery (no
changes), and visually/diagnostically compare:
- does the naive accumulator's false-peak pattern move WITH the target
  (suggesting the ghost mechanism is about the target's relationship to
  the probes, which shifts as the target moves), or stay anchored to
  the old, domain-centered locations (suggesting something fixed about
  the array/domain setup itself, independent of the target)?
- does the already-validated `backproject_no_adjacent` fix (built from
  the centered-case diagnosis) still clean up the image reasonably for
  an off-center target, or does decentering break its assumptions?

This reuses every validated piece of machinery unchanged (probes,
capture, envelope detection, direct-arrival exclusion, group-delay
correction, backproject/backproject_no_adjacent) -- only the phantom's
position changes.
"""

import numpy as np
from scipy.interpolate import RegularGridInterpolator

from phase3_backprojection_shape_fit_triangle import (
    capture_all_pairs, build_medium_homogeneous, backproject,
    backproject_no_adjacent, img_rows, img_cols, center, dx, RR, CC,
    p3cfg, labels, N, domain, cfg,
)
from jax import numpy as jnp
from jwave import FourierSeries
from jwave.geometry import Medium

from matplotlib import pyplot as plt
import os

_SQRT3_2 = 0.8660254037844386

# Offset chosen to keep the ED-sized (R=60) triangle comfortably inside
# the accumulator search grid (domain center +/- 90 cells).
OFFSET = (15, 10)  # (row, col) cells, shifts the phantom up-right-ish (toward top/right probes)
SHIFTED_CENTER = (center[0] + OFFSET[0], center[1] + OFFSET[1])


def triangle_vertices_offcenter(R):
    top = (SHIFTED_CENTER[0] - R, SHIFTED_CENTER[1])
    botleft = (SHIFTED_CENTER[0] + 0.5 * R, SHIFTED_CENTER[1] - _SQRT3_2 * R)
    botright = (SHIFTED_CENTER[0] + 0.5 * R, SHIFTED_CENTER[1] + _SQRT3_2 * R)
    return top, botleft, botright


def build_medium_triangle_offcenter(R):
    top, botleft, botright = triangle_vertices_offcenter(R)
    yy, xx = np.mgrid[0:N[0], 0:N[1]]

    def edge_sign(pt_row, pt_col, a, b):
        return (pt_row - b[0]) * (a[1] - b[1]) - (a[0] - b[0]) * (pt_col - b[1])

    d1 = edge_sign(yy, xx, top, botleft)
    d2 = edge_sign(yy, xx, botleft, botright)
    d3 = edge_sign(yy, xx, botright, top)
    has_neg = (d1 < 0) | (d2 < 0) | (d3 < 0)
    has_pos = (d1 > 0) | (d2 > 0) | (d3 > 0)
    inside = ~(has_neg & has_pos)

    sound_speed_map = np.where(inside, cfg.BLOOD.sound_speed, cfg.CHEST_WALL_PROXY.sound_speed).astype(np.float32)
    density_map = np.where(inside, cfg.BLOOD.density, cfg.CHEST_WALL_PROXY.density).astype(np.float32)
    ssm = jnp.expand_dims(jnp.array(sound_speed_map), -1)
    dm = jnp.expand_dims(jnp.array(density_map), -1)
    return Medium(domain=domain, sound_speed=FourierSeries(ssm, domain),
                  density=FourierSeries(dm, domain))


if __name__ == "__main__":
    print(f"*** {labels.PENDING_SIGNOFF_BANNER} ***")
    print(f"*** {labels.TOY_EXACT_GT_CAPTION} ***")
    print(f"Off-center triangle test: same phantom/probes/machinery as runs "
          f"-29..-34, but the triangle is shifted by {OFFSET} cells from "
          f"domain center (to {SHIFTED_CENTER}) -- testing whether the "
          f"adjacent-pair ghost pattern is tied to the target's position "
          f"or is a fixed array/domain artifact.")

    R = p3cfg.LV_RADIUS_ED_CELLS  # 60 cells, same ED case examined throughout
    print(f"\n=== Capturing pairs: R={R} cells, offset={OFFSET} + homogeneous reference ===")
    pairs_ref = capture_all_pairs(build_medium_homogeneous())
    pairs_tri = capture_all_pairs(build_medium_triangle_offcenter(R))

    accumulator_naive = backproject(pairs_tri) - backproject(pairs_ref)
    accumulator_noadj = backproject_no_adjacent(pairs_tri) - backproject_no_adjacent(pairs_ref)

    top, botleft, botright = triangle_vertices_offcenter(R)
    tri_row = [top[0], botleft[0], botright[0], top[0]]
    tri_col = [top[1], botleft[1], botright[1], top[1]]

    # Find the global peak in each image and report its position relative
    # to both the TRUE (shifted) center and the OLD domain center -- this
    # directly shows whether any false peak tracks the target's new
    # position or stays anchored to the old symmetric-case locations.
    interp_naive = RegularGridInterpolator((img_rows, img_cols), np.abs(accumulator_naive),
                                            bounds_error=False, fill_value=0.0)
    peak_idx_naive = np.unravel_index(np.argmax(np.abs(accumulator_naive)), accumulator_naive.shape)
    peak_row_naive, peak_col_naive = img_rows[peak_idx_naive[0]], img_cols[peak_idx_naive[1]]
    print(f"\nNaive accumulator global peak: (row={peak_row_naive:.1f}, col={peak_col_naive:.1f})")
    print(f"  offset from TRUE shifted center {SHIFTED_CENTER}: "
          f"({peak_row_naive - SHIFTED_CENTER[0]:.1f}, {peak_col_naive - SHIFTED_CENTER[1]:.1f})")
    print(f"  offset from OLD domain center {center}: "
          f"({peak_row_naive - center[0]:.1f}, {peak_col_naive - center[1]:.1f})")

    peak_idx_noadj = np.unravel_index(np.argmax(np.abs(accumulator_noadj)), accumulator_noadj.shape)
    peak_row_noadj, peak_col_noadj = img_rows[peak_idx_noadj[0]], img_cols[peak_idx_noadj[1]]
    print(f"\nNo-adjacent-pairs accumulator global peak: (row={peak_row_noadj:.1f}, col={peak_col_noadj:.1f})")
    print(f"  offset from TRUE shifted center {SHIFTED_CENTER}: "
          f"({peak_row_noadj - SHIFTED_CENTER[0]:.1f}, {peak_col_noadj - SHIFTED_CENTER[1]:.1f})")

    # Per-axis check (through the SHIFTED center, matching the true
    # triangle's own symmetry axis, not the old domain-centered axes).
    def direction_vector(theta_deg):
        theta = np.deg2rad(theta_deg)
        return -np.cos(theta), np.sin(theta)

    def ray_segment_intersection(orig, d_row, d_col, p1, p2):
        ax_, ay_ = orig
        bx, by = p1
        ex, ey = p2[0] - p1[0], p2[1] - p1[1]
        denom = d_row * ey - d_col * ex
        if abs(denom) < 1e-9:
            return None
        t = ((bx - ax_) * ey - (by - ay_) * ex) / denom
        s = ((bx - ax_) * d_col - (by - ay_) * d_row) / denom
        if t > 0 and 0 <= s <= 1:
            return t
        return None

    def true_dist_from_shifted_center(theta_deg):
        d_row, d_col = direction_vector(theta_deg)
        for p1, p2 in [(top, botleft), (botleft, botright), (botright, top)]:
            t = ray_segment_intersection(SHIFTED_CENTER, d_row, d_col, p1, p2)
            if t is not None:
                return t
        return None

    print("\n--- Per-axis peak search, rays from the TRUE SHIFTED center (not old domain center) ---")
    for name, theta in [("top", 0), ("right", 90), ("bottom", 180), ("left", 270)]:
        d_row, d_col = direction_vector(theta)
        d_grid = np.arange(5.0, 90.0, 0.25)
        pts_naive = np.array([(SHIFTED_CENTER[0] + d * d_row, SHIFTED_CENTER[1] + d * d_col) for d in d_grid])
        vals_naive = interp_naive(pts_naive)
        found_naive = d_grid[np.argmax(vals_naive)]
        expected = true_dist_from_shifted_center(theta)
        err_mm = abs(found_naive - expected) * dx[0] * 1e3
        print(f"  {name}: true={expected:.1f} cells, naive found={found_naive:.1f} cells, error={err_mm:.2f}mm")

    # Visual: naive vs no-adjacent, both with the TRUE shifted triangle overlaid.
    fig, axes = plt.subplots(1, 2, figsize=(11, 5))
    vmax = max(np.abs(accumulator_naive).max(), np.abs(accumulator_noadj).max())
    for ax, img, title in [
        (axes[0], accumulator_naive, "naive (all 16 pairs)"),
        (axes[1], accumulator_noadj, "no-adjacent-pairs fix"),
    ]:
        ax.imshow(np.abs(img), cmap="hot", vmin=0, vmax=vmax, origin="upper",
                  extent=[img_cols.min(), img_cols.max(), img_rows.max(), img_rows.min()])
        ax.plot(tri_col, tri_row, "c--", linewidth=1.3, alpha=0.8, label="true (shifted)")
        ax.plot(center[1], center[0], "wx", markersize=8, markeredgewidth=2, label="old domain center")
        ax.plot(SHIFTED_CENTER[1], SHIFTED_CENTER[0], "g+", markersize=10, markeredgewidth=2, label="true center")
        ax.set_title(title, fontsize=10)
        ax.legend(fontsize=7, loc="upper left")
        ax.axis("off")
    fig.suptitle(f"Off-center triangle (offset={OFFSET} cells), R=60 (ED)\n"
                "cyan dashed = true (shifted) triangle boundary\n"
                "(TOY: exact prescribed ground truth)", fontsize=10)
    plt.tight_layout(rect=[0, 0.02, 1, 0.86])
    labels.add_banner(fig)
    os.makedirs("results/figures", exist_ok=True)
    plt.savefig("results/figures/phase3_offcenter_triangle_test.png", dpi=140)
    print("\nSaved results/figures/phase3_offcenter_triangle_test.png")
