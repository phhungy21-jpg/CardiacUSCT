"""Phase 3 — off-center, CONCAVE heart-shaped phantom: an 8-frame test
of how the validated multistatic backprojection pipeline (naive +
run -34's `backproject_no_adjacent` fix) behaves on a shape harder than
anything tried so far in this thread.

Per user request: "try an 8 frame offcenter (your choice of
coordinates) heart shape. this one have a concave region and i want to
know how it behaves." Everything upstream is unchanged and already
validated (4-probe geometry, capture, envelope detection, direct-
arrival exclusion, group-delay correction, `backproject` and
`backproject_no_adjacent` from phase3_backprojection_shape_fit_triangle.py)
-- only the phantom shape changes, to a simplified 10-vertex heart
polygon (clockwise: bottom tip -> right flank -> right lobe -> NOTCH
(concave vertex, dips inward between the two lobes) -> left lobe ->
left flank -> back to tip), scaled by the same ED/ES radius schedule
used throughout this thread, and placed off-center (10, -15 cells from
domain center -- an arbitrary but grid-safe choice).

Two genuinely new things this shape can reveal that the circle/triangle
couldn't: (1) run -36 confirmed the triangle's "ghosts" are real corner
diffraction from its (convex) vertex -- a concave vertex (the notch) is
also a sharp curvature discontinuity and should diffract too, by the
same physics, so watch for a similar false-peak signature near it; (2)
concavity can genuinely block direct line-of-sight from a probe to part
of the boundary (self-occlusion/shadowing) -- something no convex shape
in this project has been able to test.

Point-in-polygon uses matplotlib.path.Path.contains_points (a general
algorithm, unlike the convex-only same-side-of-every-edge trick used
for the triangle -- that trick is NOT valid for a concave polygon).
"""

import numpy as np
from matplotlib.path import Path

from jax import numpy as jnp
from jwave import FourierSeries
from jwave.geometry import Medium

from phase3_backprojection_shape_fit_triangle import (
    capture_all_pairs, build_medium_homogeneous, backproject,
    backproject_no_adjacent, img_rows, img_cols, center, dx, N, domain,
    cfg, p3cfg, labels,
)

from matplotlib import pyplot as plt
import os

OFFSET = (10, -15)  # (row, col) cells from domain center -- arbitrary, grid-safe
SHIFTED_CENTER = (center[0] + OFFSET[0], center[1] + OFFSET[1])

# Unit heart polygon (dx, dy), dy-up convention (matches this thread's
# "row = center - dy*R" convention), clockwise from the bottom tip.
# Vertex 5 (0.00, 0.40) is the concave NOTCH -- it dips inward relative
# to both neighboring lobe-top vertices, the defining concave feature.
HEART_UNIT_VERTICES = [
    (0.00, -1.00),   # 0: bottom tip
    (0.60, -0.30),   # 1: right lower flank
    (0.95, 0.25),    # 2: right lobe, outer widest point
    (0.75, 0.70),    # 3: right lobe top (outer)
    (0.35, 0.75),    # 4: right lobe top (inner, approaching notch)
    (0.00, 0.40),    # 5: NOTCH (concave vertex)
    (-0.35, 0.75),   # 6: left lobe top (inner)
    (-0.75, 0.70),   # 7: left lobe top (outer)
    (-0.95, 0.25),   # 8: left lobe, outer widest point
    (-0.60, -0.30),  # 9: left lower flank
]


def heart_vertices(R):
    """Returns (row, col) vertices scaled by R, centered at SHIFTED_CENTER."""
    return [(SHIFTED_CENTER[0] - dy * R, SHIFTED_CENTER[1] + dx_ * R)
            for dx_, dy in HEART_UNIT_VERTICES]


def build_medium_heart(R):
    verts = heart_vertices(R)
    path = Path(verts)
    yy, xx = np.mgrid[0:N[0], 0:N[1]]
    points = np.column_stack([yy.ravel(), xx.ravel()])
    inside = path.contains_points(points).reshape(N)

    sound_speed_map = np.where(inside, cfg.BLOOD.sound_speed, cfg.CHEST_WALL_PROXY.sound_speed).astype(np.float32)
    density_map = np.where(inside, cfg.BLOOD.density, cfg.CHEST_WALL_PROXY.density).astype(np.float32)
    ssm = jnp.expand_dims(jnp.array(sound_speed_map), -1)
    dm = jnp.expand_dims(jnp.array(density_map), -1)
    return Medium(domain=domain, sound_speed=FourierSeries(ssm, domain),
                  density=FourierSeries(dm, domain))


if __name__ == "__main__":
    print(f"*** {labels.PENDING_SIGNOFF_BANNER} ***")
    print(f"*** {labels.TOY_EXACT_GT_CAPTION} ***")
    print(f"Off-center CONCAVE heart-shape test: same validated 4-probe "
          f"multistatic pipeline (naive + no-adjacent-pairs fix), phantom "
          f"shifted by {OFFSET} cells to {SHIFTED_CENTER}, 8-frame ED->ES->ED "
          f"sweep. Testing behavior on a concave boundary (the notch) and "
          f"an off-center placement together, for the first time in this "
          f"thread.")

    print("\n=== Control: homogeneous medium (no heart) ===")
    pairs_ref = capture_all_pairs(build_medium_homogeneous())
    accumulator_ref_naive = backproject(pairs_ref)
    accumulator_ref_noadj = backproject_no_adjacent(pairs_ref)

    N_FRAMES_MOVIE = 8
    phases = np.linspace(0, 1, N_FRAMES_MOVIE)
    radii_cells = [p3cfg.lv_radius_at_phase(p) for p in phases]

    frames_naive = []
    frames_noadj = []
    for i, R in enumerate(radii_cells):
        print(f"=== Frame {i+1}/{N_FRAMES_MOVIE} (R={R:.1f} cells = {R*dx[0]*1e3:.2f}mm) ===")
        pairs = capture_all_pairs(build_medium_heart(R))
        frames_naive.append(backproject(pairs) - accumulator_ref_naive)
        frames_noadj.append(backproject_no_adjacent(pairs) - accumulator_ref_noadj)

    # --- Figure 1: ED-frame naive vs no-adjacent side by side (like the
    # triangle's ghost-comparison figure) ---
    fig1, axes1 = plt.subplots(1, 2, figsize=(10, 5.2))
    R_ed = radii_cells[0]
    verts_ed = heart_vertices(R_ed)
    heart_row = [v[0] for v in verts_ed] + [verts_ed[0][0]]
    heart_col = [v[1] for v in verts_ed] + [verts_ed[0][1]]
    vmax1 = max(np.abs(frames_naive[0]).max(), np.abs(frames_noadj[0]).max())
    for ax, frame, title in [(axes1[0], frames_naive[0], "naive (all 16 pairs)"),
                              (axes1[1], frames_noadj[0], "no-adjacent-pairs fix")]:
        ax.imshow(np.abs(frame), cmap="hot", vmin=0, vmax=vmax1, origin="upper",
                  extent=[img_cols.min(), img_cols.max(), img_rows.max(), img_rows.min()])
        ax.plot(heart_col, heart_row, "c--", linewidth=1.3, alpha=0.85)
        ax.plot(SHIFTED_CENTER[1], SHIFTED_CENTER[0], "g+", markersize=10, markeredgewidth=2)
        ax.set_title(title, fontsize=10)
        ax.axis("off")
    fig1.suptitle(f"ED frame: off-center concave heart, naive vs no-adjacent-pairs\n"
                  f"cyan dashed = true heart boundary (notch at top-center)", fontsize=10)
    plt.tight_layout(rect=[0, 0.02, 1, 0.88])
    labels.add_banner(fig1)
    os.makedirs("results/figures", exist_ok=True)
    plt.savefig("results/figures/phase3_heart_shape_offcenter_ED_comparison.png", dpi=140)
    print("\nSaved results/figures/phase3_heart_shape_offcenter_ED_comparison.png")

    # --- Figure 2: full 8-frame filmstrip, no-adjacent-pairs variant ---
    n_cols = 4
    n_rows = int(np.ceil(N_FRAMES_MOVIE / n_cols))
    fig2, axes2 = plt.subplots(n_rows, n_cols, figsize=(3.2 * n_cols, 3.4 * n_rows))
    axes2 = np.array(axes2).reshape(-1)
    vmax2 = max(np.abs(f).max() for f in frames_noadj)
    for i, (ax, frame, R) in enumerate(zip(axes2, frames_noadj, radii_cells)):
        ax.imshow(np.abs(frame), cmap="hot", vmin=0, vmax=vmax2, origin="upper",
                  extent=[img_cols.min(), img_cols.max(), img_rows.max(), img_rows.min()])
        verts = heart_vertices(R)
        h_row = [v[0] for v in verts] + [verts[0][0]]
        h_col = [v[1] for v in verts] + [verts[0][1]]
        ax.plot(h_col, h_row, "c--", linewidth=1, alpha=0.7)
        ax.set_title(f"phase={phases[i]:.2f}, R={R*dx[0]*1e3:.2f}mm", fontsize=8)
        ax.axis("off")
    for ax in axes2[len(frames_noadj):]:
        ax.axis("off")
    fig2.suptitle("Off-center concave heart shape -- no-adjacent-pairs backprojection\n"
                "cyan dashed = true boundary (notch = concave feature, top-center)\n"
                "(TOY: exact prescribed ground truth)", fontsize=10)
    plt.tight_layout(rect=[0, 0.02, 1, 0.88])
    labels.add_banner(fig2)
    plt.savefig("results/figures/phase3_heart_shape_offcenter_filmstrip.png", dpi=130)
    print("\nSaved results/figures/phase3_heart_shape_offcenter_filmstrip.png")

    # --- Figure 3: same filmstrip, naive variant, for direct comparison ---
    fig3, axes3 = plt.subplots(n_rows, n_cols, figsize=(3.2 * n_cols, 3.4 * n_rows))
    axes3 = np.array(axes3).reshape(-1)
    vmax3 = max(np.abs(f).max() for f in frames_naive)
    for i, (ax, frame, R) in enumerate(zip(axes3, frames_naive, radii_cells)):
        ax.imshow(np.abs(frame), cmap="hot", vmin=0, vmax=vmax3, origin="upper",
                  extent=[img_cols.min(), img_cols.max(), img_rows.max(), img_rows.min()])
        verts = heart_vertices(R)
        h_row = [v[0] for v in verts] + [verts[0][0]]
        h_col = [v[1] for v in verts] + [verts[0][1]]
        ax.plot(h_col, h_row, "c--", linewidth=1, alpha=0.7)
        ax.set_title(f"phase={phases[i]:.2f}, R={R*dx[0]*1e3:.2f}mm", fontsize=8)
        ax.axis("off")
    for ax in axes3[len(frames_naive):]:
        ax.axis("off")
    fig3.suptitle("Off-center concave heart shape -- NAIVE backprojection (all 16 pairs)\n"
                "cyan dashed = true boundary (notch = concave feature, top-center)\n"
                "(TOY: exact prescribed ground truth)", fontsize=10)
    plt.tight_layout(rect=[0, 0.02, 1, 0.88])
    labels.add_banner(fig3)
    plt.savefig("results/figures/phase3_heart_shape_offcenter_filmstrip_naive.png", dpi=130)
    print("\nSaved results/figures/phase3_heart_shape_offcenter_filmstrip_naive.png")
