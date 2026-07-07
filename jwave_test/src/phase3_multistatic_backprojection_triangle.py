"""Phase 3 — multistatic backprojection on a beating TRIANGLE, not a circle.

Direct extension of phase3_multistatic_backprojection.py (run -28,
RMSE=0.24mm, no fixed artifact on a filled disk). Per user request: try a
triangle next, as a smoke test of whether the method generalizes past a
smooth, circularly-symmetric reflector. A triangle is a much harder,
more informative test of the SAME idea:
- the "top" probe's boresight ray hits a VERTEX head-on (a point/corner
  reflector, weaker and more diffuse than a smooth specular surface)
- the "bottom" probe's boresight ray hits the midpoint of the OPPOSITE
  EDGE, at normal incidence (should behave like the circle case)
- "left"/"right" probes each hit an oblique FACET at non-normal
  incidence (mirror images of each other by symmetry)
All three reflection geometries in one phantom, using the exact same
multistatic accumulation machinery (no changes to probe geometry,
transmit/receive capture, envelope detection, direct-arrival exclusion,
or the group-delay correction found necessary in run -28) -- only the
phantom shape and the per-direction ground-truth distance formulas
change. If this degrades gracefully (worse at the corner, fine at the
normal-incidence edge) rather than breaking, that's a meaningful
generalization result before trying real (non-convex, two-boundary)
anatomy.

Equilateral triangle, one vertex pointing at the "top" probe (up, i.e.
decreasing row), circumradius R following the same ED/ES schedule as
the beating circle (p3cfg.lv_radius_at_phase). Ground-truth distance
from center to boundary along each probe's axis (derived by hand,
standard equilateral-triangle geometry):
  - top (through the vertex):        R
  - bottom (through opposite edge):  R/2       (inradius)
  - left / right (through a facet):  R/sqrt(3) (by symmetry, identical)
"""

import numpy as np
from scipy.signal import hilbert

from jax import numpy as jnp
from jax import jit
from jwave import FourierSeries
from jwave.geometry import Domain, Medium, TimeAxis, Sources
from jwave.acoustics import simulate_wave_propagation

import phase2_config as cfg
import phase3_config as p3cfg
import labels

from matplotlib import pyplot as plt
import os

c_ref = cfg.CHEST_WALL_PROXY.sound_speed

N = (300, 300)
dx = (cfg.DX_M, cfg.DX_M)
domain = Domain(N, dx)
center = (150, 150)
PROBE_DIST_CELLS = 120

PROBES = {
    "top":    dict(row=center[0] - PROBE_DIST_CELLS, col=center[1], axis="col"),
    "bottom": dict(row=center[0] + PROBE_DIST_CELLS, col=center[1], axis="col"),
    "left":   dict(row=center[0], col=center[1] - PROBE_DIST_CELLS, axis="row"),
    "right":  dict(row=center[0], col=center[1] + PROBE_DIST_CELLS, axis="row"),
}
PROBE_NAMES = list(PROBES.keys())


def probe_source_and_receiver(probe):
    row, col, axis = probe["row"], probe["col"], probe["axis"]
    if axis == "col":
        src = (col - 5, row)
        rcv = (col + 5, row)
    else:
        src = (col, row - 5)
        rcv = (col, row + 5)
    return src, rcv


_SRC = {name: probe_source_and_receiver(p)[0] for name, p in PROBES.items()}
_RCV = {name: probe_source_and_receiver(p)[1] for name, p in PROBES.items()}

_SQRT3_2 = 0.8660254037844386  # sqrt(3)/2


def triangle_vertices(R):
    """One vertex pointing at 'top' (up = decreasing row); opposite edge
    faces 'bottom' at normal incidence; left/right are mirror-symmetric
    oblique facets."""
    top = (center[0] - R, center[1])
    botleft = (center[0] + 0.5 * R, center[1] - _SQRT3_2 * R)
    botright = (center[0] + 0.5 * R, center[1] + _SQRT3_2 * R)
    return top, botleft, botright


def build_medium_triangle(R):
    top, botleft, botright = triangle_vertices(R)
    yy, xx = np.mgrid[0:N[0], 0:N[1]]  # yy=row, xx=col

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


def build_medium_homogeneous():
    sound_speed_map = np.full(N, cfg.CHEST_WALL_PROXY.sound_speed, dtype=np.float32)
    density_map = np.full(N, cfg.CHEST_WALL_PROXY.density, dtype=np.float32)
    ssm = jnp.expand_dims(jnp.array(sound_speed_map), -1)
    dm = jnp.expand_dims(jnp.array(density_map), -1)
    return Medium(domain=domain, sound_speed=FourierSeries(ssm, domain),
                  density=FourierSeries(dm, domain))


_dummy_medium = build_medium_triangle(p3cfg.LV_RADIUS_ED_CELLS)
_base_time_axis = TimeAxis.from_medium(_dummy_medium, cfl=cfg.CFL)
dt = _base_time_axis.dt

SEARCH_RADIUS_CELLS = 80
_max_leg_cells = PROBE_DIST_CELLS + SEARCH_RADIUS_CELLS
_t_end_needed = (2 * _max_leg_cells * dx[0] / c_ref) * 1.15
time_axis = TimeAxis(dt=dt, t_end=_t_end_needed)
n_steps = int(time_axis.Nt)
t_arr = np.arange(n_steps) * dt


def toneburst(t):
    duration = cfg.N_CYCLES / cfg.F0_HZ
    sigma = duration / 6
    window = np.exp(-(t - duration / 2) ** 2 / (2 * sigma ** 2))
    return np.sin(2 * np.pi * cfg.F0_HZ * t) * window


_signal_template = jnp.array(toneburst(t_arr))[None, :]

DIRECT_EXCLUDE_MARGIN_S = 1.5e-6


def simulate_probe_transmit(tx_name):
    src = _SRC[tx_name]
    sources = Sources(positions=([src[0]], [src[1]]), signals=_signal_template,
                      dt=dt, domain=domain)

    @jit
    def run(medium):
        return simulate_wave_propagation(medium, time_axis, sources=sources)

    def capture(medium):
        pressure = run(medium)
        field = pressure.on_grid[..., 0]
        traces = {}
        for rx_name, rcv in _RCV.items():
            trace = np.array(field[:, rcv[0], rcv[1]])
            direct_time = np.hypot(src[0] - rcv[0], src[1] - rcv[1]) * dx[0] / c_ref
            mask = np.abs(t_arr - direct_time) < DIRECT_EXCLUDE_MARGIN_S
            trace = trace.copy()
            trace[mask] = 0.0
            traces[rx_name] = trace
        return traces

    return capture


def capture_all_pairs(medium):
    pairs = {}
    for tx_name in PROBE_NAMES:
        capture = simulate_probe_transmit(tx_name)
        traces = capture(medium)
        for rx_name, trace in traces.items():
            envelope = np.abs(hilbert(trace))
            pairs[(tx_name, rx_name)] = envelope
    return pairs


IMG_N = 100
_grid_lo = center[0] - SEARCH_RADIUS_CELLS - 10
_grid_hi = center[0] + SEARCH_RADIUS_CELLS + 10
img_rows = np.linspace(_grid_lo, _grid_hi, IMG_N)
img_cols = np.linspace(_grid_lo, _grid_hi, IMG_N)
RR, CC = np.meshgrid(img_rows, img_cols, indexing="ij")

_TONEBURST_DURATION_S = cfg.N_CYCLES / cfg.F0_HZ
_ENVELOPE_GROUP_DELAY_S = _TONEBURST_DURATION_S / 2  # validated fix, run -28


def backproject(pairs):
    accumulator = np.zeros(RR.shape)
    for (tx_name, rx_name), envelope in pairs.items():
        src = _SRC[tx_name]
        rcv = _RCV[rx_name]
        dist_tx = np.sqrt((CC - src[0]) ** 2 + (RR - src[1]) ** 2) * dx[0]
        dist_rx = np.sqrt((CC - rcv[0]) ** 2 + (RR - rcv[1]) ** 2) * dx[0]
        t_total = (dist_tx + dist_rx) / c_ref + _ENVELOPE_GROUP_DELAY_S
        accumulator += np.interp(t_total, t_arr, envelope, left=0, right=0)
    return accumulator


col_idx = np.argmin(np.abs(img_cols - center[1]))
row_idx = np.argmin(np.abs(img_rows - center[0]))


def track_four_directions(accumulator):
    """Peak-search along each probe's boresight ray in the FULL 2D
    accumulator (not a radial bin -- the triangle isn't circularly
    symmetric, so distance-from-center alone doesn't identify 'the'
    boundary point the way it did for the circle)."""
    vert_profile = np.abs(accumulator[:, col_idx])
    horiz_profile = np.abs(accumulator[row_idx, :])

    top_mask = img_rows < center[0]
    bottom_mask = img_rows > center[0]
    left_mask = img_cols < center[1]
    right_mask = img_cols > center[1]

    top_row = img_rows[top_mask][np.argmax(vert_profile[top_mask])]
    bottom_row = img_rows[bottom_mask][np.argmax(vert_profile[bottom_mask])]
    left_col = img_cols[left_mask][np.argmax(horiz_profile[left_mask])]
    right_col = img_cols[right_mask][np.argmax(horiz_profile[right_mask])]

    return {
        "top": center[0] - top_row,
        "bottom": bottom_row - center[0],
        "left": center[1] - left_col,
        "right": right_col - center[1],
    }


if __name__ == "__main__":
    print(f"*** {labels.PENDING_SIGNOFF_BANNER} ***")
    print(f"*** {labels.TOY_EXACT_GT_CAPTION} ***")
    print("Multistatic backprojection on a beating TRIANGLE: same 4-probe, "
          "16-pair accumulator as run -28 (filled disk), testing whether "
          "the method generalizes past circular symmetry -- vertex (top), "
          "normal-incidence edge (bottom), and oblique facets (left/right) "
          "all in one phantom.")

    print("\n=== Control: homogeneous medium (no triangle) ===")
    pairs_ref = capture_all_pairs(build_medium_homogeneous())
    accumulator_ref = backproject(pairs_ref)
    ref_dists = track_four_directions(accumulator_ref)
    print(f"  homogeneous-medium tracked distances (should be small/meaningless): {ref_dists}")

    N_FRAMES_MOVIE = 8
    phases = np.linspace(0, 1, N_FRAMES_MOVIE)
    radii_cells = [p3cfg.lv_radius_at_phase(p) for p in phases]

    frames = []
    all_tracked = []
    all_true = []
    for i, R in enumerate(radii_cells):
        print(f"=== Frame {i+1}/{N_FRAMES_MOVIE} (R={R:.1f} cells = {R*dx[0]*1e3:.2f}mm) ===")
        pairs = capture_all_pairs(build_medium_triangle(R))
        accumulator = backproject(pairs)
        accumulator_clean = accumulator - accumulator_ref
        frames.append(accumulator_clean)
        tracked = track_four_directions(accumulator_clean)
        true_dist = {"top": R, "bottom": 0.5 * R, "left": R / np.sqrt(3), "right": R / np.sqrt(3)}
        all_tracked.append(tracked)
        all_true.append(true_dist)
        for side in ["top", "bottom", "left", "right"]:
            err_mm = abs(tracked[side] - true_dist[side]) * dx[0] * 1e3
            print(f"  {side}: true={true_dist[side]:.1f} cells, tracked={tracked[side]:.1f} cells, error={err_mm:.2f}mm")

    print("\n--- Per-side RMSE across all frames ---")
    for side in ["top", "bottom", "left", "right"]:
        errs = np.array([abs(all_tracked[i][side] - all_true[i][side]) * dx[0] * 1e3
                          for i in range(N_FRAMES_MOVIE)])
        rmse = np.sqrt(np.mean(errs ** 2))
        print(f"  {side}: RMSE={rmse:.4f}mm  (per-frame: {np.round(errs, 2).tolist()})")

    # Filmstrip: full 2D accumulator, true triangle + tracked rays overlaid.
    n_cols = 4
    n_rows = int(np.ceil(N_FRAMES_MOVIE / n_cols))
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(3.2 * n_cols, 3.4 * n_rows))
    axes = np.array(axes).reshape(-1)
    vmax = max(np.abs(f).max() for f in frames)
    for i, (ax, frame, R) in enumerate(zip(axes, frames, radii_cells)):
        ax.imshow(np.abs(frame), cmap="hot", vmin=0, vmax=vmax, origin="upper",
                  extent=[img_cols.min(), img_cols.max(), img_rows.max(), img_rows.min()])
        top, botleft, botright = triangle_vertices(R)
        tri_row = [top[0], botleft[0], botright[0], top[0]]
        tri_col = [top[1], botleft[1], botright[1], top[1]]
        ax.plot(tri_col, tri_row, "c--", linewidth=1, alpha=0.7)
        t = all_tracked[i]
        ax.plot(center[1], center[0] - t["top"], "g+", markersize=10, markeredgewidth=2)
        ax.plot(center[1], center[0] + t["bottom"], "g+", markersize=10, markeredgewidth=2)
        ax.plot(center[1] - t["left"], center[0], "g+", markersize=10, markeredgewidth=2)
        ax.plot(center[1] + t["right"], center[0], "g+", markersize=10, markeredgewidth=2)
        mean_err = np.mean([abs(all_tracked[i][s] - all_true[i][s]) for s in all_true[i]]) * dx[0] * 1e3
        ax.set_title(f"phase={phases[i]:.2f}, R={R*dx[0]*1e3:.2f}mm\nmean err={mean_err:.2f}mm", fontsize=8)
        ax.axis("off")
    for ax in axes[len(frames):]:
        ax.axis("off")
    fig.suptitle("Multistatic backprojection on a beating TRIANGLE\n"
                "cyan dashed = true triangle, green + = tracked boundary per probe axis\n"
                "(TOY: exact prescribed ground truth)", fontsize=10)
    plt.tight_layout(rect=[0, 0.02, 1, 0.88])
    labels.add_banner(fig)
    os.makedirs("results/figures", exist_ok=True)
    plt.savefig("results/figures/phase3_multistatic_backprojection_triangle.png", dpi=130)
    print("\nSaved results/figures/phase3_multistatic_backprojection_triangle.png")
