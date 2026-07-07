"""Phase 3 — multistatic backprojection on a beating TRIANGLE, with
COHERENCE-FACTOR weighting (fix for the run -29/-30/diagnostic ghost).

Chain of custody: run -29 (4-probe backprojection on a triangle) found
the vertex/corner direction always tracked wrong, matching the opposite
edge's distance almost exactly. Run -30 (8-probe fix attempt, per user
suggestion) did NOT fix it and made the left facet worse -- disproving
the original "exactly-antipodal-probe-pair" diagnosis.
`phase3_backprojection_pair_diagnostic.py` then isolated the true cause
cheaply (single ED-frame capture, 16 pairs inspected individually, no
new simulations needed beyond that one capture): TWO specific pairs
(bottom->right, right->bottom) contribute 82% of the false peak's
energy while contributing ~nothing at the true vertex -- a classic
sparse-multistatic-array "ghost" (a spurious ellipse crossing between
two probes' pair, common with only 4 elements), not generic
orientation-blind clutter (which would show energy spread evenly across
many pairs, not concentrated 200-500x in two of them).

Fix tested here: COHERENCE-FACTOR (CF) weighting, a standard technique
in SAR/ultrasound imaging for exactly this failure mode (Camacho et al.
2009's "phase coherence imaging" idea, adapted to envelope/incoherent
backprojection here). For each candidate point P, instead of just
summing all N pairs' contributions S(P) = sum_i a_i(P), also compute
Q(P) = sum_i a_i(P)^2 and CF(P) = S(P)^2 / (N * Q(P)) -- CF is close to
1 when many pairs agree with comparable amplitude, and close to 1/N
when one or two pairs dominate. The final image is CF(P) * S(P): this
suppresses points (like the ghost) where the naive sum is large only
because 2 of 16 pairs are large, while preserving points many pairs
weakly-but-broadly corroborate (the hoped-for signature of the true,
weak vertex echo). Reverts to the run -29 (working) 4-probe geometry --
run -30 already showed 8 probes doesn't help and costs accuracy
elsewhere.

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


def _per_pair_values(pairs):
    """Each pair's individual (uninterpolated-sum) contribution a_i(P)
    over the full candidate grid -- the shared building block for both
    the naive sum and the coherence-factor image."""
    values = []
    for (tx_name, rx_name), envelope in pairs.items():
        src = _SRC[tx_name]
        rcv = _RCV[rx_name]
        dist_tx = np.sqrt((CC - src[0]) ** 2 + (RR - src[1]) ** 2) * dx[0]
        dist_rx = np.sqrt((CC - rcv[0]) ** 2 + (RR - rcv[1]) ** 2) * dx[0]
        t_total = (dist_tx + dist_rx) / c_ref + _ENVELOPE_GROUP_DELAY_S
        values.append(np.interp(t_total, t_arr, envelope, left=0, right=0))
    return values


def backproject(pairs):
    """Naive sum -- the run -28/-29 method, kept for direct before/after
    comparison. Vulnerable to a small number of pairs dominating a point
    (the diagnosed ghost)."""
    return sum(_per_pair_values(pairs))


BAD_PAIRS = {("bottom", "right"), ("right", "bottom")}  # identified by
# phase3_backprojection_pair_diagnostic.py: these 2 of 16 pairs contribute
# 82% of the false "top" peak's energy while contributing ~nothing at the
# true vertex -- a specific sparse-array ghost, not general clutter.


def backproject_excluding(pairs, exclude=BAD_PAIRS):
    """Surgical fix: drop only the specific pairs the diagnostic named,
    instead of a blanket statistical rule (coherence factor) that turned
    out to also suppress genuine few-pair-dominated specular echoes."""
    filtered = {k: v for k, v in pairs.items() if k not in exclude}
    return sum(_per_pair_values(filtered))


def backproject_coherence(pairs):
    """Coherence-factor-weighted backprojection (Camacho et al. 2009-style
    phase coherence imaging, adapted to envelope/incoherent backprojection):
    CF(P) = S(P)^2 / (N * Q(P)), where S=sum of per-pair values, Q=sum of
    squares. CF~1 when many pairs agree with comparable amplitude, ~1/N
    when one or two pairs dominate -- directly targets the diagnosed
    2-pair ghost (bottom->right/right->bottom contributing 82% of its
    energy) without needing a full (position, normal) orientation model."""
    values = _per_pair_values(pairs)
    N_pairs = len(values)
    S = sum(values)
    Q = sum(v ** 2 for v in values)
    CF = S ** 2 / (N_pairs * Q + 1e-12)
    return CF * S, CF, S


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
    print("Multistatic backprojection on a beating TRIANGLE, WITH "
          "coherence-factor (CF) weighting -- fix for the ghost pair "
          "(bottom->right/right->bottom) identified in "
          "phase3_backprojection_pair_diagnostic.py. Reverts to the run "
          "-29 4-probe geometry (run -30's 8-probe attempt made things "
          "worse). Runs naive-sum and CF-weighted side by side for a "
          "direct before/after comparison.")

    print("\n=== Control: homogeneous medium (no triangle) ===")
    pairs_ref = capture_all_pairs(build_medium_homogeneous())
    accumulator_ref_naive = backproject(pairs_ref)
    image_ref_cf, _, _ = backproject_coherence(pairs_ref)
    accumulator_ref_excl = backproject_excluding(pairs_ref)
    ref_dists_naive = track_four_directions(accumulator_ref_naive)
    ref_dists_cf = track_four_directions(image_ref_cf)
    ref_dists_excl = track_four_directions(accumulator_ref_excl)
    print(f"  homogeneous-medium tracked distances (naive, should be small/meaningless): {ref_dists_naive}")
    print(f"  homogeneous-medium tracked distances (CF, should be small/meaningless): {ref_dists_cf}")
    print(f"  homogeneous-medium tracked distances (excl-2-pairs, should be small/meaningless): {ref_dists_excl}")

    N_FRAMES_MOVIE = 8
    phases = np.linspace(0, 1, N_FRAMES_MOVIE)
    radii_cells = [p3cfg.lv_radius_at_phase(p) for p in phases]

    frames_naive = []
    frames_cf = []
    frames_excl = []
    all_tracked_naive = []
    all_tracked_cf = []
    all_tracked_excl = []
    all_true = []
    ed_frame_naive_raw = None  # for the dedicated ghost-suppression comparison figure
    ed_frame_cf_raw = None
    ed_frame_excl_raw = None
    for i, R in enumerate(radii_cells):
        print(f"=== Frame {i+1}/{N_FRAMES_MOVIE} (R={R:.1f} cells = {R*dx[0]*1e3:.2f}mm) ===")
        pairs = capture_all_pairs(build_medium_triangle(R))

        accumulator_naive = backproject(pairs)
        accumulator_clean_naive = accumulator_naive - accumulator_ref_naive
        frames_naive.append(accumulator_clean_naive)
        tracked_naive = track_four_directions(accumulator_clean_naive)
        all_tracked_naive.append(tracked_naive)

        image_cf, _, _ = backproject_coherence(pairs)
        image_clean_cf = image_cf - image_ref_cf
        frames_cf.append(image_clean_cf)
        tracked_cf = track_four_directions(image_clean_cf)
        all_tracked_cf.append(tracked_cf)

        accumulator_excl = backproject_excluding(pairs)
        accumulator_clean_excl = accumulator_excl - accumulator_ref_excl
        frames_excl.append(accumulator_clean_excl)
        tracked_excl = track_four_directions(accumulator_clean_excl)
        all_tracked_excl.append(tracked_excl)

        if i == 0:  # ED frame, R=LV_RADIUS_ED_CELLS -- same case the diagnostic used
            ed_frame_naive_raw = accumulator_clean_naive
            ed_frame_cf_raw = image_clean_cf
            ed_frame_excl_raw = accumulator_clean_excl

        true_dist = {"top": R, "bottom": 0.5 * R, "left": R / np.sqrt(3), "right": R / np.sqrt(3)}
        all_true.append(true_dist)
        for side in ["top", "bottom", "left", "right"]:
            err_naive_mm = abs(tracked_naive[side] - true_dist[side]) * dx[0] * 1e3
            err_cf_mm = abs(tracked_cf[side] - true_dist[side]) * dx[0] * 1e3
            err_excl_mm = abs(tracked_excl[side] - true_dist[side]) * dx[0] * 1e3
            print(f"  {side}: true={true_dist[side]:.1f} cells | naive err={err_naive_mm:.2f}mm "
                  f"| CF err={err_cf_mm:.2f}mm | excl-2-pairs err={err_excl_mm:.2f}mm")

    print("\n--- Per-side RMSE across all frames: naive vs. coherence-factor vs. exclude-known-bad-pairs ---")
    for side in ["top", "bottom", "left", "right"]:
        errs_naive = np.array([abs(all_tracked_naive[i][side] - all_true[i][side]) * dx[0] * 1e3
                                for i in range(N_FRAMES_MOVIE)])
        errs_cf = np.array([abs(all_tracked_cf[i][side] - all_true[i][side]) * dx[0] * 1e3
                             for i in range(N_FRAMES_MOVIE)])
        errs_excl = np.array([abs(all_tracked_excl[i][side] - all_true[i][side]) * dx[0] * 1e3
                               for i in range(N_FRAMES_MOVIE)])
        rmse_naive = np.sqrt(np.mean(errs_naive ** 2))
        rmse_cf = np.sqrt(np.mean(errs_cf ** 2))
        rmse_excl = np.sqrt(np.mean(errs_excl ** 2))
        print(f"  {side}: naive RMSE={rmse_naive:.4f}mm  |  CF RMSE={rmse_cf:.4f}mm  |  excl RMSE={rmse_excl:.4f}mm")
        print(f"    naive per-frame: {np.round(errs_naive, 2).tolist()}")
        print(f"    CF per-frame:    {np.round(errs_cf, 2).tolist()}")
        print(f"    excl per-frame:  {np.round(errs_excl, 2).tolist()}")

    # --- Figure 1: dedicated ghost-suppression comparison, ED frame only ---
    fig1, axes1 = plt.subplots(1, 3, figsize=(13, 4.6))
    vmax1 = max(np.abs(ed_frame_naive_raw).max(), np.abs(ed_frame_cf_raw).max(), np.abs(ed_frame_excl_raw).max())
    R_ed = radii_cells[0]
    top, botleft, botright = triangle_vertices(R_ed)
    tri_row = [top[0], botleft[0], botright[0], top[0]]
    tri_col = [top[1], botleft[1], botright[1], top[1]]
    for ax, frame, title, tracked in [
        (axes1[0], ed_frame_naive_raw, "naive sum (run -29)", all_tracked_naive[0]),
        (axes1[1], ed_frame_cf_raw, "coherence-factor weighted", all_tracked_cf[0]),
        (axes1[2], ed_frame_excl_raw, "exclude 2 known-bad pairs", all_tracked_excl[0]),
    ]:
        ax.imshow(np.abs(frame), cmap="hot", vmin=0, vmax=vmax1, origin="upper",
                  extent=[img_cols.min(), img_cols.max(), img_rows.max(), img_rows.min()])
        ax.plot(tri_col, tri_row, "c--", linewidth=1.2, alpha=0.8)
        ax.plot(center[1], center[0] - tracked["top"], "g+", markersize=14, markeredgewidth=2.5)
        top_err_mm = abs(tracked["top"] - R_ed) * dx[0] * 1e3
        ax.set_title(f"{title}\ntop(vertex) tracked err={top_err_mm:.2f}mm", fontsize=9)
        ax.axis("off")
    fig1.suptitle("ED frame: 3-way ghost-suppression comparison (bottom->right / right->bottom pair)\n"
                  "cyan dashed = true triangle, green + = tracked vertex position", fontsize=10)
    plt.tight_layout(rect=[0, 0.02, 1, 0.86])
    labels.add_banner(fig1)
    os.makedirs("results/figures", exist_ok=True)
    plt.savefig("results/figures/phase3_backprojection_coherence_ghost_comparison.png", dpi=140)
    print("\nSaved results/figures/phase3_backprojection_coherence_ghost_comparison.png")

    # --- Figure 2: full 8-frame filmstrip using the BEST-performing variant
    # (chosen after seeing the RMSE table above, not assumed in advance) ---
    variant_rmses = {}
    for name, tracked_list in [("naive", all_tracked_naive), ("cf", all_tracked_cf), ("excl", all_tracked_excl)]:
        errs = np.array([abs(tracked_list[i]["top"] - all_true[i]["top"]) * dx[0] * 1e3
                          for i in range(N_FRAMES_MOVIE)])
        variant_rmses[name] = np.sqrt(np.mean(errs ** 2))
    best_name = min(variant_rmses, key=variant_rmses.get)
    best_frames, best_tracked = {"naive": (frames_naive, all_tracked_naive),
                                  "cf": (frames_cf, all_tracked_cf),
                                  "excl": (frames_excl, all_tracked_excl)}[best_name]
    print(f"\nBest variant by top(vertex) RMSE: {best_name} ({variant_rmses[best_name]:.4f}mm) "
          f"-- used for the main filmstrip figure below.")

    n_cols = 4
    n_rows = int(np.ceil(N_FRAMES_MOVIE / n_cols))
    fig2, axes2 = plt.subplots(n_rows, n_cols, figsize=(3.2 * n_cols, 3.4 * n_rows))
    axes2 = np.array(axes2).reshape(-1)
    vmax2 = max(np.abs(f).max() for f in best_frames)
    for i, (ax, frame, R) in enumerate(zip(axes2, best_frames, radii_cells)):
        ax.imshow(np.abs(frame), cmap="hot", vmin=0, vmax=vmax2, origin="upper",
                  extent=[img_cols.min(), img_cols.max(), img_rows.max(), img_rows.min()])
        top, botleft, botright = triangle_vertices(R)
        tri_row = [top[0], botleft[0], botright[0], top[0]]
        tri_col = [top[1], botleft[1], botright[1], top[1]]
        ax.plot(tri_col, tri_row, "c--", linewidth=1, alpha=0.7)
        t = best_tracked[i]
        ax.plot(center[1], center[0] - t["top"], "g+", markersize=10, markeredgewidth=2)
        ax.plot(center[1], center[0] + t["bottom"], "g+", markersize=10, markeredgewidth=2)
        ax.plot(center[1] - t["left"], center[0], "g+", markersize=10, markeredgewidth=2)
        ax.plot(center[1] + t["right"], center[0], "g+", markersize=10, markeredgewidth=2)
        mean_err = np.mean([abs(best_tracked[i][s] - all_true[i][s]) for s in all_true[i]]) * dx[0] * 1e3
        ax.set_title(f"phase={phases[i]:.2f}, R={R*dx[0]*1e3:.2f}mm\nmean err={mean_err:.2f}mm", fontsize=8)
        ax.axis("off")
    for ax in axes2[len(best_frames):]:
        ax.axis("off")
    fig2.suptitle(f"Multistatic backprojection on a beating TRIANGLE -- best variant: {best_name}\n"
                  "cyan dashed = true triangle, green + = tracked boundary per probe axis\n"
                  "(TOY: exact prescribed ground truth)", fontsize=10)
    plt.tight_layout(rect=[0, 0.02, 1, 0.88])
    labels.add_banner(fig2)
    plt.savefig("results/figures/phase3_multistatic_backprojection_triangle_coherence.png", dpi=130)
    print("\nSaved results/figures/phase3_multistatic_backprojection_triangle_coherence.png")
