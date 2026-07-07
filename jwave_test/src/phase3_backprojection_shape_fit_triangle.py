"""Phase 3 — multistatic backprojection on a beating TRIANGLE, with a
GLOBAL SHAPE-FITTING readout (generalized-Hough-style template match)
instead of independent fixed-axis peak-picking.

Chain of custody: run -29 found the vertex direction tracked wrong
(matched the opposite edge's distance) via `track_four_directions`,
which picks one peak per fixed cardinal axis, independently. The
pair-ablation diagnostic (`phase3_backprojection_pair_diagnostic.py`)
found the cause: 2 of 16 pairs (bottom->right, right->bottom) create a
sparse-array ghost. Excluding those 2 pairs globally (run -31) fixed the
vertex but REGRESSED the left facet (0.95mm -> 2.11mm) -- because those
same pairs also carried real information for the left facet, and a
single global pair-exclusion rule can't be right for every sector.
User's correction: the deeper fix isn't another hand-tuned pair mask,
it's replacing independent per-axis peak-picking with GLOBAL SHAPE
FITTING, so a single bad local region becomes a visible outlier to a
robust fit instead of a silently-trusted final number -- and this is
the piece that's actually needed before the heart-wall phantom, which
has no natural set of 4 fixed cardinal rays at all.

This script tests that directly: since the triangle's shape family is
known exactly (equilateral, fixed center/orientation, one free
parameter R), do a GLOBAL template match -- sweep candidate R values,
and for each, sample the (uncorrected, NAIVE, no pair-exclusion)
accumulator along its ENTIRE predicted boundary across many angles (not
just 4), summing the energy. Pick the R that maximizes total boundary
energy. This integrates evidence from the whole shape at once, so a
localized ghost near one sector can't dominate the fit the way it
dominated a single local peak search -- the same principle (robust,
global aggregation over many independent observations) that will need
to generalize to fitting an unknown, non-polygonal boundary on the
heart-wall phantom later (there via a more generic curve fit; here via
a 1-parameter template match, deliberately kept simple since the shape
family is known).

Ground-truth distance from center to boundary along each cardinal axis
(same triangle geometry as run -29): top (vertex) = R; bottom (opposite
edge) = R/2; left/right (oblique facets) = R/sqrt(3).
"""

import numpy as np
from scipy.signal import hilbert
from scipy.interpolate import RegularGridInterpolator

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


def backproject_tgc(pairs):
    """TGC (time/depth-gain-compensation) fix for the range-dependent
    energy bias diagnosed in phase3_shape_fit_bias_diagnostic.py: a
    candidate point closer to the array gets systematically higher raw
    backprojected amplitude regardless of whether a true reflector is
    there (2D cylindrical wave spreading -> amplitude ~1/sqrt(r) per
    leg), which pulled the global shape fit toward under-sized R (run
    -32: 0.80mm undershoot with all 72 angles, still 0.50mm with only
    the 4 validated cardinal angles). Compensate by multiplying each
    pair's contribution by sqrt(dist_tx * dist_rx) -- the inverse of the
    round-trip 1/sqrt(dist_tx*dist_rx) geometric-spreading falloff --
    before summing, same standard fix real ultrasound systems use."""
    accumulator = np.zeros(RR.shape)
    for (tx_name, rx_name), envelope in pairs.items():
        src = _SRC[tx_name]
        rcv = _RCV[rx_name]
        dist_tx = np.sqrt((CC - src[0]) ** 2 + (RR - src[1]) ** 2) * dx[0]
        dist_rx = np.sqrt((CC - rcv[0]) ** 2 + (RR - rcv[1]) ** 2) * dx[0]
        t_total = (dist_tx + dist_rx) / c_ref + _ENVELOPE_GROUP_DELAY_S
        gain = np.sqrt(dist_tx * dist_rx)
        accumulator += gain * np.interp(t_total, t_arr, envelope, left=0, right=0)
    return accumulator


# Confirmed by phase3_backprojection_pair_diagnostic.py (TOP's ghost:
# bottom->right/right->bottom) and phase3_left_ghost_diagnostic.py
# (LEFT's ghost: left->top/bottom->left/left->bottom, 65.9% of false-
# peak energy vs 18.75% uniform expectation): ADJACENT-probe (90-degree
# separated) cross pairs create ghosts pointing toward a NEIGHBORING
# direction's true location, while monostatic (same-probe) and
# ANTIPODAL (180-degree) pairs carry genuine signal. This is a
# structural property of the 4-probe layout recurring at multiple
# sectors, not a per-target coincidence -- so exclude ALL adjacent
# cross pairs (not just the 2 originally hand-diagnosed for the
# vertex), keeping only the 4 monostatic + 4 antipodal pairs (8 of 16).
ADJACENT_GHOST_PAIRS = {
    ("top", "left"), ("left", "top"), ("top", "right"), ("right", "top"),
    ("bottom", "left"), ("left", "bottom"), ("bottom", "right"), ("right", "bottom"),
}


def backproject_no_adjacent(pairs):
    """Principled generalization of run -31's hand-picked 2-pair
    exclusion: drop ALL adjacent-probe cross pairs, keep monostatic +
    antipodal pairs only."""
    filtered = {k: v for k, v in pairs.items() if k not in ADJACENT_GHOST_PAIRS}
    accumulator = np.zeros(RR.shape)
    for (tx_name, rx_name), envelope in filtered.items():
        src = _SRC[tx_name]
        rcv = _RCV[rx_name]
        dist_tx = np.sqrt((CC - src[0]) ** 2 + (RR - src[1]) ** 2) * dx[0]
        dist_rx = np.sqrt((CC - rcv[0]) ** 2 + (RR - rcv[1]) ** 2) * dx[0]
        t_total = (dist_tx + dist_rx) / c_ref + _ENVELOPE_GROUP_DELAY_S
        accumulator += np.interp(t_total, t_arr, envelope, left=0, right=0)
    return accumulator


def direction_vector(theta_deg):
    """theta=0 -> straight up (top/vertex direction), increasing
    clockwise: 90=right, 180=down(bottom), 270=left. Matches the
    existing top/bottom/left/right axis convention exactly."""
    theta = np.deg2rad(theta_deg)
    return -np.cos(theta), np.sin(theta)  # (d_row, d_col)


def _ray_segment_intersection(d_row, d_col, p1, p2):
    """Distance t>0 along ray (center + t*(d_row,d_col)) to its
    crossing of segment p1->p2, or None if no valid crossing.
    Solves center + t*d = p1 + s*(p2-p1) for (t,s), keeps s in [0,1]."""
    ax, ay = center[0], center[1]
    bx, by = p1[0], p1[1]
    ex, ey = p2[0] - p1[0], p2[1] - p1[1]
    denom = d_row * ey - d_col * ex
    if abs(denom) < 1e-9:
        return None
    t = ((bx - ax) * ey - (by - ay) * ex) / denom
    s = ((bx - ax) * d_col - (by - ay) * d_row) / denom
    if t > 0 and 0 <= s <= 1:
        return t
    return None


def ray_triangle_distance(theta_deg, R):
    """Analytic distance from center to the triangle boundary along
    direction theta_deg, generalizing the 4 hand-derived axis formulas
    (top=R, bottom=R/2, left=right=R/sqrt(3)) to ANY angle -- the
    building block for the global template match below."""
    d_row, d_col = direction_vector(theta_deg)
    top, botleft, botright = triangle_vertices(R)
    for p1, p2 in [(top, botleft), (botleft, botright), (botright, top)]:
        t = _ray_segment_intersection(d_row, d_col, p1, p2)
        if t is not None:
            return t
    raise ValueError(f"no valid intersection at theta={theta_deg}, R={R}")


# Sanity check against the hand-derived formulas before trusting the
# general version (established practice this session: verify before use).
for _R_check in (40.0, 50.0, 60.0):
    assert abs(ray_triangle_distance(0, _R_check) - _R_check) < 1e-6
    assert abs(ray_triangle_distance(180, _R_check) - 0.5 * _R_check) < 1e-6
    assert abs(ray_triangle_distance(90, _R_check) - _R_check / np.sqrt(3)) < 1e-6
    assert abs(ray_triangle_distance(270, _R_check) - _R_check / np.sqrt(3)) < 1e-6

N_ANGLES = 72  # every 5 degrees
_THETAS = np.linspace(0, 360, N_ANGLES, endpoint=False)


def fit_triangle_radius(accumulator, R_grid):
    """Generalized-Hough-style global template match: for each candidate
    R, sample |accumulator| along its ENTIRE predicted boundary (all 72
    angles) and sum -- the reflector's true R should show high energy
    almost everywhere around the boundary, so a single bad/ghost-
    corrupted sector can't dominate the way it dominates one local
    axis-peak search. Returns (best_R, scores array, per-angle profile
    at best_R for diagnostics)."""
    interp = RegularGridInterpolator((img_rows, img_cols), np.abs(accumulator),
                                      bounds_error=False, fill_value=0.0)
    scores = np.zeros(len(R_grid))
    for i, R in enumerate(R_grid):
        pts = []
        for th in _THETAS:
            d = ray_triangle_distance(th, R)
            d_row, d_col = direction_vector(th)
            pts.append((center[0] + d * d_row, center[1] + d * d_col))
        vals = interp(np.array(pts))
        scores[i] = vals.sum()
    best_R = R_grid[np.argmax(scores)]
    per_angle_at_best = interp(np.array([
        (center[0] + ray_triangle_distance(th, best_R) * direction_vector(th)[0],
         center[1] + ray_triangle_distance(th, best_R) * direction_vector(th)[1])
        for th in _THETAS
    ]))
    return best_R, scores, per_angle_at_best


def dists_from_fit(R):
    """Per-axis distances implied by a fitted R, for apples-to-apples
    comparison with runs -29/-31's per-side table."""
    return {
        "top": ray_triangle_distance(0, R),
        "bottom": ray_triangle_distance(180, R),
        "left": ray_triangle_distance(270, R),
        "right": ray_triangle_distance(90, R),
    }


R_GRID = np.arange(25.0, 75.0, 0.25)  # candidate R sweep, covers ED..ES..ED with margin


if __name__ == "__main__":
    print(f"*** {labels.PENDING_SIGNOFF_BANNER} ***")
    print(f"*** {labels.TOY_EXACT_GT_CAPTION} ***")
    print("Multistatic backprojection on a beating TRIANGLE, GLOBAL SHAPE-FIT "
          "readout, testing the MULTI-GHOST fix for run -32's constant "
          "undershoot: phase3_backprojection_pair_diagnostic.py (TOP's "
          "ghost: bottom->right/right->bottom) and "
          "phase3_left_ghost_diagnostic.py (LEFT's ghost: "
          "left->top/bottom->left/left->bottom) both found ADJACENT-probe "
          "(90-degree) cross pairs creating false peaks at neighboring "
          "directions' true locations -- a structural, geometric effect "
          "(NOT the TGC/amplitude hypothesis, tested and rejected in run "
          "-33). Fix: exclude ALL 8 adjacent cross pairs, keep only "
          "monostatic + antipodal pairs. Naive (run -32 baseline) vs "
          "no-adjacent-pairs, side by side.")

    print("\n=== Control: homogeneous medium (no triangle) ===")
    pairs_ref = capture_all_pairs(build_medium_homogeneous())
    accumulator_ref = backproject(pairs_ref)
    accumulator_ref_noadj = backproject_no_adjacent(pairs_ref)
    ref_R, _, _ = fit_triangle_radius(accumulator_ref, R_GRID)
    ref_R_noadj, _, _ = fit_triangle_radius(accumulator_ref_noadj, R_GRID)
    print(f"  homogeneous-medium fitted R: naive={ref_R:.1f}, no-adjacent={ref_R_noadj:.1f} cells "
          f"(both should be meaningless/low-confidence)")

    N_FRAMES_MOVIE = 8
    phases = np.linspace(0, 1, N_FRAMES_MOVIE)
    radii_cells = [p3cfg.lv_radius_at_phase(p) for p in phases]

    frames_noadj = []
    all_fitted_R = []
    all_fitted_R_noadj = []
    all_fitted_dists_noadj = []
    all_true = []
    ed_scores = None
    ed_scores_noadj = None
    for i, R in enumerate(radii_cells):
        print(f"=== Frame {i+1}/{N_FRAMES_MOVIE} (R={R:.1f} cells = {R*dx[0]*1e3:.2f}mm) ===")
        pairs = capture_all_pairs(build_medium_triangle(R))

        accumulator_clean = backproject(pairs) - accumulator_ref
        fitted_R, scores, _ = fit_triangle_radius(accumulator_clean, R_GRID)
        all_fitted_R.append(fitted_R)

        accumulator_clean_noadj = backproject_no_adjacent(pairs) - accumulator_ref_noadj
        frames_noadj.append(accumulator_clean_noadj)
        fitted_R_noadj, scores_noadj, _ = fit_triangle_radius(accumulator_clean_noadj, R_GRID)
        all_fitted_R_noadj.append(fitted_R_noadj)

        if i == 0:
            ed_scores = scores
            ed_scores_noadj = scores_noadj

        fitted_dists_noadj = dists_from_fit(fitted_R_noadj)
        true_dist = {"top": R, "bottom": 0.5 * R, "left": R / np.sqrt(3), "right": R / np.sqrt(3)}
        all_fitted_dists_noadj.append(fitted_dists_noadj)
        all_true.append(true_dist)

        R_err_mm = abs(fitted_R - R) * dx[0] * 1e3
        R_err_noadj_mm = abs(fitted_R_noadj - R) * dx[0] * 1e3
        print(f"  true R={R:.1f} | naive fitted R={fitted_R:.1f} err={R_err_mm:.2f}mm "
              f"| no-adjacent fitted R={fitted_R_noadj:.1f} err={R_err_noadj_mm:.2f}mm")
        for side in ["top", "bottom", "left", "right"]:
            err_mm = abs(fitted_dists_noadj[side] - true_dist[side]) * dx[0] * 1e3
            print(f"    {side} (no-adjacent fit): true={true_dist[side]:.1f}, fitted={fitted_dists_noadj[side]:.1f}, error={err_mm:.2f}mm")

    print("\n--- Global R-fit RMSE across all frames: naive vs no-adjacent-pairs ---")
    R_errs = np.array([abs(all_fitted_R[i] - radii_cells[i]) * dx[0] * 1e3 for i in range(N_FRAMES_MOVIE)])
    R_errs_noadj = np.array([abs(all_fitted_R_noadj[i] - radii_cells[i]) * dx[0] * 1e3 for i in range(N_FRAMES_MOVIE)])
    print(f"  naive       R RMSE={np.sqrt(np.mean(R_errs ** 2)):.4f}mm  (per-frame: {np.round(R_errs, 2).tolist()})")
    print(f"  no-adjacent R RMSE={np.sqrt(np.mean(R_errs_noadj ** 2)):.4f}mm  (per-frame: {np.round(R_errs_noadj, 2).tolist()})")
    print("\n--- Per-side RMSE with no-adjacent fit (compare to run -29 naive / -31 excl / -32 shape-fit) ---")
    for side in ["top", "bottom", "left", "right"]:
        errs = np.array([abs(all_fitted_dists_noadj[i][side] - all_true[i][side]) * dx[0] * 1e3
                          for i in range(N_FRAMES_MOVIE)])
        rmse = np.sqrt(np.mean(errs ** 2))
        print(f"  {side}: no-adjacent RMSE={rmse:.4f}mm  (per-frame: {np.round(errs, 2).tolist()})")

    # --- Figure 1: score(R) vs R for the ED frame, naive vs no-adjacent ---
    fig1, ax1 = plt.subplots(figsize=(7, 4.5))
    ax1.plot(R_GRID, ed_scores / ed_scores.max(), label="naive, all 16 pairs (run -32)")
    ax1.plot(R_GRID, ed_scores_noadj / ed_scores_noadj.max(), label="no adjacent pairs (this run)")
    ax1.axvline(radii_cells[0], color="k", linestyle="--", label=f"true R={radii_cells[0]:.1f}")
    ax1.axvline(all_fitted_R[0], color="C0", linestyle=":", label=f"naive fit={all_fitted_R[0]:.1f}")
    ax1.axvline(all_fitted_R_noadj[0], color="C1", linestyle=":", label=f"no-adjacent fit={all_fitted_R_noadj[0]:.1f}")
    ax1.set_xlabel("candidate R (cells)")
    ax1.set_ylabel("normalized total boundary energy")
    ax1.set_title("ED frame: global template-match score vs. candidate R, multi-ghost fix")
    ax1.legend(fontsize=8)
    labels.add_banner(fig1)
    plt.tight_layout()
    os.makedirs("results/figures", exist_ok=True)
    plt.savefig("results/figures/phase3_backprojection_shape_fit_score_curve.png", dpi=130)
    print("\nSaved results/figures/phase3_backprojection_shape_fit_score_curve.png")

    # --- Figure 2: full 8-frame filmstrip (no-adjacent), true + fitted triangle overlaid ---
    n_cols = 4
    n_rows = int(np.ceil(N_FRAMES_MOVIE / n_cols))
    fig2, axes2 = plt.subplots(n_rows, n_cols, figsize=(3.2 * n_cols, 3.4 * n_rows))
    axes2 = np.array(axes2).reshape(-1)
    vmax = max(np.abs(f).max() for f in frames_noadj)
    for i, (ax, frame, R) in enumerate(zip(axes2, frames_noadj, radii_cells)):
        ax.imshow(np.abs(frame), cmap="hot", vmin=0, vmax=vmax, origin="upper",
                  extent=[img_cols.min(), img_cols.max(), img_rows.max(), img_rows.min()])
        top, botleft, botright = triangle_vertices(R)
        tri_row = [top[0], botleft[0], botright[0], top[0]]
        tri_col = [top[1], botleft[1], botright[1], top[1]]
        ax.plot(tri_col, tri_row, "c--", linewidth=1, alpha=0.7, label="true")

        fitted_R = all_fitted_R_noadj[i]
        ftop, fbotleft, fbotright = triangle_vertices(fitted_R)
        f_row = [ftop[0], fbotleft[0], fbotright[0], ftop[0]]
        f_col = [ftop[1], fbotleft[1], fbotright[1], ftop[1]]
        ax.plot(f_col, f_row, "g:", linewidth=1.5, alpha=0.9, label="fitted")

        R_err_mm = abs(fitted_R - R) * dx[0] * 1e3
        ax.set_title(f"phase={phases[i]:.2f}, R_true={R*dx[0]*1e3:.2f}mm\nR err={R_err_mm:.2f}mm", fontsize=8)
        ax.axis("off")
    for ax in axes2[len(frames_noadj):]:
        ax.axis("off")
    fig2.suptitle("Multistatic backprojection on a beating TRIANGLE -- GLOBAL SHAPE-FIT, multi-ghost fix\n"
                "cyan dashed = true triangle, green dotted = globally-fitted triangle (1-param R sweep)\n"
                "(TOY: exact prescribed ground truth; adjacent-probe cross pairs excluded)", fontsize=10)
    plt.tight_layout(rect=[0, 0.02, 1, 0.86])
    labels.add_banner(fig2)
    plt.savefig("results/figures/phase3_backprojection_shape_fit_triangle.png", dpi=130)
    print("\nSaved results/figures/phase3_backprojection_shape_fit_triangle.png")
