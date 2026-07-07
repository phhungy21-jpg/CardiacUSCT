"""Phase 3 — diagnostic: which tx/rx pair(s) cause LEFT's false peak
(near d=59, matching TOP's true distance) in the beating-triangle
multistatic backprojection?

Follow-on to phase3_backprojection_pair_diagnostic.py (which found
`bottom->right`/`right->bottom` causing 82% of the false TOP peak's
energy) and phase3_shape_fit_bias_diagnostic.py (run -33), which found
that a finer, interpolated free-distance search along the LEFT ray
finds its peak at d=59.25 -- almost exactly TOP's TRUE distance (60.0)
-- rather than LEFT's own true facet distance (34.64), while the coarse
native-grid search happened to land on the correct answer because the
two candidate peaks are within 1% of each other in height (a near-tied
coincidence, not robust dominance).

Per user pushback on the "timing bias" explanation for run -32's
constant undershoot ("the predicted triangle is always smaller...
regardless of systole or diastole"): a uniform additive timing/group-
delay miscalibration would apply identically to every direction, but
run -29's cardinal-axis local searches (bottom, right) were already
shown accurate to <0.2mm using the ORIGINAL coarse-grid method -- if
LEFT's own true-facet peak and a same-height competing "TOP-mimicking"
peak are BOTH present, and other non-cardinal angles show similar
cross-direction contamination, that points to the SAME kind of specific
ghost-pair mechanism already confirmed for TOP, not a generic timing
offset. If confirmed, MULTIPLE directions being quietly pulled toward
smaller apparent distances by their own ghost pairs would explain why
the GLOBAL shape-fit (which integrates all 72 angles) undershoots
consistently regardless of R/cardiac phase -- a geometric multi-ghost
effect, not a timing artifact.

This script tests that cheaply: capture all 16 pairs' traces ONCE for
the ED frame (R=60) plus the homogeneous reference, then -- pure numpy,
no new simulations -- compute each of the 16 pairs' INDIVIDUAL
backprojected contribution at (a) LEFT's false peak (d=59.25) and (b)
LEFT's true facet location (d=34.64), the same methodology that found
the original vertex ghost.
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

_SQRT3_2 = 0.8660254037844386


def triangle_vertices(R):
    top = (center[0] - R, center[1])
    botleft = (center[0] + 0.5 * R, center[1] - _SQRT3_2 * R)
    botright = (center[0] + 0.5 * R, center[1] + _SQRT3_2 * R)
    return top, botleft, botright


def build_medium_triangle(R):
    top, botleft, botright = triangle_vertices(R)
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
_ENVELOPE_GROUP_DELAY_S = _TONEBURST_DURATION_S / 2


def backproject_single_pair(tx_name, rx_name, envelope):
    src = _SRC[tx_name]
    rcv = _RCV[rx_name]
    dist_tx = np.sqrt((CC - src[0]) ** 2 + (RR - src[1]) ** 2) * dx[0]
    dist_rx = np.sqrt((CC - rcv[0]) ** 2 + (RR - rcv[1]) ** 2) * dx[0]
    t_total = (dist_tx + dist_rx) / c_ref + _ENVELOPE_GROUP_DELAY_S
    return np.interp(t_total, t_arr, envelope, left=0, right=0)


row_idx = np.argmin(np.abs(img_rows - center[0]))
left_mask = img_cols < center[1]

_SQRT3 = np.sqrt(3.0)


if __name__ == "__main__":
    print(f"*** {labels.PENDING_SIGNOFF_BANNER} ***")
    print(f"*** {labels.TOY_EXACT_GT_CAPTION} ***")
    print("Diagnostic: single-pair backprojection contributions at LEFT's "
          "false peak (d~59, matching TOP's true distance) vs. LEFT's own "
          "true facet location (d=R/sqrt(3)), for the ED frame (R=60) -- "
          "testing whether the SAME kind of specific ghost-pair mechanism "
          "already confirmed for TOP also explains LEFT's near-tied "
          "competing peaks (run -33), as an alternative to a generic "
          "timing-bias explanation for run -32's constant undershoot.")

    R = p3cfg.LV_RADIUS_ED_CELLS  # 60 cells
    print(f"\n=== Capturing all 16 pairs: R={R} cells (ED) + homogeneous reference ===")
    pairs_tri = capture_all_pairs(build_medium_triangle(R))
    pairs_ref = capture_all_pairs(build_medium_homogeneous())

    accumulator_tri = sum(backproject_single_pair(tx, rx, env) for (tx, rx), env in pairs_tri.items())
    accumulator_ref = sum(backproject_single_pair(tx, rx, env) for (tx, rx), env in pairs_ref.items())
    accumulator_clean = accumulator_tri - accumulator_ref

    horiz_profile = np.abs(accumulator_clean[row_idx, :])
    false_peak_col = img_cols[left_mask][np.argmax(horiz_profile[left_mask])]
    false_peak_dist = center[1] - false_peak_col
    true_left_dist = R / _SQRT3
    print(f"  aggregate LEFT peak (coarse grid, matches run -29/-33): "
          f"col={false_peak_col:.1f} (dist from center={false_peak_dist:.1f} cells)")
    print(f"  LEFT's true facet distance: {true_left_dist:.1f} cells")
    print(f"  (using run -33's finer free-search false-peak location, d=59.25, as the second target below)")

    # Per-pair contribution (cleaned individually) at both candidate locations.
    print(f"\n{'tx->rx':<16}{'val@d~59(false)':>16}{'val@d=34.6(true)':>18}{'ratio':>10}")
    cols_at_false = np.argmin(np.abs(img_cols - (center[1] - 59.25)))
    cols_at_true = np.argmin(np.abs(img_cols - (center[1] - true_left_dist)))
    results = []
    for (tx, rx), env_tri in pairs_tri.items():
        env_ref = pairs_ref[(tx, rx)]
        single_tri = backproject_single_pair(tx, rx, env_tri)
        single_ref = backproject_single_pair(tx, rx, env_ref)
        single_clean = single_tri - single_ref
        val_false = abs(single_clean[row_idx, cols_at_false])
        val_true = abs(single_clean[row_idx, cols_at_true])
        ratio = val_false / (val_true + 1e-12)
        results.append((f"{tx}->{rx}", val_false, val_true, ratio))

    results.sort(key=lambda r: -r[1])
    for name, val_false, val_true, ratio in results:
        print(f"{name:<16}{val_false:>16.5f}{val_true:>18.5f}{ratio:>10.2f}")

    top3 = results[:3]
    total_false = sum(r[1] for r in results)
    top3_share = sum(r[1] for r in top3) / (total_false + 1e-12)
    print(f"\nTop-3 pairs' share of energy at LEFT's false (d~59) location: {top3_share*100:.1f}%")
    print("If this is a small number of pairs (uniform expectation 3/16=18.75%), "
          "that confirms a SPECIFIC ghost-pair mechanism for LEFT too (like TOP's "
          "bottom->right/right->bottom), supporting a multi-ghost geometric "
          "explanation for the global fit's undershoot over a generic timing bias.")
