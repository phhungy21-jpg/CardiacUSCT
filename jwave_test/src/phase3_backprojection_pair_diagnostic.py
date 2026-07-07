"""Phase 3 — diagnostic: which tx/rx pair(s) cause the false "top" peak
in the beating-triangle multistatic backprojection (runs -29/-30)?

Both the 4-probe (run -29) and 8-probe (run -30) triangle tests showed
the SAME failure: the "top" (vertex) search locks onto a value matching
that frame's BOTTOM (opposite-edge) distance, almost exactly, regardless
of probe placement. That specific, reproducible, crisp match to another
feature's exact location -- not a generically elevated/broadened noise
floor -- is the signature of a specific ghost/mirror-path artifact from
one (or a few) tx/rx pairs, not (or not only) generic orientation-blind
clutter (every point treated as an isotropic scatterer, no specular-
compatibility weighting -- a real, separate architectural gap, but a
bigger fix to test first).

This script tests that cheaply: capture all 16 pairs' traces ONCE for a
single frame (ED, R=60 -- the case already measured wrong in both prior
runs) plus the homogeneous reference (needed for cleaning), then -- pure
numpy, no new simulations -- compute each of the 16 pairs' INDIVIDUAL
backprojected contribution (not summed) and read off its value at (a)
the false peak location the aggregate accumulator locked onto, and (b)
the true vertex location. If one or two pairs dominate at the false
location while contributing little at the true vertex, that identifies
the specific ghost-path mechanism (fixable by excluding/down-weighting
that pair). If contributions are spread broadly across many pairs with
no single dominant source, that instead supports the orientation-
blindness explanation, and the fuller (position, normal) joint-search
redesign is the right next step.
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


col_idx = np.argmin(np.abs(img_cols - center[1]))
top_mask = img_rows < center[0]


if __name__ == "__main__":
    print(f"*** {labels.PENDING_SIGNOFF_BANNER} ***")
    print(f"*** {labels.TOY_EXACT_GT_CAPTION} ***")
    print("Diagnostic: single-pair backprojection contributions at the "
          "false 'top' peak vs. the true vertex, for the ED frame (R=60), "
          "to identify whether a specific pair (ghost/mirror artifact) or "
          "broad multi-pair clutter (orientation-blindness) explains runs "
          "-29/-30's vertex-tracking failure.")

    R = p3cfg.LV_RADIUS_ED_CELLS  # 60 cells, the exact frame measured wrong before
    print(f"\n=== Capturing all 16 pairs: R={R} cells (ED) + homogeneous reference ===")
    pairs_tri = capture_all_pairs(build_medium_triangle(R))
    pairs_ref = capture_all_pairs(build_medium_homogeneous())

    # Reproduce the aggregate (summed) accumulator to find the exact false
    # peak location this specific run produces (should match runs -29/-30:
    # top tracked ~31-34 cells, i.e. row ~ center[0]-32).
    accumulator_tri = sum(backproject_single_pair(tx, rx, env) for (tx, rx), env in pairs_tri.items())
    accumulator_ref = sum(backproject_single_pair(tx, rx, env) for (tx, rx), env in pairs_ref.items())
    accumulator_clean = accumulator_tri - accumulator_ref

    vert_profile = np.abs(accumulator_clean[:, col_idx])
    false_peak_row = img_rows[top_mask][np.argmax(vert_profile[top_mask])]
    false_peak_dist = center[0] - false_peak_row
    true_vertex_dist = R
    print(f"  aggregate false peak: row={false_peak_row:.1f} (dist from center={false_peak_dist:.1f} cells)")
    print(f"  true vertex distance: {true_vertex_dist:.1f} cells")

    # Per-pair contribution (cleaned individually) at both locations.
    print(f"\n{'tx->rx':<16}{'val@false_peak':>16}{'val@true_vertex':>18}{'ratio':>10}")
    rows_at_false = np.argmin(np.abs(img_rows - false_peak_row))
    rows_at_true = np.argmin(np.abs(img_rows - (center[0] - true_vertex_dist)))
    results = []
    for (tx, rx), env_tri in pairs_tri.items():
        env_ref = pairs_ref[(tx, rx)]
        single_tri = backproject_single_pair(tx, rx, env_tri)
        single_ref = backproject_single_pair(tx, rx, env_ref)
        single_clean = single_tri - single_ref
        val_false = abs(single_clean[rows_at_false, col_idx])
        val_true = abs(single_clean[rows_at_true, col_idx])
        ratio = val_false / (val_true + 1e-12)
        results.append((f"{tx}->{rx}", val_false, val_true, ratio))

    results.sort(key=lambda r: -r[1])
    for name, val_false, val_true, ratio in results:
        print(f"{name:<16}{val_false:>16.5f}{val_true:>18.5f}{ratio:>10.2f}")

    top3 = results[:3]
    total_false = sum(r[1] for r in results)
    top3_share = sum(r[1] for r in top3) / (total_false + 1e-12)
    print(f"\nTop-3 pairs' share of total false-peak energy: {top3_share*100:.1f}%")
    print("If this is a small number of pairs (e.g. top-3 >> uniform 3/16=18.75%), "
          "that points to a specific ghost-path artifact (fixable by excluding "
          "those pairs). If contributions are roughly even across all 16 pairs, "
          "that instead supports generic orientation-blind clutter (needs the "
          "fuller (position, normal) joint-search redesign).")
