"""Phase 3 — ISOLATED 16-PROBE geometry (uniform circular spacing,
22.5 degrees), extending the 8-probe pattern (runs -56/-61/-70/-71).

Per user: "try the 16 probe model (4x4), or if you have a
mathematically more favourable observer field (a hex or circular vs
square field)". Answer given directly: for N discrete viewpoints
surrounding a target with the goal of minimizing the maximum angular
gap between adjacent views (the exact metric behind the "ghost cone"
problem, runs -70/-71), UNIFORM SPACING ON A CIRCLE is optimal -- NOT a
grid. Grids (square or hexagonal) are the right tool for tiling a 2D
AREA efficiently (hex packing minimizes sensors needed to cover a
plane), a different problem from surrounding a point target with even
angular coverage. This is also how real ring-array USCT systems are
built, for the same reason. So 16 probes here means 16 probes at
22.5-degree spacing around the SAME circle as the 4/8-probe layouts,
not a 4x4 grid.

Self-contained, per this thread's established discipline: own probe
geometry/domain/capture/weight-model logic, no existing file modified.
The weight model is NOT reused from the 8-probe calibration (run -60)
-- 16 probes introduces NEW baseline separations (22.5/67.5/112.5/
157.5 degrees) never measured, and run -60 already found that
calibration values do not reliably transfer across different probe
geometries even nominally-same categories. Calibration is measured
fresh in `phase3_16probe_calibration.py`.
"""

import numpy as np
from scipy.signal import hilbert, find_peaks
from scipy.interpolate import RegularGridInterpolator

from jax import numpy as jnp
from jax import jit
from jwave import FourierSeries
from jwave.geometry import Domain, Medium, TimeAxis, Sources
from jwave.acoustics import simulate_wave_propagation

import phase2_config as cfg
import phase3_config as p3cfg
import labels

c_ref = cfg.CHEST_WALL_PROXY.sound_speed
dx = (cfg.DX_M, cfg.DX_M)

PROBE_DIST_CELLS = 120  # standoff already proven irrelevant, run -52
N = (300, 300)
center = (150, 150)
domain = Domain(N, dx)

N_PROBES = 16
PROBE_ANGLES_DEG = np.linspace(0, 360, N_PROBES, endpoint=False)  # exact 22.5deg spacing


def direction_vector(theta_deg):
    theta = np.deg2rad(theta_deg)
    return -np.cos(theta), np.sin(theta)


def _probe_src_rcv(theta_deg, offset_cells=5):
    d_row, d_col = direction_vector(theta_deg)
    probe_row = center[0] + PROBE_DIST_CELLS * d_row
    probe_col = center[1] + PROBE_DIST_CELLS * d_col
    t_row, t_col = d_col, -d_row  # tangential, verified against the 4/8-probe convention
    src = (round(probe_col - offset_cells * t_col), round(probe_row - offset_cells * t_row))
    rcv = (round(probe_col + offset_cells * t_col), round(probe_row + offset_cells * t_row))
    return src, rcv


PROBE_NAMES = [f"p{i}" for i in range(N_PROBES)]
_ANGLE_OF = dict(zip(PROBE_NAMES, PROBE_ANGLES_DEG))
_SRC, _RCV = {}, {}
for name, ang in zip(PROBE_NAMES, PROBE_ANGLES_DEG):
    s, r = _probe_src_rcv(ang)
    _SRC[name] = s
    _RCV[name] = r


def build_medium_real_contour(label_map):
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


def build_medium_homogeneous():
    sound_speed_map = np.full(N, cfg.CHEST_WALL_PROXY.sound_speed, dtype=np.float32)
    density_map = np.full(N, cfg.CHEST_WALL_PROXY.density, dtype=np.float32)
    ssm = jnp.expand_dims(jnp.array(sound_speed_map), -1)
    dm = jnp.expand_dims(jnp.array(density_map), -1)
    return Medium(domain=domain, sound_speed=FourierSeries(ssm, domain),
                  density=FourierSeries(dm, domain))


_dummy_medium = build_medium_homogeneous()
_base_time_axis = TimeAxis.from_medium(_dummy_medium, cfl=cfg.CFL)
dt = _base_time_axis.dt

SEARCH_RADIUS_CELLS = 100
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
_TONEBURST_DURATION_S = cfg.N_CYCLES / cfg.F0_HZ
_ENVELOPE_GROUP_DELAY_S = _TONEBURST_DURATION_S / 2


def simulate_probe_transmit(tx_name):
    src = _SRC[tx_name]
    sources = Sources(positions=([src[0]], [src[1]]), signals=_signal_template, dt=dt, domain=domain)

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
            pairs[(tx_name, rx_name)] = np.abs(hilbert(trace))
    return pairs


def angular_separation(theta_a, theta_b):
    d = abs(theta_a - theta_b) % 360
    return min(d, 360 - d)


# Populated by phase3_16probe_calibration.py's measurement -- placeholder
# (all zero except monostatic=1.0) until calibration is run; importing
# scripts MUST call `set_calibration()` before trusting any fit.
_CAL_R = np.array([41.0, 71.0, 88.0])
_CAL_BY_SEP = {}  # sep_deg -> np.array of 3 values, filled by set_calibration()


def set_calibration(cal_by_sep):
    global _CAL_BY_SEP
    _CAL_BY_SEP = cal_by_sep


def _linear_weight(R, cal_r, cal_w):
    return float(np.clip(np.interp(R, cal_r, cal_w), 0.0, 1.0))


def pair_weight_at_R(tx_name, rx_name, R):
    sep = angular_separation(_ANGLE_OF[tx_name], _ANGLE_OF[rx_name])
    if sep < 1e-6:
        return 1.0
    if not _CAL_BY_SEP:
        raise RuntimeError("16-probe calibration not set -- run phase3_16probe_calibration.py first "
                            "and call set_calibration() before fitting.")
    # nearest calibrated separation (all 8 non-monostatic categories are
    # measured directly by phase3_16probe_calibration.py, so this should
    # always be an exact match, not an interpolation).
    nearest_sep = min(_CAL_BY_SEP.keys(), key=lambda s: abs(s - sep))
    return _linear_weight(R, _CAL_R, _CAL_BY_SEP[nearest_sep])


def select_best_local_peak(scale_grid, scores, step_tol=1.5):
    diffs = np.diff(scale_grid)
    step = np.median(diffs)
    gap_idx = np.where(diffs > step * step_tol)[0]
    boundaries = [0] + [i + 1 for i in gap_idx] + [len(scale_grid)]
    all_peak_positions, all_peak_scores = [], []
    for lo, hi in zip(boundaries[:-1], boundaries[1:]):
        seg_scores = scores[lo:hi]
        peak_idx, _ = find_peaks(seg_scores)
        for p in peak_idx:
            all_peak_positions.append(lo + p)
            all_peak_scores.append(seg_scores[p])
    if not all_peak_positions:
        best_idx = int(np.argmax(scores))
        return scale_grid[best_idx], False, 0.0
    order = np.argsort(all_peak_scores)[::-1]
    best_pos = all_peak_positions[order[0]]
    confidence = all_peak_scores[order[0]] / all_peak_scores[order[1]] if len(order) > 1 else np.inf
    return scale_grid[best_pos], True, confidence


def build_search_grid(max_extent_cells, margin=15.0, density_per_cell=100.0 / 180.0):
    n_pts = max(100, int(round(2 * max_extent_cells * density_per_cell)) + 1)
    rows = np.linspace(center[0] - max_extent_cells, center[0] + max_extent_cells, n_pts)
    cols = np.linspace(center[1] - max_extent_cells, center[1] + max_extent_cells, n_pts)
    return rows, cols
