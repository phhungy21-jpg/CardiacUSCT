"""Phase 3 — diagnostic: is patient023's noisy outer-boundary fit a
PROBE-STANDOFF artifact (fixable by moving probes farther away) or the
CURVATURE-WEIGHT CALIBRATION range limit (run -44's model only
calibrated at R=41/71 cells; patient023's outer boundary sits at
~88 cells, beyond that range)?

Per user: "if the heart is too big, why don't expand the grid?" -- a
fair, testable hypothesis, not assumed away. This builds a SEPARATE,
wider probe/domain geometry (NOT modifying the shared
`phase3_backprojection_shape_fit_triangle` module other reused scripts
depend on, to avoid silently changing any already-validated result in
this thread) with PROBE_DIST_CELLS=180 (vs the standard 120) -- giving
patient023's outer boundary (~88 cells) a standoff of ~92 cells,
proportionally MORE generous than patient001's (74 cells outer,
46-cell standoff at the standard 120).

Key physical point being tested: `pair_weight_at_R` (the curvature-
aware weight) is a function of the REFLECTOR's own radius only, not of
probe distance -- the mechanism it encodes (a larger/flatter circle
concentrates reflected energy near the monostatic direction) is a
far-field property of the reflector's curvature. If widening the probe
standoff does NOT fix the noisy outer fit, that is direct evidence the
problem is the calibration's radius RANGE (needs a new calibration
point near R=88), not a near-field/standoff artifact.
"""

import sys

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
from phase3_ring_curvature_weighted_fit import pair_weight_at_R
from phase3_mri_irregular_ring_reconstruction import _polar_resample, r_at_theta, build_search_grid

from matplotlib import pyplot as plt
import os

PATIENT_ID = sys.argv[1] if len(sys.argv) > 1 else "patient023"

c_ref = cfg.CHEST_WALL_PROXY.sound_speed
dx = (cfg.DX_M, cfg.DX_M)

PROBE_DIST_CELLS = 180  # vs. the standard 120 -- wider standoff
N = (460, 460)
center = (230, 230)
domain = Domain(N, dx)

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
        return (col - 5, row), (col + 5, row)
    return (col, row - 5), (col, row + 5)


_SRC = {name: probe_source_and_receiver(p)[0] for name, p in PROBES.items()}
_RCV = {name: probe_source_and_receiver(p)[1] for name, p in PROBES.items()}


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

SEARCH_RADIUS_CELLS = 130  # generous for patient023's ~88-cell outer radius x up to 1.31 scale
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


def direction_vector(theta_deg):
    theta = np.deg2rad(theta_deg)
    return -np.cos(theta), np.sin(theta)


N_ANGLES = 144
_THETAS = np.linspace(0, 360, N_ANGLES, endpoint=False)


def fit_scale_curvature_weighted(pairs, ext_theta, ext_r, scale_grid, origin, img_rows_g, img_cols_g):
    RR, CC = np.meshgrid(img_rows_g, img_cols_g, indexing="ij")
    per_pair_grids = {}
    for (tx, rx), envelope in pairs.items():
        src, rcv = _SRC[tx], _RCV[rx]
        dist_tx = np.sqrt((CC - src[0]) ** 2 + (RR - src[1]) ** 2) * dx[0]
        dist_rx = np.sqrt((CC - rcv[0]) ** 2 + (RR - rcv[1]) ** 2) * dx[0]
        t_total = (dist_tx + dist_rx) / c_ref + _ENVELOPE_GROUP_DELAY_S
        per_pair_grids[(tx, rx)] = np.interp(t_total, t_arr, envelope, left=0, right=0)
    interpolators = {
        key: RegularGridInterpolator((img_rows_g, img_cols_g), np.abs(grid), bounds_error=False, fill_value=0.0)
        for key, grid in per_pair_grids.items()
    }
    scores = np.zeros(len(scale_grid))
    for i, s in enumerate(scale_grid):
        d_vals = r_at_theta(_THETAS, ext_theta, ext_r) * s
        d_rows, d_cols = direction_vector(_THETAS)
        pts = np.stack([origin[0] + d_vals * d_rows, origin[1] + d_vals * d_cols], axis=1)
        total = 0.0
        for (tx, rx), interp in interpolators.items():
            w = pair_weight_at_R(tx, rx, np.mean(d_vals))
            total += w * interp(pts).sum()
        scores[i] = total
    best_idx = np.argmax(scores)
    return scale_grid[best_idx], scores


SCALE_GRID = np.arange(0.7, 1.31, 0.005)
GUARD_BAND_CELLS = 8.0

if __name__ == "__main__":
    print(f"*** {labels.PENDING_SIGNOFF_BANNER} ***")
    print(f"*** {labels.TOY_EXACT_GT_CAPTION} ***")
    print(f"Wide-probe-standoff test for {PATIENT_ID}: PROBE_DIST_CELLS={PROBE_DIST_CELLS} "
          f"(vs standard 120), testing whether patient023's noisy outer fit is a standoff "
          f"artifact or the curvature-weight calibration's radius-range limit.")

    d = np.load(f"results/mri_irregular_ring_{PATIENT_ID}_slice4.npz")
    lv_mask = d["lv_mask"].astype(bool)
    myo_mask = d["myo_mask"].astype(bool)
    ring_mask = d["ring_mask"].astype(bool)
    outer_contour = d["outer_contour"]
    inner_contour = d["inner_contour"]

    ys, xs = np.where(ring_mask)
    ring_centroid_native = (ys.mean(), xs.mean())
    lv_ys, lv_xs = np.where(lv_mask)
    lv_centroid_native = (lv_ys.mean(), lv_xs.mean())

    offset_row = int(round(center[0] - ring_centroid_native[0]))
    offset_col = int(round(center[1] - ring_centroid_native[1]))
    print(f"  placing ring centroid at wide domain center {center} via offset ({offset_row}, {offset_col})")

    rows_native, cols_native = np.mgrid[0:myo_mask.shape[0], 0:myo_mask.shape[1]]
    rows_dom, cols_dom = rows_native + offset_row, cols_native + offset_col
    valid = (rows_dom >= 0) & (rows_dom < N[0]) & (cols_dom >= 0) & (cols_dom < N[1])

    canvas_myo = np.zeros(N, dtype=bool)
    canvas_lv = np.zeros(N, dtype=bool)
    canvas_myo[rows_dom[valid], cols_dom[valid]] = myo_mask[valid]
    canvas_lv[rows_dom[valid], cols_dom[valid]] = lv_mask[valid]
    label_map = np.zeros(N, dtype=int)
    label_map[canvas_myo] = 2
    label_map[canvas_lv] = 3

    lv_centroid_dom = (lv_centroid_native[0] + offset_row, lv_centroid_native[1] + offset_col)
    ring_centroid_dom = (ring_centroid_native[0] + offset_row, ring_centroid_native[1] + offset_col)
    inner_contour_dom = inner_contour + np.array([offset_row, offset_col])
    outer_contour_dom = outer_contour + np.array([offset_row, offset_col])

    ext_theta_in, ext_r_in = _polar_resample(inner_contour_dom, lv_centroid_dom)
    ext_theta_out, ext_r_out = _polar_resample(outer_contour_dom, ring_centroid_dom)
    print(f"  mean inner radius={ext_r_in.mean():.1f} cells, mean outer radius={ext_r_out.mean():.1f} cells, "
          f"standoff to probes={PROBE_DIST_CELLS - ext_r_out.mean():.1f} cells "
          f"(vs. patient001's ~46 cells at the standard 120-cell probe distance)")

    needed_extent = max(ext_r_out.max(), ext_r_in.max()) * SCALE_GRID.max() + 15.0
    n_pts = int(2 * needed_extent * 100 / 180) + 1
    img_rows_g = np.linspace(center[0] - needed_extent, center[0] + needed_extent, n_pts)
    img_cols_g = np.linspace(center[1] - needed_extent, center[1] + needed_extent, n_pts)
    print(f"  search grid: {len(img_rows_g)}x{len(img_cols_g)}, +/-{needed_extent:.0f} cells")

    medium = build_medium_real_contour(label_map)
    print("\n=== Simulating real-contour phantom (wide probes, 16 tx/rx pairs) ===")
    pairs_real = capture_all_pairs(medium)
    print("=== Simulating homogeneous reference ===")
    pairs_ref = capture_all_pairs(build_medium_homogeneous())

    fitted_s_in, scores_in = fit_scale_curvature_weighted(pairs_real, ext_theta_in, ext_r_in, SCALE_GRID, lv_centroid_dom, img_rows_g, img_cols_g)
    fitted_s_in_ref, _ = fit_scale_curvature_weighted(pairs_ref, ext_theta_in, ext_r_in, SCALE_GRID, lv_centroid_dom, img_rows_g, img_cols_g)

    fitted_inner_mean_radius = fitted_s_in * ext_r_in.mean()
    scale_grid_guarded = SCALE_GRID[np.abs(SCALE_GRID * ext_r_out.mean() - fitted_inner_mean_radius) > GUARD_BAND_CELLS]
    fitted_s_out, scores_out = fit_scale_curvature_weighted(pairs_real, ext_theta_out, ext_r_out, scale_grid_guarded, ring_centroid_dom, img_rows_g, img_cols_g)
    fitted_s_out_ref, _ = fit_scale_curvature_weighted(pairs_ref, ext_theta_out, ext_r_out, SCALE_GRID, ring_centroid_dom, img_rows_g, img_cols_g)

    in_err_mm = abs(fitted_s_in - 1.0) * ext_r_in.mean() * dx[0] * 1e3
    out_err_mm = abs(fitted_s_out - 1.0) * ext_r_out.mean() * dx[0] * 1e3
    locked = abs(fitted_s_out * ext_r_out.mean() - fitted_s_in * ext_r_in.mean()) < 3.0

    print(f"\n--- Result (wide-probe-standoff test) ---")
    print(f"  inner: fitted scale={fitted_s_in:.3f} (true=1.000), error={in_err_mm:.2f}mm")
    print(f"  outer: fitted scale={fitted_s_out:.3f} (true=1.000), error={out_err_mm:.2f}mm, locked_to_inner={locked}")
    print(f"  homogeneous control: inner={fitted_s_in_ref:.3f}, outer={fitted_s_out_ref:.3f}")
    print(f"\n  (compare to standard-probe-distance result: outer error was 2.43mm)")

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))
    axes[0].plot(SCALE_GRID, scores_in / scores_in.max())
    axes[0].axvline(1.0, color="k", linestyle="--", label="true=1.0")
    axes[0].axvline(fitted_s_in, color="g", linestyle=":", label=f"fitted={fitted_s_in:.3f}")
    axes[0].set_title("Inner scale-fit (wide probes)")
    axes[0].legend(fontsize=8)
    axes[1].plot(scale_grid_guarded, scores_out / scores_out.max())
    axes[1].axvline(1.0, color="k", linestyle="--", label="true=1.0")
    axes[1].axvline(fitted_s_out, color="g", linestyle=":", label=f"fitted={fitted_s_out:.3f}")
    axes[1].set_title(f"Outer scale-fit (wide probes, standoff={PROBE_DIST_CELLS - ext_r_out.mean():.0f} cells)")
    axes[1].legend(fontsize=8)
    fig.suptitle(f"Wide-probe-standoff test ({PATIENT_ID}): does more standoff fix the noisy outer fit?")
    labels.add_banner(fig)
    plt.tight_layout()
    os.makedirs("results/figures", exist_ok=True)
    out_fig = f"results/figures/phase3_mri_wide_probe_standoff_test_{PATIENT_ID}.png"
    plt.savefig(out_fig, dpi=130)
    print(f"\nSaved {out_fig}")
