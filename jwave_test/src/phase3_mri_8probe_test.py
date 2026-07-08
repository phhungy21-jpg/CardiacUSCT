"""Phase 3 — ISOLATED test: does adding more probe angles (8 instead of
4) fix patient023's outer-boundary structural limitation (runs
-51/-52/-53/-54)?

Per user: "so an 8-probe parallel test? make sure you isolate the case,
clone codes before editing and leave the current code base intact."

This script is FULLY SELF-CONTAINED: it defines its own probe geometry,
domain, medium-building, capture, and curvature-weight logic, all
cloned/adapted from the existing validated 4-probe infrastructure
(`phase3_backprojection_shape_fit_triangle.py`,
`phase3_ring_curvature_weighted_fit.py`) rather than importing or
modifying them. It DOES import a few pure, unmodified helper functions
(`_polar_resample`, `r_at_theta`, `build_search_grid` from
`phase3_mri_irregular_ring_reconstruction.py`) since those are generic
math utilities with no probe-count dependency -- reusing them changes
nothing about any existing script's behavior. No existing file is
edited by this script.

RATIONALE (the actual hypothesis under test): runs -51-54 confirmed
(not assumed) that patient023's outer boundary (R=88 cells, large/flat)
returns near-zero reflected energy to all but the 4 MONOSTATIC pairs --
this is a real physical property of the reflector's curvature, not a
calibration or standoff bug. With only 4 probes, that means only 4
independent votes exist for the outer boundary. Adding 4 more probes
(8 total, at 45-degree spacing instead of 90) directly doubles the
number of monostatic-type votes (8 instead of 4) and adds new
intermediate baseline angles (45, 135 degrees) whose reliability is
unmeasured but can be reasonably interpolated between the existing
0/90/180-degree calibration points. If the outer boundary's accuracy
improves meaningfully, that confirms probe COUNT (not just geometry
angle) is the lever that matters -- if it doesn't, the limitation is
even more fundamental than probe count alone.

WEIGHT MODEL GENERALIZATION (documented approximation, not a new
measurement): `pair_weight_at_R` was only ever calibrated at 3 baseline
angles (monostatic=0, cross=90, antipodal=180 degrees). The new
45/135-degree pairs this 8-probe layout introduces are NOT
independently measured here -- their weight is a LINEAR INTERPOLATION
(in baseline angle, at the already-measured radius-dependent weight
functions) between the nearest measured anchors (0-90 and 90-180).
This is an honest, principled approximation, not a new calibration --
flagged clearly so it is not later mistaken for a measured result.
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
from phase3_mri_irregular_ring_reconstruction import _polar_resample, r_at_theta, build_search_grid

from matplotlib import pyplot as plt
import os

PATIENT_ID = "patient023"

c_ref = cfg.CHEST_WALL_PROXY.sound_speed
dx = (cfg.DX_M, cfg.DX_M)

# Same standard geometry as every other test this thread (probe
# standoff already proven irrelevant, run -52) -- ONLY the probe COUNT
# changes here, to isolate that single variable.
PROBE_DIST_CELLS = 120
N = (300, 300)
center = (150, 150)
domain = Domain(N, dx)

N_PROBES = 8
PROBE_ANGLES_DEG = np.arange(0, 360, 360 // N_PROBES)  # 0,45,90,...,315


def direction_vector(theta_deg):
    theta = np.deg2rad(theta_deg)
    return -np.cos(theta), np.sin(theta)  # (d_row, d_col), same convention as the rest of this thread


def _probe_src_rcv(theta_deg, offset_cells=5):
    d_row, d_col = direction_vector(theta_deg)
    probe_row = center[0] + PROBE_DIST_CELLS * d_row
    probe_col = center[1] + PROBE_DIST_CELLS * d_col
    # tangential direction (perpendicular to radial), verified to match
    # the existing top/bottom/left/right pitch-catch convention exactly
    # (rotate direction_vector by -90 degrees: (t_row,t_col)=(d_col,-d_row))
    t_row, t_col = d_col, -d_row
    src = (round(probe_col - offset_cells * t_col), round(probe_row - offset_cells * t_row))
    rcv = (round(probe_col + offset_cells * t_col), round(probe_row + offset_cells * t_row))
    return src, rcv


PROBE_NAMES = [f"p{int(a)}" for a in PROBE_ANGLES_DEG]
_ANGLE_OF = dict(zip(PROBE_NAMES, PROBE_ANGLES_DEG))
_SRC = {}
_RCV = {}
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


# --- Generalized curvature-weight model (baseline-angle interpolation) ---
_CAL_R = np.array([41.0, 71.0, 88.0])
_CAL_CROSS = np.array([0.136, 0.000, 0.0001])       # measured at baseline=90deg (run -44/-53)
_CAL_ANTIPODAL = np.array([0.045, 0.000, 0.0003])   # measured at baseline=180deg (run -44/-53)


def _linear_weight(R, cal_r, cal_w):
    return float(np.clip(np.interp(R, cal_r, cal_w), 0.0, 1.0))


def angular_separation(theta_a, theta_b):
    d = abs(theta_a - theta_b) % 360
    return min(d, 360 - d)


def pair_weight_at_R(tx_name, rx_name, R):
    sep = angular_separation(_ANGLE_OF[tx_name], _ANGLE_OF[rx_name])
    if sep < 1e-6:
        return 1.0  # monostatic, always trusted (per run -44 onward)
    cross_w = _linear_weight(R, _CAL_R, _CAL_CROSS)        # anchor at sep=90
    antipodal_w = _linear_weight(R, _CAL_R, _CAL_ANTIPODAL)  # anchor at sep=180
    if sep <= 90:
        frac = sep / 90.0
        return 1.0 + frac * (cross_w - 1.0)
    frac = (sep - 90.0) / 90.0
    return cross_w + frac * (antipodal_w - cross_w)


N_ANGLES = 144
_THETAS = np.linspace(0, 360, N_ANGLES, endpoint=False)


def select_best_local_peak(scale_grid, scores, step_tol=1.5):
    """Require the winning candidate to be a genuine LOCAL MAXIMUM (rises
    then falls on both sides), not just the highest score in the allowed
    range -- disqualifies a monotonic climb into a guard-band cutoff
    (leakage from the adjacent excluded region), WITHOUT presupposing
    where the true answer is. `scale_grid` may be discontinuous (the
    guard band removes a contiguous middle chunk) -- find_peaks is run
    separately on each contiguous segment so a segment's own edge (e.g.
    the point immediately next to the guard-band gap) is never mistaken
    for an interior peak, matching what find_peaks already does at a
    true array boundary.
    """
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
        # no genuine interior local max anywhere -- fall back to global
        # argmax, but this should be rare for a real reflector and is
        # worth flagging if it happens.
        best_idx = int(np.argmax(scores))
        return scale_grid[best_idx], scores[best_idx], False, 0.0  # no genuine peak -> untrustworthy, not high-confidence

    order = np.argsort(all_peak_scores)[::-1]
    best_pos = all_peak_positions[order[0]]
    best_score = all_peak_scores[order[0]]
    # Confidence: best peak vs. the next-best genuine peak (NOT vs. the
    # global minimum) -- a single, isolated peak with no real competition
    # gets confidence=inf; a peak that's only marginally ahead of another
    # real local max gets a low ratio, flagging it as borderline.
    confidence = best_score / all_peak_scores[order[1]] if len(order) > 1 else np.inf
    return scale_grid[best_pos], best_score, True, confidence


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
    best_scale, _, is_genuine_peak, confidence = select_best_local_peak(scale_grid, scores)
    return best_scale, scores, is_genuine_peak, confidence


SCALE_GRID = np.arange(0.7, 1.31, 0.005)
GUARD_BAND_CELLS = 8.0

if __name__ == "__main__":
    print(f"*** {labels.PENDING_SIGNOFF_BANNER} ***")
    print(f"*** {labels.TOY_EXACT_GT_CAPTION} ***")
    print(f"ISOLATED 8-probe test for {PATIENT_ID}: {N_PROBES} probes at "
          f"{list(PROBE_ANGLES_DEG)} degrees (vs. the standard 4 at 0/90/180/270), "
          f"same PROBE_DIST_CELLS={PROBE_DIST_CELLS}. Self-contained script -- no "
          f"existing file modified. Testing whether more monostatic-type votes "
          f"fixes patient023's confirmed structural outer-boundary limitation.")

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
    print(f"  placing ring centroid at domain center {center} via offset ({offset_row}, {offset_col})")

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
    print(f"  mean inner radius={ext_r_in.mean():.1f} cells, mean outer radius={ext_r_out.mean():.1f} cells")

    needed_extent = max(ext_r_out.max(), ext_r_in.max()) * SCALE_GRID.max() + 15.0
    img_rows_g, img_cols_g = build_search_grid(needed_extent, margin=15.0, density_per_cell=100.0 / 180.0)
    print(f"  search grid: {len(img_rows_g)}x{len(img_cols_g)}, +/-{needed_extent:.0f} cells")

    medium = build_medium_real_contour(label_map)
    print(f"\n=== Simulating real-contour phantom ({N_PROBES} transmits, {N_PROBES**2} tx/rx pairs) ===")
    pairs_real = capture_all_pairs(medium)
    print("=== Simulating homogeneous reference ===")
    pairs_ref = capture_all_pairs(build_medium_homogeneous())

    fitted_s_in, scores_in, in_is_peak, in_conf = fit_scale_curvature_weighted(pairs_real, ext_theta_in, ext_r_in, SCALE_GRID, lv_centroid_dom, img_rows_g, img_cols_g)
    fitted_s_in_ref, _, in_ref_is_peak, in_ref_conf = fit_scale_curvature_weighted(pairs_ref, ext_theta_in, ext_r_in, SCALE_GRID, lv_centroid_dom, img_rows_g, img_cols_g)

    fitted_inner_mean_radius = fitted_s_in * ext_r_in.mean()
    scale_grid_guarded = SCALE_GRID[np.abs(SCALE_GRID * ext_r_out.mean() - fitted_inner_mean_radius) > GUARD_BAND_CELLS]
    fitted_s_out, scores_out, out_is_peak, out_conf = fit_scale_curvature_weighted(pairs_real, ext_theta_out, ext_r_out, scale_grid_guarded, ring_centroid_dom, img_rows_g, img_cols_g)
    fitted_s_out_ref, _, out_ref_is_peak, out_ref_conf = fit_scale_curvature_weighted(pairs_ref, ext_theta_out, ext_r_out, SCALE_GRID, ring_centroid_dom, img_rows_g, img_cols_g)

    in_err_mm = abs(fitted_s_in - 1.0) * ext_r_in.mean() * dx[0] * 1e3
    out_err_mm = abs(fitted_s_out - 1.0) * ext_r_out.mean() * dx[0] * 1e3
    locked = abs(fitted_s_out * ext_r_out.mean() - fitted_s_in * ext_r_in.mean()) < 3.0

    print(f"\n--- Result (8-probe test, LOCAL-MAXIMUM-ONLY selection) ---")
    print(f"  inner: fitted scale={fitted_s_in:.3f} (true=1.000), error={in_err_mm:.2f}mm, "
          f"genuine_local_max={in_is_peak}, confidence={in_conf:.2f}")
    print(f"  outer: fitted scale={fitted_s_out:.3f} (true=1.000), error={out_err_mm:.2f}mm, "
          f"locked_to_inner={locked}, genuine_local_max={out_is_peak}, confidence={out_conf:.2f}")
    print(f"  homogeneous control: inner={fitted_s_in_ref:.3f} (conf={in_ref_conf:.2f}), "
          f"outer={fitted_s_out_ref:.3f} (conf={out_ref_conf:.2f}) "
          f"-- should be LOW confidence even if a plausible-looking scale is picked")
    print(f"\n  4-probe baseline (runs -51/-53, global argmax): inner err=0.45mm, outer err=2.43mm")
    print(f"  8-probe, global-argmax result (run -56): inner err=0.18mm, outer err=2.16mm")

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))
    axes[0].plot(SCALE_GRID, scores_in / scores_in.max())
    axes[0].axvline(1.0, color="k", linestyle="--", label="true=1.0")
    axes[0].axvline(fitted_s_in, color="g", linestyle=":", label=f"fitted={fitted_s_in:.3f}")
    axes[0].set_title(f"Inner scale-fit ({N_PROBES} probes)")
    axes[0].legend(fontsize=8)
    axes[1].plot(scale_grid_guarded, scores_out / scores_out.max())
    axes[1].axvline(1.0, color="k", linestyle="--", label="true=1.0")
    axes[1].axvline(fitted_s_out, color="g", linestyle=":", label=f"fitted={fitted_s_out:.3f}")
    axes[1].set_title(f"Outer scale-fit ({N_PROBES} probes)")
    axes[1].legend(fontsize=8)
    fig.suptitle(f"Isolated {N_PROBES}-probe test ({PATIENT_ID}), LOCAL-MAXIMUM-ONLY selection\n"
                 f"(disqualifies a monotonic climb into the guard-band cutoff -- must be a genuine interior peak)")
    labels.add_banner(fig)
    plt.tight_layout()
    os.makedirs("results/figures", exist_ok=True)
    out_fig = f"results/figures/phase3_mri_8probe_localmax_test_{PATIENT_ID}.png"
    plt.savefig(out_fig, dpi=130)
    print(f"\nSaved {out_fig}")
