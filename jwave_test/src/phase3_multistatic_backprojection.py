"""Phase 3 — multistatic backprojection ("LIDAR-style") reconstruction.

Per user's diagnosis of the earlier single-echo-per-probe approach: a
single probe's return trace has MANY candidate reflection points
consistent with any one arrival time (every point on the circle of
constant range from that probe) -- picking "the" echo per probe is a
structurally ambiguous, one-x-per-y guess. What actually resolves the
ambiguity is combining ALL probes: for a hypothesized reflector location
P, each (transmit probe, receive probe) pair predicts an exact expected
travel time; only the TRUE surface location is consistent with the
travel times seen by every pair simultaneously. This is standard
multistatic backprojection (the "Total Focusing Method" in ultrasonic
NDT; mathematically identical to multilateration/LIDAR point-cloud
fusion): sweep every candidate point in a 2D grid, and for each of the
tx/rx pairs, sample that pair's envelope-detected trace at the travel
time P would produce, then SUM (accumulate) across all pairs. The peak
of the accumulator is the reconstructed surface -- not a per-probe
threshold pick.

Uses the same beating-circle phantom as phase3_beating_circle_movie.py
(single filled disk, one boundary) for a clean, visually-assessable
test, and the same 4-probe geometry (top/bottom/left/right, 12mm from
center) as phase3_four_probe_tracking.py. Departure from both: each
probe's transmit element is recorded not just at its own receive
element but at ALL FOUR probes' receive elements (4 tx x 4 rx = 16
pairs), and reconstruction uses INCOHERENT (envelope) backprojection
over the full 2D domain rather than a coherent phase-summed image along
a single column -- this also tests whether incoherent multistatic
accumulation avoids the fixed coherent-summation artifact that plagued
the focused-DAS approach (runs -22/-24), since a static artifact would
need to coincidentally satisfy all 16 pairs' delay-consistency
conditions at once, which is far less likely than for a single
densely-sampled phased array.
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
PROBE_DIST_CELLS = 120  # matches phase3_four_probe_tracking.py, validated distance

PROBES = {
    "top":    dict(row=center[0] - PROBE_DIST_CELLS, col=center[1], axis="col"),
    "bottom": dict(row=center[0] + PROBE_DIST_CELLS, col=center[1], axis="col"),
    "left":   dict(row=center[0], col=center[1] - PROBE_DIST_CELLS, axis="row"),
    "right":  dict(row=center[0], col=center[1] + PROBE_DIST_CELLS, axis="row"),
}
PROBE_NAMES = list(PROBES.keys())


def probe_source_and_receiver(probe):
    """Same pitch-catch offset (10 cells) as phase3_four_probe_tracking.py
    -- src/rcv separated tangentially so the receiver isn't saturated by
    the source's own near-field at t=0."""
    row, col, axis = probe["row"], probe["col"], probe["axis"]
    if axis == "col":
        src = (col - 5, row)   # established Sources/field (x, y) convention
        rcv = (col + 5, row)
    else:
        src = (col, row - 5)
        rcv = (col, row + 5)
    return src, rcv


_SRC = {name: probe_source_and_receiver(p)[0] for name, p in PROBES.items()}
_RCV = {name: probe_source_and_receiver(p)[1] for name, p in PROBES.items()}


def build_medium_circle(radius_cells):
    yy, xx = np.mgrid[0:N[0], 0:N[1]]
    dist = np.sqrt((xx - center[1]) ** 2 + (yy - center[0]) ** 2)
    inside = dist < radius_cells
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


_dummy_medium = build_medium_circle(p3cfg.LV_RADIUS_ED_CELLS)
_base_time_axis = TimeAxis.from_medium(_dummy_medium, cfl=cfg.CFL)
dt = _base_time_axis.dt

# Search region: a box around the phantom big enough to contain the full
# ED->ES->ED radius range with margin, used both for the accumulator grid
# and to size t_end (must cover the longest tx->P->rx path within it).
SEARCH_RADIUS_CELLS = 80
_max_leg_cells = PROBE_DIST_CELLS + SEARCH_RADIUS_CELLS
_t_end_needed = (2 * _max_leg_cells * dx[0] / c_ref) * 1.15  # 15% safety margin
time_axis = TimeAxis(dt=dt, t_end=_t_end_needed)
n_steps = int(time_axis.Nt)
t_arr = np.arange(n_steps) * dt


def toneburst(t):
    duration = cfg.N_CYCLES / cfg.F0_HZ
    sigma = duration / 6
    window = np.exp(-(t - duration / 2) ** 2 / (2 * sigma ** 2))
    return np.sin(2 * np.pi * cfg.F0_HZ * t) * window


_signal_template = jnp.array(toneburst(t_arr))[None, :]

# Direct-arrival exclusion: for pair (tx, rx), the direct (unreflected)
# wave arrives at dist(src_tx, rcv_rx)/c_ref -- zero it out with a margin
# so it can't leak into the accumulator as a spurious fixed contribution
# (same principle as DIRECT_EXCLUDE_S in every earlier script this
# session, generalized to all 16 tx/rx pairs instead of just one).
DIRECT_EXCLUDE_MARGIN_S = 1.5e-6


def simulate_probe_transmit(tx_name):
    """Fire a single unfocused pulse from tx_name's source element;
    record the resulting field at ALL 4 probes' receive elements
    (multistatic capture) -- one JAX simulation covers one row of the
    4x4 tx/rx matrix."""
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
            trace = np.array(field[:, rcv[0], rcv[1]])  # established field[:, x, y] convention
            direct_time = np.hypot(src[0] - rcv[0], src[1] - rcv[1]) * dx[0] / c_ref
            mask = np.abs(t_arr - direct_time) < DIRECT_EXCLUDE_MARGIN_S
            trace = trace.copy()
            trace[mask] = 0.0
            traces[rx_name] = trace
        return traces

    return capture


def capture_all_pairs(medium):
    """Returns dict[(tx_name, rx_name)] -> envelope-detected trace, for
    all 16 tx/rx combinations."""
    pairs = {}
    for tx_name in PROBE_NAMES:
        capture = simulate_probe_transmit(tx_name)
        traces = capture(medium)
        for rx_name, trace in traces.items():
            envelope = np.abs(hilbert(trace))
            pairs[(tx_name, rx_name)] = envelope
    return pairs


# Accumulator grid, centered on the phantom, radius SEARCH_RADIUS_CELLS + margin.
IMG_N = 100
_grid_lo = center[0] - SEARCH_RADIUS_CELLS - 10
_grid_hi = center[0] + SEARCH_RADIUS_CELLS + 10
img_rows = np.linspace(_grid_lo, _grid_hi, IMG_N)
img_cols = np.linspace(_grid_lo, _grid_hi, IMG_N)
RR, CC = np.meshgrid(img_rows, img_cols, indexing="ij")


_TONEBURST_DURATION_S = cfg.N_CYCLES / cfg.F0_HZ
_ENVELOPE_GROUP_DELAY_S = _TONEBURST_DURATION_S / 2  # Hilbert envelope of a
# windowed toneburst peaks half a pulse-duration AFTER the naive geometric
# (instantaneous) arrival time -- the transmitted envelope itself is
# centered at t=duration/2, not t=0, so a reflection of the true geometric
# path length dist/c shows its ENVELOPE peak at dist/c + duration/2, not
# dist/c. Uncorrected, this biases every backprojected point's predicted
# arrival time early, which (found empirically, run -28) biases the
# recovered radius outward by ~1-1.7mm -- a diagnosable, fixable offset,
# not a structural limitation of the method.


def backproject(pairs):
    """Sum, over all 16 tx/rx pairs, each pair's envelope sampled at the
    travel time a reflector at each grid point would produce. Peak =
    reconstructed surface (multistatic agreement), not a per-probe pick."""
    accumulator = np.zeros(RR.shape)
    for (tx_name, rx_name), envelope in pairs.items():
        src = _SRC[tx_name]
        rcv = _RCV[rx_name]
        dist_tx = np.sqrt((CC - src[0]) ** 2 + (RR - src[1]) ** 2) * dx[0]
        dist_rx = np.sqrt((CC - rcv[0]) ** 2 + (RR - rcv[1]) ** 2) * dx[0]
        t_total = (dist_tx + dist_rx) / c_ref + _ENVELOPE_GROUP_DELAY_S
        accumulator += np.interp(t_total, t_arr, envelope, left=0, right=0)
    return accumulator


def radial_peak(accumulator):
    """Bin the accumulator by radius from center and return the radius
    (cells) of peak mean accumulated energy -- a single number per frame
    for direct RMSE comparison against ground truth, using the FULL 2D
    multistatic image rather than one column."""
    r_from_center = np.sqrt((RR - center[0]) ** 2 + (CC - center[1]) ** 2)
    bins = np.arange(0, SEARCH_RADIUS_CELLS + 10, 1.0)
    bin_idx = np.digitize(r_from_center.ravel(), bins)
    vals = accumulator.ravel()
    bin_means = np.array([vals[bin_idx == b].mean() if np.any(bin_idx == b) else 0.0
                           for b in range(1, len(bins))])
    peak_bin = np.argmax(bin_means)
    return bins[peak_bin] + 0.5, bin_means[peak_bin]


if __name__ == "__main__":
    print(f"*** {labels.PENDING_SIGNOFF_BANNER} ***")
    print(f"*** {labels.TOY_EXACT_GT_CAPTION} ***")
    print("Multistatic backprojection: 4 probes (top/bottom/left/right), "
          "each firing in turn, all 4 receiving -- 16 tx/rx pairs "
          "backprojected over the full 2D domain and summed. Peak = "
          "where all pairs agree, not a per-probe threshold pick.")

    print("\n=== Control: homogeneous medium (no circle) ===")
    pairs_ref = capture_all_pairs(build_medium_homogeneous())
    accumulator_ref = backproject(pairs_ref)
    ref_peak_radius, ref_peak_val = radial_peak(accumulator_ref)
    print(f"  homogeneous-medium accumulator peak: radius={ref_peak_radius:.1f} cells, "
          f"value={ref_peak_val:.4f} (should be small/flat -- no reflector present)")

    N_FRAMES_MOVIE = 8
    phases = np.linspace(0, 1, N_FRAMES_MOVIE)
    radii_cells = [p3cfg.lv_radius_at_phase(p) for p in phases]

    frames = []
    tracked_radii = []
    tracked_vals = []
    for i, r in enumerate(radii_cells):
        print(f"=== Frame {i+1}/{N_FRAMES_MOVIE} (radius={r:.1f} cells = {r*dx[0]*1e3:.2f}mm) ===")
        pairs = capture_all_pairs(build_medium_circle(r))
        accumulator = backproject(pairs)
        accumulator_clean = accumulator - accumulator_ref
        frames.append(accumulator_clean)
        peak_radius, peak_val = radial_peak(accumulator_clean)
        tracked_radii.append(peak_radius)
        tracked_vals.append(peak_val)
        print(f"  tracked radius={peak_radius:.1f} cells (true={r:.1f}), "
              f"peak value={peak_val:.4f}")

    tracked_radii = np.array(tracked_radii)
    true_radii = np.array(radii_cells)
    errors_mm = np.abs(tracked_radii - true_radii) * dx[0] * 1e3
    rmse_mm = np.sqrt(np.mean(errors_mm ** 2))
    print(f"\n--- Multistatic backprojection radial tracking: RMSE={rmse_mm:.4f}mm ---")
    for i in range(N_FRAMES_MOVIE):
        print(f"  frame {i}: true={true_radii[i]:.1f} cells, tracked={tracked_radii[i]:.1f} cells, "
              f"error={errors_mm[i]:.2f}mm")

    # Filmstrip: each frame's full 2D accumulator with true boundary overlaid.
    n_cols = 4
    n_rows = int(np.ceil(N_FRAMES_MOVIE / n_cols))
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(3.2 * n_cols, 3.4 * n_rows))
    axes = np.array(axes).reshape(-1)
    vmax = max(np.abs(f).max() for f in frames)
    theta = np.linspace(0, 2 * np.pi, 100)
    for i, (ax, frame, r) in enumerate(zip(axes, frames, radii_cells)):
        ax.imshow(np.abs(frame), cmap="hot", vmin=0, vmax=vmax, origin="upper",
                  extent=[img_cols.min(), img_cols.max(), img_rows.max(), img_rows.min()])
        true_row = center[0] + r * np.sin(theta)
        true_col = center[1] + r * np.cos(theta)
        ax.plot(true_col, true_row, "c--", linewidth=1, alpha=0.7)
        tracked_row = center[0] + tracked_radii[i] * np.sin(theta)
        tracked_col = center[1] + tracked_radii[i] * np.cos(theta)
        ax.plot(tracked_col, tracked_row, "g:", linewidth=1.3, alpha=0.9)
        ax.set_title(f"phase={phases[i]:.2f}, r_true={r*dx[0]*1e3:.2f}mm\n"
                     f"err={errors_mm[i]:.2f}mm", fontsize=8)
        ax.axis("off")
    for ax in axes[len(frames):]:
        ax.axis("off")
    fig.suptitle("Multistatic backprojection (4 probes, 16 tx/rx pairs, incoherent envelope sum)\n"
                "cyan dashed = true boundary, green dotted = tracked radius\n"
                "(TOY: exact prescribed ground truth)", fontsize=10)
    plt.tight_layout(rect=[0, 0.02, 1, 0.88])
    labels.add_banner(fig)
    os.makedirs("results/figures", exist_ok=True)
    plt.savefig("results/figures/phase3_multistatic_backprojection.png", dpi=130)
    print("\nSaved results/figures/phase3_multistatic_backprojection.png")
