"""Replaces the reflection channel's per-angle "pick one peak, assume
it's the radial specular point" approach (runs -08 through -27) with
DELAY-AND-SUM (Kirchhoff-migration-style) reflectivity imaging -- the
standard synthetic-aperture ultrasound/seismic answer to exactly the
ambiguity the user identified: one pitch-catch firing can return
on-axis outer, off-axis outer, inner, concavity, or reverberation
echoes, all in the SAME waveform, and a single-peak-per-angle detector
has no way to know which is which.

DAS sidesteps the peak-identity question entirely instead of solving it
directly: for EVERY candidate pixel in the image, and EVERY one of the
36 pitch-catch firings, compute the PREDICTED bistatic round-trip delay
(src -> pixel -> rcv) and sample that shot's matched-filter envelope at
that exact delay. Accumulate across all 36 shots. A true reflecting
surface point is consistent with a genuine echo at its predicted delay
from MANY different firing angles at once (including off-axis and
non-radial ones), so it builds up strong accumulated energy; a point
that is not a real reflector only matches by chance for a few angles at
best, so it stays low. This is the correct way to fold in exactly the
effects the user listed (directivity, specular geometry, off-axis echo
families, concavity) WITHOUT needing an explicit per-shot classifier --
concave points are handled automatically since nothing about DAS assumes
a star-shaped/single-valued boundary.

This is the STRAIGHT-RAY (homogeneous water) version -- travel time
between src/pixel/rcv computed via simple Euclidean distance / c_water.
A bent-ray (scikit-fmm-informed) version is the natural next upgrade
(reusing this project's runs -28/-29/-30 eikonal-solver work), tested
separately once this baseline is characterized.

Runs a FRESH reflection-channel (pitch-catch) simulation on the
off-center concave heart phantom (same shape as runs -27/-29/-30) --
this project has never saved raw pitch-catch traces for this specific
phantom before, only classified peaks -- and SAVES the raw traces for
reuse by the bent-ray DAS upgrade and any future variant.
"""

import numpy as np
from scipy.signal import correlate, hilbert

from jax import jit
from jwave.geometry import Sources
from jwave.acoustics import simulate_wave_propagation

from phase1_offcenter_heart_blind_test import (
    build_medium_heart, build_medium_water_only, ray_heart_distance, heart_vertices,
    HEART_R, SHIFTED_CENTER,
)
from phase1_reflection_channel_scout import thetas, pitch_catch_positions, direction_vector
from phase1_matched_filter_echo_extraction import _lag_t_arr, _template
from phase1_rotating_transmission_scout import (
    dx, N, domain, time_axis, dt, t_arr, _signal_template, PROBE_RADIUS_CELLS,
)
from phase1_reflection_channel_scout import DIRECT_EXCLUDE_MARGIN_S
import phase2_config as cfg
import labels

from matplotlib import pyplot as plt
import os

IMG_SIZE = 150
_nonneg = _lag_t_arr >= 0


def simulate_pitch_catch_raw_at(medium, theta_deg):
    src, rcv = pitch_catch_positions(theta_deg)
    sources = Sources(positions=([src[0]], [src[1]]), signals=_signal_template, dt=dt, domain=domain)

    @jit
    def run(m):
        return simulate_wave_propagation(m, time_axis, sources=sources)

    pressure = run(medium)
    trace = np.array(pressure.on_grid[:, rcv[0], rcv[1], 0])
    direct_time = np.hypot(src[0] - rcv[0], src[1] - rcv[1]) * dx[0] / cfg.WATER.sound_speed
    mask = np.abs(t_arr - direct_time) < DIRECT_EXCLUDE_MARGIN_S
    trace = trace.copy()
    trace[mask] = 0.0
    return trace


def matched_filter_envelope(trace):
    correlated = correlate(trace, _template, mode="full")
    return np.abs(hilbert(correlated))


def das_straight_ray_image(phantom_envelopes, water_envelopes, img_size=IMG_SIZE, angle_indices=None):
    """Straight-ray delay-and-sum reflectivity image: for each pixel and
    each firing (all 36 by default, or a subset via `angle_indices` --
    used to test view-angle-count sensitivity without new simulation),
    sample that shot's (background-subtracted) matched-filter envelope
    at the predicted src->pixel->rcv bistatic delay (homogeneous water
    speed), and accumulate."""
    img_rows = np.linspace(0, N[0], img_size)
    img_cols = np.linspace(0, N[1], img_size)
    RR, CC = np.meshgrid(img_rows, img_cols, indexing="ij")
    accumulator = np.zeros((img_size, img_size))

    indices = angle_indices if angle_indices is not None else range(len(thetas))
    for i in indices:
        theta = thetas[i]
        src, rcv = pitch_catch_positions(theta)
        dist_src = np.hypot(RR - src[0], CC - src[1]) * dx[0]
        dist_rcv = np.hypot(RR - rcv[0], CC - rcv[1]) * dx[0]
        t_pred = (dist_src + dist_rcv) / cfg.WATER.sound_speed

        env_p = phantom_envelopes[i]
        env_w = water_envelopes[i]
        excess_env = np.clip(env_p - env_w, 0, None)
        sampled = np.interp(t_pred.ravel(), _lag_t_arr, excess_env, left=0.0, right=0.0)
        accumulator += sampled.reshape(img_size, img_size)

    return accumulator, img_rows, img_cols


def extract_boundary_from_image(image, img_rows, img_cols, origin, thetas_arr,
                                 r_min_cells=20.0, r_max_cells=140.0, step_cells=0.5):
    """BLIND per-angle boundary extraction from a reflectivity image:
    for each angle, march radially outward from `origin` and report the
    radius of PEAK accumulated intensity (the most likely true
    reflector location along that ray), not the first threshold
    crossing -- appropriate for a reflectivity image where the true
    boundary should be the dominant peak, not a step function."""
    def sample(rr, cc):
        ri = np.argmin(np.abs(img_rows - rr))
        ci = np.argmin(np.abs(img_cols - cc))
        return image[ri, ci]

    radii = []
    r_vals = np.arange(r_min_cells, r_max_cells, step_cells)
    for theta in thetas_arr:
        d_row, d_col = direction_vector(theta)
        vals = np.array([sample(origin[0] + r * d_row, origin[1] + r * d_col) for r in r_vals])
        radii.append(r_vals[np.argmax(vals)])
    return np.array(radii)


if __name__ == "__main__":
    print(f"*** {labels.PENDING_SIGNOFF_BANNER} ***")
    print(f"*** {labels.TOY_EXACT_GT_CAPTION} ***")
    print("DAS (DELAY-AND-SUM) REFLECTIVITY IMAGING: replaces per-angle single-peak "
          "detection with cross-angle accumulation at every candidate pixel -- the standard "
          "synthetic-aperture answer to 'which surface point produced this peak.' "
          "Off-center concave heart phantom (same shape as runs -27/-29/-30). STRAIGHT-RAY "
          "(homogeneous water) travel-time model -- bent-ray upgrade is a separate next step.")
    print("  compute estimate: 36 angles x 2 media (pitch-catch reflection) = 72 forward sims "
          "-- ~15-20 minutes based on prior-run precedent")

    medium_water = build_medium_water_only()
    medium_phantom = build_medium_heart(HEART_R)

    print("\n=== REFLECTION channel: water-only control, pitch-catch at 36 angles (raw traces) ===")
    water_traces = [simulate_pitch_catch_raw_at(medium_water, th) for th in thetas]
    print("=== REFLECTION channel: heart phantom, pitch-catch at 36 angles (raw traces) ===")
    phantom_traces = [simulate_pitch_catch_raw_at(medium_phantom, th) for th in thetas]

    os.makedirs("results", exist_ok=True)
    np.savez("results/offcenter_heart_reflection_raw_traces.npz",
             thetas=thetas, water_traces=np.array(water_traces), phantom_traces=np.array(phantom_traces))
    print("  saved results/offcenter_heart_reflection_raw_traces.npz (for reuse by bent-ray DAS upgrade)")

    water_env = [matched_filter_envelope(tr) for tr in water_traces]
    phantom_env = [matched_filter_envelope(tr) for tr in phantom_traces]

    print("\n=== Building straight-ray DAS reflectivity image (36-angle accumulation) ===")
    image, img_rows, img_cols = das_straight_ray_image(phantom_env, water_env)

    true_r_by_angle = np.array([ray_heart_distance(th, HEART_R) for th in thetas])
    r_das = extract_boundary_from_image(image, img_rows, img_cols, SHIFTED_CENTER, thetas)
    errs_mm = (r_das - true_r_by_angle) * cfg.DX_M * 1e3
    rmse_mm = np.sqrt(np.mean(errs_mm ** 2))

    print(f"\n--- Result: DAS reflectivity image, blind per-angle boundary (peak-intensity radius) ---")
    print(f"  RMSE={rmse_mm:.4f}mm across {len(thetas)} angles")
    print(f"  (compare: this project's per-angle single-peak reflection method, run -27: RMSE=1.3047mm)")
    print(f"  (compare: iterative bent-ray TRANSMISSION-channel image extraction, run -30: RMSE=3.5502mm)")
    print(f"  (compare: jwave_test's sparse-probe blind reconstruction: 8-probe=1.544mm run -72, "
          f"16-probe=1.674mm run -73)")

    d_rows, d_cols = direction_vector(thetas)
    pt_row = SHIFTED_CENTER[0] + r_das * d_rows
    pt_col = SHIFTED_CENTER[1] + r_das * d_cols

    fig, axes = plt.subplots(1, 2, figsize=(13, 5.5))
    im = axes[0].imshow(image, cmap="hot", origin="upper",
                         extent=[img_cols.min(), img_cols.max(), img_rows.max(), img_rows.min()])
    verts = heart_vertices(HEART_R)
    h_row = [v[0] for v in verts] + [verts[0][0]]
    h_col = [v[1] for v in verts] + [verts[0][1]]
    axes[0].plot(h_col, h_row, "c--", linewidth=1.5, label="true heart boundary")
    axes[0].set_title("DAS reflectivity image (straight-ray)\n(accumulated matched-filter energy)")
    axes[0].legend(fontsize=7)
    plt.colorbar(im, ax=axes[0], shrink=0.7)

    axes[1].imshow(image, cmap="hot", origin="upper",
                   extent=[img_cols.min(), img_cols.max(), img_rows.max(), img_rows.min()])
    axes[1].plot(h_col, h_row, "c--", linewidth=1.5, label="true heart boundary")
    axes[1].scatter(pt_col, pt_row, c="lime", marker="s", s=20, edgecolor="k", linewidth=0.4,
                     label=f"DAS-extracted boundary, RMSE={rmse_mm:.2f}mm", zorder=5)
    axes[1].set_title("DAS-extracted boundary vs. truth\n(compare: single-peak method run -27, RMSE=1.30mm)")
    axes[1].legend(fontsize=7)

    fig.suptitle("DAS (delay-and-sum) reflectivity imaging: off-center concave heart, straight-ray")
    labels.add_banner(fig)
    plt.tight_layout()
    os.makedirs("results/figures", exist_ok=True)
    out_fig = "results/figures/phase1_das_reflectivity_imaging_straight_ray.png"
    plt.savefig(out_fig, dpi=140)
    print(f"\nSaved {out_fig}")
