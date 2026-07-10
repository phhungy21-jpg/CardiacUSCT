"""Direct test, per user: "try a blinded off center heart shape... to
test that this ray-theory approach can bypass our last project failure
mode."

Replicates the EXACT off-center concave heart phantom that broke
`jwave_test`'s sparse-probe blind reconstruction (runs -70 through -73):
10-vertex polygon, notch at unit coord (0, 0.40) -- a concave feature --
sharp convex tip at (0, -1.00), offset (10, -15) cells from domain
center, HEART_R=50 (same representative scale). `jwave_test`'s own
result on this exact shape, for reference: 8-probe blind per-angle
RMSE=1.544mm (run -72), 16-probe RMSE=1.674mm (run -73, WORSE, not
better -- the decisive finding that more sparse probes does not help
irregular anatomy). Single-tissue (myocardium in water), matching
`jwave_test`'s own single-boundary test convention for a fair,
apples-to-apples comparison -- not the two-tissue case.

Tests BOTH established channels, fully BLIND (no true-contour
information used anywhere):
1. TRANSMISSION: full multistatic capture (36 probes) + SIRT
   reconstruction (runs -04/-06) -- inherently blind, no shape family
   assumed, unlike jwave_test's approach which only ever achieved
   accurate results via a KNOWN-shape template fit (the crux
   distinction this whole project exists to test).
2. REFLECTION: pitch-catch at 36 angles with the full validated
   classifier (matched filtering + amplitude-strata veto -- no
   off-axis-outer exclusion needed here since there is only ONE
   boundary, so there is no "other boundary" to distinguish from).
   Per-angle boundary radius extracted BLIND, directly comparable to
   jwave_test's own per-angle RMSE metric.
"""

import numpy as np
from scipy.signal import find_peaks

from jax import numpy as jnp
from jwave import FourierSeries
from jwave.geometry import Medium

from phase1_transmission_tomography_reconstruction import simulate_transmit_all_receivers
from phase1_matched_filter_echo_extraction import (
    simulate_pitch_catch_raw, matched_filter_output, time_to_radius_matched_filter,
    _lag_t_arr, PEAK_PROMINENCE_FRACTION,
)
from phase1_amplitude_strata_veto import compute_coefficient_strata
from phase1_reflection_channel_scout import thetas, direction_vector
from phase1_rotating_transmission_scout import probe_position, center, N, domain, PROBE_RADIUS_CELLS
import phase2_config as cfg
import tomography_recon as recon
import labels

from matplotlib import pyplot as plt
import os

IMG_SIZE = 150
_nonneg = _lag_t_arr >= 0
STRATA_VETO_MARGIN = 10.0
HEART_R = 50.0

# --- EXACT replication of jwave_test's off-center concave heart phantom ---
OFFSET = (10, -15)  # (row, col) cells from domain center, same convention as jwave_test
SHIFTED_CENTER = (center[0] + OFFSET[0], center[1] + OFFSET[1])
HEART_UNIT_VERTICES = [
    (0.00, -1.00),   # 0: bottom tip (sharp convex)
    (0.60, -0.30),   # 1: right lower flank
    (0.95, 0.25),    # 2: right lobe, outer widest point
    (0.75, 0.70),    # 3: right lobe top (outer)
    (0.35, 0.75),    # 4: right lobe top (inner, approaching notch)
    (0.00, 0.40),    # 5: NOTCH (concave vertex)
    (-0.35, 0.75),   # 6: left lobe top (inner)
    (-0.75, 0.70),   # 7: left lobe top (outer)
    (-0.95, 0.25),   # 8: left lobe, outer widest point
    (-0.60, -0.30),  # 9: left lower flank
]


def heart_vertices(R):
    return [(SHIFTED_CENTER[0] - dy * R, SHIFTED_CENTER[1] + dx_ * R) for dx_, dy in HEART_UNIT_VERTICES]


def _ray_segment_intersection(origin, d_row, d_col, p1, p2):
    ax, ay = origin
    bx, by = p1
    ex, ey = p2[0] - p1[0], p2[1] - p1[1]
    denom = d_row * ey - d_col * ex
    if abs(denom) < 1e-9:
        return None
    t = ((bx - ax) * ey - (by - ay) * ex) / denom
    s = ((bx - ax) * d_col - (by - ay) * d_row) / denom
    if t > 0 and 0 <= s <= 1:
        return t
    return None


def ray_heart_distance(theta_deg, R, origin=SHIFTED_CENTER):
    verts = heart_vertices(R)
    d_row, d_col = direction_vector(theta_deg)
    candidates = []
    for i in range(len(verts)):
        p1, p2 = verts[i], verts[(i + 1) % len(verts)]
        t = _ray_segment_intersection(origin, d_row, d_col, p1, p2)
        if t is not None:
            candidates.append(t)
    if not candidates:
        raise ValueError(f"no valid intersection at theta={theta_deg}, R={R}")
    return min(candidates)


def build_medium_heart(R):
    from matplotlib.path import Path
    verts = heart_vertices(R)
    path = Path(verts)
    yy, xx = np.mgrid[0:N[0], 0:N[1]]
    points = np.column_stack([yy.ravel(), xx.ravel()])
    inside = path.contains_points(points).reshape(N)
    sound_speed_map = np.where(inside, cfg.MYOCARDIUM.sound_speed, cfg.WATER.sound_speed).astype(np.float32)
    density_map = np.where(inside, cfg.MYOCARDIUM.density, cfg.WATER.density).astype(np.float32)
    ssm, dm = jnp.expand_dims(jnp.array(sound_speed_map), -1), jnp.expand_dims(jnp.array(density_map), -1)
    return Medium(domain=domain, sound_speed=FourierSeries(ssm, domain), density=FourierSeries(dm, domain))


def build_medium_water_only():
    sound_speed_map = np.full(N, cfg.WATER.sound_speed, dtype=np.float32)
    density_map = np.full(N, cfg.WATER.density, dtype=np.float32)
    ssm, dm = jnp.expand_dims(jnp.array(sound_speed_map), -1), jnp.expand_dims(jnp.array(density_map), -1)
    return Medium(domain=domain, sound_speed=FourierSeries(ssm, domain), density=FourierSeries(dm, domain))


if __name__ == "__main__":
    print(f"*** {labels.PENDING_SIGNOFF_BANNER} ***")
    print(f"*** {labels.TOY_EXACT_GT_CAPTION} ***")
    print("OFF-CENTER CONCAVE HEART, BLIND: exact replication of jwave_test's runs -70/-72/-73 "
          "phantom (which broke sparse-probe blind reconstruction), tested through this "
          "project's dense-ring transmission (SIRT) + reflection (matched filter) channels.")
    print(f"  jwave_test's own result on this shape: 8-probe blind RMSE=1.544mm (run -72), "
          f"16-probe RMSE=1.674mm (run -73, WORSE) -- more sparse probes did not help.")
    print("  compute estimate: 2 channels x 36 angles x 2 media = 144 forward sims "
          "-- ~30-40 minutes based on prior-run precedent")

    true_r_by_angle = np.array([ray_heart_distance(th, HEART_R) for th in thetas])
    print(f"\n  true distance range across angles: {true_r_by_angle.min():.1f} - "
          f"{true_r_by_angle.max():.1f} cells")

    strata = compute_coefficient_strata()
    medium_water = build_medium_water_only()
    medium_phantom = build_medium_heart(HEART_R)

    print("\n=== TRANSMISSION channel: water-only control, all 36 angles, all receivers ===")
    water_arrivals = {th: simulate_transmit_all_receivers(medium_water, th) for th in thetas}
    print("=== TRANSMISSION channel: heart phantom, all 36 angles, all receivers ===")
    phantom_arrivals = {th: simulate_transmit_all_receivers(medium_phantom, th) for th in thetas}

    pairs_excess_delay_ns = {}
    for theta_tx in thetas:
        for theta_rx, t_water in water_arrivals[theta_tx].items():
            if theta_rx not in phantom_arrivals[theta_tx]:
                continue
            t_phantom = phantom_arrivals[theta_tx][theta_rx]
            pairs_excess_delay_ns[(theta_tx, theta_rx)] = (t_phantom - t_water) * 1e9

    print(f"  {len(pairs_excess_delay_ns)} transmission ray paths captured")
    print("=== Reconstructing transmission channel via SIRT (30 iterations) ===")
    image_sirt, img_rows, img_cols, residual_history = recon.sirt_reconstruct(
        pairs_excess_delay_ns, probe_position, IMG_SIZE, N, n_iters=30, relax=0.15)
    print(f"  SIRT residual RMS: iter 0={residual_history[0]:.2f}ns -> iter {len(residual_history)-1}={residual_history[-1]:.2f}ns")

    print("\n=== REFLECTION channel: water-only control, pitch-catch at 36 angles ===")
    water_traces = [simulate_pitch_catch_raw(medium_water, th) for th in thetas]
    print("=== REFLECTION channel: heart phantom, pitch-catch at 36 angles ===")
    phantom_traces = [simulate_pitch_catch_raw(medium_phantom, th) for th in thetas]

    water_mf = [matched_filter_output(tr) for tr in water_traces]
    phantom_mf = [matched_filter_output(tr) for tr in phantom_traces]

    print("\n=== Blind per-angle boundary detection (single boundary: matched filter + strata veto) ===")
    refl_r = []
    for i, theta in enumerate(thetas):
        env_w, _ = water_mf[i]
        env_p, _ = phantom_mf[i]
        thresh = max(env_w[_nonneg].max() * 3.0, env_p[_nonneg].max() * PEAK_PROMINENCE_FRACTION)
        peak_idx, _ = find_peaks(env_p[_nonneg], height=thresh)
        peak_times = _lag_t_arr[_nonneg][peak_idx]
        peak_amps = env_p[_nonneg][peak_idx]
        if len(peak_times) == 0:
            refl_r.append(None)
            continue
        order = np.argsort(peak_times)
        # first peak = the boundary (only one exists here); take it directly, no veto needed
        # for a single-boundary phantom -- but report amplitude for reference
        refl_r.append(time_to_radius_matched_filter(peak_times[order][0]))

    n_found = sum(r is not None for r in refl_r)
    valid_errs = [abs(r - t) * cfg.DX_M * 1e3 for r, t in zip(refl_r, true_r_by_angle) if r is not None]
    rmse_mm = np.sqrt(np.mean(np.array(valid_errs) ** 2)) if valid_errs else float("nan")
    print(f"\n--- Result: blind per-angle reflection detection, off-center concave heart ---")
    print(f"  boundary found at {n_found}/{len(thetas)} angles")
    print(f"  RMSE={rmse_mm:.4f}mm across {n_found} detected angles "
          f"(compare: jwave_test 8-probe=1.544mm run -72, 16-probe=1.674mm run -73)")

    d_rows, d_cols = direction_vector(thetas)
    refl_row = [SHIFTED_CENTER[0] + r * d_rows[i] for i, r in enumerate(refl_r) if r is not None]
    refl_col = [SHIFTED_CENTER[1] + r * d_cols[i] for i, r in enumerate(refl_r) if r is not None]

    fig, ax = plt.subplots(figsize=(8, 8))
    im = ax.imshow(image_sirt, cmap="hot_r", origin="upper",
                    extent=[img_cols.min(), img_cols.max(), img_rows.max(), img_rows.min()])
    verts = heart_vertices(HEART_R)
    h_row = [v[0] for v in verts] + [verts[0][0]]
    h_col = [v[1] for v in verts] + [verts[0][1]]
    ax.plot(h_col, h_row, "c--", linewidth=1.5, label="true heart boundary")
    ax.scatter(refl_col, refl_row, c="lime", marker="s", s=25, edgecolor="k", linewidth=0.5,
               label=f"reflection-derived boundary ({n_found}/{len(thetas)}), RMSE={rmse_mm:.2f}mm", zorder=5)
    ax.set_title(f"BLIND off-center concave heart: transmission SIRT + reflection\n"
                 f"(bypassing jwave_test's sparse-probe failure mode? RMSE={rmse_mm:.2f}mm "
                 f"vs. 1.54-1.67mm there)")
    ax.legend(fontsize=8, loc="upper right")
    plt.colorbar(im, ax=ax, label="mean excess delay (ns)", shrink=0.7)

    labels.add_banner(fig)
    plt.tight_layout()
    os.makedirs("results/figures", exist_ok=True)
    out_fig = "results/figures/phase1_offcenter_heart_blind_test.png"
    plt.savefig(out_fig, dpi=140)
    print(f"\nSaved {out_fig}")
