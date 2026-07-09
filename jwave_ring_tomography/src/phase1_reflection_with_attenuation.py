"""Phase 1/2 scout, re-testing runs -08/-09's reflection channel WITH
real attenuation physics now switched on (channel 2, validated in run
-10), directly testing the user's hypothesis: does absorption further
weaken inner-boundary detectability and/or change the +7.8 cell timing
bias found when attenuation was completely absent?

Same two-tissue patient001 phantom, same pitch-catch geometry, same
blind two-peak detection method as run -09 -- ONLY the solver changes
(`simulate_wave_propagation_attenuating` instead of
`simulate_wave_propagation`), so any difference in the result is
attributable to attenuation, not a confounded methodology change.
"""

import numpy as np
from scipy.signal import hilbert, find_peaks

from jax import jit
from jax import numpy as jnp
from jwave import FourierSeries
from jwave.geometry import Medium, Sources

from attenuation_solver import simulate_wave_propagation_attenuating, attenuation_field_np_per_m
from phase1_reflection_channel_scout import (
    thetas, pitch_catch_positions, DIRECT_EXCLUDE_MARGIN_S, _ENVELOPE_GROUP_DELAY_S,
    polar_resample, r_at_theta, predicted_reflection_times, peak_in_window, SEARCH_WINDOW_S,
)
from phase1_fused_channel_reconstruction import blind_two_peak_distances
from phase1_two_tissue_reconstruction import load_real_contours_two_tissue
from phase1_rotating_transmission_scout import dx, center, N, domain, time_axis, dt, t_arr, _signal_template
import phase2_config as cfg
import labels

from matplotlib import pyplot as plt
import os


def build_attenuating_medium(canvas_lv, canvas_myo, tissue_water, tissue_myo, tissue_blood):
    sound_speed_map = np.where(canvas_myo, tissue_myo.sound_speed, tissue_water.sound_speed).astype(np.float32)
    density_map = np.where(canvas_myo, tissue_myo.density, tissue_water.density).astype(np.float32)
    sound_speed_map = np.where(canvas_lv, tissue_blood.sound_speed, sound_speed_map).astype(np.float32)
    density_map = np.where(canvas_lv, tissue_blood.density, density_map).astype(np.float32)

    alpha_water = attenuation_field_np_per_m(tissue_water.attenuation, cfg.F0_HZ, y=1.0)
    alpha_myo = attenuation_field_np_per_m(tissue_myo.attenuation, cfg.F0_HZ, y=1.0)
    alpha_blood = attenuation_field_np_per_m(tissue_blood.attenuation, cfg.F0_HZ, y=1.0)
    atten_map = np.where(canvas_myo, alpha_myo, alpha_water).astype(np.float32)
    atten_map = np.where(canvas_lv, alpha_blood, atten_map).astype(np.float32)

    ssm = jnp.expand_dims(jnp.array(sound_speed_map), -1)
    dm = jnp.expand_dims(jnp.array(density_map), -1)
    am = jnp.expand_dims(jnp.array(atten_map), -1)
    medium = Medium(domain=domain, sound_speed=FourierSeries(ssm, domain), density=FourierSeries(dm, domain))
    atten_field = FourierSeries(am, domain)
    return medium, atten_field


def build_water_only_attenuating():
    sound_speed_map = np.full(N, cfg.WATER.sound_speed, dtype=np.float32)
    density_map = np.full(N, cfg.WATER.density, dtype=np.float32)
    alpha_water = attenuation_field_np_per_m(cfg.WATER.attenuation, cfg.F0_HZ, y=1.0)
    atten_map = np.full(N, alpha_water, dtype=np.float32)
    ssm = jnp.expand_dims(jnp.array(sound_speed_map), -1)
    dm = jnp.expand_dims(jnp.array(density_map), -1)
    am = jnp.expand_dims(jnp.array(atten_map), -1)
    medium = Medium(domain=domain, sound_speed=FourierSeries(ssm, domain), density=FourierSeries(dm, domain))
    atten_field = FourierSeries(am, domain)
    return medium, atten_field


def simulate_pitch_catch_attenuating(medium, atten_field, theta_deg):
    src, rcv = pitch_catch_positions(theta_deg)
    sources = Sources(positions=([src[0]], [src[1]]), signals=_signal_template, dt=dt, domain=domain)

    @jit
    def run(m, a):
        return simulate_wave_propagation_attenuating(m, time_axis, a, cfg.F0_HZ, sources=sources)

    pressure = run(medium, atten_field)
    trace = np.array(pressure.on_grid[:, rcv[0], rcv[1], 0])
    direct_time = np.hypot(src[0] - rcv[0], src[1] - rcv[1]) * dx[0] / cfg.WATER.sound_speed
    mask = np.abs(t_arr - direct_time) < DIRECT_EXCLUDE_MARGIN_S
    trace = trace.copy()
    trace[mask] = 0.0
    return np.abs(hilbert(trace))


if __name__ == "__main__":
    print(f"*** {labels.PENDING_SIGNOFF_BANNER} ***")
    print(f"*** {labels.TOY_EXACT_GT_CAPTION} ***")
    print("PHASE 1/2, CHANNEL 2 (ABSORPTION) RE-TEST: same pitch-catch reflection scan as "
          "runs -08/-09, WITH real attenuation physics now switched on (validated run -10). "
          "Testing: does absorption weaken inner-boundary detectability and/or change the "
          "+7.8 cell timing bias found when attenuation was completely absent?")
    print("  compute estimate: 36 angles x 2 media = 72 forward sims (same as prior runs) "
          "-- ~15-20 minutes based on that precedent")

    canvas_lv, canvas_myo, outer_contour_dom, inner_contour_dom, max_radius_cells = load_real_contours_two_tissue()
    ext_theta_out, ext_r_out = polar_resample(outer_contour_dom, center)
    ext_theta_in, ext_r_in = polar_resample(inner_contour_dom, center)

    medium_water, atten_water = build_water_only_attenuating()
    medium_phantom, atten_phantom = build_attenuating_medium(canvas_lv, canvas_myo, cfg.WATER, cfg.MYOCARDIUM, cfg.BLOOD)

    print("\n=== Simulating water-only control (attenuating), pitch-catch at 36 angles ===")
    water_envelopes = [simulate_pitch_catch_attenuating(medium_water, atten_water, th) for th in thetas]
    print("=== Simulating two-tissue phantom (attenuating), pitch-catch at 36 angles ===")
    phantom_envelopes = [simulate_pitch_catch_attenuating(medium_phantom, atten_phantom, th) for th in thetas]

    print("\n=== Blind per-angle reflection peak detection (with attenuation) ===")
    refl_r_outer, refl_r_inner = [], []
    outer_amp, inner_amp = [], []
    for i, theta in enumerate(thetas):
        water_max = water_envelopes[i].max()
        r_out, r_in = blind_two_peak_distances(phantom_envelopes[i], water_max)
        refl_r_outer.append(r_out)
        refl_r_inner.append(r_in)
        r_out_true = r_at_theta(theta, ext_theta_out, ext_r_out)
        r_in_true = r_at_theta(theta, ext_theta_in, ext_r_in)
        t_outer, t_inner = predicted_reflection_times(theta, r_out_true, r_in_true)
        outer_amp.append(peak_in_window(phantom_envelopes[i], t_outer))
        inner_amp.append(peak_in_window(phantom_envelopes[i], t_inner))

    n_outer_found = sum(r is not None for r in refl_r_outer)
    n_inner_found = sum(r is not None for r in refl_r_inner)
    print(f"  outer boundary candidate found (WITH attenuation): {n_outer_found}/{len(thetas)}")
    print(f"  inner boundary candidate found (WITH attenuation): {n_inner_found}/{len(thetas)} "
          f"(run -09, no attenuation, found: 36/36)")

    true_r_outer_by_angle = [r_at_theta(th, ext_theta_out, ext_r_out) for th in thetas]
    true_r_inner_by_angle = [r_at_theta(th, ext_theta_in, ext_r_in) for th in thetas]
    outer_err = np.array([r - t for r, t in zip(refl_r_outer, true_r_outer_by_angle) if r is not None])
    inner_matched_true = [t for r, t in zip(refl_r_inner, true_r_inner_by_angle) if r is not None]
    inner_err = np.array([r - t for r, t in zip(refl_r_inner, inner_matched_true) if r is not None])
    print(f"\n--- Radius bias, WITH attenuation ---")
    print(f"  outer boundary: mean error={outer_err.mean():+.1f} cells (run -09 without attenuation: +0.1)")
    print(f"  inner boundary: mean error={inner_err.mean():+.1f} cells (run -09 without attenuation: +7.8)")

    outer_amp, inner_amp = np.array(outer_amp), np.array(inner_amp)
    print(f"\n--- Amplitude, WITH attenuation ---")
    print(f"  outer mean amplitude={outer_amp.mean():.4g}, inner mean amplitude={inner_amp.mean():.4g}")
    print(f"  inner/outer amplitude ratio={inner_amp.mean()/outer_amp.mean():.3f} "
          f"(run -08 without attenuation: 0.235)")

    fig, ax = plt.subplots(figsize=(7, 5))
    rep_idx = 0
    ax.plot(t_arr * 1e6, phantom_envelopes[rep_idx], label="two-tissue phantom (WITH attenuation)", color="C1")
    ax.plot(t_arr * 1e6, water_envelopes[rep_idx], label="water-only control", color="C0", alpha=0.7)
    t_outer_rep, t_inner_rep = predicted_reflection_times(
        thetas[rep_idx], r_at_theta(thetas[rep_idx], ext_theta_out, ext_r_out),
        r_at_theta(thetas[rep_idx], ext_theta_in, ext_r_in))
    ax.axvline(t_outer_rep * 1e6, color="c", linestyle="--", label="predicted outer reflection")
    ax.axvline(t_inner_rep * 1e6, color="lime", linestyle="--", label="predicted inner reflection")
    ax.set_xlabel("time (us)")
    ax.set_ylabel("envelope amplitude")
    ax.set_title(f"WITH attenuation: representative A-scan, theta={thetas[rep_idx]:.0f}deg")
    ax.legend(fontsize=7)

    fig.suptitle("Channel 2 (absorption) re-test: reflection A-scan WITH real attenuation physics")
    labels.add_banner(fig)
    plt.tight_layout()
    os.makedirs("results/figures", exist_ok=True)
    out_fig = "results/figures/phase1_reflection_with_attenuation.png"
    plt.savefig(out_fig, dpi=140)
    print(f"\nSaved {out_fig}")
