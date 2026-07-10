"""Tests CHANNEL 2-alternate (backscatter/speckle texture), per user:
"but static echo can still diff myo and blood?" -- following the
finding that the smooth-interface reflection coefficient (~0.25%) is
too weak for DAS or any method tried so far to recover the inner
(blood/myocardium) boundary (run -32). Real clinical echocardiography
does NOT primarily rely on that interface reflection to see the
endocardial border -- it relies on VOLUME BACKSCATTER CONTRAST:
myocardium's fibrous microstructure scatters ultrasound diffusely
(the "speckle" texture filling the wall in any real echo image);
blood, a near-homogeneous fluid at these scales/frequencies, does not,
and appears nearly anechoic. This is a DIFFERENT physical channel from
anything tested in this project so far -- every phantom built here
(heart, patient001, patient023) has modeled myocardium and blood as
perfectly homogeneous regions, so speckle CANNOT appear in any of those
simulations by construction; only the boundary between them can do
anything, and that's the channel just shown to be too weak.

Direct test: build a two-tissue circular phantom (same R_OUTER=80,
R_INNER=60 geometry as the run -12 positive control) with the
myocardium ring given RANDOMIZED per-grid-cell sound-speed/density
fluctuations (a standard coarse proxy for unresolved microstructure
scattering) -- blood stays perfectly homogeneous, matching the real
near-anechoic-blood assumption. Compares the received echo's energy
WITHIN THE MYOCARDIUM-WALL TIME WINDOW (between the predicted outer and
inner boundary arrivals -- i.e., echoes that could ONLY come from
inside the wall's volume, not from either boundary) against the SAME
window for a perfectly homogeneous myocardium wall (run -12's own
phantom, reused unchanged).

PREDICTION, stated before running: the homogeneous phantom's within-
wall window should be close to the noise floor (a perfectly homogeneous
region has nothing to scatter from except its two boundaries, both
excluded from this window). The speckle phantom's within-wall window
should show detectably ELEVATED energy if volume backscatter is a real,
simulable effect at this grid resolution (dx=0.1mm, ~6 cells/wavelength
at 2.5MHz) -- if it does NOT elevate, that specifically means either
the grid resolution is too coarse to resolve wavelength-scale scattering
in this simulation, or the chosen fluctuation magnitude is too weak, not
that the physical mechanism itself is invalid (real speckle is a
well-established, ubiquitous clinical phenomenon).

Fixed seed logged for reproducibility, per this project's standing
seed-discipline: SPECKLE_SEED = 42.
"""

import numpy as np
from scipy.signal import correlate, hilbert

from jax import numpy as jnp, jit
from jwave import FourierSeries
from jwave.geometry import Medium, Sources
from jwave.acoustics import simulate_wave_propagation

from phase1_circular_positive_control import build_medium_concentric_circles, R_OUTER, R_INNER
from phase1_reflection_channel_scout import thetas, pitch_catch_positions, DIRECT_EXCLUDE_MARGIN_S
from phase1_matched_filter_echo_extraction import _lag_t_arr, _template, predicted_times_no_group_delay
from phase1_rotating_transmission_scout import (
    center, N, domain, time_axis, dt, t_arr, _signal_template, dx,
)
import phase2_config as cfg
import labels

from matplotlib import pyplot as plt
import os

SPECKLE_SEED = 42
SPECKLE_STD_FRAC = 0.03  # 3% relative random fluctuation, sound speed and density independently
_nonneg = _lag_t_arr >= 0
_WINDOW_MARGIN_S = 0.3e-6  # trim this much off each end of the [t_outer, t_inner] window, to
                            # reduce contamination from the two boundary echoes' own matched-filter width


def build_medium_speckle_myocardium(seed=SPECKLE_SEED, std_frac=SPECKLE_STD_FRAC):
    rng = np.random.default_rng(seed)
    yy, xx = np.mgrid[0:N[0], 0:N[1]]
    dist = np.sqrt((xx - center[1]) ** 2 + (yy - center[0]) ** 2)
    inside_myo = dist < R_OUTER
    inside_lv = dist < R_INNER
    myo_only = inside_myo & ~inside_lv

    sound_speed_map = np.full(N, cfg.WATER.sound_speed, dtype=np.float32)
    density_map = np.full(N, cfg.WATER.density, dtype=np.float32)

    speed_noise = rng.normal(1.0, std_frac, size=N).astype(np.float32)
    density_noise = rng.normal(1.0, std_frac, size=N).astype(np.float32)
    sound_speed_map = np.where(myo_only, cfg.MYOCARDIUM.sound_speed * speed_noise, sound_speed_map).astype(np.float32)
    density_map = np.where(myo_only, cfg.MYOCARDIUM.density * density_noise, density_map).astype(np.float32)

    sound_speed_map = np.where(inside_lv, cfg.BLOOD.sound_speed, sound_speed_map).astype(np.float32)
    density_map = np.where(inside_lv, cfg.BLOOD.density, density_map).astype(np.float32)

    ssm = jnp.expand_dims(jnp.array(sound_speed_map), -1)
    dm = jnp.expand_dims(jnp.array(density_map), -1)
    return Medium(domain=domain, sound_speed=FourierSeries(ssm, domain), density=FourierSeries(dm, domain))


def simulate_pitch_catch_raw(medium, theta_deg):
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


def within_wall_energy(envelope, t_outer, t_inner):
    lo, hi = t_outer + _WINDOW_MARGIN_S, t_inner - _WINDOW_MARGIN_S
    if hi <= lo:
        return np.nan
    mask = (_lag_t_arr >= lo) & (_lag_t_arr <= hi)
    if not mask.any():
        return np.nan
    return np.sqrt(np.mean(envelope[mask] ** 2))


if __name__ == "__main__":
    print(f"*** {labels.PENDING_SIGNOFF_BANNER} ***")
    print(f"*** {labels.TOY_EXACT_GT_CAPTION} ***")
    print("BACKSCATTER/SPECKLE CHANNEL TEST: does randomized within-tissue microstructure "
          "(a standard coarse proxy for real myocardial fiber scattering) produce detectable "
          "echo energy from WITHIN the myocardium wall's volume, distinct from the too-weak "
          "boundary reflection coefficient (run -32)? Circular/centered phantom (R_outer=80, "
          f"R_inner=60), speckle seed={SPECKLE_SEED}, std_frac={SPECKLE_STD_FRAC}.")
    print("  compute estimate: 36 angles x 2 media (speckle vs. homogeneous myocardium) = "
          "72 forward sims -- ~15-20 minutes based on prior-run precedent")

    medium_homogeneous = build_medium_concentric_circles()
    medium_speckle = build_medium_speckle_myocardium()

    print("\n=== Simulating HOMOGENEOUS myocardium phantom, pitch-catch at 36 angles ===")
    homog_traces = [simulate_pitch_catch_raw(medium_homogeneous, th) for th in thetas]
    print("=== Simulating SPECKLE myocardium phantom, pitch-catch at 36 angles ===")
    speckle_traces = [simulate_pitch_catch_raw(medium_speckle, th) for th in thetas]

    os.makedirs("results", exist_ok=True)
    np.savez("results/speckle_channel_raw_traces.npz",
             thetas=thetas, homogeneous_traces=np.array(homog_traces), speckle_traces=np.array(speckle_traces),
             seed=SPECKLE_SEED, std_frac=SPECKLE_STD_FRAC)

    homog_env = [matched_filter_envelope(tr) for tr in homog_traces]
    speckle_env = [matched_filter_envelope(tr) for tr in speckle_traces]

    print("\n=== Comparing within-wall echo energy: speckle vs. homogeneous myocardium ===")
    homog_energy, speckle_energy = [], []
    for i, theta in enumerate(thetas):
        t_outer, t_inner = predicted_times_no_group_delay(theta, R_OUTER, R_INNER)
        homog_energy.append(within_wall_energy(homog_env[i], t_outer, t_inner))
        speckle_energy.append(within_wall_energy(speckle_env[i], t_outer, t_inner))
    homog_energy, speckle_energy = np.array(homog_energy), np.array(speckle_energy)

    ratio = np.nanmean(speckle_energy) / np.nanmean(homog_energy) if np.nanmean(homog_energy) > 0 else float("inf")
    print(f"  homogeneous-wall within-window RMS energy: mean={np.nanmean(homog_energy):.4g}")
    print(f"  speckle-wall within-window RMS energy:     mean={np.nanmean(speckle_energy):.4g}")
    print(f"  ratio (speckle/homogeneous): {ratio:.2f}x")
    if ratio > 3.0:
        print("  -> CLEAR elevation: volume backscatter is a real, detectable channel at this "
              "grid resolution/fluctuation magnitude.")
    elif ratio > 1.3:
        print("  -> Modest elevation: a real but weak effect at this fluctuation magnitude/resolution.")
    else:
        print("  -> NO clear elevation: either resolution or fluctuation magnitude insufficient "
              "at this setting, not evidence the mechanism itself is wrong.")

    rep_idx = 0
    t_outer_rep, t_inner_rep = predicted_times_no_group_delay(thetas[rep_idx], R_OUTER, R_INNER)
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    axes[0].plot(_lag_t_arr[_nonneg] * 1e6, speckle_env[rep_idx][_nonneg], label="speckle myocardium", color="C1")
    axes[0].plot(_lag_t_arr[_nonneg] * 1e6, homog_env[rep_idx][_nonneg], label="homogeneous myocardium", color="C0", alpha=0.7)
    axes[0].axvspan(t_outer_rep * 1e6, t_inner_rep * 1e6, color="yellow", alpha=0.15, label="within-wall window")
    axes[0].set_xlim(0, t_arr.max() * 1e6)
    axes[0].set_xlabel("time (us)")
    axes[0].set_ylabel("matched-filter envelope amplitude")
    axes[0].set_title(f"Representative A-scan, theta={thetas[rep_idx]:.0f}deg")
    axes[0].legend(fontsize=7)

    axes[1].plot(thetas, homog_energy, "o-", label="homogeneous wall")
    axes[1].plot(thetas, speckle_energy, "s-", label="speckle wall")
    axes[1].set_xlabel("angle (deg)")
    axes[1].set_ylabel("within-wall window RMS energy")
    axes[1].set_title(f"Within-wall energy vs. angle (ratio={ratio:.2f}x)")
    axes[1].legend(fontsize=8)

    fig.suptitle("Backscatter/speckle channel test: does within-wall microstructure scattering appear?")
    labels.add_banner(fig)
    plt.tight_layout()
    os.makedirs("results/figures", exist_ok=True)
    out_fig = "results/figures/phase1_backscatter_speckle_channel.png"
    plt.savefig(out_fig, dpi=140)
    print(f"\nSaved {out_fig}")
