"""Step 1 of speckle-informed surface selection (patient023): builds the
real, irregular two-tissue anatomy with randomized myocardium
microstructure (same 3% relative fluctuation, seed=42, as the circular
positive control in run -33/-35) and runs the 36-angle pitch-catch scan.

Reuses the ALREADY-CACHED homogeneous patient023 reflection traces
(`results/patient023_reflection_raw_traces.npz`, run -32) as the
background/comparison half -- only the SPECKLE variant needs a fresh
simulation (36 sims, not 72).
"""

import numpy as np

from jax import numpy as jnp, jit
from jwave import FourierSeries
from jwave.geometry import Medium, Sources
from jwave.acoustics import simulate_wave_propagation

from phase1_patient023_validation import load_real_contours, MRI_NPZ
from phase1_reflection_channel_scout import thetas, pitch_catch_positions, DIRECT_EXCLUDE_MARGIN_S
from phase1_rotating_transmission_scout import center, N, domain, time_axis, dt, t_arr, _signal_template, dx
import phase2_config as cfg
import labels

import os

SPECKLE_SEED = 42
SPECKLE_STD_FRAC = 0.03


def build_medium_speckle_two_tissue(canvas_lv, canvas_myo, seed=SPECKLE_SEED, std_frac=SPECKLE_STD_FRAC):
    rng = np.random.default_rng(seed)
    sound_speed_map = np.full(N, cfg.WATER.sound_speed, dtype=np.float32)
    density_map = np.full(N, cfg.WATER.density, dtype=np.float32)

    speed_noise = rng.normal(1.0, std_frac, size=N).astype(np.float32)
    density_noise = rng.normal(1.0, std_frac, size=N).astype(np.float32)
    sound_speed_map = np.where(canvas_myo, cfg.MYOCARDIUM.sound_speed * speed_noise, sound_speed_map).astype(np.float32)
    density_map = np.where(canvas_myo, cfg.MYOCARDIUM.density * density_noise, density_map).astype(np.float32)

    # blood stays perfectly homogeneous (near-anechoic assumption, same as run -33)
    sound_speed_map = np.where(canvas_lv, cfg.BLOOD.sound_speed, sound_speed_map).astype(np.float32)
    density_map = np.where(canvas_lv, cfg.BLOOD.density, density_map).astype(np.float32)

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


if __name__ == "__main__":
    print(f"*** {labels.PENDING_SIGNOFF_BANNER} ***")
    print(f"*** {labels.TOY_EXACT_GT_CAPTION} ***")
    print("SPECKLE PATIENT023 SIMULATION: real irregular anatomy with randomized myocardium "
          f"microstructure (seed={SPECKLE_SEED}, std_frac={SPECKLE_STD_FRAC}). Reuses cached "
          "homogeneous patient023 traces (run -32) as the background half -- only 36 NEW sims "
          "needed here, not 72.")
    print("  compute estimate: 36 angles x 1 medium (speckle only) = 36 forward sims -- "
          "~7-10 minutes based on half of the usual 72-sim precedent")

    canvas_lv, canvas_myo, outer_contour_dom, inner_contour_dom = load_real_contours(MRI_NPZ)
    medium_speckle = build_medium_speckle_two_tissue(canvas_lv, canvas_myo)

    print("\n=== Simulating patient023 SPECKLE two-tissue phantom, pitch-catch at 36 angles ===")
    speckle_traces = [simulate_pitch_catch_raw(medium_speckle, th) for th in thetas]

    os.makedirs("results", exist_ok=True)
    np.savez("results/patient023_speckle_raw_traces.npz", thetas=thetas,
             speckle_traces=np.array(speckle_traces), seed=SPECKLE_SEED, std_frac=SPECKLE_STD_FRAC)
    print("  saved results/patient023_speckle_raw_traces.npz")
