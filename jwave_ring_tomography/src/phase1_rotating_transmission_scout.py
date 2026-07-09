"""Phase 1 -- scout smoke test for the ring/water-bath transmission
tomography acquisition, per user: "test it first on a static model.
try it on a 2d cross section first. a rotating probe transmitter
beaming through the tissue and project to the opposite side, record
it, move the probe a bit, project, record. until the probe completes a
full Perimeter."

STATIC phantom only (no cardiac motion yet -- that is Phase 3+, once
this basic geometry is confirmed sane). 2D only, per protocol.
SEQUENTIAL acquisition (decided, run -02): one transmitter fires at a
time; its receiver is diametrically opposite, at the SAME probe
radius; the pair rotates together in fixed angular steps around a full
360-degree perimeter. This captures TRANSMISSION only (the through-
tissue arrival), not reflection -- matches exactly what was asked, not
more.

Deliberately the simplest possible phantom for a first sanity check: a
single, CENTERED, circular myocardium-like disk in a water bath. This
gives an analytically predictable, checkable result -- every
transmitter/receiver pair's straight-line path is a diameter of the
probe circle passing through the exact domain center, so it traverses
the SAME chord length of tissue (the phantom's full diameter) at EVERY
angle. Two independent predictions to check the simulation against
before trusting anything:
  1. A homogeneous (water-only) control's arrival time should be
     EXACTLY constant across all angles (pure rotational symmetry --
     any variation flags a geometry bug, not real physics).
  2. The phantom's arrival time should ALSO be constant across all
     angles (same chord length every time), shifted from the
     water-only baseline by a predictable amount: myocardium's cited
     sound speed (1576 m/s) is FASTER than the water bath's (1520
     m/s), so the phantom path should arrive EARLIER, not later, by
     chord_length * (1/c_water - 1/c_tissue).

Attenuation is NOT modeled in this test (jWave's base transient solver
does not implement it -- jwave_test's attenuation_solver.py would need
to be ported here separately, future work, not needed for this
geometry/timing sanity check). Reported peak amplitudes therefore
reflect only geometric spreading + transmission-coefficient effects,
not true tissue attenuation -- do not read them as an attenuation map.
"""

import numpy as np
from scipy.signal import hilbert

from jax import numpy as jnp
from jax import jit
from jwave import FourierSeries
from jwave.geometry import Domain, Medium, TimeAxis, Sources
from jwave.acoustics import simulate_wave_propagation

import phase2_config as cfg
import labels

from matplotlib import pyplot as plt
import os

dx = (cfg.DX_M, cfg.DX_M)
N = (300, 300)
center = (150, 150)
domain = Domain(N, dx)

PROBE_RADIUS_CELLS = 120.0   # rotating tx/rx radius from center
PHANTOM_RADIUS_CELLS = 60.0  # centered, static myocardium-like disk
N_ANGLES = 36                # 10-degree steps, full 360-degree perimeter


def direction_vector(theta_deg):
    theta = np.deg2rad(theta_deg)
    return -np.cos(theta), np.sin(theta)  # (d_row, d_col), same convention as jwave_test


def probe_position(theta_deg):
    d_row, d_col = direction_vector(theta_deg)
    return (round(center[0] + PROBE_RADIUS_CELLS * d_row),
            round(center[1] + PROBE_RADIUS_CELLS * d_col))


def build_medium_water_only():
    sound_speed_map = np.full(N, cfg.WATER.sound_speed, dtype=np.float32)
    density_map = np.full(N, cfg.WATER.density, dtype=np.float32)
    ssm = jnp.expand_dims(jnp.array(sound_speed_map), -1)
    dm = jnp.expand_dims(jnp.array(density_map), -1)
    return Medium(domain=domain, sound_speed=FourierSeries(ssm, domain),
                  density=FourierSeries(dm, domain))


def build_medium_static_phantom():
    yy, xx = np.mgrid[0:N[0], 0:N[1]]
    dist = np.sqrt((xx - center[1]) ** 2 + (yy - center[0]) ** 2)
    inside = dist < PHANTOM_RADIUS_CELLS
    sound_speed_map = np.where(inside, cfg.MYOCARDIUM.sound_speed, cfg.WATER.sound_speed).astype(np.float32)
    density_map = np.where(inside, cfg.MYOCARDIUM.density, cfg.WATER.density).astype(np.float32)
    ssm = jnp.expand_dims(jnp.array(sound_speed_map), -1)
    dm = jnp.expand_dims(jnp.array(density_map), -1)
    return Medium(domain=domain, sound_speed=FourierSeries(ssm, domain),
                  density=FourierSeries(dm, domain))


_dummy_medium = build_medium_water_only()
_base_time_axis = TimeAxis.from_medium(_dummy_medium, cfl=cfg.CFL)
dt = _base_time_axis.dt

_c_min = min(cfg.WATER.sound_speed, cfg.MYOCARDIUM.sound_speed)
_straight_line_dist_cells = 2 * PROBE_RADIUS_CELLS  # tx -> rx, straight through center
_t_end_needed = (_straight_line_dist_cells * dx[0] / _c_min) * 1.3
time_axis = TimeAxis(dt=dt, t_end=_t_end_needed)
n_steps = int(time_axis.Nt)
t_arr = np.arange(n_steps) * dt


def toneburst(t):
    duration = cfg.N_CYCLES / cfg.F0_HZ
    sigma = duration / 6
    window = np.exp(-(t - duration / 2) ** 2 / (2 * sigma ** 2))
    return np.sin(2 * np.pi * cfg.F0_HZ * t) * window


_signal_template = jnp.array(toneburst(t_arr))[None, :]


def simulate_transmission(medium, theta_tx_deg):
    tx = probe_position(theta_tx_deg)
    rx = probe_position(theta_tx_deg + 180.0)
    sources = Sources(positions=([tx[0]], [tx[1]]), signals=_signal_template, dt=dt, domain=domain)

    @jit
    def run(m):
        return simulate_wave_propagation(m, time_axis, sources=sources)

    pressure = run(medium)
    trace = np.array(pressure.on_grid[:, rx[0], rx[1], 0])
    envelope = np.abs(hilbert(trace))
    arrival_idx = int(np.argmax(envelope))
    return t_arr[arrival_idx], envelope[arrival_idx], trace, envelope


if __name__ == "__main__":
    print(f"*** {labels.PENDING_SIGNOFF_BANNER} ***")
    print("PHASE 1 SCOUT: rotating-probe transmission tomography, STATIC 2D phantom.")
    print(f"  probe radius={PROBE_RADIUS_CELLS} cells, phantom radius={PHANTOM_RADIUS_CELLS} cells, "
          f"{N_ANGLES} angles around a full 360-degree perimeter")
    print(f"  attenuation NOT modeled in this test -- amplitude reflects geometric/transmission "
          f"effects only, see module docstring")

    thetas = np.linspace(0, 360, N_ANGLES, endpoint=False)

    medium_water = build_medium_water_only()
    medium_phantom = build_medium_static_phantom()

    print("\n=== Sweeping water-only control (homogeneous, no phantom) ===")
    water_times, water_amps = [], []
    for theta in thetas:
        t_arrival, amp, _, _ = simulate_transmission(medium_water, theta)
        water_times.append(t_arrival)
        water_amps.append(amp)
    water_times = np.array(water_times)
    water_amps = np.array(water_amps)

    print("=== Sweeping static phantom (centered myocardium-like disk) ===")
    phantom_times, phantom_amps = [], []
    for theta in thetas:
        t_arrival, amp, _, _ = simulate_transmission(medium_phantom, theta)
        phantom_times.append(t_arrival)
        phantom_amps.append(amp)
    phantom_times = np.array(phantom_times)
    phantom_amps = np.array(phantom_amps)

    # --- Sanity check 1: water-only control should be flat (rotational symmetry) ---
    water_std_ns = water_times.std() * 1e9
    print(f"\n--- Sanity check 1: water-only arrival time vs. angle ---")
    print(f"  mean={water_times.mean()*1e6:.4f}us, std={water_std_ns:.2f}ns across {N_ANGLES} angles")
    print(f"  (should be ~0ns std by pure rotational symmetry -- any real spread flags a geometry bug)")

    # --- Sanity check 2: phantom arrival time should ALSO be flat, shifted from water-only ---
    phantom_std_ns = phantom_times.std() * 1e9
    measured_excess_delay_ns = (phantom_times.mean() - water_times.mean()) * 1e9

    tissue_chord_m = 2 * PHANTOM_RADIUS_CELLS * dx[0]
    # excess delay = mixed_time - water_only_time = chord/c_tissue - chord/c_water
    # (a FASTER tissue than water gives a NEGATIVE excess delay -- arrives sooner)
    predicted_excess_delay_ns = tissue_chord_m * (1.0 / cfg.MYOCARDIUM.sound_speed - 1.0 / cfg.WATER.sound_speed) * 1e9

    print(f"\n--- Sanity check 2: phantom arrival time vs. angle, and excess delay vs. prediction ---")
    print(f"  phantom mean={phantom_times.mean()*1e6:.4f}us, std={phantom_std_ns:.2f}ns across {N_ANGLES} angles")
    print(f"  (should ALSO be ~flat -- same tissue chord length at every angle for a centered circular phantom)")
    print(f"  measured excess delay (phantom - water_only) = {measured_excess_delay_ns:+.2f}ns")
    print(f"  ANALYTIC PREDICTION (straight-ray, chord={tissue_chord_m*1e3:.2f}mm, "
          f"c_water={cfg.WATER.sound_speed}, c_myocardium={cfg.MYOCARDIUM.sound_speed}) = "
          f"{predicted_excess_delay_ns:+.2f}ns")
    rel_err = abs(measured_excess_delay_ns - predicted_excess_delay_ns) / abs(predicted_excess_delay_ns)
    print(f"  relative error = {rel_err*100:.2f}% "
          f"({'MATCHES (geometry/timing sane)' if rel_err < 0.10 else 'DOES NOT MATCH -- investigate before trusting anything further'})")

    # --- Amplitude (NOT a validated attenuation map -- see docstring) ---
    print(f"\n--- Amplitude (informational only, NOT attenuation -- not modeled in this test) ---")
    print(f"  water-only: mean={water_amps.mean():.4g}, std={water_amps.std():.4g}")
    print(f"  phantom:    mean={phantom_amps.mean():.4g}, std={phantom_amps.std():.4g}")

    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    axes[0].plot(thetas, (water_times - water_times.mean()) * 1e9, "o-", label="water-only (control)")
    axes[0].plot(thetas, (phantom_times - water_times.mean()) * 1e9, "s-", label="static phantom")
    axes[0].axhline(predicted_excess_delay_ns, color="k", linestyle="--",
                     label=f"predicted phantom excess delay ({predicted_excess_delay_ns:+.1f}ns)")
    axes[0].set_xlabel("transmitter angle (deg)")
    axes[0].set_ylabel("arrival time relative to water-only mean (ns)")
    axes[0].set_title("Transmission arrival time vs. rotating-probe angle\n(both curves should be FLAT)")
    axes[0].legend(fontsize=8)

    axes[1].plot(thetas, water_amps, "o-", label="water-only (control)")
    axes[1].plot(thetas, phantom_amps, "s-", label="static phantom")
    axes[1].set_xlabel("transmitter angle (deg)")
    axes[1].set_ylabel("received peak envelope amplitude (a.u.)")
    axes[1].set_title("Transmission amplitude vs. angle\n(informational -- attenuation not modeled)")
    axes[1].legend(fontsize=8)

    fig.suptitle("Phase 1 scout: rotating-probe transmission tomography, STATIC centered phantom, 2D")
    labels.add_banner(fig)
    plt.tight_layout()
    os.makedirs("results/figures", exist_ok=True)
    out_fig = "results/figures/phase1_rotating_transmission_scout.png"
    plt.savefig(out_fig, dpi=140)
    print(f"\nSaved {out_fig}")
