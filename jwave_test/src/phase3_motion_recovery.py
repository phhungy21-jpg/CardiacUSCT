"""Phase 3.2 — simulate -> recover on the toy moving ring phantom.

MOTION-INJECTION METHOD (the item flagged as needing a deliberate decision
in ../PHASE2_TO_PHASE3_DIAGNOSTIC_HANDOFF.md, item 5):

Each transmit is simulated as an independent, fully "frozen" static medium
at the phantom's instantaneous geometry for that frame — a fresh Medium is
built from scratch per frame, with zero initial velocity/pressure
carried over from any other frame. This is not a numerical shortcut: it is
the standard "frozen scene" approximation used throughout real pulse-echo
ultrasound, justified because a single pulse's transit time (microseconds,
c*T_pulse ~ mm) is many orders of magnitude shorter than the cardiac motion
timescale (hundreds of ms, cm-scale excursions) -- the medium genuinely
does not move meaningfully during one transmit. Because each frame's
simulation is fully independent (no shared state), there is no mechanism
for a discretization "jump" between frames to leak into within-transmit
physics. The remaining risk the diagnostic handoff named -- a
motion-CORRELATED artifact inflating recovery accuracy -- is checked
explicitly below via a null test (Section 3): a zero-motion phantom run
through the IDENTICAL per-frame rebuild-and-transmit pipeline. If recovery
reports non-zero "motion" there, the pipeline itself is suspect and no
Phase 3 recovery number should be trusted.

Recovery method: a simple two-element pitch-catch (source + receiver 1mm
apart, both near the top/anterior edge) pulse-echo range measurement. The
first strong echo after the direct-arrival window is attributed to the
near (outer) myocardial boundary; its round-trip time converts to a range,
and range converts to the recovered outer-radius via the known, fixed
transducer-to-phantom-center distance. This directly tests whether the
loop can recover a KNOWN prescribed boundary position from simulated
pulse-echo data -- exactly protocol 3.2's requirement -- without needing
a full image-formation pipeline.

STATUS: PENDING ACOUSTIC-PHYSICS SIGNOFF. This is a toy, non-physical-
realism result -- attenuation is inert, source amplitude is uncalibrated,
staircasing is untested at real-anatomy resolution (all deferred to the
Phase 3->4 hard gate, see ../PHASE2_TO_PHASE3_DIAGNOSTIC_HANDOFF.md). Its
RMSE numbers are against an EXACTLY prescribed toy ground truth, not
Phase I's registration-derived motion -- no ground-truth-motion floor
applies here (see src/labels.py TOY_EXACT_GT_CAPTION). That floor becomes
relevant starting at Phase 4.
"""

import os
import time

import numpy as np
from scipy.signal import hilbert
from jax import numpy as jnp
from jax import jit
from matplotlib import pyplot as plt

from jwave import FourierSeries
from jwave.geometry import Domain, Medium, TimeAxis, Sources

from jwave.acoustics import simulate_wave_propagation

import phase2_config as cfg
import phase3_config as p3cfg
import labels

N = (300, 300)
dx = (cfg.DX_M, cfg.DX_M)
domain = Domain(N, dx)
center = (N[0] // 2, N[1] // 2)

array_y = 30  # 3mm from top edge, matches phase2_forward_model.py
src_x = center[1] - 5   # source 0.5mm left of center
rcv_x = center[1] + 5   # receiver 0.5mm right of center, 1mm pitch-catch spacing

vertical_dist_mm = (center[0] - array_y) * dx[0] * 1e3  # transducer -> phantom center


def build_medium(lv_radius_cells, wall_thickness_cells):
    yy, xx = np.mgrid[0:N[0], 0:N[1]]
    dist = np.sqrt((xx - center[1]) ** 2 + (yy - center[0]) ** 2)
    label_map = np.zeros(N, dtype=int)
    label_map[dist < lv_radius_cells + wall_thickness_cells] = 2  # myocardium
    label_map[dist < lv_radius_cells] = 3                          # LV cavity

    sound_speed_map = np.zeros(N, dtype=np.float32)
    density_map = np.zeros(N, dtype=np.float32)
    for label, tissue in cfg.ACDC_LABEL_TO_TISSUE.items():
        mask = label_map == label
        sound_speed_map[mask] = tissue.sound_speed
        density_map[mask] = tissue.density

    sound_speed_map = jnp.expand_dims(jnp.array(sound_speed_map), -1)
    density_map = jnp.expand_dims(jnp.array(density_map), -1)
    return Medium(domain=domain,
                  sound_speed=FourierSeries(sound_speed_map, domain),
                  density=FourierSeries(density_map, domain))


# Fixed timing: shortened to cover transmit -> boundary -> back with margin,
# not the full domain crossing (keeps every frame's compute cost bounded).
_dummy_medium = build_medium(p3cfg.LV_RADIUS_ED_CELLS, p3cfg.WALL_THICKNESS_CELLS)
_base_time_axis = TimeAxis.from_medium(_dummy_medium, cfl=cfg.CFL)
dt = _base_time_axis.dt
t_end = 0.35 * _base_time_axis.t_end
time_axis = TimeAxis(dt=dt, t_end=t_end)
n_steps = int(np.round(t_end / dt))
t_arr = np.arange(n_steps) * dt


def toneburst(t, t_delay=0.0):
    """Gaussian-windowed toneburst, WITHOUT a hard truncation.

    Bug caught during Phase 3 validation: the original version here (and
    in ../jwave/toy_2d_array_source.py / phase2_forward_model.py) multiplied
    the Gaussian window by a hard (tau>=0)*(tau<=duration) indicator. With
    sigma=duration/4, the Gaussian is still ~13.5% of peak at those edges
    (2 sigma out), so the indicator created a genuine discontinuity in the
    source signal -- which excited persistent broadband numerical ringing
    that swamped the real (much weaker) reflected echo, causing
    peak-detection to lock onto ringing instead of the true reflection
    (recovered range was identical across all frames regardless of
    geometry -- the tell that something was wrong). Fix: use a tighter
    sigma (duration/6, ~1.1% of peak at 3-sigma) and no hard truncation --
    the window decays smoothly to near-zero instead of jumping.
    """
    tau = t - t_delay
    duration = cfg.N_CYCLES / cfg.F0_HZ
    sigma = duration / 6
    window = np.exp(-(tau - duration / 2) ** 2 / (2 * sigma ** 2))
    return np.sin(2 * np.pi * cfg.F0_HZ * tau) * window


_signal = jnp.array(toneburst(t_arr))[None, :]  # single source element
sources = Sources(positions=([src_x], [array_y]), signals=_signal, dt=dt,
                   domain=domain)


@jit
def run(medium):
    return simulate_wave_propagation(medium, time_axis, sources=sources)


def simulate_frame(lv_radius_cells):
    medium = build_medium(lv_radius_cells, p3cfg.WALL_THICKNESS_CELLS)
    pressure = run(medium)
    field = pressure.on_grid[..., 0]
    trace = np.array(field[:, rcv_x, array_y])
    return trace


# Direct-arrival exclusion window: source-receiver are 1mm apart (transit
# ~0.63us) but the toneburst itself lasts n_cycles/f0 ~1.2us, so the direct
# pulse's tail is still present at the receiver until ~0.63+1.2=1.83us.
# Bug caught during Phase 3 validation (see LOG.md): an earlier cutoff of
# 1.5us was inside that tail, causing peak-detection to lock onto the
# direct pulse's tail instead of the real reflection -- recovered range
# was IDENTICAL across all frames regardless of true geometry, a red flag
# caught by comparing against ground truth before trusting the result.
DIRECT_EXCLUDE_S = 3.0e-6


ECHO_THRESHOLD_FRAC = 0.3  # fraction of the post-exclusion envelope peak


def recover_range_mm(trace, c_ref):
    """Envelope-based FIRST-threshold-crossing detection.

    Bug caught during Phase 3 validation: simple argmax-of-amplitude
    picked whichever arrival was LARGEST in the post-exclusion window, not
    the FIRST (nearest) one. With two similarly-weak interfaces
    (chest-wall/myocardium R~0.0035, blood/myocardium R~0.0025, both from
    ../jwave/LOG.md run -05's cited values) a later multipath/reverberation
    was sometimes larger than the true near-boundary echo, causing the
    recovered range to lock onto the wrong (and non-monotonic, sometimes
    geometry-independent) feature. Fix: use the analytic envelope's FIRST
    threshold crossing, matching how real pulse-echo systems detect range
    (first-break / leading-edge detection), not peak amplitude.
    """
    mask = t_arr > DIRECT_EXCLUDE_S
    sub = trace[mask]
    envelope = np.abs(hilbert(sub))
    threshold = ECHO_THRESHOLD_FRAC * envelope.max()
    idx_local = np.argmax(envelope > threshold)  # first True
    t_echo = t_arr[mask][idx_local]
    return t_echo * c_ref / 2 * 1e3  # meters -> mm


def recover_outer_radius_mm(trace, c_ref):
    return vertical_dist_mm - recover_range_mm(trace, c_ref)


def run_sweep(radii_cells, label):
    traces = []
    t0 = time.time()
    for i, r in enumerate(radii_cells):
        traces.append(simulate_frame(r))
        print(f"  [{label}] frame {i+1}/{len(radii_cells)} "
              f"(lv_radius={r} cells) done, {time.time()-t0:.1f}s elapsed")
    return traces


if __name__ == "__main__":
    print(f"*** {labels.PENDING_SIGNOFF_BANNER} ***")
    print(f"*** {labels.TOY_EXACT_GT_CAPTION} ***")
    c_ref = cfg.CHEST_WALL_PROXY.sound_speed
    phases = np.linspace(0, 1, p3cfg.N_FRAMES)
    lv_radii_motion = [p3cfg.lv_radius_at_phase(p) for p in phases]
    ground_truth_outer_radius_mm = [
        (r + p3cfg.WALL_THICKNESS_CELLS) * cfg.DX_M * 1e3 for r in lv_radii_motion
    ]

    print("=== Motion sweep ===")
    traces_motion = run_sweep(lv_radii_motion, "motion")

    print("=== Null test (zero motion, identical pipeline) ===")
    lv_radii_null = [p3cfg.LV_RADIUS_ED_CELLS] * p3cfg.N_FRAMES
    traces_null = run_sweep(lv_radii_null, "null")
    ground_truth_null_mm = [
        (r + p3cfg.WALL_THICKNESS_CELLS) * cfg.DX_M * 1e3 for r in lv_radii_null
    ]

    rng = np.random.default_rng(0)  # fixed seed, per CLAUDE.md

    def recover_with_noise(traces, noise_level):
        results = []
        for trace in traces:
            peak = np.max(np.abs(trace))
            noisy = trace + rng.normal(0, noise_level * peak, size=trace.shape)
            results.append(recover_outer_radius_mm(noisy, c_ref))
        return np.array(results)

    # --- Motion sweep results across noise levels ---
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(phases, ground_truth_outer_radius_mm, "k--", label="ground truth")
    baseline_pred = np.mean(ground_truth_outer_radius_mm)
    baseline_rmse = np.sqrt(np.mean(
        (np.array(ground_truth_outer_radius_mm) - baseline_pred) ** 2))
    for noise_level in p3cfg.NOISE_LEVELS:
        recovered = recover_with_noise(traces_motion, noise_level)
        rmse = np.sqrt(np.mean(
            (recovered - np.array(ground_truth_outer_radius_mm)) ** 2))
        ax.plot(phases, recovered, "o-",
                label=f"recovered, noise={noise_level} (RMSE={rmse:.3f}mm)")
        print(f"Motion sweep, noise={noise_level}: RMSE={rmse:.4f}mm "
              f"(naive constant-baseline RMSE={baseline_rmse:.4f}mm)")
    ax.axhline(baseline_pred, color="gray", linestyle=":",
               label=f"naive constant baseline (RMSE={baseline_rmse:.3f}mm)")
    ax.set_xlabel("cardiac phase (0=ED, 0.5=ES, 1=ED)")
    ax.set_ylabel("outer myocardial radius (mm)")
    ax.set_title("Phase 3: recovered vs. prescribed motion\n"
                 "(TOY: exactly-prescribed ground truth, no motion-floor applies)")
    ax.legend(fontsize=8)
    plt.tight_layout(rect=[0, 0.06, 1, 1])
    labels.add_banner(fig)
    os.makedirs("results/figures", exist_ok=True)
    plt.savefig("results/figures/phase3_motion_recovery.png", dpi=150)
    print("Saved results/figures/phase3_motion_recovery.png")

    # --- Null test ---
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(phases, ground_truth_null_mm, "k--", label="ground truth (zero motion)")
    for noise_level in [0.0, 0.05]:
        recovered_null = recover_with_noise(traces_null, noise_level)
        drift_std = np.std(recovered_null)
        ax.plot(phases, recovered_null, "o-",
                label=f"recovered, noise={noise_level} (std={drift_std:.4f}mm)")
        print(f"NULL TEST, noise={noise_level}: recovered outer-radius std "
              f"across frames = {drift_std:.4f}mm "
              f"(motion signal amplitude = "
              f"{(max(ground_truth_outer_radius_mm)-min(ground_truth_outer_radius_mm)):.3f}mm)")
    ax.set_xlabel("frame index (as phase, all frames identical geometry)")
    ax.set_ylabel("outer myocardial radius (mm)")
    ax.set_title("Phase 3 NULL TEST: zero-motion phantom, same pipeline")
    ax.legend(fontsize=8)
    plt.tight_layout(rect=[0, 0.06, 1, 1])
    labels.add_banner(fig)
    plt.savefig("results/figures/phase3_null_test.png", dpi=150)
    print("Saved results/figures/phase3_null_test.png")
