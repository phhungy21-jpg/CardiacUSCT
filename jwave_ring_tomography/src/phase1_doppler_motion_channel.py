"""Tests the Doppler/MOTION channel -- ranked #1 for blood-vs-myocardium
discrimination in this session's own channel ranking, and never touched
until now (everything else this session used single, static frames).

SCOPE, stated honestly upfront: true continuous-wave/pulsed Doppler
measures a frequency shift accumulated WITHIN a single pulse from a
continuously-moving scatterer. Modeling that properly would require
jWave to simulate a medium that is actually moving DURING wave
propagation -- nothing in this project's infrastructure does that (every
simulation uses one static, frozen medium per run). What this script
tests instead, using ALREADY-SIMULATED data (zero new jWave simulation):
the DISCRETE analog -- does the inner-boundary echo's arrival TIME shift
between ADJACENT cardiac phases in a way that tracks the real, known
anatomical motion? This is the core mechanism behind real M-mode/tissue-
tracking imaging (frame-to-frame displacement tracking), not literally
continuous-wave Doppler, but a legitimate, directly-testable proxy for
"does motion information show up in the acoustic signal at all."

Uses an ORACLE-assisted peak match (the analytically-predicted true
inner-echo time, from the ALREADY-KNOWN true contours) to identify which
observed peak is the genuine inner echo at each phase/angle -- this
project's own established discipline of validating a mechanism exists
BEFORE building a blind detector for it (e.g., the circular positive
control before real anatomy, runs -12 onward).

Reuses `results/beating_patient001/023_reflection_traces.npz` (already
simulated) and the cached water-only baseline -- no new jWave simulation.
"""

import numpy as np
from scipy.signal import correlate, hilbert, find_peaks

from phase1_reflection_channel_scout import polar_resample, r_at_theta
from phase1_matched_filter_echo_extraction import _lag_t_arr, _template, PEAK_PROMINENCE_FRACTION
from phase1_rotating_transmission_scout import center, PROBE_RADIUS_CELLS, dx
import phase2_config as cfg
import labels

from matplotlib import pyplot as plt
import os

ANGLE_INDICES = list(range(0, 36, 36 // 18))
_nonneg = _lag_t_arr >= 0
MATCH_WINDOW_S = 3e-7

# Real cardiac-cycle timing (informational, for context -- not used in the
# correlation test itself, which only needs RELATIVE motion between phases):
# a typical resting heart rate ~60-75 bpm -> cycle ~0.8-1.0s; the 8 phases
# span one full cycle (ED->ES->ED), so adjacent phases are ~0.1-0.125s apart.


def matched_filter_envelope(trace):
    correlated = correlate(trace, _template, mode="full")
    return np.abs(hilbert(correlated))


def predicted_inner_time_two_leg(r_outer_cells, r_inner_cells):
    leg_water_m = (PROBE_RADIUS_CELLS - r_outer_cells) * dx[0]
    leg_myo_m = (r_outer_cells - r_inner_cells) * dx[0]
    return 2 * (leg_water_m / cfg.WATER.sound_speed + leg_myo_m / cfg.MYOCARDIUM.sound_speed)


def oracle_match_time(env_water, env_phantom, t_predicted, lag_t_local, nonneg_local):
    thresh = max(env_water[nonneg_local].max() * 3.0, env_phantom[nonneg_local].max() * PEAK_PROMINENCE_FRACTION)
    peak_idx, _ = find_peaks(env_phantom[nonneg_local], height=thresh)
    if len(peak_idx) == 0:
        return np.nan
    times = lag_t_local[nonneg_local][peak_idx]
    d = np.abs(times - t_predicted)
    best = np.argmin(d)
    if d[best] > MATCH_WINDOW_S:
        return np.nan
    return times[best]


def analyze_patient(patient_id, water_env):
    d = np.load(f"results/beating_{patient_id}_reflection_traces.npz", allow_pickle=True)
    traces = d["traces"]
    thetas = d["thetas"]
    n_phases, n_angles, _ = traces.shape

    r_in_all = np.zeros((n_phases, n_angles))
    r_out_all = np.zeros((n_phases, n_angles))
    t_obs_all = np.full((n_phases, n_angles), np.nan)

    for phase_idx in range(n_phases):
        inner_dom = d["inner_contours_dom"][phase_idx]
        outer_dom = d["outer_contours_dom"][phase_idx]
        ext_theta_in, ext_r_in = polar_resample(inner_dom, center)
        ext_theta_out, ext_r_out = polar_resample(outer_dom, center)
        for angle_idx, theta in enumerate(thetas):
            r_in = r_at_theta(theta, ext_theta_in, ext_r_in)
            r_out = r_at_theta(theta, ext_theta_out, ext_r_out)
            r_in_all[phase_idx, angle_idx] = r_in
            r_out_all[phase_idx, angle_idx] = r_out

            trace = traces[phase_idx, angle_idx]
            env = matched_filter_envelope(trace)
            min_len = min(len(env), len(water_env[angle_idx]), len(_lag_t_arr))
            lag_t_local = _lag_t_arr[:min_len]
            nonneg_local = lag_t_local >= 0
            t_pred = predicted_inner_time_two_leg(r_out, r_in)
            t_obs = oracle_match_time(water_env[angle_idx][:min_len], env[:min_len], t_pred,
                                       lag_t_local, nonneg_local)
            t_obs_all[phase_idx, angle_idx] = t_obs

    # frame-to-frame (adjacent phase) differences
    dr_true_mm, dt_obs_s = [], []
    for phase_idx in range(n_phases - 1):
        for angle_idx in range(n_angles):
            t0, t1 = t_obs_all[phase_idx, angle_idx], t_obs_all[phase_idx + 1, angle_idx]
            if np.isnan(t0) or np.isnan(t1):
                continue
            dr = (r_in_all[phase_idx + 1, angle_idx] - r_in_all[phase_idx, angle_idx]) * cfg.DX_M * 1e3
            dt = t1 - t0
            dr_true_mm.append(dr)
            dt_obs_s.append(dt)

    return np.array(dr_true_mm), np.array(dt_obs_s), t_obs_all


if __name__ == "__main__":
    print(f"*** {labels.PENDING_SIGNOFF_BANNER} ***")
    print(f"*** {labels.TOY_EXACT_GT_CAPTION} ***")
    print("DOPPLER/MOTION CHANNEL TEST (discrete/M-mode proxy): does the inner-boundary echo's "
          "arrival time shift between adjacent cardiac phases track the TRUE known anatomical "
          "motion? Oracle-assisted peak match (validating the mechanism exists before building "
          "a blind detector, per this project's established discipline). No new jWave simulation "
          "-- reuses cached beating-heart traces (both patients) + cached water baseline.")

    d_water = np.load("results/patient023_reflection_raw_traces.npz")
    water_traces = d_water["water_traces"][ANGLE_INDICES]
    water_env = np.array([matched_filter_envelope(tr) for tr in water_traces])

    fig, axes = plt.subplots(1, 2, figsize=(13, 5.5))
    all_dr, all_dt = [], []
    for ax, patient_id in zip(axes, ["patient001", "patient023"]):
        dr_true_mm, dt_obs_s, t_obs_all = analyze_patient(patient_id, water_env)
        n_valid = len(dr_true_mm)
        n_total = 7 * 18
        if n_valid >= 3:
            corr = np.corrcoef(dr_true_mm, dt_obs_s)[0, 1]
        else:
            corr = float("nan")
        print(f"\n--- {patient_id} ---")
        print(f"  valid frame-to-frame pairs: {n_valid}/{n_total}")
        print(f"  true radial motion range: [{dr_true_mm.min():.3f}, {dr_true_mm.max():.3f}] mm")
        print(f"  observed echo-time shift range: [{dt_obs_s.min()*1e9:.1f}, {dt_obs_s.max()*1e9:.1f}] ns")
        print(f"  correlation(true radial motion, observed echo-time shift): {corr:.3f}")
        expected_slope_ns_per_mm = -2.0 / cfg.MYOCARDIUM.sound_speed * 1e-3 * 1e9
        print(f"  (expected slope from M-mode physics, -2/c_myo: {expected_slope_ns_per_mm:.3f} ns/mm -- "
              f"NEGATIVE slope expected: the probe sits OUTSIDE at r=PROBE_RADIUS, so a "
              f"shrinking inner radius (contraction, dr<0) moves the boundary AWAY from the "
              f"probe, lengthening the round trip -> LATER echo (dt>0))")

        all_dr.append(dr_true_mm)
        all_dt.append(dt_obs_s)

        ax.scatter(dr_true_mm, dt_obs_s * 1e9, alpha=0.6)
        if n_valid >= 2:
            fit = np.polyfit(dr_true_mm, dt_obs_s * 1e9, 1)
            xs = np.linspace(dr_true_mm.min(), dr_true_mm.max(), 10)
            ax.plot(xs, np.polyval(fit, xs), "r--", label=f"fit slope={fit[0]:.2f} ns/mm")
        ax.axhline(0, color="gray", linewidth=0.6)
        ax.axvline(0, color="gray", linewidth=0.6)
        ax.set_xlabel("true radial motion, phase(i+1)-phase(i) (mm)")
        ax.set_ylabel("observed echo-time shift (ns)")
        ax.set_title(f"{patient_id}: corr={corr:.3f}, n={n_valid}/{n_total}")
        ax.legend(fontsize=8)

    dr_pooled = np.concatenate(all_dr)
    dt_pooled = np.concatenate(all_dt)
    corr_pooled = np.corrcoef(dr_pooled, dt_pooled)[0, 1]
    print(f"\n--- POOLED (both patients) ---")
    print(f"  n={len(dr_pooled)}, correlation={corr_pooled:.3f}")
    if abs(corr_pooled) > 0.5:
        print("  -> STRONG: motion tracking via echo-timing shift is a real, usable channel.")
    elif abs(corr_pooled) > 0.2:
        print("  -> MODERATE: real but noisy signal.")
    else:
        print("  -> WEAK/NONE at this setup.")

    fig.suptitle("Doppler/motion channel (discrete M-mode proxy): does echo timing track true motion?")
    labels.add_banner(fig)
    plt.tight_layout()
    os.makedirs("results/figures", exist_ok=True)
    plt.savefig("results/figures/phase1_doppler_motion_channel.png", dpi=140)
    print("\nSaved results/figures/phase1_doppler_motion_channel.png")
