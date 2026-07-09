"""Tests whether run -13's "extra peaks before the true inner echo"
are genuine distinct physical bounces, or just ringing/sidelobes of
the strong outer reflection's own waveform being mistaken for separate
echoes by raw envelope-peak detection.

Per user's proposed direction (a "complete-picture inverse equation"
linking every echo in a single pulse-fire back to the specific surface
and bounce count that produced it): that kind of path-inversion
reasoning REQUIRES a clean, reliable set of (echo time, amplitude)
pairs first -- you cannot reason about which physical bounce produced
which echo if some of your "echoes" are actually ringing artifacts of
one real echo's own pulse shape, not separate physical events. Matched
filtering (cross-correlating the received trace against the KNOWN
transmitted toneburst, standard radar/sonar pulse compression) is the
necessary first step: it collapses each GENUINE echo into one sharp
peak and suppresses a single echo's own ringing tail (which does not
correlate well with a full copy of the transmit pulse), unlike raw
envelope detection.

Validates the matched-filter method against the ALREADY-KNOWN-GOOD
outer boundary timing before trusting it on the harder inner-boundary
question (same "verify before trust on a known case" discipline as
every other run in this project).
"""

import numpy as np
from scipy.signal import hilbert, find_peaks, correlate

from phase1_reflection_channel_scout import (
    thetas, predicted_reflection_times, pitch_catch_positions, _ENVELOPE_GROUP_DELAY_S,
    DIRECT_EXCLUDE_MARGIN_S,
)
from phase1_circular_positive_control import build_medium_concentric_circles, build_medium_water_only, R_OUTER, R_INNER
from phase1_rotating_transmission_scout import (
    PROBE_RADIUS_CELLS, dx, time_axis, dt, t_arr, _signal_template, domain,
)
import phase2_config as cfg
import labels

from jax import jit
from jwave.acoustics import simulate_wave_propagation
from jwave.geometry import Sources

from matplotlib import pyplot as plt
import os

PEAK_PROMINENCE_FRACTION = 0.05
MATCH_WINDOW_S = 3e-7
_template = np.array(_signal_template[0])  # the known transmitted toneburst, as a plain array


def simulate_pitch_catch_raw(medium, theta_deg):
    """Same as simulate_pitch_catch, but returns the RAW trace (not just
    its envelope) so matched filtering can be applied."""
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


_n = len(t_arr)
_dt_val = t_arr[1] - t_arr[0]
_lag_t_arr = np.arange(-(_n - 1), _n) * _dt_val  # correct index->time mapping for mode="full"


def matched_filter_output(trace):
    """CORRECTED (see LOG.md run -14): mode='full', not 'same' -- 'same'
    mode's raw index does NOT correspond to actual delay time, confirmed
    via a synthetic known-delay self-test before trusting this. Returns
    (envelope, lag_time_array) since the output's time axis is NOT
    t_arr (it spans negative lags too, from the correlation itself)."""
    correlated = correlate(trace, _template, mode="full")
    return np.abs(hilbert(correlated)), _lag_t_arr


def time_to_radius_matched_filter(t):
    """CORRECTED (see LOG.md run -14): matched filtering against the
    FULL known template shape recovers the TRUE ballistic round-trip
    delay directly -- NO group-delay subtraction needed (that
    correction is specific to raw envelope-peak detection, which peaks
    at the pulse's CENTER, not its onset; matched filtering against the
    complete pulse shape has no such offset, confirmed via the
    synthetic self-test)."""
    one_way_m = cfg.WATER.sound_speed * t / 2.0
    return PROBE_RADIUS_CELLS - one_way_m / dx[0]


def predicted_times_no_group_delay(theta_deg, r_outer, r_inner):
    c_water, c_myo = cfg.WATER.sound_speed, cfg.MYOCARDIUM.sound_speed
    leg_to_outer_m = (PROBE_RADIUS_CELLS - r_outer) * dx[0]
    leg_outer_to_inner_m = (r_outer - r_inner) * dx[0]
    t_outer = 2 * (leg_to_outer_m / c_water)
    t_inner = 2 * (leg_to_outer_m / c_water + leg_outer_to_inner_m / c_myo)
    return t_outer, t_inner


if __name__ == "__main__":
    print(f"*** {labels.PENDING_SIGNOFF_BANNER} ***")
    print(f"*** {labels.TOY_EXACT_GT_CAPTION} ***")
    print("MATCHED FILTER ECHO EXTRACTION: testing whether run -13's extra peaks "
          "before the true inner echo are genuine bounces or ringing/sidelobe artifacts "
          "of raw envelope detection. Validates against the known-good outer boundary first.")
    print("  compute estimate: 36 angles x 2 media = 72 forward sims (same as prior runs) "
          "-- ~15-20 minutes based on that precedent")

    medium_water = build_medium_water_only()
    medium_phantom = build_medium_concentric_circles()

    print("\n=== Simulating water-only control, pitch-catch at 36 angles ===")
    water_traces = [simulate_pitch_catch_raw(medium_water, th) for th in thetas]
    print("=== Simulating concentric-circle phantom, pitch-catch at 36 angles ===")
    phantom_traces = [simulate_pitch_catch_raw(medium_phantom, th) for th in thetas]

    water_mf = [matched_filter_output(tr) for tr in water_traces]  # each: (envelope, lag_t_arr)
    phantom_mf = [matched_filter_output(tr) for tr in phantom_traces]

    # only non-negative lags are physically possible (an echo can't arrive before t=0)
    _nonneg = _lag_t_arr >= 0

    # --- Sanity check: does matched-filter peak position for the OUTER
    # boundary still match its already-validated predicted time? ---
    outer_mf_errs = []
    for i, theta in enumerate(thetas):
        t_outer_true, _ = predicted_times_no_group_delay(theta, R_OUTER, R_INNER)
        env_w, _ = water_mf[i]
        env_p, _ = phantom_mf[i]
        thresh = max(env_w[_nonneg].max() * 3.0, env_p[_nonneg].max() * PEAK_PROMINENCE_FRACTION)
        peak_idx, _ = find_peaks(env_p[_nonneg], height=thresh)
        if len(peak_idx) == 0:
            continue
        times = np.sort(_lag_t_arr[_nonneg][peak_idx])
        r0 = time_to_radius_matched_filter(times[0])
        outer_mf_errs.append(r0 - R_OUTER)
    print(f"\n--- Sanity check: matched-filter outer-boundary timing (should be near 0, "
          f"like run -12's -0.17 cells) ---")
    print(f"  mean error={np.mean(outer_mf_errs):+.2f} cells (n={len(outer_mf_errs)}/{len(thetas)})")

    # --- Main test: does the number of peaks BEFORE the true inner echo drop? ---
    matched_positions = []
    n_total_peaks = []
    naive_vs_matched = []
    for i, theta in enumerate(thetas):
        env_w, _ = water_mf[i]
        env_p, _ = phantom_mf[i]
        thresh = max(env_w[_nonneg].max() * 3.0, env_p[_nonneg].max() * PEAK_PROMINENCE_FRACTION)
        peak_idx, _ = find_peaks(env_p[_nonneg], height=thresh)
        order = np.argsort(_lag_t_arr[_nonneg][peak_idx])
        peak_times = _lag_t_arr[_nonneg][peak_idx[order]]
        n_total_peaks.append(len(peak_times))

        _, t_inner_true = predicted_times_no_group_delay(theta, R_OUTER, R_INNER)
        if len(peak_times) == 0:
            matched_positions.append(None)
            naive_vs_matched.append((None, None))
            continue
        dists = np.abs(peak_times - t_inner_true)
        match_idx = int(np.argmin(dists))
        if dists[match_idx] > MATCH_WINDOW_S:
            match_idx = None
        matched_positions.append(match_idx)

        naive_r = time_to_radius_matched_filter(peak_times[1]) if len(peak_times) >= 2 else None
        matched_r = time_to_radius_matched_filter(peak_times[match_idx]) if match_idx is not None else None
        naive_vs_matched.append((naive_r, matched_r))

    valid_matches = [m for m in matched_positions if m is not None]
    print(f"\n--- Matched-filter: total peaks found per angle ---")
    print(f"  mean={np.mean(n_total_peaks):.1f}, (run -13's raw envelope: typically 4+ before/at true echo)")
    print(f"\n--- Which chronological peak position matches the TRUE predicted inner echo (matched filter) ---")
    print(f"  matched at SOME peak in {len(valid_matches)}/{len(thetas)} angles "
          f"(run -13 raw envelope: 12/36)")
    if valid_matches:
        from collections import Counter
        counts = Counter(valid_matches)
        for pos in sorted(counts):
            print(f"    position #{pos+1}: {counts[pos]}/{len(valid_matches)} angles")

    naive_radii = np.array([r for r, m in naive_vs_matched if r is not None])
    matched_radii = np.array([m for r, m in naive_vs_matched if m is not None])
    print(f"\n--- Radius accuracy (matched filter) ---")
    print(f"  naive (position #2): mean error={np.mean(naive_radii - R_INNER):+.2f} cells "
          f"(run -13 raw envelope: +11.79)")
    print(f"  prediction-matched peak: mean error={np.mean(matched_radii - R_INNER):+.2f} cells "
          f"(run -13 raw envelope: +1.31)")

    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    rep_idx = 0
    env_p_rep, _ = phantom_mf[rep_idx]
    axes[0].plot(_lag_t_arr[_nonneg] * 1e6, env_p_rep[_nonneg] / env_p_rep[_nonneg].max(),
                 label="matched filter output", color="C1")
    axes[0].plot(t_arr * 1e6, np.abs(hilbert(phantom_traces[rep_idx])) / np.abs(hilbert(phantom_traces[rep_idx])).max(),
                 label="raw envelope (run -12/-13's method)", color="C0", alpha=0.6)
    t_outer_rep, t_inner_rep = predicted_times_no_group_delay(thetas[rep_idx], R_OUTER, R_INNER)
    axes[0].axvline(t_outer_rep * 1e6, color="c", linestyle="--", label="predicted outer")
    axes[0].axvline(t_inner_rep * 1e6, color="lime", linestyle="--", label="predicted inner")
    axes[0].set_xlim(0, t_arr.max() * 1e6)
    axes[0].set_xlabel("time (us)")
    axes[0].set_ylabel("normalized amplitude")
    axes[0].set_title(f"Matched filter vs. raw envelope, theta={thetas[rep_idx]:.0f}deg")
    axes[0].legend(fontsize=7)

    positions_hist = [m + 1 if m is not None else 0 for m in matched_positions]
    axes[1].hist(positions_hist, bins=np.arange(0, max(positions_hist) + 2) - 0.5, edgecolor="black")
    axes[1].set_xlabel("matched-filter peak position for the TRUE inner echo\n(0 = no match)")
    axes[1].set_ylabel("count (of 36 angles)")
    axes[1].set_title("Does matched filtering move the true echo to an earlier,\nmore reliable position?")

    fig.suptitle("Matched filter echo extraction: ringing artifact vs. genuine reverberation")
    labels.add_banner(fig)
    plt.tight_layout()
    os.makedirs("results/figures", exist_ok=True)
    out_fig = "results/figures/phase1_matched_filter_echo_extraction.png"
    plt.savefig(out_fig, dpi=140)
    print(f"\nSaved {out_fig}")
