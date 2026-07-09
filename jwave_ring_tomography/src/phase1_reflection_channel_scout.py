"""Phase 1 scout, CHANNEL 1 (reflection): per user, "note those channel
down and proceed with the first one" -- see the "Multi-channel
information roadmap" at the top of `../ring_tomography_phase_protocol.md`.

Every prior run in this project (runs -03 through -07) excluded
near-angle receiver pairs as a "near-field artifact." That exclusion
was wrong for genuine reflection: a receiver near the transmitter has
no straight-line transmission path through the tissue at all, so
anything it picks up beyond the direct src-rcv coupling (excluded
separately, same convention as `jwave_test`) IS reflected/scattered
energy -- real data this project had been discarding.

This script builds a classic MONOSTATIC-style pitch-catch reflection
A-scan at each of the 36 ring positions (reusing `jwave_test`'s proven
pulse-echo methodology: closely-spaced src/rcv pair, direct-arrival
exclusion, envelope detection), on the SAME two-tissue real-anatomy
phantom as run -07 (patient001, myocardium wall + LV/blood cavity).
Tests directly whether reflection can see the INNER (blood/myocardium,
~0.5% contrast) boundary that run -07 confirmed transmission cannot,
by predicting both boundaries' expected reflection arrival times
geometrically and checking for a detectable peak at each.
"""

import numpy as np
from scipy.signal import hilbert

from jax import jit
from jwave.geometry import Sources
from jwave.acoustics import simulate_wave_propagation

from phase1_rotating_transmission_scout import (
    direction_vector, dx, center, N, domain, time_axis, dt, t_arr, _signal_template,
    PROBE_RADIUS_CELLS,
)
from phase1_two_tissue_reconstruction import (
    load_real_contours_two_tissue, build_medium_two_tissue, build_medium_water_only,
)
import phase2_config as cfg
import labels

from matplotlib import pyplot as plt
import os

N_ANGLES = 36
thetas = np.linspace(0, 360, N_ANGLES, endpoint=False)
OFFSET_CELLS = 5.0  # tangential src/rcv separation, same magnitude as jwave_test's pitch-catch convention
DIRECT_EXCLUDE_MARGIN_S = 1.5e-6  # same convention as jwave_test
_TONEBURST_DURATION_S = cfg.N_CYCLES / cfg.F0_HZ
_ENVELOPE_GROUP_DELAY_S = _TONEBURST_DURATION_S / 2
SEARCH_WINDOW_S = 4e-7  # +/- window around each predicted reflection time to search for the actual peak


def polar_resample(contour_rowcol, origin):
    rows = contour_rowcol[:, 0] - origin[0]
    cols = contour_rowcol[:, 1] - origin[1]
    r = np.hypot(rows, cols)
    theta = np.degrees(np.arctan2(cols, -rows)) % 360
    order = np.argsort(theta)
    theta_sorted, r_sorted = theta[order], r[order]
    keep = np.concatenate([[True], np.diff(theta_sorted) > 1e-9])
    theta_sorted, r_sorted = theta_sorted[keep], r_sorted[keep]
    ext_theta = np.concatenate([theta_sorted, [theta_sorted[0] + 360]])
    ext_r = np.concatenate([r_sorted, [r_sorted[0]]])
    return ext_theta, ext_r


def r_at_theta(theta_deg, ext_theta, ext_r):
    return np.interp(np.mod(theta_deg, 360), ext_theta, ext_r)


def pitch_catch_positions(theta_deg):
    d_row, d_col = direction_vector(theta_deg)
    t_row, t_col = d_col, -d_row  # tangential (perpendicular to radial)
    probe_row = center[0] + PROBE_RADIUS_CELLS * d_row
    probe_col = center[1] + PROBE_RADIUS_CELLS * d_col
    src = (round(probe_row - OFFSET_CELLS * t_row), round(probe_col - OFFSET_CELLS * t_col))
    rcv = (round(probe_row + OFFSET_CELLS * t_row), round(probe_col + OFFSET_CELLS * t_col))
    return src, rcv


def simulate_pitch_catch(medium, theta_deg):
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
    return np.abs(hilbert(trace))


def predicted_reflection_times(theta_deg, r_outer, r_inner):
    c_water, c_myo = cfg.WATER.sound_speed, cfg.MYOCARDIUM.sound_speed
    leg_to_outer_m = (PROBE_RADIUS_CELLS - r_outer) * dx[0]
    leg_outer_to_inner_m = (r_outer - r_inner) * dx[0]
    t_outer = 2 * (leg_to_outer_m / c_water) + _ENVELOPE_GROUP_DELAY_S
    t_inner = 2 * (leg_to_outer_m / c_water + leg_outer_to_inner_m / c_myo) + _ENVELOPE_GROUP_DELAY_S
    return t_outer, t_inner


def peak_in_window(envelope, t_center):
    mask = np.abs(t_arr - t_center) < SEARCH_WINDOW_S
    if not mask.any():
        return 0.0
    return envelope[mask].max()


if __name__ == "__main__":
    print(f"*** {labels.PENDING_SIGNOFF_BANNER} ***")
    print(f"*** {labels.TOY_EXACT_GT_CAPTION} ***")
    print("PHASE 1 SCOUT, CHANNEL 1 (REFLECTION): monostatic-style pitch-catch A-scan at "
          "each of 36 ring positions, patient001 two-tissue phantom (same as run -07). "
          "Tests whether reflection detects the INNER (blood/myocardium) boundary that "
          "run -07 confirmed transmission cannot.")
    print("  compute estimate: 36 angles x 2 media = 72 forward sims (same as prior runs) "
          "-- ~15-20 minutes based on that precedent")

    canvas_lv, canvas_myo, outer_contour_dom, inner_contour_dom, max_radius_cells = load_real_contours_two_tissue()
    ext_theta_out, ext_r_out = polar_resample(outer_contour_dom, center)
    ext_theta_in, ext_r_in = polar_resample(inner_contour_dom, center)

    medium_water = build_medium_water_only()
    medium_phantom = build_medium_two_tissue(canvas_lv, canvas_myo)

    print("\n=== Simulating water-only control, pitch-catch at 36 angles ===")
    water_envelopes = [simulate_pitch_catch(medium_water, th) for th in thetas]
    print("=== Simulating two-tissue phantom, pitch-catch at 36 angles ===")
    phantom_envelopes = [simulate_pitch_catch(medium_phantom, th) for th in thetas]

    outer_phantom_peaks, outer_water_peaks = [], []
    inner_phantom_peaks, inner_water_peaks = [], []
    for i, theta in enumerate(thetas):
        r_out = r_at_theta(theta, ext_theta_out, ext_r_out)
        r_in = r_at_theta(theta, ext_theta_in, ext_r_in)
        t_outer, t_inner = predicted_reflection_times(theta, r_out, r_in)
        outer_phantom_peaks.append(peak_in_window(phantom_envelopes[i], t_outer))
        outer_water_peaks.append(peak_in_window(water_envelopes[i], t_outer))
        inner_phantom_peaks.append(peak_in_window(phantom_envelopes[i], t_inner))
        inner_water_peaks.append(peak_in_window(water_envelopes[i], t_inner))

    outer_phantom_peaks = np.array(outer_phantom_peaks)
    outer_water_peaks = np.array(outer_water_peaks)
    inner_phantom_peaks = np.array(inner_phantom_peaks)
    inner_water_peaks = np.array(inner_water_peaks)

    outer_excess = outer_phantom_peaks - outer_water_peaks
    inner_excess = inner_phantom_peaks - inner_water_peaks

    print(f"\n--- OUTER (myocardium/water, ~3.6% contrast) reflection ---")
    print(f"  phantom peak: mean={outer_phantom_peaks.mean():.4g}, water-only baseline: mean={outer_water_peaks.mean():.4g}")
    print(f"  excess (phantom - water-only): mean={outer_excess.mean():.4g}, "
          f"detectable at {int((outer_excess > 0.1*outer_excess.max()).sum())}/{N_ANGLES} angles")

    print(f"\n--- INNER (blood/myocardium, ~0.5% contrast) reflection ---")
    print(f"  phantom peak: mean={inner_phantom_peaks.mean():.4g}, water-only baseline: mean={inner_water_peaks.mean():.4g}")
    print(f"  excess (phantom - water-only): mean={inner_excess.mean():.4g}, "
          f"detectable at {int((inner_excess > 0.1*outer_excess.max()).sum())}/{N_ANGLES} angles "
          f"(threshold = 10% of outer's own peak excess, for a fair relative comparison)")

    ratio = inner_excess.mean() / outer_excess.mean() if outer_excess.mean() != 0 else float("nan")
    print(f"\n  inner/outer excess-signal ratio: {ratio:.3f} "
          f"(compare to the ~0.5%/3.6% = 0.14 sound-speed-contrast ratio -- "
          f"reflection amplitude need not scale the same way as transmission delay)")

    fig, axes = plt.subplots(1, 2, figsize=(13, 5.5))
    rep_idx = 0  # representative angle for a raw trace plot
    axes[0].plot(t_arr * 1e6, phantom_envelopes[rep_idx], label="two-tissue phantom", color="C1")
    axes[0].plot(t_arr * 1e6, water_envelopes[rep_idx], label="water-only control", color="C0", alpha=0.7)
    t_outer_rep, t_inner_rep = predicted_reflection_times(
        thetas[rep_idx], r_at_theta(thetas[rep_idx], ext_theta_out, ext_r_out),
        r_at_theta(thetas[rep_idx], ext_theta_in, ext_r_in))
    axes[0].axvline(t_outer_rep * 1e6, color="c", linestyle="--", label="predicted outer reflection")
    axes[0].axvline(t_inner_rep * 1e6, color="lime", linestyle="--", label="predicted inner reflection")
    axes[0].set_xlabel("time (us)")
    axes[0].set_ylabel("envelope amplitude")
    axes[0].set_title(f"Representative A-scan, theta={thetas[rep_idx]:.0f}deg")
    axes[0].legend(fontsize=7)

    axes[1].plot(thetas, outer_excess, "o-", label="outer (myocardium/water) excess")
    axes[1].plot(thetas, inner_excess, "s-", label="inner (blood/myocardium) excess")
    axes[1].set_xlabel("angle (deg)")
    axes[1].set_ylabel("reflection excess amplitude (phantom - water-only)")
    axes[1].set_title("Reflection excess signal vs. angle")
    axes[1].legend(fontsize=8)

    fig.suptitle("Phase 1, CHANNEL 1 (reflection): does pitch-catch echo see the inner boundary\n"
                 "transmission (run -07) could not?")
    labels.add_banner(fig)
    plt.tight_layout()
    os.makedirs("results/figures", exist_ok=True)
    out_fig = "results/figures/phase1_reflection_channel_scout.png"
    plt.savefig(out_fig, dpi=140)
    print(f"\nSaved {out_fig}")
