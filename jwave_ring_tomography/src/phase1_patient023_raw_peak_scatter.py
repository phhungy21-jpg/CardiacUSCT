"""Quick diagnostic, per user: "do a quick xy plot for incoming
amplitude and time for every signal" -- plots every detected peak's
(arrival time, amplitude) across all 36 angles, on patient023's real
anatomy, with NO classification applied (no matched-filter exclusion
logic, no strata veto) -- just the raw population, to visually inspect
for clustering/structure before deciding what run -25's spatially-wrong
"inner" peaks actually are.

Reflection channel only (not transmission) -- this question is
specifically about the pitch-catch echo population.
"""

import numpy as np
from scipy.signal import find_peaks

from phase1_matched_filter_echo_extraction import (
    simulate_pitch_catch_raw, matched_filter_output, _lag_t_arr, PEAK_PROMINENCE_FRACTION,
)
from phase1_patient023_validation import load_real_contours, build_medium_two_tissue, build_medium_water_only, MRI_NPZ
from phase1_reflection_channel_scout import thetas
import labels

from matplotlib import pyplot as plt
import os

_nonneg = _lag_t_arr >= 0

if __name__ == "__main__":
    print(f"*** {labels.PENDING_SIGNOFF_BANNER} ***")
    print(f"*** {labels.TOY_EXACT_GT_CAPTION} ***")
    print("RAW PEAK SCATTER: every detected (time, amplitude) pair across all 36 angles, "
          "patient023 real anatomy, reflection channel, NO classification applied.")
    print("  compute estimate: 36 angles x 2 media = 72 forward sims (reflection only, "
          "no transmission channel) -- ~15-20 minutes based on prior-run precedent")

    canvas_lv, canvas_myo, outer_contour_dom, inner_contour_dom = load_real_contours(MRI_NPZ)
    medium_water = build_medium_water_only()
    medium_phantom = build_medium_two_tissue(canvas_lv, canvas_myo)

    print("\n=== Simulating water-only control, pitch-catch at 36 angles ===")
    water_traces = [simulate_pitch_catch_raw(medium_water, th) for th in thetas]
    print("=== Simulating patient023 phantom, pitch-catch at 36 angles ===")
    phantom_traces = [simulate_pitch_catch_raw(medium_phantom, th) for th in thetas]

    water_mf = [matched_filter_output(tr) for tr in water_traces]
    phantom_mf = [matched_filter_output(tr) for tr in phantom_traces]

    all_t, all_amp, all_theta = [], [], []
    for i, theta in enumerate(thetas):
        env_w, _ = water_mf[i]
        env_p, _ = phantom_mf[i]
        thresh = max(env_w[_nonneg].max() * 3.0, env_p[_nonneg].max() * PEAK_PROMINENCE_FRACTION)
        peak_idx, _ = find_peaks(env_p[_nonneg], height=thresh)
        peak_times = _lag_t_arr[_nonneg][peak_idx]
        peak_amps = env_p[_nonneg][peak_idx]
        all_t.extend(peak_times.tolist())
        all_amp.extend(peak_amps.tolist())
        all_theta.extend([theta] * len(peak_times))

    all_t, all_amp, all_theta = np.array(all_t), np.array(all_amp), np.array(all_theta)
    print(f"\n  total peaks across all 36 angles: {len(all_t)}")

    os.makedirs("results", exist_ok=True)
    np.savez("results/patient023_raw_peaks.npz", t=all_t, amp=all_amp, theta=all_theta)

    fig, axes = plt.subplots(1, 2, figsize=(14, 5.5))
    sc = axes[0].scatter(all_t * 1e6, all_amp, c=all_theta, cmap="hsv", s=15, alpha=0.7)
    plt.colorbar(sc, ax=axes[0], label="probe angle (deg)")
    axes[0].set_xlabel("arrival time (us)")
    axes[0].set_ylabel("amplitude")
    axes[0].set_title(f"All detected peaks (n={len(all_t)}): time vs. amplitude")

    axes[1].scatter(all_t * 1e6, all_amp, c=all_theta, cmap="hsv", s=15, alpha=0.7)
    axes[1].set_yscale("log")
    axes[1].set_xlabel("arrival time (us)")
    axes[1].set_ylabel("amplitude (log scale)")
    axes[1].set_title("Same data, log amplitude (reveals strata separation)")

    fig.suptitle("Patient023: every detected peak, raw (no classification applied)")
    labels.add_banner(fig)
    plt.tight_layout()
    os.makedirs("results/figures", exist_ok=True)
    out_fig = "results/figures/phase1_patient023_raw_peak_scatter.png"
    plt.savefig(out_fig, dpi=140)
    print(f"\nSaved {out_fig}")
