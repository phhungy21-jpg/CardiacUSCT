"""Diagnoses run -12's finding directly: per user, "we need a time-of-
impact to measure which hits first, second, third etc. to
differentiate first-order bounce vs echoes."

Clarifies a subtlety: run -09/-12's blind detector ALREADY sorts peaks
chronologically (first peak = outer, second peak = inner) -- the
problem isn't ignoring arrival order, it's that a reverberation echo
can genuinely occupy that same "second position" for a thin,
high-contrast wall. What actually distinguishes a true first-order
bounce from a multi-path echo is PREDICTED time-of-impact (does a peak
exist near where the real single-bounce reflection analytically SHOULD
arrive), not just counting peaks in order.

This finds ALL peaks per angle (not just the first two), compares each
to the analytically predicted true single-bounce inner-reflection time,
and identifies which peak POSITION actually corresponds to the genuine
echo -- directly testing whether it's a different position than the
naive method's "#2", which would confirm reverberation is inserting an
extra peak before the true inner echo.
"""

import numpy as np
from scipy.signal import find_peaks

from phase1_reflection_channel_scout import (
    thetas, predicted_reflection_times, _ENVELOPE_GROUP_DELAY_S,
)
from phase1_circular_positive_control import (
    build_medium_concentric_circles, build_medium_water_only, R_OUTER, R_INNER,
)
from phase1_reflection_channel_scout import simulate_pitch_catch
from phase1_rotating_transmission_scout import PROBE_RADIUS_CELLS, dx, t_arr
import phase2_config as cfg
import labels

from matplotlib import pyplot as plt
import os

PEAK_PROMINENCE_FRACTION = 0.05
MATCH_WINDOW_S = 3e-7  # how close a peak must be to the prediction to count as "matched"


def time_to_radius_water_only(t):
    one_way_m = cfg.WATER.sound_speed * (t - _ENVELOPE_GROUP_DELAY_S) / 2.0
    return PROBE_RADIUS_CELLS - one_way_m / dx[0]


if __name__ == "__main__":
    print(f"*** {labels.PENDING_SIGNOFF_BANNER} ***")
    print(f"*** {labels.TOY_EXACT_GT_CAPTION} ***")
    print("DIAGNOSING run -12's bias: finding ALL peaks per angle (not just the first two) "
          "and matching against the ANALYTICALLY PREDICTED true single-bounce inner-reflection "
          "time, to see which peak POSITION is the genuine echo vs. a reverberation artifact.")
    print(f"  compute estimate: 36 angles x 2 media = 72 forward sims (same as prior runs) "
          f"-- ~15-20 minutes based on that precedent")

    medium_water = build_medium_water_only()
    medium_phantom = build_medium_concentric_circles()

    print("\n=== Simulating water-only control, pitch-catch at 36 angles ===")
    water_envelopes = [simulate_pitch_catch(medium_water, th) for th in thetas]
    print("=== Simulating concentric-circle phantom, pitch-catch at 36 angles ===")
    phantom_envelopes = [simulate_pitch_catch(medium_phantom, th) for th in thetas]

    print("\n=== Finding ALL peaks per angle, matching against predicted true echo times ===")
    matched_positions = []  # which chronological peak index matched the TRUE inner echo, per angle
    n_peaks_before_match = []
    naive_vs_matched_radius = []

    for i, theta in enumerate(thetas):
        envelope = phantom_envelopes[i]
        water_max = water_envelopes[i].max()
        thresh = max(water_max * 3.0, envelope.max() * PEAK_PROMINENCE_FRACTION)
        peak_idx, _ = find_peaks(envelope, height=thresh)
        order = np.argsort(t_arr[peak_idx])
        peak_idx_sorted = peak_idx[order]
        peak_times = t_arr[peak_idx_sorted]

        t_outer_true, t_inner_true = predicted_reflection_times(theta, R_OUTER, R_INNER)

        # which peak (chronological position) is closest to the TRUE predicted inner-echo time?
        if len(peak_times) == 0:
            continue
        dists_to_true_inner = np.abs(peak_times - t_inner_true)
        match_idx = int(np.argmin(dists_to_true_inner))
        if dists_to_true_inner[match_idx] > MATCH_WINDOW_S:
            match_idx = None  # no peak close enough to the true prediction at all

        matched_positions.append(match_idx)
        n_peaks_before_match.append(match_idx if match_idx is not None else -1)

        if len(peak_times) >= 2:
            naive_r = time_to_radius_water_only(peak_times[1])  # position #2, the naive method's pick
        else:
            naive_r = None
        matched_r = time_to_radius_water_only(peak_times[match_idx]) if match_idx is not None else None
        naive_vs_matched_radius.append((naive_r, matched_r))

    valid_matches = [m for m in matched_positions if m is not None]
    print(f"\n--- Which chronological peak position matches the TRUE predicted inner echo? ---")
    print(f"  matched at SOME peak in {len(valid_matches)}/{len(thetas)} angles")
    if valid_matches:
        from collections import Counter
        counts = Counter(valid_matches)
        for pos in sorted(counts):
            print(f"    position #{pos+1} (0-indexed {pos}): {counts[pos]}/{len(valid_matches)} angles")

    naive_radii = np.array([r for r, m in naive_vs_matched_radius if r is not None])
    matched_radii = np.array([m for r, m in naive_vs_matched_radius if m is not None])
    print(f"\n--- Radius accuracy: naive position-#2 pick vs. prediction-matched peak ---")
    print(f"  naive (position #2): mean error={np.mean(naive_radii - R_INNER):+.2f} cells "
          f"(matches run -12's +11.79)")
    print(f"  prediction-matched peak: mean error={np.mean(matched_radii - R_INNER):+.2f} cells")

    fig, ax = plt.subplots(figsize=(8, 5))
    positions_hist = [m + 1 if m is not None else 0 for m in matched_positions]  # 1-indexed, 0=no match
    ax.hist(positions_hist, bins=np.arange(0, max(positions_hist) + 2) - 0.5, edgecolor="black")
    ax.set_xlabel("chronological peak position matching the TRUE predicted inner echo\n(0 = no peak matched)")
    ax.set_ylabel("count (out of 36 angles)")
    ax.set_title("Which peak position is the genuine inner-boundary echo?\n"
                 "(naive method always picks position #2 -- mismatch here confirms reverberation)")
    labels.add_banner(fig)
    plt.tight_layout()
    os.makedirs("results/figures", exist_ok=True)
    out_fig = "results/figures/phase1_diagnose_reverberation.png"
    plt.savefig(out_fig, dpi=140)
    print(f"\nSaved {out_fig}")
