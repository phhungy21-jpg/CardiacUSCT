"""The "complete-picture inverse" the user proposed: instead of trying
to isolate ONE target echo (the inner boundary), predict the FULL
cascade of physically possible bounce paths for this two-interface
(outer water/myocardium, inner myocardium/blood) geometry, and match
ALL observed peaks against that whole pattern at once.

With only two interfaces and purely radial propagation, the possible
paths are: (0) outer reflection, never enters tissue; (1) inner
reflection, one bounce; (2), (3), ... inner reflection with 1, 2, ...
EXTRA internal round-trips inside the myocardium wall before finally
exiting. Each extra round-trip adds exactly 2*wall_thickness/c_myo to
the arrival time -- so ALL reverberation echoes are predicted to arrive
LATER than the direct (k=0) inner echo, never earlier. This is a
concrete, falsifiable prediction: if run -13/-14's "mystery" peaks
sitting BEFORE the direct inner echo are real, this simple radial
cascade CANNOT explain them, isolating them as a genuinely different,
still-unmodeled mechanism (most likely the pitch-catch pair's small
tangential offset, which this and all prior scripts approximated as
perfectly monostatic).

Reuses the matched-filter extraction (run -14, alignment bug fixed and
validated) since it resolves more genuine distinct echoes than raw
envelope detection.
"""

import numpy as np
from scipy.signal import find_peaks

from phase1_matched_filter_echo_extraction import (
    simulate_pitch_catch_raw, matched_filter_output, time_to_radius_matched_filter,
    _lag_t_arr, PEAK_PROMINENCE_FRACTION,
)
from phase1_reflection_channel_scout import thetas
from phase1_circular_positive_control import build_medium_concentric_circles, build_medium_water_only, R_OUTER, R_INNER
from phase1_rotating_transmission_scout import PROBE_RADIUS_CELLS, dx
import phase2_config as cfg
import labels

from matplotlib import pyplot as plt
import os

MAX_K = 4  # number of extra internal round-trips to predict (0 = direct inner echo)
MATCH_WINDOW_S = 2e-7


def predicted_bounce_cascade(r_outer, r_inner):
    c_water, c_myo = cfg.WATER.sound_speed, cfg.MYOCARDIUM.sound_speed
    d1 = (PROBE_RADIUS_CELLS - r_outer) * dx[0]
    d2 = (r_outer - r_inner) * dx[0]
    t_outer = 2 * d1 / c_water
    labels_times = [("outer", t_outer)]
    for k in range(MAX_K):
        t_k = 2 * d1 / c_water + 2 * (2 * k + 1) * d2 / c_myo
        labels_times.append((f"inner_k{k}", t_k))
    return labels_times


if __name__ == "__main__":
    print(f"*** {labels.PENDING_SIGNOFF_BANNER} ***")
    print(f"*** {labels.TOY_EXACT_GT_CAPTION} ***")
    print("MULTI-BOUNCE CASCADE MODEL: predicts the FULL set of physically possible echo "
          "paths (outer, direct inner, +1/+2/+3/+4 internal reverberation round-trips) and "
          "matches ALL observed peaks against the whole pattern at once, instead of "
          "isolating one target echo.")
    print("  compute estimate: 36 angles x 2 media = 72 forward sims (same as prior runs) "
          "-- ~15-20 minutes based on that precedent")

    cascade = predicted_bounce_cascade(R_OUTER, R_INNER)
    print("\n  predicted cascade (theta-independent leg lengths, circular/centered phantom):")
    for name, t in cascade:
        print(f"    {name}: {t*1e6:.4f} us")

    medium_water = build_medium_water_only()
    medium_phantom = build_medium_concentric_circles()

    print("\n=== Simulating water-only control, pitch-catch at 36 angles ===")
    water_traces = [simulate_pitch_catch_raw(medium_water, th) for th in thetas]
    print("=== Simulating concentric-circle phantom, pitch-catch at 36 angles ===")
    phantom_traces = [simulate_pitch_catch_raw(medium_phantom, th) for th in thetas]

    water_mf = [matched_filter_output(tr) for tr in water_traces]
    phantom_mf = [matched_filter_output(tr) for tr in phantom_traces]
    _nonneg = _lag_t_arr >= 0

    print("\n=== Matching ALL observed peaks against the FULL predicted cascade ===")
    all_matches = []  # (angle_idx, peak_time, matched_label, residual)
    unexplained_peaks = []  # peaks that don't match ANY cascade entry
    n_peaks_per_angle = []

    for i, theta in enumerate(thetas):
        env_w, _ = water_mf[i]
        env_p, _ = phantom_mf[i]
        thresh = max(env_w[_nonneg].max() * 3.0, env_p[_nonneg].max() * PEAK_PROMINENCE_FRACTION)
        peak_idx, _ = find_peaks(env_p[_nonneg], height=thresh)
        peak_times = np.sort(_lag_t_arr[_nonneg][peak_idx])
        n_peaks_per_angle.append(len(peak_times))

        for pt in peak_times:
            dists = [(name, abs(pt - t)) for name, t in cascade]
            best_name, best_dist = min(dists, key=lambda x: x[1])
            if best_dist < MATCH_WINDOW_S:
                all_matches.append((i, pt, best_name, pt - dict(cascade)[best_name]))
            else:
                unexplained_peaks.append((i, pt))

    total_peaks = sum(n_peaks_per_angle)
    print(f"\n--- Cascade match summary ---")
    print(f"  total peaks across all angles: {total_peaks} (mean {np.mean(n_peaks_per_angle):.1f}/angle)")
    print(f"  matched to SOME cascade entry: {len(all_matches)}/{total_peaks} "
          f"({100*len(all_matches)/total_peaks:.0f}%)")
    print(f"  UNEXPLAINED by the radial cascade: {len(unexplained_peaks)}/{total_peaks} "
          f"({100*len(unexplained_peaks)/total_peaks:.0f}%)")

    from collections import Counter
    match_counts = Counter(m[2] for m in all_matches)
    print(f"\n  matches by cascade entry:")
    for name, _ in cascade:
        c = match_counts.get(name, 0)
        print(f"    {name}: {c}/{len(all_matches)}")

    if any(m[2].startswith("inner_k0") for m in all_matches):
        k0_residuals = np.array([m[3] for m in all_matches if m[2] == "inner_k0"])
        k0_radii = [time_to_radius_matched_filter(m[1]) for m in all_matches if m[2] == "inner_k0"]
        print(f"\n  inner_k0 (direct inner echo) matches: {len(k0_residuals)}, "
              f"mean radius error={np.mean(k0_radii) - R_INNER:+.2f} cells")

    print(f"\n--- Diagnosis of unexplained peaks (arrive BEFORE inner_k0, per runs -13/-14) ---")
    t_inner_k0 = dict(cascade)["inner_k0"]
    before_k0 = [(i, pt) for i, pt in unexplained_peaks if pt < t_inner_k0]
    print(f"  unexplained peaks arriving BEFORE the direct inner echo (t<{t_inner_k0*1e6:.3f}us): "
          f"{len(before_k0)}/{len(unexplained_peaks)}")
    print(f"  CONCLUSION: {'the radial reverberation cascade explains most peaks -- remaining mystery peaks are a small residual' if len(unexplained_peaks) < total_peaks*0.3 else 'a SUBSTANTIAL fraction of peaks are NOT explained by simple radial reverberation -- a different mechanism (likely the pitch-catch tangential offset, not perfectly monostatic) is real and significant, not a minor residual'}")

    fig, ax = plt.subplots(figsize=(10, 6))
    for name, t in cascade:
        ax.axvline(t * 1e6, color="k", linestyle="--", alpha=0.4)
        ax.text(t * 1e6, 37, name, rotation=90, fontsize=7, ha="right", va="top")
    matched_times = [m[1] * 1e6 for m in all_matches]
    matched_angles = [thetas[m[0]] for m in all_matches]
    unmatched_times = [pt * 1e6 for i, pt in unexplained_peaks]
    unmatched_angles = [thetas[i] for i, pt in unexplained_peaks]
    ax.scatter(matched_times, matched_angles, c="green", s=10, label="matched to cascade", zorder=5)
    ax.scatter(unmatched_times, unmatched_angles, c="red", s=10, label="UNEXPLAINED", zorder=5)
    ax.set_xlabel("peak arrival time (us)")
    ax.set_ylabel("transmit angle (deg)")
    ax.set_title("All detected peaks vs. predicted multi-bounce cascade\n"
                 "(dashed lines = predicted cascade times; green = explained, red = not)")
    ax.legend(fontsize=8)

    fig.suptitle("Multi-bounce cascade model: does the full radial reverberation picture explain the data?")
    labels.add_banner(fig)
    plt.tight_layout()
    os.makedirs("results/figures", exist_ok=True)
    out_fig = "results/figures/phase1_multibounce_cascade_model.png"
    plt.savefig(out_fig, dpi=140)
    print(f"\nSaved {out_fig}")
