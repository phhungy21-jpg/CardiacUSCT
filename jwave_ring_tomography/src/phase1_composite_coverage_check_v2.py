"""Corrected composite coverage check, per user's diagnosis of run
-18's flaw ("overfitting via continuity" -- a dense continuous
off-axis search out-competes a single discrete prediction near their
shared time boundary, cannibalizing genuine inner-boundary echoes).

Three-part fix, per user's specified hierarchy:
1. HARD-GATE the off-axis search: compute phi_critical where the
   off-axis outer-wall path time exactly equals the direct inner-echo
   time (t_inner_k0), and cap the off-axis search at
   phi_critical - GATE_MARGIN_DEG (2.5 degrees, within the user's
   specified 2-3 degree buffer) -- the off-axis model is no longer
   ALLOWED to compete for territory that close to the inner echo.
2. PRIORITY: if a peak matches a discrete cascade entry within
   MATCH_WINDOW_S, prefer it over an off-axis match by default.
3. GREY-ZONE AMPLITUDE TIEBREAKER: per user, a genuine ambiguous case
   is still possible near the boundary (a longer outer-wall path and a
   faster inner-echo path CAN land at the same time with DISTINCT
   energy) -- if a peak matches BOTH a discrete entry and an off-axis
   angle within the window, use the off-axis model's OWN fitted
   amplitude prediction (calibrated from run -17's saved data,
   amplitude ~ A0*cos(phi)^n) to check whether the OBSERVED amplitude
   is consistent with a weak off-axis bounce or is anomalously strong
   (more consistent with a genuine direct-cascade echo).
"""

import numpy as np
from scipy.signal import find_peaks
from scipy.optimize import brentq

from phase1_matched_filter_echo_extraction import (
    simulate_pitch_catch_raw, matched_filter_output, _lag_t_arr, PEAK_PROMINENCE_FRACTION,
)
from phase1_multibounce_cascade_model import predicted_bounce_cascade
from phase1_offaxis_outer_bounce_model import dist_cells, offaxis_time, implied_phi
from phase1_reflection_channel_scout import thetas
from phase1_circular_positive_control import build_medium_concentric_circles, build_medium_water_only, R_OUTER, R_INNER
from phase1_rotating_transmission_scout import PROBE_RADIUS_CELLS
import phase2_config as cfg
import labels

from matplotlib import pyplot as plt
import os

MATCH_WINDOW_S = 1.5e-7
GATE_MARGIN_DEG = 2.5
AMPLITUDE_ANOMALY_FACTOR = 2.0  # observed amp must exceed this multiple of the off-axis prediction to override it


def fit_offaxis_amplitude_model():
    """Calibrate amplitude ~ A0*cos(phi)^n from run -17's saved data
    (RAW amplitude, not spreading-corrected -- we want to predict the
    actually-OBSERVED amplitude at a given phi for comparison)."""
    d = np.load("results/offaxis_outer_bounce_data.npz")
    phi, amp = d["phi"], d["amp"]
    valid = amp > 0
    log_amp = np.log(amp[valid])
    log_cos = np.log(np.clip(np.cos(np.deg2rad(phi[valid])), 1e-6, None))
    n_fit, log_a0 = np.polyfit(log_cos, log_amp, 1)
    return np.exp(log_a0), n_fit


if __name__ == "__main__":
    print(f"*** {labels.PENDING_SIGNOFF_BANNER} ***")
    print(f"*** {labels.TOY_EXACT_GT_CAPTION} ***")
    print("COMPOSITE COVERAGE CHECK v2: hard-gated off-axis search + cascade priority + "
          "amplitude-based grey-zone tiebreaker, per user's diagnosis of run -18's "
          "'overfitting via continuity' flaw.")

    cascade = predicted_bounce_cascade(R_OUTER, R_INNER)
    cascade_dict = dict(cascade)
    t_inner_k0 = cascade_dict["inner_k0"]

    phi_critical = implied_phi(t_inner_k0, R_OUTER)
    max_phi_safe = phi_critical - GATE_MARGIN_DEG
    print(f"\n  phi_critical (off-axis outer time == direct inner echo time): {phi_critical:.2f} deg")
    print(f"  HARD-GATED off-axis search range: [0, {max_phi_safe:.2f}] deg "
          f"(margin={GATE_MARGIN_DEG} deg)")

    a0_fit, n_fit = fit_offaxis_amplitude_model()
    print(f"  off-axis amplitude model (from run -17's data): amplitude = {a0_fit:.5f} * cos(phi)^{n_fit:.2f}")

    def gated_offaxis_match(t_observed, r_outer, n_search=300):
        phis = np.linspace(0, np.deg2rad(max_phi_safe), n_search)
        times = offaxis_time(phis, r_outer)
        idx = np.argmin(np.abs(times - t_observed))
        resid = abs(times[idx] - t_observed)
        if resid < MATCH_WINDOW_S:
            return np.degrees(phis[idx]), resid
        return None, None

    print("  compute estimate: 36 angles x 2 media = 72 forward sims (same as prior runs) "
          "-- ~15-20 minutes based on that precedent")

    medium_water = build_medium_water_only()
    medium_phantom = build_medium_concentric_circles()

    print("\n=== Simulating water-only control, pitch-catch at 36 angles ===")
    water_traces = [simulate_pitch_catch_raw(medium_water, th) for th in thetas]
    print("=== Simulating concentric-circle phantom, pitch-catch at 36 angles ===")
    phantom_traces = [simulate_pitch_catch_raw(medium_phantom, th) for th in thetas]

    water_mf = [matched_filter_output(tr) for tr in water_traces]
    phantom_mf = [matched_filter_output(tr) for tr in phantom_traces]
    _nonneg = _lag_t_arr >= 0

    print("\n=== Classifying ALL detected peaks (hierarchy: discrete > off-axis, "
          "amplitude tiebreaker in grey zone) ===")
    total_peaks = 0
    explained_discrete = {name: 0 for name, _ in cascade}
    explained_offaxis = 0
    grey_zone_count = 0
    grey_zone_to_discrete = 0
    unexplained = 0

    for i, theta in enumerate(thetas):
        env_w, _ = water_mf[i]
        env_p, _ = phantom_mf[i]
        thresh = max(env_w[_nonneg].max() * 3.0, env_p[_nonneg].max() * PEAK_PROMINENCE_FRACTION)
        peak_idx, _ = find_peaks(env_p[_nonneg], height=thresh)
        order = np.argsort(_lag_t_arr[_nonneg][peak_idx])
        peak_times = _lag_t_arr[_nonneg][peak_idx[order]]
        peak_amps = env_p[_nonneg][peak_idx[order]]

        for pt, pa in zip(peak_times, peak_amps):
            total_peaks += 1
            discrete_matches = [(name, abs(pt - t_pred)) for name, t_pred in cascade if abs(pt - t_pred) < MATCH_WINDOW_S]
            phi_off, resid_off = gated_offaxis_match(pt, R_OUTER)

            has_discrete = len(discrete_matches) > 0
            has_offaxis = phi_off is not None

            if has_discrete and not has_offaxis:
                best_name = min(discrete_matches, key=lambda x: x[1])[0]
                explained_discrete[best_name] += 1
            elif has_offaxis and not has_discrete:
                explained_offaxis += 1
            elif has_discrete and has_offaxis:
                # GREY ZONE: both plausible -- use amplitude to disambiguate
                grey_zone_count += 1
                predicted_amp = a0_fit * np.cos(np.deg2rad(phi_off)) ** n_fit
                if pa > AMPLITUDE_ANOMALY_FACTOR * predicted_amp:
                    # anomalously strong for a weak off-axis bounce -- genuine cascade echo
                    best_name = min(discrete_matches, key=lambda x: x[1])[0]
                    explained_discrete[best_name] += 1
                    grey_zone_to_discrete += 1
                else:
                    explained_offaxis += 1
            else:
                unexplained += 1

    total_explained = sum(explained_discrete.values()) + explained_offaxis
    print(f"\n--- Composite coverage (v2, hierarchy-corrected) ---")
    print(f"  total peaks: {total_peaks}")
    print(f"  grey-zone (both discrete AND off-axis plausible): {grey_zone_count}, "
          f"of which {grey_zone_to_discrete} resolved to discrete via amplitude anomaly")
    print(f"  explained (discrete cascade): {sum(explained_discrete.values())} -- breakdown:")
    for name, count in explained_discrete.items():
        print(f"    {name}: {count}")
    print(f"  explained (off-axis outer family, gated <= {max_phi_safe:.1f} deg): {explained_offaxis}")
    print(f"  TOTAL explained: {total_explained}/{total_peaks} ({100*total_explained/total_peaks:.0f}%)")
    print(f"  UNEXPLAINED: {unexplained}/{total_peaks} ({100*unexplained/total_peaks:.0f}%)")
    print(f"\n  COMPARE run -15 (discrete only): 13% explained | "
          f"run -18 (uncorrected composite): 40% explained, but inner_k0=0 (flawed)")

    fig, ax = plt.subplots(figsize=(9, 5))
    categories = list(explained_discrete.keys()) + ["offaxis_outer", "UNEXPLAINED"]
    counts = list(explained_discrete.values()) + [explained_offaxis, unexplained]
    colors = ["C0"] * len(explained_discrete) + ["C1", "red"]
    ax.bar(categories, counts, color=colors)
    ax.set_ylabel("number of peaks")
    ax.set_title(f"Composite model coverage (v2, hierarchy-corrected): "
                 f"{total_explained}/{total_peaks} ({100*total_explained/total_peaks:.0f}%) explained\n"
                 f"grey-zone peaks: {grey_zone_count} ({grey_zone_to_discrete} resolved to discrete by amplitude)")
    plt.xticks(rotation=20, ha="right")
    labels.add_banner(fig)
    plt.tight_layout()
    os.makedirs("results/figures", exist_ok=True)
    out_fig = "results/figures/phase1_composite_coverage_check_v2.png"
    plt.savefig(out_fig, dpi=140)
    print(f"\nSaved {out_fig}")
