"""Amplitude-strata veto, per user: "if a bounce cause 10x decrease,
it should be a clear difference between 1st, 2nd, 3rd order bounce,
and if its just a few% drop, it should be an attenuated angle bounce.
with such a clear strata it would be an easi job for the classifier."

Uses the REAL reflection/transmission coefficients derived from this
project's own cited tissue properties (impedance Z=density*sound_speed)
to predict the expected amplitude SCALE for each bounce order:
  R_outer (water/myo) = 5.02%
  R_inner (myo/blood) = 0.253% (confirms the long-standing weak-contrast finding)
  direct inner (T*R_inner*T) ~= 20x weaker than direct outer
  k1 (1 extra internal round-trip) ~= 8000x weaker than direct inner
    (~160,000x weaker than direct outer -- almost certainly below any
    realistic simulation noise floor)

This is a DECISIVE, well-separated gap (many orders of magnitude, not
a close call), unlike the not-yet-validated inner off-axis ANGULAR
model (run -20, null result) -- so this veto targets specifically
whether the "inner_k2" matches from runs -15/-18/-19 (8 peaks each
run) are amplitude-consistent with genuine 2nd-order reverberation, or
are actually something else (off-axis contribution, numerical noise)
that happened to coincide in TIME with the k2 prediction.
"""

import numpy as np
from scipy.signal import find_peaks

from phase1_matched_filter_echo_extraction import (
    simulate_pitch_catch_raw, matched_filter_output, _lag_t_arr, PEAK_PROMINENCE_FRACTION,
)
from phase1_multibounce_cascade_model import predicted_bounce_cascade
from phase1_reflection_channel_scout import thetas
from phase1_circular_positive_control import build_medium_concentric_circles, build_medium_water_only, R_OUTER, R_INNER
import phase2_config as cfg
import labels

from matplotlib import pyplot as plt
import os

MATCH_WINDOW_S = 1.5e-7
VETO_MARGIN = 10.0  # observed amplitude must be within this multiple of the coefficient-predicted scale to survive


def acoustic_impedance(tissue):
    return tissue.density * tissue.sound_speed


def compute_coefficient_strata():
    Z_water = acoustic_impedance(cfg.WATER)
    Z_myo = acoustic_impedance(cfg.MYOCARDIUM)
    Z_blood = acoustic_impedance(cfg.BLOOD)
    R_outer_coeff = (Z_myo - Z_water) / (Z_myo + Z_water)
    R_inner_coeff = (Z_blood - Z_myo) / (Z_blood + Z_myo)
    T_in = 2 * Z_myo / (Z_water + Z_myo)
    T_out = 2 * Z_water / (Z_myo + Z_water)
    direct_outer = abs(R_outer_coeff)
    direct_inner = abs(T_in * R_inner_coeff * T_out)
    k1 = abs(T_in * R_inner_coeff * (-R_outer_coeff) * R_inner_coeff * T_out)
    k2 = abs(T_in * R_inner_coeff * (-R_outer_coeff) * R_inner_coeff * (-R_outer_coeff) * R_inner_coeff * T_out)
    return {"outer": direct_outer, "inner_k0": direct_inner, "inner_k1": k1, "inner_k2": k2,
            "inner_k3": k2 * abs(R_inner_coeff * (-R_outer_coeff))}


if __name__ == "__main__":
    print(f"*** {labels.PENDING_SIGNOFF_BANNER} ***")
    print(f"*** {labels.TOY_EXACT_GT_CAPTION} ***")
    print("AMPLITUDE-STRATA VETO: checks whether run -15/-18/-19's 'inner_k2' matches are "
          "amplitude-consistent with genuine 2nd-order reverberation, using coefficient-"
          "predicted amplitude ratios from this project's own cited tissue properties.")

    strata = compute_coefficient_strata()
    print("\n  coefficient-predicted relative amplitude strata (relative to direct outer):")
    for name, val in strata.items():
        ratio_to_outer = val / strata["outer"]
        print(f"    {name}: {val:.3e}  (ratio to direct outer: {ratio_to_outer:.3e})")

    cascade = predicted_bounce_cascade(R_OUTER, R_INNER)
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

    # calibrate the ABSOLUTE amplitude scale using the measured direct-outer echo
    # (the best-established, most-validated amplitude in this whole investigation)
    outer_amps_measured = []
    for i, theta in enumerate(thetas):
        env_p, _ = phantom_mf[i]
        t_outer = cascade[0][1]
        idx = np.argmin(np.abs(_lag_t_arr[_nonneg] - t_outer))
        outer_amps_measured.append(env_p[_nonneg][idx])
    outer_baseline_amp = np.median(outer_amps_measured)
    print(f"\n  calibrated direct-outer amplitude baseline (median observed): {outer_baseline_amp:.4g}")

    predicted_abs_amp = {name: outer_baseline_amp * (val / strata["outer"]) for name, val in strata.items()}
    print("  coefficient-predicted ABSOLUTE amplitude per bounce order (calibrated to outer baseline):")
    for name, val in predicted_abs_amp.items():
        print(f"    {name}: predicted~{val:.3e}, VETO if observed > {VETO_MARGIN}x that = {val*VETO_MARGIN:.3e}")

    print("\n=== Re-checking each cascade category's matches against the amplitude strata ===")
    category_amps = {name: [] for name, _ in cascade}
    for i, theta in enumerate(thetas):
        env_w, _ = water_mf[i]
        env_p, _ = phantom_mf[i]
        thresh = max(env_w[_nonneg].max() * 3.0, env_p[_nonneg].max() * PEAK_PROMINENCE_FRACTION)
        peak_idx, _ = find_peaks(env_p[_nonneg], height=thresh)
        peak_times = _lag_t_arr[_nonneg][peak_idx]
        peak_amps = env_p[_nonneg][peak_idx]

        for pt, pa in zip(peak_times, peak_amps):
            for name, t_pred in cascade:
                if abs(pt - t_pred) < MATCH_WINDOW_S:
                    category_amps[name].append(pa)

    print(f"\n--- Per-category match count and amplitude vs. strata veto ---")
    survived, vetoed = {}, {}
    for name, amps in category_amps.items():
        amps = np.array(amps)
        pred = predicted_abs_amp[name]
        veto_thresh = pred * VETO_MARGIN
        n_survive = int((amps <= veto_thresh).sum()) if len(amps) else 0
        n_veto = len(amps) - n_survive
        survived[name] = n_survive
        vetoed[name] = n_veto
        if len(amps) > 0:
            print(f"  {name}: {len(amps)} raw matches, predicted amp~{pred:.3e}, "
                  f"observed range [{amps.min():.3e}, {amps.max():.3e}], "
                  f"SURVIVE veto: {n_survive}, VETOED (too strong to be this order): {n_veto}")
        else:
            print(f"  {name}: 0 raw matches")

    print(f"\n--- Verdict on run -15/-18/-19's inner_k2 matches ---")
    print(f"  inner_k2: {vetoed.get('inner_k2', 0)}/{len(category_amps['inner_k2'])} matches VETOED "
          f"(observed amplitude far too strong for genuine 2nd-order reverberation)")
    print(f"  CONCLUSION: {'CONFIRMED -- these matches are NOT genuine k2 reverberation, consistent with the ~160,000x predicted gap' if vetoed.get('inner_k2',0)==len(category_amps['inner_k2']) and len(category_amps['inner_k2'])>0 else 'mixed result -- see per-category detail above'}")

    fig, ax = plt.subplots(figsize=(9, 5.5))
    names = list(category_amps.keys())
    x = np.arange(len(names))
    surv_counts = [survived[n] for n in names]
    veto_counts = [vetoed[n] for n in names]
    ax.bar(x, surv_counts, label="survives amplitude strata", color="C2")
    ax.bar(x, veto_counts, bottom=surv_counts, label="VETOED (amplitude inconsistent)", color="red")
    ax.set_xticks(x)
    ax.set_xticklabels(names, rotation=20, ha="right")
    ax.set_ylabel("number of time-matched peaks")
    ax.set_title("Amplitude-strata veto: which cascade matches survive a physically-grounded\n"
                 "amplitude check (coefficient-predicted, calibrated to measured outer baseline)?")
    ax.legend(fontsize=8)
    labels.add_banner(fig)
    plt.tight_layout()
    os.makedirs("results/figures", exist_ok=True)
    out_fig = "results/figures/phase1_amplitude_strata_veto.png"
    plt.savefig(out_fig, dpi=140)
    print(f"\nSaved {out_fig}")
