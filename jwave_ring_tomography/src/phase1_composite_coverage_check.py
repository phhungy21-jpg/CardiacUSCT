"""Composite coverage check: how much of run -15's 87%-unexplained
figure does adding the off-axis outer-wall bounce model (runs -16/-17,
now with the geometric-spreading/angular-efficiency separation) account
for, combined with the already-modeled discrete cascade (direct outer,
direct inner, +1/+2/+3 internal reverberations)?

Composite classifier, per detected peak: check against the discrete
cascade (inner_k0..k3, matched within a tight window) AND the
CONTINUOUS off-axis-outer family (any phi in [0, MAX_PHI_COMPOSITE_DEG]
-- capped at the EMPIRICALLY-VALIDATED range from runs -16/-17, ~27
degrees observed, not extended to the full 75-degree search range used
there, which would start encroaching on inner_k1/k2 territory and risk
false-positive "explanations" of genuine reverberation echoes). Uses
whichever candidate gives the smallest timing residual.
"""

import numpy as np
from scipy.signal import find_peaks

from phase1_matched_filter_echo_extraction import (
    simulate_pitch_catch_raw, matched_filter_output, _lag_t_arr, PEAK_PROMINENCE_FRACTION,
)
from phase1_multibounce_cascade_model import predicted_bounce_cascade
from phase1_offaxis_outer_bounce_model import dist_cells, offaxis_time
from phase1_reflection_channel_scout import thetas
from phase1_circular_positive_control import build_medium_concentric_circles, build_medium_water_only, R_OUTER, R_INNER
from phase1_rotating_transmission_scout import PROBE_RADIUS_CELLS
import phase2_config as cfg
import labels

from matplotlib import pyplot as plt
import os

MATCH_WINDOW_S = 1.5e-7
MAX_PHI_COMPOSITE_DEG = 30.0  # capped at runs -16/-17's empirically-validated range, not the full 75-deg search


def best_offaxis_match(t_observed, r_outer, n_search=300):
    phis = np.linspace(0, np.deg2rad(MAX_PHI_COMPOSITE_DEG), n_search)
    times = offaxis_time(phis, r_outer)
    idx = np.argmin(np.abs(times - t_observed))
    resid = abs(times[idx] - t_observed)
    if resid < MATCH_WINDOW_S:
        return np.degrees(phis[idx]), resid
    return None, None


if __name__ == "__main__":
    print(f"*** {labels.PENDING_SIGNOFF_BANNER} ***")
    print(f"*** {labels.TOY_EXACT_GT_CAPTION} ***")
    print("COMPOSITE COVERAGE CHECK: discrete cascade (outer/inner/reverberation, run -15) "
          f"+ off-axis outer-wall family (capped at {MAX_PHI_COMPOSITE_DEG} deg, runs -16/-17) "
          "-- how much of run -15's 87%-unexplained figure does this combination account for?")
    print("  compute estimate: 36 angles x 2 media = 72 forward sims (same as prior runs) "
          "-- ~15-20 minutes based on that precedent")

    cascade = predicted_bounce_cascade(R_OUTER, R_INNER)
    cascade_dict = dict(cascade)

    medium_water = build_medium_water_only()
    medium_phantom = build_medium_concentric_circles()

    print("\n=== Simulating water-only control, pitch-catch at 36 angles ===")
    water_traces = [simulate_pitch_catch_raw(medium_water, th) for th in thetas]
    print("=== Simulating concentric-circle phantom, pitch-catch at 36 angles ===")
    phantom_traces = [simulate_pitch_catch_raw(medium_phantom, th) for th in thetas]

    water_mf = [matched_filter_output(tr) for tr in water_traces]
    phantom_mf = [matched_filter_output(tr) for tr in phantom_traces]
    _nonneg = _lag_t_arr >= 0

    print("\n=== Classifying ALL detected peaks against the composite model ===")
    total_peaks = 0
    explained_discrete = {name: 0 for name, _ in cascade}
    explained_offaxis = 0
    unexplained = 0
    unexplained_list = []

    for i, theta in enumerate(thetas):
        env_w, _ = water_mf[i]
        env_p, _ = phantom_mf[i]
        thresh = max(env_w[_nonneg].max() * 3.0, env_p[_nonneg].max() * PEAK_PROMINENCE_FRACTION)
        peak_idx, _ = find_peaks(env_p[_nonneg], height=thresh)
        peak_times = np.sort(_lag_t_arr[_nonneg][peak_idx])

        for pt in peak_times:
            total_peaks += 1
            candidates = []
            for name, t_pred in cascade:
                resid = abs(pt - t_pred)
                if resid < MATCH_WINDOW_S:
                    candidates.append((resid, "discrete", name))
            phi, resid_offaxis = best_offaxis_match(pt, R_OUTER)
            if phi is not None:
                candidates.append((resid_offaxis, "offaxis", phi))

            if not candidates:
                unexplained += 1
                unexplained_list.append((i, pt))
                continue
            candidates.sort(key=lambda c: c[0])
            _, kind, label = candidates[0]
            if kind == "discrete":
                explained_discrete[label] += 1
            else:
                explained_offaxis += 1

    total_explained = sum(explained_discrete.values()) + explained_offaxis
    print(f"\n--- Composite coverage ---")
    print(f"  total peaks: {total_peaks}")
    print(f"  explained (discrete cascade): {sum(explained_discrete.values())} -- breakdown:")
    for name, count in explained_discrete.items():
        print(f"    {name}: {count}")
    print(f"  explained (off-axis outer family, <= {MAX_PHI_COMPOSITE_DEG} deg): {explained_offaxis}")
    print(f"  TOTAL explained: {total_explained}/{total_peaks} ({100*total_explained/total_peaks:.0f}%)")
    print(f"  UNEXPLAINED: {unexplained}/{total_peaks} ({100*unexplained/total_peaks:.0f}%)")
    print(f"\n  COMPARE run -15 (discrete cascade only): 44/332 (13%) explained, 87% unexplained")

    fig, ax = plt.subplots(figsize=(9, 5))
    categories = list(explained_discrete.keys()) + ["offaxis_outer", "UNEXPLAINED"]
    counts = list(explained_discrete.values()) + [explained_offaxis, unexplained]
    colors = ["C0"] * len(explained_discrete) + ["C1", "red"]
    ax.bar(categories, counts, color=colors)
    ax.set_ylabel("number of peaks")
    ax.set_title(f"Composite model coverage: {total_explained}/{total_peaks} "
                 f"({100*total_explained/total_peaks:.0f}%) explained\n"
                 f"(run -15 discrete-only baseline: 13% explained)")
    plt.xticks(rotation=20, ha="right")
    labels.add_banner(fig)
    plt.tight_layout()
    os.makedirs("results/figures", exist_ok=True)
    out_fig = "results/figures/phase1_composite_coverage_check.png"
    plt.savefig(out_fig, dpi=140)
    print(f"\nSaved {out_fig}")
