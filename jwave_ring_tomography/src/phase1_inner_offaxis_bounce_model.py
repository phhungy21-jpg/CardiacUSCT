"""The inner boundary's own off-axis family, per user: "model the
inner boundary's own off-axis family." Analogous to runs -16/-17's
outer-wall off-axis model, but geometrically more involved -- a path
to an off-axis point on the INNER circle must first cross the OUTER
interface (a real, different-speed boundary), not travel through
uniform water the whole way.

Geometry (straight-line/no-refraction approximation, consistent with
every other prediction in this project so far -- a real simplification
worth flagging, since the wave should genuinely bend via Snell's law
at the outer interface for an oblique path, not travel in a straight
line switching only speed): for an off-axis angle psi, find where the
straight line from the probe to the off-axis inner-boundary point Q
crosses the outer circle, split the path into a water leg (probe to
crossing point) and a myocardium leg (crossing point to Q), each at
its own sound speed. VERIFIED analytically before running anything:
at psi=0 this exactly reproduces the already-validated direct
inner-echo leg lengths (40.0/20.0 cells for this phantom's R_outer=80,
R_inner=60) -- confirmed to machine precision before trusting the
general formula.

Tests this model directly against currently-UNEXPLAINED peaks (run
-19's classification) arriving at or after the direct inner-echo time,
mirroring run -16's standalone validation approach before considering
integration into the full composite classifier.
"""

import numpy as np
from scipy.signal import find_peaks
from scipy.optimize import brentq

from phase1_matched_filter_echo_extraction import (
    simulate_pitch_catch_raw, matched_filter_output, _lag_t_arr, PEAK_PROMINENCE_FRACTION,
)
from phase1_multibounce_cascade_model import predicted_bounce_cascade
from phase1_reflection_channel_scout import thetas
from phase1_circular_positive_control import build_medium_concentric_circles, build_medium_water_only, R_OUTER, R_INNER
from phase1_rotating_transmission_scout import PROBE_RADIUS_CELLS, dx
import phase2_config as cfg
import labels

from matplotlib import pyplot as plt
import os

MATCH_WINDOW_S = 1.5e-7
GATE_MARGIN_DEG = 2.5
MAX_PSI_SEARCH_DEG = 40.0  # generous outer search bound before gating


def leg_split_cells(psi_rad, r_outer, r_inner, probe_radius=PROBE_RADIUS_CELLS):
    """Straight-line (no-refraction) split of the probe-to-inner-point
    path into a water leg and a myocardium leg, via line/circle
    intersection with the outer boundary."""
    P = np.array([probe_radius, 0.0])
    Q = np.array([r_inner * np.cos(psi_rad), r_inner * np.sin(psi_rad)])
    d = Q - P
    A = d @ d
    B = 2 * P @ d
    C = P @ P - r_outer ** 2
    disc = B ** 2 - 4 * A * C
    t = (-B - np.sqrt(disc)) / (2 * A)
    dist = np.linalg.norm(d)
    return t * dist, (1 - t) * dist  # leg1 (water), leg2 (myocardium)


def inner_offaxis_time(psi_rad, r_outer, r_inner):
    leg1, leg2 = leg_split_cells(psi_rad, r_outer, r_inner)
    return 2 * (leg1 * dx[0] / cfg.WATER.sound_speed + leg2 * dx[0] / cfg.MYOCARDIUM.sound_speed)


def implied_psi(t_observed, r_outer, r_inner, max_psi_deg=MAX_PSI_SEARCH_DEG):
    t_min = inner_offaxis_time(0.0, r_outer, r_inner)
    t_max = inner_offaxis_time(np.deg2rad(max_psi_deg), r_outer, r_inner)
    if t_observed < t_min or t_observed > t_max:
        return None
    f = lambda psi: inner_offaxis_time(psi, r_outer, r_inner) - t_observed
    return np.degrees(brentq(f, 0.0, np.deg2rad(max_psi_deg)))


if __name__ == "__main__":
    print(f"*** {labels.PENDING_SIGNOFF_BANNER} ***")
    print(f"*** {labels.TOY_EXACT_GT_CAPTION} ***")

    # sanity check before running anything, matching this project's established discipline
    l1_check, l2_check = leg_split_cells(0.0, R_OUTER, R_INNER)
    expected_l1, expected_l2 = PROBE_RADIUS_CELLS - R_OUTER, R_OUTER - R_INNER
    print(f"SANITY CHECK: at psi=0, leg1={l1_check:.4f} (expect {expected_l1}), "
          f"leg2={l2_check:.4f} (expect {expected_l2}) -- "
          f"{'PASS' if abs(l1_check-expected_l1)<1e-6 and abs(l2_check-expected_l2)<1e-6 else 'FAIL, DO NOT TRUST FURTHER RESULTS'}")

    cascade = predicted_bounce_cascade(R_OUTER, R_INNER)
    cascade_dict = dict(cascade)
    t_inner_k0, t_inner_k1 = cascade_dict["inner_k0"], cascade_dict["inner_k1"]

    # gate the inner off-axis search before it encroaches on the NEXT cascade term (inner_k1),
    # same discipline as the outer model's gating against inner_k0 (run -19)
    psi_critical = implied_psi(t_inner_k1, R_OUTER, R_INNER)
    max_psi_safe = psi_critical - GATE_MARGIN_DEG if psi_critical is not None else MAX_PSI_SEARCH_DEG
    print(f"\n  psi_critical (inner off-axis time == inner_k1 reverberation time): "
          f"{psi_critical if psi_critical is not None else 'not reached within search range'}")
    print(f"  gated inner off-axis search range: [0, {max_psi_safe:.2f}] deg")
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

    print("\n=== Testing inner off-axis model against peaks at/after the direct inner echo ===")
    all_psi, all_amp, all_t = [], [], []
    n_tested_region = 0
    for i, theta in enumerate(thetas):
        env_w, _ = water_mf[i]
        env_p, _ = phantom_mf[i]
        thresh = max(env_w[_nonneg].max() * 3.0, env_p[_nonneg].max() * PEAK_PROMINENCE_FRACTION)
        peak_idx, _ = find_peaks(env_p[_nonneg], height=thresh)
        peak_times = _lag_t_arr[_nonneg][peak_idx]
        peak_amps = env_p[_nonneg][peak_idx]
        order = np.argsort(peak_times)
        peak_times, peak_amps = peak_times[order], peak_amps[order]

        for pt, pa in zip(peak_times, peak_amps):
            if pt < t_inner_k0 - MATCH_WINDOW_S:
                continue  # this model only applies at/after the direct inner echo
            n_tested_region += 1
            psi = implied_psi(pt, R_OUTER, R_INNER, max_psi_deg=max_psi_safe)
            if psi is not None:
                all_psi.append(psi)
                all_amp.append(pa)
                all_t.append(pt)

    print(f"\n--- Inner off-axis model: coverage in its applicable region ---")
    print(f"  peaks at/after direct inner echo: {n_tested_region}")
    print(f"  explained by inner off-axis family: {len(all_psi)}/{n_tested_region}")

    all_psi, all_amp = np.array(all_psi), np.array(all_amp)
    if len(all_psi) > 3:
        corr = np.corrcoef(all_psi, all_amp)[0, 1]
        print(f"\n--- Implied angle vs. amplitude relationship ---")
        print(f"  correlation(implied psi, amplitude) = {corr:+.3f} "
              f"(expect NEGATIVE if the mechanism is real)")
        valid = all_amp > 0
        log_amp = np.log(all_amp[valid])
        log_cos = np.log(np.clip(np.cos(np.deg2rad(all_psi[valid])), 1e-6, None))
        if np.std(log_cos) > 1e-9:
            n_fit = np.polyfit(log_cos, log_amp, 1)[0]
            print(f"  fitted power-law exponent (amplitude ~ cos(psi)^n): n={n_fit:.2f} "
                  f"(compare to the outer model's n=20.42, run -17)")

        os.makedirs("results", exist_ok=True)
        np.savez("results/inner_offaxis_bounce_data.npz", psi=all_psi, amp=all_amp, t=np.array(all_t))

    fig, axes = plt.subplots(1, 2, figsize=(13, 5.5))
    if len(all_psi) > 0:
        sc = axes[0].scatter(np.array(all_t) * 1e6, all_psi, c=all_amp, cmap="viridis", s=25)
        plt.colorbar(sc, ax=axes[0], label="peak amplitude")
    axes[0].axvline(t_inner_k0 * 1e6, color="lime", linestyle="--", alpha=0.6, label="direct inner echo")
    if psi_critical is not None:
        axes[0].axvline(t_inner_k1 * 1e6, color="m", linestyle="--", alpha=0.6, label="inner_k1 (reverberation)")
    axes[0].set_xlabel("peak arrival time (us)")
    axes[0].set_ylabel("implied off-axis angle psi (deg)")
    axes[0].set_title("Peaks at/after direct inner echo: implied inner off-axis angle vs. time")
    axes[0].legend(fontsize=8)

    if len(all_psi) > 0:
        axes[1].scatter(all_psi, all_amp, alpha=0.6)
        axes[1].set_xlabel("implied off-axis angle psi (deg)")
        axes[1].set_ylabel("peak amplitude")
        axes[1].set_title("Implied angle vs. amplitude (inner boundary family)")

    fig.suptitle("Inner boundary's own off-axis bounce family (analogous to runs -16/-17's outer model)")
    labels.add_banner(fig)
    plt.tight_layout()
    os.makedirs("results/figures", exist_ok=True)
    out_fig = "results/figures/phase1_inner_offaxis_bounce_model.png"
    plt.savefig(out_fig, dpi=140)
    print(f"\nSaved {out_fig}")
