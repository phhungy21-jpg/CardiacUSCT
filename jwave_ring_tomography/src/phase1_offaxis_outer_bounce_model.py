"""Tests the user's proposed mechanism for run -15's unexplained
"mystery" peaks (87% of all detected peaks, including 64 arriving
BEFORE the direct inner echo, which the radial reverberation cascade
cannot produce by construction): OFF-AXIS reflections from the SAME
outer boundary, not from the inner boundary or internal reverberation.

A real point source illuminates the WHOLE nearby arc of the outer
circle, not just the single radial point directly "in front." A point
Q on that circle at angular offset phi from the true radial point sits
FARTHER from the probe (by basic geometry, the radial point is the
unique nearest point on a circle), so its round-trip water-only path
is LONGER, arriving LATER than the direct (phi=0) outer echo -- and,
per the user, at correspondingly REDUCED amplitude (increasing
obliqueness/incidence angle reduces reflection efficiency). Critically,
this uses ONLY water-path propagation (never enters tissue), so it can
plausibly reach times all the way up to (and even past) the direct
inner-echo time using modest, physically ordinary angles -- confirmed
analytically before building: phi~26 degrees already reaches the
direct-inner-echo time (7.8us) using pure water-path geometry.

Method: derive the round-trip time as a function of angular offset phi
analytically (law of cosines, monostatic approximation), invert it for
each observed "mystery" peak to get an IMPLIED phi, and check whether
implied-phi vs. observed amplitude follows a sensible, monotonically
decreasing relationship (the "equation to normalize" the user asked
for) -- confirming or refuting the mechanism on its own predictive
structure, not just aggregate match-rate.
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
MAX_PHI_DEG = 75.0  # cap the search -- beyond this, shadowing/grazing effects make the simple model unreliable


def dist_cells(phi_rad, r_outer):
    """One-way distance (cells) from probe to the outer circle at
    angular offset phi. Law of cosines: |P-Q|^2 = R_probe^2 + r_outer^2
    - 2*R_probe*r_outer*cos(phi)."""
    return np.sqrt(PROBE_RADIUS_CELLS ** 2 + r_outer ** 2
                   - 2 * PROBE_RADIUS_CELLS * r_outer * np.cos(phi_rad))


def offaxis_time(phi_rad, r_outer):
    """Round-trip water-only travel time for a monostatic pair
    reflecting off the outer circle at angular offset phi."""
    return 2 * dist_cells(phi_rad, r_outer) * dx[0] / cfg.WATER.sound_speed


def implied_phi(t_observed, r_outer):
    """Invert offaxis_time(phi) for phi, given an observed round-trip
    time. Returns None if outside the modeled phi range."""
    t_min = offaxis_time(0.0, r_outer)
    t_max = offaxis_time(np.deg2rad(MAX_PHI_DEG), r_outer)
    if t_observed < t_min or t_observed > t_max:
        return None
    f = lambda phi: offaxis_time(phi, r_outer) - t_observed
    return np.degrees(brentq(f, 0.0, np.deg2rad(MAX_PHI_DEG)))


if __name__ == "__main__":
    print(f"*** {labels.PENDING_SIGNOFF_BANNER} ***")
    print(f"*** {labels.TOY_EXACT_GT_CAPTION} ***")
    print("OFF-AXIS OUTER-BOUNCE MODEL: tests whether run -15's unexplained peaks are "
          "off-axis reflections from the SAME outer boundary (wider-angle, longer water-only "
          "path, weaker amplitude), not inner-boundary echoes or tissue reverberation.")
    t_at_26deg = offaxis_time(np.deg2rad(26.0), R_OUTER)
    print(f"  analytic check: a {26}-degree off-axis outer bounce arrives at "
          f"{t_at_26deg*1e6:.2f}us (direct inner echo is at ~7.80us per run -15)")
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

    cascade = dict(predicted_bounce_cascade(R_OUTER, R_INNER))
    t_outer, t_inner_k0 = cascade["outer"], cascade["inner_k0"]

    print("\n=== Testing off-axis outer-bounce model against all detected peaks ===")
    all_phi, all_amp, all_t, all_dist = [], [], [], []
    n_explained_by_offaxis = 0
    n_total = 0
    for i, theta in enumerate(thetas):
        env_w, _ = water_mf[i]
        env_p, _ = phantom_mf[i]
        thresh = max(env_w[_nonneg].max() * 3.0, env_p[_nonneg].max() * PEAK_PROMINENCE_FRACTION)
        peak_idx, props = find_peaks(env_p[_nonneg], height=thresh)
        peak_times = _lag_t_arr[_nonneg][peak_idx]
        peak_amps = env_p[_nonneg][peak_idx]
        order = np.argsort(peak_times)
        peak_times, peak_amps = peak_times[order], peak_amps[order]

        for pt, pa in zip(peak_times, peak_amps):
            n_total += 1
            # only test peaks between the direct outer and direct inner echoes -- the "mystery gap"
            if pt < t_outer - MATCH_WINDOW_S or pt > t_inner_k0 + MATCH_WINDOW_S:
                continue
            phi = implied_phi(pt, R_OUTER)
            if phi is not None:
                n_explained_by_offaxis += 1
                all_phi.append(phi)
                all_amp.append(pa)
                all_t.append(pt)
                all_dist.append(dist_cells(np.deg2rad(phi), R_OUTER))

    print(f"\n--- Off-axis outer-bounce model: coverage ---")
    print(f"  peaks in the 'mystery gap' (between direct outer and direct inner echo) "
          f"explained by SOME phi in [0,{MAX_PHI_DEG}] deg: {n_explained_by_offaxis}")

    all_phi, all_amp, all_dist = np.array(all_phi), np.array(all_amp), np.array(all_dist)
    dist0 = dist_cells(0.0, R_OUTER)
    # Undo 2D monostatic round-trip geometric spreading (~1/distance: two legs of
    # ~1/sqrt(dist) cylindrical spreading each) BEFORE fitting the angular
    # (reflection-efficiency) falloff -- per user's correction, the raw fit
    # conflates "mild" distance-based spreading loss with the "dramatic"
    # angle-based reflection-efficiency loss; these must be separated.
    all_amp_corrected = all_amp * (all_dist / dist0)

    if len(all_phi) > 3:
        corr_raw = np.corrcoef(all_phi, all_amp)[0, 1]
        corr_corrected = np.corrcoef(all_phi, all_amp_corrected)[0, 1]
        print(f"\n--- Implied angle vs. amplitude relationship (RAW, conflates spreading+angle) ---")
        print(f"  correlation(implied phi, RAW amplitude) = {corr_raw:+.3f}")

        def fit_power_law(amp):
            valid = amp > 0
            log_amp = np.log(amp[valid] / amp[valid].max())
            log_cos = np.log(np.clip(np.cos(np.deg2rad(all_phi[valid])), 1e-6, None))
            if np.std(log_cos) < 1e-9:
                return None
            return np.polyfit(log_cos, log_amp, 1)[0]

        n_fit_raw = fit_power_law(all_amp)
        print(f"  fitted power-law exponent (RAW amplitude ~ cos(phi)^n): n={n_fit_raw:.2f}")

        print(f"\n--- Implied angle vs. amplitude relationship (SPREADING-CORRECTED, isolates angle-only effect) ---")
        print(f"  correlation(implied phi, spreading-corrected amplitude) = {corr_corrected:+.3f}")
        n_fit_corrected = fit_power_law(all_amp_corrected)
        print(f"  fitted power-law exponent (corrected amplitude ~ cos(phi)^n): n={n_fit_corrected:.2f}")
        print(f"\n  spreading (mild, 1/distance) accounted for {all_dist.max()/dist0:.2f}x range in distance; "
              f"the TRUE angle-only reflection-efficiency exponent is n={n_fit_corrected:.2f} "
              f"(vs. the conflated raw fit's n={n_fit_raw:.2f})")

        os.makedirs("results", exist_ok=True)
        np.savez("results/offaxis_outer_bounce_data.npz", phi=all_phi, amp=all_amp,
                 amp_corrected=all_amp_corrected, dist=all_dist, t=np.array(all_t))

    fig, axes = plt.subplots(1, 3, figsize=(18, 5.5))
    if len(all_phi) > 0:
        sc = axes[0].scatter(np.array(all_t) * 1e6, all_phi, c=all_amp, cmap="viridis", s=25)
        plt.colorbar(sc, ax=axes[0], label="peak amplitude")
    axes[0].axvline(t_outer * 1e6, color="c", linestyle="--", alpha=0.6, label="direct outer echo")
    axes[0].axvline(t_inner_k0 * 1e6, color="lime", linestyle="--", alpha=0.6, label="direct inner echo")
    axes[0].set_xlabel("peak arrival time (us)")
    axes[0].set_ylabel("implied off-axis angle (deg)")
    axes[0].set_title("Mystery-gap peaks: implied off-axis angle vs. time")
    axes[0].legend(fontsize=8)

    if len(all_phi) > 0:
        axes[1].scatter(all_phi, all_amp, alpha=0.6, label=f"raw (n={n_fit_raw:.1f})")
        axes[1].set_xlabel("implied off-axis angle (deg)")
        axes[1].set_ylabel("peak amplitude (RAW)")
        axes[1].set_title("RAW amplitude vs. angle\n(conflates distance spreading + angular reflection loss)")
        axes[1].legend(fontsize=8)

        axes[2].scatter(all_phi, all_amp_corrected, alpha=0.6, color="C1",
                        label=f"spreading-corrected (n={n_fit_corrected:.1f})")
        axes[2].set_xlabel("implied off-axis angle (deg)")
        axes[2].set_ylabel("peak amplitude (spreading-corrected)")
        axes[2].set_title("CORRECTED amplitude vs. angle\n(distance effect removed -- isolates angular falloff)")
        axes[2].legend(fontsize=8)

    fig.suptitle("Off-axis outer-boundary bounce model: testing the user's proposed mechanism")
    labels.add_banner(fig)
    plt.tight_layout()
    os.makedirs("results/figures", exist_ok=True)
    out_fig = "results/figures/phase1_offaxis_outer_bounce_model.png"
    plt.savefig(out_fig, dpi=140)
    print(f"\nSaved {out_fig}")
