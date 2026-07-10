"""Patient023 version of run -09's fused transmission+reflection
visualization, per user: "now i want the 023 scan that looks like
phase1_fused_channel_reconstruction.png" -- followed by the explicit
instruction "make sure you incorporated all mechanisms we discovered."

Incorporates every established, validated mechanism from this
investigation, applied BLINDLY (no true-contour information used):

1. TRANSMISSION channel: full multistatic capture + SIRT reconstruction
   (runs -04/-06), unchanged.
2. REFLECTION channel: per-angle pitch-catch, but NOT a naive "take
   peak #1 and #2" pick (that mistake was diagnosed and corrected in
   runs -12/-13/-14) -- a proper per-trace classifier:
   a. MATCHED FILTERING (runs -14 onward) for peak extraction, not raw
      envelope.
   b. First (earliest, strongest) peak = direct outer echo. Its own
      (time, amplitude) calibrates a LOCAL off-axis-outer model for
      THIS trace (the circular law-of-cosines geometry, runs -16/-17,
      using the LOCALLY-ESTIMATED outer radius from this same peak as
      an effective local radius of curvature -- a reasonable
      approximation for real, irregular anatomy, since the off-axis
      mechanism is inherently a LOCAL beam-illumination effect, not a
      global-shape one).
   c. Every SUBSEQUENT peak is tested against the off-axis-outer
      family FIRST (excluded if consistent -- same wider-angle,
      weaker-amplitude signature confirmed in runs -16/-17) before
      being considered a genuine second-boundary (inner) candidate.
   d. Any peak surviving (c) is tested against the coefficient-derived
      amplitude-strata veto (runs -21/-22/-24) -- only accepted as a
      genuine inner echo if its amplitude is consistent with the
      predicted order-1 inner-boundary scale, calibrated to THIS
      trace's own outer baseline.
"""

import numpy as np
from scipy.signal import find_peaks
from scipy.optimize import brentq

from phase1_transmission_tomography_reconstruction import simulate_transmit_all_receivers
from phase1_matched_filter_echo_extraction import (
    simulate_pitch_catch_raw, matched_filter_output, time_to_radius_matched_filter,
    _lag_t_arr, PEAK_PROMINENCE_FRACTION,
)
from phase1_amplitude_strata_veto import compute_coefficient_strata
from phase1_patient023_validation import load_real_contours, build_medium_two_tissue, build_medium_water_only, MRI_NPZ
from phase1_reflection_channel_scout import thetas, direction_vector
from phase1_rotating_transmission_scout import probe_position, center, N, PROBE_RADIUS_CELLS, dx
import phase2_config as cfg
import tomography_recon as recon
import labels

from matplotlib import pyplot as plt
import os

IMG_SIZE = 150
_nonneg = _lag_t_arr >= 0
OFFAXIS_N_EXPONENT = 20.42  # reused from run -17's circular-phantom fit (not re-calibrated for patient023 specifically)
OFFAXIS_AMP_MARGIN = 8.0    # observed amp must be within this multiple of the off-axis prediction to count as "explained"
STRATA_VETO_MARGIN = 10.0   # same margin as runs -21/-24
MAX_PHI_SEARCH_DEG = 75.0


def offaxis_time_local(phi_rad, r_outer_local):
    dist = np.sqrt(PROBE_RADIUS_CELLS ** 2 + r_outer_local ** 2
                    - 2 * PROBE_RADIUS_CELLS * r_outer_local * np.cos(phi_rad))
    return 2 * dist * dx[0] / cfg.WATER.sound_speed


def implied_phi_local(t_observed, r_outer_local):
    t_min = offaxis_time_local(0.0, r_outer_local)
    t_max = offaxis_time_local(np.deg2rad(MAX_PHI_SEARCH_DEG), r_outer_local)
    if t_observed < t_min or t_observed > t_max:
        return None
    f = lambda phi: offaxis_time_local(phi, r_outer_local) - t_observed
    try:
        return np.degrees(brentq(f, 0.0, np.deg2rad(MAX_PHI_SEARCH_DEG)))
    except ValueError:
        return None


def classify_trace(peak_times, peak_amps, strata):
    """Per-trace blind classifier: first peak = outer; subsequent peaks
    tested against (a) the off-axis-outer family, excluded if
    consistent, then (b) the amplitude-strata veto for a genuine inner
    echo. Returns (r_outer, r_inner) or (r_outer, None)."""
    if len(peak_times) == 0:
        return None, None
    order = np.argsort(peak_times)
    times, amps = peak_times[order], peak_amps[order]

    r_outer = time_to_radius_matched_filter(times[0])
    amp_outer = amps[0]
    predicted_inner_amp = amp_outer * (strata["inner_k0"] / strata["outer"])

    for t_k, amp_k in zip(times[1:], amps[1:]):
        phi = implied_phi_local(t_k, r_outer)
        if phi is not None:
            predicted_offaxis_amp = amp_outer * np.cos(np.deg2rad(phi)) ** OFFAXIS_N_EXPONENT
            if amp_k <= OFFAXIS_AMP_MARGIN * max(predicted_offaxis_amp, 1e-12):
                continue  # explained by off-axis outer family -- not the target, keep looking
        if amp_k <= STRATA_VETO_MARGIN * predicted_inner_amp:
            return r_outer, time_to_radius_matched_filter(t_k)
        # else: too strong/inconsistent for either explanation -- skip, keep looking
    return r_outer, None


if __name__ == "__main__":
    print(f"*** {labels.PENDING_SIGNOFF_BANNER} ***")
    print(f"*** {labels.TOY_EXACT_GT_CAPTION} ***")
    print("PATIENT023 FUSED RECONSTRUCTION (all mechanisms incorporated): transmission (SIRT) "
          "+ reflection with matched filtering, off-axis-outer exclusion, and the "
          "amplitude-strata veto, all applied BLINDLY.")
    print("  compute estimate: 2 channels x 36 angles x 2 media = 144 forward sims "
          "-- ~30-40 minutes based on prior-run precedent (run sequentially here)")

    strata = compute_coefficient_strata()
    canvas_lv, canvas_myo, outer_contour_dom, inner_contour_dom = load_real_contours(MRI_NPZ)
    medium_water = build_medium_water_only()
    medium_phantom = build_medium_two_tissue(canvas_lv, canvas_myo)

    print("\n=== TRANSMISSION channel: water-only control, all 36 angles, all receivers ===")
    water_arrivals = {th: simulate_transmit_all_receivers(medium_water, th) for th in thetas}
    print("=== TRANSMISSION channel: patient023 phantom, all 36 angles, all receivers ===")
    phantom_arrivals = {th: simulate_transmit_all_receivers(medium_phantom, th) for th in thetas}

    pairs_excess_delay_ns = {}
    for theta_tx in thetas:
        for theta_rx, t_water in water_arrivals[theta_tx].items():
            if theta_rx not in phantom_arrivals[theta_tx]:
                continue
            t_phantom = phantom_arrivals[theta_tx][theta_rx]
            pairs_excess_delay_ns[(theta_tx, theta_rx)] = (t_phantom - t_water) * 1e9

    print(f"  {len(pairs_excess_delay_ns)} transmission ray paths captured")
    print("=== Reconstructing transmission channel via SIRT (30 iterations) ===")
    image_sirt, img_rows, img_cols, residual_history = recon.sirt_reconstruct(
        pairs_excess_delay_ns, probe_position, IMG_SIZE, N, n_iters=30, relax=0.15)
    print(f"  SIRT residual RMS: iter 0={residual_history[0]:.2f}ns -> iter {len(residual_history)-1}={residual_history[-1]:.2f}ns")

    print("\n=== REFLECTION channel: water-only control, pitch-catch at 36 angles ===")
    water_traces = [simulate_pitch_catch_raw(medium_water, th) for th in thetas]
    print("=== REFLECTION channel: patient023 phantom, pitch-catch at 36 angles ===")
    phantom_traces = [simulate_pitch_catch_raw(medium_phantom, th) for th in thetas]

    water_mf = [matched_filter_output(tr) for tr in water_traces]
    phantom_mf = [matched_filter_output(tr) for tr in phantom_traces]

    print("\n=== Classifying each trace (matched filter + off-axis-outer exclusion + strata veto) ===")
    refl_r_outer, refl_r_inner = [], []
    for i, theta in enumerate(thetas):
        env_w, _ = water_mf[i]
        env_p, _ = phantom_mf[i]
        thresh = max(env_w[_nonneg].max() * 3.0, env_p[_nonneg].max() * PEAK_PROMINENCE_FRACTION)
        peak_idx, _ = find_peaks(env_p[_nonneg], height=thresh)
        peak_times = _lag_t_arr[_nonneg][peak_idx]
        peak_amps = env_p[_nonneg][peak_idx]
        r_out, r_in = classify_trace(peak_times, peak_amps, strata)
        refl_r_outer.append(r_out)
        refl_r_inner.append(r_in)

    n_outer_found = sum(r is not None for r in refl_r_outer)
    n_inner_found = sum(r is not None for r in refl_r_inner)
    print(f"  outer boundary candidate found at {n_outer_found}/{len(thetas)} angles")
    print(f"  inner boundary candidate found at {n_inner_found}/{len(thetas)} angles "
          f"(after off-axis-outer exclusion + strata veto)")

    d_rows, d_cols = direction_vector(thetas)
    refl_outer_row, refl_outer_col, refl_inner_row, refl_inner_col = [], [], [], []
    for i in range(len(thetas)):
        if refl_r_outer[i] is not None:
            refl_outer_row.append(center[0] + refl_r_outer[i] * d_rows[i])
            refl_outer_col.append(center[1] + refl_r_outer[i] * d_cols[i])
        if refl_r_inner[i] is not None:
            refl_inner_row.append(center[0] + refl_r_inner[i] * d_rows[i])
            refl_inner_col.append(center[1] + refl_r_inner[i] * d_cols[i])

    fig, ax = plt.subplots(figsize=(8, 8))
    im = ax.imshow(image_sirt, cmap="hot_r", origin="upper",
                    extent=[img_cols.min(), img_cols.max(), img_rows.max(), img_rows.min()])
    ax.plot(outer_contour_dom[:, 1], outer_contour_dom[:, 0], "c--", linewidth=1.5, label="true outer (epicardium)")
    ax.plot(inner_contour_dom[:, 1], inner_contour_dom[:, 0], "lime", linestyle="--", linewidth=1.5, label="true inner (LV/blood)")
    ax.scatter(refl_outer_col, refl_outer_row, c="cyan", marker="o", s=25, edgecolor="k", linewidth=0.5,
               label=f"reflection-derived outer ({n_outer_found}/{len(thetas)})", zorder=5)
    ax.scatter(refl_inner_col, refl_inner_row, c="lime", marker="s", s=25, edgecolor="k", linewidth=0.5,
               label=f"reflection-derived inner ({n_inner_found}/{len(thetas)})", zorder=5)
    ax.set_title("FUSED (all mechanisms): transmission SIRT + BLIND reflection\n"
                 "(matched filter + off-axis-outer exclusion + strata veto), patient023")
    ax.legend(fontsize=8, loc="upper right")
    plt.colorbar(im, ax=ax, label="mean excess delay (ns)", shrink=0.7)

    labels.add_banner(fig)
    plt.tight_layout()
    os.makedirs("results/figures", exist_ok=True)
    out_fig = "results/figures/phase1_patient023_fused_reconstruction.png"
    plt.savefig(out_fig, dpi=140)
    print(f"\nSaved {out_fig}")
