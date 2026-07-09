"""Phase 1 scout, FUSING channel 0 (transmission) + channel 1
(reflection): per user, "proceed, and do visualise this one for me to
see" -- combines run -07's transmission-based SIRT reconstruction
(reused from its saved dataset, no resimulation) with a genuinely BLIND
per-angle reflection boundary estimate (not run -08's known-location
check -- this time finds peaks without being told where to look, the
fair test of whether reflection alone could locate a boundary in
practice).

Reflection boundary estimate, per angle: take the two most prominent
envelope peaks after the direct-arrival exclusion, convert their
arrival times to radial distance assuming an ALL-WATER round trip (the
simplest assumption a real device without prior tissue knowledge would
make). This is a KNOWN, honest source of bias for the inner boundary:
part of that round trip actually happened through FASTER myocardium,
so a water-only time-to-distance conversion UNDERESTIMATES the true
inner radius (the same conversion assumes slower propagation than
what really happened for that leg) -- exactly the "refraction/sound-
speed-model" limitation flagged in the multi-channel roadmap's channel
3 entry. Reported explicitly, not hidden.
"""

import numpy as np
from scipy.signal import find_peaks

from phase1_reflection_channel_scout import (
    simulate_pitch_catch, thetas, DIRECT_EXCLUDE_MARGIN_S, _ENVELOPE_GROUP_DELAY_S,
    polar_resample, r_at_theta,
)
from phase1_two_tissue_reconstruction import load_real_contours_two_tissue, build_medium_two_tissue, build_medium_water_only
from phase1_rotating_transmission_scout import direction_vector, dx, center, N, PROBE_RADIUS_CELLS, t_arr
import phase2_config as cfg
import tomography_recon as recon
import labels

from matplotlib import pyplot as plt
import os

IMG_SIZE = 150
PEAK_PROMINENCE_FRACTION = 0.05  # relative to that angle's own max envelope value


def blind_two_peak_distances(envelope, water_baseline_max):
    thresh = max(water_baseline_max * 3.0, envelope.max() * PEAK_PROMINENCE_FRACTION)
    peak_idx, props = find_peaks(envelope, height=thresh)
    if len(peak_idx) == 0:
        return None, None
    order = np.argsort(t_arr[peak_idx])  # chronological: first hit = outer, second = inner
    peak_idx_sorted = peak_idx[order]
    times = t_arr[peak_idx_sorted]

    def time_to_radius(t):
        # ALL-WATER round-trip assumption (see module docstring for the known bias this introduces)
        one_way_m = cfg.WATER.sound_speed * (t - _ENVELOPE_GROUP_DELAY_S) / 2.0
        return PROBE_RADIUS_CELLS - one_way_m / dx[0]

    r_outer = time_to_radius(times[0]) if len(times) >= 1 else None
    r_inner = time_to_radius(times[1]) if len(times) >= 2 else None
    return r_outer, r_inner


if __name__ == "__main__":
    print(f"*** {labels.PENDING_SIGNOFF_BANNER} ***")
    print(f"*** {labels.TOY_EXACT_GT_CAPTION} ***")
    print("PHASE 1: FUSING transmission (channel 0, reused from run -07's saved data) "
          "+ reflection (channel 1, BLIND per-angle peak detection this time, not "
          "checked against known locations like run -08).")
    print("  compute estimate: 36 angles x 2 media = 72 forward sims for reflection "
          "(transmission reused, free) -- ~15-20 minutes based on prior-run precedent")

    canvas_lv, canvas_myo, outer_contour_dom, inner_contour_dom, max_radius_cells = load_real_contours_two_tissue()
    ext_theta_out, ext_r_out = polar_resample(outer_contour_dom, center)
    ext_theta_in, ext_r_in = polar_resample(inner_contour_dom, center)

    medium_water = build_medium_water_only()
    medium_phantom = build_medium_two_tissue(canvas_lv, canvas_myo)

    print("\n=== Simulating water-only control, pitch-catch at 36 angles ===")
    water_envelopes = [simulate_pitch_catch(medium_water, th) for th in thetas]
    print("=== Simulating two-tissue phantom, pitch-catch at 36 angles ===")
    phantom_envelopes = [simulate_pitch_catch(medium_phantom, th) for th in thetas]

    print("\n=== Blind per-angle reflection peak detection (no known-location check) ===")
    refl_r_outer, refl_r_inner = [], []
    for i, theta in enumerate(thetas):
        water_max = water_envelopes[i].max()
        r_out, r_in = blind_two_peak_distances(phantom_envelopes[i], water_max)
        refl_r_outer.append(r_out)
        refl_r_inner.append(r_in)
    n_outer_found = sum(r is not None for r in refl_r_outer)
    n_inner_found = sum(r is not None for r in refl_r_inner)
    print(f"  outer boundary candidate found at {n_outer_found}/{len(thetas)} angles")
    print(f"  inner boundary candidate found at {n_inner_found}/{len(thetas)} angles")

    print("\n=== Loading transmission dataset from run -07 (reused, not resimulated) ===")
    d = np.load("results/patient001_two_tissue_rays.npz")
    pairs_excess_delay_ns = {(tt, tr): v for tt, tr, v in zip(d["theta_tx"], d["theta_rx"], d["excess_delay_ns"])}
    from phase1_rotating_transmission_scout import probe_position
    print("=== Reconstructing transmission channel via SIRT (30 iterations) ===")
    image_sirt, img_rows, img_cols, residual_history = recon.sirt_reconstruct(
        pairs_excess_delay_ns, probe_position, IMG_SIZE, N, n_iters=30, relax=0.15)

    d_rows, d_cols = direction_vector(thetas)
    refl_outer_row, refl_outer_col = [], []
    refl_inner_row, refl_inner_col = [], []
    for i, theta in enumerate(thetas):
        if refl_r_outer[i] is not None:
            refl_outer_row.append(center[0] + refl_r_outer[i] * d_rows[i])
            refl_outer_col.append(center[1] + refl_r_outer[i] * d_cols[i])
        if refl_r_inner[i] is not None:
            refl_inner_row.append(center[0] + refl_r_inner[i] * d_rows[i])
            refl_inner_col.append(center[1] + refl_r_inner[i] * d_cols[i])

    # Honest bias check: compare blind reflection-derived radii (all-water assumption)
    # against the TRUE radii, to quantify (not just predict) the expected undershoot.
    true_r_outer_by_angle = [r_at_theta(th, ext_theta_out, ext_r_out) for th in thetas]
    true_r_inner_by_angle = [r_at_theta(th, ext_theta_in, ext_r_in) for th in thetas]
    outer_err = np.array([r - t for r, t in zip(refl_r_outer, true_r_outer_by_angle) if r is not None])
    inner_matched_true = [t for r, t in zip(refl_r_inner, true_r_inner_by_angle) if r is not None]
    inner_err = np.array([r - t for r, t in zip(refl_r_inner, inner_matched_true) if r is not None])
    print(f"\n--- Blind reflection radius bias (all-water time-to-distance assumption) ---")
    print(f"  outer boundary: mean error={outer_err.mean():+.1f} cells (water-path leg, should be near 0)")
    print(f"  inner boundary: mean error={inner_err.mean():+.1f} cells "
          f"(expected UNDERSHOOT -- part of that path was actually through faster myocardium)")

    fig, ax = plt.subplots(figsize=(8, 8))
    im = ax.imshow(image_sirt, cmap="hot_r", origin="upper",
                    extent=[img_cols.min(), img_cols.max(), img_rows.max(), img_rows.min()])
    ax.plot(outer_contour_dom[:, 1], outer_contour_dom[:, 0], "c--", linewidth=1.5, label="true outer (epicardium)")
    ax.plot(inner_contour_dom[:, 1], inner_contour_dom[:, 0], "lime", linestyle="--", linewidth=1.5, label="true inner (LV/blood)")
    ax.scatter(refl_outer_col, refl_outer_row, c="cyan", marker="o", s=25, edgecolor="k", linewidth=0.5,
               label=f"reflection-derived outer ({n_outer_found}/{len(thetas)})", zorder=5)
    ax.scatter(refl_inner_col, refl_inner_row, c="lime", marker="s", s=25, edgecolor="k", linewidth=0.5,
               label=f"reflection-derived inner ({n_inner_found}/{len(thetas)})", zorder=5)
    ax.set_title("FUSED: transmission SIRT image (background) + BLIND reflection-derived\n"
                 "boundary points (markers), patient001 two-tissue, static")
    ax.legend(fontsize=8, loc="upper right")
    plt.colorbar(im, ax=ax, label="mean excess delay (ns)", shrink=0.7)

    labels.add_banner(fig)
    plt.tight_layout()
    os.makedirs("results/figures", exist_ok=True)
    out_fig = "results/figures/phase1_fused_channel_reconstruction.png"
    plt.savefig(out_fig, dpi=140)
    print(f"\nSaved {out_fig}")
