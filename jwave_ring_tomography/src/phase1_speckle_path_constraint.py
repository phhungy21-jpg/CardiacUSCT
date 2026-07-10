"""Per user: reflection/refraction has "many suitable candidates along
the function" (many plausible peaks per angle -- off-axis spillover,
reverberation, the genuine inner echo, etc., runs -36/-37/-38's core
problem) -- "but since speckle have located the object, can't it
localise the sound path?"

Previously (runs -36/-37/-38), speckle was only ever used as a
POINTWISE "rising-edge score" evaluated AT each reflection candidate's
own implied location -- never as an independent GEOMETRIC constraint in
its own right. This tests the more direct idea: extract the speckle
field's OWN per-angle radial localization (where does the backscatter
intensity itself rise from blood-like/low to wall-like/high, along each
angle's own ray?) as a standalone geometric estimate of the inner
boundary, then use THAT location -- not amplitude -- to pick which
reflection-channel candidate is real.

Three comparisons, all on patient023's real anatomy, all reusing cached
data (no new jWave simulation):
1. Speckle-alone: does the per-angle radial profile's rising edge, by
   itself, estimate the true inner boundary?
2. Speckle-constrained candidate selection: pick the reflection
   candidate closest to the speckle-derived location (not the naive
   strongest-amplitude one).
3. Compare both to run -38's amplitude+smoothness result (corr=0.566)
   and run -36's naive amplitude baseline (corr=0.262 uncorrected /
   0.272 corrected).
"""

import numpy as np
from scipy.signal import correlate, hilbert, find_peaks

from phase1_patient023_validation import load_real_contours, MRI_NPZ
from phase1_reflection_channel_scout import (
    thetas, direction_vector, polar_resample, r_at_theta, pitch_catch_positions,
)
from phase1_matched_filter_echo_extraction import _lag_t_arr, _template, PEAK_PROMINENCE_FRACTION
from phase1_das_reflectivity_imaging import extract_boundary_from_image
from phase1_speckle_informed_surface_selection_v2 import das_energy_field
from phase1_rotating_transmission_scout import center, PROBE_RADIUS_CELLS, dx
import phase2_config as cfg
import labels

from matplotlib import pyplot as plt
import os

R_MAX_CELLS = 0.9 * PROBE_RADIUS_CELLS
MIN_SEP_CELLS = 6.0
R_STEP = 0.5
_nonneg = _lag_t_arr >= 0


def matched_filter_envelope(trace):
    correlated = correlate(trace, _template, mode="full")
    return np.abs(hilbert(correlated))


def time_to_radius_two_leg(t, r_outer_cells):
    leg_water_m = (PROBE_RADIUS_CELLS - r_outer_cells) * dx[0]
    t_half = t / 2.0
    leg_tissue_m = cfg.MYOCARDIUM.sound_speed * (t_half - leg_water_m / cfg.WATER.sound_speed)
    return r_outer_cells - leg_tissue_m / dx[0]


def generate_candidates(env_water, env_phantom, r_outer_known):
    thresh = max(env_water[_nonneg].max() * 3.0, env_phantom[_nonneg].max() * PEAK_PROMINENCE_FRACTION)
    peak_idx, _ = find_peaks(env_phantom[_nonneg], height=thresh)
    if len(peak_idx) == 0:
        return []
    times = _lag_t_arr[_nonneg][peak_idx]
    amps = env_phantom[_nonneg][peak_idx]
    radii = time_to_radius_two_leg(times, r_outer_known)
    keep = (radii > 0) & (radii <= r_outer_known - MIN_SEP_CELLS)
    return list(zip(radii[keep], amps[keep]))


def sample_field(field, img_rows, img_cols, row, col):
    ri = np.argmin(np.abs(img_rows - row))
    ci = np.argmin(np.abs(img_cols - col))
    return field[ri, ci]


def speckle_rising_edge_radius(speckle_field, img_rows, img_cols, theta, r_min=20.0, r_max=R_MAX_CELLS):
    """Per-angle radial profile of the speckle field along this angle's
    own ray from center; returns the radius where intensity crosses
    halfway between the profile's own local min and max (a "rising
    edge" -- transition from blood-like/low to wall-like/high)."""
    d_row, d_col = direction_vector(theta)
    r_vals = np.arange(r_min, r_max, R_STEP)
    vals = np.array([sample_field(speckle_field, img_rows, img_cols,
                                   center[0] + r * d_row, center[1] + r * d_col) for r in r_vals])
    vmin, vmax = vals.min(), vals.max()
    if vmax - vmin < 1e-12:
        return np.nan
    half = vmin + 0.5 * (vmax - vmin)
    above = vals >= half
    crossings = np.where(np.diff(above.astype(int)) == 1)[0]
    if len(crossings) == 0:
        return np.nan
    return r_vals[crossings[0]]


if __name__ == "__main__":
    print(f"*** {labels.PENDING_SIGNOFF_BANNER} ***")
    print(f"*** {labels.TOY_EXACT_GT_CAPTION} ***")
    print("SPECKLE AS A GEOMETRIC PATH CONSTRAINT: extracts the speckle field's OWN per-angle "
          "radial rising-edge location as a standalone estimate, then uses it to select among "
          "reflection candidates (rather than amplitude). No new jWave simulation -- reuses "
          "cached patient023 data (runs -32/-36).")

    canvas_lv, canvas_myo, outer_contour_dom, inner_contour_dom = load_real_contours(MRI_NPZ)
    ext_theta_in, ext_r_in = polar_resample(inner_contour_dom, center)
    true_r_in = np.array([r_at_theta(th, ext_theta_in, ext_r_in) for th in thetas])

    d_das = np.load("results/patient023_das_images.npz")
    image_sr, img_rows_das, img_cols_das = d_das["image_straight_ray"], d_das["img_rows"], d_das["img_cols"]
    r_outer_est = extract_boundary_from_image(image_sr, img_rows_das, img_cols_das, center, thetas, r_max_cells=R_MAX_CELLS)

    d_refl = np.load("results/patient023_reflection_raw_traces.npz")
    water_traces, homog_traces = d_refl["water_traces"], d_refl["phantom_traces"]
    d_speckle = np.load("results/patient023_speckle_raw_traces.npz")
    speckle_traces = d_speckle["speckle_traces"]
    water_env = [matched_filter_envelope(tr) for tr in water_traces]
    homog_env = [matched_filter_envelope(tr) for tr in homog_traces]
    speckle_env = [matched_filter_envelope(tr) for tr in speckle_traces]

    print("\n=== Building incoherent (energy) speckle field ===")
    speckle_field, img_rows, img_cols = das_energy_field(speckle_env, homog_env)

    print("=== Extracting per-angle speckle rising-edge radius ===")
    r_speckle = np.array([speckle_rising_edge_radius(speckle_field, img_rows, img_cols, th) for th in thetas])
    valid_speckle = ~np.isnan(r_speckle)
    corr_speckle = np.corrcoef(r_speckle[valid_speckle], true_r_in[valid_speckle])[0, 1]
    rmse_speckle_mm = np.sqrt(np.mean((r_speckle[valid_speckle] - true_r_in[valid_speckle]) ** 2)) * cfg.DX_M * 1e3
    print(f"  speckle-alone: {valid_speckle.sum()}/{len(thetas)} valid, "
          f"corr={corr_speckle:.3f}, RMSE={rmse_speckle_mm:.3f}mm")

    print("\n=== Speckle-constrained candidate selection (vs. naive amplitude) ===")
    r_naive, r_speckle_constrained = [], []
    for i, theta in enumerate(thetas):
        cands = generate_candidates(water_env[i], homog_env[i], r_outer_est[i])
        if not cands:
            r_naive.append(np.nan)
            r_speckle_constrained.append(np.nan)
            continue
        r_naive.append(max(cands, key=lambda ra: ra[1])[0])
        if np.isnan(r_speckle[i]):
            r_speckle_constrained.append(np.nan)
        else:
            radii = np.array([r for r, _ in cands])
            best = np.argmin(np.abs(radii - r_speckle[i]))
            r_speckle_constrained.append(radii[best])

    r_naive, r_speckle_constrained = np.array(r_naive), np.array(r_speckle_constrained)

    def eval_method(r_est, name):
        valid = ~np.isnan(r_est)
        corr = np.corrcoef(r_est[valid], true_r_in[valid])[0, 1]
        rmse_mm = np.sqrt(np.mean((r_est[valid] - true_r_in[valid]) ** 2)) * cfg.DX_M * 1e3
        print(f"  {name}: {valid.sum()}/{len(thetas)}, corr={corr:.3f}, RMSE={rmse_mm:.3f}mm")
        return corr, rmse_mm

    eval_method(r_naive, "naive strongest-amplitude (baseline)")
    eval_method(r_speckle_constrained, "speckle-CONSTRAINED candidate selection")
    print(f"\n  (compare: run -38 amplitude+smoothness joint optimization: corr=0.566, RMSE~1.9mm)")

    d_rows, d_cols = direction_vector(thetas)
    fig, axes = plt.subplots(1, 2, figsize=(13, 5.5))
    im = axes[0].imshow(speckle_field, cmap="hot", origin="upper",
                         extent=[img_cols.min(), img_cols.max(), img_rows.max(), img_rows.min()])
    valid = ~np.isnan(r_speckle)
    axes[0].scatter([center[1] + r_speckle[i] * d_cols[i] for i in range(len(thetas)) if valid[i]],
                     [center[0] + r_speckle[i] * d_rows[i] for i in range(len(thetas)) if valid[i]],
                     c="cyan", marker="o", s=20, edgecolor="k", linewidth=0.4, label="speckle rising-edge")
    ext_theta_out, ext_r_out = polar_resample(outer_contour_dom, center)
    true_out_row = [center[0] + r_at_theta(th, ext_theta_out, ext_r_out) * direction_vector(th)[0] for th in thetas]
    true_out_col = [center[1] + r_at_theta(th, ext_theta_out, ext_r_out) * direction_vector(th)[1] for th in thetas]
    axes[0].plot(true_out_col + [true_out_col[0]], true_out_row + [true_out_row[0]], "c--", linewidth=1, label="true outer")
    true_in_row = [center[0] + true_r_in[i] * d_rows[i] for i in range(len(thetas))]
    true_in_col = [center[1] + true_r_in[i] * d_cols[i] for i in range(len(thetas))]
    axes[0].plot(true_in_col + [true_in_col[0]], true_in_row + [true_in_row[0]], "b--", linewidth=1, label="true inner")
    axes[0].set_title(f"Speckle field + rising-edge estimate\ncorr={corr_speckle:.2f}, RMSE={rmse_speckle_mm:.2f}mm")
    axes[0].legend(fontsize=7)

    axes[1].plot(thetas, true_r_in, "k-", linewidth=2, label="true inner contour")
    axes[1].plot(thetas, r_naive, "o-", color="red", alpha=0.6, label="naive amplitude")
    axes[1].plot(thetas, r_speckle_constrained, "s-", color="lime", alpha=0.8, label="speckle-constrained")
    axes[1].set_xlabel("angle (deg)")
    axes[1].set_ylabel("inner-boundary radius estimate (cells)")
    axes[1].legend(fontsize=8)
    axes[1].set_title("Per-angle: naive vs. speckle-constrained candidate selection")

    fig.suptitle("Speckle as a geometric path constraint for reflection-candidate selection (patient023)")
    labels.add_banner(fig)
    plt.tight_layout()
    os.makedirs("results/figures", exist_ok=True)
    plt.savefig("results/figures/phase1_speckle_path_constraint.png", dpi=140)
    print("\nSaved results/figures/phase1_speckle_path_constraint.png")
