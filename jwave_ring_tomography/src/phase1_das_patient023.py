"""Retests DAS (delay-and-sum) reflectivity imaging (run -31) on REAL,
irregular, TWO-TISSUE anatomy: patient023 (myocardium wall + LV/blood
cavity), the same patient this project's single-peak reflection method
has struggled with since run -07/-08 -- the inner (blood/myocardium,
~0.5% sound-speed contrast, ~0.25% reflection-coefficient) boundary has
been this project's stubborn weak point, detected inconsistently and
only ever validated via narrow known-location search windows or the
amplitude-strata veto, never as a genuinely standalone blind detector.

Per user: "go ahead. see if this time inner wall is good" -- testing
directly whether DAS's cross-angle accumulation (many weak-but-
consistent echoes reinforcing each other across 36 firing angles) can
pull the inner boundary's genuinely weak signal above the noise floor
in a way single-shot peak-picking never reliably could, since DAS does
not require any one shot's peak to be strong or unambiguous on its own.

Builds the reflectivity image ONCE and looks for BOTH boundaries in the
SAME image (unlike every single-peak method in this project, which
predicted specific outer/inner arrival times and searched narrow windows
around each) -- a genuinely blind, dual-boundary radial peak search per
angle.

Runs a FRESH pitch-catch simulation (this project has never saved raw
reflection traces for patient023 before, only classified peaks) and
reuses the ALREADY-SIMULATED transmission-channel data (`results/
patient023_transmission_rays.npz`, run -29) for the bent-ray upgrade's
sound-speed field, no new transmission simulation needed.
"""

import numpy as np
from scipy.signal import correlate, hilbert, find_peaks

from phase1_patient023_validation import (
    load_real_contours, build_medium_two_tissue, build_medium_water_only, MRI_NPZ,
)
from phase1_reflection_channel_scout import (
    thetas, pitch_catch_positions, direction_vector, polar_resample, r_at_theta,
)
from phase1_matched_filter_echo_extraction import _lag_t_arr, _template
from phase1_das_reflectivity_imaging import simulate_pitch_catch_raw_at, das_straight_ray_image
from phase1_bent_ray_correction import nearest_grid_index, fmm_travel_time_field, _threshold_and_clean
from phase1_rotating_transmission_scout import probe_position, dx, N, PROBE_RADIUS_CELLS
import tomography_recon as recon
import phase2_config as cfg
import labels

from matplotlib import pyplot as plt
import os

IMG_SIZE = 150
THRESHOLD_FRACTION = 0.3
_nonneg = _lag_t_arr >= 0


def matched_filter_envelope(trace):
    correlated = correlate(trace, _template, mode="full")
    return np.abs(hilbert(correlated))


def extract_two_boundaries_from_image(image, img_rows, img_cols, origin, thetas_arr,
                                       r_min_cells=20.0, r_max_cells=140.0, step_cells=0.5,
                                       min_sep_cells=6.0, min_inner_prominence_frac=0.05):
    """BLIND per-angle dual-boundary extraction: for each angle, find ALL
    local maxima along the radial intensity profile. The OUTERMOST
    sufficiently-prominent peak is the outer-boundary candidate (nothing
    physically should reflect from farther out than the true exterior
    wall). The strongest peak at least `min_sep_cells` INSIDE that is the
    inner-boundary candidate, if its own peak height clears
    `min_inner_prominence_frac` of the outer peak's height (otherwise:
    not detected, reported honestly rather than forced)."""
    def sample(rr, cc):
        ri = np.argmin(np.abs(img_rows - rr))
        ci = np.argmin(np.abs(img_cols - cc))
        return image[ri, ci]

    r_vals = np.arange(r_min_cells, r_max_cells, step_cells)
    r_outer, r_inner = [], []
    for theta in thetas_arr:
        d_row, d_col = direction_vector(theta)
        vals = np.array([sample(origin[0] + r * d_row, origin[1] + r * d_col) for r in r_vals])
        peak_idx, props = find_peaks(vals, prominence=vals.max() * 0.03)
        if len(peak_idx) == 0:
            r_outer.append(np.nan)
            r_inner.append(np.nan)
            continue
        peak_r = r_vals[peak_idx]
        peak_h = vals[peak_idx]
        outer_i = int(np.argmax(peak_r))  # outermost candidate = outer boundary
        r_out = peak_r[outer_i]
        h_out = peak_h[outer_i]
        r_outer.append(r_out)

        inner_candidates = [(r, h) for r, h in zip(peak_r, peak_h)
                             if r <= r_out - min_sep_cells]
        if inner_candidates:
            r_in, h_in = max(inner_candidates, key=lambda rh: rh[1])
            if h_in >= min_inner_prominence_frac * h_out:
                r_inner.append(r_in)
            else:
                r_inner.append(np.nan)
        else:
            r_inner.append(np.nan)
    return np.array(r_outer), np.array(r_inner)


if __name__ == "__main__":
    print(f"*** {labels.PENDING_SIGNOFF_BANNER} ***")
    print(f"*** {labels.TOY_EXACT_GT_CAPTION} ***")
    print("DAS REFLECTIVITY IMAGING: patient023 real two-tissue anatomy. Does cross-angle "
          "accumulation reveal the inner (blood/myocardium) boundary that single-peak "
          "detection (runs -08/-09/-22/-25) has struggled with since this project began?")
    print("  compute estimate: 36 angles x 2 media (pitch-catch reflection) = 72 forward sims "
          "-- ~15-20 minutes based on prior-run precedent")

    from phase1_rotating_transmission_scout import center
    canvas_lv, canvas_myo, outer_contour_dom, inner_contour_dom = load_real_contours(MRI_NPZ)
    ext_theta_out, ext_r_out = polar_resample(outer_contour_dom, center)
    ext_theta_in, ext_r_in = polar_resample(inner_contour_dom, center)

    medium_water = build_medium_water_only()
    medium_phantom = build_medium_two_tissue(canvas_lv, canvas_myo)

    print("\n=== REFLECTION channel: water-only control, pitch-catch at 36 angles (raw traces) ===")
    water_traces = [simulate_pitch_catch_raw_at(medium_water, th) for th in thetas]
    print("=== REFLECTION channel: patient023 two-tissue phantom, pitch-catch at 36 angles ===")
    phantom_traces = [simulate_pitch_catch_raw_at(medium_phantom, th) for th in thetas]

    os.makedirs("results", exist_ok=True)
    np.savez("results/patient023_reflection_raw_traces.npz",
             thetas=thetas, water_traces=np.array(water_traces), phantom_traces=np.array(phantom_traces))
    print("  saved results/patient023_reflection_raw_traces.npz")

    water_env = [matched_filter_envelope(tr) for tr in water_traces]
    phantom_env = [matched_filter_envelope(tr) for tr in phantom_traces]

    print("\n=== Building straight-ray DAS reflectivity image (36-angle accumulation) ===")
    image_sr, img_rows, img_cols = das_straight_ray_image(phantom_env, water_env, img_size=IMG_SIZE)

    print("=== Rebuilding the CLEAN straight-ray SIRT sound-speed field (transmission channel) ===")
    d_tx = np.load("results/patient023_transmission_rays.npz")
    pairs_excess_delay_ns = {(tt, tr): v for tt, tr, v in zip(d_tx["theta_tx"], d_tx["theta_rx"], d_tx["excess_delay_ns"])}
    image_sirt, img_rows2, img_cols2, _ = recon.sirt_reconstruct(
        pairs_excess_delay_ns, probe_position, IMG_SIZE, N, n_iters=30, relax=0.15)
    is_tissue = _threshold_and_clean(image_sirt, THRESHOLD_FRACTION)
    sound_speed_grid = np.where(is_tissue, cfg.MYOCARDIUM.sound_speed, cfg.WATER.sound_speed).astype(np.float64)
    print(f"  sound-speed field ready ({is_tissue.mean()*100:.1f}% tissue fraction)")

    print("=== Solving the eikonal equation from every pitch-catch src/rcv position (72 FMM solves) ===")
    cell_size_m = ((img_rows[1] - img_rows[0]) * dx[0], (img_cols[1] - img_cols[0]) * dx[0])
    accumulator_br = np.zeros((IMG_SIZE, IMG_SIZE))
    for i, theta in enumerate(thetas):
        src, rcv = pitch_catch_positions(theta)
        sr, sc = nearest_grid_index(src[0], src[1], img_rows, img_cols)
        rr_, rc_ = nearest_grid_index(rcv[0], rcv[1], img_rows, img_cols)
        t_src = fmm_travel_time_field(sound_speed_grid, sr, sc, cell_size_m)
        t_rcv = fmm_travel_time_field(sound_speed_grid, rr_, rc_, cell_size_m)
        t_pred = t_src + t_rcv
        excess_env = np.clip(phantom_env[i] - water_env[i], 0, None)
        sampled = np.interp(t_pred.ravel(), _lag_t_arr, excess_env, left=0.0, right=0.0)
        accumulator_br += sampled.reshape(IMG_SIZE, IMG_SIZE)

    print("\n=== BLIND dual-boundary extraction (both channels) ===")
    true_r_out = np.array([r_at_theta(th, ext_theta_out, ext_r_out) for th in thetas])
    true_r_in = np.array([r_at_theta(th, ext_theta_in, ext_r_in) for th in thetas])

    for tag, image in [("straight-ray", image_sr), ("bent-ray", accumulator_br)]:
        r_out, r_in = extract_two_boundaries_from_image(image, img_rows, img_cols, center, thetas)
        outer_valid = ~np.isnan(r_out)
        inner_valid = ~np.isnan(r_in)
        outer_rmse = np.sqrt(np.mean(((r_out[outer_valid] - true_r_out[outer_valid]) * cfg.DX_M * 1e3) ** 2))
        print(f"\n--- {tag} DAS result ---")
        print(f"  outer boundary: {outer_valid.sum()}/{len(thetas)} detected, RMSE={outer_rmse:.4f}mm")
        if inner_valid.sum() > 0:
            inner_rmse = np.sqrt(np.mean(((r_in[inner_valid] - true_r_in[inner_valid]) * cfg.DX_M * 1e3) ** 2))
            print(f"  INNER boundary: {inner_valid.sum()}/{len(thetas)} detected, RMSE={inner_rmse:.4f}mm")
        else:
            print(f"  INNER boundary: 0/{len(thetas)} detected")
        if tag == "bent-ray":
            r_out_final, r_in_final = r_out, r_in
            outer_rmse_final, inner_valid_final = outer_rmse, inner_valid

    print(f"\n  (compare: prior single-peak inner-boundary results -- run -08: 25/36 'detectable' via "
          f"known-location search window (not blind); run -22/-26: amplitude-strata veto applied to "
          f"candidate matches, spatially inconsistent; no prior run has reported a genuine blind "
          f"inner-boundary RMSE on real anatomy)")

    d_rows, d_cols = direction_vector(thetas)
    fig, axes = plt.subplots(1, 2, figsize=(13, 5.5))
    for ax, (tag, image, r_out, r_in) in zip(axes, [
        ("straight-ray", image_sr, *extract_two_boundaries_from_image(image_sr, img_rows, img_cols, center, thetas)),
        ("bent-ray", accumulator_br, r_out_final, r_in_final),
    ]):
        im = ax.imshow(image, cmap="hot", origin="upper",
                        extent=[img_cols.min(), img_cols.max(), img_rows.max(), img_rows.min()])
        h_row = [center[0] + r * d for r, d in zip(true_r_out, d_rows)]
        h_col = [center[1] + r * d for r, d in zip(true_r_out, d_cols)]
        ax.plot(h_col + [h_col[0]], h_row + [h_row[0]], "c--", linewidth=1.2, label="true outer")
        hi_row = [center[0] + r * d for r, d in zip(true_r_in, d_rows)]
        hi_col = [center[1] + r * d for r, d in zip(true_r_in, d_cols)]
        ax.plot(hi_col + [hi_col[0]], hi_row + [hi_row[0]], "b--", linewidth=1.2, label="true inner")
        outer_valid = ~np.isnan(r_out)
        inner_valid = ~np.isnan(r_in)
        ax.scatter([center[1] + r * d_cols[i] for i, r in enumerate(r_out) if outer_valid[i]],
                   [center[0] + r * d_rows[i] for i, r in enumerate(r_out) if outer_valid[i]],
                   c="lime", marker="s", s=15, edgecolor="k", linewidth=0.3, label="DAS outer", zorder=5)
        ax.scatter([center[1] + r * d_cols[i] for i, r in enumerate(r_in) if inner_valid[i]],
                   [center[0] + r * d_rows[i] for i, r in enumerate(r_in) if inner_valid[i]],
                   c="yellow", marker="o", s=15, edgecolor="k", linewidth=0.3, label="DAS inner", zorder=5)
        ax.set_title(f"{tag} DAS ({inner_valid.sum()}/{len(thetas)} inner detected)")
        ax.legend(fontsize=6, loc="upper right")

    fig.suptitle("DAS reflectivity imaging, patient023 real two-tissue anatomy: does inner-wall detection improve?")
    labels.add_banner(fig)
    plt.tight_layout()
    os.makedirs("results/figures", exist_ok=True)
    out_fig = "results/figures/phase1_das_patient023.png"
    plt.savefig(out_fig, dpi=140)
    print(f"\nSaved {out_fig}")
