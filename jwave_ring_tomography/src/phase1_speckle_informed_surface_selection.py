"""Direct test of the user's proposed integration: "Reflection/
transmission propose surfaces. Speckle validates the tissue region
between surfaces." Per the user's specific immediate experiment: "Take
your existing reflection-derived candidate boundaries and rescore them
with a speckle wall-likelihood term. If the correct inner contour gets
better ranking than the false reflection candidates, then speckle has
become useful."

Two things run -32 established, reused here unchanged: (1) the outer
boundary is reliably recovered (RMSE=1.04mm) from the DAS reflectivity
image; (2) naive single-peak (or naive DAS-peak) INNER-boundary
selection is essentially noise (correlation=0.25 with the true contour)
-- there are multiple candidate peaks per angle (off-axis outer-wall
spillover, reverberation, genuine inner echo, etc.) and no reliable way
to tell them apart from amplitude/timing alone.

This script: (1) generates MULTIPLE inner-boundary candidates per angle
(not just the single strongest peak) from the classic single-shot
pitch-catch matched-filter trace; (2) builds the INCOHERENT (energy)
speckle field (run -35's fix) for patient023's real anatomy; (3) scores
each candidate by a RISING-EDGE speckle likelihood -- a genuine inner
boundary should have LOW speckle energy just inside it (blood, no
scattering) and HIGH speckle energy just outside it (myocardium wall,
real scattering) -- and picks the top-scoring candidate per angle;
(4) compares the speckle-rescored selection's accuracy against the
naive "strongest amplitude peak" selection, the same comparison the
user's proposed experiment calls for.

No new jWave simulation beyond `phase1_speckle_patient023_sim.py`'s 36
sims -- reuses cached homogeneous/water traces (run -32) and the new
speckle traces, pure post-processing otherwise.
"""

import numpy as np
from scipy.signal import correlate, hilbert, find_peaks

from phase1_patient023_validation import load_real_contours, MRI_NPZ
from phase1_reflection_channel_scout import (
    thetas, direction_vector, polar_resample, r_at_theta, pitch_catch_positions,
)
from phase1_matched_filter_echo_extraction import (
    _lag_t_arr, _template, time_to_radius_matched_filter, PEAK_PROMINENCE_FRACTION,
)
from phase1_rotating_transmission_scout import center, N, dx, PROBE_RADIUS_CELLS
import phase2_config as cfg
import labels

from matplotlib import pyplot as plt
import os

IMG_SIZE = 150
R_MAX_CELLS = 0.9 * PROBE_RADIUS_CELLS  # generic physical constraint (run -32's fix), excludes near-probe artifacts
MIN_SEP_CELLS = 6.0
EDGE_MARGIN_CELLS = 4.0  # how far inside/outside a candidate to sample the speckle rising-edge score
_nonneg = _lag_t_arr >= 0


def matched_filter_envelope(trace):
    correlated = correlate(trace, _template, mode="full")
    return np.abs(hilbert(correlated))


def generate_inner_candidates(env_water, env_phantom):
    """BLIND multi-candidate generation from a single pitch-catch A-scan:
    finds ALL sufficiently-prominent peaks, converts each to a candidate
    radius (all-water round-trip assumption, this project's standard
    convention since run -09), takes the outermost credible one as the
    outer-boundary estimate, and returns every OTHER candidate inward of
    it (not just the single strongest) as inner-boundary candidates."""
    thresh = max(env_water[_nonneg].max() * 3.0, env_phantom[_nonneg].max() * PEAK_PROMINENCE_FRACTION)
    peak_idx, props = find_peaks(env_phantom[_nonneg], height=thresh)
    if len(peak_idx) == 0:
        return None, []
    times = _lag_t_arr[_nonneg][peak_idx]
    amps = env_phantom[_nonneg][peak_idx]
    radii = time_to_radius_matched_filter(times)
    keep = (radii > 0) & (radii <= R_MAX_CELLS)
    radii, amps = radii[keep], amps[keep]
    if len(radii) == 0:
        return None, []
    outer_i = int(np.argmax(radii))
    r_outer = radii[outer_i]
    candidates = [(r, a) for r, a in zip(radii, amps) if r <= r_outer - MIN_SEP_CELLS]
    return r_outer, candidates


def das_energy_field(speckle_env, homog_env, img_size=IMG_SIZE):
    img_rows = np.linspace(0, N[0], img_size)
    img_cols = np.linspace(0, N[1], img_size)
    RR, CC = np.meshgrid(img_rows, img_cols, indexing="ij")
    accumulator = np.zeros((img_size, img_size))
    for i, theta in enumerate(thetas):
        src, rcv = pitch_catch_positions(theta)
        dist_src = np.hypot(RR - src[0], CC - src[1]) * dx[0]
        dist_rcv = np.hypot(RR - rcv[0], CC - rcv[1]) * dx[0]
        t_pred = (dist_src + dist_rcv) / cfg.WATER.sound_speed
        excess_env = np.clip(speckle_env[i] - homog_env[i], 0, None)
        sampled = np.interp(t_pred.ravel(), _lag_t_arr, excess_env, left=0.0, right=0.0)
        accumulator += (sampled.reshape(img_size, img_size)) ** 2
    return accumulator, img_rows, img_cols


def sample_field(field, img_rows, img_cols, row, col):
    ri = np.argmin(np.abs(img_rows - row))
    ci = np.argmin(np.abs(img_cols - col))
    return field[ri, ci]


if __name__ == "__main__":
    print(f"*** {labels.PENDING_SIGNOFF_BANNER} ***")
    print(f"*** {labels.TOY_EXACT_GT_CAPTION} ***")
    print("SPECKLE-INFORMED SURFACE SELECTION: rescores multiple reflection-derived inner-"
          "boundary candidates per angle using a speckle rising-edge likelihood, testing "
          "directly whether this beats naive strongest-peak selection (run -32: correlation "
          "with true contour = 0.25, effectively noise). No new jWave simulation beyond the "
          "36-sim speckle patient023 run.")

    canvas_lv, canvas_myo, outer_contour_dom, inner_contour_dom = load_real_contours(MRI_NPZ)
    ext_theta_out, ext_r_out = polar_resample(outer_contour_dom, center)
    ext_theta_in, ext_r_in = polar_resample(inner_contour_dom, center)
    true_r_in = np.array([r_at_theta(th, ext_theta_in, ext_r_in) for th in thetas])

    d_refl = np.load("results/patient023_reflection_raw_traces.npz")
    water_traces, homog_traces = d_refl["water_traces"], d_refl["phantom_traces"]
    d_speckle = np.load("results/patient023_speckle_raw_traces.npz")
    speckle_traces = d_speckle["speckle_traces"]

    water_env = [matched_filter_envelope(tr) for tr in water_traces]
    homog_env = [matched_filter_envelope(tr) for tr in homog_traces]
    speckle_env = [matched_filter_envelope(tr) for tr in speckle_traces]

    print("\n=== Building INCOHERENT (energy) speckle field for patient023 ===")
    speckle_field, img_rows, img_cols = das_energy_field(speckle_env, homog_env)

    print("=== Generating multi-candidate inner-boundary peaks per angle (homogeneous traces) ===")
    d_rows, d_cols = direction_vector(thetas)
    naive_r, speckle_r, n_candidates_per_angle = [], [], []
    for i, theta in enumerate(thetas):
        r_outer, candidates = generate_inner_candidates(water_env[i], homog_env[i])
        n_candidates_per_angle.append(len(candidates))
        if not candidates:
            naive_r.append(np.nan)
            speckle_r.append(np.nan)
            continue
        # naive: strongest-amplitude candidate (this project's established baseline approach)
        r_naive = max(candidates, key=lambda ra: ra[1])[0]
        naive_r.append(r_naive)

        # speckle-informed: rising-edge score per candidate, pick the best
        best_r, best_score = None, -np.inf
        for r_cand, amp in candidates:
            row_out = center[0] + (r_cand + EDGE_MARGIN_CELLS) * d_rows[i]
            col_out = center[1] + (r_cand + EDGE_MARGIN_CELLS) * d_cols[i]
            row_in = center[0] + (r_cand - EDGE_MARGIN_CELLS) * d_rows[i]
            col_in = center[1] + (r_cand - EDGE_MARGIN_CELLS) * d_cols[i]
            e_out = sample_field(speckle_field, img_rows, img_cols, row_out, col_out)
            e_in = sample_field(speckle_field, img_rows, img_cols, row_in, col_in)
            score = e_out - e_in
            if score > best_score:
                best_score, best_r = score, r_cand
        speckle_r.append(best_r)

    naive_r, speckle_r = np.array(naive_r), np.array(speckle_r)
    print(f"  mean candidates per angle: {np.mean(n_candidates_per_angle):.1f}")

    def eval_method(r_est, name):
        valid = ~np.isnan(r_est)
        if valid.sum() < 2:
            print(f"  {name}: too few valid detections to evaluate")
            return
        corr = np.corrcoef(r_est[valid], true_r_in[valid])[0, 1]
        rmse_mm = np.sqrt(np.mean(((r_est[valid] - true_r_in[valid]) * cfg.DX_M * 1e3) ** 2))
        print(f"  {name}: {valid.sum()}/{len(thetas)} detected, "
              f"correlation with true inner contour = {corr:.3f}, RMSE={rmse_mm:.3f}mm")
        return corr, rmse_mm

    print("\n--- Result: naive (strongest-amplitude peak) vs. speckle-rescored inner-boundary selection ---")
    naive_result = eval_method(naive_r, "naive strongest-peak")
    speckle_result = eval_method(speckle_r, "speckle-rescored")

    if speckle_result and naive_result and speckle_result[0] > naive_result[0] + 0.1:
        print("  -> speckle rescoring IMPROVES candidate selection over naive amplitude ranking.")
    elif speckle_result and naive_result and speckle_result[0] < naive_result[0] - 0.1:
        print("  -> speckle rescoring is WORSE than naive amplitude ranking at this setting.")
    else:
        print("  -> no clear improvement either way -- speckle field still too artifact-dominated "
              "or too weakly localized to usefully rescore these candidates.")

    fig, axes = plt.subplots(1, 2, figsize=(13, 5.5))
    im = axes[0].imshow(speckle_field, cmap="hot", origin="upper",
                         extent=[img_cols.min(), img_cols.max(), img_rows.max(), img_rows.min()])
    h_row = [center[0] + r_at_theta(th, ext_theta_out, ext_r_out) * direction_vector(th)[0] for th in thetas]
    h_col = [center[1] + r_at_theta(th, ext_theta_out, ext_r_out) * direction_vector(th)[1] for th in thetas]
    axes[0].plot(h_col + [h_col[0]], h_row + [h_row[0]], "c--", linewidth=1.2, label="true outer")
    hi_row = [center[0] + r_at_theta(th, ext_theta_in, ext_r_in) * direction_vector(th)[0] for th in thetas]
    hi_col = [center[1] + r_at_theta(th, ext_theta_in, ext_r_in) * direction_vector(th)[1] for th in thetas]
    axes[0].plot(hi_col + [hi_col[0]], hi_row + [hi_row[0]], "b--", linewidth=1.2, label="true inner")
    axes[0].set_title("Incoherent speckle field, patient023")
    axes[0].legend(fontsize=7)
    plt.colorbar(im, ax=axes[0], shrink=0.7)

    valid_n = ~np.isnan(naive_r)
    valid_s = ~np.isnan(speckle_r)
    axes[1].plot(thetas, true_r_in, "k-", linewidth=2, label="true inner contour")
    axes[1].plot(thetas[valid_n], naive_r[valid_n], "o", color="red", alpha=0.6, label="naive strongest-peak")
    axes[1].plot(thetas[valid_s], speckle_r[valid_s], "s", color="lime", alpha=0.8, label="speckle-rescored")
    axes[1].set_xlabel("angle (deg)")
    axes[1].set_ylabel("inner-boundary radius estimate (cells)")
    axes[1].set_title("Per-angle inner-boundary candidate: naive vs. speckle-rescored")
    axes[1].legend(fontsize=8)

    fig.suptitle("Speckle-informed surface selection: does rescoring beat naive amplitude ranking?")
    labels.add_banner(fig)
    plt.tight_layout()
    os.makedirs("results/figures", exist_ok=True)
    out_fig = "results/figures/phase1_speckle_informed_surface_selection.png"
    plt.savefig(out_fig, dpi=140)
    print(f"\nSaved {out_fig}")
