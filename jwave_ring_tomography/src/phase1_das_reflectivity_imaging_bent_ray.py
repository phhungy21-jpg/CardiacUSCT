"""Upgrades the straight-ray DAS reflectivity image (phase1_das_
reflectivity_imaging.py) with BENT-RAY (scikit-fmm eikonal) travel
times instead of homogeneous-water Euclidean distance -- this is what
"fold the bent-ray correction into the reflection channel" concretely
means: instead of predicting each candidate pixel's src->pixel->rcv
delay assuming pure water, predict it through the best-available
sound-speed FIELD (reusing this project's runs -28/-29/-30 eikonal
infrastructure), so a pixel whose true path partly crosses faster
tissue gets correctly credited with a SHORTER predicted delay, not
penalized against a water-only assumption.

Sound-speed field used: the CLEAN (non-fragmented) straight-ray SIRT
threshold from the transmission channel's already-simulated data
(`results/offcenter_heart_rays.npz`, no new jWave sim) -- run -30's
own iterative bent-ray refinement of this field turned out visually
FRAGMENTED/noisy (a straight-ray-backprojection-as-update artifact), so
using that noisy field here would inject spurious sound-speed
variation into the travel-time prediction; the clean iteration-0
estimate is the more trustworthy choice available right now.

Reuses the ALREADY-SIMULATED reflection-channel raw traces from
`results/offcenter_heart_reflection_raw_traces.npz` (the straight-ray
DAS run) -- no new jWave simulation needed for this upgrade, pure
post-processing.
"""

import numpy as np
from scipy.signal import correlate, hilbert

from phase1_offcenter_heart_blind_test import (
    ray_heart_distance, heart_vertices, HEART_R, SHIFTED_CENTER,
)
from phase1_reflection_channel_scout import thetas, pitch_catch_positions, direction_vector
from phase1_matched_filter_echo_extraction import _lag_t_arr, _template
from phase1_bent_ray_correction import (
    nearest_grid_index, fmm_travel_time_field, _threshold_and_clean,
)
from phase1_das_reflectivity_imaging import extract_boundary_from_image
from phase1_rotating_transmission_scout import probe_position, dx, N
import tomography_recon as recon
import phase2_config as cfg
import labels

from matplotlib import pyplot as plt
import os

IMG_SIZE = 150
THRESHOLD_FRACTION = 0.3


def matched_filter_envelope(trace):
    correlated = correlate(trace, _template, mode="full")
    return np.abs(hilbert(correlated))


if __name__ == "__main__":
    print(f"*** {labels.PENDING_SIGNOFF_BANNER} ***")
    print(f"*** {labels.TOY_EXACT_GT_CAPTION} ***")
    print("DAS REFLECTIVITY IMAGING, BENT-RAY UPGRADE: same accumulation as the straight-ray "
          "version, but predicted src->pixel->rcv delays now go through a real (data-derived) "
          "sound-speed field via scikit-fmm, instead of assuming homogeneous water. No new "
          "jWave simulation -- reuses cached transmission rays + reflection raw traces.")

    print("\n=== Rebuilding the CLEAN straight-ray SIRT sound-speed field (transmission channel) ===")
    d_tx = np.load("results/offcenter_heart_rays.npz")
    pairs_excess_delay_ns = {(tt, tr): v for tt, tr, v in zip(d_tx["theta_tx"], d_tx["theta_rx"], d_tx["excess_delay_ns"])}
    image_sirt, img_rows, img_cols, hist = recon.sirt_reconstruct(
        pairs_excess_delay_ns, probe_position, IMG_SIZE, N, n_iters=30, relax=0.15)
    is_tissue = _threshold_and_clean(image_sirt, THRESHOLD_FRACTION)
    sound_speed_grid = np.where(is_tissue, cfg.MYOCARDIUM.sound_speed, cfg.WATER.sound_speed).astype(np.float64)
    print(f"  sound-speed field ready ({is_tissue.mean()*100:.1f}% tissue fraction)")

    print("\n=== Loading cached reflection raw traces, computing matched-filter envelopes ===")
    d_refl = np.load("results/offcenter_heart_reflection_raw_traces.npz")
    water_traces, phantom_traces = d_refl["water_traces"], d_refl["phantom_traces"]
    water_env = [matched_filter_envelope(tr) for tr in water_traces]
    phantom_env = [matched_filter_envelope(tr) for tr in phantom_traces]

    print("\n=== Solving the eikonal equation from every pitch-catch src/rcv position (72 FMM solves) ===")
    cell_size_m = ((img_rows[1] - img_rows[0]) * dx[0], (img_cols[1] - img_cols[0]) * dx[0])
    fmm_src, fmm_rcv = {}, {}
    for theta in thetas:
        src, rcv = pitch_catch_positions(theta)
        sr, sc = nearest_grid_index(src[0], src[1], img_rows, img_cols)
        rr_, rc_ = nearest_grid_index(rcv[0], rcv[1], img_rows, img_cols)
        fmm_src[theta] = fmm_travel_time_field(sound_speed_grid, sr, sc, cell_size_m)
        fmm_rcv[theta] = fmm_travel_time_field(sound_speed_grid, rr_, rc_, cell_size_m)

    print("=== Building BENT-RAY DAS reflectivity image (36-angle accumulation) ===")
    accumulator = np.zeros((IMG_SIZE, IMG_SIZE))
    for i, theta in enumerate(thetas):
        t_pred = fmm_src[theta] + fmm_rcv[theta]
        excess_env = np.clip(phantom_env[i] - water_env[i], 0, None)
        sampled = np.interp(t_pred.ravel(), _lag_t_arr, excess_env, left=0.0, right=0.0)
        accumulator += sampled.reshape(IMG_SIZE, IMG_SIZE)

    true_r_by_angle = np.array([ray_heart_distance(th, HEART_R) for th in thetas])
    r_das_bent = extract_boundary_from_image(accumulator, img_rows, img_cols, SHIFTED_CENTER, thetas)
    errs_mm = (r_das_bent - true_r_by_angle) * cfg.DX_M * 1e3
    rmse_mm = np.sqrt(np.mean(errs_mm ** 2))

    print(f"\n--- Result: BENT-RAY DAS reflectivity image, blind per-angle boundary ---")
    print(f"  RMSE={rmse_mm:.4f}mm across {len(thetas)} angles")
    print(f"  (compare: STRAIGHT-ray DAS, this session's own baseline -- see its printed RMSE)")
    print(f"  (compare: per-angle single-peak reflection method, run -27: RMSE=1.3047mm)")
    print(f"  (compare: jwave_test's sparse-probe blind reconstruction: 8-probe=1.544mm run -72, "
          f"16-probe=1.674mm run -73)")

    d_rows, d_cols = direction_vector(thetas)
    pt_row = SHIFTED_CENTER[0] + r_das_bent * d_rows
    pt_col = SHIFTED_CENTER[1] + r_das_bent * d_cols

    fig, axes = plt.subplots(1, 2, figsize=(13, 5.5))
    im = axes[0].imshow(accumulator, cmap="hot", origin="upper",
                         extent=[img_cols.min(), img_cols.max(), img_rows.max(), img_rows.min()])
    verts = heart_vertices(HEART_R)
    h_row = [v[0] for v in verts] + [verts[0][0]]
    h_col = [v[1] for v in verts] + [verts[0][1]]
    axes[0].plot(h_col, h_row, "c--", linewidth=1.5, label="true heart boundary")
    axes[0].set_title("BENT-RAY DAS reflectivity image\n(accumulated matched-filter energy)")
    axes[0].legend(fontsize=7)
    plt.colorbar(im, ax=axes[0], shrink=0.7)

    axes[1].imshow(accumulator, cmap="hot", origin="upper",
                   extent=[img_cols.min(), img_cols.max(), img_rows.max(), img_rows.min()])
    axes[1].plot(h_col, h_row, "c--", linewidth=1.5, label="true heart boundary")
    axes[1].scatter(pt_col, pt_row, c="lime", marker="s", s=20, edgecolor="k", linewidth=0.4,
                     label=f"bent-ray DAS boundary, RMSE={rmse_mm:.2f}mm", zorder=5)
    axes[1].set_title("Bent-ray DAS-extracted boundary vs. truth")
    axes[1].legend(fontsize=7)

    fig.suptitle("DAS reflectivity imaging, BENT-RAY upgrade: off-center concave heart")
    labels.add_banner(fig)
    plt.tight_layout()
    os.makedirs("results/figures", exist_ok=True)
    out_fig = "results/figures/phase1_das_reflectivity_imaging_bent_ray.png"
    plt.savefig(out_fig, dpi=140)
    print(f"\nSaved {out_fig}")
