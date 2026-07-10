"""Cheap test (no new jWave simulation -- reuses `results/
offcenter_heart_reflection_raw_traces.npz`) of whether DAS reflectivity
imaging is view-angle-count limited: subsample the ALREADY-SIMULATED
36-angle dataset down to 18, 12, and 9 angles (evenly spaced subsets)
and rebuild the straight-ray DAS image + boundary RMSE at each count.

Motivates the "more view angles" lever (flagged since runs -06/-15 for
the transmission channel's star artifact) directly for the reflection
channel's NEW DAS method: if RMSE degrades smoothly as angle count
drops, that's direct evidence more angles would further improve DAS
accuracy beyond 36; if RMSE is roughly flat, the current bottleneck is
something else (e.g. the single guessed background sound speed, or
matched-filter timing precision), not view count.
"""

import numpy as np
from scipy.signal import correlate, hilbert

from phase1_offcenter_heart_blind_test import ray_heart_distance, HEART_R, SHIFTED_CENTER
from phase1_reflection_channel_scout import thetas
from phase1_matched_filter_echo_extraction import _template
from phase1_das_reflectivity_imaging import das_straight_ray_image, extract_boundary_from_image
import phase2_config as cfg
import labels

from matplotlib import pyplot as plt
import os


def matched_filter_envelope(trace):
    correlated = correlate(trace, _template, mode="full")
    return np.abs(hilbert(correlated))


if __name__ == "__main__":
    print(f"*** {labels.PENDING_SIGNOFF_BANNER} ***")
    print(f"*** {labels.TOY_EXACT_GT_CAPTION} ***")
    print("DAS VIEW-ANGLE-COUNT SENSITIVITY: subsamples the already-simulated 36-angle "
          "reflection dataset down to 18/12/9 angles, no new jWave simulation.")

    d = np.load("results/offcenter_heart_reflection_raw_traces.npz")
    water_traces, phantom_traces = d["water_traces"], d["phantom_traces"]
    water_env = [matched_filter_envelope(tr) for tr in water_traces]
    phantom_env = [matched_filter_envelope(tr) for tr in phantom_traces]

    true_r_by_angle = np.array([ray_heart_distance(th, HEART_R) for th in thetas])

    n_angles_tested = [36, 18, 12, 9]
    rmse_by_count = {}
    for n_use in n_angles_tested:
        step = len(thetas) // n_use
        indices = list(range(0, len(thetas), step))[:n_use]
        image, img_rows, img_cols = das_straight_ray_image(phantom_env, water_env, angle_indices=indices)
        r_das = extract_boundary_from_image(image, img_rows, img_cols, SHIFTED_CENTER, thetas)
        errs_mm = (r_das - true_r_by_angle) * cfg.DX_M * 1e3
        rmse = np.sqrt(np.mean(errs_mm ** 2))
        rmse_by_count[n_use] = rmse
        print(f"  {n_use} angles: boundary RMSE={rmse:.4f}mm")

    fig, ax = plt.subplots(figsize=(7, 5.5))
    ns = sorted(rmse_by_count.keys())
    ax.plot(ns, [rmse_by_count[n] for n in ns], "o-")
    ax.axhline(1.3047, color="gray", linestyle="--", label="run -27 single-peak method (36 angles): 1.30mm")
    ax.set_xlabel("number of pitch-catch view angles used")
    ax.set_ylabel("DAS boundary RMSE (mm)")
    ax.set_title("DAS reflectivity imaging: does more view angles help?\n(off-center concave heart, subsampled from the same 36-angle dataset)")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)

    labels.add_banner(fig)
    plt.tight_layout()
    os.makedirs("results/figures", exist_ok=True)
    out_fig = "results/figures/phase1_das_view_angle_sensitivity.png"
    plt.savefig(out_fig, dpi=140)
    print(f"\nSaved {out_fig}")
