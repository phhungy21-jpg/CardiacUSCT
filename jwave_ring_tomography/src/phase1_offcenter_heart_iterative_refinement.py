"""Folds the bent-ray/eikonal correction (runs -28/-29) into an
ITERATIVE refinement loop, and derives an actual per-angle BOUNDARY
ESTIMATE from the corrected sound-speed field -- rather than only
measuring travel-time-prediction error, as runs -28/-29 did.

Per user: fold the bent-ray correction into an iterative loop, derive a
real boundary estimate from the corrected field, and consider whether
the TRANSMISSION channel is subject to the same "one pulse, many
possible paths" ambiguity the user identified as the deeper unsolved
problem for the REFLECTION channel (sequential acquisition tells you
WHICH TRANSMITTER produced a waveform, but not WHICH SURFACE/PATH did --
a pitch-catch pulse-echo shot really can bounce off the outer wall
on-axis, the outer wall off-axis, an inner wall, a concave notch, or
reverberate, all within one firing).

**Why this test uses the TRANSMISSION/bent-ray channel specifically,
not reflection**: transmission tomography does NOT have the reflection
channel's multi-specular-point ambiguity. Each tx/rx pair's signal is
governed by exactly ONE physical quantity -- Fermat's least-time path
through the medium -- not a family of candidate bounce points on a
possibly-concave boundary. scikit-fmm's fast-marching solver computes
that single least-time path directly (the first-arrival eikonal
solution), so "which surface point produced this" is not a live
question for this channel the way it is for reflection. The residual
analogous simplification here is DIFFRACTION/MULTIPATH -- a real
wavefront can split around a concave notch or complex boundary and
arrive via more than one near-equal-time path, and first-arrival-only
FMM captures just the fastest one -- but this is a smaller, different,
and NOT yet directly tested effect, not the same specular-ambiguity
problem. The reflection channel's open problem (peak-to-surface-point
identity) is NOT solved by anything in this script.

Reuses the ALREADY-SIMULATED off-center heart transmission dataset
(`results/offcenter_heart_rays.npz`, from the run -29 retest) -- no new
jWave simulation needed, pure post-processing, fast to iterate.
"""

import numpy as np

from phase1_offcenter_heart_blind_test import (
    ray_heart_distance, HEART_R, SHIFTED_CENTER,
)
from phase1_reflection_channel_scout import thetas, direction_vector
from phase1_bent_ray_correction import (
    iterative_bent_ray_refinement, extract_boundary_radius_per_angle,
)
import phase2_config as cfg
import labels

from matplotlib import pyplot as plt
import os

N_OUTER_ITERS = 4

if __name__ == "__main__":
    print(f"*** {labels.PENDING_SIGNOFF_BANNER} ***")
    print(f"*** {labels.TOY_EXACT_GT_CAPTION} ***")
    print("ITERATIVE BENT-RAY REFINEMENT + BOUNDARY EXTRACTION: off-center concave heart phantom "
          "(same exact shape as runs -27/-29). No new jWave simulation -- reuses "
          "results/offcenter_heart_rays.npz.")

    d = np.load("results/offcenter_heart_rays.npz")
    pairs_excess_delay_ns = {(tt, tr): v for tt, tr, v in zip(d["theta_tx"], d["theta_rx"], d["excess_delay_ns"])}
    print(f"  loaded {len(pairs_excess_delay_ns)} cached transmission ray paths")

    result = iterative_bent_ray_refinement(
        pairs_excess_delay_ns, cfg.MYOCARDIUM.sound_speed, tag="offcenter_heart",
        n_outer_iters=N_OUTER_ITERS)

    true_r_by_angle = np.array([ray_heart_distance(th, HEART_R) for th in thetas])

    r_baseline = extract_boundary_radius_per_angle(
        result["is_tissue_baseline"], result["img_rows"], result["img_cols"], SHIFTED_CENTER, thetas)
    r_final = extract_boundary_radius_per_angle(
        result["is_tissue_final"], result["img_rows"], result["img_cols"], SHIFTED_CENTER, thetas)

    def rmse_mm(r_est):
        valid = ~np.isnan(r_est)
        errs = (r_est[valid] - true_r_by_angle[valid]) * cfg.DX_M * 1e3
        return np.sqrt(np.mean(errs ** 2)), valid.sum()

    rmse_baseline, n_baseline = rmse_mm(r_baseline)
    rmse_final, n_final = rmse_mm(r_final)

    print(f"\n--- Boundary RMSE from the TRANSMISSION channel's own reconstructed tissue mask ---")
    print(f"  straight-ray SIRT (iteration 0):        RMSE={rmse_baseline:.4f}mm ({n_baseline}/{len(thetas)} angles found)")
    print(f"  iterative bent-ray (after {N_OUTER_ITERS} outer iters): RMSE={rmse_final:.4f}mm ({n_final}/{len(thetas)} angles found)")
    print(f"  (compare: this project's REFLECTION channel on the SAME phantom, run -27: RMSE=1.3047mm)")
    print(f"  (compare: jwave_test's sparse-probe blind reconstruction on the SAME phantom: "
          f"8-probe=1.544mm run -72, 16-probe=1.674mm run -73)")

    fig, axes = plt.subplots(1, 3, figsize=(18, 5.5))
    axes[0].plot(range(1, N_OUTER_ITERS + 1), result["data_residual_rms_ns"], "o-")
    axes[0].set_xlabel("outer iteration")
    axes[0].set_ylabel("bent-ray data-residual RMS (ns)")
    axes[0].set_title("Convergence: observed vs. bent-ray-predicted time")
    axes[0].grid(alpha=0.3)

    im = axes[1].imshow(result["is_tissue_baseline"], cmap="viridis", origin="upper",
                         extent=[result["img_cols"].min(), result["img_cols"].max(),
                                 result["img_rows"].max(), result["img_rows"].min()])
    axes[1].set_title(f"Straight-ray SIRT tissue mask (iter 0)\nboundary RMSE={rmse_baseline:.2f}mm")

    axes[2].imshow(result["is_tissue_final"], cmap="viridis", origin="upper",
                    extent=[result["img_cols"].min(), result["img_cols"].max(),
                            result["img_rows"].max(), result["img_rows"].min()])
    from matplotlib.path import Path
    from phase1_offcenter_heart_blind_test import heart_vertices
    verts = heart_vertices(HEART_R)
    h_row = [v[0] for v in verts] + [verts[0][0]]
    h_col = [v[1] for v in verts] + [verts[0][1]]
    axes[2].plot(h_col, h_row, "c--", linewidth=1.5, label="true heart boundary")
    d_rows, d_cols = direction_vector(thetas)
    valid = ~np.isnan(r_final)
    pt_row = SHIFTED_CENTER[0] + r_final[valid] * d_rows[valid]
    pt_col = SHIFTED_CENTER[1] + r_final[valid] * d_cols[valid]
    axes[2].scatter(pt_col, pt_row, c="lime", marker="s", s=20, edgecolor="k", linewidth=0.4,
                     label=f"extracted boundary ({n_final}/{len(thetas)})", zorder=5)
    axes[2].set_title(f"Iterative bent-ray tissue mask (iter {N_OUTER_ITERS})\nboundary RMSE={rmse_final:.2f}mm")
    axes[2].legend(fontsize=7, loc="upper right")

    fig.suptitle("Iterative bent-ray refinement + blind boundary extraction (off-center concave heart)")
    labels.add_banner(fig)
    plt.tight_layout()
    os.makedirs("results/figures", exist_ok=True)
    out_fig = "results/figures/phase1_offcenter_heart_iterative_refinement.png"
    plt.savefig(out_fig, dpi=140)
    print(f"\nSaved {out_fig}")
