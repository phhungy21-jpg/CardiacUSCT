"""Phase 1 scout, addressing run -04/-05's known unfiltered-
backprojection star artifact: re-runs patient001's single-tissue
real-anatomy case (identical medium to run -05) and compares the
ORIGINAL unfiltered backprojection against a new SIRT-style iterative
reconstruction, side by side against the same true boundary.

Also saves the raw per-ray excess-delay dataset to disk (not done in
runs -04/-05) so future reconstruction-algorithm experiments can reuse
this expensive simulated data without resimulating.
"""

import numpy as np

from phase1_transmission_tomography_reconstruction import simulate_transmit_all_receivers, thetas
from phase1_real_mri_transmission_tomography import (
    load_real_contour_single_tissue, build_medium_real_contour_single_tissue, build_medium_water_only,
)
from phase1_rotating_transmission_scout import probe_position, N
import tomography_recon as recon
import labels

from matplotlib import pyplot as plt
import os

IMG_SIZE = 150


if __name__ == "__main__":
    print(f"*** {labels.PENDING_SIGNOFF_BANNER} ***")
    print(f"*** {labels.TOY_EXACT_GT_CAPTION} ***")
    print("PHASE 1 SCOUT: SIRT vs. unfiltered backprojection, patient001 single-tissue "
          "(same medium as run -05) -- addresses the known 36-point star artifact.")
    print("  compute estimate: 36 simulations x 2 media = 72 forward sims "
          "(same as runs -03/-04/-05) -- ~15-20 minutes based on that precedent")

    canvas, outer_contour_dom, max_radius_cells = load_real_contour_single_tissue()
    print(f"  real contour placed at domain center, max radius={max_radius_cells:.1f} cells")

    medium_water = build_medium_water_only()
    medium_phantom = build_medium_real_contour_single_tissue(canvas)

    print("\n=== Simulating water-only control, all transmit angles, all receivers ===")
    water_arrivals = {theta_tx: simulate_transmit_all_receivers(medium_water, theta_tx) for theta_tx in thetas}
    print("=== Simulating real-anatomy phantom, all transmit angles, all receivers ===")
    phantom_arrivals = {theta_tx: simulate_transmit_all_receivers(medium_phantom, theta_tx) for theta_tx in thetas}

    pairs_excess_delay_ns = {}
    for theta_tx in thetas:
        for theta_rx, t_water in water_arrivals[theta_tx].items():
            if theta_rx not in phantom_arrivals[theta_tx]:
                continue
            t_phantom = phantom_arrivals[theta_tx][theta_rx]
            pairs_excess_delay_ns[(theta_tx, theta_rx)] = (t_phantom - t_water) * 1e9

    n_pairs = len(pairs_excess_delay_ns)
    print(f"\n--- {n_pairs} ray paths captured ---")

    os.makedirs("results", exist_ok=True)
    keys_arr = np.array(list(pairs_excess_delay_ns.keys()))
    vals_arr = np.array(list(pairs_excess_delay_ns.values()))
    np.savez("results/patient001_single_tissue_rays.npz",
             theta_tx=keys_arr[:, 0], theta_rx=keys_arr[:, 1], excess_delay_ns=vals_arr,
             outer_contour_dom=outer_contour_dom, max_radius_cells=max_radius_cells)
    print("Saved results/patient001_single_tissue_rays.npz (for future reuse without resimulating)")

    print("\n=== Reconstructing: unfiltered backprojection (run -05's method) ===")
    image_unfiltered, img_rows, img_cols = recon.unfiltered_backprojection(
        pairs_excess_delay_ns, probe_position, IMG_SIZE, N)

    print("=== Reconstructing: SIRT (30 iterations) ===")
    image_sirt, _, _, residual_history = recon.sirt_reconstruct(
        pairs_excess_delay_ns, probe_position, IMG_SIZE, N, n_iters=30, relax=0.15)
    print(f"  SIRT residual RMS: iter 0={residual_history[0]:.2f}ns -> "
          f"iter {len(residual_history)-1}={residual_history[-1]:.2f}ns")

    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    for ax, image, title in [(axes[0], image_unfiltered, "Unfiltered backprojection (run -05's method)"),
                              (axes[1], image_sirt, "SIRT (30 iterations)")]:
        im = ax.imshow(image, cmap="hot_r", origin="upper",
                        extent=[img_cols.min(), img_cols.max(), img_rows.max(), img_rows.min()])
        ax.plot(outer_contour_dom[:, 1], outer_contour_dom[:, 0], "c--", linewidth=1.5, label="true boundary")
        ax.set_title(title, fontsize=10)
        ax.legend(fontsize=8)
        plt.colorbar(im, ax=ax, shrink=0.7)

    axes[2].plot(residual_history, "o-")
    axes[2].set_xlabel("SIRT iteration")
    axes[2].set_ylabel("residual RMS (ns)")
    axes[2].set_title("SIRT convergence")

    fig.suptitle("Phase 1: SIRT vs. unfiltered backprojection, patient001 single-tissue, static")
    labels.add_banner(fig)
    plt.tight_layout()
    os.makedirs("results/figures", exist_ok=True)
    out_fig = "results/figures/phase1_sirt_reconstruction_patient001.png"
    plt.savefig(out_fig, dpi=140)
    print(f"\nSaved {out_fig}")
