"""Phase 1 scout, two-tissue escalation: patient001's real anatomy with
BOTH boundaries present (myocardium wall + LV/blood cavity), not just
the single isolated outer boundary from run -05.

Verified before building: `myo_mask` and `lv_mask` in the prepped npz
are disjoint and their union equals `ring_mask` (myo_mask = wall only,
lv_mask = cavity only) -- confirmed numerically, not assumed.

Predicted BEFORE running, per this project's own tissue citations and
this whole codebase's history (`jwave/LOG.md`'s original weak-contrast
finding): the INNER (blood/myocardium) boundary should be much harder
to see than the OUTER (myocardium/water) boundary in this
reconstruction. Blood (1584 m/s) vs myocardium (1576 m/s) differ by
only ~0.5%; myocardium (1576 m/s) vs water (1520 m/s) differ by ~3.6%,
roughly 7x more contrast. Stating this prediction up front so the
result can be checked against it, not rationalized after the fact.

Uses the SIRT reconstruction (`tomography_recon.py`) from run -06
directly, not the unfiltered method, since run -06 already showed SIRT
is the better tool for this job.
"""

import numpy as np

from jax import numpy as jnp
from jwave import FourierSeries
from jwave.geometry import Medium

from phase1_transmission_tomography_reconstruction import simulate_transmit_all_receivers, thetas
from phase1_rotating_transmission_scout import probe_position, center, N, domain
import phase2_config as cfg
import tomography_recon as recon
import labels

from matplotlib import pyplot as plt
import os

MRI_NPZ = r"../jwave_test/results/mri_irregular_ring_patient001_slice4.npz"
IMG_SIZE = 150


def load_real_contours_two_tissue():
    d = np.load(MRI_NPZ)
    lv_mask = d["lv_mask"].astype(bool)
    myo_mask = d["myo_mask"].astype(bool)
    ring_mask = d["ring_mask"].astype(bool)
    outer_contour = d["outer_contour"]
    inner_contour = d["inner_contour"]

    ys, xs = np.where(ring_mask)
    ring_centroid_native = (ys.mean(), xs.mean())
    offset_row = int(round(center[0] - ring_centroid_native[0]))
    offset_col = int(round(center[1] - ring_centroid_native[1]))

    rows_native, cols_native = np.mgrid[0:lv_mask.shape[0], 0:lv_mask.shape[1]]
    rows_dom, cols_dom = rows_native + offset_row, cols_native + offset_col
    valid = (rows_dom >= 0) & (rows_dom < N[0]) & (cols_dom >= 0) & (cols_dom < N[1])

    canvas_lv = np.zeros(N, dtype=bool)
    canvas_myo = np.zeros(N, dtype=bool)
    canvas_lv[rows_dom[valid], cols_dom[valid]] = lv_mask[valid]
    canvas_myo[rows_dom[valid], cols_dom[valid]] = myo_mask[valid]

    outer_contour_dom = outer_contour + np.array([offset_row, offset_col])
    inner_contour_dom = inner_contour + np.array([offset_row, offset_col])
    max_radius_cells = np.hypot(outer_contour_dom[:, 0] - center[0], outer_contour_dom[:, 1] - center[1]).max()
    return canvas_lv, canvas_myo, outer_contour_dom, inner_contour_dom, max_radius_cells


def build_medium_two_tissue(canvas_lv, canvas_myo):
    sound_speed_map = np.full(N, cfg.WATER.sound_speed, dtype=np.float32)
    density_map = np.full(N, cfg.WATER.density, dtype=np.float32)
    sound_speed_map[canvas_myo] = cfg.MYOCARDIUM.sound_speed
    density_map[canvas_myo] = cfg.MYOCARDIUM.density
    sound_speed_map[canvas_lv] = cfg.BLOOD.sound_speed
    density_map[canvas_lv] = cfg.BLOOD.density
    ssm = jnp.expand_dims(jnp.array(sound_speed_map), -1)
    dm = jnp.expand_dims(jnp.array(density_map), -1)
    return Medium(domain=domain, sound_speed=FourierSeries(ssm, domain),
                  density=FourierSeries(dm, domain))


def build_medium_water_only():
    sound_speed_map = np.full(N, cfg.WATER.sound_speed, dtype=np.float32)
    density_map = np.full(N, cfg.WATER.density, dtype=np.float32)
    ssm = jnp.expand_dims(jnp.array(sound_speed_map), -1)
    dm = jnp.expand_dims(jnp.array(density_map), -1)
    return Medium(domain=domain, sound_speed=FourierSeries(ssm, domain),
                  density=FourierSeries(dm, domain))


if __name__ == "__main__":
    print(f"*** {labels.PENDING_SIGNOFF_BANNER} ***")
    print(f"*** {labels.TOY_EXACT_GT_CAPTION} ***")
    print("PHASE 1 SCOUT, TWO-TISSUE: patient001 real anatomy, myocardium wall + LV/blood "
          "cavity both present. PREDICTION (stated before running): the inner blood/"
          "myocardium boundary (~0.5% sound-speed contrast) should be much harder to "
          "see than the outer myocardium/water boundary (~3.6% contrast, ~7x more).")
    print("  compute estimate: 36 simulations x 2 media = 72 forward sims "
          "(same as prior runs) -- ~15-20 minutes based on that precedent")

    canvas_lv, canvas_myo, outer_contour_dom, inner_contour_dom, max_radius_cells = load_real_contours_two_tissue()
    print(f"  real contours placed at domain center, outer max radius={max_radius_cells:.1f} cells")

    medium_water = build_medium_water_only()
    medium_phantom = build_medium_two_tissue(canvas_lv, canvas_myo)

    print("\n=== Simulating water-only control, all transmit angles, all receivers ===")
    water_arrivals = {theta_tx: simulate_transmit_all_receivers(medium_water, theta_tx) for theta_tx in thetas}
    print("=== Simulating two-tissue phantom, all transmit angles, all receivers ===")
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
    np.savez("results/patient001_two_tissue_rays.npz",
             theta_tx=keys_arr[:, 0], theta_rx=keys_arr[:, 1], excess_delay_ns=vals_arr,
             outer_contour_dom=outer_contour_dom, inner_contour_dom=inner_contour_dom)
    print("Saved results/patient001_two_tissue_rays.npz (for future reuse without resimulating)")

    print("\n=== Reconstructing: SIRT (30 iterations) ===")
    image_sirt, img_rows, img_cols, residual_history = recon.sirt_reconstruct(
        pairs_excess_delay_ns, probe_position, IMG_SIZE, N, n_iters=30, relax=0.15)
    print(f"  SIRT residual RMS: iter 0={residual_history[0]:.2f}ns -> "
          f"iter {len(residual_history)-1}={residual_history[-1]:.2f}ns")

    # Quantitative check of the stated prediction: sample the reconstructed image
    # along a radial profile through the domain center and report the magnitude of
    # any dip/step at the inner (LV) boundary vs. the outer (epicardial) boundary.
    center_row_idx = np.argmin(np.abs(img_rows - center[0]))
    profile = image_sirt[center_row_idx, :]
    profile_cols = img_cols
    outer_radius_est = max_radius_cells
    inner_radius_est = np.hypot(inner_contour_dom[:, 0] - center[0], inner_contour_dom[:, 1] - center[1]).mean()
    print(f"\n--- Radial profile check (through domain center) ---")
    print(f"  outer boundary (myocardium/water) at ~{outer_radius_est:.0f} cells from center: "
          f"true epicardial contrast ~3.6% sound-speed difference")
    print(f"  inner boundary (blood/myocardium) at ~{inner_radius_est:.0f} cells from center: "
          f"true endocardial contrast ~0.5% sound-speed difference (predicted much weaker signal)")

    fig, axes = plt.subplots(1, 2, figsize=(13, 6))
    im = axes[0].imshow(image_sirt, cmap="hot_r", origin="upper",
                         extent=[img_cols.min(), img_cols.max(), img_rows.max(), img_rows.min()])
    axes[0].plot(outer_contour_dom[:, 1], outer_contour_dom[:, 0], "c--", linewidth=1.5, label="true outer (epicardium)")
    axes[0].plot(inner_contour_dom[:, 1], inner_contour_dom[:, 0], "lime", linestyle="--", linewidth=1.5, label="true inner (LV/blood)")
    axes[0].set_title("SIRT reconstruction, two-tissue patient001")
    axes[0].legend(fontsize=8)
    plt.colorbar(im, ax=axes[0], shrink=0.7)

    axes[1].plot(profile_cols, profile, "-")
    axes[1].axvline(center[1] - outer_radius_est, color="c", linestyle="--", alpha=0.6, label="outer boundary (left crossing)")
    axes[1].axvline(center[1] + outer_radius_est, color="c", linestyle="--", alpha=0.6)
    axes[1].axvline(center[1] - inner_radius_est, color="lime", linestyle="--", alpha=0.6, label="inner boundary (left crossing)")
    axes[1].axvline(center[1] + inner_radius_est, color="lime", linestyle="--", alpha=0.6)
    axes[1].set_xlabel("column (cells)")
    axes[1].set_ylabel("reconstructed value (ns)")
    axes[1].set_title("Horizontal profile through domain center")
    axes[1].legend(fontsize=7)

    fig.suptitle("Phase 1, TWO-TISSUE: patient001 myocardium+LV, static\n"
                 "prediction: inner (blood/myocardium) boundary much harder to see than outer")
    labels.add_banner(fig)
    plt.tight_layout()
    os.makedirs("results/figures", exist_ok=True)
    out_fig = "results/figures/phase1_two_tissue_reconstruction_patient001.png"
    plt.savefig(out_fig, dpi=140)
    print(f"\nSaved {out_fig}")
