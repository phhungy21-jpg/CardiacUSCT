"""Phase 1 scout, escalated to REAL anatomy: same full multistatic
transmission-tomography + straight-ray backprojection pipeline as run
-04, but on a real MRI-derived cross-section instead of a synthetic
circle.

Per user, after being corrected on run -04's phantom: "why is it just
a circle? i thought you put a crossection of the mri reconstructed
heart?" -- correct catch; run -04 used a synthetic circular disk
deliberately, as the simplest first sanity check (matching how
`jwave_test` itself escalated: circle -> eccentric ring -> real MRI).
This script does that escalation for the FIRST time in this project:
loads patient001's already-prepped real contour
(`jwave_test/results/mri_irregular_ring_patient001_slice4.npz`, the
same file `jwave_test`'s own real-anatomy runs used) and places it in
THIS project's water-bath domain.

Still a SINGLE-TISSUE test (the whole ring_mask interior -- myocardium
+ LV cavity combined -- treated as one MYOCARDIUM-property region, not
yet split into two separate boundaries), for a fair, isolated
comparison to run -04: this isolates the effect of REAL, IRREGULAR
shape vs. run -04's perfect circle, without also introducing a second
boundary at the same time (that two-tissue escalation is still a
separate, later step per run -04's "next action" list). STATIC (no
motion) still, per the original request; the real anatomy here is only
used for its real, irregular SHAPE, not yet its motion.
"""

import numpy as np

from jax import numpy as jnp
from jwave import FourierSeries
from jwave.geometry import Medium

from phase1_transmission_tomography_reconstruction import (
    simulate_transmit_all_receivers, backproject_excess_delay, thetas,
)
from phase1_rotating_transmission_scout import dx, center, N, domain
import phase2_config as cfg
import labels

from matplotlib import pyplot as plt
import os

MRI_NPZ = r"../jwave_test/results/mri_irregular_ring_patient001_slice4.npz"


def load_real_contour_single_tissue():
    d = np.load(MRI_NPZ)
    ring_mask = d["ring_mask"].astype(bool)
    outer_contour = d["outer_contour"]

    ys, xs = np.where(ring_mask)
    ring_centroid_native = (ys.mean(), xs.mean())
    offset_row = int(round(center[0] - ring_centroid_native[0]))
    offset_col = int(round(center[1] - ring_centroid_native[1]))

    rows_native, cols_native = np.mgrid[0:ring_mask.shape[0], 0:ring_mask.shape[1]]
    rows_dom, cols_dom = rows_native + offset_row, cols_native + offset_col
    valid = (rows_dom >= 0) & (rows_dom < N[0]) & (cols_dom >= 0) & (cols_dom < N[1])

    canvas = np.zeros(N, dtype=bool)
    canvas[rows_dom[valid], cols_dom[valid]] = ring_mask[valid]

    outer_contour_dom = outer_contour + np.array([offset_row, offset_col])
    max_radius_cells = np.hypot(outer_contour_dom[:, 0] - center[0], outer_contour_dom[:, 1] - center[1]).max()
    return canvas, outer_contour_dom, max_radius_cells


def build_medium_real_contour_single_tissue(canvas):
    sound_speed_map = np.where(canvas, cfg.MYOCARDIUM.sound_speed, cfg.WATER.sound_speed).astype(np.float32)
    density_map = np.where(canvas, cfg.MYOCARDIUM.density, cfg.WATER.density).astype(np.float32)
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
    print("PHASE 1 SCOUT, REAL ANATOMY: full multistatic transmission tomography "
          "on patient001's real MRI-derived outer boundary, single-tissue (isolated) "
          "test -- same pipeline as run -04, real shape instead of a synthetic circle.")

    canvas, outer_contour_dom, max_radius_cells = load_real_contour_single_tissue()
    print(f"  real contour placed at domain center, max radius={max_radius_cells:.1f} cells "
          f"(probe radius=120 cells -- must stay clear of the phantom)")
    from phase1_rotating_transmission_scout import PROBE_RADIUS_CELLS
    if max_radius_cells >= PROBE_RADIUS_CELLS * 0.9:
        print(f"  WARNING: phantom extent ({max_radius_cells:.1f}) is close to probe radius "
              f"({PROBE_RADIUS_CELLS}) -- check for overlap before trusting results")

    n_pairs_expected = sum(1 for tt in thetas for tr in thetas if tt != tr)
    print(f"\n  compute estimate: 36 simulations x 2 media = 72 forward sims "
          f"(same as runs -03/-04) -- ~15-20 minutes based on that precedent")

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
    delays = np.array(list(pairs_excess_delay_ns.values()))
    print(f"\n--- {n_pairs} ray paths captured ---")
    print(f"  excess delay range: {delays.min():.1f} to {delays.max():.1f} ns, mean={delays.mean():.1f}ns")

    print("\n=== Backprojecting excess delay (unfiltered straight-ray) ===")
    image, img_rows, img_cols = backproject_excess_delay(pairs_excess_delay_ns)

    fig, axes = plt.subplots(1, 2, figsize=(13, 6))
    axes[0].hist(delays, bins=40)
    axes[0].set_xlabel("excess delay (ns)")
    axes[0].set_ylabel("count")
    axes[0].set_title(f"Per-ray excess delay distribution ({n_pairs} rays)")

    im = axes[1].imshow(image, cmap="hot_r", origin="upper",
                         extent=[img_cols.min(), img_cols.max(), img_rows.max(), img_rows.min()])
    axes[1].plot(outer_contour_dom[:, 1], outer_contour_dom[:, 0], "c--", linewidth=1.5,
                 label="true real-MRI boundary (patient001)")
    axes[1].set_title("Unfiltered straight-ray backprojection\n(more negative = faster/tissue-like)")
    axes[1].legend(fontsize=8)
    plt.colorbar(im, ax=axes[1], label="mean excess delay (ns)", shrink=0.8)

    fig.suptitle("Phase 1 scout, REAL ANATOMY: patient001 outer boundary, static, single-tissue\n"
                 "full multistatic transmission tomography (same pipeline as run -04)")
    labels.add_banner(fig)
    plt.tight_layout()
    os.makedirs("results/figures", exist_ok=True)
    out_fig = "results/figures/phase1_real_mri_transmission_tomography_patient001.png"
    plt.savefig(out_fig, dpi=140)
    print(f"\nSaved {out_fig}")
