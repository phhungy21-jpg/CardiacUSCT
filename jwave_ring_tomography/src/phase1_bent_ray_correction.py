"""Resolves the "how do you infer a medium's unknown properties from
residuals" circularity, per user: fixes the straight-line ("no
refraction") approximation used everywhere in this project so far,
without needing to already know the medium.

Method (standard bent-ray/iterative travel-time tomography, and the
direct answer to "how do you infer the medium"): start from the
STRAIGHT-ray SIRT reconstruction we already have (an imperfect but real
starting guess, built from data, not assumed). Convert it into an
estimated sound-speed FIELD. Then find the TRUE travel time between
every probe pair through that estimated field, via a proper FAST
MARCHING (eikonal-equation) solver (`scikit-fmm`) -- this is Fermat's
principle computed numerically the RIGHT way: a real upwind eikonal
solver on a continuous field, not a naive 8-connected graph shortest
path.

CORRECTION, logged honestly: the first version of this script used
`scipy.sparse.csgraph.dijkstra` on an 8-connected grid graph instead --
it made predictions 3-4x WORSE, not better. Diagnosed as NOT primarily
the SIRT star-artifact (a cleaning step removing isolated components
made no difference) but the graph discretization itself: naive
grid-Dijkstra can find unphysical "shortcuts" through a fast medium
that no real, continuously-refracting wavefront could actually take
(only 8 fixed bend directions per step, no real angular constraint) --
a known, documented failure mode of graph-Dijkstra as an eikonal-
equation stand-in. `scikit-fmm` solves the actual eikonal equation
|grad T| = 1/c(x) with a proper upwind finite-difference scheme, which
does not have this artifact.

Validates directly against ALREADY-SIMULATED real data (`results/
patient001_single_tissue_rays.npz`, no new jWave simulation needed):
does the BENT-ray prediction explain the actually-observed excess delay
better than the naive straight-ray (homogeneous water) prediction did?
"""

import numpy as np
import skfmm

from phase1_rotating_transmission_scout import probe_position, center, N, dx
from phase1_reflection_channel_scout import direction_vector
import phase2_config as cfg
import tomography_recon as recon
import labels

from matplotlib import pyplot as plt
import os

IMG_SIZE = 150
THRESHOLD_FRACTION = 0.3  # fraction of the SIRT image's own min value, for the water/myocardium split


def nearest_grid_index(pos_row, pos_col, img_rows, img_cols):
    r = np.argmin(np.abs(img_rows - pos_row))
    c = np.argmin(np.abs(img_cols - pos_col))
    return r, c


def fmm_travel_time_field(sound_speed_grid, src_r, src_c, cell_size_m):
    """Solves the eikonal equation |grad T| = 1/c(x) from a point source
    at (src_r, src_c) using scikit-fmm's fast marching solver -- the
    proper numerical answer to 'what is the true travel time through
    this heterogeneous medium', without the grid-Dijkstra shortcut
    artifact."""
    phi = np.ones_like(sound_speed_grid)
    phi[src_r, src_c] = -1.0
    return skfmm.travel_time(phi, sound_speed_grid, dx=cell_size_m)


def evaluate_bent_ray_correction(pairs_excess_delay_ns, tissue_sound_speed, tag,
                                  img_size=IMG_SIZE, threshold_fraction=THRESHOLD_FRACTION,
                                  n_iters=30, relax=0.15, save_fig=True):
    """Reusable core of the patient001 bent-ray validation (run -28): builds
    a straight-ray SIRT reconstruction, cleans it, converts it into a
    binary water/tissue sound-speed field (using ONE guessed tissue speed
    -- `tissue_sound_speed` -- since the reconstruction doesn't yet know
    which tissue it's looking at), solves the eikonal equation from every
    probe position via scikit-fmm, and compares straight-ray vs. bent-ray
    travel-time predictions against the real observed data. Returns a
    dict of results; also saves a figure unless save_fig=False."""
    print("\n=== Rebuilding the straight-ray SIRT reconstruction (fast, no simulation) ===")
    image_sirt, img_rows, img_cols, residual_history = recon.sirt_reconstruct(
        pairs_excess_delay_ns, probe_position, img_size, N, n_iters=n_iters, relax=relax)
    print(f"  SIRT residual RMS: iter 0={residual_history[0]:.2f}ns -> iter {len(residual_history)-1}={residual_history[-1]:.2f}ns")

    print("\n=== Converting SIRT image into an estimated sound-speed field ===")
    thresh = image_sirt.min() * threshold_fraction
    is_tissue_raw = image_sirt < thresh
    print(f"  raw thresholded tissue fraction: {is_tissue_raw.mean()*100:.1f}% of pixels "
          f"(threshold={thresh:.1f}ns, {threshold_fraction} x SIRT min={image_sirt.min():.1f}ns)")

    print("=== Cleaning the sound-speed estimate: removing isolated star-artifact islands ===")
    from scipy import ndimage
    labeled, n_components = ndimage.label(is_tissue_raw, structure=np.ones((3, 3)))
    print(f"  found {n_components} connected 'tissue' components in the raw threshold")
    if n_components > 0:
        component_sizes = ndimage.sum(is_tissue_raw, labeled, range(1, n_components + 1))
        largest_label = np.argmax(component_sizes) + 1
        is_tissue = labeled == largest_label
        print(f"  keeping only the largest component ({int(component_sizes.max())} pixels) -- "
              f"discarding {n_components - 1} smaller ones (the star-artifact islands)")
    else:
        is_tissue = is_tissue_raw
    print(f"  cleaned tissue fraction: {is_tissue.mean()*100:.1f}% of pixels")
    sound_speed_grid = np.where(is_tissue, tissue_sound_speed, cfg.WATER.sound_speed).astype(np.float64)

    cell_size_m = ((img_rows[1] - img_rows[0]) * dx[0], (img_cols[1] - img_cols[0]) * dx[0])
    thetas_unique = sorted(set(tt for tt, _ in pairs_excess_delay_ns.keys()))
    print(f"\n=== Computing bent-ray (fast marching / eikonal) travel times from each of "
          f"{len(thetas_unique)} probe positions ===")
    dist_from_source = {}
    for theta in thetas_unique:
        src_pos = probe_position(theta)
        sr, sc = nearest_grid_index(src_pos[0], src_pos[1], img_rows, img_cols)
        dist_from_source[theta] = fmm_travel_time_field(sound_speed_grid, sr, sc, cell_size_m)

    print("\n=== Comparing straight-ray vs. bent-ray predictions against the REAL observed data ===")
    straight_errs, bent_errs = [], []
    for (theta_tx, theta_rx), excess_delay_ns in pairs_excess_delay_ns.items():
        tx_pos = probe_position(theta_tx)
        rx_pos = probe_position(theta_rx)
        straight_dist_m = np.hypot(tx_pos[0] - rx_pos[0], tx_pos[1] - rx_pos[1]) * dx[0]
        t_water_only = straight_dist_m / cfg.WATER.sound_speed
        t_observed = t_water_only + excess_delay_ns * 1e-9

        n_samples = 60
        rows_s = np.linspace(tx_pos[0], rx_pos[0], n_samples)
        cols_s = np.linspace(tx_pos[1], rx_pos[1], n_samples)
        seg_len = straight_dist_m / (n_samples - 1)
        t_straight_pred = 0.0
        for rr, cc in zip(rows_s, cols_s):
            ri, ci = nearest_grid_index(rr, cc, img_rows, img_cols)
            t_straight_pred += seg_len / sound_speed_grid[ri, ci]

        rx_r, rx_c = nearest_grid_index(rx_pos[0], rx_pos[1], img_rows, img_cols)
        t_bent_pred = dist_from_source[theta_tx][rx_r, rx_c]

        straight_errs.append(abs(t_straight_pred - t_observed))
        bent_errs.append(abs(t_bent_pred - t_observed))

    straight_errs, bent_errs = np.array(straight_errs), np.array(bent_errs)
    straight_rms = np.sqrt(np.mean(straight_errs ** 2)) * 1e9
    bent_rms = np.sqrt(np.mean(bent_errs ** 2)) * 1e9
    improvement = (1 - bent_errs.mean() / straight_errs.mean()) * 100
    print(f"\n--- Result: prediction error vs. real observed data ({tag}) ---")
    print(f"  straight-ray: mean error={straight_errs.mean()*1e9:.3f}ns, RMS={straight_rms:.3f}ns")
    print(f"  bent-ray:     mean error={bent_errs.mean()*1e9:.3f}ns, RMS={bent_rms:.3f}ns")
    print(f"  improvement: {improvement:+.1f}%")

    if save_fig:
        fig, axes = plt.subplots(1, 2, figsize=(13, 5.5))
        axes[0].imshow(sound_speed_grid, cmap="viridis", origin="upper",
                        extent=[img_cols.min(), img_cols.max(), img_rows.max(), img_rows.min()])
        axes[0].set_title(f"Estimated sound-speed field ({tag})\n(from thresholded straight-ray SIRT image)")

        axes[1].scatter(straight_errs * 1e9, bent_errs * 1e9, alpha=0.4, s=15)
        max_val = max(straight_errs.max(), bent_errs.max()) * 1e9
        axes[1].plot([0, max_val], [0, max_val], "k--", alpha=0.5, label="no improvement line")
        axes[1].set_xlabel("straight-ray prediction error (ns)")
        axes[1].set_ylabel("bent-ray prediction error (ns)")
        axes[1].set_title(f"Per-ray error: bent vs. straight ({tag})\n({improvement:+.1f}% mean improvement)")
        axes[1].legend(fontsize=8)

        fig.suptitle(f"Bent-ray correction ({tag}): does inferring the medium from residuals improve travel-time prediction?")
        labels.add_banner(fig)
        plt.tight_layout()
        os.makedirs("results/figures", exist_ok=True)
        out_fig = f"results/figures/phase1_bent_ray_correction_{tag}.png"
        plt.savefig(out_fig, dpi=140)
        print(f"\nSaved {out_fig}")

    return {
        "image_sirt": image_sirt, "sound_speed_grid": sound_speed_grid,
        "straight_rms_ns": straight_rms, "bent_rms_ns": bent_rms, "improvement_pct": improvement,
    }


def extract_boundary_radius_per_angle(is_tissue, img_rows, img_cols, origin, thetas,
                                       max_r_cells=140.0, step_cells=0.5):
    """BLIND boundary extraction from a reconstructed tissue mask: for
    each angle, march radially outward from `origin` and report the
    distance to the first tissue pixel hit. Uses the SAME convention
    (radial distance from the phantom's own center, per angle) as this
    project's established reflection-channel RMSE metric (e.g. run -27),
    so the two channels' accuracy numbers are directly comparable -- the
    "origin known" assumption is not a NEW blindness violation, it's the
    same one already baked into that existing benchmark."""
    radii = []
    for theta in thetas:
        d_row, d_col = direction_vector(theta)
        found = np.nan
        r = 0.0
        while r <= max_r_cells:
            rr, cc = origin[0] + r * d_row, origin[1] + r * d_col
            ri, ci = nearest_grid_index(rr, cc, img_rows, img_cols)
            if 0 <= ri < is_tissue.shape[0] and 0 <= ci < is_tissue.shape[1] and is_tissue[ri, ci]:
                found = r
                break
            r += step_cells
        radii.append(found)
    return np.array(radii)


def _threshold_and_clean(image, threshold_fraction):
    from scipy import ndimage
    thresh = image.min() * threshold_fraction
    is_tissue_raw = image < thresh
    labeled, n_components = ndimage.label(is_tissue_raw, structure=np.ones((3, 3)))
    if n_components > 0:
        sizes = ndimage.sum(is_tissue_raw, labeled, range(1, n_components + 1))
        is_tissue = labeled == (np.argmax(sizes) + 1)
    else:
        is_tissue = is_tissue_raw
    return is_tissue


def iterative_bent_ray_refinement(pairs_excess_delay_ns, tissue_sound_speed, tag,
                                   img_size=IMG_SIZE, threshold_fraction=THRESHOLD_FRACTION,
                                   n_sirt_iters=30, relax=0.15, n_outer_iters=4):
    """Folds the single-shot bent-ray correction (run -28/-29) into an
    iterative loop: at each outer iteration, solve the eikonal equation
    through the CURRENT sound-speed estimate (the exact forward model
    for that estimate, via scikit-fmm), compute the data residual
    against the real observed arrival times, and backproject that
    residual via straight-ray SIRT to update the image.

    HONEST LIMITATION, stated up front rather than glossed over: the
    correct update would backproject each ray's residual along its
    ACTUAL bent path (the Frechet derivative of travel time w.r.t.
    slowness along the true ray -- the standard approach in real
    iterative/adjoint-state travel-time tomography). scikit-fmm's fast-
    marching solver gives travel TIMES only, not explicit ray paths, so
    straight-ray backprojection is used here as a practical stand-in for
    the update step. This is expected to under-correct relative to a
    full bent-ray update, not to be wrong in direction -- convergence of
    the DATA residual (observed vs. bent-ray-predicted time) is the
    honest thing to check, not assumed.
    """
    print(f"\n=== Iterative bent-ray refinement ({tag}): outer iteration 0 (straight-ray SIRT baseline) ===")
    image, img_rows, img_cols, hist0 = recon.sirt_reconstruct(
        pairs_excess_delay_ns, probe_position, img_size, N, n_iters=n_sirt_iters, relax=relax)
    print(f"  baseline SIRT residual RMS: {hist0[0]:.2f}ns -> {hist0[-1]:.2f}ns")

    thetas_unique = sorted(set(tt for tt, _ in pairs_excess_delay_ns.keys()))
    cell_size_m = ((img_rows[1] - img_rows[0]) * dx[0], (img_cols[1] - img_cols[0]) * dx[0])

    is_tissue_baseline = _threshold_and_clean(image, threshold_fraction)
    is_tissue = is_tissue_baseline
    sound_speed_grid = np.where(is_tissue, tissue_sound_speed, cfg.WATER.sound_speed).astype(np.float64)

    data_residual_rms_ns = []
    for outer in range(1, n_outer_iters + 1):
        dist_from_source = {}
        for theta in thetas_unique:
            src_pos = probe_position(theta)
            sr, sc = nearest_grid_index(src_pos[0], src_pos[1], img_rows, img_cols)
            dist_from_source[theta] = fmm_travel_time_field(sound_speed_grid, sr, sc, cell_size_m)

        residual_target_ns = {}
        errs = []
        for (theta_tx, theta_rx), excess_delay_ns in pairs_excess_delay_ns.items():
            tx_pos, rx_pos = probe_position(theta_tx), probe_position(theta_rx)
            straight_dist_m = np.hypot(tx_pos[0] - rx_pos[0], tx_pos[1] - rx_pos[1]) * dx[0]
            t_obs = straight_dist_m / cfg.WATER.sound_speed + excess_delay_ns * 1e-9
            rx_r, rx_c = nearest_grid_index(rx_pos[0], rx_pos[1], img_rows, img_cols)
            t_bent_pred = dist_from_source[theta_tx][rx_r, rx_c]
            residual_s = t_obs - t_bent_pred
            residual_target_ns[(theta_tx, theta_rx)] = residual_s * 1e9
            errs.append(residual_s)
        rms_ns = np.sqrt(np.mean(np.array(errs) ** 2)) * 1e9
        data_residual_rms_ns.append(rms_ns)
        print(f"  outer iter {outer}: bent-ray data-residual RMS = {rms_ns:.3f}ns")

        delta_image, _, _, _ = recon.sirt_reconstruct(
            residual_target_ns, probe_position, img_size, N, n_iters=n_sirt_iters, relax=relax)
        image = image + delta_image
        is_tissue = _threshold_and_clean(image, threshold_fraction)
        sound_speed_grid = np.where(is_tissue, tissue_sound_speed, cfg.WATER.sound_speed).astype(np.float64)

    return {
        "image_baseline": None, "is_tissue_baseline": is_tissue_baseline,
        "image_final": image, "is_tissue_final": is_tissue, "sound_speed_grid_final": sound_speed_grid,
        "img_rows": img_rows, "img_cols": img_cols,
        "data_residual_rms_ns": data_residual_rms_ns, "baseline_sirt_history": hist0,
    }


if __name__ == "__main__":
    print(f"*** {labels.PENDING_SIGNOFF_BANNER} ***")
    print(f"*** {labels.TOY_EXACT_GT_CAPTION} ***")
    print("BENT-RAY CORRECTION: does a Fermat-principle shortest-path solve through the "
          "SIRT-estimated medium predict the ALREADY-SIMULATED real data better than the "
          "naive straight-ray (homogeneous water) assumption? No new jWave simulation --  "
          "reuses results/patient001_single_tissue_rays.npz.")

    d = np.load("results/patient001_single_tissue_rays.npz")
    pairs_excess_delay_ns = {(tt, tr): v for tt, tr, v in zip(d["theta_tx"], d["theta_rx"], d["excess_delay_ns"])}

    print("\n=== Rebuilding the straight-ray SIRT reconstruction (fast, no simulation) ===")
    image_sirt, img_rows, img_cols, residual_history = recon.sirt_reconstruct(
        pairs_excess_delay_ns, probe_position, IMG_SIZE, N, n_iters=30, relax=0.15)
    print(f"  SIRT residual RMS: iter 0={residual_history[0]:.2f}ns -> iter {len(residual_history)-1}={residual_history[-1]:.2f}ns")

    print("\n=== Converting SIRT image into an estimated sound-speed field ===")
    thresh = image_sirt.min() * THRESHOLD_FRACTION
    is_tissue_raw = image_sirt < thresh
    print(f"  raw thresholded tissue fraction: {is_tissue_raw.mean()*100:.1f}% of pixels "
          f"(threshold={thresh:.1f}ns, {THRESHOLD_FRACTION} x SIRT min={image_sirt.min():.1f}ns)")

    print("=== Cleaning the sound-speed estimate: removing isolated star-artifact islands ===")
    from scipy import ndimage
    labeled, n_components = ndimage.label(is_tissue_raw, structure=np.ones((3, 3)))
    print(f"  found {n_components} connected 'tissue' components in the raw threshold")
    if n_components > 0:
        component_sizes = ndimage.sum(is_tissue_raw, labeled, range(1, n_components + 1))
        largest_label = np.argmax(component_sizes) + 1
        is_tissue = labeled == largest_label
        print(f"  keeping only the largest component ({int(component_sizes.max())} pixels) -- "
              f"discarding {n_components - 1} smaller ones (the star-artifact islands)")
    else:
        is_tissue = is_tissue_raw
    print(f"  cleaned tissue fraction: {is_tissue.mean()*100:.1f}% of pixels")
    sound_speed_grid = np.where(is_tissue, cfg.MYOCARDIUM.sound_speed, cfg.WATER.sound_speed).astype(np.float64)

    cell_size_m = ((img_rows[1] - img_rows[0]) * dx[0], (img_cols[1] - img_cols[0]) * dx[0])
    thetas_unique = sorted(set(d["theta_tx"].tolist()))
    print(f"\n=== Computing bent-ray (fast marching / eikonal) travel times from each of "
          f"{len(thetas_unique)} probe positions ===")
    dist_from_source = {}
    for theta in thetas_unique:
        src_pos = probe_position(theta)
        sr, sc = nearest_grid_index(src_pos[0], src_pos[1], img_rows, img_cols)
        dist_from_source[theta] = fmm_travel_time_field(sound_speed_grid, sr, sc, cell_size_m)

    print("\n=== Comparing straight-ray vs. bent-ray predictions against the REAL observed data ===")
    straight_errs, bent_errs = [], []
    for (theta_tx, theta_rx), excess_delay_ns in pairs_excess_delay_ns.items():
        tx_pos = probe_position(theta_tx)
        rx_pos = probe_position(theta_rx)
        straight_dist_m = np.hypot(tx_pos[0] - rx_pos[0], tx_pos[1] - rx_pos[1]) * dx[0]
        t_water_only = straight_dist_m / cfg.WATER.sound_speed
        t_observed = t_water_only + excess_delay_ns * 1e-9

        # straight-ray prediction: integrate estimated slowness along the STRAIGHT line
        n_samples = 60
        rows_s = np.linspace(tx_pos[0], rx_pos[0], n_samples)
        cols_s = np.linspace(tx_pos[1], rx_pos[1], n_samples)
        seg_len = straight_dist_m / (n_samples - 1)
        t_straight_pred = 0.0
        for rr, cc in zip(rows_s, cols_s):
            ri, ci = nearest_grid_index(rr, cc, img_rows, img_cols)
            t_straight_pred += seg_len / sound_speed_grid[ri, ci]

        # bent-ray prediction: shortest path through the estimated field
        rx_r, rx_c = nearest_grid_index(rx_pos[0], rx_pos[1], img_rows, img_cols)
        t_bent_pred = dist_from_source[theta_tx][rx_r, rx_c]

        straight_errs.append(abs(t_straight_pred - t_observed))
        bent_errs.append(abs(t_bent_pred - t_observed))

    straight_errs, bent_errs = np.array(straight_errs), np.array(bent_errs)
    print(f"\n--- Result: prediction error vs. real observed data ---")
    print(f"  straight-ray: mean error={straight_errs.mean()*1e9:.3f}ns, RMS={np.sqrt(np.mean(straight_errs**2))*1e9:.3f}ns")
    print(f"  bent-ray:     mean error={bent_errs.mean()*1e9:.3f}ns, RMS={np.sqrt(np.mean(bent_errs**2))*1e9:.3f}ns")
    improvement = (1 - bent_errs.mean() / straight_errs.mean()) * 100
    print(f"  improvement: {improvement:+.1f}%")

    fig, axes = plt.subplots(1, 2, figsize=(13, 5.5))
    axes[0].imshow(sound_speed_grid, cmap="viridis", origin="upper",
                    extent=[img_cols.min(), img_cols.max(), img_rows.max(), img_rows.min()])
    axes[0].set_title("Estimated sound-speed field\n(from thresholded straight-ray SIRT image)")

    axes[1].scatter(straight_errs * 1e9, bent_errs * 1e9, alpha=0.4, s=15)
    max_val = max(straight_errs.max(), bent_errs.max()) * 1e9
    axes[1].plot([0, max_val], [0, max_val], "k--", alpha=0.5, label="no improvement line")
    axes[1].set_xlabel("straight-ray prediction error (ns)")
    axes[1].set_ylabel("bent-ray prediction error (ns)")
    axes[1].set_title(f"Per-ray error: bent vs. straight\n({improvement:+.1f}% mean improvement)")
    axes[1].legend(fontsize=8)

    fig.suptitle("Bent-ray correction: does inferring the medium from residuals improve travel-time prediction?")
    labels.add_banner(fig)
    plt.tight_layout()
    os.makedirs("results/figures", exist_ok=True)
    out_fig = "results/figures/phase1_bent_ray_correction.png"
    plt.savefig(out_fig, dpi=140)
    print(f"\nSaved {out_fig}")
