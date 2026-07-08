"""Phase 3 — TWO-COMPONENT reconstruction test: LV cavity/myocardium
ring AND RV cavity, simultaneously, on the STANDARD (validated) 4-probe
model.

Per user: "try a 2-component run for [patient001's prep figure]
including both the lv and the appendage/la on a 4-probe model." ACDC
(patient001's only available real data) has no LA/appendage
segmentation at all -- confirmed by checking the label set (0=background,
1=RV, 2=myocardium, 3=LV cavity only). Per user's explicit choice when
asked: RV is used as the real, available second chamber, not a
synthetic appendage shape.

This is a genuinely new class of test for this thread: every prior
real-MRI test (runs -47 onward) deliberately EXCLUDED RV to keep a
direct 2-tissue-boundary analog of the synthetic ring phantoms. Here,
RV is included as a THIRD tissue region, spatially SEPARATE from the
LV/myocardium ring (not concentric, not overlapping) -- testing whether
the validated 4-probe multistatic backprojection + curvature-weighted +
local-max pipeline (run -59's official, patched version) can correctly
localize TWO INDEPENDENT anatomical structures at once, each fit
against its own real measured shape, without one contaminating the
other's fit (an analogous risk to the previously-diagnosed inner/outer
"locking", but between two disjoint bodies rather than two concentric
boundaries).

RV and LV cavities are BOTH blood (identical cited acoustic properties,
Mast 2000) -- acoustically indistinguishable except by shape/position,
same as real anatomy.
"""

import numpy as np
from scipy import ndimage
from skimage import measure

import phase2_config as cfg
from phase3_backprojection_shape_fit_triangle import (
    capture_all_pairs, build_medium_homogeneous, direction_vector,
    _SRC, _RCV, t_arr, c_ref, _ENVELOPE_GROUP_DELAY_S,
    img_rows, img_cols, center, dx, N, domain, labels,
)
from phase3_mri_irregular_ring_reconstruction import (
    _polar_resample, r_at_theta, build_medium_real_contour, build_search_grid,
    fit_scale_curvature_weighted, SCALE_GRID, GUARD_BAND_CELLS,
)

from jax import numpy as jnp
from jwave import FourierSeries
from jwave.geometry import Medium

from matplotlib import pyplot as plt
import os

PATIENT_ID = "patient001"
SLICE_Z = 4
DX_M = 0.1e-3
TARGET_LV_RADIUS_CELLS = 60.0


def build_medium_multi_label(label_map):
    sound_speed_map = np.zeros(N, dtype=np.float32)
    density_map = np.zeros(N, dtype=np.float32)
    for label, tissue in cfg.ACDC_LABEL_TO_TISSUE.items():
        m = label_map == label
        sound_speed_map[m] = tissue.sound_speed
        density_map[m] = tissue.density
    ssm = jnp.expand_dims(jnp.array(sound_speed_map), -1)
    dm = jnp.expand_dims(jnp.array(density_map), -1)
    return Medium(domain=domain, sound_speed=FourierSeries(ssm, domain),
                  density=FourierSeries(dm, domain))


if __name__ == "__main__":
    print(f"*** {labels.PENDING_SIGNOFF_BANNER} ***")
    print(f"*** {labels.TOY_EXACT_GT_CAPTION} ***")
    print(f"TWO-COMPONENT test ({PATIENT_ID}): LV cavity/myocardium ring + RV cavity, "
          f"STANDARD 4-probe model. RV used as the real second chamber (ACDC has no "
          f"LA/appendage label).")

    d = np.load(f"../pilot/data/processed/ACDC/{PATIENT_ID}.npz")
    mask3d = d["ed_mask"]
    spacing = d["spacing"]
    slice2d = mask3d[SLICE_Z]
    rv_native = (slice2d == 1)
    lv_native = (slice2d == 3)
    myo_native = (slice2d == 2)
    ring_native = myo_native | lv_native

    lv_area_mm2 = lv_native.sum() * (spacing[0] * spacing[1])
    real_lv_radius_mm = np.sqrt(lv_area_mm2 / np.pi)
    target_lv_radius_mm = TARGET_LV_RADIUS_CELLS * DX_M * 1e3
    zoom_factor = (spacing[0] / (DX_M * 1e3)) * (target_lv_radius_mm / real_lv_radius_mm)
    print(f"  zoom_factor={zoom_factor:.3f}x (same formula as run -47, should match patient001's 2.531x)")

    ring_up = ndimage.zoom(ring_native.astype(np.uint8), zoom_factor, order=0)
    lv_up = ndimage.zoom(lv_native.astype(np.uint8), zoom_factor, order=0)
    myo_up = ndimage.zoom(myo_native.astype(np.uint8), zoom_factor, order=0)
    rv_up = ndimage.zoom(rv_native.astype(np.uint8), zoom_factor, order=0)

    smooth_sigma = zoom_factor / 2.0
    ring_smooth = ndimage.gaussian_filter(ring_up.astype(float), sigma=smooth_sigma) >= 0.5
    lv_smooth = ndimage.gaussian_filter(lv_up.astype(float), sigma=smooth_sigma) >= 0.5
    myo_smooth = ndimage.gaussian_filter(myo_up.astype(float), sigma=smooth_sigma) >= 0.5
    rv_smooth = ndimage.gaussian_filter(rv_up.astype(float), sigma=smooth_sigma) >= 0.5
    print(f"  smoothing sigma={smooth_sigma:.2f} cells (same as run -47)")

    overlap = (rv_smooth & ring_smooth).sum()
    print(f"  RV/ring pixel overlap after smoothing: {overlap} px (should be 0 or near-0)")

    ys, xs = np.where(ring_smooth)
    ring_centroid_native = (ys.mean(), xs.mean())
    lys, lxs = np.where(lv_smooth)
    lv_centroid_native = (lys.mean(), lxs.mean())
    rys, rxs = np.where(rv_smooth)
    rv_centroid_native = (rys.mean(), rxs.mean())

    offset_row = int(round(center[0] - ring_centroid_native[0]))
    offset_col = int(round(center[1] - ring_centroid_native[1]))
    print(f"  placing ring centroid at domain center {center} via offset ({offset_row}, {offset_col})")

    rows_native, cols_native = np.mgrid[0:ring_smooth.shape[0], 0:ring_smooth.shape[1]]
    rows_dom, cols_dom = rows_native + offset_row, cols_native + offset_col
    valid = (rows_dom >= 0) & (rows_dom < N[0]) & (cols_dom >= 0) & (cols_dom < N[1])

    canvas_myo = np.zeros(N, dtype=bool)
    canvas_lv = np.zeros(N, dtype=bool)
    canvas_rv = np.zeros(N, dtype=bool)
    canvas_myo[rows_dom[valid], cols_dom[valid]] = myo_smooth[valid]
    canvas_lv[rows_dom[valid], cols_dom[valid]] = lv_smooth[valid]
    canvas_rv[rows_dom[valid], cols_dom[valid]] = rv_smooth[valid]

    label_map = np.zeros(N, dtype=int)
    label_map[canvas_myo] = 2
    label_map[canvas_lv] = 3
    label_map[canvas_rv] = 1  # RV -- placed AFTER myo/LV so any accidental overlap favors RV, flagged above if nonzero

    lv_centroid_dom = (lv_centroid_native[0] + offset_row, lv_centroid_native[1] + offset_col)
    ring_centroid_dom = (ring_centroid_native[0] + offset_row, ring_centroid_native[1] + offset_col)
    rv_centroid_dom = (rv_centroid_native[0] + offset_row, rv_centroid_native[1] + offset_col)
    print(f"  RV centroid (domain coords): {rv_centroid_dom}, distance from ring centroid: "
          f"{np.hypot(rv_centroid_dom[0]-ring_centroid_dom[0], rv_centroid_dom[1]-ring_centroid_dom[1]):.1f} cells")

    outer_contour = max(measure.find_contours(ring_smooth.astype(np.uint8), 0.5), key=len) + np.array([offset_row, offset_col])
    inner_contour = max(measure.find_contours(lv_smooth.astype(np.uint8), 0.5), key=len) + np.array([offset_row, offset_col])
    rv_contour = max(measure.find_contours(rv_smooth.astype(np.uint8), 0.5), key=len) + np.array([offset_row, offset_col])

    ext_theta_in, ext_r_in = _polar_resample(inner_contour, lv_centroid_dom)
    ext_theta_out, ext_r_out = _polar_resample(outer_contour, ring_centroid_dom)
    ext_theta_rv, ext_r_rv = _polar_resample(rv_contour, rv_centroid_dom)
    print(f"  mean radii (cells): LV inner={ext_r_in.mean():.1f}, epicardium outer={ext_r_out.mean():.1f}, "
          f"RV={ext_r_rv.mean():.1f} (max={ext_r_rv.max():.1f})")

    needed_extent = max(ext_r_out.max(), ext_r_in.max(), ext_r_rv.max(),
                         np.hypot(rv_centroid_dom[0]-center[0], rv_centroid_dom[1]-center[1]) + ext_r_rv.max()
                         ) * SCALE_GRID.max() + 15.0
    if needed_extent > (img_rows.max() - center[0]):
        img_rows_g, img_cols_g = build_search_grid(needed_extent)
        print(f"  using wider search grid: {len(img_rows_g)}x{len(img_cols_g)}, +/-{needed_extent:.0f} cells")
    else:
        img_rows_g, img_cols_g = img_rows, img_cols

    medium = build_medium_multi_label(label_map)
    print(f"\n=== Simulating LV+myo+RV phantom (16 tx/rx pairs, standard 4-probe) ===")
    pairs_real = capture_all_pairs(medium)
    print("=== Simulating homogeneous reference ===")
    pairs_ref = capture_all_pairs(build_medium_homogeneous())

    # --- Fit 1: LV inner (validated method) ---
    s_in, _, in_is_peak, in_conf = fit_scale_curvature_weighted(pairs_real, ext_theta_in, ext_r_in, SCALE_GRID, lv_centroid_dom, img_rows_g, img_cols_g)
    fitted_inner_mean_radius = s_in * ext_r_in.mean()

    # --- Fit 2: epicardium outer (validated method, guard-banded around LV inner) ---
    scale_grid_guarded_out = SCALE_GRID[np.abs(SCALE_GRID * ext_r_out.mean() - fitted_inner_mean_radius) > GUARD_BAND_CELLS]
    s_out, _, out_is_peak, out_conf = fit_scale_curvature_weighted(pairs_real, ext_theta_out, ext_r_out, scale_grid_guarded_out, ring_centroid_dom, img_rows_g, img_cols_g)

    # --- Fit 3: RV (NEW -- a spatially separate structure, own centroid, own template) ---
    # Guard band vs BOTH already-fitted structures (their fitted physical
    # radius, not just LV's) -- RV is far from the ring, so this is
    # expected to have little effect in practice, but checked rather than
    # assumed.
    fitted_outer_mean_radius = s_out * ext_r_out.mean()
    scale_grid_guarded_rv = SCALE_GRID[
        (np.abs(SCALE_GRID * ext_r_rv.mean() - fitted_inner_mean_radius) > GUARD_BAND_CELLS) &
        (np.abs(SCALE_GRID * ext_r_rv.mean() - fitted_outer_mean_radius) > GUARD_BAND_CELLS)
    ]
    s_rv, _, rv_is_peak, rv_conf = fit_scale_curvature_weighted(pairs_real, ext_theta_rv, ext_r_rv, scale_grid_guarded_rv, rv_centroid_dom, img_rows_g, img_cols_g)

    in_err_mm = abs(s_in - 1.0) * ext_r_in.mean() * dx[0] * 1e3
    out_err_mm = abs(s_out - 1.0) * ext_r_out.mean() * dx[0] * 1e3
    rv_err_mm = abs(s_rv - 1.0) * ext_r_rv.mean() * dx[0] * 1e3

    print(f"\n--- Result (2-component: LV/myo + RV, standard 4-probe) ---")
    print(f"  LV inner:    fitted scale={s_in:.3f} (true=1.000), error={in_err_mm:.2f}mm, "
          f"genuine_peak={in_is_peak}, conf={in_conf:.2f}")
    print(f"  epicardium:  fitted scale={s_out:.3f} (true=1.000), error={out_err_mm:.2f}mm, "
          f"genuine_peak={out_is_peak}, conf={out_conf:.2f}")
    print(f"  RV:          fitted scale={s_rv:.3f} (true=1.000), error={rv_err_mm:.2f}mm, "
          f"genuine_peak={rv_is_peak}, conf={rv_conf:.2f}")
    print(f"\n  (compare LV/epicardium to run -55's single-component (RV-excluded) patient001 result: "
          f"inner=0.995/0.03mm, outer=1.035/0.26mm -- if RV's presence contaminates the fit, "
          f"these numbers will differ meaningfully from that baseline)")

    # --- Figure: accumulator with all 3 true/fitted contours overlaid ---
    RR_full, CC_full = np.meshgrid(img_rows_g, img_cols_g, indexing="ij")
    accumulator = np.zeros(RR_full.shape)
    for (tx, rx), envelope in pairs_real.items():
        src, rcv = _SRC[tx], _RCV[rx]
        dist_tx = np.sqrt((CC_full - src[0]) ** 2 + (RR_full - src[1]) ** 2) * dx[0]
        dist_rx = np.sqrt((CC_full - rcv[0]) ** 2 + (RR_full - rcv[1]) ** 2) * dx[0]
        t_total = (dist_tx + dist_rx) / c_ref + _ENVELOPE_GROUP_DELAY_S
        accumulator += np.interp(t_total, t_arr, envelope, left=0, right=0)
    accumulator_ref = np.zeros(RR_full.shape)
    for (tx, rx), envelope in pairs_ref.items():
        src, rcv = _SRC[tx], _RCV[rx]
        dist_tx = np.sqrt((CC_full - src[0]) ** 2 + (RR_full - src[1]) ** 2) * dx[0]
        dist_rx = np.sqrt((CC_full - rcv[0]) ** 2 + (RR_full - rcv[1]) ** 2) * dx[0]
        t_total = (dist_tx + dist_rx) / c_ref + _ENVELOPE_GROUP_DELAY_S
        accumulator_ref += np.interp(t_total, t_arr, envelope, left=0, right=0)
    accumulator_clean = accumulator - accumulator_ref

    fig, ax = plt.subplots(figsize=(8, 8))
    ax.imshow(np.abs(accumulator_clean), cmap="hot", origin="upper",
              extent=[img_cols_g.min(), img_cols_g.max(), img_rows_g.max(), img_rows_g.min()])
    ax.plot(inner_contour[:, 1], inner_contour[:, 0], "c--", linewidth=1, label="true LV inner")
    ax.plot(outer_contour[:, 1], outer_contour[:, 0], "c-", linewidth=1, label="true epicardium")
    ax.plot(rv_contour[:, 1], rv_contour[:, 0], "y-", linewidth=1.5, label="true RV")

    N_ANGLES = 144
    thetas_plot = np.linspace(0, 360, N_ANGLES, endpoint=False)
    fit_in_pts = np.array([(lv_centroid_dom[0] + r_at_theta(th, ext_theta_in, ext_r_in) * s_in * direction_vector(th)[0],
                             lv_centroid_dom[1] + r_at_theta(th, ext_theta_in, ext_r_in) * s_in * direction_vector(th)[1]) for th in thetas_plot])
    fit_out_pts = np.array([(ring_centroid_dom[0] + r_at_theta(th, ext_theta_out, ext_r_out) * s_out * direction_vector(th)[0],
                              ring_centroid_dom[1] + r_at_theta(th, ext_theta_out, ext_r_out) * s_out * direction_vector(th)[1]) for th in thetas_plot])
    fit_rv_pts = np.array([(rv_centroid_dom[0] + r_at_theta(th, ext_theta_rv, ext_r_rv) * s_rv * direction_vector(th)[0],
                             rv_centroid_dom[1] + r_at_theta(th, ext_theta_rv, ext_r_rv) * s_rv * direction_vector(th)[1]) for th in thetas_plot])
    ax.plot(fit_in_pts[:, 1], fit_in_pts[:, 0], "g:", linewidth=1.5, label="fitted LV inner")
    ax.plot(fit_out_pts[:, 1], fit_out_pts[:, 0], "g-.", linewidth=1.5, label="fitted epicardium")
    ax.plot(fit_rv_pts[:, 1], fit_rv_pts[:, 0], "m:", linewidth=1.5, label="fitted RV")
    ax.set_title(f"2-component reconstruction ({PATIENT_ID}): LV/myo + RV, standard 4-probe\n"
                 f"LV_err={in_err_mm:.2f}mm epi_err={out_err_mm:.2f}mm RV_err={rv_err_mm:.2f}mm")
    ax.legend(fontsize=8, loc="upper right")
    ax.axis("off")
    labels.add_banner(fig)
    plt.tight_layout()
    os.makedirs("results/figures", exist_ok=True)
    out_fig = f"results/figures/phase3_lv_rv_twocomponent_test_{PATIENT_ID}.png"
    plt.savefig(out_fig, dpi=140)
    print(f"\nSaved {out_fig}")
