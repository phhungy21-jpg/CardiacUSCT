"""Phase 3 — acoustic reconstruction test on the REAL, irregular MRI-
derived myocardial ring (patient001, slice 4, smoothed contour from
`phase3_mri_irregular_ring_prep.py`), the escalation from the smooth
eccentric ring (run -46) per user: "maybe reconstruct the irregular
ring from one of the mri, a sound escalation from smooth eccentric
off-center rings", followed by "stick with 1 first ... apply
smoothing, mimicking true tissue-like irregularities".

This is a SINGLE STATIC FRAME test (the real segmentation is one ED
timepoint, not a cardiac cycle) -- deliberately: the question here is
"does the multistatic-backprojection + shape-fit method survive a
real, non-parametric, irregular boundary shape at all, or where does
it break", not yet a full motion-cycle stress test (that is a separate,
later escalation).

READOUT GENERALIZATION (since a real contour isn't a circle or a
simple polygon family): each boundary (inner LV cavity, outer
epicardium) is represented by its own measured r(theta) function
(radius vs. angle from that boundary's own centroid, built by
polar-resampling the extracted contour points) -- this is the same
"known shape family, one free parameter" template-match principle
validated on the circle/triangle/heart-cartoon/ring phantoms, just
with r(theta) taken from the real measured contour instead of a
closed-form formula. The one free parameter swept is a SCALE FACTOR s
applied uniformly to r(theta); recovering s=1.0 means the method
correctly locks onto the true real shape at its true size, not a
smoothed/idealized approximation of it. This deliberately does NOT
attempt full non-parametric boundary recovery (fitting the whole curve
shape blind) -- that remains a harder, unattempted future step (see
run -46's caveat list).

Medium built directly from the smoothed real binary masks (not a
synthetic formula), reusing the SAME probe geometry, tissue properties,
and curvature-weighted + guard-band fit (run -46's validated method)
as every other ring test in this thread. The real ring's centroid is
translated to the existing domain center (150,150) so it sits within
the already-validated probe/search-grid geometry; each boundary uses
its OWN measured centroid as its own ray-sweep origin (same convention
as the eccentric ring test, run -46) -- this tests scale recovery under
real eccentricity/irregularity, not blind joint center+shape fitting.
"""

import numpy as np
from scipy.interpolate import RegularGridInterpolator

from jax import numpy as jnp
from jwave import FourierSeries
from jwave.geometry import Medium

import phase2_config as cfg
from phase3_backprojection_shape_fit_triangle import (
    capture_all_pairs, build_medium_homogeneous, direction_vector,
    _SRC, _RCV, t_arr, c_ref, _ENVELOPE_GROUP_DELAY_S,
    img_rows, img_cols, center, dx, N, domain, p3cfg, labels,
)
from phase3_ring_curvature_weighted_fit import pair_weight_at_R

from matplotlib import pyplot as plt
import os

import sys
import glob

PATIENT_ID = sys.argv[1] if len(sys.argv) > 1 else "patient001"
_matches = glob.glob(f"results/mri_irregular_ring_{PATIENT_ID}_slice*.npz")
if not _matches:
    raise FileNotFoundError(f"no prep npz found for {PATIENT_ID} -- run phase3_mri_irregular_ring_prep.py {PATIENT_ID} first")
MRI_NPZ = _matches[0]

N_ANGLES = 144  # finer than the smooth ring's 72, given real irregularity
_THETAS = np.linspace(0, 360, N_ANGLES, endpoint=False)


def _polar_resample(contour_rowcol, origin):
    """Build a wraparound-interpolatable r(theta) function from a closed
    contour's (row, col) points, relative to `origin` -- same angle
    convention as direction_vector (theta=0 -> up/-row, clockwise)."""
    rows = contour_rowcol[:, 0] - origin[0]
    cols = contour_rowcol[:, 1] - origin[1]
    r = np.sqrt(rows ** 2 + cols ** 2)
    theta = np.degrees(np.arctan2(cols, -rows)) % 360
    order = np.argsort(theta)
    theta_sorted, r_sorted = theta[order], r[order]
    # de-duplicate identical angles (keep first) so np.interp's x is strictly increasing
    keep = np.concatenate([[True], np.diff(theta_sorted) > 1e-9])
    theta_sorted, r_sorted = theta_sorted[keep], r_sorted[keep]
    ext_theta = np.concatenate([theta_sorted, [theta_sorted[0] + 360]])
    ext_r = np.concatenate([r_sorted, [r_sorted[0]]])
    return ext_theta, ext_r


def r_at_theta(theta_deg, ext_theta, ext_r):
    return np.interp(np.mod(theta_deg, 360), ext_theta, ext_r)


def build_medium_real_contour(label_map):
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


def build_search_grid(max_extent_cells, margin=15.0, density_per_cell=100.0 / 180.0):
    """Build a domain-centered search grid wide enough to cover
    `max_extent_cells` + margin -- the default imported img_rows/img_cols
    (+/-90 cells) is sized for patient001's proportions and clips a
    patient with a thicker wall / larger outer radius (silently, via
    RegularGridInterpolator's fill_value=0.0), so this must be checked
    per-patient rather than assumed. Density matches the default grid's
    (~0.56 points/cell) so results stay comparable across patients."""
    n_pts = max(100, int(round(2 * max_extent_cells * density_per_cell)) + 1)
    rows = np.linspace(center[0] - max_extent_cells, center[0] + max_extent_cells, n_pts)
    cols = np.linspace(center[1] - max_extent_cells, center[1] + max_extent_cells, n_pts)
    return rows, cols


def fit_scale_curvature_weighted(pairs, ext_theta, ext_r, scale_grid, origin, img_rows_g=img_rows, img_cols_g=img_cols):
    """Same curvature-weighted global template match as run -46, with
    the shape family being the real measured r(theta) scaled by s
    instead of a circle's constant R -- pair_weight_at_R still keyed off
    the LOCAL radius at each angle (r(theta)*s), not a single global R,
    since curvature (and thus reflection divergence) varies with radius
    the same way regardless of shape."""
    RR, CC = np.meshgrid(img_rows_g, img_cols_g, indexing="ij")
    per_pair_grids = {}
    for (tx, rx), envelope in pairs.items():
        src, rcv = _SRC[tx], _RCV[rx]
        dist_tx = np.sqrt((CC - src[0]) ** 2 + (RR - src[1]) ** 2) * dx[0]
        dist_rx = np.sqrt((CC - rcv[0]) ** 2 + (RR - rcv[1]) ** 2) * dx[0]
        t_total = (dist_tx + dist_rx) / c_ref + _ENVELOPE_GROUP_DELAY_S
        per_pair_grids[(tx, rx)] = np.interp(t_total, t_arr, envelope, left=0, right=0)
    interpolators = {
        key: RegularGridInterpolator((img_rows_g, img_cols_g), np.abs(grid), bounds_error=False, fill_value=0.0)
        for key, grid in per_pair_grids.items()
    }

    scores = np.zeros(len(scale_grid))
    for i, s in enumerate(scale_grid):
        d_vals = r_at_theta(_THETAS, ext_theta, ext_r) * s
        d_rows, d_cols = direction_vector(_THETAS)
        pts = np.stack([origin[0] + d_vals * d_rows, origin[1] + d_vals * d_cols], axis=1)
        total = 0.0
        for (tx, rx), interp in interpolators.items():
            w = pair_weight_at_R(tx, rx, np.mean(d_vals))  # mean radius at this scale as the curvature proxy
            total += w * interp(pts).sum()
        scores[i] = total
    best_idx = np.argmax(scores)
    return scale_grid[best_idx], scores


SCALE_GRID = np.arange(0.7, 1.31, 0.005)
GUARD_BAND_CELLS = 8.0  # same guard band as run -45/46, in physical cells (not scale units)


if __name__ == "__main__":
    print(f"*** {labels.PENDING_SIGNOFF_BANNER} ***")
    print(f"*** {labels.TOY_EXACT_GT_CAPTION} ***")
    print(f"Acoustic reconstruction of a REAL, MRI-derived, smoothed-irregular "
          f"myocardial ring (ACDC {PATIENT_ID}), single static frame. "
          f"Escalation from the smooth eccentric ring (run -46). Readout: "
          f"scale-factor sweep against the real measured r(theta) shape, "
          f"curvature-weighted + guard-band fit (run -46's validated method).")

    d = np.load(MRI_NPZ)
    lv_mask = d["lv_mask"].astype(bool)
    myo_mask = d["myo_mask"].astype(bool)
    ring_mask = d["ring_mask"].astype(bool)
    outer_contour = d["outer_contour"]
    inner_contour = d["inner_contour"]

    ys, xs = np.where(ring_mask)
    ring_centroid_native = (ys.mean(), xs.mean())
    lv_ys, lv_xs = np.where(lv_mask)
    lv_centroid_native = (lv_ys.mean(), lv_xs.mean())
    print(f"  native ring centroid={ring_centroid_native}, LV centroid={lv_centroid_native}")

    offset_row = int(round(center[0] - ring_centroid_native[0]))
    offset_col = int(round(center[1] - ring_centroid_native[1]))
    print(f"  placing ring centroid at domain center {center} via offset ({offset_row}, {offset_col})")

    rows_native, cols_native = np.mgrid[0:myo_mask.shape[0], 0:myo_mask.shape[1]]
    rows_dom, cols_dom = rows_native + offset_row, cols_native + offset_col
    valid = (rows_dom >= 0) & (rows_dom < N[0]) & (cols_dom >= 0) & (cols_dom < N[1])

    canvas_myo = np.zeros(N, dtype=bool)
    canvas_lv = np.zeros(N, dtype=bool)
    canvas_myo[rows_dom[valid], cols_dom[valid]] = myo_mask[valid]
    canvas_lv[rows_dom[valid], cols_dom[valid]] = lv_mask[valid]

    label_map = np.zeros(N, dtype=int)
    label_map[canvas_myo] = 2
    label_map[canvas_lv] = 3

    lv_centroid_dom = (lv_centroid_native[0] + offset_row, lv_centroid_native[1] + offset_col)
    ring_centroid_dom = (ring_centroid_native[0] + offset_row, ring_centroid_native[1] + offset_col)

    inner_contour_dom = inner_contour + np.array([offset_row, offset_col])
    outer_contour_dom = outer_contour + np.array([offset_row, offset_col])

    ext_theta_in, ext_r_in = _polar_resample(inner_contour_dom, lv_centroid_dom)
    ext_theta_out, ext_r_out = _polar_resample(outer_contour_dom, ring_centroid_dom)
    print(f"  mean inner radius={ext_r_in.mean():.1f} cells, mean outer radius={ext_r_out.mean():.1f} cells")

    # Search grid sized to THIS patient's actual anatomy -- the default
    # imported img_rows/img_cols (+/-90 cells) was sized for patient001's
    # proportions and silently clips a patient with a proportionally
    # thicker wall (RegularGridInterpolator's fill_value=0.0 masks this
    # rather than erroring), so this must be checked, not assumed.
    needed_extent = max(ext_r_out.max(), ext_r_in.max()) * SCALE_GRID.max() + 15.0
    if needed_extent > (img_rows.max() - center[0]):
        img_rows_g, img_cols_g = build_search_grid(needed_extent)
        print(f"  default search grid too small for this patient's anatomy (needs +/-{needed_extent:.0f} cells) "
              f"-- using a wider {len(img_rows_g)}x{len(img_cols_g)} grid instead")
    else:
        img_rows_g, img_cols_g = img_rows, img_cols

    medium = build_medium_real_contour(label_map)
    print("\n=== Simulating real-contour phantom (16 tx/rx pairs) ===")
    pairs_real = capture_all_pairs(medium)
    print("=== Simulating homogeneous reference ===")
    pairs_ref = capture_all_pairs(build_medium_homogeneous())

    fitted_s_in, scores_in = fit_scale_curvature_weighted(pairs_real, ext_theta_in, ext_r_in, SCALE_GRID, lv_centroid_dom, img_rows_g, img_cols_g)
    fitted_s_in_ref, _ = fit_scale_curvature_weighted(pairs_ref, ext_theta_in, ext_r_in, SCALE_GRID, lv_centroid_dom, img_rows_g, img_cols_g)

    fitted_inner_mean_radius = fitted_s_in * ext_r_in.mean()
    scale_grid_guarded = SCALE_GRID[np.abs(SCALE_GRID * ext_r_out.mean() - fitted_inner_mean_radius) > GUARD_BAND_CELLS]
    fitted_s_out, scores_out = fit_scale_curvature_weighted(pairs_real, ext_theta_out, ext_r_out, scale_grid_guarded, ring_centroid_dom, img_rows_g, img_cols_g)
    fitted_s_out_ref, _ = fit_scale_curvature_weighted(pairs_ref, ext_theta_out, ext_r_out, SCALE_GRID, ring_centroid_dom, img_rows_g, img_cols_g)

    in_err_mm = abs(fitted_s_in - 1.0) * ext_r_in.mean() * dx[0] * 1e3
    out_err_mm = abs(fitted_s_out - 1.0) * ext_r_out.mean() * dx[0] * 1e3
    locked = abs(fitted_s_out * ext_r_out.mean() - fitted_s_in * ext_r_in.mean()) < 3.0

    print(f"\n--- Result ---")
    print(f"  inner (LV cavity): fitted scale={fitted_s_in:.3f} (true=1.000), "
          f"mean-radius error={in_err_mm:.2f}mm")
    print(f"  outer (epicardium): fitted scale={fitted_s_out:.3f} (true=1.000), "
          f"mean-radius error={out_err_mm:.2f}mm")
    print(f"  outer locked to inner? {locked}")
    print(f"  homogeneous-medium control: inner fit scale={fitted_s_in_ref:.3f}, "
          f"outer fit scale={fitted_s_out_ref:.3f} (should be meaningless/low-confidence)")

    # --- Figure: score(s) curves + accumulator with true/fitted contour overlaid ---
    fig, axes = plt.subplots(1, 3, figsize=(16, 5.5))

    axes[0].plot(SCALE_GRID, scores_in / scores_in.max())
    axes[0].axvline(1.0, color="k", linestyle="--", label="true scale=1.0")
    axes[0].axvline(fitted_s_in, color="g", linestyle=":", label=f"fitted={fitted_s_in:.3f}")
    axes[0].set_title("Inner (LV) scale-fit score")
    axes[0].set_xlabel("candidate scale s")
    axes[0].legend(fontsize=8)

    axes[1].plot(scale_grid_guarded, scores_out / scores_out.max())
    axes[1].axvline(1.0, color="k", linestyle="--", label="true scale=1.0")
    axes[1].axvline(fitted_s_out, color="g", linestyle=":", label=f"fitted={fitted_s_out:.3f}")
    axes[1].set_title("Outer (epicardium) scale-fit score\n(guard-banded around inner fit)")
    axes[1].set_xlabel("candidate scale s")
    axes[1].legend(fontsize=8)

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

    axes[2].imshow(np.abs(accumulator_clean), cmap="hot", origin="upper",
                    extent=[img_cols_g.min(), img_cols_g.max(), img_rows_g.max(), img_rows_g.min()])
    axes[2].plot(inner_contour_dom[:, 1], inner_contour_dom[:, 0], "c--", linewidth=1, label="true inner")
    axes[2].plot(outer_contour_dom[:, 1], outer_contour_dom[:, 0], "c-", linewidth=1, label="true outer")

    fitted_in_pts = np.array([
        (lv_centroid_dom[0] + r_at_theta(th, ext_theta_in, ext_r_in) * fitted_s_in * direction_vector(th)[0],
         lv_centroid_dom[1] + r_at_theta(th, ext_theta_in, ext_r_in) * fitted_s_in * direction_vector(th)[1])
        for th in _THETAS])
    fitted_out_pts = np.array([
        (ring_centroid_dom[0] + r_at_theta(th, ext_theta_out, ext_r_out) * fitted_s_out * direction_vector(th)[0],
         ring_centroid_dom[1] + r_at_theta(th, ext_theta_out, ext_r_out) * fitted_s_out * direction_vector(th)[1])
        for th in _THETAS])
    axes[2].plot(fitted_in_pts[:, 1], fitted_in_pts[:, 0], "g:", linewidth=1.5, label="fitted inner")
    axes[2].plot(fitted_out_pts[:, 1], fitted_out_pts[:, 0], "g-.", linewidth=1.5, label="fitted outer")
    axes[2].set_title(f"Real-contour reconstruction\nin_err={in_err_mm:.2f}mm, out_err={out_err_mm:.2f}mm")
    axes[2].legend(fontsize=7)
    axes[2].axis("off")

    fig.suptitle(f"MRI-derived irregular ring ({PATIENT_ID}, smoothed) -- "
                  "acoustic backprojection + scale-factor shape-fit\n"
                  "(TOY: exact prescribed ground truth is the real segmented shape, not registration-derived motion)",
                  fontsize=10)
    labels.add_banner(fig)
    plt.tight_layout()
    os.makedirs("results/figures", exist_ok=True)
    out_fig = f"results/figures/phase3_mri_irregular_ring_reconstruction_{PATIENT_ID}.png"
    plt.savefig(out_fig, dpi=140)
    print(f"\nSaved {out_fig}")
