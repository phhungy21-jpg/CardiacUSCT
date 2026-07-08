"""Phase 3 -- run -74's injectivity probe (does the acoustic data
actually encode a local boundary perturbation, or is the region
ghost-dominated/information-poor?), applied to REAL anatomy instead of
the synthetic heart-cartoon phantom.

Per user's objection to run -74 ("that's just tested information in 1
patient") and the follow-up ("sure, why not both... because they are 2
contrast cases"): this tests patient001 (mild/typical contraction, used
throughout runs -47/-48/-49/-50) and patient023 (strong ~45%
contraction, and the patient whose OUTER/epicardial boundary was
confirmed structurally hard to recover with 4 probes -- runs
-51/-52/-53), at two curvature-selected landmarks per patient (sharpest
local feature vs. smoothest local region) on the OUTER (epicardial)
boundary -- the historically problematic one for this project.

Method, identical in spirit to run -74 but adapted for a non-polygon
real contour: instead of moving one polygon vertex, add a smooth,
LOCALIZED angular Gaussian bump of +PERTURB_CELLS to the real measured
r(theta) at the landmark angle only (leaving the rest of the real
contour untouched), rasterize baseline vs. perturbed as an ISOLATED
single-boundary phantom (myocardium-like tissue vs. chest-wall-proxy
only -- same isolation convention as the calibration measurements in
runs -44/-53/-60/-73, deliberately not the full two-boundary ring, to
keep this test about the outer boundary's OWN acoustic recoverability
rather than inner/outer interference), and compare the raw
curvature-weighted score curve along that exact ray, before vs. after,
bypassing the peak-selection algorithm -- exactly run -74's test.

Self-contained per this thread's discipline: duplicates the small
prep-npz-loading logic from `phase3_mri_irregular_ring_reconstruction.py`
(no function there is reusable for this without modification) and
otherwise imports only pure, unmodified 8-probe primitives.
"""

import numpy as np
from matplotlib.path import Path

from jax import numpy as jnp
from jwave import FourierSeries
from jwave.geometry import Medium

from phase3_mri_8probe_test import (
    capture_all_pairs, direction_vector, N, domain, center, cfg, labels,
)
from phase3_mri_irregular_ring_reconstruction import _polar_resample
from phase3_tip_notch_sensitivity_test import score_along_ray

from matplotlib import pyplot as plt
import os

PATIENTS = ["patient001", "patient023"]
PERTURB_CELLS = 5.0  # same magnitude as run -74, for direct comparability
ANG_SIGMA_DEG = 8.0  # localization width of the perturbation bump


def angdiff(a, b):
    return (a - b + 180) % 360 - 180


def add_angular_bump(ext_theta, ext_r, theta_center, amplitude, sigma_deg=ANG_SIGMA_DEG):
    diffs = angdiff(ext_theta, theta_center)
    bump = amplitude * np.exp(-0.5 * (diffs / sigma_deg) ** 2)
    return ext_r + bump


def curvature_landmarks(ext_theta, ext_r, pad=5):
    """Sharpest (max |curvature|) and smoothest (min |curvature|) angles
    on the real measured r(theta), via the standard polar-curve
    curvature formula, with circular padding for stable finite
    differences at the wraparound boundary."""
    theta_u, r_u = ext_theta[:-1], ext_r[:-1]  # drop the duplicated wrap point
    theta_pad = np.concatenate([theta_u[-pad:] - 360, theta_u, theta_u[:pad] + 360])
    r_pad = np.concatenate([r_u[-pad:], r_u, r_u[:pad]])
    th = np.deg2rad(theta_pad)
    rp = np.gradient(r_pad, th)
    rpp = np.gradient(rp, th)
    kappa_pad = (r_pad ** 2 + 2 * rp ** 2 - r_pad * rpp) / (r_pad ** 2 + rp ** 2) ** 1.5
    kappa = kappa_pad[pad:-pad]
    theta_sharp = theta_u[np.argmax(np.abs(kappa))]
    theta_smooth = theta_u[np.argmin(np.abs(kappa))]
    return theta_sharp, theta_smooth, kappa, theta_u


def contour_points_from_r_theta(ext_theta, ext_r, origin, n=360):
    thetas = np.linspace(0, 360, n, endpoint=False)
    r = np.interp(thetas, ext_theta, ext_r)
    d_row = -np.cos(np.deg2rad(thetas))
    d_col = np.sin(np.deg2rad(thetas))
    rows = origin[0] + r * d_row
    cols = origin[1] + r * d_col
    return list(zip(rows, cols))


def build_medium_isolated_boundary(ext_theta, ext_r, origin):
    verts = contour_points_from_r_theta(ext_theta, ext_r, origin)
    path = Path(verts)
    yy, xx = np.mgrid[0:N[0], 0:N[1]]
    points = np.column_stack([yy.ravel(), xx.ravel()])
    inside = path.contains_points(points).reshape(N)
    sound_speed_map = np.where(inside, cfg.MYOCARDIUM.sound_speed, cfg.CHEST_WALL_PROXY.sound_speed).astype(np.float32)
    density_map = np.where(inside, cfg.MYOCARDIUM.density, cfg.CHEST_WALL_PROXY.density).astype(np.float32)
    ssm = jnp.expand_dims(jnp.array(sound_speed_map), -1)
    dm = jnp.expand_dims(jnp.array(density_map), -1)
    return Medium(domain=domain, sound_speed=FourierSeries(ssm, domain),
                  density=FourierSeries(dm, domain))


def load_outer_contour(patient_id):
    d = np.load(f"results/mri_irregular_ring_{patient_id}_slice4.npz")
    ring_mask = d["ring_mask"].astype(bool)
    outer_contour = d["outer_contour"]
    ys, xs = np.where(ring_mask)
    ring_centroid_native = (ys.mean(), xs.mean())
    offset_row = int(round(center[0] - ring_centroid_native[0]))
    offset_col = int(round(center[1] - ring_centroid_native[1]))
    ring_centroid_dom = (ring_centroid_native[0] + offset_row, ring_centroid_native[1] + offset_col)
    outer_contour_dom = outer_contour + np.array([offset_row, offset_col])
    ext_theta, ext_r = _polar_resample(outer_contour_dom, ring_centroid_dom)
    return ext_theta, ext_r, ring_centroid_dom


def test_landmark(patient_id, label, ext_theta, ext_r, origin, theta_landmark, R_grid):
    r_before = np.interp(theta_landmark, ext_theta, ext_r)
    r_after = r_before + PERTURB_CELLS
    print(f"  [{patient_id}/{label}] theta={theta_landmark:.1f}deg, true R {r_before:.1f} -> {r_after:.1f} cells")

    ext_r_pert = add_angular_bump(ext_theta, ext_r, theta_landmark, PERTURB_CELLS)

    medium_base = build_medium_isolated_boundary(ext_theta, ext_r, origin)
    medium_pert = build_medium_isolated_boundary(ext_theta, ext_r_pert, origin)
    pairs_base = capture_all_pairs(medium_base)
    pairs_pert = capture_all_pairs(medium_pert)

    scores_base = score_along_ray(pairs_base, theta_landmark, R_grid, origin=origin)
    scores_pert = score_along_ray(pairs_pert, theta_landmark, R_grid, origin=origin)

    idx_old = np.argmin(np.abs(R_grid - r_before))
    idx_new = np.argmin(np.abs(R_grid - r_after))
    s_old_base, s_old_pert = scores_base[idx_old], scores_pert[idx_old]
    s_new_base, s_new_pert = scores_base[idx_new], scores_pert[idx_new]
    argmax_base = R_grid[np.argmax(scores_base)]
    argmax_pert = R_grid[np.argmax(scores_pert)]
    argmax_shift = argmax_pert - argmax_base
    direction_correct = (s_old_pert < s_old_base) and (s_new_pert > s_new_base)
    tracks = abs(argmax_shift - PERTURB_CELLS) < PERTURB_CELLS * 0.5

    print(f"    score at OLD true R: base={s_old_base:.4g} -> pert={s_old_pert:.4g} "
          f"({'decreased (expected)' if s_old_pert < s_old_base else 'INCREASED (unexpected)'})")
    print(f"    score at NEW true R: base={s_new_base:.4g} -> pert={s_new_pert:.4g} "
          f"({'increased (expected)' if s_new_pert > s_new_base else 'DECREASED (unexpected)'})")
    print(f"    raw argmax: base={argmax_base:.1f} -> pert={argmax_pert:.1f} "
          f"(shift={argmax_shift:+.1f} vs true {PERTURB_CELLS:+.1f}) "
          f"-- {'TRACKS' if tracks else 'DOES NOT TRACK'}, "
          f"direction {'CORRECT' if direction_correct else 'INCONSISTENT'}")

    return dict(patient=patient_id, label=label, theta=theta_landmark, r_before=r_before,
                scores_base=scores_base, scores_pert=scores_pert, tracks=tracks,
                direction_correct=direction_correct, argmax_shift=argmax_shift)


if __name__ == "__main__":
    print(f"*** {labels.PENDING_SIGNOFF_BANNER} ***")
    print(f"*** {labels.TOY_EXACT_GT_CAPTION} ***")
    print("INJECTIVITY PROBE ON REAL ANATOMY: run -74's tip/notch method, "
          "applied to the OUTER (epicardial) boundary of patient001 (mild "
          "contraction) and patient023 (strong contraction, confirmed "
          "structural outer-boundary limitation, runs -51/-52/-53), at "
          "curvature-selected landmarks (sharpest vs. smoothest local "
          f"feature) instead of a synthetic polygon vertex. +{PERTURB_CELLS:.0f}-"
          f"cell localized (sigma={ANG_SIGMA_DEG:.0f}deg) angular bump, isolated "
          "single-boundary phantom (myocardium vs. chest-wall-proxy only, "
          "same isolation convention as the calibration measurements).")

    all_results = []
    fig, axes = plt.subplots(2, 2, figsize=(13, 9))
    for pi, patient_id in enumerate(PATIENTS):
        print(f"\n=== {patient_id} ===")
        ext_theta, ext_r, origin = load_outer_contour(patient_id)
        theta_sharp, theta_smooth, kappa, theta_u = curvature_landmarks(ext_theta, ext_r)
        print(f"  outer boundary mean radius={ext_r[:-1].mean():.1f} cells")
        print(f"  sharpest-curvature landmark: theta={theta_sharp:.1f}deg (|kappa|={np.abs(kappa).max():.4f})")
        print(f"  smoothest-curvature landmark: theta={theta_smooth:.1f}deg (|kappa|={np.abs(kappa).min():.4f})")

        needed = ext_r[:-1].max() * 1.3 + 20.0
        R_grid = np.arange(10.0, needed, 0.5)

        r_sharp = test_landmark(patient_id, "sharp", ext_theta, ext_r, origin, theta_sharp, R_grid)
        r_smooth = test_landmark(patient_id, "smooth", ext_theta, ext_r, origin, theta_smooth, R_grid)
        all_results += [r_sharp, r_smooth]

        for j, r in enumerate([r_sharp, r_smooth]):
            ax = axes[pi, j]
            ax.plot(R_grid, r["scores_base"], label="baseline", color="C0")
            ax.plot(R_grid, r["scores_pert"], label=f"+{PERTURB_CELLS:.0f} cell bump", color="C1")
            ax.axvline(r["r_before"], color="C0", linestyle="--", alpha=0.6)
            ax.axvline(r["r_before"] + PERTURB_CELLS, color="C1", linestyle="--", alpha=0.6)
            clean = r["tracks"] and r["direction_correct"]
            ax.set_title(f"{patient_id} / {r['label']} (theta={r['theta']:.0f}deg)\n"
                         f"argmax shift={r['argmax_shift']:+.1f} -- "
                         f"{'CLEAN' if clean else 'GHOST/PARTIAL'}", fontsize=9)
            ax.set_xlabel("candidate R (cells)")
            ax.legend(fontsize=7)

    print("\n--- Summary ---")
    for r in all_results:
        clean = r["tracks"] and r["direction_correct"]
        print(f"  {r['patient']:12s} {r['label']:7s} theta={r['theta']:6.1f}deg "
              f"argmax_shift={r['argmax_shift']:+5.1f} "
              f"{'CLEAN' if clean else ('PARTIAL' if (r['tracks'] or r['direction_correct']) else 'GHOST-DOMINATED')}")

    fig.suptitle("Injectivity probe on REAL anatomy: sharp vs. smooth curvature landmarks,\n"
                 "patient001 (mild) and patient023 (strong contraction) outer boundary")
    labels.add_banner(fig)
    plt.tight_layout()
    os.makedirs("results/figures", exist_ok=True)
    out_fig = "results/figures/phase3_real_contour_sensitivity_test.png"
    plt.savefig(out_fig, dpi=140)
    print(f"\nSaved {out_fig}")
