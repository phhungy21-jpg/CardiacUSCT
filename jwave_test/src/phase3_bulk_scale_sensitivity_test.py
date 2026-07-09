"""Phase 3 -- the discriminating experiment between "this target is
dead" and "the hypothesis is dead": does a BULK/global contraction
signal survive where runs -74/-75 showed fine PER-POINT boundary
localization mostly does not (1/14 clean)?

Runs -74/-75 tested per-vertex/per-landmark specular localization --
can the data pin down where ONE point of the boundary moved. The
project's actual registered hypothesis (LOG lines 23-27) was coarser:
does learned motion recovery survive real acoustic physics for BULK
myocardial contraction, not fine boundary tracing. A fixed ghost
artifact that destroys per-point argmax localization does not
necessarily destroy a GLOBAL/aggregate signal -- the already-validated
global shape-fit method throughout this thread (runs -32/-38/-46
onward) integrates energy across ALL angles precisely because a single
bad sector gets diluted by many other correctly-voting angles. This
script tests that dilution claim directly, in the same injectivity-
probe spirit as runs -74/-75/-76 (perturb the true geometry by a known
amount, read the RAW aggregate score curve before any peak-selection
algorithm), but with a BULK/global perturbation (uniform contraction of
the whole boundary) instead of one local vertex/landmark, and an
AGGREGATE (144-angle-summed) readout instead of one ray or one pair.

Per-pair check (`phase3_raw_pair_sensitivity_test.py`) came back
AMBIGUOUS (even the known-clean positive control scored at or below the
ghost-dominated locations) -- not a real answer either way, since
single-pair single-timepoint sampling is fragile. This test uses the
far more robust AGGREGATE statistic instead.

Tests: synthetic heart phantom (true R: 50 -> 45 cells, ~10% bulk
contraction, ED-like -> ES-like) and both real patients' outer boundary
(true scale: 1.0 -> 0.95, 5% bulk contraction), all on the isolated
single-boundary 8-probe setup already used throughout this line of
work.
"""

import numpy as np
from scipy.interpolate import RegularGridInterpolator

from phase3_mri_8probe_test import (
    _SRC, _RCV, capture_all_pairs, direction_vector, pair_weight_at_R,
    center, dx, c_ref, t_arr, _ENVELOPE_GROUP_DELAY_S, labels,
)
from phase3_tip_notch_sensitivity_test import build_medium_from_vertices
from phase3_heart_shape_offcenter_test import heart_vertices, SHIFTED_CENTER
from phase3_heart_shape_shapefit import ray_heart_distance
from phase3_mri_irregular_ring_reconstruction import r_at_theta
from phase3_real_contour_sensitivity_test import load_outer_contour, build_medium_isolated_boundary

from matplotlib import pyplot as plt
import os

N_ANGLES = 144
_THETAS = np.linspace(0, 360, N_ANGLES, endpoint=False)


def aggregate_score_curve(pairs, boundary_fn, param_grid, origin, img_rows_g, img_cols_g):
    """Same vectorized pattern as the already-validated global shape-fit
    (`fit_scale_curvature_weighted`), generalized to any boundary_fn(theta,
    param) -- sum curvature-weighted backprojected energy along the
    ENTIRE predicted boundary (all 144 angles) at each candidate
    parameter value, not one ray."""
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
    d_rows, d_cols = direction_vector(_THETAS)
    scores = np.zeros(len(param_grid))
    for i, param in enumerate(param_grid):
        dists = np.array([boundary_fn(th, param) for th in _THETAS])
        pts = np.stack([origin[0] + dists * d_rows, origin[1] + dists * d_cols], axis=1)
        total = 0.0
        for (tx, rx), interp in interpolators.items():
            w = pair_weight_at_R(tx, rx, np.mean(dists))
            total += w * interp(pts).sum()
        scores[i] = total
    return scores


def evaluate(label, scores_base, scores_pert, param_grid, param_old, param_new):
    idx_old = np.argmin(np.abs(param_grid - param_old))
    idx_new = np.argmin(np.abs(param_grid - param_new))
    s_old_base, s_old_pert = scores_base[idx_old], scores_pert[idx_old]
    s_new_base, s_new_pert = scores_base[idx_new], scores_pert[idx_new]
    argmax_base = param_grid[np.argmax(scores_base)]
    argmax_pert = param_grid[np.argmax(scores_pert)]
    print(f"  [{label}] score at OLD param={param_old:.3f}: base={s_old_base:.4g} -> pert={s_old_pert:.4g} "
          f"({'decreased (expected)' if s_old_pert < s_old_base else 'INCREASED (unexpected)'})")
    print(f"  [{label}] score at NEW param={param_new:.3f}: base={s_new_base:.4g} -> pert={s_new_pert:.4g} "
          f"({'increased (expected)' if s_new_pert > s_new_base else 'DECREASED (unexpected)'})")
    print(f"  [{label}] raw argmax: base={argmax_base:.3f} -> pert={argmax_pert:.3f} "
          f"(true param: {param_old:.3f} -> {param_new:.3f})")
    return argmax_base, argmax_pert


if __name__ == "__main__":
    print(f"*** {labels.PENDING_SIGNOFF_BANNER} ***")
    print(f"*** {labels.TOY_EXACT_GT_CAPTION} ***")
    print("BULK/GLOBAL CONTRACTION INJECTIVITY PROBE: does an AGGREGATE "
          "(144-angle-summed) readout track a uniform bulk contraction, even "
          "though runs -74/-75 found fine per-point localization mostly "
          "ghost-dominated (1/14 clean)? Same raw-curve-before-selection "
          "spirit, bulk parameter instead of one local point.")

    # --- Synthetic heart: true R 50 (ED-like) -> 45 (ES-like, ~10% bulk contraction) ---
    print("\n=== Synthetic heart phantom ===")
    R_ED, R_ES = 50.0, 45.0
    medium_ed = build_medium_from_vertices(heart_vertices(R_ED))
    medium_es = build_medium_from_vertices(heart_vertices(R_ES))
    print("  simulating ED-shape capture (8 probes)...")
    pairs_ed = capture_all_pairs(medium_ed)
    print("  simulating ES-shape capture (8 probes)...")
    pairs_es = capture_all_pairs(medium_es)

    R_grid_heart = np.arange(20.0, 80.0, 0.5)
    needed_extent = R_ED * 1.3 + 15.0
    img_rows_g = np.linspace(center[0] - needed_extent, center[0] + needed_extent, 160)
    img_cols_g = np.linspace(center[1] - needed_extent, center[1] + needed_extent, 160)

    boundary_fn_heart = lambda th, R: ray_heart_distance(th, R, origin=SHIFTED_CENTER)
    scores_ed = aggregate_score_curve(pairs_ed, boundary_fn_heart, R_grid_heart, SHIFTED_CENTER, img_rows_g, img_cols_g)
    scores_es = aggregate_score_curve(pairs_es, boundary_fn_heart, R_grid_heart, SHIFTED_CENTER, img_rows_g, img_cols_g)
    heart_argmax_ed, heart_argmax_es_capture = evaluate("heart: ED-capture", scores_ed, scores_ed, R_grid_heart, R_ED, R_ES)
    _, heart_argmax_es = evaluate("heart: ES-capture (the real test)", scores_ed, scores_es, R_grid_heart, R_ED, R_ES)
    print(f"  ED-capture argmax={R_grid_heart[np.argmax(scores_ed)]:.1f} (true={R_ED}); "
          f"ES-capture argmax={R_grid_heart[np.argmax(scores_es)]:.1f} (true={R_ES})")

    # --- Real anatomy: true scale 1.0 -> 0.95 (5% bulk contraction) ---
    scale_grid = np.arange(0.7, 1.31, 0.005)
    real_results = {}
    for patient_id in ["patient001", "patient023"]:
        print(f"\n=== {patient_id} outer boundary ===")
        ext_theta, ext_r, origin = load_outer_contour(patient_id)
        medium_full = build_medium_isolated_boundary(ext_theta, ext_r, origin)
        medium_contracted = build_medium_isolated_boundary(ext_theta, ext_r * 0.95, origin)
        print("  simulating scale=1.0 capture (8 probes)...")
        pairs_full = capture_all_pairs(medium_full)
        print("  simulating scale=0.95 capture (8 probes)...")
        pairs_contracted = capture_all_pairs(medium_contracted)

        boundary_fn_real = lambda th, s: r_at_theta(th, ext_theta, ext_r) * s
        scores_full = aggregate_score_curve(pairs_full, boundary_fn_real, scale_grid, origin, img_rows_g, img_cols_g)
        scores_contracted = aggregate_score_curve(pairs_contracted, boundary_fn_real, scale_grid, origin, img_rows_g, img_cols_g)
        evaluate(f"{patient_id}: scale=1.0 capture", scores_full, scores_full, scale_grid, 1.0, 0.95)
        evaluate(f"{patient_id}: scale=0.95 capture (the real test)", scores_full, scores_contracted, scale_grid, 1.0, 0.95)
        argmax_full = scale_grid[np.argmax(scores_full)]
        argmax_contracted = scale_grid[np.argmax(scores_contracted)]
        print(f"  full-size capture argmax scale={argmax_full:.3f} (true=1.000); "
              f"contracted capture argmax scale={argmax_contracted:.3f} (true=0.950)")
        real_results[patient_id] = dict(scores_full=scores_full, scores_contracted=scores_contracted,
                                         argmax_full=argmax_full, argmax_contracted=argmax_contracted)

    print("\n--- Summary: does the AGGREGATE readout track bulk contraction? ---")
    print(f"  heart phantom:  ED-capture argmax={R_grid_heart[np.argmax(scores_ed)]:.1f} (true 50.0), "
          f"ES-capture argmax={R_grid_heart[np.argmax(scores_es)]:.1f} (true 45.0)")
    for patient_id, r in real_results.items():
        print(f"  {patient_id}: full-capture argmax={r['argmax_full']:.3f} (true 1.000), "
              f"contracted-capture argmax={r['argmax_contracted']:.3f} (true 0.950)")

    fig, axes = plt.subplots(1, 3, figsize=(16, 5))
    axes[0].plot(R_grid_heart, scores_ed / scores_ed.max(), label="ED capture (true R=50)", color="C0")
    axes[0].plot(R_grid_heart, scores_es / scores_es.max(), label="ES capture (true R=45)", color="C1")
    axes[0].axvline(R_ED, color="C0", linestyle="--", alpha=0.6)
    axes[0].axvline(R_ES, color="C1", linestyle="--", alpha=0.6)
    axes[0].set_title("Synthetic heart: aggregate score vs. candidate R")
    axes[0].set_xlabel("candidate R (cells)")
    axes[0].legend(fontsize=8)

    for j, patient_id in enumerate(["patient001", "patient023"]):
        r = real_results[patient_id]
        ax = axes[j + 1]
        ax.plot(scale_grid, r["scores_full"] / r["scores_full"].max(), label="scale=1.0 capture", color="C0")
        ax.plot(scale_grid, r["scores_contracted"] / r["scores_contracted"].max(), label="scale=0.95 capture", color="C1")
        ax.axvline(1.0, color="C0", linestyle="--", alpha=0.6)
        ax.axvline(0.95, color="C1", linestyle="--", alpha=0.6)
        ax.set_title(f"{patient_id}: aggregate score vs. candidate scale")
        ax.set_xlabel("candidate scale")
        ax.legend(fontsize=8)

    fig.suptitle("BULK/GLOBAL contraction injectivity probe: aggregate (144-angle) readout\n"
                 "vs. runs -74/-75's per-point localization (1/14 clean)")
    labels.add_banner(fig)
    plt.tight_layout()
    os.makedirs("results/figures", exist_ok=True)
    out_fig = "results/figures/phase3_bulk_scale_sensitivity_test.png"
    plt.savefig(out_fig, dpi=140)
    print(f"\nSaved {out_fig}")
