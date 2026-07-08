"""Validation (fast, synthetic, standard-domain): does the robust
temporal estimator (`phase3_robust_temporal_estimator.py`) correctly
NOT mask a genuine, abrupt, real anatomical change -- directly
addressing the user's safety concern ("wouldn't [borrowing from
confident neighbors] be a concern [the same way a strong in-frame
signal artifact biased a whole prediction]?").

Design: an 8-frame synthetic ring cycle where the inner radius is
CONSTANT at every frame except one (frame 3), which gets a deliberate,
large, one-off jump -- modeling something like a single ectopic beat.
Unlike patient023's phase 2/5 (run -64: no real acoustic peak at all),
this jump is a genuinely prescribed, real toy-phantom geometry, so it
should produce a strong, coherent acoustic signal (high CF) at that
frame, same as any other well-behaved frame in this thread's synthetic
tests. If the robust estimator preserves this jump (because its own
precision is high) rather than smoothing it toward its neighbors
(which are all at the constant baseline), that confirms the design is
safe. Uses the SAME 8-probe + local-max + real-calibration pipeline as
runs -56 onward, applied to a synthetic (not real-MRI) ring so it stays
within the standard, fast N=(300,300) domain -- no oversized real-
anatomy search grid needed.
"""

import numpy as np

from phase3_mri_8probe_test import (
    capture_all_pairs, build_medium_homogeneous, build_medium_real_contour,
    fit_scale_curvature_weighted, N, center, dx, labels,
)
from phase3_robust_temporal_estimator import robust_temporal_fuse

BASELINE_INNER_R = 60.0  # cells, matches this thread's toy LV radius convention
WALL_THICKNESS = 30.0
JUMP_FRAME = 3
JUMP_INNER_R = 45.0  # a deliberate, large, one-off contraction at frame 3 only

N_ANGLES = 144
_THETAS = np.linspace(0, 360, N_ANGLES, endpoint=False)


def build_synthetic_ring_label_map(inner_r):
    yy, xx = np.mgrid[0:N[0], 0:N[1]]
    dist = np.sqrt((xx - center[1]) ** 2 + (yy - center[0]) ** 2)
    outer_r = inner_r + WALL_THICKNESS
    label_map = np.zeros(N, dtype=int)
    label_map[dist < outer_r] = 2
    label_map[dist < inner_r] = 3
    return label_map


def uniform_r_theta(r_value):
    ext_theta = np.array([0.0, 360.0])
    ext_r = np.array([r_value, r_value])
    return ext_theta, ext_r


SCALE_GRID = np.arange(0.7, 1.31, 0.005)
GUARD_BAND_CELLS = 8.0

if __name__ == "__main__":
    print(f"*** {labels.PENDING_SIGNOFF_BANNER} ***")
    print(f"*** {labels.TOY_EXACT_GT_CAPTION} ***")
    print(f"Synthetic validation: 8-frame ring cycle, CONSTANT inner_R={BASELINE_INNER_R} "
          f"at every frame EXCEPT frame {JUMP_FRAME} (deliberate jump to inner_R={JUMP_INNER_R}). "
          f"Testing whether the robust temporal estimator preserves this genuine abrupt "
          f"change rather than smoothing it toward its (unchanged) neighbors.")

    true_inner_r_cells = [BASELINE_INNER_R] * 8
    true_inner_r_cells[JUMP_FRAME] = JUMP_INNER_R

    pairs_ref = capture_all_pairs(build_medium_homogeneous())
    ext_theta_in_template, ext_r_in_template = uniform_r_theta(BASELINE_INNER_R)  # FIXED template (like the ED template)

    fitted_r_mm, cfs = [], []
    for i, true_r in enumerate(true_inner_r_cells):
        print(f"\n=== Frame {i} (true inner_R={true_r} cells) ===")
        label_map = build_synthetic_ring_label_map(true_r)
        medium = build_medium_real_contour(label_map)
        pairs_real = capture_all_pairs(medium)

        # Fixed-template scale fit -- same design as the real-motion-cycle
        # scripts: one scale parameter against a FIXED (baseline) template.
        img_rows_g = np.linspace(center[0] - 100, center[0] + 100, 100)
        img_cols_g = np.linspace(center[1] - 100, center[1] + 100, 100)
        s_fit, scores, is_peak, conf, prom, cf = fit_scale_curvature_weighted(
            pairs_real, ext_theta_in_template, ext_r_in_template, SCALE_GRID, center, img_rows_g, img_cols_g)
        fitted_r = s_fit * BASELINE_INNER_R * dx[0] * 1e3
        fitted_r_mm.append(fitted_r)
        cfs.append(cf)
        true_r_mm = true_r * dx[0] * 1e3
        print(f"  true_r={true_r_mm:.2f}mm fitted_r={fitted_r:.2f}mm err={abs(fitted_r-true_r_mm):.2f}mm "
              f"CF={cf:.3f} prominence={prom:.2f} genuine_peak={is_peak}")

    true_r_mm_all = [r * dx[0] * 1e3 for r in true_inner_r_cells]
    raw_err = np.abs(np.array(fitted_r_mm) - np.array(true_r_mm_all))
    print(f"\n--- RAW fit RMSE={np.sqrt(np.mean(raw_err**2)):.4f}mm (per-frame: {np.round(raw_err,2).tolist()}) ---")

    posterior, prec_own, prior, prec_prior = robust_temporal_fuse(fitted_r_mm, cfs)
    post_err = np.abs(posterior - np.array(true_r_mm_all))
    print(f"\n--- After robust temporal fusion ---")
    for i in range(8):
        marker = "  <== DELIBERATE JUMP FRAME" if i == JUMP_FRAME else ""
        print(f"  frame {i}: true={true_r_mm_all[i]:.2f}mm raw={fitted_r_mm[i]:.2f}mm (err={raw_err[i]:.2f}) -> "
              f"posterior={posterior[i]:.2f}mm (err={post_err[i]:.2f}) "
              f"[own_prec={prec_own[i]:.2f} vs prior_prec={prec_prior[i]:.2f}]{marker}")
    print(f"\n  POSTERIOR RMSE={np.sqrt(np.mean(post_err**2)):.4f}mm")
    print(f"\n  Jump frame {JUMP_FRAME}: raw error={raw_err[JUMP_FRAME]:.2f}mm, "
          f"posterior error={post_err[JUMP_FRAME]:.2f}mm -- "
          f"{'PRESERVED (safe)' if post_err[JUMP_FRAME] < raw_err[JUMP_FRAME] + 0.5 else 'SMOOTHED AWAY (unsafe!)'}")
