"""Phase 3 -- direct forward-model INJECTIVITY probe: is the tip
region's failure (runs -72/-73) a genuine information deficit in the
acoustic data, or a readout-algorithm limitation a learned model could
fix?

Per user: given blind reconstruction (no prior) failed at the heart
phantom's convex tip but succeeded at the concave notch, and a
prior-provided fit worked well under high confidence -- what would a
U-Net-learned prior do for blind reconstruction, or is this
unpredictable? Answer: decompose into (1) structured, deterministic
corruption of information that IS present (learnable, e.g. the
ghost-cone smearing that improved cleanly with probe count for the
circle) vs. (2) genuine information absence (not learnable without
falling back to a population prior, i.e. hallucination). This script
tests which regime the tip/notch failure is in, directly, without
training anything: perturb the TRUE heart geometry at one vertex only
(tip or notch) by a known amount, resimulate, and measure whether that
perturbation is even visible in the raw backprojection score curve
along that ray. If perturbing the true tip position doesn't move the
score curve, no method -- classical or learned -- can recover it
without hallucinating; if it does move (just isn't being read out
correctly by the current local-max selector), a learned readout has
real signal to exploit.

Self-contained per this thread's discipline: reuses the already-
validated 8-probe capture/scoring primitives (`phase3_mri_8probe_test`)
and the already-validated heart phantom geometry
(`phase3_heart_shape_offcenter_test`), but adds its own single-vertex
perturbation logic -- no existing file is modified.
"""

import numpy as np
from matplotlib.path import Path
from scipy.interpolate import RegularGridInterpolator

from jax import numpy as jnp
from jwave import FourierSeries
from jwave.geometry import Medium

from phase3_mri_8probe_test import (
    _SRC, _RCV, capture_all_pairs, direction_vector, pair_weight_at_R,
    select_best_local_peak, center, dx, c_ref, t_arr,
    _ENVELOPE_GROUP_DELAY_S, labels,
)
from phase3_heart_shape_offcenter_test import (
    heart_vertices, N, domain, cfg, SHIFTED_CENTER,
)

from matplotlib import pyplot as plt
import os

HEART_R = 50.0
PERTURB_CELLS = 5.0  # ~0.5mm, a non-trivial, clinically-relevant displacement
R_GRID = np.arange(10.0, 90.0, 0.5)

TIP_IDX = 0    # unit vertex (0.00, -1.00) -- theta=180, sharp convex tip
NOTCH_IDX = 5  # unit vertex (0.00,  0.40) -- theta=0,   concave notch


def perturb_vertex(base_verts, idx, delta_cells, origin=SHIFTED_CENTER):
    """Move ONE vertex radially outward from `origin` by delta_cells,
    holding every other vertex fixed -- isolates the effect of a local
    boundary change, unlike scaling the whole shape by R."""
    row, col = base_verts[idx]
    vrow, vcol = row - origin[0], col - origin[1]
    norm = np.hypot(vrow, vcol)
    urow, ucol = vrow / norm, vcol / norm
    new_verts = list(base_verts)
    new_verts[idx] = (row + delta_cells * urow, col + delta_cells * ucol)
    return new_verts


def build_medium_from_vertices(verts):
    path = Path(verts)
    yy, xx = np.mgrid[0:N[0], 0:N[1]]
    points = np.column_stack([yy.ravel(), xx.ravel()])
    inside = path.contains_points(points).reshape(N)
    sound_speed_map = np.where(inside, cfg.BLOOD.sound_speed, cfg.CHEST_WALL_PROXY.sound_speed).astype(np.float32)
    density_map = np.where(inside, cfg.BLOOD.density, cfg.CHEST_WALL_PROXY.density).astype(np.float32)
    ssm = jnp.expand_dims(jnp.array(sound_speed_map), -1)
    dm = jnp.expand_dims(jnp.array(density_map), -1)
    return Medium(domain=domain, sound_speed=FourierSeries(ssm, domain),
                  density=FourierSeries(dm, domain))


def score_along_ray(pairs, theta_deg, R_grid, origin=SHIFTED_CENTER):
    """Same curvature-weighted scoring as the blind per-angle scan
    (runs -70/-71/-72), but for ONE ray only, returned as a raw score
    curve (no peak selection) so the underlying information content is
    visible directly, independent of any readout algorithm's quirks."""
    d_row, d_col = direction_vector(theta_deg)
    pts = np.array([(origin[0] + R * d_row, origin[1] + R * d_col) for R in R_grid])
    per_pair_grids = {}
    for (tx, rx), envelope in pairs.items():
        src, rcv = _SRC[tx], _RCV[rx]
        dist_tx = np.hypot(pts[:, 1] - src[0], pts[:, 0] - src[1]) * dx[0]
        dist_rx = np.hypot(pts[:, 1] - rcv[0], pts[:, 0] - rcv[1]) * dx[0]
        t_total = (dist_tx + dist_rx) / c_ref + _ENVELOPE_GROUP_DELAY_S
        per_pair_grids[(tx, rx)] = np.interp(t_total, t_arr, envelope, left=0, right=0)
    scores = np.zeros(len(R_grid))
    for i, R in enumerate(R_grid):
        total = 0.0
        for (tx, rx), grid in per_pair_grids.items():
            w = pair_weight_at_R(tx, rx, R)
            total += w * np.abs(grid[i])
        scores[i] = total
    return scores


if __name__ == "__main__":
    print(f"*** {labels.PENDING_SIGNOFF_BANNER} ***")
    print(f"*** {labels.TOY_EXACT_GT_CAPTION} ***")
    print("INJECTIVITY PROBE: perturbing ONE vertex of the heart phantom "
          f"(+{PERTURB_CELLS} cells radially) at a time -- tip (run -72's "
          "failure region) and notch (run -72's success region) -- to test "
          "whether the true-boundary information is PRESENT but misread "
          "(learnable), or genuinely ABSENT (hallucination hazard for any "
          "learned model), independent of the current local-max selector.")

    base_verts = heart_vertices(HEART_R)
    tip_pert_verts = perturb_vertex(base_verts, TIP_IDX, PERTURB_CELLS)
    notch_pert_verts = perturb_vertex(base_verts, NOTCH_IDX, PERTURB_CELLS)

    print("\n=== Simulating baseline heart (8 probes) ===")
    pairs_base = capture_all_pairs(build_medium_from_vertices(base_verts))
    print("=== Simulating TIP-perturbed heart (8 probes) ===")
    pairs_tip_pert = capture_all_pairs(build_medium_from_vertices(tip_pert_verts))
    print("=== Simulating NOTCH-perturbed heart (8 probes) ===")
    pairs_notch_pert = capture_all_pairs(build_medium_from_vertices(notch_pert_verts))

    true_tip_R_base, true_tip_R_pert = HEART_R * 1.00, HEART_R * 1.00 + PERTURB_CELLS
    true_notch_R_base, true_notch_R_pert = HEART_R * 0.40, HEART_R * 0.40 + PERTURB_CELLS

    print(f"\n  tip true distance:   {true_tip_R_base:.1f} -> {true_tip_R_pert:.1f} cells "
          f"({PERTURB_CELLS/true_tip_R_base*100:.0f}% change)")
    print(f"  notch true distance: {true_notch_R_base:.1f} -> {true_notch_R_pert:.1f} cells "
          f"({PERTURB_CELLS/true_notch_R_base*100:.0f}% change)")

    print("\n=== Scoring along the TIP ray (theta=180) ===")
    scores_tip_base = score_along_ray(pairs_base, 180.0, R_GRID)
    scores_tip_pert = score_along_ray(pairs_tip_pert, 180.0, R_GRID)

    print("=== Scoring along the NOTCH ray (theta=0) ===")
    scores_notch_base = score_along_ray(pairs_base, 0.0, R_GRID)
    scores_notch_pert = score_along_ray(pairs_notch_pert, 0.0, R_GRID)

    print("=== Locality control: does perturbing the TIP change the NOTCH ray? ===")
    scores_notch_from_tip_pert = score_along_ray(pairs_tip_pert, 0.0, R_GRID)

    def report(name, R_grid, scores_base, scores_pert, R_old, R_new):
        idx_old = np.argmin(np.abs(R_grid - R_old))
        idx_new = np.argmin(np.abs(R_grid - R_new))
        s_old_base, s_old_pert = scores_base[idx_old], scores_pert[idx_old]
        s_new_base, s_new_pert = scores_base[idx_new], scores_pert[idx_new]
        argmax_base = R_grid[np.argmax(scores_base)]
        argmax_pert = R_grid[np.argmax(scores_pert)]
        rel_gain_at_new = (s_new_pert - s_new_base) / (s_old_base + 1e-12)
        print(f"  [{name}] score at OLD true R={R_old:.1f}: base={s_old_base:.4g} -> pert={s_old_pert:.4g}")
        print(f"  [{name}] score at NEW true R={R_new:.1f}: base={s_new_base:.4g} -> pert={s_new_pert:.4g}")
        print(f"  [{name}] raw argmax: base={argmax_base:.1f} -> pert={argmax_pert:.1f} "
              f"(true shift was {R_new-R_old:+.1f})")
        print(f"  [{name}] relative signal gain at new true R (normalized to old base level): {rel_gain_at_new:+.3f}")
        return argmax_base, argmax_pert

    print("\n--- Results ---")
    tip_argmax_base, tip_argmax_pert = report("TIP", R_GRID, scores_tip_base, scores_tip_pert,
                                               true_tip_R_base, true_tip_R_pert)
    notch_argmax_base, notch_argmax_pert = report("NOTCH", R_GRID, scores_notch_base, scores_notch_pert,
                                                    true_notch_R_base, true_notch_R_pert)
    print(f"  [locality control] notch ray argmax: unperturbed-medium={notch_argmax_base:.1f}, "
          f"tip-perturbed-medium={R_GRID[np.argmax(scores_notch_from_tip_pert)]:.1f} "
          f"(should be ~unchanged if the tip perturbation doesn't leak into the notch ray)")

    tip_argmax_shift = tip_argmax_pert - tip_argmax_base
    notch_argmax_shift = notch_argmax_pert - notch_argmax_base
    true_shift = PERTURB_CELLS
    print(f"\n--- Verdict ---")
    print(f"  TIP:   raw argmax shift = {tip_argmax_shift:+.1f} cells vs true shift {true_shift:+.1f} "
          f"({'TRACKS' if abs(tip_argmax_shift - true_shift) < true_shift * 0.5 else 'DOES NOT TRACK'})")
    print(f"  NOTCH: raw argmax shift = {notch_argmax_shift:+.1f} cells vs true shift {true_shift:+.1f} "
          f"({'TRACKS' if abs(notch_argmax_shift - true_shift) < true_shift * 0.5 else 'DOES NOT TRACK'})")

    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    axes[0].plot(R_GRID, scores_tip_base, label="baseline heart", color="C0")
    axes[0].plot(R_GRID, scores_tip_pert, label=f"tip pushed out +{PERTURB_CELLS:.0f} cells", color="C1")
    axes[0].axvline(true_tip_R_base, color="C0", linestyle="--", alpha=0.6, label="true R (base)")
    axes[0].axvline(true_tip_R_pert, color="C1", linestyle="--", alpha=0.6, label="true R (perturbed)")
    axes[0].set_title(f"TIP ray (theta=180)\nargmax shift={tip_argmax_shift:+.1f} vs true {true_shift:+.1f}")
    axes[0].set_xlabel("candidate R (cells)")
    axes[0].set_ylabel("curvature-weighted score")
    axes[0].legend(fontsize=8)

    axes[1].plot(R_GRID, scores_notch_base, label="baseline heart", color="C0")
    axes[1].plot(R_GRID, scores_notch_pert, label=f"notch pushed out +{PERTURB_CELLS:.0f} cells", color="C1")
    axes[1].axvline(true_notch_R_base, color="C0", linestyle="--", alpha=0.6, label="true R (base)")
    axes[1].axvline(true_notch_R_pert, color="C1", linestyle="--", alpha=0.6, label="true R (perturbed)")
    axes[1].set_title(f"NOTCH ray (theta=0)\nargmax shift={notch_argmax_shift:+.1f} vs true {true_shift:+.1f}")
    axes[1].set_xlabel("candidate R (cells)")
    axes[1].legend(fontsize=8)

    fig.suptitle("Injectivity probe: does the acoustic data actually encode a "
                 f"{PERTURB_CELLS:.0f}-cell boundary shift at the tip vs. the notch?")
    labels.add_banner(fig)
    plt.tight_layout()
    os.makedirs("results/figures", exist_ok=True)
    out_fig = "results/figures/phase3_tip_notch_sensitivity_test.png"
    plt.savefig(out_fig, dpi=140)
    print(f"\nSaved {out_fig}")
