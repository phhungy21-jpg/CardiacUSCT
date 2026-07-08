"""Phase 3 -- extends run -74's tip/notch injectivity probe to ALL 10
vertices of the synthetic heart phantom, not just those two locations.

Per user's objection to run -74: "that's just tested information in 1
patient" -- correct, and more precisely it was 1 synthetic shape, 1
location each for "tip"/"notch", 1 perturbation magnitude, 1 probe
count. This script closes the narrowest part of that gap first (cheap,
reuses the same phantom/infrastructure, no new simulation setup): does
the tip-vs-notch asymmetry reflect a real, general relationship with
local boundary geometry (convex vs. concave, sharp vs. smooth), or was
it a coincidence of which two vertices got picked? Perturbs each of the
10 vertices independently (one at a time, others held fixed, same
`perturb_vertex`/`build_medium_from_vertices` machinery as run -74),
and reports the same raw-score-curve tracking test at each one.

Companion script `phase3_real_contour_sensitivity_test.py` addresses
the other half of the objection (real anatomy, not a synthetic
cartoon).
"""

import numpy as np

from phase3_tip_notch_sensitivity_test import (
    perturb_vertex, build_medium_from_vertices, score_along_ray,
    HEART_R, PERTURB_CELLS, R_GRID,
)
from phase3_mri_8probe_test import capture_all_pairs, labels
from phase3_heart_shape_offcenter_test import heart_vertices, HEART_UNIT_VERTICES, SHIFTED_CENTER

from matplotlib import pyplot as plt
import os

VERTEX_NAMES = [
    "0: bottom tip (sharp convex)",
    "1: right lower flank",
    "2: right lobe outer widest",
    "3: right lobe top (outer)",
    "4: right lobe top (inner, near notch)",
    "5: NOTCH (concave)",
    "6: left lobe top (inner, near notch)",
    "7: left lobe top (outer)",
    "8: left lobe outer widest",
    "9: left lower flank",
]


def vertex_theta_deg(dx_, dy):
    """Angle convention matching direction_vector: theta=0 -> up (-row).
    Derived analytically from the unit vertex's (dx_, dy) components
    (row = center - dy*R, col = center + dx_*R), verified against run
    -74's tip (180deg) and notch (0deg) hardcoded values."""
    return np.degrees(np.arctan2(dx_, dy)) % 360


if __name__ == "__main__":
    print(f"*** {labels.PENDING_SIGNOFF_BANNER} ***")
    print(f"*** {labels.TOY_EXACT_GT_CAPTION} ***")
    print("INJECTIVITY PROBE, ALL 10 VERTICES: does run -74's tip/notch "
          "asymmetry reflect a general convex/concave (or sharp/smooth) "
          "relationship, or was it specific to those two locations? Same "
          f"method as run -74 (8 probes, +{PERTURB_CELLS:.0f}-cell radial "
          "perturbation, raw score curve, no peak-selector involved).")

    base_verts = heart_vertices(HEART_R)
    print("\n=== Simulating baseline heart (8 probes) ===")
    pairs_base = capture_all_pairs(build_medium_from_vertices(base_verts))

    results = []
    for idx in range(10):
        dx_, dy = HEART_UNIT_VERTICES[idx]
        theta = vertex_theta_deg(dx_, dy)
        mag = np.hypot(dx_, dy)
        true_R_base = HEART_R * mag
        true_R_pert = true_R_base + PERTURB_CELLS

        print(f"\n=== Vertex {idx} ({VERTEX_NAMES[idx]}), theta={theta:.1f}deg, "
              f"true R={true_R_base:.1f}->{true_R_pert:.1f} cells ===")
        pert_verts = perturb_vertex(base_verts, idx, PERTURB_CELLS)
        pairs_pert = capture_all_pairs(build_medium_from_vertices(pert_verts))

        scores_base = score_along_ray(pairs_base, theta, R_GRID)
        scores_pert = score_along_ray(pairs_pert, theta, R_GRID)

        idx_old = np.argmin(np.abs(R_GRID - true_R_base))
        idx_new = np.argmin(np.abs(R_GRID - true_R_pert))
        s_old_base, s_old_pert = scores_base[idx_old], scores_pert[idx_old]
        s_new_base, s_new_pert = scores_base[idx_new], scores_pert[idx_new]
        argmax_base = R_GRID[np.argmax(scores_base)]
        argmax_pert = R_GRID[np.argmax(scores_pert)]
        argmax_shift = argmax_pert - argmax_base
        direction_correct = (s_old_pert < s_old_base) and (s_new_pert > s_new_base)
        tracks = abs(argmax_shift - PERTURB_CELLS) < PERTURB_CELLS * 0.5

        print(f"  score at OLD true R: base={s_old_base:.4g} -> pert={s_old_pert:.4g} "
              f"({'decreased (expected)' if s_old_pert < s_old_base else 'INCREASED (unexpected)'})")
        print(f"  score at NEW true R: base={s_new_base:.4g} -> pert={s_new_pert:.4g} "
              f"({'increased (expected)' if s_new_pert > s_new_base else 'DECREASED (unexpected)'})")
        print(f"  raw argmax: base={argmax_base:.1f} -> pert={argmax_pert:.1f} "
              f"(shift={argmax_shift:+.1f} vs true {PERTURB_CELLS:+.1f}) "
              f"-- {'TRACKS' if tracks else 'DOES NOT TRACK'}, "
              f"direction {'CORRECT' if direction_correct else 'INCONSISTENT'}")

        results.append(dict(idx=idx, name=VERTEX_NAMES[idx], theta=theta, mag=mag,
                             true_R_base=true_R_base, argmax_base=argmax_base, argmax_pert=argmax_pert,
                             argmax_shift=argmax_shift, tracks=tracks, direction_correct=direction_correct,
                             s_old_base=s_old_base, s_old_pert=s_old_pert,
                             s_new_base=s_new_base, s_new_pert=s_new_pert))

    print("\n--- Summary across all 10 vertices ---")
    n_tracks = sum(r["tracks"] for r in results)
    n_direction_correct = sum(r["direction_correct"] for r in results)
    print(f"  {n_tracks}/10 vertices TRACK the true perturbation in argmax")
    print(f"  {n_direction_correct}/10 vertices show physically-CONSISTENT direction of change")
    for r in results:
        clean = r["tracks"] and r["direction_correct"]
        print(f"  vertex {r['idx']} ({r['name']:42s}) theta={r['theta']:6.1f}deg "
              f"mag={r['mag']:.2f} argmax_shift={r['argmax_shift']:+5.1f} "
              f"{'CLEAN' if clean else ('PARTIAL' if (r['tracks'] or r['direction_correct']) else 'GHOST-DOMINATED')}")

    fig, ax = plt.subplots(figsize=(9, 5))
    thetas_plot = [r["theta"] for r in results]
    shifts_plot = [r["argmax_shift"] for r in results]
    colors = ["green" if (r["tracks"] and r["direction_correct"]) else
              ("orange" if (r["tracks"] or r["direction_correct"]) else "red") for r in results]
    ax.bar(range(10), shifts_plot, color=colors)
    ax.axhline(PERTURB_CELLS, color="k", linestyle="--", label=f"true shift ({PERTURB_CELLS:.0f} cells)")
    ax.set_xticks(range(10))
    ax.set_xticklabels([f"v{r['idx']}\n{r['theta']:.0f}deg" for r in results], fontsize=8)
    ax.set_ylabel("raw argmax shift (cells)")
    ax.set_title("Injectivity probe across all 10 heart-phantom vertices\n"
                 "green=clean (tracks + correct direction), orange=partial, red=ghost-dominated")
    ax.legend(fontsize=8)
    labels.add_banner(fig)
    plt.tight_layout()
    os.makedirs("results/figures", exist_ok=True)
    out_fig = "results/figures/phase3_heart_all_vertices_sensitivity_test.png"
    plt.savefig(out_fig, dpi=140)
    print(f"\nSaved {out_fig}")
