"""Phase 3 -- decisive follow-up to run -75: is "ghost-dominated" a
genuine physical information VOID, or just a feature the current
curvature-weighted LINEAR SUM discards while the raw data still
carries real signal?

Run -75's injectivity probe tested one specific handcrafted statistic
(the curvature-weighted sum of pairwise envelope amplitudes along a
ray). A ghost-dominated result there means THAT statistic is
insensitive/backwards -- it does NOT by itself prove no combination of
the raw per-pair data could recover the boundary, since a learned model
isn't restricted to the same linear weighting. This script checks the
RAW per-pair envelope traces directly (bypassing pair_weight_at_R
entirely, not just the peak selector) at the 7 locations run -75
classified GHOST-DOMINATED, plus the synthetic notch (vertex 5) as a
POSITIVE CONTROL (known-clean, to confirm this per-pair method actually
shows a clear signature where one is known to exist before trusting its
verdict elsewhere).

For each of the 64 (tx, rx) pairs at each location: does that
INDIVIDUAL pair's envelope amplitude, at the predicted arrival time for
the true boundary point, drop where the boundary moved away from and
rise where it moved toward -- even if the WEIGHTED SUM across all pairs
does not show this? If most/all pairs are flat or backwards too, that
is real evidence of a genuine information void (no method, learned or
not, can recover it without hallucinating). If a meaningful fraction of
pairs DO respond correctly despite the summed statistic being
ghost-dominated, that is direct, concrete evidence that a learned
(nonlinear, per-pair) combination could recover real signal the current
formula throws away -- the strongest legitimate case for ML on this
data so far.
"""

import numpy as np

from phase3_mri_8probe_test import (
    _SRC, _RCV, capture_all_pairs, direction_vector, center, dx, c_ref,
    t_arr, _ENVELOPE_GROUP_DELAY_S, labels,
)
from phase3_tip_notch_sensitivity_test import perturb_vertex, build_medium_from_vertices, PERTURB_CELLS
from phase3_heart_shape_offcenter_test import heart_vertices, HEART_UNIT_VERTICES, SHIFTED_CENTER
from phase3_real_contour_sensitivity_test import (
    load_outer_contour, add_angular_bump, build_medium_isolated_boundary,
)

from matplotlib import pyplot as plt
import os


def vertex_theta_deg(dx_, dy):
    return np.degrees(np.arctan2(dx_, dy)) % 360


def per_pair_response(pairs_base, pairs_pert, theta_deg, r_old, r_new, origin):
    d_row, d_col = direction_vector(theta_deg)
    pt_old = (origin[0] + r_old * d_row, origin[1] + r_old * d_col)
    pt_new = (origin[0] + r_new * d_row, origin[1] + r_new * d_col)
    rows = []
    for (tx, rx), env_base in pairs_base.items():
        env_pert = pairs_pert[(tx, rx)]
        src, rcv = _SRC[tx], _RCV[rx]

        def t_for(pt):
            d_tx = np.hypot(pt[1] - src[0], pt[0] - src[1]) * dx[0]
            d_rx = np.hypot(pt[1] - rcv[0], pt[0] - rcv[1]) * dx[0]
            return (d_tx + d_rx) / c_ref + _ENVELOPE_GROUP_DELAY_S

        t_old, t_new = t_for(pt_old), t_for(pt_new)
        v_old_base = np.interp(t_old, t_arr, env_base, left=0, right=0)
        v_old_pert = np.interp(t_old, t_arr, env_pert, left=0, right=0)
        v_new_base = np.interp(t_new, t_arr, env_base, left=0, right=0)
        v_new_pert = np.interp(t_new, t_arr, env_pert, left=0, right=0)
        direction_correct = (v_old_pert < v_old_base) and (v_new_pert > v_new_base)
        rel_change = (v_new_pert - v_new_base) / (v_old_base + 1e-12)
        rows.append(dict(tx=tx, rx=rx, direction_correct=direction_correct, rel_change=rel_change))
    return rows


def report(label, rows):
    n_correct = sum(r["direction_correct"] for r in rows)
    n_total = len(rows)
    mean_rel = np.mean([r["rel_change"] for r in rows])
    median_rel_correct = np.median([r["rel_change"] for r in rows if r["direction_correct"]]) if n_correct else 0.0
    print(f"  [{label}] {n_correct}/{n_total} pairs ({100*n_correct/n_total:.0f}%) show CORRECT-direction response")
    print(f"  [{label}] mean relative change (all pairs) = {mean_rel:+.3f}, "
          f"median relative change (correct-direction pairs only) = {median_rel_correct:+.3f}")
    return n_correct, n_total, mean_rel


if __name__ == "__main__":
    print(f"*** {labels.PENDING_SIGNOFF_BANNER} ***")
    print(f"*** {labels.TOY_EXACT_GT_CAPTION} ***")
    print("RAW PER-PAIR SENSITIVITY CHECK: does information exist in the raw "
          "pair envelopes at run -75's GHOST-DOMINATED locations, even though "
          "the curvature-weighted SUM across pairs doesn't show it? Tests all "
          "64 (tx,rx) pairs individually, bypassing pair_weight_at_R entirely. "
          "Includes the synthetic notch (vertex 5) as a POSITIVE CONTROL.")

    summary = []

    # --- Synthetic heart: vertices 0,1,6,9 (ghost-dominated) + 5 (positive control) ---
    HEART_R = 50.0
    base_verts = heart_vertices(HEART_R)
    print("\n=== Simulating baseline heart (8 probes) ===")
    pairs_heart_base = capture_all_pairs(build_medium_from_vertices(base_verts))

    synthetic_targets = [(0, "tip (ghost-dominated)"), (1, "flank v1 (ghost-dominated)"),
                          (6, "left-inner v6 (ghost-dominated)"), (9, "flank v9 (ghost-dominated)"),
                          (5, "NOTCH (positive control, known CLEAN)")]
    for idx, desc in synthetic_targets:
        dx_, dy = HEART_UNIT_VERTICES[idx]
        theta = vertex_theta_deg(dx_, dy)
        mag = np.hypot(dx_, dy)
        r_old = HEART_R * mag
        r_new = r_old + PERTURB_CELLS
        print(f"\n=== Synthetic vertex {idx}: {desc}, theta={theta:.1f}deg, R={r_old:.1f}->{r_new:.1f} ===")
        pert_verts = perturb_vertex(base_verts, idx, PERTURB_CELLS)
        pairs_pert = capture_all_pairs(build_medium_from_vertices(pert_verts))
        rows = per_pair_response(pairs_heart_base, pairs_pert, theta, r_old, r_new, SHIFTED_CENTER)
        n_correct, n_total, mean_rel = report(f"vertex{idx}", rows)
        summary.append(dict(label=f"synthetic v{idx} ({desc.split(' (')[0]})", n_correct=n_correct,
                             n_total=n_total, mean_rel=mean_rel, rows=rows))

    # --- Real anatomy: patient001 (sharp, smooth), patient023 (smooth) ---
    real_targets = [("patient001", "sharp"), ("patient001", "smooth"), ("patient023", "smooth")]
    loaded = {}
    for patient_id, label in real_targets:
        if patient_id not in loaded:
            ext_theta, ext_r, origin = load_outer_contour(patient_id)
            print(f"\n=== Simulating baseline {patient_id} outer boundary (8 probes) ===")
            pairs_base_real = capture_all_pairs(build_medium_isolated_boundary(ext_theta, ext_r, origin))
            loaded[patient_id] = (ext_theta, ext_r, origin, pairs_base_real)
        ext_theta, ext_r, origin, pairs_base_real = loaded[patient_id]

        # recover the same curvature landmark used in run -75
        from phase3_real_contour_sensitivity_test import curvature_landmarks
        theta_sharp, theta_smooth, _, _ = curvature_landmarks(ext_theta, ext_r)
        theta_landmark = theta_sharp if label == "sharp" else theta_smooth
        r_old = np.interp(theta_landmark, ext_theta, ext_r)
        r_new = r_old + PERTURB_CELLS

        print(f"\n=== {patient_id}/{label}: theta={theta_landmark:.1f}deg, R={r_old:.1f}->{r_new:.1f} ===")
        ext_r_pert = add_angular_bump(ext_theta, ext_r, theta_landmark, PERTURB_CELLS)
        pairs_pert_real = capture_all_pairs(build_medium_isolated_boundary(ext_theta, ext_r_pert, origin))
        rows = per_pair_response(pairs_base_real, pairs_pert_real, theta_landmark, r_old, r_new, origin)
        n_correct, n_total, mean_rel = report(f"{patient_id}/{label}", rows)
        summary.append(dict(label=f"{patient_id}/{label}", n_correct=n_correct, n_total=n_total,
                             mean_rel=mean_rel, rows=rows))

    print("\n--- Summary: fraction of individual pairs showing CORRECT-direction raw response ---")
    for s in summary:
        print(f"  {s['label']:35s} {s['n_correct']:2d}/{s['n_total']:2d} "
              f"({100*s['n_correct']/s['n_total']:5.1f}%) correct-direction, mean_rel={s['mean_rel']:+.3f}")

    fig, ax = plt.subplots(figsize=(10, 5))
    labels_x = [s["label"] for s in summary]
    fracs = [100 * s["n_correct"] / s["n_total"] for s in summary]
    colors = ["green" if "NOTCH" in s["label"] or "v5" in s["label"] else "C1" for s in summary]
    ax.bar(range(len(summary)), fracs, color=colors)
    ax.axhline(50, color="gray", linestyle=":", label="50% (chance level for a binary direction test)")
    ax.set_xticks(range(len(summary)))
    ax.set_xticklabels(labels_x, rotation=30, ha="right", fontsize=8)
    ax.set_ylabel("% of 64 raw pairs showing correct-direction response")
    ax.set_title("Raw per-pair sensitivity at run -75's ghost-dominated locations\n"
                 "(green = positive control, known clean from the weighted-sum test)")
    ax.legend(fontsize=8)
    labels.add_banner(fig)
    plt.tight_layout()
    os.makedirs("results/figures", exist_ok=True)
    out_fig = "results/figures/phase3_raw_pair_sensitivity_test.png"
    plt.savefig(out_fig, dpi=140)
    print(f"\nSaved {out_fig}")
