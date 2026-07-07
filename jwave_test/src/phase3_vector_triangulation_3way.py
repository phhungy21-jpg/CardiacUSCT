"""Phase 3 — 3-sub-aperture vector triangulation: establish the angle-vs-
robustness trend, and an over-determined least-squares vector solve.

Per user request: 2 angles only gives a single A-vs-B data point (run
-20's surprising finding that near-normal A was MORE noise-fragile than
oblique B). A 3rd angle at an INTERMEDIATE look-angle (35.5 degrees,
between A's 0 and B's 66.1) lets us see whether robustness trends
monotonically with angle (supporting the wrong-echo-ambiguity-near-
normal-incidence hypothesis from run -20) or something non-monotonic.
Also enables an over-determined (3 equations, 2 unknowns) least-squares
vector solve instead of the exact 2x2 solve used in run -19/-20, which
should average out some noise via redundancy.

Reuses phase3_vector_triangulation.py's validated infrastructure
(delay-focusing, sequential tracking, local extended time axis) --
only adds sub-aperture C and the 3-way comparison/solve.
"""

import numpy as np

import phase3_vector_triangulation as vt
import phase2_config as cfg
import phase3_config as p3cfg
import labels

from matplotlib import pyplot as plt
import os

c_ref = vt.c_ref


if __name__ == "__main__":
    print(f"*** {labels.PENDING_SIGNOFF_BANNER} ***")
    print(f"*** {labels.TOY_EXACT_GT_CAPTION} ***")
    print("3-sub-aperture vector triangulation: establishing the angle-vs-"
          "robustness trend (0, 35.5, 66.1 degrees).")

    phases = np.linspace(0, 1, p3cfg.N_FRAMES)
    lv_radii_motion = [p3cfg.lv_radius_at_phase(p) for p in phases]
    r_boundary = np.array([r + p3cfg.WALL_THICKNESS_CELLS for r in lv_radii_motion])

    target_row_ref, target_col_ref = vt.target_position_at_radius(r_boundary[0])
    print(f"Reference (ED) target position: row={target_row_ref:.2f}, col={target_col_ref:.2f}")

    center_col_A = target_col_ref
    center_col_C = target_col_ref - 30   # 35.5 deg
    center_col_B = target_col_ref - 95   # 66.1 deg (same as run -19/-20)

    sources_A, _, _ = vt.build_focused_sources(center_col_A, target_row_ref, target_col_ref)
    sources_C, _, _ = vt.build_focused_sources(center_col_C, target_row_ref, target_col_ref)
    sources_B, _, _ = vt.build_focused_sources(center_col_B, target_row_ref, target_col_ref)

    u_A = vt.look_direction(center_col_A, target_row_ref, target_col_ref)
    u_C = vt.look_direction(center_col_C, target_row_ref, target_col_ref)
    u_B = vt.look_direction(center_col_B, target_row_ref, target_col_ref)
    ang_A = np.degrees(np.arccos(np.clip(u_A @ [1, 0], -1, 1)))
    ang_C = np.degrees(np.arccos(np.clip(u_C @ [1, 0], -1, 1)))
    ang_B = np.degrees(np.arccos(np.clip(u_B @ [1, 0], -1, 1)))
    print(f"u_A={u_A} ({ang_A:.1f}deg), u_C={u_C} ({ang_C:.1f}deg), u_B={u_B} ({ang_B:.1f}deg)")

    rcv_col_A = int(center_col_A) + 5
    rcv_col_C = int(center_col_C) + 5
    rcv_col_B = int(center_col_B) + 5

    print("=== Simulating sub-aperture A (0 deg) ===")
    traces_A = [np.array(vt.simulate_subaperture(r - p3cfg.WALL_THICKNESS_CELLS, sources_A)[:, rcv_col_A, vt.pmr.array_y])
                for r in r_boundary]
    print("=== Simulating sub-aperture C (35.5 deg) ===")
    traces_C = [np.array(vt.simulate_subaperture(r - p3cfg.WALL_THICKNESS_CELLS, sources_C)[:, rcv_col_C, vt.pmr.array_y])
                for r in r_boundary]
    print("=== Simulating sub-aperture B (66.1 deg) ===")
    traces_B = [np.array(vt.simulate_subaperture(r - p3cfg.WALL_THICKNESS_CELLS, sources_B)[:, rcv_col_B, vt.pmr.array_y])
                for r in r_boundary]

    def approx_t_ref(rcv_col):
        return 2 * np.sqrt((target_row_ref - vt.pmr.array_y) ** 2 +
                           (target_col_ref - rcv_col) ** 2) * cfg.DX_M / c_ref

    def refine(trace, t_guess, margin=1e-6):
        mask = (vt.t_arr_local > t_guess - margin) & (vt.t_arr_local < t_guess + margin)
        idx = np.argmax(np.abs(trace[mask]))
        return vt.t_arr_local[mask][idx]

    t_ref_A = refine(traces_A[0], approx_t_ref(rcv_col_A))
    t_ref_C = refine(traces_C[0], approx_t_ref(rcv_col_C))
    t_ref_B = refine(traces_B[0], approx_t_ref(rcv_col_B))
    print(f"t_ref: A={t_ref_A:.3e}s, C={t_ref_C:.3e}s, B={t_ref_B:.3e}s")

    template_halfwidth = 0.5e-6
    search_margin = 1.2e-6

    true_disp_mag_mm = (r_boundary - r_boundary[0]) * cfg.DX_M * 1e3
    true_disp_row_mm = true_disp_mag_mm * (-np.cos(vt.THETA))
    true_disp_col_mm = true_disp_mag_mm * (np.sin(vt.THETA))
    expected_rc = {
        "A": true_disp_row_mm * u_A[0] + true_disp_col_mm * u_A[1],
        "C": true_disp_row_mm * u_C[0] + true_disp_col_mm * u_C[1],
        "B": true_disp_row_mm * u_B[0] + true_disp_col_mm * u_B[1],
    }

    U = np.array([u_A, u_C, u_B])  # 3x2, for least-squares solve
    U_pinv = np.linalg.pinv(U)

    reference_peaks = {
        "A": np.max(np.abs(traces_A[0])),
        "C": np.max(np.abs(traces_C[0])),
        "B": np.max(np.abs(traces_B[0])),
    }

    N_REALIZATIONS = 20
    print(f"\n(RMSE averaged over {N_REALIZATIONS} independent noise realizations)")
    print(f"\n{'noise':>8} {'scalar RMSE A(0deg)':>20} {'C(35.5deg)':>12} {'B(66.1deg)':>12} "
          f"{'LSQ row':>10} {'LSQ col':>10}")
    scalar_rmse_trend = {"A": [], "C": [], "B": []}
    lsq_row_rmses, lsq_col_rmses = [], []
    for noise_level in p3cfg.NOISE_LEVELS:
        trials = {"A": [], "C": [], "B": [], "lsq_row": [], "lsq_col": []}
        seed_offsets = {"A": 11, "C": 12, "B": 13}  # fixed, reproducible (not Python's hash())
        for trial in range(N_REALIZATIONS):
            noisy = {}
            for name, traces, t_ref in [("A", traces_A, t_ref_A), ("C", traces_C, t_ref_C),
                                        ("B", traces_B, t_ref_B)]:
                rng = np.random.default_rng(1000 * trial + seed_offsets[name])
                noisy_traces = [t + rng.normal(0, noise_level * reference_peaks[name], size=t.shape)
                                for t in traces]
                tt = vt.track_sequential_range(noisy_traces, t_ref, template_halfwidth, search_margin)
                rc = tt * c_ref / 2 * 1e3 - tt[0] * c_ref / 2 * 1e3
                noisy[name] = rc
                trials[name].append(np.sqrt(np.mean((rc - expected_rc[name]) ** 2)))

            rec_row, rec_col = [], []
            for i in range(len(phases)):
                rc_vec = np.array([noisy["A"][i], noisy["C"][i], noisy["B"][i]])
                d = U_pinv @ rc_vec  # least-squares solve, 3 eqns 2 unknowns
                rec_row.append(d[0]); rec_col.append(d[1])
            rec_row = np.array(rec_row); rec_col = np.array(rec_col)
            trials["lsq_row"].append(np.sqrt(np.mean((rec_row - true_disp_row_mm) ** 2)))
            trials["lsq_col"].append(np.sqrt(np.mean((rec_col - true_disp_col_mm) ** 2)))

        for name in ["A", "C", "B"]:
            scalar_rmse_trend[name].append(np.mean(trials[name]))
        lsq_row_rmses.append(np.mean(trials["lsq_row"]))
        lsq_col_rmses.append(np.mean(trials["lsq_col"]))
        print(f"{noise_level:>8} {np.mean(trials['A']):>20.4f} {np.mean(trials['C']):>12.4f} "
              f"{np.mean(trials['B']):>12.4f} {np.mean(trials['lsq_row']):>10.4f} "
              f"{np.mean(trials['lsq_col']):>10.4f}")

    print("\nFor comparison, run -20's 2-aperture EXACT solve (noise 0.02/0.05/0.10): "
          "row=0.977/0.959/0.959mm, col=0.686/0.675/0.674mm")

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    angles = [ang_A, ang_C, ang_B]
    colors_map = plt.cm.viridis(np.linspace(0.2, 0.9, len(p3cfg.NOISE_LEVELS)))
    for i, noise_level in enumerate(p3cfg.NOISE_LEVELS):
        vals = [scalar_rmse_trend["A"][i], scalar_rmse_trend["C"][i], scalar_rmse_trend["B"][i]]
        axes[0].plot(angles, vals, "o-", color=colors_map[i], label=f"noise={noise_level}")
    axes[0].set_xlabel("look angle from boresight (deg)")
    axes[0].set_ylabel("scalar RMSE (mm)")
    axes[0].set_title("Angle-vs-robustness trend")
    axes[0].legend(fontsize=8)

    axes[1].plot(p3cfg.NOISE_LEVELS, lsq_row_rmses, "o-", label="row (3-way LSQ)")
    axes[1].plot(p3cfg.NOISE_LEVELS, lsq_col_rmses, "s-", label="col (3-way LSQ)")
    axes[1].axhline(0.977, color="gray", linestyle=":", alpha=0.5, label="2-way row (run -20)")
    axes[1].axhline(0.686, color="gray", linestyle="--", alpha=0.5, label="2-way col (run -20)")
    axes[1].set_xlabel("noise level")
    axes[1].set_ylabel("RMSE vs. true displacement (mm)")
    axes[1].set_title("3-way least-squares vs. run -20's 2-way exact solve")
    axes[1].legend(fontsize=8)

    fig.suptitle("3-sub-aperture triangulation: trend + over-determined solve\n(TOY: exact prescribed ground truth)")
    plt.tight_layout(rect=[0, 0.08, 1, 0.92])
    labels.add_banner(fig)
    os.makedirs("results/figures", exist_ok=True)
    plt.savefig("results/figures/phase3_vector_triangulation_3way.png", dpi=150)
    print("Saved results/figures/phase3_vector_triangulation_3way.png")
