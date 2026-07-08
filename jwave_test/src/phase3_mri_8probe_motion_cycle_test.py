"""Phase 3 — ISOLATED test: 8-probe + local-max + fully self-consistent
calibration (runs -56/-57/-60/-61) applied across patient023's real
8-phase motion cycle (`phase3_mri_motion_cycle_prep.py`), not just the
single static ED frame.

Per user: "and if all works out, generate the 8 frames for id23 and
log" -- after patient001's static 8-probe result confirmed the model
"still stands" (run -62), this extends the validated 8-probe pipeline
to the full real motion cycle, mirroring
`phase3_mri_motion_cycle_reconstruction.py`'s design (fixed ED
template, per-frame own centroid, frozen-scene medium rebuild per
phase) but built on the 8-probe geometry/capture/weight model instead
of the official 4-probe pipeline. Self-contained -- reuses
`phase3_mri_8probe_test.py`'s probe geometry, capture, and weight
model via import, no existing file modified.
"""

import numpy as np

from phase3_mri_8probe_test import (
    _SRC, _RCV, capture_all_pairs, build_medium_homogeneous, build_medium_real_contour,
    build_search_grid, direction_vector, fit_scale_curvature_weighted,
    _polar_resample, r_at_theta, center, dx, N, c_ref, t_arr, _ENVELOPE_GROUP_DELAY_S,
    SCALE_GRID, GUARD_BAND_CELLS, N_PROBES, labels,
)

from matplotlib import pyplot as plt
import os

PATIENT_ID = "patient023"

if __name__ == "__main__":
    print(f"*** {labels.PENDING_SIGNOFF_BANNER} ***")
    print(f"*** {labels.GT_FLOOR_CAPTION} ***")
    print(f"ISOLATED 8-probe motion-cycle test for {PATIENT_ID}: {N_PROBES} probes, "
          f"local-max-only selection, fully self-consistent calibration (run -60). "
          f"Fixed-ED-template scale-factor fit across the real 8-phase registration-"
          f"derived motion cycle.")

    d_static = np.load(f"results/mri_irregular_ring_{PATIENT_ID}_slice4.npz")
    d_motion = np.load(f"results/mri_motion_cycle_{PATIENT_ID}_slice4.npz", allow_pickle=True)

    myo_frames = d_motion["myo_frames"]
    lv_frames = d_motion["lv_frames"]
    ring_frames = d_motion["ring_frames"]
    outer_contours_true = d_motion["outer_contours"]
    inner_contours_true = d_motion["inner_contours"]
    true_inner_radius_mm = d_motion["true_inner_radius_mm"]
    true_outer_radius_mm = d_motion["true_outer_radius_mm"]
    phases = d_motion["phases"]
    fractions = d_motion["fractions"]
    N_PHASES = len(phases)
    print(f"  registration quality: mean_dice={float(d_motion['mean_dice']):.3f} "
          f"(myo={float(d_motion['myo_dice']):.3f}, lv={float(d_motion['lv_dice']):.3f})")

    ring_ed = d_static["ring_mask"].astype(bool)
    lv_ed = d_static["lv_mask"].astype(bool)
    ys, xs = np.where(ring_ed)
    ring_centroid_native_ed = (ys.mean(), xs.mean())
    lys, lxs = np.where(lv_ed)
    lv_centroid_native_ed = (lys.mean(), lxs.mean())
    offset_row = int(round(center[0] - ring_centroid_native_ed[0]))
    offset_col = int(round(center[1] - ring_centroid_native_ed[1]))
    print(f"  fixed placement offset (from ED ring centroid): ({offset_row}, {offset_col})")

    outer_contour_ed_dom = d_static["outer_contour"] + np.array([offset_row, offset_col])
    inner_contour_ed_dom = d_static["inner_contour"] + np.array([offset_row, offset_col])
    lv_centroid_ed_dom = (lv_centroid_native_ed[0] + offset_row, lv_centroid_native_ed[1] + offset_col)
    ring_centroid_ed_dom = (ring_centroid_native_ed[0] + offset_row, ring_centroid_native_ed[1] + offset_col)
    ext_theta_in, ext_r_in = _polar_resample(inner_contour_ed_dom, lv_centroid_ed_dom)
    ext_theta_out, ext_r_out = _polar_resample(outer_contour_ed_dom, ring_centroid_ed_dom)
    ed_mean_r_in = ext_r_in.mean()
    ed_mean_r_out = ext_r_out.mean()
    print(f"  fixed ED template: mean inner radius={ed_mean_r_in:.1f} cells, mean outer radius={ed_mean_r_out:.1f} cells")

    needed_extent = max(ed_mean_r_out, ed_mean_r_in) * SCALE_GRID.max() + 15.0
    img_rows_g, img_cols_g = build_search_grid(needed_extent, margin=15.0, density_per_cell=100.0 / 180.0)
    print(f"  search grid: {len(img_rows_g)}x{len(img_cols_g)}, +/-{needed_extent:.0f} cells")

    print(f"\n=== Simulating homogeneous reference (shared across all phases, {N_PROBES} transmits) ===")
    pairs_ref = capture_all_pairs(build_medium_homogeneous())

    rows_native, cols_native = np.mgrid[0:ring_frames.shape[1], 0:ring_frames.shape[2]]
    rows_dom, cols_dom = rows_native + offset_row, cols_native + offset_col
    valid = (rows_dom >= 0) & (rows_dom < N[0]) & (cols_dom >= 0) & (cols_dom < N[1])

    fitted_inner_scale, fitted_outer_scale = [], []
    fitted_inner_r_mm, fitted_outer_r_mm = [], []
    frame_centroids_lv, frame_centroids_ring = [], []
    accumulators = []
    in_confidences, out_confidences = [], []
    in_prominences, out_prominences = [], []
    in_snrs, out_snrs = [], []
    in_cfs, out_cfs = [], []

    for i in range(N_PHASES):
        print(f"\n=== Phase {i+1}/{N_PHASES} (frac={fractions[i]:.2f}) ===")
        myo_i, lv_i, ring_i = myo_frames[i], lv_frames[i], ring_frames[i]

        canvas_myo = np.zeros(N, dtype=bool)
        canvas_lv = np.zeros(N, dtype=bool)
        canvas_myo[rows_dom[valid], cols_dom[valid]] = myo_i[valid]
        canvas_lv[rows_dom[valid], cols_dom[valid]] = lv_i[valid]
        label_map = np.zeros(N, dtype=int)
        label_map[canvas_myo] = 2
        label_map[canvas_lv] = 3

        lys_i, lxs_i = np.where(lv_i)
        ys_i, xs_i = np.where(ring_i)
        lv_centroid_dom = (lys_i.mean() + offset_row, lxs_i.mean() + offset_col)
        ring_centroid_dom = (ys_i.mean() + offset_row, xs_i.mean() + offset_col)
        frame_centroids_lv.append(lv_centroid_dom)
        frame_centroids_ring.append(ring_centroid_dom)

        medium = build_medium_real_contour(label_map)
        pairs_real = capture_all_pairs(medium)

        s_in, scores_in_real, in_is_peak, in_conf, in_prom, in_cf = fit_scale_curvature_weighted(pairs_real, ext_theta_in, ext_r_in, SCALE_GRID, lv_centroid_dom, img_rows_g, img_cols_g)
        _, scores_in_ref, _, _, _, _ = fit_scale_curvature_weighted(pairs_ref, ext_theta_in, ext_r_in, SCALE_GRID, lv_centroid_dom, img_rows_g, img_cols_g)
        in_peak_idx = int(np.argmin(np.abs(SCALE_GRID - s_in)))
        in_snr = scores_in_real[in_peak_idx] / (scores_in_ref[in_peak_idx] + 1e-12)

        fitted_inner_mean_radius = s_in * ed_mean_r_in
        scale_grid_guarded = SCALE_GRID[np.abs(SCALE_GRID * ed_mean_r_out - fitted_inner_mean_radius) > GUARD_BAND_CELLS]
        s_out, scores_out_real, out_is_peak, out_conf, out_prom, out_cf = fit_scale_curvature_weighted(pairs_real, ext_theta_out, ext_r_out, scale_grid_guarded, ring_centroid_dom, img_rows_g, img_cols_g)
        _, scores_out_ref, _, _, _, _ = fit_scale_curvature_weighted(pairs_ref, ext_theta_out, ext_r_out, scale_grid_guarded, ring_centroid_dom, img_rows_g, img_cols_g)
        out_peak_idx = int(np.argmin(np.abs(scale_grid_guarded - s_out)))
        out_snr = scores_out_real[out_peak_idx] / (scores_out_ref[out_peak_idx] + 1e-12)

        in_prominences.append(in_prom)
        out_prominences.append(out_prom)
        in_cfs.append(in_cf)
        out_cfs.append(out_cf)
        in_snrs.append(in_snr)
        out_snrs.append(out_snr)

        fitted_inner_scale.append(s_in)
        fitted_outer_scale.append(s_out)
        fitted_inner_r_mm.append(s_in * ed_mean_r_in * dx[0] * 1e3)
        fitted_outer_r_mm.append(s_out * ed_mean_r_out * dx[0] * 1e3)
        in_confidences.append(in_conf)
        out_confidences.append(out_conf)

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
        accumulators.append(accumulator - accumulator_ref)

        in_err_mm = abs(fitted_inner_r_mm[-1] - true_inner_radius_mm[i])
        out_err_mm = abs(fitted_outer_r_mm[-1] - true_outer_radius_mm[i])
        print(f"  inner: true_r={true_inner_radius_mm[i]:.2f}mm fitted_r={fitted_inner_r_mm[-1]:.2f}mm err={in_err_mm:.2f}mm "
              f"conf={in_conf:.2f} prominence={in_prom:.2f} snr={in_snr:.2f} CF={in_cf:.3f}")
        print(f"  outer: true_r={true_outer_radius_mm[i]:.2f}mm fitted_r={fitted_outer_r_mm[-1]:.2f}mm err={out_err_mm:.2f}mm "
              f"conf={out_conf:.2f} prominence={out_prom:.2f} snr={out_snr:.2f} CF={out_cf:.3f}")

    inner_errs = np.abs(np.array(fitted_inner_r_mm) - true_inner_radius_mm)
    outer_errs = np.abs(np.array(fitted_outer_r_mm) - true_outer_radius_mm)
    print(f"\n--- RMSE across {N_PHASES} real-motion phases (8-probe + local-max + real calibration) ---")
    print(f"  inner boundary RMSE={np.sqrt(np.mean(inner_errs**2)):.4f}mm  (per-phase: {np.round(inner_errs,2).tolist()})")
    print(f"  outer boundary RMSE={np.sqrt(np.mean(outer_errs**2)):.4f}mm  (per-phase: {np.round(outer_errs,2).tolist()})")
    print(f"  (compare to 4-probe patched-pipeline result, run -59: inner RMSE=0.8354mm, outer RMSE=1.9053mm)")
    print(f"  (compare to naive-4-probe baseline, run -54: inner RMSE=0.8014mm, outer RMSE=2.2940mm)")

    # --- Richer confidence report: prominence (absolute peak height vs.
    # curve's own dynamic range) + SNR (real peak vs. homogeneous-control
    # curve at that same candidate), per user diagnosis: "confidence =
    # best/second is not enough -- a single wrong peak can produce
    # infinite confidence if there is no second local peak."
    print(f"\n--- Richer confidence report (prominence, SNR, Coherence Factor vs. homogeneous control) ---")
    print(f"  (CF = Mallart-Fink-style coherence factor across tx/rx PAIRS at the winning candidate --")
    print(f"   CF near 1 means many pairs agree/contribute coherently; CF near 1/N_pairs means the score")
    print(f"   is dominated by a few outlier pairs -- the signature of a false/noise-level peak)")
    for i in range(N_PHASES):
        flag_in = " <-- LOW PROMINENCE/SNR/CF, UNTRUSTWORTHY" if (in_prominences[i] < 0.7 or in_snrs[i] < 1.3 or in_cfs[i] < 0.3) else ""
        flag_out = " <-- LOW PROMINENCE/SNR/CF, UNTRUSTWORTHY" if (out_prominences[i] < 0.7 or out_snrs[i] < 1.3 or out_cfs[i] < 0.3) else ""
        print(f"  phase {i} (frac={fractions[i]:.2f}): inner prom={in_prominences[i]:.2f} snr={in_snrs[i]:.2f} CF={in_cfs[i]:.3f}{flag_in}")
        print(f"  phase {i} (frac={fractions[i]:.2f}): outer prom={out_prominences[i]:.2f} snr={out_snrs[i]:.2f} CF={out_cfs[i]:.3f}{flag_out}")

    # --- Temporal-consistency check: flag any frame whose fitted radius
    # is a large outlier relative to its neighbors -- a DIAGNOSTIC flag
    # only (not a forced correction, per this project's standing rule
    # against shaping results toward an assumed-correct answer).
    print(f"\n--- Temporal-consistency check (deviation from neighbor-smoothed trajectory) ---")
    for label, seq in [("inner", fitted_inner_r_mm), ("outer", fitted_outer_r_mm)]:
        seq = np.array(seq)
        smoothed = np.array([np.mean([seq[(j - 1) % N_PHASES], seq[(j + 1) % N_PHASES]]) for j in range(N_PHASES)])
        deviation = np.abs(seq - smoothed)
        mad = np.median(np.abs(deviation - np.median(deviation))) + 1e-9
        for i in range(N_PHASES):
            outlier_score = deviation[i] / mad
            flag = " <-- TEMPORAL OUTLIER" if outlier_score > 3.0 else ""
            print(f"  {label} phase {i} (frac={fractions[i]:.2f}): fitted={seq[i]:.2f}mm, "
                  f"neighbor-avg={smoothed[i]:.2f}mm, deviation={deviation[i]:.2f}mm, outlier_score={outlier_score:.1f}{flag}")

    n_cols = 4
    n_rows = int(np.ceil(N_PHASES / n_cols))
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(3.4 * n_cols, 3.6 * n_rows))
    axes = np.array(axes).reshape(-1)
    vmax = max(np.abs(a).max() for a in accumulators)
    for i, ax in enumerate(axes[:N_PHASES]):
        ax.imshow(np.abs(accumulators[i]), cmap="hot", vmin=0, vmax=vmax, origin="upper",
                  extent=[img_cols_g.min(), img_cols_g.max(), img_rows_g.max(), img_rows_g.min()])
        true_in = inner_contours_true[i] + np.array([offset_row, offset_col])
        true_out = outer_contours_true[i] + np.array([offset_row, offset_col])
        ax.plot(true_in[:, 1], true_in[:, 0], "c--", linewidth=1, label="true inner")
        ax.plot(true_out[:, 1], true_out[:, 0], "c-", linewidth=1, label="true outer")

        s_in, s_out = fitted_inner_scale[i], fitted_outer_scale[i]
        lv_c, ring_c = frame_centroids_lv[i], frame_centroids_ring[i]
        N_ANGLES_PLOT = 144
        thetas_plot = np.linspace(0, 360, N_ANGLES_PLOT, endpoint=False)
        fit_in_pts = np.array([
            (lv_c[0] + r_at_theta(th, ext_theta_in, ext_r_in) * s_in * direction_vector(th)[0],
             lv_c[1] + r_at_theta(th, ext_theta_in, ext_r_in) * s_in * direction_vector(th)[1]) for th in thetas_plot])
        fit_out_pts = np.array([
            (ring_c[0] + r_at_theta(th, ext_theta_out, ext_r_out) * s_out * direction_vector(th)[0],
             ring_c[1] + r_at_theta(th, ext_theta_out, ext_r_out) * s_out * direction_vector(th)[1]) for th in thetas_plot])
        ax.plot(fit_in_pts[:, 1], fit_in_pts[:, 0], "g:", linewidth=1.5, label="fitted inner")
        ax.plot(fit_out_pts[:, 1], fit_out_pts[:, 0], "g-.", linewidth=1.5, label="fitted outer")

        ax.set_title(f"phase={phases[i]:.2f}, frac={fractions[i]:.2f}\n"
                     f"in_err={inner_errs[i]:.2f}mm out_err={outer_errs[i]:.2f}mm (conf={out_confidences[i]:.2f})", fontsize=8)
        ax.axis("off")
        if i == 0:
            ax.legend(fontsize=6, loc="lower left")
    for ax in axes[N_PHASES:]:
        ax.axis("off")
    fig.suptitle(f"8-probe + local-max + real calibration, real motion cycle ({PATIENT_ID})\n"
                 f"cyan = true (non-uniform real deformation), green = fitted (scale x FIXED ED template)", fontsize=10)
    plt.tight_layout(rect=[0, 0.03, 1, 0.87])
    labels.add_banner(fig)
    os.makedirs("results/figures", exist_ok=True)
    out_fig = f"results/figures/phase3_mri_8probe_motion_cycle_test_{PATIENT_ID}.png"
    plt.savefig(out_fig, dpi=130)
    print(f"\nSaved {out_fig}")
