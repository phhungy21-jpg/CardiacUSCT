"""Phase 3.2 upgrade — Level 2: reference-anchored frame-to-frame tracking.

Per PIPELINE_STATUS_AND_ROADMAP.md's run -14 update: Level 1 (matched
filtering) genuinely helped (~30-40% RMSE reduction) but didn't fix the
fragility, because the real bottleneck is that TWO reflecting interfaces
(chest-wall/myocardium, blood/myocardium) are comparably weak and any
single-echo detector searching the FULL post-exclusion window is
ambiguous between them under noise.

Level 2 fix: don't search the whole window. Use the known ED (frame 0)
reference trace to locate the near-boundary echo ONCE (a "calibration"
step, analogous to acquiring a clean reference frame in real elastography/
speckle tracking), then for every other frame, restrict the search to a
NARROW window around that reference location -- sized to the physically
plausible motion range (computed below: max round-trip time shift over
the whole prescribed cardiac cycle is ~2.53us), which is comfortably
smaller than the ~3.7us separation to the far multipath echo identified in
LOG.md run -14. This structurally excludes the far multipath rather than
relying on amplitude/correlation-strength thresholds to avoid it.
"""

import numpy as np
from scipy.signal import correlate

import phase3_motion_recovery as pmr
import phase2_config as cfg
import phase3_config as p3cfg
import labels

from matplotlib import pyplot as plt
import os

_duration = cfg.N_CYCLES / cfg.F0_HZ
_template_t = np.arange(0, _duration, pmr.dt)
_template = pmr.toneburst(_template_t)

# Search-window margin: computed from the actual prescribed motion range
# (not guessed) -- see chat log: max outer-radius change over the cycle
# is 2.0mm, giving max round-trip time shift ~2.53us. Add modest margin,
# stay safely below the ~3.7us separation to the far multipath (run -14).
SEARCH_MARGIN_S = 3.0e-6


def find_reference_echo_time(reference_trace, c_ref):
    """Locate the near-boundary echo in a clean (noiseless) reference
    trace using the full-window matched filter (run -14's Level 1
    method) -- a one-time calibration step, not repeated per frame."""
    mask = pmr.t_arr > pmr.DIRECT_EXCLUDE_S
    segment = reference_trace[mask]
    corr = np.abs(correlate(segment, _template, mode="valid"))
    threshold = 0.3 * corr.max()
    idx_local = np.argmax(corr > threshold)
    return pmr.t_arr[mask][idx_local] + _duration / 2


def recover_range_mm_reference_tracking(trace, t_ref, c_ref):
    """Cross-correlate against the transmitted-pulse template, but ONLY
    within a narrow window around t_ref (+/- SEARCH_MARGIN_S) -- not the
    full post-exclusion trace.

    Bug caught during validation (see LOG.md): the window's lower bound
    (t_ref - SEARCH_MARGIN_S =~ 1.05us) fell BELOW DIRECT_EXCLUDE_S
    (3.0us), so the window dipped into the direct-pulse-contaminated
    region and locked onto that instead -- recovered range was IDENTICAL
    across every frame regardless of true geometry, the exact same
    tell as the original truncated-toneburst bug. Fixed by clamping the
    window's lower bound to DIRECT_EXCLUDE_S."""
    window_mask = ((pmr.t_arr > max(t_ref - SEARCH_MARGIN_S, pmr.DIRECT_EXCLUDE_S)) &
                   (pmr.t_arr < t_ref + SEARCH_MARGIN_S))
    segment = trace[window_mask]
    if len(segment) < len(_template):
        return np.nan
    corr = np.abs(correlate(segment, _template, mode="valid"))
    idx_local = np.argmax(corr)  # within this narrow window, global argmax
                                  # is safe -- the far multipath is excluded
                                  # by construction, not by thresholding.
    t_window_start = pmr.t_arr[window_mask][0]
    t_echo = t_window_start + idx_local * pmr.dt + _duration / 2
    return t_echo * c_ref / 2 * 1e3


def recover_outer_radius_mm_reference_tracking(trace, t_ref, c_ref):
    return pmr.vertical_dist_mm - recover_range_mm_reference_tracking(trace, t_ref, c_ref)


if __name__ == "__main__":
    print(f"*** {labels.PENDING_SIGNOFF_BANNER} ***")
    print(f"*** {labels.TOY_EXACT_GT_CAPTION} ***")
    print(f"Level 2: reference-anchored tracking, search margin={SEARCH_MARGIN_S*1e6:.2f}us")

    c_ref = cfg.CHEST_WALL_PROXY.sound_speed
    phases = np.linspace(0, 1, p3cfg.N_FRAMES)
    lv_radii_motion = [p3cfg.lv_radius_at_phase(p) for p in phases]
    ground_truth_outer_radius_mm = np.array([
        (r + p3cfg.WALL_THICKNESS_CELLS) * cfg.DX_M * 1e3 for r in lv_radii_motion
    ])

    print("=== Motion sweep (reusing phase3_motion_recovery's simulate_frame) ===")
    traces_motion = pmr.run_sweep(lv_radii_motion, "motion")

    # One-time calibration: locate the reference echo in the CLEAN (noiseless)
    # frame-0 (ED) trace.
    t_ref = find_reference_echo_time(traces_motion[0], c_ref)
    print(f"Reference echo time (frame 0, ED, noiseless): {t_ref:.3e}s "
          f"-> range={t_ref*c_ref/2*1e3:.3f}mm")

    rng = np.random.default_rng(0)

    def recover_with_noise(traces, noise_level, recover_fn):
        results = []
        for trace in traces:
            peak = np.max(np.abs(trace))
            noisy = trace + rng.normal(0, noise_level * peak, size=trace.shape)
            results.append(recover_fn(noisy))
        return np.array(results)

    baseline_pred = np.mean(ground_truth_outer_radius_mm)
    baseline_rmse = np.sqrt(np.mean((ground_truth_outer_radius_mm - baseline_pred) ** 2))

    print(f"\n{'noise':>8} {'envelope (mm)':>16} {'matched-filter (mm)':>20} {'ref-tracking (mm)':>20}")
    import phase3_matched_filter_recovery as mfr
    env_rmses, mf_rmses, rt_rmses = [], [], []
    for noise_level in p3cfg.NOISE_LEVELS:
        state = rng.bit_generator.state
        rec_env = recover_with_noise(traces_motion, noise_level,
                                      lambda t: pmr.recover_outer_radius_mm(t, c_ref))
        rng.bit_generator.state = state
        rec_mf = recover_with_noise(traces_motion, noise_level,
                                    lambda t: mfr.recover_outer_radius_mm_matched_filter(t, c_ref))
        rng.bit_generator.state = state
        rec_rt = recover_with_noise(traces_motion, noise_level,
                                    lambda t: recover_outer_radius_mm_reference_tracking(t, t_ref, c_ref))

        rmse_env = np.sqrt(np.mean((rec_env - ground_truth_outer_radius_mm) ** 2))
        rmse_mf = np.sqrt(np.mean((rec_mf - ground_truth_outer_radius_mm) ** 2))
        rmse_rt = np.sqrt(np.mean((rec_rt - ground_truth_outer_radius_mm) ** 2))
        env_rmses.append(rmse_env); mf_rmses.append(rmse_mf); rt_rmses.append(rmse_rt)
        print(f"{noise_level:>8} {rmse_env:>16.4f} {rmse_mf:>20.4f} {rmse_rt:>20.4f}")

    print(f"\nnaive constant-baseline RMSE: {baseline_rmse:.4f}mm")

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(p3cfg.NOISE_LEVELS, env_rmses, "o-", label="envelope threshold (Level 0)")
    ax.plot(p3cfg.NOISE_LEVELS, mf_rmses, "s-", label="matched filter, full window (Level 1)")
    ax.plot(p3cfg.NOISE_LEVELS, rt_rmses, "^-", label="reference-anchored, narrow window (Level 2)")
    ax.axhline(baseline_rmse, color="gray", linestyle=":", label="naive constant baseline")
    ax.set_xlabel("noise level (arbitrary, fraction of trace peak)")
    ax.set_ylabel("RMSE vs. prescribed motion (mm)")
    ax.set_title("Detector comparison across 3 levels\n(TOY: exact prescribed ground truth)")
    ax.legend(fontsize=9)
    plt.tight_layout(rect=[0, 0.06, 1, 1])
    labels.add_banner(fig)
    os.makedirs("results/figures", exist_ok=True)
    plt.savefig("results/figures/phase3_detector_comparison_3level.png", dpi=150)
    print("Saved results/figures/phase3_detector_comparison_3level.png")
