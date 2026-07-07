"""Phase 3.2 upgrade — RF matched-filter / cross-correlation echo detection.

Per PIPELINE_STATUS_AND_ROADMAP.md's recommendation (Level 1): replace the
envelope first-threshold-crossing detector (phase3_motion_recovery.py,
fragile -- cliff-edge RMSE collapse at just 2% noise, see LOG.md run -07)
with a matched filter -- cross-correlating the received trace against the
KNOWN transmitted pulse shape. This is the classical optimal detector for
recovering a known waveform buried in noise (processing gain scales with
the number of independent cycles in the pulse), and it uses the full
waveform, not just its envelope's first threshold crossing.

Reuses phase3_motion_recovery.py's simulation infrastructure (geometry,
medium, sources, timing) unchanged -- only the DETECTOR is new. This lets
the comparison below isolate the effect of the detection method alone.
"""

import numpy as np
from scipy.signal import correlate

import phase3_motion_recovery as pmr  # reuse geometry/sim/timing, unchanged
import phase2_config as cfg
import phase3_config as p3cfg
import labels

from matplotlib import pyplot as plt
import os

# Matched-filter template: the clean transmitted pulse shape, same dt as
# the simulation, spanning one pulse duration.
_duration = cfg.N_CYCLES / cfg.F0_HZ
_template_t = np.arange(0, _duration, pmr.dt)
_template = pmr.toneburst(_template_t)


MF_THRESHOLD_FRAC = 0.3  # fraction of the post-exclusion |correlation| peak


def recover_range_mm_matched_filter(trace, c_ref):
    """Cross-correlate the post-exclusion trace against the known
    transmitted pulse shape; take the FIRST |correlation| threshold
    crossing (not global argmax) as the echo arrival time.

    Bug caught during validation (see LOG.md): a first attempt used
    global argmax(|corr|), reintroducing the exact same failure mode that
    motivated switching the envelope detector to first-crossing earlier
    (LOG.md run -07) -- the near-boundary echo's correlation peak
    (~1.145e-3) and a later multipath's peak (~1.054e-3) were close
    enough that argmax was one noise realization away from picking the
    wrong one, exactly like the earlier argmax(|amplitude|) bug. Fixed by
    using first-threshold-crossing on |correlation|, mirroring the
    envelope detector's fix."""
    mask = pmr.t_arr > pmr.DIRECT_EXCLUDE_S
    segment = trace[mask]
    if len(segment) < len(_template):
        return np.nan
    corr = np.abs(correlate(segment, _template, mode="valid"))
    threshold = MF_THRESHOLD_FRAC * corr.max()
    idx_local = np.argmax(corr > threshold)  # first True
    t_echo = pmr.t_arr[mask][idx_local] + _duration / 2
    return t_echo * c_ref / 2 * 1e3  # meters -> mm


def recover_outer_radius_mm_matched_filter(trace, c_ref):
    return pmr.vertical_dist_mm - recover_range_mm_matched_filter(trace, c_ref)


if __name__ == "__main__":
    print(f"*** {labels.PENDING_SIGNOFF_BANNER} ***")
    print(f"*** {labels.TOY_EXACT_GT_CAPTION} ***")
    print("Comparing envelope-threshold vs. matched-filter detection across "
          "the same noise sweep (falsifiable test: does the cliff-edge "
          "failure at 2% noise, LOG.md run -07, become gradual instead?)")

    c_ref = cfg.CHEST_WALL_PROXY.sound_speed
    phases = np.linspace(0, 1, p3cfg.N_FRAMES)
    lv_radii_motion = [p3cfg.lv_radius_at_phase(p) for p in phases]
    ground_truth_outer_radius_mm = np.array([
        (r + p3cfg.WALL_THICKNESS_CELLS) * cfg.DX_M * 1e3 for r in lv_radii_motion
    ])

    print("=== Motion sweep (reusing phase3_motion_recovery's simulate_frame) ===")
    traces_motion = pmr.run_sweep(lv_radii_motion, "motion")

    rng = np.random.default_rng(0)  # fixed seed, per CLAUDE.md

    def recover_with_noise(traces, noise_level, recover_fn):
        results = []
        for trace in traces:
            peak = np.max(np.abs(trace))
            noisy = trace + rng.normal(0, noise_level * peak, size=trace.shape)
            results.append(recover_fn(noisy, c_ref))
        return np.array(results)

    baseline_pred = np.mean(ground_truth_outer_radius_mm)
    baseline_rmse = np.sqrt(np.mean((ground_truth_outer_radius_mm - baseline_pred) ** 2))

    print(f"\n{'noise':>8} {'envelope RMSE (mm)':>20} {'matched-filter RMSE (mm)':>26}")
    envelope_rmses, mf_rmses = [], []
    for noise_level in p3cfg.NOISE_LEVELS:
        rng_state = rng.bit_generator.state  # reuse identical noise draws for both detectors
        recovered_env = recover_with_noise(traces_motion, noise_level,
                                            pmr.recover_outer_radius_mm)
        rng.bit_generator.state = rng_state
        recovered_mf = recover_with_noise(traces_motion, noise_level,
                                          recover_outer_radius_mm_matched_filter)

        rmse_env = np.sqrt(np.mean((recovered_env - ground_truth_outer_radius_mm) ** 2))
        rmse_mf = np.sqrt(np.mean((recovered_mf - ground_truth_outer_radius_mm) ** 2))
        envelope_rmses.append(rmse_env)
        mf_rmses.append(rmse_mf)
        print(f"{noise_level:>8} {rmse_env:>20.4f} {rmse_mf:>26.4f}")

    print(f"\nnaive constant-baseline RMSE: {baseline_rmse:.4f}mm (for reference)")

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(p3cfg.NOISE_LEVELS, envelope_rmses, "o-", label="envelope threshold (old)")
    ax.plot(p3cfg.NOISE_LEVELS, mf_rmses, "s-", label="matched filter (new)")
    ax.axhline(baseline_rmse, color="gray", linestyle=":", label="naive constant baseline")
    ax.set_xlabel("noise level (arbitrary, fraction of trace peak)")
    ax.set_ylabel("RMSE vs. prescribed motion (mm)")
    ax.set_title("Detector comparison: does matched filtering fix the\n"
                 "cliff-edge noise fragility? (TOY: exact prescribed ground truth)")
    ax.legend(fontsize=9)
    plt.tight_layout(rect=[0, 0.06, 1, 1])
    labels.add_banner(fig)
    os.makedirs("results/figures", exist_ok=True)
    plt.savefig("results/figures/phase3_detector_comparison.png", dpi=150)
    print("Saved results/figures/phase3_detector_comparison.png")
