"""Prepares the (X, y) training/test tensors from the beating-heart
Colab simulation output: X = matched-filter envelope per (phase, angle)
trace (a fixed, physically-justified pulse-compression front end --
correlating against the KNOWN transmit waveform, not a learned or
hand-picked feature -- everything downstream of that is left entirely
to the CNN, unlike every earlier candidate-generation/rescoring method
in this project), y = the true inner-boundary radius at that
(phase, angle), in this project's domain cell units.

patient001 (TRAIN, ~1.6% real contraction) -> 8 phases x 18 angles = 144
examples. patient023 (TEST, held out entirely, ~18% real contraction,
never touched during training) -> 144 examples. This is the direct
generalization test: does a simple CNN trained on one beating heart
"light up" (predict meaningfully) on a genuinely different, unleaked
heart's cross sections?

Reuses the cached water-only baseline (`results/patient023_reflection_
raw_traces.npz`, subsampled to the same 18-angle subset used for the
beating-heart sims) for background subtraction -- no new simulation.
"""

import numpy as np
from scipy.signal import correlate, hilbert

from phase1_reflection_channel_scout import direction_vector, polar_resample, r_at_theta
from phase1_matched_filter_echo_extraction import _lag_t_arr, _template
from phase1_rotating_transmission_scout import center
import phase2_config as cfg

ANGLE_INDICES = list(range(0, 36, 36 // 18))


def matched_filter_envelope(trace):
    correlated = correlate(trace, _template, mode="full")
    return np.abs(hilbert(correlated))


def build_dataset(patient_id):
    d = np.load(f"results/beating_{patient_id}_reflection_traces.npz", allow_pickle=True)
    traces = d["traces"]  # (n_phases, n_angles, n_samples)
    thetas = d["thetas"]
    n_phases, n_angles, _ = traces.shape

    d_water = np.load("results/patient023_reflection_raw_traces.npz")
    water_traces_full = d_water["water_traces"]
    water_traces = water_traces_full[ANGLE_INDICES]
    water_env = np.array([matched_filter_envelope(tr) for tr in water_traces])

    X, y = [], []
    for phase_idx in range(n_phases):
        inner_dom = d["inner_contours_dom"][phase_idx]
        ext_theta_in, ext_r_in = polar_resample(inner_dom, center)
        for angle_idx, theta in enumerate(thetas):
            trace = traces[phase_idx, angle_idx]
            env = matched_filter_envelope(trace)
            # align lengths: the Colab-simulated traces differ from the cached local
            # water baseline by 1 sample (minor cross-environment time-axis precision
            # difference), so truncate both to their shared length before subtracting.
            min_len = min(len(env), len(water_env[angle_idx]))
            excess = np.clip(env[:min_len] - water_env[angle_idx][:min_len], 0, None)
            X.append(excess)
            y.append(r_at_theta(theta, ext_theta_in, ext_r_in))

    return np.array(X, dtype=np.float32), np.array(y, dtype=np.float32)


if __name__ == "__main__":
    X_train, y_train = build_dataset("patient001")
    X_test, y_test = build_dataset("patient023")
    print(f"train (patient001): X={X_train.shape}, y range=[{y_train.min():.1f}, {y_train.max():.1f}] cells")
    print(f"test  (patient023): X={X_test.shape}, y range=[{y_test.min():.1f}, {y_test.max():.1f}] cells")
    np.savez("results/beating_heart_dataset.npz",
             X_train=X_train, y_train=y_train, X_test=X_test, y_test=y_test)
    print("Saved results/beating_heart_dataset.npz")
