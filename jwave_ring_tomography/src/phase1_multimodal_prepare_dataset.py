"""Extends the single-channel CNN dataset (`phase1_beating_heart_
prepare_dataset.py`, reflection envelope only, in isolation per phase)
into a MULTI-MODAL one, per user: "put all live echo measured
information we did so far into cnn to see if any multi-model
influence." Combines two of this session's channels into one input:

1. REFLECTION: the current phase's matched-filter envelope (background-
   subtracted against the water-only baseline) -- what the original CNN
   (this project's first CNN experiment) saw alone.
2. MOTION/DOPPLER: the PREVIOUS phase's envelope (cyclically -- the
   8-phase cycle already closes ED->ES->ED, fractions[0]==fractions[7]
   ==0, so phase 0's "previous" is phase 7), stacked as a second input
   channel. This gives the network DIRECT access to differential/motion
   information (run -39's validated, by far the strongest single result
   this session found: echo-timing shift tracks true motion to within
   0.03% of the analytically predicted physical slope for patient023) --
   information the original single-phase CNN never had access to at all.

Network sees both channels together and is free to learn any combination
(including an implicit difference, ratio, or anything else) -- this is
NOT the same as explicitly hand-computing a timing-shift feature; it is
the raw signals from both channels, fused at the input, letting the CNN
itself discover whatever cross-channel relationship helps.
"""

import numpy as np

from phase1_beating_heart_prepare_dataset import matched_filter_envelope, ANGLE_INDICES
from phase1_reflection_channel_scout import polar_resample, r_at_theta
from phase1_rotating_transmission_scout import center


def build_multimodal_dataset(patient_id):
    d = np.load(f"results/beating_{patient_id}_reflection_traces.npz", allow_pickle=True)
    traces = d["traces"]  # (n_phases, n_angles, n_samples)
    thetas = d["thetas"]
    n_phases, n_angles, _ = traces.shape

    d_water = np.load("results/patient023_reflection_raw_traces.npz")
    water_traces_full = d_water["water_traces"]
    water_traces = water_traces_full[ANGLE_INDICES]
    water_env = np.array([matched_filter_envelope(tr) for tr in water_traces])

    # precompute background-subtracted envelope for every phase/angle once
    excess_all = [[None] * n_angles for _ in range(n_phases)]
    for p in range(n_phases):
        for a in range(n_angles):
            env = matched_filter_envelope(traces[p, a])
            min_len = min(len(env), len(water_env[a]))
            excess_all[p][a] = np.clip(env[:min_len] - water_env[a][:min_len], 0, None)

    X, y = [], []
    for phase_idx in range(n_phases):
        prev_idx = (phase_idx - 1) % n_phases
        inner_dom = d["inner_contours_dom"][phase_idx]
        ext_theta_in, ext_r_in = polar_resample(inner_dom, center)
        for angle_idx, theta in enumerate(thetas):
            cur = excess_all[phase_idx][angle_idx]
            prev = excess_all[prev_idx][angle_idx]
            X.append(np.stack([cur, prev], axis=-1))  # (n_samples, 2)
            y.append(r_at_theta(theta, ext_theta_in, ext_r_in))

    return np.array(X, dtype=np.float32), np.array(y, dtype=np.float32)


if __name__ == "__main__":
    X_train, y_train = build_multimodal_dataset("patient001")
    X_test, y_test = build_multimodal_dataset("patient023")
    print(f"train (patient001): X={X_train.shape}, y range=[{y_train.min():.1f}, {y_train.max():.1f}] cells")
    print(f"test  (patient023): X={X_test.shape}, y range=[{y_test.min():.1f}, {y_test.max():.1f}] cells")
    np.savez("results/multimodal_heart_dataset.npz",
             X_train=X_train, y_train=y_train, X_test=X_test, y_test=y_test)
    print("Saved results/multimodal_heart_dataset.npz")
