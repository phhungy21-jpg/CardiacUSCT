"""Phase 3 — robust/outlier-aware temporal estimator (the 3rd
literature-grounded direction from run -67: robust Kalman filtering +
cardiac speckle-tracking's confidence-gated spatiotemporal
regularization), turning run -65's diagnostic-only temporal-
consistency FLAG into an actual estimator.

DESIGN (precision-weighted fusion, a single-step Kalman-style update,
per user's explicit safety concern: "wouldn't [borrowing from confident
neighbors] be a concern [the same way strong-signal-bias-the-whole-
prediction was]?"):

    posterior_i = (precision_own_i * own_i + precision_prior_i * prior_i)
                  / (precision_own_i + precision_prior_i)

`precision_own_i` is derived from that frame's OWN Coherence Factor
(run -67) -- NOT a fixed constant -- so a frame with strong, genuine
acoustic evidence keeps its own estimate almost unchanged regardless of
whether it disagrees with its neighbors (a real abrupt event is NOT
masked), while a frame with near-zero coherence (no real peak at all,
like patient023's phase 2/5, run -64) gets pulled toward the neighbor-
based prior, which is the only informative evidence available for that
frame. This is the same design principle as run -46's guard band and
run -57's local-max rule: use a MEASURED quantity to decide how much to
trust something, never a fixed assumption about what the answer should
be.

`prior_i` is the average of the two temporal neighbors' OWN estimates
(same construction as run -65's diagnostic check) -- not a hard-coded
"expected smooth value", just the best available alternative evidence
when a frame's own data is uninformative.
"""

import numpy as np


def precision_from_cf(cf, floor=0.02, sharpness=1.0):
    """Monotonic map from Coherence Factor (bounded, roughly [1/N, 1])
    to an unbounded precision-like weight -- CF near 0 (incoherent,
    noise-level) maps to near-zero precision; CF near 1 maps to high
    precision. `floor` avoids exactly-zero precision. A `sharpness`
    exponent was tried (3.0) to more decisively favor "okay" frames
    over the neighbor prior, but TESTED WORSE (posterior RMSE rose from
    0.44mm to 0.52mm inner, 0.95mm to 1.39mm outer) -- when every CF in
    a cycle is only moderate (this dataset: 0.13-0.51, none near 1.0),
    a power >1 on a ratio that's often <1 shrinks ALL precisions
    together rather than separating good from poor. Kept at 1.0 (the
    tested, better-performing choice) rather than the untested,
    plausible-sounding "sharper must be better" assumption."""
    cf_c = max(cf, floor)
    return (cf_c / (1.0 - min(cf, 0.999) + 1e-9)) ** sharpness


def robust_temporal_fuse(values, cfs, prior_precision_scale=1.0):
    """Precision-weighted fusion across a periodic cycle (frame i's
    neighbors wrap around, matching the cardiac-cycle structure already
    used in run -65's diagnostic). Returns (posterior_values,
    own_precisions, prior_values) for full transparency -- the raw
    per-frame value is NEVER discarded, only combined."""
    n = len(values)
    values = np.asarray(values, dtype=float)
    own_precisions = np.array([precision_from_cf(cf) for cf in cfs])
    prior_values = np.array([np.mean([values[(i - 1) % n], values[(i + 1) % n]]) for i in range(n)])
    # Prior precision: informed by the AVERAGE precision of the two
    # neighbors used to build it -- a prior built from two ALSO-low-
    # confidence neighbors should itself be trusted less.
    neighbor_precisions = np.array([
        np.mean([own_precisions[(i - 1) % n], own_precisions[(i + 1) % n]]) for i in range(n)
    ])
    prior_precisions = prior_precision_scale * neighbor_precisions

    posterior = (own_precisions * values + prior_precisions * prior_values) / (own_precisions + prior_precisions)
    return posterior, own_precisions, prior_values, prior_precisions


if __name__ == "__main__":
    print("Testing robust temporal fusion on patient023's ALREADY-COMPUTED "
          "8-probe motion-cycle results (run -67 console output, copied here "
          "directly -- no new simulation) before building a synthetic "
          "abrupt-event validation test.")

    fractions = [0.00, 0.19, 0.61, 0.95, 0.95, 0.61, 0.19, 0.00]
    true_inner_mm = [6.02, 5.93, 5.40, 4.93, 4.93, 5.40, 5.93, 6.02]
    fitted_inner_mm = [5.72, 4.94, 7.10, 5.03, 5.03, 7.10, 4.94, 5.72]
    cf_inner = [0.428, 0.496, 0.270, 0.507, 0.507, 0.270, 0.496, 0.428]

    true_outer_mm = [8.82, 8.82, 8.67, 8.55, 8.55, 8.67, 8.82, 8.82]
    fitted_outer_mm = [8.16, 6.48, 11.16, 8.20, 8.20, 11.16, 6.48, 8.16]
    cf_outer = [0.183, 0.323, 0.134, 0.168, 0.168, 0.134, 0.323, 0.183]

    for label, true_mm, fitted_mm, cf in [
        ("inner", true_inner_mm, fitted_inner_mm, cf_inner),
        ("outer", true_outer_mm, fitted_outer_mm, cf_outer),
    ]:
        posterior, prec_own, prior, prec_prior = robust_temporal_fuse(fitted_mm, cf)
        print(f"\n=== {label} ===")
        raw_err = np.abs(np.array(fitted_mm) - np.array(true_mm))
        post_err = np.abs(posterior - np.array(true_mm))
        for i in range(8):
            flag = " <-- own evidence weak, prior dominates" if prec_own[i] < prec_prior[i] else ""
            print(f"  phase {i} (frac={fractions[i]:.2f}): true={true_mm[i]:.2f} raw={fitted_mm[i]:.2f} "
                  f"(err={raw_err[i]:.2f}) -> posterior={posterior[i]:.2f} (err={post_err[i]:.2f}) "
                  f"[own_prec={prec_own[i]:.2f} vs prior_prec={prec_prior[i]:.2f}]{flag}")
        print(f"  RAW RMSE={np.sqrt(np.mean(raw_err**2)):.4f}mm  ->  POSTERIOR RMSE={np.sqrt(np.mean(post_err**2)):.4f}mm")
