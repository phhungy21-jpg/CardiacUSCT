"""Validates the established mechanisms from the circular positive
control (runs -08 through -22) on REAL, irregular anatomy: patient023
(strong ~45% contraction, the same patient jwave_test's own sparse-
probe investigation used as its hardest real case).

Per user: "move to patient023 to validate the established mechanism we
noted so far" -- tests, on real anatomy rather than a synthetic circle:
(1) direct outer-boundary echo detection (expected: strong, accurate,
per every prior real/synthetic result in this whole project); (2)
direct inner-boundary echo detection (expected: present but much
weaker, ~20x per the coefficient-derived strata from run -21/-22 --
this is the first REAL-anatomy test of that specific, decisive
prediction); (3) the amplitude-strata veto applied to any extra
candidate matches, to check whether the same qualitative separation
(order-1 detectable, order-2+ essentially undetectable) holds
regardless of boundary shape, since the coefficients themselves are
shape-independent physics.

Since the real contour is NOT a perfect circle, each angle's predicted
echo times use that angle's OWN measured r_outer(theta)/r_inner(theta)
(polar-resampled from the real contour, same convention as
`phase1_real_mri_transmission_tomography.py`), not a single global
R_OUTER/R_INNER constant.
"""

import numpy as np
from scipy.signal import find_peaks

from jax import numpy as jnp
from jwave import FourierSeries
from jwave.geometry import Medium

from phase1_matched_filter_echo_extraction import (
    simulate_pitch_catch_raw, matched_filter_output, _lag_t_arr, PEAK_PROMINENCE_FRACTION,
)
from phase1_reflection_channel_scout import predicted_reflection_times, polar_resample, r_at_theta
from phase1_amplitude_strata_veto import compute_coefficient_strata
from phase1_rotating_transmission_scout import center, N, domain, PROBE_RADIUS_CELLS
import phase2_config as cfg
import labels

from matplotlib import pyplot as plt
import os

PATIENT_ID = "patient023"
MRI_NPZ = f"../jwave_test/results/mri_irregular_ring_{PATIENT_ID}_slice4.npz"
N_ANGLES = 36
thetas = np.linspace(0, 360, N_ANGLES, endpoint=False)
MATCH_WINDOW_S = 1.5e-7
VETO_MARGIN = 10.0


def load_real_contours(npz_path):
    d = np.load(npz_path)
    lv_mask, myo_mask, ring_mask = d["lv_mask"].astype(bool), d["myo_mask"].astype(bool), d["ring_mask"].astype(bool)
    outer_contour, inner_contour = d["outer_contour"], d["inner_contour"]

    ys, xs = np.where(ring_mask)
    ring_centroid_native = (ys.mean(), xs.mean())
    offset_row = int(round(center[0] - ring_centroid_native[0]))
    offset_col = int(round(center[1] - ring_centroid_native[1]))

    rows_native, cols_native = np.mgrid[0:lv_mask.shape[0], 0:lv_mask.shape[1]]
    rows_dom, cols_dom = rows_native + offset_row, cols_native + offset_col
    valid = (rows_dom >= 0) & (rows_dom < N[0]) & (cols_dom >= 0) & (cols_dom < N[1])

    canvas_lv, canvas_myo = np.zeros(N, dtype=bool), np.zeros(N, dtype=bool)
    canvas_lv[rows_dom[valid], cols_dom[valid]] = lv_mask[valid]
    canvas_myo[rows_dom[valid], cols_dom[valid]] = myo_mask[valid]

    outer_contour_dom = outer_contour + np.array([offset_row, offset_col])
    inner_contour_dom = inner_contour + np.array([offset_row, offset_col])
    return canvas_lv, canvas_myo, outer_contour_dom, inner_contour_dom


def build_medium_two_tissue(canvas_lv, canvas_myo):
    sound_speed_map = np.full(N, cfg.WATER.sound_speed, dtype=np.float32)
    density_map = np.full(N, cfg.WATER.density, dtype=np.float32)
    sound_speed_map = np.where(canvas_myo, cfg.MYOCARDIUM.sound_speed, sound_speed_map).astype(np.float32)
    density_map = np.where(canvas_myo, cfg.MYOCARDIUM.density, density_map).astype(np.float32)
    sound_speed_map = np.where(canvas_lv, cfg.BLOOD.sound_speed, sound_speed_map).astype(np.float32)
    density_map = np.where(canvas_lv, cfg.BLOOD.density, density_map).astype(np.float32)
    ssm, dm = jnp.expand_dims(jnp.array(sound_speed_map), -1), jnp.expand_dims(jnp.array(density_map), -1)
    return Medium(domain=domain, sound_speed=FourierSeries(ssm, domain), density=FourierSeries(dm, domain))


def build_medium_water_only():
    sound_speed_map = np.full(N, cfg.WATER.sound_speed, dtype=np.float32)
    density_map = np.full(N, cfg.WATER.density, dtype=np.float32)
    ssm, dm = jnp.expand_dims(jnp.array(sound_speed_map), -1), jnp.expand_dims(jnp.array(density_map), -1)
    return Medium(domain=domain, sound_speed=FourierSeries(ssm, domain), density=FourierSeries(dm, domain))


if __name__ == "__main__":
    print(f"*** {labels.PENDING_SIGNOFF_BANNER} ***")
    print(f"*** {labels.TOY_EXACT_GT_CAPTION} ***")
    print(f"PATIENT023 VALIDATION: real irregular anatomy, testing the established mechanisms "
          "(direct outer/inner echo detection, amplitude-strata veto) from the circular "
          "positive control (runs -08 through -22).")

    canvas_lv, canvas_myo, outer_contour_dom, inner_contour_dom = load_real_contours(MRI_NPZ)
    ext_theta_out, ext_r_out = polar_resample(outer_contour_dom, center)
    ext_theta_in, ext_r_in = polar_resample(inner_contour_dom, center)
    max_r_outer = ext_r_out.max()
    print(f"  {PATIENT_ID} outer boundary: mean radius={ext_r_out[:-1].mean():.1f} cells, "
          f"max={max_r_outer:.1f} cells (probe radius={PROBE_RADIUS_CELLS})")
    if max_r_outer > PROBE_RADIUS_CELLS * 0.85:
        print(f"  WARNING: outer boundary extent close to probe radius -- check for overlap")

    strata = compute_coefficient_strata()
    print(f"\n  coefficient-predicted amplitude strata (ratio to direct outer): "
          f"inner_k0={strata['inner_k0']/strata['outer']:.3e}, "
          f"k1={strata['inner_k1']/strata['outer']:.3e}")
    print("  compute estimate: 36 angles x 2 media = 72 forward sims (same as prior runs) "
          "-- ~15-20 minutes based on that precedent")

    medium_water = build_medium_water_only()
    medium_phantom = build_medium_two_tissue(canvas_lv, canvas_myo)

    print(f"\n=== Simulating water-only control, pitch-catch at {N_ANGLES} angles ===")
    water_traces = [simulate_pitch_catch_raw(medium_water, th) for th in thetas]
    print(f"=== Simulating {PATIENT_ID} two-tissue phantom, pitch-catch at {N_ANGLES} angles ===")
    phantom_traces = [simulate_pitch_catch_raw(medium_phantom, th) for th in thetas]

    water_mf = [matched_filter_output(tr) for tr in water_traces]
    phantom_mf = [matched_filter_output(tr) for tr in phantom_traces]
    _nonneg = _lag_t_arr >= 0

    print("\n=== Detecting outer and inner echoes per angle (real, angle-varying radii) ===")
    outer_matches, inner_matches = [], []  # each: (theta, amp, err_or_None)
    for i, theta in enumerate(thetas):
        r_out = r_at_theta(theta, ext_theta_out, ext_r_out)
        r_in = r_at_theta(theta, ext_theta_in, ext_r_in)
        t_outer, t_inner = predicted_reflection_times(theta, r_out, r_in)

        env_w, lag_t = water_mf[i]
        env_p, _ = phantom_mf[i]
        thresh = max(env_w[_nonneg].max() * 3.0, env_p[_nonneg].max() * PEAK_PROMINENCE_FRACTION)
        peak_idx, _ = find_peaks(env_p[_nonneg], height=thresh)
        peak_times = lag_t[_nonneg][peak_idx]
        peak_amps = env_p[_nonneg][peak_idx]

        if len(peak_times) == 0:
            continue
        d_outer = np.abs(peak_times - t_outer)
        if d_outer.min() < MATCH_WINDOW_S:
            idx = np.argmin(d_outer)
            r_est = PROBE_RADIUS_CELLS - (cfg.WATER.sound_speed * peak_times[idx] / 2) / (cfg.DX_M)
            outer_matches.append((theta, peak_amps[idx], r_est - r_out))
        d_inner = np.abs(peak_times - t_inner)
        if d_inner.min() < MATCH_WINDOW_S:
            idx = np.argmin(d_inner)
            inner_matches.append((theta, peak_amps[idx], None))

    outer_amps = [m[1] for m in outer_matches]
    inner_amps = [m[1] for m in inner_matches]
    outer_errs = [m[2] for m in outer_matches]

    print(f"\n--- Detection results ({PATIENT_ID}, real anatomy) ---")
    print(f"  outer boundary: detected {len(outer_matches)}/{N_ANGLES} angles, "
          f"mean radius error={np.mean(outer_errs):+.2f} cells" if outer_errs else "  outer: none detected")
    print(f"  inner boundary: detected {len(inner_matches)}/{N_ANGLES} angles")
    if outer_amps and inner_amps:
        ratio = np.mean(inner_amps) / np.mean(outer_amps)
        print(f"  mean outer amplitude={np.mean(outer_amps):.4g}, mean inner amplitude={np.mean(inner_amps):.4g}, "
              f"ratio={ratio:.3f} (coefficient prediction: {strata['inner_k0']/strata['outer']:.3f}; "
              f"circular phantom run -08 measured: 0.235)")

    print(f"\n=== Applying the amplitude-strata veto (runs -21/-22 methodology) to patient023's matches ===")
    outer_baseline_amp = np.median(outer_amps) if outer_amps else None
    if outer_baseline_amp is not None:
        predicted_inner_amp = outer_baseline_amp * (strata["inner_k0"] / strata["outer"])
        veto_thresh = predicted_inner_amp * VETO_MARGIN
        print(f"  calibrated outer baseline (median observed): {outer_baseline_amp:.4g}")
        print(f"  predicted inner_k0 amplitude (coefficient-scaled): {predicted_inner_amp:.4g}, "
              f"veto threshold ({VETO_MARGIN}x): {veto_thresh:.4g}")
        survive, veto = [], []
        for theta, amp, _ in inner_matches:
            (survive if amp <= veto_thresh else veto).append((theta, amp))
        print(f"  inner matches: {len(survive)}/{len(inner_matches)} SURVIVE the strata veto, "
              f"{len(veto)}/{len(inner_matches)} VETOED (amplitude too strong for a genuine order-1 inner echo)")
        if veto:
            print(f"  vetoed angles/amplitudes: {[(f'{t:.0f}deg', f'{a:.4g}') for t, a in veto]}")
        if survive:
            print(f"  surviving angles/amplitudes: {[(f'{t:.0f}deg', f'{a:.4g}') for t, a in survive]}")
    else:
        print("  no outer matches -- cannot calibrate the strata veto")
        survive, veto = [], list(zip([m[0] for m in inner_matches], inner_amps))

    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    axes[0].bar(["outer", "inner (raw)"], [len(outer_matches), len(inner_matches)], color=["C0", "C1"])
    axes[0].set_ylabel(f"angles detected (of {N_ANGLES})")
    axes[0].set_title(f"{PATIENT_ID}: outer vs. inner boundary detection (raw)")

    axes[1].bar(["inner: survives veto", "inner: VETOED"], [len(survive), len(veto)], color=["C2", "red"])
    axes[1].set_ylabel("number of inner matches")
    axes[1].set_title(f"{PATIENT_ID}: inner matches after amplitude-strata veto")

    fig.suptitle(f"{PATIENT_ID} real anatomy: reflection detection + amplitude-strata veto")
    labels.add_banner(fig)
    plt.tight_layout()
    os.makedirs("results/figures", exist_ok=True)
    out_fig = f"results/figures/phase1_patient023_validation.png"
    plt.savefig(out_fig, dpi=140)
    print(f"\nSaved {out_fig}")
