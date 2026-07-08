"""Phase 3 — measuring the REAL (not interpolated) 45-degree and
135-degree baseline amplitude ratios for the 8-probe layout, closing
the gap flagged in runs -56/-58/-59 before the 8-probe geometry itself
can be considered for the official pipeline.

Per user: "so do both then, 8 probe on a selector?" -- combining 8
probes with the local-max selector (run -57) gave a near-complete fix
(0.04mm), but that result relied on an UNMEASURED interpolation
assumption for the new 45/135-degree baseline pairs the 8-probe layout
introduces (only 0/90/180-degree baselines were ever measured, runs
-44/-53 -- a 4-probe layout has no 45-degree-separated pairs at all,
so this measurement was never possible before now).

METHOD: identical to run -44/-53's isolated single-boundary calibration
(myocardium disk alone, no competing boundary, standard geometry) --
just generalized to the 8-probe layout so 45/135-degree-separated pairs
actually exist, and generalized `predicted_time` (run -44's nearest-
point specular search) to work for this domain's own probe positions/
center instead of the 4-probe module's hardcoded (150,150).
"""

import numpy as np

from jax import numpy as jnp
from jwave import FourierSeries
from jwave.geometry import Medium

import phase2_config as cfg
import labels
from phase3_mri_8probe_test import (
    _SRC, _RCV, PROBE_NAMES, _ANGLE_OF, angular_separation, capture_all_pairs,
    build_medium_homogeneous, direction_vector, domain, N, center, dx, c_ref,
    t_arr, _ENVELOPE_GROUP_DELAY_S,
)

RADII = [41.0, 71.0, 88.0]  # same 3 radii already calibrated for 0/90/180


def build_medium_myocardium_disk(R):
    """Isolated myocardium disk (no competing boundary) -- identical
    construction to run -44's `build_medium_myocardium_disk`, just
    built in the 8-probe module's own domain/center."""
    yy, xx = np.mgrid[0:N[0], 0:N[1]]
    dist = np.sqrt((xx - center[1]) ** 2 + (yy - center[0]) ** 2)
    inside = dist < R
    sound_speed_map = np.where(inside, cfg.MYOCARDIUM.sound_speed, cfg.CHEST_WALL_PROXY.sound_speed).astype(np.float32)
    density_map = np.where(inside, cfg.MYOCARDIUM.density, cfg.CHEST_WALL_PROXY.density).astype(np.float32)
    ssm = jnp.expand_dims(jnp.array(sound_speed_map), -1)
    dm = jnp.expand_dims(jnp.array(density_map), -1)
    return Medium(domain=domain, sound_speed=FourierSeries(ssm, domain),
                  density=FourierSeries(dm, domain))


def predicted_time(tx_name, rx_name, R):
    """Nearest-point specular round-trip time -- generalized from run
    -44's version to use THIS module's own center, not a hardcoded
    (150,150)."""
    src, rcv = _SRC[tx_name], _RCV[rx_name]
    thetas = np.linspace(0, 2 * np.pi, 720, endpoint=False)
    best_t, best_dist_sum = None, np.inf
    for th in thetas:
        row = center[0] + R * np.cos(th)
        col = center[1] + R * np.sin(th)
        d_tx = np.hypot(col - src[0], row - src[1]) * dx[0]
        d_rx = np.hypot(col - rcv[0], row - rcv[1]) * dx[0]
        if d_tx + d_rx < best_dist_sum:
            best_dist_sum = d_tx + d_rx
            best_t = (d_tx + d_rx) / c_ref + _ENVELOPE_GROUP_DELAY_S
    return best_t


PAIRS_BY_SEP = {0: [], 45: [], 90: [], 135: [], 180: []}
for tx in PROBE_NAMES:
    for rx in PROBE_NAMES:
        sep = angular_separation(_ANGLE_OF[tx], _ANGLE_OF[rx])
        PAIRS_BY_SEP[sep].append((tx, rx))
# sep=0 includes the monostatic (tx==rx) pairs -- needed as the ratio
# denominator; the official weight model still always assigns monostatic
# weight=1.0 by construction (run -44 onward), this measurement is just
# for the ratio itself.


if __name__ == "__main__":
    print(f"*** {labels.PENDING_SIGNOFF_BANNER} ***")
    print(f"*** {labels.TOY_EXACT_GT_CAPTION} ***")
    print("Measuring REAL 45-degree and 135-degree baseline amplitude ratios "
          "(8-probe layout only supports these baselines; a 4-probe layout "
          "has none) -- closing the interpolation-assumption gap from runs "
          "-56/-58/-59 before considering the 8-probe geometry for the "
          "official pipeline. Same isolated single-boundary method as run "
          "-44/-53.")
    for sep, pairs in PAIRS_BY_SEP.items():
        print(f"  {sep} degrees: {len(pairs)} pairs")

    pairs_ref = capture_all_pairs(build_medium_homogeneous())

    results = {}
    for R in RADII:
        print(f"\n=== Capturing isolated myocardium disk: R={R} ===")
        pairs_disk = capture_all_pairs(build_medium_myocardium_disk(R))
        amp_by_sep = {}
        for sep, pair_list in PAIRS_BY_SEP.items():
            amps = []
            for tx, rx in pair_list:
                env_clean = pairs_disk[(tx, rx)] - pairs_ref[(tx, rx)]
                t_pred = predicted_time(tx, rx, R)
                amp = abs(np.interp(t_pred, t_arr, env_clean))
                amps.append(amp)
            amp_by_sep[sep] = np.mean(amps)
        results[R] = amp_by_sep

    print(f"\n{'R':<6}{'0deg (mono)':>14}{'45deg':>14}{'90deg':>14}{'135deg':>14}{'180deg':>14}")
    for R in RADII:
        r = results[R]
        print(f"{R:<6}{r[0]:>14.6f}{r[45]:>14.6f}{r[90]:>14.6f}{r[135]:>14.6f}{r[180]:>14.6f}")

    print(f"\n{'R':<6}{'45/mono ratio':>16}{'90/mono ratio':>16}{'135/mono ratio':>16}{'180/mono ratio':>16}")
    measured_45, measured_90, measured_135, measured_180 = [], [], [], []
    for R in RADII:
        r = results[R]
        ratio_45 = r[45] / (r[0] + 1e-12)
        ratio_90 = r[90] / (r[0] + 1e-12)
        ratio_135 = r[135] / (r[0] + 1e-12)
        ratio_180 = r[180] / (r[0] + 1e-12)
        measured_45.append(ratio_45)
        measured_90.append(ratio_90)
        measured_135.append(ratio_135)
        measured_180.append(ratio_180)
        print(f"{R:<6}{ratio_45:>16.4f}{ratio_90:>16.4f}{ratio_135:>16.4f}{ratio_180:>16.4f}")

    print(f"\nFor comparison, run -44/-53's already-measured 90/180-degree values:")
    print(f"  R=41: cross(90)=0.136, antipodal(180)=0.045")
    print(f"  R=71: cross(90)=0.000, antipodal(180)=0.000")
    print(f"  R=88: cross(90)=0.0001, antipodal(180)=0.0003")
    print(f"\nCurrent INTERPOLATION assumption used in phase3_mri_8probe_test.py "
          f"(midpoint between monostatic=1.0 and cross(90) for 45deg, "
          f"between cross(90) and antipodal(180) for 135deg):")
    for R, m45, m135 in zip(RADII, measured_45, measured_135):
        cross_r = {41.0: 0.136, 71.0: 0.000, 88.0: 0.0001}[R]
        antipodal_r = {41.0: 0.045, 71.0: 0.000, 88.0: 0.0003}[R]
        interp_45 = 1.0 + 0.5 * (cross_r - 1.0)
        interp_135 = cross_r + 0.5 * (antipodal_r - cross_r)
        print(f"  R={R}: measured 45deg={m45:.4f} vs assumed=0.5000 (interp={interp_45:.4f} incl. cross_r) "
              f"| measured 135deg={m135:.4f} vs interp={interp_135:.4f}")
