"""Phase 3 — self-consistent calibration measurement for the 16-probe
geometry (uniform 22.5-degree spacing), extending run -60's method
(measure ALL non-monostatic baseline categories fresh within the SAME
geometry, rather than reusing/interpolating values measured under a
different probe layout -- run -60 found that invalid even for
nominally-same categories).

16 probes at 22.5-degree spacing produce baseline separations of
exactly 0 (monostatic), 22.5, 45, 67.5, 90, 112.5, 135, 157.5, 180
degrees -- 8 non-monostatic categories, each measured here at the same
3 radii used throughout this thread (41, 71, 88 cells) via the
isolated single-boundary myocardium-disk method (run -44/-53/-60).
"""

import numpy as np

from jax import numpy as jnp
from jwave import FourierSeries
from jwave.geometry import Medium

import phase2_config as cfg
import labels
from phase3_mri_16probe_test import (
    _SRC, _RCV, PROBE_NAMES, _ANGLE_OF, angular_separation, capture_all_pairs,
    build_medium_homogeneous, domain, N, center, dx, c_ref, t_arr, _ENVELOPE_GROUP_DELAY_S,
    set_calibration,
)

RADII = [41.0, 71.0, 88.0]
SEPARATIONS = [22.5, 45.0, 67.5, 90.0, 112.5, 135.0, 157.5, 180.0]


def build_medium_myocardium_disk(R):
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


PAIRS_BY_SEP = {sep: [] for sep in [0] + SEPARATIONS}
for tx in PROBE_NAMES:
    for rx in PROBE_NAMES:
        sep = angular_separation(_ANGLE_OF[tx], _ANGLE_OF[rx])
        nearest = min(PAIRS_BY_SEP.keys(), key=lambda s: abs(s - sep))
        PAIRS_BY_SEP[nearest].append((tx, rx))


if __name__ == "__main__":
    print(f"*** {labels.PENDING_SIGNOFF_BANNER} ***")
    print(f"*** {labels.TOY_EXACT_GT_CAPTION} ***")
    print(f"Measuring self-consistent calibration for the 16-probe geometry "
          f"(8 non-monostatic separations: {SEPARATIONS}), same isolated "
          f"single-boundary method as runs -44/-53/-60, at R={RADII}.")
    for sep, pairs in PAIRS_BY_SEP.items():
        print(f"  {sep} degrees: {len(pairs)} pairs")

    pairs_ref = capture_all_pairs(build_medium_homogeneous())

    results = {}
    for R in RADII:
        print(f"\n=== Capturing isolated myocardium disk: R={R} (16 transmits) ===")
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

    print(f"\n{'R':<6}" + "".join(f"{s:>10.1f}deg" for s in [0] + SEPARATIONS))
    for R in RADII:
        r = results[R]
        print(f"{R:<6}" + "".join(f"{r[s]:>14.6f}" for s in [0] + SEPARATIONS))

    print(f"\nRatios (to monostatic, R-by-R):")
    cal_by_sep = {}
    for sep in SEPARATIONS:
        ratios = np.array([results[R][sep] / (results[R][0] + 1e-12) for R in RADII])
        cal_by_sep[sep] = ratios
        print(f"  {sep} degrees: {ratios}")

    # Save for reuse without re-simulating
    np.savez("results/mri_16probe_calibration.npz",
             radii=np.array(RADII), separations=np.array(SEPARATIONS),
             **{f"ratio_{sep}": cal_by_sep[sep] for sep in SEPARATIONS})
    print("\nSaved results/mri_16probe_calibration.npz")
