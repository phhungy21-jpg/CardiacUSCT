"""Loads a single cardiac-cycle PHASE from jwave_test's already-prepped
8-phase motion-cycle datasets (`mri_motion_cycle_patient001_slice4.npz`,
`mri_motion_cycle_patient023_slice4.npz`) and places it onto this
project's domain, using the SAME centroid-offset convention as
`phase1_patient023_validation.load_real_contours`. Confirmed directly
(not assumed): the motion-cycle masks are at exactly 0.1mm/pixel, IDENTICAL
to this project's grid spacing (`cfg.DX_M`), so placement is a pure
translation, no rescaling needed.

Per-frame (not single fixed) centroid alignment: uses that PHASE's own
`ring_frames[i]` centroid, in case the heart's centroid drifts slightly
frame to frame within a cardiac cycle (a small correctness improvement
over reusing one static offset for all phases).
"""

import numpy as np

from jax import numpy as jnp
from jwave import FourierSeries
from jwave.geometry import Medium

from phase1_rotating_transmission_scout import center, N, domain
import phase2_config as cfg

MOTION_CYCLE_NPZ = {
    "patient001": "../jwave_test/results/mri_motion_cycle_patient001_slice4.npz",
    "patient023": "../jwave_test/results/mri_motion_cycle_patient023_slice4.npz",
}


def load_motion_cycle(patient_id):
    return np.load(MOTION_CYCLE_NPZ[patient_id], allow_pickle=True)


def load_phase_contours(d, phase_idx):
    """Places phase `phase_idx`'s myo/lv masks onto this project's domain
    canvas. Returns canvas_lv, canvas_myo (bool arrays, shape N) and the
    outer/inner contours in DOMAIN coordinates (cells)."""
    myo_mask = d["myo_frames"][phase_idx].astype(bool)
    lv_mask = d["lv_frames"][phase_idx].astype(bool)
    ring_mask = d["ring_frames"][phase_idx].astype(bool)
    outer_contour = d["outer_contours"][phase_idx]
    inner_contour = d["inner_contours"][phase_idx]

    ys, xs = np.where(ring_mask)
    ring_centroid_native = (ys.mean(), xs.mean())
    offset_row = int(round(center[0] - ring_centroid_native[0]))
    offset_col = int(round(center[1] - ring_centroid_native[1]))

    rows_native, cols_native = np.mgrid[0:myo_mask.shape[0], 0:myo_mask.shape[1]]
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
