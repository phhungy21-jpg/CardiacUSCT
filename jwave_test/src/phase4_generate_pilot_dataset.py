"""Phase 4.2 — PILOT dataset generation (resumable driver).

STATUS: PENDING FULL-SCALE SIGNOFF. Gate 2 (acoustic-physics correctness)
PASSED 2026-07-07 (real collaborator review, see LOG.md run -12,
PROXY_AUDIT.md). This unblocks the PHYSICS. It does NOT by itself unblock
the full 150-patient/full-resolution Phase 4.2 run: Gate 4's separate
checklist item ("compute estimate agreed with collaborators before the
full run") is still open -- only a preliminary CPU/small-GPU estimate
exists (LOG.md runs -09/-10), and real full-resolution simulation still
OOMs on this local machine (run -09, 700x700 -> 3.9GB; extrapolated
full-heart N~900 -> ~13.8GB).

This script generates a small, locally-feasible PILOT dataset instead: a
few real patients, their real (non-empty) cardiac-cycle frames, at
N=350 (35mm FOV, the one size confirmed to contain a genuine tissue
boundary -- see PROXY_AUDIT.md item 6), using the validated attenuating
solver and calibrated amplitude. Purpose: build and prove out the
RESUMABLE dataset-generation driver itself (protocol 4.2's explicit
requirement -- "Long GPU runs will be interrupted -- make the driver skip
already-completed cases"), and produce a small real dataset usable for
further pipeline development, NOT a dataset whose scale has been
budget-agreed with collaborators for Phase 5's actual characterization
study.

Per-patient/per-frame heart centroid is computed fresh each time (not a
fixed array-center assumption) since patients' hearts sit at slightly
different positions within the 128x128 ACDC grid (checked across 6
patients: centroid row 60-69, col 66-76 out of 128 -- see chat log).
"""

import os
import time

import numpy as np
from jax import numpy as jnp
from jax import jit

from jwave import FourierSeries
from jwave.geometry import Domain, Medium, TimeAxis, Sources

import phase2_config as cfg
import calibration
import labels
from attenuation_solver import (simulate_wave_propagation_attenuating,
                                attenuation_field_np_per_m)

ACDC_DIR = "../pilot/data/processed/ACDC_reg"
OUT_DIR = "results/phase4_pilot_dataset"

PILOT_PATIENTS = ["patient001.npz", "patient002.npz", "patient003.npz"]
N = 350
DX = (cfg.DX_M, cfg.DX_M)


def heart_centroid(mask_frame_upsampled):
    """Centroid of labels {2,3}, directly in the frame's own coordinate
    space -- caller must pass an already-upsampled frame (no re-scaling
    here). BUG FIX (see chat log): an earlier version multiplied this by
    zoom_factors again after being called on an already-upsampled frame,
    double-scaling the center by ~15.6x and putting every crop off-canvas
    (e.g. center (15117, 17553) on a 2000x2000 canvas) -- caught because
    the pilot run produced 0 cases despite no exceptions being raised."""
    ys, xs = np.where(np.isin(mask_frame_upsampled, [2, 3]))
    if len(ys) == 0:
        return None
    return int(ys.mean()), int(xs.mean())


def load_upsampled_mask(npz_path):
    from scipy.ndimage import zoom
    d = np.load(npz_path)
    spacing_mm = d["spacing"][:2]
    zoom_factors = spacing_mm / (cfg.DX_M * 1e3)
    mask = d["warped_ed_mask"]
    upsampled = np.stack(
        [zoom(mask[i], zoom_factors, order=0) for i in range(mask.shape[0])])
    return upsampled.astype(int), zoom_factors


def crop_around(arr, center, size):
    cy, cx = center
    h = size // 2
    y0, y1 = cy - h, cy - h + size
    x0, x1 = cx - h, cx - h + size
    if y0 < 0 or x0 < 0 or y1 > arr.shape[0] or x1 > arr.shape[1]:
        return None  # crop would run off the upsampled canvas edge
    return arr[y0:y1, x0:x1]


def build_medium_and_atten(label_map, domain):
    sound_speed_map = np.zeros((N, N), dtype=np.float32)
    density_map = np.zeros((N, N), dtype=np.float32)
    atten_map = np.zeros((N, N), dtype=np.float32)
    for label, tissue in cfg.ACDC_LABEL_TO_TISSUE.items():
        m = label_map == label
        sound_speed_map[m] = tissue.sound_speed
        density_map[m] = tissue.density
        atten_map[m] = attenuation_field_np_per_m(tissue.attenuation, cfg.F0_HZ, y=1.0)
    medium = Medium(
        domain=domain,
        sound_speed=FourierSeries(jnp.expand_dims(jnp.array(sound_speed_map), -1), domain),
        density=FourierSeries(jnp.expand_dims(jnp.array(density_map), -1), domain),
    )
    atten_field = FourierSeries(jnp.expand_dims(jnp.array(atten_map), -1), domain)
    return medium, atten_field


def toneburst(t, pa_per_unit):
    duration = cfg.N_CYCLES / cfg.F0_HZ
    sigma = duration / 6
    window = np.exp(-(t - duration / 2) ** 2 / (2 * sigma ** 2))
    return pa_per_unit * np.sin(2 * np.pi * cfg.F0_HZ * t) * window


def run_one_case(label_map):
    domain = Domain((N, N), DX)
    medium, atten_field = build_medium_and_atten(label_map, domain)
    time_axis = TimeAxis.from_medium(medium, cfl=cfg.CFL)
    t_end = 0.5 * time_axis.t_end
    time_axis = TimeAxis(dt=time_axis.dt, t_end=t_end)
    n_steps = int(np.round(t_end / time_axis.dt))
    t_arr = np.arange(n_steps) * time_axis.dt

    pa_per_unit = calibration.calibrate_arbitrary_units(cfg.F0_HZ)
    src_x, src_y = N // 2 - 5, 10
    signal = jnp.array(toneburst(t_arr, pa_per_unit))[None, :]
    sources = Sources(positions=([src_x], [src_y]), signals=signal,
                       dt=time_axis.dt, domain=domain)

    @jit
    def run(medium):
        return simulate_wave_propagation_attenuating(
            medium, time_axis, atten_field, cfg.F0_HZ, sources=sources)

    pressure = run(medium)
    field = pressure.on_grid[..., 0]
    n_nan = int(jnp.sum(jnp.isnan(field)))
    max_p = float(jnp.max(jnp.abs(field)))
    # Save receiver trace (small) + a coarse downsample of the full field
    # (every 4th pixel) for spot-checking, not the full field, to keep
    # pilot-dataset disk usage bounded.
    rcv_x, rcv_y = N // 2 + 5, 10
    receiver_trace = np.array(field[:, rcv_y, rcv_x])
    field_thumbnail = np.array(field[::20, ::4, ::4])  # sparse spacetime thumbnail
    return {
        "n_nan": n_nan, "max_pressure_pa": max_p,
        "receiver_trace": receiver_trace, "field_thumbnail": field_thumbnail,
        "n_steps": n_steps, "dt": time_axis.dt,
    }


if __name__ == "__main__":
    print(f"*** {labels.PENDING_SIGNOFF_BANNER} ***")
    print("Phase 4.2 PILOT dataset (small, locally-feasible subset -- NOT "
          "the full 150-patient/full-resolution run, see module docstring)")
    os.makedirs(OUT_DIR, exist_ok=True)

    n_run, n_skipped, n_failed = 0, 0, 0
    t_start = time.time()
    for patient_file in PILOT_PATIENTS:
        npz_path = os.path.join(ACDC_DIR, patient_file)
        if not os.path.exists(npz_path):
            print(f"  [skip] {patient_file} not found")
            continue
        patient_id = patient_file.replace(".npz", "")
        upsampled_mask, zoom_factors = load_upsampled_mask(npz_path)

        for frame_idx in range(upsampled_mask.shape[0]):
            out_path = os.path.join(OUT_DIR, f"{patient_id}_frame{frame_idx:02d}.npz")
            if os.path.exists(out_path):
                n_skipped += 1
                continue  # RESUMABLE: skip already-completed cases

            frame = upsampled_mask[frame_idx]
            if frame.max() == 0:
                continue  # empty frame (e.g. frame 0 in some patients), not a real case

            center = heart_centroid(frame)
            label_map = crop_around(upsampled_mask[frame_idx], center, N)
            if label_map is None:
                print(f"  [skip] {patient_id} frame {frame_idx}: crop runs off canvas edge")
                continue

            t0 = time.time()
            try:
                result = run_one_case(label_map)
            except Exception as e:
                print(f"  [FAILED] {patient_id} frame {frame_idx}: {e}")
                n_failed += 1
                continue
            elapsed = time.time() - t0

            np.savez_compressed(
                out_path,
                receiver_trace=result["receiver_trace"],
                field_thumbnail=result["field_thumbnail"],
                n_nan=result["n_nan"], max_pressure_pa=result["max_pressure_pa"],
                n_steps=result["n_steps"], dt=result["dt"],
                label_map=label_map.astype(np.int8),
            )
            n_run += 1
            status = "OK" if result["n_nan"] == 0 else "NAN DETECTED"
            print(f"  [{status}] {patient_id} frame {frame_idx}: "
                  f"{elapsed:.1f}s, max|p|={result['max_pressure_pa']:.0f}Pa "
                  f"-> {out_path}")

    total_elapsed = time.time() - t_start
    print(f"\nDone. {n_run} cases run, {n_skipped} skipped (already done), "
          f"{n_failed} failed. Total time: {total_elapsed:.1f}s")
    print(f"Pilot dataset saved to {OUT_DIR}/ "
          f"({len(PILOT_PATIENTS)} patients -- NOT the full 150-patient cohort)")
