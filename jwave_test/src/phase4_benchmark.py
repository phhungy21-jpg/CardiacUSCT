"""Phase 4.1 — benchmark-then-multiply (protocol 4.1).

STATUS: PENDING ACOUSTIC-PHYSICS SIGNOFF (see labels.py). This is Phase
4.1 (Prepare) ONLY -- timing/scope estimation. Per the Phase 3->4 hard
gate (../PHASE2_TO_PHASE3_DIAGNOSTIC_HANDOFF.md), Phase 4.2 (generating
the actual dataset used for Phase 5 conclusions) is BLOCKED until
attenuation, source-scaling, and staircasing get collaborator signoff.
Nothing here is a real dataset -- it is a compute-budget estimate to bring
to the collaborators, per protocol 4.1's explicit instruction to do this
BEFORE committing to a full run.

Real cardiac anatomy (patient001, Phase I ACDC registration output) is
used for the FIRST time in this project here, per protocol Phase 4.1
("reuse Phase I anatomy"). Mask resampled with NEAREST-NEIGHBOR only, per
CLAUDE.md's explicit rule (never linear/cubic -- silently corrupts labels).

Compute reality check: patient001's real LV+myocardium bounding box is
~83x80mm. At the Phase-2-specified dx=0.1mm, that's >800x800 grid points --
this local CPU machine OOM'd at 700x700 (Run 2026-07-07-06). Full-size,
full-resolution real-anatomy simulation is NOT attempted here; instead,
several smaller memory-safe crops of the SAME real anatomy are timed, and
the scaling relationship is used to extrapolate a full-size estimate. This
extrapolation is a stand-in for an actual GPU benchmark (Gate 1's
GPU-timed reproduction is also still outstanding, see
../jwave/notebooks/phase1_gate1_reference_repro.ipynb) -- bring both to
the collaborators before trusting this number.
"""

import time

import numpy as np
from jax import numpy as jnp
from jax import jit
from scipy.ndimage import zoom

from jwave import FourierSeries
from jwave.geometry import Domain, Medium, TimeAxis, Sources
from jwave.acoustics import simulate_wave_propagation

import phase2_config as cfg
import labels

PATIENT_NPZ = "../pilot/data/processed/ACDC_reg/patient001.npz"
FRAME_IDX = 2  # non-empty frame with substantial foreground (see LOG.md)


def load_real_mask_upsampled():
    d = np.load(PATIENT_NPZ)
    mask = d["warped_ed_mask"][FRAME_IDX]
    spacing_mm = d["spacing"][:2]  # in-plane spacing, (row, col)
    zoom_factors = spacing_mm / (cfg.DX_M * 1e3)  # native px -> 0.1mm px
    # NEAREST-NEIGHBOR only (order=0), per CLAUDE.md -- linear/cubic would
    # corrupt integer label values (e.g. blend label 2 and 3 into 2.5).
    upsampled = zoom(mask, zoom_factors, order=0)
    return upsampled.astype(int)


def crop_center(arr, size):
    cy, cx = arr.shape[0] // 2, arr.shape[1] // 2
    h = size // 2
    return arr[cy - h:cy + h, cx - h:cx + h]


def build_medium_from_labels(label_map, domain):
    sound_speed_map = np.zeros(label_map.shape, dtype=np.float32)
    density_map = np.zeros(label_map.shape, dtype=np.float32)
    for label, tissue in cfg.ACDC_LABEL_TO_TISSUE.items():
        m = label_map == label
        sound_speed_map[m] = tissue.sound_speed
        density_map[m] = tissue.density
    sound_speed_map = jnp.expand_dims(jnp.array(sound_speed_map), -1)
    density_map = jnp.expand_dims(jnp.array(density_map), -1)
    return Medium(domain=domain,
                  sound_speed=FourierSeries(sound_speed_map, domain),
                  density=FourierSeries(density_map, domain))


def toneburst(t):
    duration = cfg.N_CYCLES / cfg.F0_HZ
    sigma = duration / 6
    window = np.exp(-(t - duration / 2) ** 2 / (2 * sigma ** 2))
    return np.sin(2 * np.pi * cfg.F0_HZ * t) * window


def benchmark_one_size(N):
    dx = (cfg.DX_M, cfg.DX_M)
    domain = Domain((N, N), dx)
    full_map = load_real_mask_upsampled()
    label_map = crop_center(full_map, N)
    if label_map.shape != (N, N):
        raise ValueError(f"real anatomy crop too small for N={N} "
                          f"(got {label_map.shape})")

    medium = build_medium_from_labels(label_map, domain)
    time_axis = TimeAxis.from_medium(medium, cfl=cfg.CFL)  # full domain crossing
    n_steps = int(np.round(time_axis.t_end / time_axis.dt))
    t_arr = np.arange(n_steps) * time_axis.dt

    src_x, src_y = N // 2 - 5, 10
    signal = jnp.array(toneburst(t_arr))[None, :]
    sources = Sources(positions=([src_x], [src_y]), signals=signal,
                       dt=time_axis.dt, domain=domain)

    @jit
    def run(medium):
        return simulate_wave_propagation(medium, time_axis, sources=sources)

    t0 = time.time()
    pressure = run(medium)
    pressure.on_grid.block_until_ready()
    elapsed = time.time() - t0

    bytes_estimate = n_steps * N * N * 4
    return {
        "N": N, "n_steps": n_steps, "elapsed_s": elapsed,
        "memory_estimate_gb": bytes_estimate / 1e9,
    }


if __name__ == "__main__":
    print(f"*** {labels.PENDING_SIGNOFF_BANNER} ***")
    print("Phase 4.1 benchmark-then-multiply (PREPARE only, not 4.2 dataset "
          "generation). Real anatomy: patient001, frame", FRAME_IDX)

    results = []
    for N in [150, 250, 350]:
        r = benchmark_one_size(N)
        results.append(r)
        print(f"N={r['N']:4d}: {r['n_steps']:5d} steps, "
              f"{r['elapsed_s']:.2f}s, "
              f"~{r['memory_estimate_gb']:.3f}GB time-history array")

    # Empirical scaling fit: elapsed ~ a * N^p (log-log linear fit)
    Ns = np.array([r["N"] for r in results], dtype=float)
    ts = np.array([r["elapsed_s"] for r in results])
    p, log_a = np.polyfit(np.log(Ns), np.log(ts), 1)
    a = np.exp(log_a)
    print(f"\nEmpirical scaling fit: time ~ {a:.3e} * N^{p:.2f}")

    real_anatomy_N_estimate = 900  # ~90mm bbox + margin at dx=0.1mm
    extrapolated_s = a * real_anatomy_N_estimate ** p
    extrapolated_mem_gb = (
        (extrapolated_s / results[-1]["elapsed_s"]) ** 0  # placeholder, see note
    )
    # Memory scales differently (N^2 * n_steps, n_steps ~ N), i.e. N^3, not
    # the same power as wall-clock (FFT cost differs) -- reported
    # separately via direct N^3 scaling from the largest benchmarked point.
    mem_scale = (real_anatomy_N_estimate / results[-1]["N"]) ** 3
    extrapolated_mem_gb = results[-1]["memory_estimate_gb"] * mem_scale

    print(f"\nExtrapolated single-transmit estimate at N~{real_anatomy_N_estimate} "
          f"(full real-anatomy FOV, ~90mm bbox+margin at dx=0.1mm):")
    print(f"  time ~ {extrapolated_s:.1f}s, memory ~ {extrapolated_mem_gb:.1f}GB")
    print("  (This CPU extrapolation, and Gate 1's still-outstanding GPU-timed "
          "reproduction, both need validating on the lab's actual GPU before "
          "this number is trusted for real budgeting -- see LOG.md.)")

    print("\n=== Compute-budget FORMULA (fill in with collaborators, not "
          "invented here) ===")
    print("  total_time = per_transmit_time * transmits_per_frame * "
          "frames_per_patient(10, from ACDC_reg data) * n_patients(<=150 "
          "available) * n_conditions(Phase 5 sweep, not yet finalized)")
    print("  Bring this formula + the extrapolated per-transmit number to "
          "the collaborators per Gate 4's first checkbox, before committing "
          "to any Phase 4.2 run.")
