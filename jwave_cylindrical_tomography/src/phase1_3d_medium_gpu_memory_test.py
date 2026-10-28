"""Colab GPU version: builds patient023's real, cropped, resampled 3D
heterogeneous medium (N=150, dx=0.867mm, f0=0.8MHz -- this project's
design point from run -05) and runs ONE real 3D forward simulation with
it, to confirm this HETEROGENEOUS medium fits GPU memory the same way
the homogeneous-water timing test (run -03) did -- not yet verified,
since a 3-tissue FourierSeries medium might have a different memory
footprint than a uniform one.

Expects `patient023_ed_only.npz` (just `ed_mask` + `spacing`, ~148KB) in
the same directory -- embedded as base64 in the Colab notebook cell
that wraps this script, not uploaded separately.
"""

import time
import numpy as np
from scipy import ndimage

from jax import numpy as jnp, jit
from jwave import FourierSeries
from jwave.geometry import Domain, Medium, TimeAxis, Sources
from jwave.acoustics import simulate_wave_propagation

# inlined tissue constants (Mast 2000, verified this session -- see jwave_ring_tomography/src/phase2_config.py)
WATER_SOUND_SPEED, WATER_DENSITY = 1520.0, 994.0
BLOOD_SOUND_SPEED, BLOOD_DENSITY = 1584.0, 1060.0
MYOCARDIUM_SOUND_SPEED, MYOCARDIUM_DENSITY = 1576.0, 1060.0

N_GRID = 150
DOMAIN_MM = 130.0
DX_MM = DOMAIN_MM / N_GRID
DX_M = DX_MM * 1e-3
F0_HZ = 0.8e6
CFL = 0.3
MARGIN_MM = 20.0


def load_and_crop(npz_path, margin_mm=MARGIN_MM):
    d = np.load(npz_path, allow_pickle=True)
    mask = d["ed_mask"]
    native_spacing_mm = np.array(d["spacing"])[::-1]
    tissue = mask > 0
    coords = np.argwhere(tissue)
    mins, maxs = coords.min(axis=0), coords.max(axis=0)
    margin_vox = np.ceil(margin_mm / native_spacing_mm).astype(int)
    mins = np.maximum(mins - margin_vox, 0)
    maxs = np.minimum(maxs + margin_vox, np.array(mask.shape) - 1)
    cropped = mask[mins[0]:maxs[0]+1, mins[1]:maxs[1]+1, mins[2]:maxs[2]+1]
    return cropped, native_spacing_mm


def resample_to_target(cropped_mask, native_spacing_mm, target_n=N_GRID):
    zoom_factors = native_spacing_mm / DX_MM
    resampled = ndimage.zoom(cropped_mask, zoom_factors, order=0, mode="nearest")
    out = np.zeros((target_n, target_n, target_n), dtype=resampled.dtype)
    src_shape, tgt_shape = np.array(resampled.shape), np.array(out.shape)
    start_src = np.maximum((src_shape - tgt_shape) // 2, 0)
    start_tgt = np.maximum((tgt_shape - src_shape) // 2, 0)
    copy_shape = np.minimum(src_shape - start_src, tgt_shape - start_tgt)
    out[start_tgt[0]:start_tgt[0]+copy_shape[0], start_tgt[1]:start_tgt[1]+copy_shape[1], start_tgt[2]:start_tgt[2]+copy_shape[2]] = \
        resampled[start_src[0]:start_src[0]+copy_shape[0], start_src[1]:start_src[1]+copy_shape[1], start_src[2]:start_src[2]+copy_shape[2]]
    return out


if __name__ == "__main__":
    import jax
    print("jax devices:", jax.devices())
    print(f"BUILD 3D MEDIUM + GPU MEMORY TEST: N={N_GRID}, domain={DOMAIN_MM}mm, dx={DX_MM:.3f}mm, f0={F0_HZ/1e6}MHz")

    cropped, native_spacing = load_and_crop("patient023_ed_only.npz")
    resampled = resample_to_target(cropped, native_spacing)
    print(f"  resampled shape: {resampled.shape}")
    for label, name in [(0, "water"), (1, "RV/blood"), (2, "MYO"), (3, "LV/blood")]:
        print(f"    {name}: {(resampled == label).mean()*100:.1f}%")

    sound_speed = np.where(resampled == 2, MYOCARDIUM_SOUND_SPEED,
                   np.where((resampled == 1) | (resampled == 3), BLOOD_SOUND_SPEED, WATER_SOUND_SPEED)).astype(np.float32)
    density = np.where(resampled == 2, MYOCARDIUM_DENSITY,
               np.where((resampled == 1) | (resampled == 3), BLOOD_DENSITY, WATER_DENSITY)).astype(np.float32)

    domain = Domain((N_GRID, N_GRID, N_GRID), (DX_M, DX_M, DX_M))
    ssm = jnp.expand_dims(jnp.array(sound_speed), -1)
    dm = jnp.expand_dims(jnp.array(density), -1)
    medium = Medium(domain=domain, sound_speed=FourierSeries(ssm, domain), density=FourierSeries(dm, domain))
    print("  heterogeneous Medium constructed successfully.")

    time_axis = TimeAxis.from_medium(medium, cfl=CFL)
    dt = time_axis.dt
    c_min = WATER_SOUND_SPEED
    traversal_s = (DOMAIN_MM * 1e-3 / c_min) * 2 * 1.3
    time_axis = TimeAxis(dt=dt, t_end=traversal_s)
    n_steps = int(time_axis.Nt)
    print(f"  dt={dt:.3e}s, n_steps for full round trip={n_steps}")

    center = N_GRID // 2
    src_pos = (center - N_GRID // 4, center, center)
    t = jnp.arange(n_steps) * dt
    n_cycles = 3
    signal = jnp.sin(2 * jnp.pi * F0_HZ * t) * jnp.exp(-((t - n_cycles/F0_HZ) ** 2) / (2 * (n_cycles/3/F0_HZ) ** 2))
    sources = Sources(positions=([src_pos[0]], [src_pos[1]], [src_pos[2]]),
                       signals=jnp.expand_dims(signal, 0), dt=dt, domain=domain)

    # CRITICAL FIX (caught after the first attempt's OOM): jWave's default (no `sensors`
    # given) returns/stores the FULL pressure field at EVERY timestep -- memory scales as
    # N^3 * n_steps, not just N^3. At N=150 x 1355 steps that's 18.3GB for ONE field alone,
    # exactly matching the earlier OOM error. Fix: supply a `sensors` callback that extracts
    # only the specific receiver point(s) actually needed (the standard, correct way large-
    # scale ultrasound simulation is done -- k-Wave's own "sensor mask" convention) -- this
    # project's 2D work never needed this because N was small enough for full-field
    # recording to fit comfortably; verified the API mechanics work correctly at small scale
    # (locally, N=40) before spending another GPU round-trip on this fix.
    rx_positions = [
        (center + N_GRID // 4, center, center),           # diametrically opposite the source
        (center, center + N_GRID // 4, center),
        (center, center - N_GRID // 4, center),
    ]
    rx_rows = jnp.array([p[0] for p in rx_positions])
    rx_cols = jnp.array([p[1] for p in rx_positions])
    rx_deps = jnp.array([p[2] for p in rx_positions])

    def sensors(p, u, rho):
        return p.on_grid[rx_rows, rx_cols, rx_deps, 0]

    @jit
    def run(m):
        return simulate_wave_propagation(m, time_axis, sources=sources, sensors=sensors)

    print(f"\n  running full {n_steps}-step simulation on the REAL heterogeneous medium "
          f"(sensor-based recording, {len(rx_positions)} receiver points, NOT full-field)...")
    t0 = time.time()
    traces = run(medium)
    traces.block_until_ready()
    elapsed = time.time() - t0
    print(f"  SUCCESS: {elapsed:.2f}s for the full simulation (N={N_GRID}, heterogeneous medium, GPU)")
    print(f"  traces shape: {traces.shape} (expect ({n_steps}, {len(rx_positions)}))")
    print(f"  peak trace magnitude: {float(jnp.max(jnp.abs(traces))):.4e}")
