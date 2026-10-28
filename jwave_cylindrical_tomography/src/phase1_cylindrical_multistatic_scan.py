"""Phase 1: the first real multistatic 3D scan on patient023's static
ED volume. SEQUENTIAL acquisition (96 cylindrical scanner elements,
`phase1_cylindrical_scanner.py`): each element fires once, recording ALL
95 other elements simultaneously via jWave's `sensors` API (run -07's
fix -- mandatory at this scale, the default full-field recording would
blow GPU memory). Water-only baseline + the real heterogeneous phantom,
96 elements x 2 media = 192 forward simulations, ~19s each (confirmed,
run -08) -> ~51 minutes total, run on Colab GPU.

No transmission/reflection distinction yet at this stage -- this
captures the RAW multistatic dataset (every tx/rx pair's received
trace); channel-specific processing (DAS reflectivity imaging, the most
directly portable method from `jwave_ring_tomography` run -31) happens
as a separate, later post-processing step, reusing this same raw data
without re-simulating.
"""

import os
import time
import numpy as np
from scipy import ndimage

# If a mounted Google Drive path is set (Colab), save checkpoints there directly --
# survives a mid-run disconnect, unlike Colab's local (ephemeral) disk. Falls back to
# the current directory if not set (e.g. local testing).
OUTPUT_DIR = os.environ.get("SCAN_OUTPUT_DIR", ".")

from jax import numpy as jnp, jit

from jwave import FourierSeries
from jwave.geometry import Domain, Medium, TimeAxis, Sources
from jwave.acoustics import simulate_wave_propagation

# --- tissue constants (Mast 2000, verified this session) ---
WATER_SOUND_SPEED, WATER_DENSITY = 1520.0, 994.0
BLOOD_SOUND_SPEED, BLOOD_DENSITY = 1584.0, 1060.0
MYOCARDIUM_SOUND_SPEED, MYOCARDIUM_DENSITY = 1576.0, 1060.0

# --- design point (runs -05/-07/-08) ---
N_GRID = 150
DOMAIN_MM = 130.0
DX_MM = DOMAIN_MM / N_GRID
DX_M = DX_MM * 1e-3
F0_HZ = 0.8e6
CFL = 0.3
MARGIN_MM = 20.0

# --- cylindrical scanner layout (this run) ---
N_RINGS = 8
N_PER_RING = 12
CENTER_CELL = N_GRID // 2
PROBE_RADIUS_MM = 56.0
PROBE_RADIUS_CELLS = PROBE_RADIUS_MM / DX_MM
Z_SPAN_MM = 80.0
Z_SPAN_CELLS = Z_SPAN_MM / DX_MM


def element_positions():
    positions = []
    ring_thetas = np.linspace(0, 360, N_PER_RING, endpoint=False)
    ring_zs = np.linspace(CENTER_CELL - Z_SPAN_CELLS / 2, CENTER_CELL + Z_SPAN_CELLS / 2, N_RINGS)
    for z in ring_zs:
        for theta_deg in ring_thetas:
            theta = np.deg2rad(theta_deg)
            row = CENTER_CELL - PROBE_RADIUS_CELLS * np.cos(theta)
            col = CENTER_CELL + PROBE_RADIUS_CELLS * np.sin(theta)
            positions.append((round(row), round(col), round(z)))
    return positions


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


def build_media(resampled):
    sound_speed = np.where(resampled == 2, MYOCARDIUM_SOUND_SPEED,
                   np.where((resampled == 1) | (resampled == 3), BLOOD_SOUND_SPEED, WATER_SOUND_SPEED)).astype(np.float32)
    density = np.where(resampled == 2, MYOCARDIUM_DENSITY,
               np.where((resampled == 1) | (resampled == 3), BLOOD_DENSITY, WATER_DENSITY)).astype(np.float32)
    water_speed = np.full_like(sound_speed, WATER_SOUND_SPEED)
    water_density = np.full_like(density, WATER_DENSITY)
    return (sound_speed, density), (water_speed, water_density)


if __name__ == "__main__":
    import jax
    print("jax devices:", jax.devices())

    positions = element_positions()
    n_elements = len(positions)
    print(f"N_ELEMENTS={n_elements} ({N_RINGS} rings x {N_PER_RING}/ring), "
          f"PROBE_RADIUS_CELLS={PROBE_RADIUS_CELLS:.1f}")
    print(f"compute estimate: {n_elements} elements x 2 media = {n_elements*2} forward sims, "
          f"~19s each -> ~{n_elements*2*19/60:.0f} minutes")

    cropped, native_spacing = load_and_crop("patient023_ed_only.npz")
    resampled = resample_to_target(cropped, native_spacing)
    (phantom_c, phantom_rho), (water_c, water_rho) = build_media(resampled)

    domain = Domain((N_GRID, N_GRID, N_GRID), (DX_M, DX_M, DX_M))

    def make_medium(c_map, rho_map):
        ssm = jnp.expand_dims(jnp.array(c_map), -1)
        dm = jnp.expand_dims(jnp.array(rho_map), -1)
        return Medium(domain=domain, sound_speed=FourierSeries(ssm, domain), density=FourierSeries(dm, domain))

    medium_phantom = make_medium(phantom_c, phantom_rho)
    medium_water = make_medium(water_c, water_rho)

    time_axis = TimeAxis.from_medium(medium_water, cfl=CFL)
    dt = time_axis.dt
    c_min = WATER_SOUND_SPEED
    traversal_s = (DOMAIN_MM * 1e-3 / c_min) * 2 * 1.3
    time_axis = TimeAxis(dt=dt, t_end=traversal_s)
    n_steps = int(time_axis.Nt)
    print(f"dt={dt:.3e}s, n_steps={n_steps}")

    all_rows = jnp.array([p[0] for p in positions])
    all_cols = jnp.array([p[1] for p in positions])
    all_zs = jnp.array([p[2] for p in positions])

    def sensors_all(p, u, rho):
        return p.on_grid[all_rows, all_cols, all_zs, 0]

    t_arr = jnp.arange(n_steps) * dt
    n_cycles = 3
    signal = jnp.sin(2 * jnp.pi * F0_HZ * t_arr) * jnp.exp(
        -((t_arr - n_cycles / F0_HZ) ** 2) / (2 * (n_cycles / 3 / F0_HZ) ** 2))

    def simulate_one(medium, tx_idx):
        tx = positions[tx_idx]
        sources = Sources(positions=([tx[0]], [tx[1]], [tx[2]]),
                           signals=jnp.expand_dims(signal, 0), dt=dt, domain=domain)

        @jit
        def run(m):
            return simulate_wave_propagation(m, time_axis, sources=sources, sensors=sensors_all)

        return np.array(run(medium))  # (n_steps, n_elements)

    all_water_traces = np.zeros((n_elements, n_steps, n_elements), dtype=np.float32)
    all_phantom_traces = np.zeros((n_elements, n_steps, n_elements), dtype=np.float32)
    CHECKPOINT_EVERY = 20  # resilience against a Colab disconnect partway through this ~1hr job

    out_path = os.path.join(OUTPUT_DIR, "cylindrical_multistatic_scan_patient023.npz")

    def save_checkpoint():
        np.savez_compressed(out_path,
                             water_traces=all_water_traces, phantom_traces=all_phantom_traces,
                             positions=np.array(positions), dt=dt, n_steps=n_steps,
                             n_grid=N_GRID, dx_m=DX_M, f0_hz=F0_HZ)

    t0 = time.time()
    for i in range(n_elements):
        all_water_traces[i] = simulate_one(medium_water, i)
        elapsed = time.time() - t0
        eta_s = elapsed / (i + 1) * (2 * n_elements - (i + 1))
        print(f"  water   tx={i+1}/{n_elements}  (elapsed {elapsed:.0f}s, ETA {eta_s/60:.1f}min)")
        if (i + 1) % CHECKPOINT_EVERY == 0:
            save_checkpoint()
            print(f"    [checkpoint saved at water tx={i+1}]")
    for i in range(n_elements):
        all_phantom_traces[i] = simulate_one(medium_phantom, i)
        elapsed = time.time() - t0
        eta_s = elapsed / (n_elements + i + 1) * (2 * n_elements - (n_elements + i + 1))
        print(f"  phantom tx={i+1}/{n_elements}  (elapsed {elapsed:.0f}s, ETA {eta_s/60:.1f}min)")
        if (i + 1) % CHECKPOINT_EVERY == 0:
            save_checkpoint()
            print(f"    [checkpoint saved at phantom tx={i+1}]")

    total_s = time.time() - t0
    print(f"\nTotal scan time: {total_s:.0f}s ({total_s/60:.1f} min)")
    save_checkpoint()
    print(f"Saved {out_path} (final)")
