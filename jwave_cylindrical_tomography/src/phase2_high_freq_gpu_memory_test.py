"""Tests whether N=300 and N=470 (the grid sizes needed for 2.5MHz at
this project's own ~2.2 points/wavelength standard, vs. the current
N=150/0.8MHz design point) fit in GPU memory NOW that sensor-based
recording is used (run -07's fix) -- the original "N=200/300 OOM"
finding (run -05) was measured BEFORE that fix, using the old default
full-field recording, which was later proven to be the actual memory
hog, not grid size alone (run -07). Nobody has tested whether these
larger grids fit under the corrected (sensors-based) recording scheme
-- this is that test, homogeneous water only (no patient data needed,
matching phase0_3d_gpu_timing_sweep.py's convention), so it can run
standalone with no embedded data.

If both sizes fit, a full batched 2.5MHz/36-element/8-ring re-scan
becomes a real option to design. If they don't, that sets the real
current ceiling before committing to anything.
"""

import time
import numpy as np

DOMAIN_MM = 130.0
WATER_SOUND_SPEED, WATER_DENSITY = 1520.0, 994.0
CFL = 0.3
TARGET_POINTS_PER_WAVELENGTH = 2.2
F0_HZ_TARGET = 2.5e6

CANDIDATE_N = [300, 470]


def test_one(N_GRID):
    from jax import numpy as jnp, jit
    from jwave import FourierSeries
    from jwave.geometry import Domain, Medium, TimeAxis, Sources
    from jwave.acoustics import simulate_wave_propagation

    dx_mm = DOMAIN_MM / N_GRID
    dx_m = dx_mm * 1e-3
    wavelength_mm = WATER_SOUND_SPEED / F0_HZ_TARGET * 1e3
    pts_per_wavelength = wavelength_mm / dx_mm
    print(f"\n=== N={N_GRID}: dx={dx_mm:.4f}mm, {pts_per_wavelength:.2f} points/wavelength at {F0_HZ_TARGET/1e6}MHz ===")

    domain = Domain((N_GRID, N_GRID, N_GRID), (dx_m, dx_m, dx_m))
    sound_speed = jnp.full((N_GRID, N_GRID, N_GRID, 1), WATER_SOUND_SPEED)
    density = jnp.full((N_GRID, N_GRID, N_GRID, 1), WATER_DENSITY)
    medium = Medium(domain=domain, sound_speed=FourierSeries(sound_speed, domain),
                     density=FourierSeries(density, domain))

    time_axis = TimeAxis.from_medium(medium, cfl=CFL)
    dt = time_axis.dt
    traversal_s = (DOMAIN_MM * 1e-3 / WATER_SOUND_SPEED) * 2 * 1.3
    time_axis = TimeAxis(dt=dt, t_end=traversal_s)
    n_steps = int(time_axis.Nt)
    print(f"dt={dt:.3e}s, n_steps={n_steps}")

    center = N_GRID // 2
    quarter = N_GRID // 4
    rx_rows = jnp.array([center + quarter, center, center])
    rx_cols = jnp.array([center, center + quarter, center - quarter])
    rx_deps = jnp.array([center, center, center])

    def sensors(p, u, rho):
        return p.on_grid[rx_rows, rx_cols, rx_deps, 0]

    t_arr = jnp.arange(n_steps) * dt
    n_cycles = 3
    signal = jnp.sin(2 * jnp.pi * F0_HZ_TARGET * t_arr) * jnp.exp(
        -((t_arr - n_cycles / F0_HZ_TARGET) ** 2) / (2 * (n_cycles / 3 / F0_HZ_TARGET) ** 2))
    sources = Sources(positions=([center - quarter], [center], [center]),
                       signals=jnp.expand_dims(signal, 0), dt=dt, domain=domain)

    @jit
    def run(m):
        return simulate_wave_propagation(m, time_axis, sources=sources, sensors=sensors)

    t0 = time.time()
    try:
        result = np.array(run(medium))
        elapsed = time.time() - t0
        print(f"SUCCESS: {elapsed:.2f}s, output shape {result.shape}, "
              f"peak magnitude {np.abs(result).max():.4e}")
        return True, elapsed
    except Exception as e:
        elapsed = time.time() - t0
        print(f"FAILED after {elapsed:.2f}s: {type(e).__name__}: {str(e)[:300]}")
        return False, elapsed


if __name__ == "__main__":
    import jax
    print("jax devices:", jax.devices())
    results = {}
    for N in CANDIDATE_N:
        ok, elapsed = test_one(N)
        results[N] = (ok, elapsed)
        if not ok:
            print(f"\nStopping sweep -- N={N} failed, larger N would fail too.")
            break
    print("\n=== Summary ===")
    for N, (ok, elapsed) in results.items():
        print(f"  N={N}: {'OK' if ok else 'FAILED'} ({elapsed:.1f}s)")
