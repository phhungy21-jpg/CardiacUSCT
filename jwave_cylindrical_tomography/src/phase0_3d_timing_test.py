"""Phase 0 (scoping): the single blocking question before any 3D
design decision -- does jWave 3D simulation run in a practical amount
of time? Runs ONE small 3D forward simulation (homogeneous water,
single point source, single receiver) and times it, on CPU (this
session's default execution environment) -- extrapolate from here
before committing to any domain size, resolution, or scanner layout.

Deliberately starts SMALL: N=(60,60,60) = 216,000 grid points, roughly
2.4x jwave_ring_tomography's 2D domain (300x300=90,000 points) -- NOT
yet at a clinically realistic scale, just enough to get a first real
per-point-cost data point on THIS machine before scaling up.
"""

import time
import numpy as np

from jax import numpy as jnp, jit
from jwave import FourierSeries
from jwave.geometry import Domain, Medium, TimeAxis, Sources
from jwave.acoustics import simulate_wave_propagation

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "jwave_ring_tomography", "src"))
import phase2_config as cfg

N3D = (60, 60, 60)
DX3D = (cfg.DX_M, cfg.DX_M, cfg.DX_M)

if __name__ == "__main__":
    print(f"PHASE 0 3D TIMING TEST: N={N3D} ({np.prod(N3D)} grid points, "
          f"vs. jwave_ring_tomography's 2D 300x300={300*300} points)")
    print("  homogeneous water medium, single point source, single receiver, CPU execution")

    t0 = time.time()
    domain = Domain(N3D, DX3D)
    sound_speed_map = np.full(N3D, cfg.WATER.sound_speed, dtype=np.float32)
    density_map = np.full(N3D, cfg.WATER.density, dtype=np.float32)
    ssm = jnp.expand_dims(jnp.array(sound_speed_map), -1)
    dm = jnp.expand_dims(jnp.array(density_map), -1)
    medium = Medium(domain=domain, sound_speed=FourierSeries(ssm, domain), density=FourierSeries(dm, domain))
    print(f"  medium construction: {time.time()-t0:.2f}s")

    t0 = time.time()
    time_axis = TimeAxis.from_medium(medium, cfl=cfg.CFL)
    # short duration -- just enough for the wave to cross a modest fraction of the domain,
    # this is a TIMING test, not a physically meaningful simulation
    dt = time_axis.dt
    t_end_short = dt * 200  # 200 steps, enough to see real per-step cost, not a full traversal
    time_axis = TimeAxis(dt=dt, t_end=t_end_short)
    n_steps = int(time_axis.Nt)
    print(f"  time_axis construction: {time.time()-t0:.2f}s, dt={dt:.3e}s, n_steps={n_steps}")

    center = tuple(n // 2 for n in N3D)
    src_pos = (center[0] - 15, center[1], center[2])
    signal = jnp.sin(2 * jnp.pi * cfg.F0_HZ * jnp.arange(n_steps) * dt) * jnp.exp(
        -((jnp.arange(n_steps) * dt - 10 / cfg.F0_HZ) ** 2) / (2 * (3 / cfg.F0_HZ) ** 2))
    sources = Sources(positions=([src_pos[0]], [src_pos[1]], [src_pos[2]]),
                       signals=jnp.expand_dims(signal, 0), dt=dt, domain=domain)

    @jit
    def run(m):
        return simulate_wave_propagation(m, time_axis, sources=sources)

    print(f"\n  compiling + running {n_steps} steps (first call includes JIT compile time)...")
    t0 = time.time()
    pressure = run(medium)
    pressure.on_grid.block_until_ready()
    first_call_s = time.time() - t0
    print(f"  first call (compile + run): {first_call_s:.2f}s")

    t0 = time.time()
    pressure2 = run(medium)
    pressure2.on_grid.block_until_ready()
    second_call_s = time.time() - t0
    print(f"  second call (post-JIT, pure execution): {second_call_s:.2f}s")

    per_step_s = second_call_s / n_steps
    print(f"\n  per-timestep cost (post-JIT): {per_step_s*1e3:.3f} ms/step at N={N3D}")
    print(f"  ({np.prod(N3D):.2e} grid points -> {per_step_s/np.prod(N3D)*1e9:.4f} ns/point/step)")
