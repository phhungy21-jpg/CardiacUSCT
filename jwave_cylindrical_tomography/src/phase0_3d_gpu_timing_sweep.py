"""Phase 0 (scoping), GPU version: per user's decision (2026-07-10) to
go GPU-only from the start rather than compromise resolution/domain
size/angle density, this replaces the assumed "30x GPU speedup" guess
in run -02's CPU extrapolation with REAL measured GPU numbers across a
sweep of domain sizes -- run this on a Colab GPU kernel, not locally
(no patient data dependency at all -- homogeneous water medium only --
so this needs no upload/embedding workaround, unlike the beating-heart
notebook).
"""

import time
import numpy as np

from jax import numpy as jnp, jit
from jwave import FourierSeries
from jwave.geometry import Domain, Medium, TimeAxis, Sources
from jwave.acoustics import simulate_wave_propagation

# inlined tissue constants (avoids needing jwave_ring_tomography/src on the Colab path)
WATER_SOUND_SPEED = 1520.0
WATER_DENSITY = 994.0
DX_M = 0.1e-3
CFL = 0.3
F0_HZ = 2.5e6
N_STEPS_TEST = 200

SIZES_TO_TEST = [40, 60, 100, 150, 200, 300]

if __name__ == "__main__":
    import jax
    print("jax devices:", jax.devices())

    results = []
    for n in SIZES_TO_TEST:
        N3D = (n, n, n)
        dx3d = (DX_M, DX_M, DX_M)
        try:
            domain = Domain(N3D, dx3d)
            sound_speed_map = np.full(N3D, WATER_SOUND_SPEED, dtype=np.float32)
            density_map = np.full(N3D, WATER_DENSITY, dtype=np.float32)
            ssm = jnp.expand_dims(jnp.array(sound_speed_map), -1)
            dm = jnp.expand_dims(jnp.array(density_map), -1)
            medium = Medium(domain=domain, sound_speed=FourierSeries(ssm, domain), density=FourierSeries(dm, domain))

            time_axis = TimeAxis.from_medium(medium, cfl=CFL)
            dt = time_axis.dt
            time_axis = TimeAxis(dt=dt, t_end=dt * N_STEPS_TEST)
            n_steps = int(time_axis.Nt)

            center = tuple(k // 2 for k in N3D)
            src_pos = (center[0] - n // 4, center[1], center[2])
            t = jnp.arange(n_steps) * dt
            signal = jnp.sin(2 * jnp.pi * F0_HZ * t) * jnp.exp(-((t - 10 / F0_HZ) ** 2) / (2 * (3 / F0_HZ) ** 2))
            sources = Sources(positions=([src_pos[0]], [src_pos[1]], [src_pos[2]]),
                               signals=jnp.expand_dims(signal, 0), dt=dt, domain=domain)

            @jit
            def run(m):
                return simulate_wave_propagation(m, time_axis, sources=sources)

            t0 = time.time()
            pressure = run(medium)
            pressure.on_grid.block_until_ready()
            first_s = time.time() - t0

            t0 = time.time()
            pressure2 = run(medium)
            pressure2.on_grid.block_until_ready()
            second_s = time.time() - t0

            per_step_ms = second_s / n_steps * 1e3
            n_points = n ** 3
            print(f"N={n:4d} ({n_points:>10,} points): first(compile+run)={first_s:6.2f}s, "
                  f"post-JIT={second_s:6.2f}s ({n_steps} steps), {per_step_ms:.3f} ms/step")
            results.append((n, n_points, first_s, second_s, n_steps, per_step_ms))
        except Exception as e:
            print(f"N={n:4d}: FAILED -- {type(e).__name__}: {e}")
            results.append((n, n**3, None, None, None, None))

    print("\n--- Summary: extrapolated single-simulation cost for a full 30mm-equivalent round trip ---")
    c_min = WATER_SOUND_SPEED
    for n, n_points, first_s, second_s, n_steps, per_step_ms in results:
        if per_step_ms is None:
            continue
        domain_mm = n * DX_M * 1e3
        traversal_s = (domain_mm * 1e-3 / c_min) * 2 * 1.3
        dt_this = second_s / n_steps / (per_step_ms / 1e3) if per_step_ms else None
        # steps needed for a full traversal at this domain's own physical size
        steps_needed = traversal_s / (second_s / n_steps)
        est_total_s = (per_step_ms / 1e3) * steps_needed
        print(f"  N={n:4d} (domain={domain_mm:.1f}mm): estimated full single-sim time "
              f"~{est_total_s:.1f}s ({est_total_s/60:.2f} min)")
