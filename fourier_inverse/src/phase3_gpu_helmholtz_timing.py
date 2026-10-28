"""Phase 3 -> nonlinear scoping: time jWave's frequency-domain Helmholtz
solvers on a GPU (run on Colab; also runs on CPU at small size for checking).

Times, per grid size:
  - GMRES helmholtz_solver   (supports density; differentiable)
  - convergent Born series   (FFT-only; ignores density)
  - cross-check: relative difference GMRES vs Born series (same medium)
  - one gradient d(misfit)/d(sound speed) through GMRES (the cost of one
    reconstruction step per source)
  - 8 sources batched with vmap
Phantom: heart-like, density-matched (Born series has no density term):
water background, myocardium-speed annulus, blood-speed core, from the
cited values in tissue.py. Synthetic only -- no patient data.
Grid: 4 points per wavelength (spectral method), 130 mm heart domain.
"""

import sys
import time

import numpy as np

C_WATER, C_MYO, C_BLOOD = 1520.0, 1576.0, 1584.0   # tissue.py (Mast 2000 / Duck 1990)
RHO = 1000.0
PPW = 4
CASES = [("small test, 64 mm @ 0.7 MHz", 64e-3, 0.7e6),
         ("2D heart, 130 mm @ 0.8 MHz", 130e-3, 0.8e6),
         ("2D heart, 130 mm @ 1.2 MHz", 130e-3, 1.2e6),
         ("2D heart, 130 mm @ 2.5 MHz", 130e-3, 2.5e6)]


def grid_for(L, f):
    n = int(np.ceil(L / (C_WATER / f) * PPW / 32) * 32)
    return n, L / n


def build(n, dx):
    from jax import numpy as jnp
    from jwave import FourierSeries
    from jwave.geometry import Domain, Medium
    domain = Domain((n, n), (dx, dx))
    r = np.hypot(*np.meshgrid(*(2 * [(np.arange(n) - n / 2) * dx]), indexing="ij"))
    R = 0.3 * n * dx
    c = np.where(r < R, C_MYO, C_WATER)
    c = np.where(r < 0.65 * R, C_BLOOD, c)
    medium = Medium(domain=domain,
                    sound_speed=FourierSeries(jnp.asarray(c)[..., None], domain),
                    density=FourierSeries(jnp.full((n, n, 1), RHO), domain),
                    pml_size=16)
    src = np.zeros((n, n, 1), dtype=np.complex64)
    src[n // 2, 20] = 1.0
    return domain, medium, FourierSeries(jnp.asarray(src), domain), jnp.asarray(c)


def timed(fn, reps=3):
    out = fn()
    _block(out)                       # compile + warm-up
    t = time.time()
    for _ in range(reps):
        out = fn()
        _block(out)
    return (time.time() - t) / reps, out


def _block(x):
    import jax
    jax.tree_util.tree_map(lambda a: a.block_until_ready() if hasattr(a, "block_until_ready") else a, x)


def run_case(label, L, f, do_cbs=True):
    import jax
    from jax import numpy as jnp
    from jwave import FourierSeries
    from jwave.acoustics.time_harmonic import born_series, helmholtz_solver

    n, dx = grid_for(L, f)
    domain, medium, src, c0 = build(n, dx)
    omega = 2 * np.pi * f
    print(f"\n== {label}: grid {n}x{n} ({n*n/1e6:.2f}M points), dx={dx*1e3:.3f} mm", flush=True)

    gm = jax.jit(lambda m, s: helmholtz_solver(m, omega, s, tol=1e-3, maxiter=1000))
    t_gm, u_gm = timed(lambda: gm(medium, src))
    print(f"   GMRES solve:        {t_gm:7.2f} s", flush=True)

    if do_cbs:
        cb = jax.jit(lambda m, s: born_series(m, s, omega=omega, max_iter=2000, tol=1e-3))
        t_cb, u_cb = timed(lambda: cb(medium, src))
        # GMRES puts its PML INSIDE the domain, the Born series OUTSIDE, so
        # compare only the interior (a 32^2 check grid was mostly PML)
        k = 16 + 8
        a = jnp.squeeze(u_gm.on_grid)[k:-k, k:-k]
        b = jnp.squeeze(u_cb.on_grid)[k:-k, k:-k]
        diff = float(jnp.linalg.norm(a - b) / jnp.linalg.norm(a))
        print(f"   Born series solve:  {t_cb:7.2f} s   (rel. diff vs GMRES {diff:.2e})", flush=True)

    rec = (slice(None), n - 21)       # a receiver column opposite the source

    def misfit(c):
        m = medium.replace("sound_speed", FourierSeries(c[..., None], domain))
        u = helmholtz_solver(m, omega, src, tol=1e-3, maxiter=1000)
        return jnp.sum(jnp.abs(jnp.squeeze(u.on_grid)[rec]) ** 2)

    g = jax.jit(jax.grad(misfit))
    t_g, grad = timed(lambda: g(c0), reps=2)
    print(f"   gradient (1 src):   {t_g:7.2f} s   (finite: {bool(jnp.all(jnp.isfinite(grad)))})",
          flush=True)

    srcs = jnp.stack([jnp.roll(src.on_grid, k * n // 8, axis=0) for k in range(8)])
    vm = jax.jit(jax.vmap(lambda s: helmholtz_solver(
        medium, omega, FourierSeries(s, domain), tol=1e-3, maxiter=1000).on_grid))
    t_v, _ = timed(lambda: vm(srcs), reps=2)
    print(f"   8 sources batched:  {t_v:7.2f} s  ({t_v/8:.2f} s per source)", flush=True)
    return dict(label=label, n=n, gmres=t_gm, grad=t_g, batch8=t_v)


def main(cases):
    import jax
    print("devices:", jax.devices(), flush=True)
    out = []
    for label, L, f in cases:
        try:
            out.append(run_case(label, L, f))
        except Exception as e:                      # e.g. GPU out of memory at the largest size
            print(f"   FAILED: {type(e).__name__}: {str(e)[:300]}", flush=True)
            break
    # one reconstruction ~ 10 freqs x 10 iters x 1.5 (line search) = 150 steps,
    # each = gradient for every source; 128 sources, batched cost per source
    print("\nEstimated one blind reconstruction (128 sources, ~150 steps):")
    for r in out:
        per_src = max(r["batch8"] / 8, 1e-9) * (r["grad"] / max(r["gmres"], 1e-9))
        print(f"   {r['label']:32s} ~{150 * 128 * per_src / 3600:6.1f} GPU-hours")


if __name__ == "__main__":
    if "--cpu-check" in sys.argv:       # tiny local sanity run
        main([("CPU check, 60 mm @ 0.6 MHz", 60e-3, 0.6e6)])
    else:
        main(CASES)
