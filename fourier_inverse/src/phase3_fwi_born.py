"""Nonlinear (full-wave) sound-speed inversion with jWave's convergent Born
series (validated vs. the exact point-source field to ~2.5%, LOG run -10).
Runs on a GPU (Colab). Density and attenuation are ignored by the Born
series, so this inverts SOUND SPEED only.

Physics/units: the solver returns P = K * G s, where G is the exact Green's
operator of L = -(lap + k^2), k^2 = (w/c)^2, and K is a solver normalization
constant. K is MEASURED, not fitted: apply L spectrally to a solved field and
read L P / s at the source pixel.

Gradient of J = 1/2 sum_{s,r} |P_s(x_r) - D_{s,r}|^2 (reciprocity / adjoint):
  dP = G(dk^2 P)  =>  dJ = Re sum_x (G R)(x) dk^2(x) P(x),
  R = conj(residual) injected at the receivers, G R = Solve(R) / K
  (G is complex-symmetric, G^T = G), dk^2/dc = -2 w^2 / c^3.
So one gradient = 1 forward + 1 adjoint Born-series solve per source.
Verified against forward-mode (jvp) directional derivatives, which JAX can
take through the Born series' while_loop (reverse mode it cannot).
"""

import numpy as np
import jax
import jax.numpy as jnp
from jwave import FourierSeries
from jwave.geometry import Domain, Medium
from jwave.acoustics.time_harmonic import born_series

C_WATER = 1520.0   # tissue.py (Duck 1990), the known coupling medium


class Ring:
    """Square grid with point sources/receivers on a ring (pixel positions)."""

    def __init__(self, n, L, ring_r, n_src, n_rec):
        self.n, self.dx = n, L / n
        self.domain = Domain((n, n), (self.dx, self.dx))

        def ring(m):
            a = 2 * np.pi * np.arange(m) / m
            i = np.round(n / 2 + ring_r / self.dx * np.sin(a)).astype(int)
            j = np.round(n / 2 + ring_r / self.dx * np.cos(a)).astype(int)
            return i, j

        self.si, self.sj = ring(n_src)
        self.ri, self.rj = ring(n_rec)
        S = np.zeros((n_src, n, n), np.complex64)
        S[np.arange(n_src), self.si, self.sj] = 1.0
        self.S = jnp.asarray(S)


def make_solver(ring, omega, tol=1e-3, max_iter=5000):
    dom = ring.domain

    def solve(c, s):
        m = Medium(domain=dom, sound_speed=FourierSeries(c[..., None], dom),
                   density=FourierSeries(jnp.full(c.shape + (1,), 1000.0), dom), pml_size=16)
        u = born_series(m, FourierSeries(s[..., None], dom), omega=omega,
                        max_iter=max_iter, tol=tol)
        return jnp.squeeze(u.on_grid)

    return jax.jit(jax.vmap(solve, in_axes=(None, 0)))


def helmholtz_L(P, c, omega, dx):
    """L P = -(lap P + k^2 P), spectral Laplacian on the grid."""
    n = P.shape[-1]
    k = 2 * np.pi * jnp.fft.fftfreq(n, dx)
    k2 = k[:, None] ** 2 + k[None, :] ** 2
    lap = jnp.fft.ifft2(-k2 * jnp.fft.fft2(P))
    return -(lap + (omega / c) ** 2 * P)


def measure_K(P, c, ring, omega):
    """Normalization K = (L P)/s at each source pixel (s = 1 there)."""
    LP = helmholtz_L(P, c, omega, ring.dx)
    Ks = LP[jnp.arange(len(ring.si)), ring.si, ring.sj]
    return jnp.mean(Ks), jnp.std(jnp.abs(Ks)) / jnp.abs(jnp.mean(Ks))


def misfit_and_grad(c, D, ring, solve, omega):
    P = solve(c, ring.S)
    res = P[:, ring.ri, ring.rj] - D
    J = 0.5 * jnp.sum(jnp.abs(res) ** 2)
    R = jnp.zeros_like(ring.S).at[:, ring.ri, ring.rj].add(jnp.conj(res))   # add: receivers may share pixels (run -39)
    Lam = solve(c, R)
    K, _ = measure_K(P, c, ring, omega)
    g_k2 = jnp.real(jnp.sum(Lam * P, axis=0) / K)
    return J, -2 * omega ** 2 / c ** 3 * g_k2


def misfit(c, D, ring, solve):
    res = solve(c, ring.S)[:, ring.ri, ring.rj] - D
    return 0.5 * jnp.sum(jnp.abs(res) ** 2)


# --- Fast path (LOG run -15): ~40x faster AND more accurate than make_solver ---
# jwave's default 16-cell PML (alpha 1) makes the Born series need ~2000-5000
# iterations (often hitting the cap) and leaves ~3% error vs the exact field.
# A 64-cell, alpha 0.5 PML needs ~90 iterations at <0.05% error. Some faster
# settings converge to WRONG fields (16 cells, alpha 0.5: 65% error), so any
# change here must be re-checked against the exact H0 water solution.
PML_FAST, ALPHA_FAST = 64, 0.5


def make_fast_solver(ring, omega, tol=1e-3):
    """vmapped (c, sources, guesses) -> (fields, padded fields for warm start, iterations)."""
    from cbs_warm import born_series_warm
    dom = ring.domain

    def solve1(c, s, g):
        m = Medium(domain=dom, sound_speed=FourierSeries(c[..., None], dom),
                   density=FourierSeries(jnp.full(c.shape + (1,), 1000.0), dom), pml_size=PML_FAST)
        u, padded, its = born_series_warm(m, FourierSeries(s[..., None], dom), g,
                                          omega=omega, tol=tol, alpha=ALPHA_FAST)
        return jnp.squeeze(u.on_grid), padded, its

    return jax.jit(jax.vmap(solve1, in_axes=(None, 0, 0)))


def zero_guess(ring):
    n = ring.n + 2 * PML_FAST
    return jnp.zeros((len(ring.si), n, n, 1), jnp.complex64)


def misfit_and_grad_warm(c, D, ring, solve, omega, cache):
    """As misfit_and_grad, warm-starting forward/adjoint from cache['P'], cache['L'] (updated in place).

    BROKEN -- DO NOT USE (LOG run -16). The Born series stops when its residual
    is below tol relative to the SOURCE norm. After a small model update the
    previous field already passes, so the solve returns after ~2 iterations
    with an essentially stale field (J barely changes; L-BFGS line searches
    fail). Use make_fast_solver with zero_guess(ring) every call (cold solves).
    A correct warm start needs a stopping rule on the change, not the field."""
    P, cache["P"], _ = solve(c, ring.S, cache["P"])
    res = P[:, ring.ri, ring.rj] - D
    R = jnp.zeros_like(ring.S).at[:, ring.ri, ring.rj].add(jnp.conj(res))   # add: receivers may share pixels (run -39)
    Lam, cache["L"], _ = solve(c, R, cache["L"])
    K, _ = measure_K(P, c, ring, omega)
    g_k2 = jnp.real(jnp.sum(Lam * P, axis=0) / K)
    return 0.5 * jnp.sum(jnp.abs(res) ** 2), -2 * omega ** 2 / c ** 3 * g_k2
