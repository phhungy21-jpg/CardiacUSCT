# This file is DERIVED from j-Wave 0.2.1 (jwave/acoustics/time_harmonic.py,
# `born_series`), Copyright the j-Wave authors, licensed under the GNU Lesser
# General Public License v3 (LGPL-3.0-only). Modified 2026-10-24 for the
# fourier_inverse project: adds an ATTENUATION map (jWave's born_series
# ignores attenuation). As a modified LGPL work, this file stays LGPL-3.0.
"""Convergent Born series with attenuation.

Attenuation enters as a complex wavenumber k = w/c + i*alpha (alpha in Np/m,
field ~ exp(i k x) decays as exp(-alpha x)), i.e. k^2 = (w/c + i alpha)^2 with
Im(k^2) > 0, the same sign as the series' own absorbing layer, which keeps the
series convergent. Cold start only (the warm start in cbs_warm.py is broken,
LOG run -16).
"""

from jax import numpy as jnp
from jax.lax import while_loop
from jaxdf.discretization import FourierSeries
from jaxdf.geometry import Domain
from jaxdf.operators import functional
from jaxdf.operators.differential import laplacian

from jwave.acoustics.operators import scale_source_helmholtz
from jwave.acoustics.time_harmonic import (_cbs_norm_units, _cbs_pml,
                                           _cbs_unnorm_units, born_iteration)


def born_series_att(medium, src, alpha_np=None, *, omega, max_iter=5000, tol=1e-3, alpha=1.0):
    """medium.sound_speed (m/s), alpha_np: attenuation map (Np/m, same grid) or None.
    `alpha` is the PML strength parameter, as in jwave. Returns (field, iterations)."""
    k_max = omega / functional(medium.sound_speed)(jnp.amax)
    k_min = omega / functional(medium.sound_speed)(jnp.amin)
    k0 = jnp.sqrt(0.5 * (k_max ** 2 + k_min ** 2))
    medium, omega, k0, src, conversion = _cbs_norm_units(medium, omega, k0, src)   # omega -> 1, dx -> 1
    pml = medium.pml_size
    # jwave's scale_source_helmholtz divides by the SLOWEST speed anywhere in the grid, so a slow
    # inclusion (lung) rescales the whole field vs the water reference (factor 1520/658, LOG run -33).
    # Scale by the boundary (water-bath) speed instead: identical when nothing is slower than water.
    src = src * 2 / (src.domain.dx[0] * medium.sound_speed.on_grid[0, 0, 0])
    pad = tuple([(pml, pml) for _ in range(len(src.domain.N))] + [(0, 0)])
    big = Domain(tuple(x + 2 * pml for x in src.domain.N), src.domain.dx)
    src = FourierSeries(jnp.pad(src.on_grid, pad), big)
    norm_initial = jnp.linalg.norm(src.on_grid)
    sos = medium.sound_speed.on_grid if type(medium.sound_speed) is FourierSeries else medium.sound_speed
    k_in = omega / sos + 0j
    if alpha_np is not None:
        k_in = k_in + 1j * alpha_np[..., None] * conversion["dx"]      # Np/m -> normalized (per grid cell)
    k_sq = _cbs_pml(src, omega / sos[0, 0, 0], pml, alpha)   # PML matched to the boundary (water) medium, not max k (LOG run -33)
    k_sq = k_sq.at[pml:-pml, pml:-pml].set(k_in ** 2)
    k_sq = FourierSeries(k_sq, src.domain)
    epsilon = jnp.amax(jnp.abs(k_sq.on_grid - k0 ** 2))
    resid = lambda u: laplacian(u) + k_sq * u + src

    def cond(carry):
        it, u = carry
        return (it < max_iter) * (jnp.linalg.norm(resid(u).on_grid) / norm_initial > tol)

    def body(carry):
        it, u = carry
        return it + 1, born_iteration(u, k_sq, src, k0=k0, epsilon=epsilon)

    its, u = while_loop(cond, body, (0, FourierSeries.empty(src.domain) + 0j))
    out = -1j * omega * FourierSeries(u.on_grid[pml:-pml, pml:-pml], medium.domain)
    return _cbs_unnorm_units(out, conversion), its
