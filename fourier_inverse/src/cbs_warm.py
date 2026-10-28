# This file is DERIVED from j-Wave 0.2.1 (jwave/acoustics/time_harmonic.py,
# `born_series`), Copyright the j-Wave authors, licensed under the GNU Lesser
# General Public License v3 (LGPL-3.0-only). Modified 2026-10-24 for the
# fourier_inverse project: (1) accepts an initial `guess` on the PML-padded
# grid (warm start), (2) returns the padded field and the iteration count.
# Everything else follows the original algorithm (Osnabrugge et al. 2016).
# This file, as a modified LGPL work, stays under LGPL-3.0.
"""Warm-startable convergent Born series."""

from jax import numpy as jnp
from jax.lax import while_loop
from jaxdf.discretization import FourierSeries
from jaxdf.geometry import Domain
from jaxdf.operators import functional
from jaxdf.operators.differential import laplacian

from jwave.acoustics.operators import scale_source_helmholtz
from jwave.acoustics.time_harmonic import (_cbs_norm_units, _cbs_pml,
                                           _cbs_unnorm_units, born_iteration)


def born_series_warm(medium, src, guess=None, *, omega, max_iter=5000, tol=1e-3, alpha=1.0):
    """Returns (field on the medium domain, padded field for the next warm start, iterations).
    `guess` is a padded-grid array (N + 2*pml, ..., 1) from a previous call, or None (= zeros,
    identical to jwave.born_series)."""
    k_max = omega / functional(medium.sound_speed)(jnp.amax)
    k_min = omega / functional(medium.sound_speed)(jnp.amin)
    k0 = jnp.sqrt(0.5 * (k_max ** 2 + k_min ** 2))
    medium, omega, k0, src, conversion = _cbs_norm_units(medium, omega, k0, src)   # omega is now 1 (normalized)
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
    k_sq = _cbs_pml(src, omega / sos[0, 0, 0], pml, alpha)   # PML matched to the boundary (water) medium, not max k (LOG run -33)
    k_sq = k_sq.at[pml:-pml, pml:-pml].set((omega / sos) ** 2 + 0j)
    k_sq = FourierSeries(k_sq, src.domain)
    epsilon = jnp.amax(jnp.abs(k_sq.on_grid - k0 ** 2))

    field0 = FourierSeries.empty(src.domain) + 0j
    if guess is not None:
        field0 = FourierSeries(guess, src.domain)

    resid = lambda u: laplacian(u) + k_sq * u + src

    def cond(carry):
        it, u = carry
        return (it < max_iter) * (jnp.linalg.norm(resid(u).on_grid) / norm_initial > tol)

    def body(carry):
        it, u = carry
        return it + 1, born_iteration(u, k_sq, src, k0=k0, epsilon=epsilon)

    its, u = while_loop(cond, body, (0, field0))
    padded = u.on_grid
    # normalized omega (=1), as in jwave; first version used the physical omega -> output off by exactly w
    out = -1j * omega * FourierSeries(padded[pml:-pml, pml:-pml], medium.domain)
    return _cbs_unnorm_units(out, conversion), padded, its
