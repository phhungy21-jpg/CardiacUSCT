# This file is DERIVED from j-Wave 0.2.1 (jwave/acoustics/time_harmonic.py,
# `born_series`), Copyright the j-Wave authors, licensed under the GNU Lesser
# General Public License v3 (LGPL-3.0-only). Modified 2026-10-26 for the
# fourier_inverse project: adds ATTENUATION (as cbs_ext.py) and variable
# DENSITY (jWave's born_series ignores both). As a modified LGPL work, this
# file stays LGPL-3.0.
"""Convergent Born series with sound speed, density and attenuation.

Density enters through the standard Liouville transform. The variable-density
equation  rho div(rho^-1 grad p) + k^2 p = -s  becomes, with p = sqrt(rho) psi,

    lap psi + (k^2 + q) psi = -s / sqrt(rho),
    q = 1/2 lap(ln rho) - 1/4 |grad ln rho|^2,

an ordinary scalar Helmholtz equation, so the Born series applies unchanged
with k^2 -> k^2 + q. Sources and receivers sit in the uniform-density water
bath, where sqrt(rho) is a constant that cancels in the water-shot calibration,
so the solver returns psi with the plain source s.

ln rho is Gaussian-smoothed (sigma = rho_smooth cells) before differentiation:
q contains the derivative of a step at every tissue interface, which a spectral
derivative turns into Gibbs ringing. The smoothing is a modelling choice; how
well it works is checked against the independent time-domain solver (LOG run -29).
Cold start only (the warm start in cbs_warm.py is broken, LOG run -16).
"""

import numpy as np
from jax import numpy as jnp
from jax.lax import while_loop
from jaxdf.discretization import FourierSeries
from jaxdf.geometry import Domain
from jaxdf.operators.differential import laplacian

from jwave.acoustics.operators import scale_source_helmholtz
from jwave.acoustics.time_harmonic import (_cbs_norm_units, _cbs_pml,
                                           _cbs_unnorm_units, born_iteration)


def density_potential(rho, dx, rho_smooth=1.0):
    """q = 1/2 lap(ln rho) - 1/4 |grad ln rho|^2 (1/m^2) on a 2D grid, spectral derivatives."""
    n0, n1 = rho.shape
    k0 = 2 * np.pi * jnp.fft.fftfreq(n0, dx)[:, None]
    k1 = 2 * np.pi * jnp.fft.fftfreq(n1, dx)[None, :]
    L = jnp.fft.fft2(jnp.log(rho)) * jnp.exp(-0.5 * (rho_smooth * dx) ** 2 * (k0 ** 2 + k1 ** 2))
    lap = jnp.real(jnp.fft.ifft2(-(k0 ** 2 + k1 ** 2) * L))
    g0, g1 = jnp.real(jnp.fft.ifft2(1j * k0 * L)), jnp.real(jnp.fft.ifft2(1j * k1 * L))
    return 0.5 * lap - 0.25 * (g0 ** 2 + g1 ** 2)


def born_series_full(medium, src, alpha_np=None, rho=None, *, omega, max_iter=20000, tol=1e-3,
                     alpha=1.0, rho_smooth=1.0):
    """medium.sound_speed (m/s); alpha_np: attenuation (Np/m) or None; rho: density (kg/m^3) or None.
    All maps on the medium grid. `alpha` is the PML strength, as in jwave. Returns (field, iterations)."""
    sos_phys = medium.sound_speed.on_grid[..., 0]
    q_phys = None if rho is None else density_potential(rho, medium.domain.dx[0], rho_smooth)
    medium, omega, _, src, conversion = _cbs_norm_units(medium, omega, 1.0, src)   # omega -> 1, dx -> 1
    pml = medium.pml_size
    # jwave's scale_source_helmholtz divides by the SLOWEST speed anywhere in the grid, so a slow
    # inclusion (lung) rescales the whole field vs the water reference (factor 1520/658, LOG run -33).
    # Scale by the boundary (water-bath) speed instead: identical when nothing is slower than water.
    src = src * 2 / (src.domain.dx[0] * medium.sound_speed.on_grid[0, 0, 0])
    pad = tuple([(pml, pml) for _ in range(len(src.domain.N))] + [(0, 0)])
    big = Domain(tuple(x + 2 * pml for x in src.domain.N), src.domain.dx)
    src = FourierSeries(jnp.pad(src.on_grid, pad), big)
    norm_initial = jnp.linalg.norm(src.on_grid)
    sos = medium.sound_speed.on_grid
    k_in = omega / sos + 0j
    if alpha_np is not None:
        k_in = k_in + 1j * alpha_np[..., None] * conversion["dx"]      # Np/m -> per grid cell
    k_in_sq = k_in ** 2
    if q_phys is not None:
        k_in_sq = k_in_sq + q_phys[..., None] * conversion["dx"] ** 2   # 1/m^2 -> per cell^2
    # k0 from the actual in-domain potential (with density, k^2 + q can reach beyond the speed-only range)
    re = jnp.real(k_in_sq)
    k0 = jnp.sqrt(0.5 * (jnp.amax(re) + jnp.maximum(jnp.amin(re), 0.0)))
    # Absorbing layer matched to the BOUNDARY medium (corner pixel = water bath), not to the largest
    # wavenumber anywhere as in jwave: with a slow inclusion (lung, 658 m/s) the jwave choice makes
    # the layer 2.3x mismatched to the water it borders, it reflects, and the series converges to a
    # wrong field (300-400% error vs the exact cylinder solution, LOG run -33).
    k_sq = _cbs_pml(src, omega / sos[0, 0, 0], pml, alpha)
    k_sq = k_sq.at[pml:-pml, pml:-pml].set(k_in_sq)
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
