"""Attenuating extension of jWave's time-domain solver.

PROXY AUDIT (Claude acting as a technical stand-in reviewer for this SOLO
DEV RUN ONLY -- explicitly NOT a substitute for Gate 2's real
acoustic-physics collaborator signoff; see labels.py and
PHASE2_TO_PHASE3_DIAGNOSTIC_HANDOFF.md "Phase 3->4 hard gate"). This module
is the fix for diagnostic item 4: jWave's `simulate_wave_propagation`
(jwave/acoustics/time_varying.py) never references `medium.attenuation` in
its PDE update (confirmed by reading the source, see LOG.md run -06) --
attenuation is only implemented in jWave's separate frequency-domain
Helmholtz solver, unusable for transient pulse-echo simulation.

APPROACH: rather than switch toolkits (k-Wave) or leave attenuation
unmodeled, this reimplements jWave's own scan loop (same
momentum_conservation_rhs / mass_conservation_rhs / pressure_from_density
operators, imported directly from jwave.acoustics.time_varying, not
duplicated logic) with ONE addition: after each PML update, the density
field is multiplied by a per-cell exponential damping factor

    damp = exp(-alpha_Np_per_m * c0 * dt)

This is the standard "frequency-independent relaxation absorption"
approximation: over one timestep the wave travels ~c0*dt, and a plane
wave's pressure amplitude decays as exp(-alpha*distance) for absorption
coefficient alpha (Nepers/m). It assumes alpha scales LINEARLY with
frequency (power-law exponent y=1), NOT the y~1.1-1.5 more accurate for
real soft tissue -- a documented simplification, not a hidden one. This is
a legitimate, literature-standard first-order absorption model (used e.g.
in basic FDTD acoustic solvers before adding full multi-relaxation/
fractional-Laplacian dispersion terms), but a real acoustic-physics
reviewer should assess whether y=1 is adequate for this study's claims
before this is used for anything beyond solo-dev-run characterization.

Cited attenuation values (dB/cm at 1MHz) come from phase2_config.py
(Mast 2000, Table 1). Conversion: 1 dB = 0.11513 Np (amplitude
convention), extrapolated to f0 assuming y=1: alpha(f0) = alpha(1MHz) *
(f0/1MHz).
"""

from typing import Union

import numpy as np
from jax import numpy as jnp
from jax.lax import scan
from jax import checkpoint as jax_checkpoint

from jwave.acoustics.time_varying import (
    ongrid_wave_prop_params,
    momentum_conservation_rhs,
    mass_conservation_rhs,
    pressure_from_density,
    TimeWavePropagationSettings,
)
from jwave.geometry import Medium, TimeAxis
from jwave.signal_processing import smooth

DB_PER_1MHZ_TO_NP_PER_M = 0.11513 * 100  # dB/cm -> Np/m, at the SAME frequency


def attenuation_field_np_per_m(attenuation_db_cm_at_1mhz, f0_hz, y=1.0):
    """Convert cited dB/cm-at-1MHz values to Np/m at f0_hz, assuming a
    power-law frequency dependence with exponent y (y=1: linear-with-
    frequency, the simplification used here -- see module docstring)."""
    f0_mhz = f0_hz / 1e6
    alpha_db_cm_at_f0 = attenuation_db_cm_at_1mhz * (f0_mhz ** y)
    return alpha_db_cm_at_f0 * DB_PER_1MHZ_TO_NP_PER_M


def simulate_wave_propagation_attenuating(
    medium: Medium,
    time_axis: TimeAxis,
    attenuation_np_per_m,  # array/Field, same grid shape as medium.sound_speed
    f0_hz: float,
    *,
    settings: TimeWavePropagationSettings = TimeWavePropagationSettings(),
    sources=None,
    sensors=None,
    u0=None,
    p0=None,
):
    """Same as jwave.acoustics.simulate_wave_propagation, plus per-step
    exponential density damping from attenuation_np_per_m (Np/m, already
    converted to f0_hz -- see attenuation_field_np_per_m)."""
    if sensors is None:
        sensors = lambda p, u, rho: p

    dt = time_axis.dt
    output_steps = jnp.arange(0, time_axis.Nt, 1)

    params = ongrid_wave_prop_params(medium, time_axis, settings=settings)
    c_ref = params["c_ref"]
    pml_rho = params["pml_rho"]
    pml_u = params["pml_u"]

    c0_field = medium.sound_speed.on_grid[..., 0] if hasattr(
        medium.sound_speed, "on_grid") else medium.sound_speed
    atten_arr = attenuation_np_per_m.on_grid[..., 0] if hasattr(
        attenuation_np_per_m, "on_grid") else attenuation_np_per_m
    damp_grid = jnp.exp(-atten_arr * c0_field * dt)  # per-cell scalar array
    damp_rho = pml_rho.replace_params(jnp.expand_dims(damp_grid, -1))
    # BUG CAUGHT BY validate_attenuation.py (see LOG.md): damping only rho
    # (pressure) and not u (velocity) under-damps by roughly 2x, because
    # the undamped velocity field keeps re-injecting energy into pressure
    # via mass_conservation_rhs each step. Both fields must be damped.
    ndim = len(medium.domain.N)
    damp_u = pml_u.replace_params(jnp.stack([damp_grid] * ndim, axis=-1))

    shape = tuple(list(medium.domain.N) + [len(medium.domain.N)])
    shape_one = tuple(list(medium.domain.N) + [1])
    if u0 is None:
        u0 = pml_u.replace_params(jnp.zeros(shape))
    if p0 is None:
        p0 = pml_rho.replace_params(jnp.zeros(shape_one))
    else:
        if settings.smooth_initial:
            p0_params = p0.params[..., 0]
            p0_params = jnp.expand_dims(smooth(p0_params), -1)
            p0 = p0.replace_params(p0_params)
        u0 = -dt * momentum_conservation_rhs(
            p0, u0, medium, c_ref=c_ref, dt=dt) / 2

    rho = (p0.replace_params(
        jnp.stack([p0.params[..., i] for i in range(p0.domain.ndim)],
                  axis=-1)) / p0.domain.ndim)
    rho = rho / (medium.sound_speed ** 2)

    fields = [p0, u0, rho]

    def scan_fun(fields, n):
        p, u, rho = fields
        mass_src_field = 0.0 if sources is None else sources.on_grid(n)

        du = momentum_conservation_rhs(p, u, medium, c_ref=c_ref, dt=dt)
        u = pml_u * (pml_u * u + dt * du)
        u = damp_u * u  # absorption loss on velocity

        drho = mass_conservation_rhs(p, u, mass_src_field, medium,
                                     c_ref=c_ref, dt=dt)
        rho = pml_rho * (pml_rho * rho + dt * drho)
        rho = damp_rho * rho  # absorption loss on density/pressure

        p = pressure_from_density(rho, medium)
        return [p, u, rho], sensors(p, u, rho)

    if settings.checkpoint:
        scan_fun = jax_checkpoint(scan_fun)

    _, ys = scan(scan_fun, fields, output_steps)
    return ys
