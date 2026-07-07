"""Phase 3 — asymmetric (regional hypokinesis) ring phantom.

Extends the isotropic ring phantom (phase3_four_probe_tracking.py) with
an ANGLE-DEPENDENT contraction amplitude, modeling a common, clinically
realistic scenario: regional hypokinesis (e.g. post-infarct reduced wall
motion in one region while the rest of the wall contracts normally).
This makes the 4-probe test genuinely informative -- for the isotropic
phantom, 4 probes agreeing was a symmetry check; here, they should
DISAGREE in a specific, predictable way if the method correctly resolves
regional motion differences.

Convention: theta=0 at the TOP probe direction, +90 at RIGHT, +/-180 at
BOTTOM, -90 at LEFT (matches phase3_four_probe_tracking.py's probe
placement exactly).

Hypokinetic region centered at the LEFT probe's direction (theta=-90),
with HYPOKINETIC_FACTOR=0.3 (30% of normal contraction amplitude, i.e. a
70% reduction -- hypokinesis, not full akinesis) and a 60-degree
half-width smooth (raised-cosine) taper, so neighboring probes (top,
bottom) are not meaningfully affected.
"""

import numpy as np
from jax import numpy as jnp

from jwave import FourierSeries
from jwave.geometry import Medium

import phase2_config as cfg
import phase3_config as p3cfg
import phase3_four_probe_tracking as fpt

HYPOKINETIC_CENTER_DEG = -90.0  # LEFT probe direction
HYPOKINETIC_HALFWIDTH_DEG = 60.0
HYPOKINETIC_FACTOR = 0.3  # 30% of normal contraction amplitude at the center


def regional_factor(theta_deg):
    """1.0 (normal) everywhere except smoothly reduced to
    HYPOKINETIC_FACTOR within HYPOKINETIC_HALFWIDTH_DEG of the
    hypokinetic center, via a raised-cosine taper (no sharp discontinuity)."""
    angular_dist = np.abs(((theta_deg - HYPOKINETIC_CENTER_DEG + 180) % 360) - 180)
    bump = np.where(
        angular_dist < HYPOKINETIC_HALFWIDTH_DEG,
        0.5 * (1 + np.cos(np.pi * angular_dist / HYPOKINETIC_HALFWIDTH_DEG)),
        0.0,
    )
    return 1.0 - (1.0 - HYPOKINETIC_FACTOR) * bump


def local_lv_radius(theta_deg, phase):
    """Angle-dependent LV radius: normal contraction scaled by the
    regional factor at that angle. At ED (phase=0 or 1) all angles agree
    (radius = ED radius, since contraction depth is 0 there) -- the
    asymmetry only shows up DURING contraction."""
    ed, es = p3cfg.LV_RADIUS_ED_CELLS, p3cfg.LV_RADIUS_ES_CELLS
    normal_radius = p3cfg.lv_radius_at_phase(phase)
    contraction_depth = ed - normal_radius  # 0 at ED, max at ES
    factor = regional_factor(theta_deg)
    return ed - factor * contraction_depth


def build_medium_asymmetric(phase, wall_thickness_cells):
    yy, xx = np.mgrid[0:fpt.N[0], 0:fpt.N[1]]
    dx_pixel = xx - fpt.center[1]
    dy_pixel = yy - fpt.center[0]
    dist = np.sqrt(dx_pixel ** 2 + dy_pixel ** 2)
    theta_deg = np.degrees(np.arctan2(dx_pixel, -dy_pixel))

    lv_radius_local = local_lv_radius(theta_deg, phase)
    outer_radius_local = lv_radius_local + wall_thickness_cells

    label_map = np.zeros(fpt.N, dtype=int)
    label_map[dist < outer_radius_local] = 2
    label_map[dist < lv_radius_local] = 3

    sound_speed_map = np.zeros(fpt.N, dtype=np.float32)
    density_map = np.zeros(fpt.N, dtype=np.float32)
    for label, tissue in cfg.ACDC_LABEL_TO_TISSUE.items():
        m = label_map == label
        sound_speed_map[m] = tissue.sound_speed
        density_map[m] = tissue.density
    ssm = jnp.expand_dims(jnp.array(sound_speed_map), -1)
    dm = jnp.expand_dims(jnp.array(density_map), -1)
    return Medium(domain=fpt.domain, sound_speed=FourierSeries(ssm, fpt.domain),
                  density=FourierSeries(dm, fpt.domain)), label_map
