"""Phase 2 acoustic model configuration for the ring/water-bath
tomography project — cited tissue properties, ring geometry, and
grid/timestep parameters (see `../ring_tomography_phase_protocol.md`
Phase 2.1).

Tissue properties (BLOOD, MYOCARDIUM, CHEST_WALL_PROXY) are CLONED
from `jwave_test/src/phase2_config.py`, not re-derived — those values
were already Gate-2-reviewed once (run -12, for the anterior-arc
pulse-echo setup) and the values themselves don't change with
acquisition geometry, only the surrounding medium does. WATER is NEW
(a water-bath/full-surround setup needs a coupling medium the prior
anterior-arc project never did) and is NOT yet Gate-2-reviewed for
THIS setup.

GATE 2 STATUS: NOT PASSED for this project. Every value below has a
cited source, but per `ring_tomography_phase_protocol.md`, this
project's water-bath/transmission-tomography setup requires its OWN
collaborator signoff — `jwave_test`'s Gate 2 (run -12) does not carry
over. Treat everything here as a documented proposal awaiting that
review, not an established result.
"""

from dataclasses import dataclass


@dataclass(frozen=True)
class TissueProperties:
    name: str
    sound_speed: float   # m/s
    density: float       # kg/m^3
    attenuation: float    # dB/cm at 1 MHz
    source: str


# --- Cloned unchanged from jwave_test/src/phase2_config.py --------------
# Cited from: Mast, T.D. (2000). "Empirical relationships between acoustic
# parameters in human soft tissues." Acoustics Research Letters Online,
# 1(2), 37-42, Table 1. That table itself compiles from ICRU Report 61
# (1998) (blood, cardiac muscle, skeletal muscle rows) and Duck, F.A.
# (1990) "Physical Properties of Tissue: A Comprehensive Reference Book"
# (Academic, London).
BLOOD = TissueProperties(
    name="blood", sound_speed=1584.0, density=1060.0, attenuation=0.20,
    source="Mast (2000) Table 1, 'Blood' row, via ICRU Report 61 (1998)",
)
MYOCARDIUM = TissueProperties(
    name="myocardium (cardiac muscle)", sound_speed=1576.0, density=1060.0,
    attenuation=0.52,
    source="Mast (2000) Table 1, 'Muscle, cardiac' row, via ICRU Report 61 (1998)",
)
CHEST_WALL_PROXY = TissueProperties(
    name="chest wall (skeletal-muscle proxy, simplification — see jwave_test note)",
    sound_speed=1580.0, density=1050.0, attenuation=0.74,
    source="Mast (2000) Table 1, 'Muscle, skeletal' row, via ICRU Report 61 (1998)",
)

# --- NEW for this project: water-bath coupling medium --------------------
# PROPOSED, not yet collaborator-confirmed. Sound speed in water is
# strongly temperature-dependent (unlike soft tissue); a real bath would
# be temperature-controlled near body temperature for patient comfort and
# to keep the water/tissue impedance mismatch small. Value chosen for
# ~30-37C bath temperature.
WATER = TissueProperties(
    name="water (bath coupling medium, ~30-37C)",
    sound_speed=1520.0, density=994.0, attenuation=0.0022,
    source="Duck, F.A. (1990) 'Physical Properties of Tissue,' water entry "
           "(temperature-adjusted toward body-bath range; a real bath's exact "
           "temperature must be confirmed with collaborators, not assumed)",
)

# ACDC label convention (from pilot/data/processed/ACDC_reg/*.npz
# 'warped_ed_mask'): 0=background, 1=RV cavity, 2=myocardium, 3=LV cavity.
# Background is now WATER (the bath), not CHEST_WALL_PROXY — the chest
# wall itself becomes an explicit intermediate layer once real anatomy
# with a torso outline is used (Phase 4), not modeled in the Phase 2/3
# toy phantoms below.
ACDC_LABEL_TO_TISSUE = {
    0: WATER,
    1: BLOOD,   # RV cavity
    2: MYOCARDIUM,
    3: BLOOD,   # LV cavity
}

# --- Transducer geometry (NEW: ring/full-surround, not anterior-arc) -----
# Per protocol Phase 0.1: simultaneous-ring-capture vs. sequential/rotating
# acquisition must be decided before building the forward model -- NOT YET
# DECIDED here; both N_ELEMENTS and the capture-mode flag are placeholders
# until that decision is made and logged.
F0_HZ = 2.5e6  # unchanged from jwave_test -- revisit if collaborators want
N_CYCLES = 3   # a different frequency for a water-bath full-body scanner

N_ELEMENTS = None       # TBD -- budget a real per-transmit cost first (Phase 0.1)
CAPTURE_MODE = None     # "simultaneous" | "sequential" -- TBD (Phase 0.1)

# --- Grid resolution + timestep -----------------------------------------
# Wavelength at max sound speed in this project's tissue set (chest-wall
# proxy, 1580 m/s -- water is slower, 1520 m/s, so still the limiting case
# unless a faster tissue is added) and F0:
#   lambda = c / f0 = 1580 / 2.5e6 = 6.32e-4 m = 0.632 mm
# Same safety-margin choice as jwave_test: dx = lambda / 6 =~ 0.105mm ->
# rounded to 0.1mm. NOTE: a full-surround water-bath domain is physically
# LARGER (whole torso, not one anterior arc's worth of near-field) --
# recompute the actual domain size and memory/compute cost before assuming
# this resolution is affordable at the new domain scale (protocol 0.1).
DX_M = 0.1e-3
CFL = 0.3

# --- Dimensionality -------------------------------------------------------
DIMENSIONALITY = "2D"  # 2D first, per protocol -- same discipline as jwave_test
