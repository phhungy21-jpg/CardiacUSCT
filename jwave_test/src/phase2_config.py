"""Phase 2 acoustic model configuration — cited tissue properties, transducer
geometry, and grid/timestep parameters (protocol 2.1).

Per protocol Phase 2.1: "Use cited literature values, not invented ones...
Document every value and its source." All tissue values below are cited;
none are invented. Per Phase 3's explicit guidance ("before touching real
cardiac anatomy, prove the whole loop works on something simple"), this
module and phase2_forward_model.py operate on a synthetic ring phantom, NOT
real ACDC/M&Ms segmentations — those are reserved for Phase 4.

GATE 2 STATUS: NOT PASSED. Every value here has a cited source and the
sanity-physics check (point source in homogeneous medium; two-tissue
reflection) was already run in ../jwave/ (frozen Phase 1 scout runs), but
Gate 2 explicitly requires "a collaborator with acoustic-physics expertise
[to have] reviewed and signed off on the setup" — this has NOT happened.
Treat everything here as a documented proposal awaiting that review, not
an established result.
"""

from dataclasses import dataclass


@dataclass(frozen=True)
class TissueProperties:
    name: str
    sound_speed: float   # m/s
    density: float       # kg/m^3
    attenuation: float    # dB/cm at 1 MHz
    source: str


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
# Proxy for the anterior chest-wall path (skin/fat/intercostal muscle) the
# beam crosses before reaching the heart. Simplification: modeled as a
# single skeletal-muscle-like layer rather than a layered
# skin/fat/muscle/lung-avoidance stack. Flagged explicitly for collaborator
# review — a real chest-wall model may need multiple layers (protocol 2.1
# transducer-geometry section references the Phase I anterior-arc,
# no-posterior-window rationale, which assumes an acoustic window that
# avoids lung).
CHEST_WALL_PROXY = TissueProperties(
    name="chest wall (skeletal-muscle proxy, simplification — see note above)",
    sound_speed=1580.0, density=1050.0, attenuation=0.74,
    source="Mast (2000) Table 1, 'Muscle, skeletal' row, via ICRU Report 61 (1998)",
)

# ACDC label convention (from pilot/data/processed/ACDC_reg/*.npz
# 'warped_ed_mask'): 0=background, 1=RV cavity, 2=myocardium, 3=LV cavity.
# Both cavities are blood-filled -> same acoustic properties.
ACDC_LABEL_TO_TISSUE = {
    0: CHEST_WALL_PROXY,
    1: BLOOD,   # RV cavity
    2: MYOCARDIUM,
    3: BLOOD,   # LV cavity
}

# --- Transducer geometry -----------------------------------------------
# Frequency: 2.5 MHz is a standard adult transthoracic cardiac imaging
# frequency (typical clinical phased-array cardiac probes operate ~1-5 MHz;
# this is an engineering/design choice, not a cited physical constant, so
# no literature citation is claimed here — flagged for collaborator input
# on whether the lab's target hardware/frequency differs).
F0_HZ = 2.5e6
N_CYCLES = 3  # toneburst length, matches ../jwave/ scout convention

# Anterior-arc line array, carried over from Phase I's anterior-arc
# rationale (no posterior window — real transthoracic acoustic access is
# anterior only). Single focused transmit for Phase 2's "single slice /
# single transmit" requirement (protocol 2.2); full sector-scan transmit
# sequences are a later Phase 4 concern.
N_ELEMENTS = 32
ARRAY_ARC_DEG = 60.0  # angular extent of the anterior arc

# --- Grid resolution + timestep -----------------------------------------
# Wavelength at max sound speed (chest-wall proxy, 1580 m/s) and F0:
#   lambda = c / f0 = 1580 / 2.5e6 = 6.32e-4 m = 0.632 mm
# jWave's Fourier pseudospectral method needs only ~2 points/wavelength in
# principle, but a safety margin is used here for accuracy in a
# heterogeneous, reflecting medium: dx = lambda / 6 =~ 0.105 mm -> rounded
# to 0.1 mm (matches ../jwave/ scout-run grid spacing, so the earlier
# sanity-physics checks are grid-resolution-consistent with this config).
DX_M = 0.1e-3
CFL = 0.3  # matches jWave/k-Wave convention; stability handled by
           # TimeAxis.from_medium via max(sound_speed), see jwave/LOG.md
           # run 2026-07-07-03 observations.

# --- Dimensionality -------------------------------------------------------
# 2D first, per protocol 2.1. Full 3D deferred to a later phase.
DIMENSIONALITY = "2D"
