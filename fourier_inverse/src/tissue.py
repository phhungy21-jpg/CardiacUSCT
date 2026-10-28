"""Cited tissue properties, cloned UNCHANGED from
`jwave_ring_tomography/src/phase2_config.py` (which cloned BLOOD/MYOCARDIUM/
CHEST_WALL_PROXY from `jwave_test/src/phase2_config.py`).

Sources: Mast, T.D. (2000), ARLO 1(2), 37-42, Table 1 (via ICRU Report 61,
1998) for the three soft tissues; Duck, F.A. (1990) for water.

Do NOT add tissues (fat, bone, air, ...) without a cited primary source.
GATE 2 STATUS for this project: NOT PASSED.
"""

from dataclasses import dataclass


@dataclass(frozen=True)
class TissueProperties:
    name: str
    sound_speed: float   # m/s
    density: float       # kg/m^3
    attenuation: float   # dB/cm at 1 MHz (LUNG/BONE below: at their stated frequency)
    source: str

    @property
    def impedance(self) -> float:
        return self.density * self.sound_speed


BLOOD = TissueProperties(
    "blood", 1584.0, 1060.0, 0.20,
    "Mast (2000) Table 1, 'Blood' row, via ICRU Report 61 (1998)",
)
MYOCARDIUM = TissueProperties(
    "myocardium", 1576.0, 1060.0, 0.52,
    "Mast (2000) Table 1, 'Muscle, cardiac' row, via ICRU Report 61 (1998)",
)
CHEST_WALL_PROXY = TissueProperties(
    "chest wall (skeletal-muscle proxy)", 1580.0, 1050.0, 0.74,
    "Mast (2000) Table 1, 'Muscle, skeletal' row, via ICRU Report 61 (1998)",
)
WATER = TissueProperties(
    "water (bath, ~30-37C)", 1520.0, 994.0, 0.0022,
    "Duck (1990) water entry, temperature-adjusted (not collaborator-confirmed)",
)

ALL_TISSUES = (WATER, BLOOD, MYOCARDIUM, CHEST_WALL_PROXY)


# --- Added 2026-10-25 for the chest (lung/rib) test; NOT collaborator-reviewed ---
# `attenuation` here is the value AT THE STATED FREQUENCY (see source), not per MHz.
#
# LUNG (aerated, bulk). Miller DL (2016) Ultrasound Med Biol 42(12):2743-57,
# doi:10.1016/j.ultrasmedbio.2016.08.006 (review), summarizing Dunn F (1986)
# J Acoust Soc Am 80(4):1248-50, doi:10.1121/1.393818 and related work:
#   speed 658 m/s at 1 MHz (rising to 1180 m/s at 5 MHz); attenuation 37.3 dB/cm
#   at 1 MHz (101 dB/cm at 5 MHz); bulk density ~200-400 kg/m^3 depending on
#   inflation (midpoint 300 used).
# CAVEATS: measured at 1-5 MHz; our 0.15-0.8 MHz is an EXTRAPOLATION. Bulk lung
# is a homogenized model of a highly heterogeneous air/septa structure.
LUNG = TissueProperties(
    "lung (aerated, bulk; values at 1 MHz)", 658.0, 300.0, 37.3,
    "Miller 2016 UMB review (doi:10.1016/j.ultrasmedbio.2016.08.006) citing Dunn 1986 "
    "(doi:10.1121/1.393818); density = midpoint of quoted 200-400 kg/m^3",
)
# BONE (cortical). Pichardo S, Sin VW, Hynynen K (2011) Phys Med Biol 56(1):219-50,
# doi:10.1088/0031-9155/56/1/014: fresh human calvaria, cortical bone speed
# 2384 +- 130 m/s at 270 kHz and 2471 +- 90 m/s at 836 kHz; attenuation 33 +- 9 Np/m
# at 270 kHz and 240 +- 9 Np/m at 836 kHz; mean apparent density 1904 +- 117 kg/m^3.
# CAVEAT: SKULL bone, not rib. Ribs are thinner cortical shells around marrow, so
# this is a stand-in for the schematic ribs.
BONE_CORTICAL = TissueProperties(
    "cortical bone (skull; values at 836 kHz)", 2471.0, 1904.0, 240.0 * 8.686 / 100,
    "Pichardo, Sin & Hynynen 2011 PMB (doi:10.1088/0031-9155/56/1/014); "
    "attenuation 240 Np/m at 836 kHz converted to dB/cm",
)


# --- Added 2026-10-26 for the MRI-traced heart phantom (dim lining); NOT collaborator-reviewed ---
# FAT (porcine, fresh/refrigerated): CRC Press / Taylor & Francis textbook chapter 5
# "Acoustical Properties of Biological Tissue", Table 5.1 "Typical Values of Acoustic
# Parameters for Mammalian Tissues" (publisher sample chapter, routledge.com file
# ATQ03_-_K14770__Chapter.pdf; book title not confirmed from the file):
#   density 0.937, 0.95 g/cm^3; speed 1479, 1450 m/s; attenuation 0.07 +- 0.02 cm^-1 at
#   1 MHz, 37 C, ~ f^1. Values used = means of the two quoted measurements.
#   Attenuation 0.07 cm^-1 (taken as Np/cm) = 0.608 dB/cm at 1 MHz.
# The same table gives bovine heart muscle 1546 +- 4.7 m/s (and 1570 m/s): the +-4.7 m/s
# spread is used ONLY as the amplitude of the optional myocardial speed texture.
# CAVEAT: used as a PROXY for the MRI's less-bright lining (pericardium / epicardial fat);
# that identity is an assumption.
FAT_PERICARDIUM_PROXY = TissueProperties(
    "fat (porcine; proxy for pericardium/epicardial fat)", 1464.5, 943.5, 0.07 * 8.686,
    "CRC Press ch. 5 'Acoustical Properties of Biological Tissue', Table 5.1 (porcine fat; "
    "means of 1479/1450 m/s and 0.937/0.95 g/cm^3; 0.07 cm^-1 at 1 MHz)",
)
MYOCARDIUM_SPEED_SPREAD = 4.7   # m/s, same Table 5.1, bovine heart muscle (1546 +- 4.7 m/s)


# --- Added 2026-10-27 for the two-thigh phantom (LOG run -44); NOT collaborator-reviewed ---
# SKIN (dermis). Moran CM, Bush NL, Bamber JC (1995) Ultrasound Med Biol 21(9):1177-90,
# doi:10.1016/0301-5629(95)00049-6, PMID 8849832: excised human skin (8 samples, 2 cadavers), 20-30 MHz, 24 C.
#   speed: epidermis 1645 m/s, dermis 1595 m/s (dermis used: it is most of the skin thickness);
#   attenuation alpha = alpha1 f^n, dermis alpha1 = 0.264 +- 0.17 dB/cm, n = 1.69 +- 0.084.
#   Density NOT reported -> PROXY: skeletal-muscle density (Mast 2000), an assumption.
# CAVEATS: measured at 20-30 MHz and 24 C; our 0.1-0.8 MHz use of the power law is an EXTRAPOLATION.
SKIN_DERMIS = TissueProperties(
    "skin (dermis; density proxy = skeletal muscle)", 1595.0, 1050.0, 0.264,
    "Moran, Bush & Bamber 1995 UMB (doi:10.1016/0301-5629(95)00049-6); alpha = 0.264 f^1.69 dB/cm; "
    "density = Mast 2000 skeletal muscle (proxy)",
)
SKIN_ATTEN_EXPONENT = 1.69
# BONE MARROW (human femur/tibia; elderly; fatty). Kawasaki S et al. (2015) J Acoust Soc Am 138(1):EL83-7,
# doi:10.1121/1.4922764, PMID 26233067: 4-10 MHz, 30-40 C; speed 1410 m/s and attenuation 4.4 dB/cm at 10 MHz
# (36 C); power-law exponent n = 2.0 (30-38 C). => alpha = 0.044 f^2 dB/cm (f in MHz).
#   Density NOT reported -> PROXY: fat density (CRC Table 5.1, as FAT_PERICARDIUM_PROXY); the same group reported
#   marrow properties similar to bovine fat (2013 conference abstract), which motivates the proxy.
# CAVEATS: 4-10 MHz data extrapolated to 0.1-0.8 MHz; elderly marrow (fatty).
BONE_MARROW = TissueProperties(
    "bone marrow (femur/tibia, fatty; density proxy = fat)", 1410.0, 943.5, 0.044,
    "Kawasaki et al. 2015 JASA (doi:10.1121/1.4922764); alpha = 0.044 f^2 dB/cm; density = CRC fat (proxy)",
)
MARROW_ATTEN_EXPONENT = 2.0
