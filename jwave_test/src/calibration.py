"""Source-amplitude calibration proposal (diagnostic item 3).

PROXY AUDIT (solo-dev stand-in, NOT Gate 2 -- see labels.py).

Every script in this project so far uses an arbitrary source amplitude
(p0/toneburst peak = 1.0), which is fine for internal relative comparisons
but not for absolute SNR/MI claims (flagged in
PHASE2_TO_PHASE3_DIAGNOSTIC_HANDOFF.md item 3).

Proposed anchor: the FDA's Mechanical Index (MI), a standard clinical
safety/output metric:

    MI = P_r,derated (MPa) / sqrt(f_MHz)

FDA regulatory limit: MI <= 1.9 for all applications except the eye.
Typical MEASURED transthoracic cardiac imaging MI: ~0.18 (fundamental),
~0.25 (harmonic) -- both far below the regulatory ceiling, i.e. typical
clinical practice is much gentler than the legal maximum.
Source: FDA MI definition and cardiac fundamental/harmonic MI
measurements (0.18/0.25) -- see chat log for search citations (Wikipedia
"Mechanical index"; PMC "Conditionally Increased Acoustic Pressures in
Nonfetal Diagnostic Ultrasound Examinations").

This proposes anchoring our arbitrary source amplitude (=1.0) to a
representative clinical MI of 0.2 at our f0=2.5MHz, giving a derated peak
rarefactional pressure of MI * sqrt(f_MHz) = 0.2 * sqrt(2.5) = 0.316 MPa.

THIS IS A PROPOSAL, NOT A COLLABORATOR-CONFIRMED CALIBRATION. A real
acoustic-physics reviewer should confirm whether anchoring to a
"typical clinical MI" (rather than e.g. a specific target device's
measured output) is an appropriate calibration choice for this study's
claims.
"""

TARGET_MI = 0.2  # representative clinical transthoracic cardiac MI
                  # (measured range ~0.18-0.25 fundamental/harmonic)


def peak_pressure_pa(f0_hz, mi=TARGET_MI):
    f0_mhz = f0_hz / 1e6
    p_r_mpa = mi * (f0_mhz ** 0.5)
    return p_r_mpa * 1e6  # Pa


def calibrate_arbitrary_units(f0_hz, arbitrary_peak_amplitude=1.0, mi=TARGET_MI):
    """Returns Pa per 1.0 arbitrary source-amplitude unit."""
    return peak_pressure_pa(f0_hz, mi) / arbitrary_peak_amplitude
