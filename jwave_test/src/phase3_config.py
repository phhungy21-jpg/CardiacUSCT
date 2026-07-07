"""Phase 3 config — toy moving-phantom proof-of-concept (protocol 3.1).

Extends the Phase 2 ring phantom (phase2_config.py, phase2_forward_model.py)
with a prescribed, exactly-known LV contraction cycle. Per protocol Phase 3
guidance and the pre-Phase-3 diagnostic handoff
(../PHASE2_TO_PHASE3_DIAGNOSTIC_HANDOFF.md), this toy explicitly does NOT
claim physical realism for absolute pressure/SNR scaling or absorption
power-law — those are deferred to Phase 4 with collaborator input. What
DOES matter for Phase 3, and is handled deliberately here, is the
motion-injection method (see phase3_motion_recovery.py docstring).

Motion model: myocardial wall thickness held CONSTANT (simplification —
real myocardium thickens during systole; not modeling that mechanics here,
only testing whether the recovery loop can track a boundary that moves by
a known amount). LV cavity radius follows a smooth half-cosine "cardiac
cycle": largest at end-diastole (ED), smallest at end-systole (ES), back to
ED. Frame count and contraction fraction are self-chosen toy parameters,
not cited physiological values — flagged as such.
"""

import numpy as np

# Self-chosen toy contraction (documented as such, not a cited physiological
# ejection-fraction value): ED radius 6mm (matches Phase 2's static ring),
# ES radius 4mm -> ~33% radius reduction, a clearly-recoverable toy signal.
LV_RADIUS_ED_CELLS = 60   # 6mm
LV_RADIUS_ES_CELLS = 40   # 4mm
WALL_THICKNESS_CELLS = 30  # 3mm, held constant (simplification, see above)

N_FRAMES = 8  # self-chosen, sparse toy sampling across one cycle

# Self-chosen noise levels (NOT physically calibrated SNR/MI — arbitrary
# fractions of local trace amplitude, per the diagnostic handoff's guidance
# that absolute scaling is deferred to Phase 4). Swept explicitly so the
# toy's noise-sensitivity is at least visible, not claimed realistic.
NOISE_LEVELS = [0.0, 0.02, 0.05, 0.10]


def lv_radius_at_phase(phase: float) -> float:
    """phase in [0,1]: 0=ED, 0.5=ES, 1=ED (one full cycle). Smooth
    half-cosine profile — chosen for smoothness, not fit to real LV volume
    curves."""
    ed, es = LV_RADIUS_ED_CELLS, LV_RADIUS_ES_CELLS
    return es + (ed - es) * 0.5 * (1 + np.cos(2 * np.pi * phase))
