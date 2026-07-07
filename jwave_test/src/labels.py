"""Shared status labels, applied to every figure/result in this codebase
until collaborator signoff happens. See
../PHASE2_TO_PHASE3_DIAGNOSTIC_HANDOFF.md, "Phase 3->4 hard gate" section.

Purpose: no phase transition should silently upgrade a toy/preliminary
result into an established one. Every figure carries an explicit banner;
every accuracy number reported from Phase 4 onward must also carry the
ground-truth-motion-floor caption.
"""

PENDING_SIGNOFF_BANNER = (
    "PENDING ACOUSTIC-PHYSICS SIGNOFF -- not collaborator-reviewed "
    "(Gate 2). Toy/preliminary result: attenuation solver gap, "
    "uncalibrated source amplitude, and (pre-Phase-4) untested "
    "staircasing all still open -- see PHASE2_TO_PHASE3_DIAGNOSTIC_HANDOFF.md"
)

GT_FLOOR_CAPTION = (
    "Ground truth = Phase I registration-derived motion, itself imperfect: "
    "median boundary error ~1 voxel (~1.5mm), rising to ~2-3 voxels "
    "(~3-4.5mm) for high-contraction (NOR/HCM) patients, RV worst "
    "(~1.1 voxel median) -- pilot/LIMITATIONS.md Gap 1. Reported accuracy "
    "is relative to this imperfect floor, not to true cardiac motion."
)

TOY_EXACT_GT_CAPTION = (
    "Ground truth here is a toy phantom's EXACTLY prescribed motion (not "
    "Phase I registration-derived) -- no ground-truth-motion floor applies "
    "to this number. This does not carry over to Phase 4, where ground "
    "truth switches to imperfect registration-derived motion; see "
    "GT_FLOOR_CAPTION."
)


def add_banner(fig, caption=PENDING_SIGNOFF_BANNER):
    """Adds a small pending-signoff banner to the bottom of a figure."""
    fig.text(0.5, 0.01, caption, ha="center", va="bottom", fontsize=7,
              color="firebrick", wrap=True)
