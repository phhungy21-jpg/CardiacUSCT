"""Shared status labels, applied to every figure/result in this project
until ITS OWN collaborator signoff happens. Cloned from
`jwave_test/src/labels.py` (same purpose, same discipline) rather than
imported, since this project's Gate 2 status is independent --
`jwave_test`'s Gate 2 signoff does NOT carry over to this water-bath/
transmission-tomography setup. See `../ring_tomography_phase_protocol.md`.

Purpose: no phase transition should silently upgrade a toy/preliminary
result into an established one. Every figure carries an explicit
banner; every accuracy number reported once real (non-toy) ground
truth is used must also carry the ground-truth-motion-floor caption.
"""

PENDING_SIGNOFF_BANNER = (
    "PENDING ACOUSTIC-PHYSICS SIGNOFF -- not collaborator-reviewed "
    "(Gate 2, THIS project's own -- jwave_test's Gate 2 does not carry "
    "over). Toy/preliminary result: water tissue property not yet "
    "confirmed, attenuation not modeled, motion-during-acquisition not "
    "yet implemented -- see ring_tomography_phase_protocol.md"
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
    "to this number. This does not carry over once real anatomy is used, "
    "where ground truth switches to imperfect registration-derived motion; "
    "see GT_FLOOR_CAPTION."
)


def add_banner(fig, caption=PENDING_SIGNOFF_BANNER):
    """Adds a small pending-signoff banner to the bottom of a figure."""
    fig.text(0.5, 0.01, caption, ha="center", va="bottom", fontsize=7,
              color="firebrick", wrap=True)
