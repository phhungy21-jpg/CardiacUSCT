# MANIFEST — jwave_ring_tomography/ (ring/water-bath ultrasound tomography, new project)

Tracks on-disk state so a new session doesn't have to re-derive it by
exploring the filesystem. Update this when the directory structure or
generated artifacts change materially. See `../ring_tomography_phase_protocol.md`
for the full phase-gate structure and the reasoning for why this is a
NEW project rather than a continuation of `../jwave_test/`.

## Origin and what's reused vs. NOT reused

Created 2026-07-09, in direct response to `jwave_test/`'s PROJECT
CLOSURE finding (fine blind boundary reconstruction from sparse
multistatic acoustic data is a dead end for irregular anatomy) and a
parallel ChatGPT-consulted exploration of denser acquisition geometries
(two-opposite-probe scanning, rotating/ring-array transmission +
reflection tomography). Per explicit user decision, this project
assumes a **water-bath / full-surround acquisition geometry**
(Butterfly/Midjourney-style whole-body scanner model), not the
clinically standard anterior-arc-only handheld transthoracic probe —
this resolves the physical conflict a literal "ring around the chest"
would otherwise have with the prior project's anterior-arc rationale,
at the honest cost of being a research/investigational setup.

**Reused from prior projects (per user: "we can use heart constructed
data and tissue information from old projects")**:
- Cited tissue properties (BLOOD, MYOCARDIUM, CHEST_WALL_PROXY) —
  cloned from `jwave_test/src/phase2_config.py` into
  `src/phase2_config.py` here, unchanged values, same citations (Mast
  2000 / ICRU 61 / Duck 1990). These were already Gate-2-reviewed once
  (`jwave_test` run -12) — the values don't change with acquisition
  geometry, only the medium around them does.
- Real cardiac anatomy pathway: `pilot/data/processed/ACDC_reg/` and
  `MandMs_reg/` (registration-derived real motion, same source Phase I
  and `jwave_test` both used) — not yet copied/referenced by any script
  here, but this is the intended Phase 4 anatomy source, same
  nearest-neighbor-resampling rule (CLAUDE.md) as before.

**NOT reused (deliberately left behind in `jwave_test/`, frozen)**:
- The sparse 4/8/16-probe multistatic backprojection code, the
  curvature-weighted calibration/weight model, the blind per-angle
  discovery machinery, and the injectivity-probe scripts — all of that
  is specific to the closed sparse pulse-echo problem and does not
  transfer to a ring/transmission-tomography setup. If any of its
  PRINCIPLES turn out useful here (e.g. the ghost-cone/angular-coverage
  mechanism, or the injectivity-probe discipline of checking information
  content before trusting a readout), re-derive and re-cite them
  explicitly rather than importing the old code.
- The anterior-arc-only transducer geometry (`ARRAY_ARC_DEG=60.0` in
  `jwave_test`'s config) — superseded by the ring/full-surround decision
  above; `N_ELEMENTS` remains a placeholder pending a real compute-budget
  estimate.

## Motion-capture mode: SEQUENTIAL (decided 2026-07-09)

Per explicit user choice ("lets go with sequential for data clarity"):
each element/angle fires and records in its own turn, keeping every
measurement cleanly attributable to one transmit angle at one moment —
NOT a simultaneous multi-element snapshot. The tradeoff, made explicit
rather than discovered later: this reintroduces a CT-style
motion-during-acquisition problem `jwave_test`'s frozen-scene sparse
setup never had to face. Consequence for Phase 2: the forward model
needs a scan-time -> cardiac-phase mapping (a decided scan rate and
cardiac cycle length) so each element's capture uses the medium state
at ITS OWN moment in the scan, not one static scene reused for every
element. See `../ring_tomography_phase_protocol.md`'s Phase 2.1 for the
updated requirement.

## New, not previously needed

- **WATER tissue property** (`src/phase2_config.py`) — the water-bath
  coupling medium. Proposed citation: Duck (1990), water entry,
  temperature-adjusted toward a body-comfortable bath range. NOT yet
  collaborator-confirmed — flagged exactly like every other cited value
  awaiting this project's own Gate 2.
- **Motion-during-acquisition modeling, required from Phase 2 onward**
  (protocol Phase 0.1/2.1) — this is the one lesson from `jwave_test`'s
  closure most directly acted on here: that project never modeled
  motion-during-scan because its sparse setup was implicitly
  simultaneous; a dense angular/ring scan cannot make that same
  implicit assumption without deciding it explicitly first.

## Layout

- `jwave_ring_tomography/src/` — `phase2_config.py` (tissue properties +
  ring geometry placeholders, see above). No forward-model or capture
  code written yet — next action is Phase 0.1's access-geometry/
  compute-budget/motion-mode decisions, then Phase 1's scout smoke tests
  (point source in water, two-tissue reflection, full-ring source/receive
  geometry sanity check), mirroring `jwave/`'s original Phase 1 approach.
- `jwave_ring_tomography/results/` — will hold generated artifacts;
  gitignored per the project-wide `data/`/`results/` convention once
  anything is written there (double-check `.gitignore` covers this path
  before the first real anatomy data lands here, per CLAUDE.md's
  "never commit patient data" rule).
- `jwave_ring_tomography/LOG.md` — running lab notebook for this
  project, starting fresh (not cloned from `jwave_test/LOG.md` — this is
  a new project, not a continuation).

## Status

**Phase 0 (setup/scoping) IN PROGRESS.** Access geometry decided
(water-bath/full-surround, user decision 2026-07-09). NOT yet decided:
simultaneous-ring-capture vs. sequential/rotating acquisition mode
(protocol Phase 0.1) — this must be settled before any Phase 2 forward-
model code is written, not discovered empirically. No simulation code
exists yet. No gates passed yet — Gate 2 in particular will need a
FRESH collaborator signoff specific to this water-bath/transmission-
tomography setup, independent of `jwave_test`'s existing Gate 2.
