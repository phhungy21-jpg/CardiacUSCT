# LOG — jwave_ring_tomography/ (ring/water-bath ultrasound tomography)

Running lab notebook for this project. Each entry follows the Appendix
C template from `../ring_tomography_phase_protocol.md` (phase, seed,
config, dataset/split, result, gate pass/fail, observations, next
action). This is a NEW project's log, started fresh on 2026-07-09 — it
does not continue `../jwave_test/LOG.md`'s run numbering. See that
file's "PROJECT CLOSURE" entry for the full history and reasoning that
led here, and `MANIFEST.md` in this directory for what was reused vs.
deliberately left behind.

### Run 2026-07-09-01 — Project created; access geometry decided; scaffolding only, no simulation yet
- Phase: 0 (setup/scoping). Created in direct response to
  `jwave_test/`'s PROJECT CLOSURE finding and a parallel ChatGPT-
  consulted exploration of denser acquisition geometries (two-opposite-
  probe scanning, rotating/ring-array transmission+reflection
  tomography). Per explicit user request ("new project folder for
  this... we can use heart constructed data and tissue information from
  old projects").
- Decision made this run (via AskUserQuestion, user selected the
  recommended option): **water-bath / full-surround acquisition
  geometry** (Butterfly/Midjourney-style whole-body scanner model),
  not the clinically standard anterior-arc-only handheld transthoracic
  probe. This resolves a real physical conflict: `jwave_test`'s tissue
  config explicitly carried over Phase I's "no posterior acoustic
  window, anterior access only" rationale, which a literal 360° ring
  around a standard chest-probe setup would have violated without
  addressing it. The honest cost of this choice, to carry into any
  future writeup: this is now a research/investigational acquisition
  mode, not standard point-of-care cardiac ultrasound.
- Work done: `ring_tomography_phase_protocol.md` written (root),
  adapting `acoustic_simulation_phase_protocol.md`'s phase-gate
  structure for the 3 things this project changes at once (acquisition
  geometry, acquisition mode [transmission+reflection, not reflection-
  only], and explicit motion-during-acquisition modeling from the
  start — the one lesson most directly acted on from `jwave_test`'s
  closure). `src/phase2_config.py` written: clones `jwave_test`'s cited
  tissue properties (BLOOD/MYOCARDIUM/CHEST_WALL_PROXY) unchanged, adds
  a NEW water-bath coupling-medium property (proposed citation: Duck
  1990, NOT yet collaborator-confirmed), and leaves `N_ELEMENTS`/
  `CAPTURE_MODE` as explicit placeholders pending Phase 0.1's
  compute-budget and motion-handling decisions. `MANIFEST.md` written,
  documenting what's reused (tissue properties, the Phase I/`jwave_test`
  real-anatomy pathway via `pilot/data/processed/`) vs. deliberately
  NOT reused (the closed sparse-probe backprojection/blind-discovery
  code stays in `jwave_test/`, frozen).
- Physical sanity checked? by whom?: N/A — no simulation run yet, pure
  scaffolding.
- Gate passed? (Y/N): N/A. Phase 0 not yet complete — the
  simultaneous-vs-sequential capture-mode decision (protocol 0.1) is
  still open.
- Next action: decide simultaneous-ring-capture vs. sequential/rotating
  acquisition mode (Phase 0.1) BEFORE writing any Phase 1 scout code;
  get a real per-transmit compute estimate at the new, larger
  (full-torso, not one anterior arc) domain size before committing to
  an element count; then proceed to Phase 1 scout smoke tests (point
  source in water, two-tissue water/chest-wall reflection, full-ring
  source/receive geometry sanity check), mirroring `jwave/`'s original
  Phase 1 approach.
