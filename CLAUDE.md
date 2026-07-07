# Instructions for Claude Code sessions on this repo

This repo now spans two phases, each self-contained in its own subfolder:

- **`pilot/`** — Phase I pilot (cardiac motion recovery from synthetic Doppler
  projections, ACDC → M&Ms). Complete, frozen, written up. Source of truth:
  `pilot/cardiac_motion_pilot_protocol.md`, `pilot/LOG.md`, `pilot/MANIFEST.md`.
  Do not modify pilot code/results except to fix a documentation error — it is
  a finished, reported result.
- **`jwave/`** — Phase II Phase 1 exploratory scout record (k-Wave/jWave
  toolchain smoke tests: point source, two-tissue reflection, array source).
  Frozen as of 2026-07-07 — findings here (e.g. the blood/myocardium
  weak-contrast result) are informative but NOT collaborator-reviewed, so
  don't treat them as established. See `jwave/LOG.md` / `jwave/MANIFEST.md`.
- **`jwave_test/`** — Phase II active workspace (cloned from `jwave/` on
  2026-07-07). Phase 2 (acoustic model definition) work happens here. Source
  of truth: `acoustic_simulation_phase_protocol.md` (root),
  `jwave_test/LOG.md`, `jwave_test/MANIFEST.md`.

Read the relevant protocol (and that phase's `LOG.md` + `MANIFEST.md`) before
doing substantive work in either folder — they tell you which phase-gate is
active and what's already been validated.

## Rules that apply across sessions

- **Do not skip a validation gate.** Each protocol phase ends in a ⛔ gate.
  If asked to move to the next phase, check that phase's `LOG.md` first — if
  the current phase's gate isn't logged as passed, say so instead of
  proceeding.
- **M&Ms is evaluated exactly once**, in pilot Phase 6, after the model is
  fully frozen from ACDC-only tuning. This is already done and frozen —
  never re-tune the pilot model against M&Ms.
- **Fixed seeds everywhere.** Use `pilot/src/seed.py::set_all_seeds()` (pilot)
  or the jWave-phase equivalent once created. Log the seed in that phase's
  `LOG.md`.
- **Never commit patient data.** `data/` (at any depth, e.g. `pilot/data/`,
  `jwave/data/`) is gitignored — if you add a new data path outside an
  existing `data/` dir, gitignore it too, and double check `git status`
  before staging.
- **After any run** (registration, simulation, training, evaluation), append
  an entry to that phase's `LOG.md` using the relevant protocol's Appendix C
  template — phase, seed, config, dataset/split, result, gate pass/fail,
  observations, next action.
- **Update that phase's `MANIFEST.md`** when the on-disk state changes
  materially (new dataset added, results generated, directory structure
  changes) so the next session doesn't have to re-derive it by exploring the
  filesystem.
- Masks resample with **nearest-neighbor**, never linear/cubic — protocol
  pitfall, silently corrupts labels if violated.
- Keep models small given low N (pilot protocol Phase 5) — a large
  architecture makes the cross-cohort generalization gap uninterpretable.
- **jWave phase (Phase II) specific:** Gate 2 (acoustic model correctness)
  requires a collaborator with acoustic-physics expertise to sign off — it is
  not passable by Claude or the user alone. Never invent tissue acoustic
  property values; every value needs a cited source.
