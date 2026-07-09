# Instructions for Claude Code sessions on this repo

This repo now spans three phases, each self-contained in its own subfolder:

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
- **`jwave_test/`** — Phase II workspace (cloned from `jwave/` on
  2026-07-07). **Frozen as of 2026-07-09.** Gate 2 (acoustic-physics
  signoff) and Gate 3's literal checkbox both passed, and Phase 4.1
  (benchmark) is done — but the extensive reconstruction-methodology
  investigation built on top of that (runs -14 through -77) concluded
  that fine blind boundary-shape reconstruction from sparse multistatic
  acoustic data is a well-diagnosed dead end for irregular real cardiac
  anatomy, with only a narrower, anatomy-dependent positive for
  bulk/aggregate contraction recovery. **Do not commit Phase 4.2's
  budget-gated compute to the current approach without first reading
  the "PROJECT CLOSURE" entry at the end of `jwave_test/LOG.md`** — it
  lays out the full reasoning and the two legitimate paths forward
  (neither of which is a plain continuation of Phase 4 as scoped). See
  also `jwave_test/MANIFEST.md`'s closure section.
- **`jwave_ring_tomography/`** — Phase II, NEW active workspace (started
  2026-07-09, in direct response to `jwave_test/`'s closure). Explores
  a fundamentally different acquisition geometry: dense angular/ring
  coverage with a water-bath/full-surround setup (Butterfly/Midjourney-
  style whole-body scanner model, per explicit user decision — NOT the
  clinically standard anterior-arc-only transthoracic probe), and
  transmission+reflection tomography rather than sparse pulse-echo
  backprojection. Cited tissue properties are reused from `jwave_test/`
  unchanged; the sparse-probe backprojection/blind-discovery code is
  deliberately NOT reused (it's specific to the closed problem). Source
  of truth: `ring_tomography_phase_protocol.md` (root),
  `jwave_ring_tomography/LOG.md`, `jwave_ring_tomography/MANIFEST.md`.
  Requires its OWN Gate 2 collaborator signoff — `jwave_test`'s Gate 2
  does not carry over to this new geometry/medium.

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
