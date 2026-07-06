# Instructions for Claude Code sessions on this repo

This project follows `cardiac_motion_pilot_protocol.md` as the source of
truth. Read it (and `LOG.md` + `MANIFEST.md`) before doing substantive work —
they tell you which phase is active and what's already been validated.

## Rules that apply across sessions

- **Do not skip a validation gate.** Each protocol phase ends in a ⛔ gate.
  If asked to move to the next phase, check `LOG.md` first — if the current
  phase's gate isn't logged as passed, say so instead of proceeding.
- **M&Ms is evaluated exactly once**, in Phase 6, after the model is fully
  frozen from ACDC-only tuning. Never write code that evaluates against M&Ms
  before that point, even for "just checking."
- **Fixed seeds everywhere.** Use `src/seed.py::set_all_seeds()` before any
  split, model init, or training run. Log the seed in `LOG.md`.
- **Never commit patient data.** `data/` is gitignored — if you add a new
  data path outside `data/`, gitignore it too, and double check `git status`
  before staging.
- **After any run** (registration, training, evaluation), append an entry to
  `LOG.md` using the template in the protocol's Appendix C — phase, seed,
  config, dataset/split, result, gate pass/fail, observations, next action.
- **Update `MANIFEST.md`** when the on-disk state changes materially (new
  dataset added, results generated, directory structure changes) so the next
  session doesn't have to re-derive it by exploring the filesystem.
- Masks resample with **nearest-neighbor**, never linear/cubic — protocol
  Phase 2 pitfall, silently corrupts labels if violated.
- Keep models small given low N (protocol Phase 5) — a large architecture
  makes the cross-cohort generalization gap uninterpretable.
