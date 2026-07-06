# Manifest

Snapshot of what's actually on disk. Update this when data/results/structure
changes materially — it's the fastest way for a new session (human or Claude)
to get oriented without re-deriving state from scratch.

_Last updated: 2026-07-06_

## Code (tracked in git)

| Path | Purpose |
|---|---|
| `cardiac_motion_pilot_protocol.md` | Source of truth protocol — phases + validation gates |
| `Cardiac_Motion_Pilot_Protocol.docx` | Original doc export of the protocol (reference only, prefer the .md) |
| `README.md` | Setup + orientation for humans and future sessions |
| `MANIFEST.md` | This file |
| `LOG.md` | Hypothesis, success criteria, phase checklist, per-run log — update every run |
| `requirements.txt` | Pinned dependencies |
| `.gitignore` | Excludes `data/`, `venv/`, caches |
| `src/__init__.py` | Package marker |
| `src/seed.py` | `set_all_seeds()` — call before any split/init/training |

`notebooks/` and `results/` exist but are empty as of this snapshot — no
Phase 2+ work has started yet.

## Data (gitignored — never committed, regenerate/redownload per-machine)

| Path | Contents | Size |
|---|---|---|
| `data/ACDC/ACDC/database/training/` | 100 ACDC patients, each with 4D cine volume, ED/ES frames + GT masks, `Info.cfg` | — |
| `data/ACDC/ACDC/database/testing/` | 50 ACDC patients, same structure | — |
| `data/ACDC/` (total) | | ~2.3 GB |
| `data/M&Ms/` | **Not yet downloaded** — Phase 1.2 step 2, to be added later | — |

Sanity-checked one sample: `patient001_frame01.nii.gz` shape `(216, 256, 10)`,
spacing `(1.5625, 1.5625, 10.0)` mm — matches its GT mask. Full Gate 1
overlay-and-inspect check across both cohorts is still outstanding.

## Environment

- Python 3.11.9, venv at `venv/` (gitignored, machine-local — recreate with
  `python -m venv venv && pip install -r requirements.txt`)
- Installed versions pinned in `requirements.txt`; exact resolved set is in
  `pip freeze` output, currently matches pins exactly except transitive deps
  (torch 2.0.1 is CPU-only build on this machine).

## Protocol phase status

See the checklist in `LOG.md` for the authoritative up-to-date state. As of
this snapshot: Phase 1 environment setup done; ACDC data in place; M&Ms not
yet acquired; no validation gates passed yet (Gate 1 overlay check not yet
run).
