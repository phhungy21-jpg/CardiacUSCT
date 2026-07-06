# Cardiac Motion Pilot Study

Feasibility pilot testing whether a model trained to recover cardiac tissue
displacement from synthetic multi-angle Doppler-like velocity projections —
using MRI-derived ground-truth motion from one cohort — generalizes to a
different center/vendor's cohort.

Full protocol: [`cardiac_motion_pilot_protocol.md`](cardiac_motion_pilot_protocol.md).
This is a proxy study — no acoustic simulation, no real Doppler hardware, not
a clinical claim. See protocol Section 0 for exact scope.

## Status

See [`LOG.md`](LOG.md) for phase checklist and run history. See
[`MANIFEST.md`](MANIFEST.md) for what's currently on disk (data, code, results).

## Setup

```
python -m venv venv
venv\Scripts\Activate.ps1        # Windows PowerShell
pip install -r requirements.txt
```

Verify:

```
python -c "import SimpleITK, nibabel, numpy, scipy, sklearn, torch, matplotlib; print('ok')"
```

## Repository layout

```
data/          gitignored — ACDC / M&Ms datasets go here, never committed
src/           shared code (registration, projection, models, seeding)
notebooks/     exploratory / validation-gate notebooks
results/       generated metrics, figures, checkpoints (gitignored except summaries — see MANIFEST)
LOG.md         hypothesis, success criteria, phase checklist, per-run log
MANIFEST.md    inventory of what's on disk right now
```

## Working conventions

- Fixed seeds everywhere — use `src/seed.py::set_all_seeds()` before any
  split, init, or training run. Log the seed used in `LOG.md`.
- Every phase has a validation gate in the protocol — do not proceed past a
  gate that hasn't passed. Log gate pass/fail in `LOG.md`.
- M&Ms is touched exactly once (Phase 6). Do not evaluate against it while
  tuning on ACDC.
- Never commit patient data — `data/` is gitignored; double-check `git status`
  before staging if you've added new data paths.
