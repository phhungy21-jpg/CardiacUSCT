# MANIFEST — jwave/ (Phase II, acoustic-simulation layer)

Tracks on-disk state for the acoustic-simulation phase so a new session
doesn't have to re-derive it by exploring the filesystem. Update this when
the directory structure or generated artifacts change materially.

## Layout

- `jwave/src/` — simulation, forward-model, and recovery code for this phase.
  Contains `toy_2d_homogeneous.py` and `toy_3d_homogeneous.py` — exploratory
  CPU smoke tests (point/sphere source in a homogeneous 1500 m/s medium,
  jWave's own published example) confirming the local jWave install
  produces physically sane wavefronts. Also `toy_2d_array_source.py` —
  16-element line array, plane-wave vs. delay-focused transmit, confirming
  jWave's `Sources` API and validating a geometric focusing delay law.
  Also `toy_2d_two_tissue_reflection.py` — blood/myocardium interface using
  **cited real tissue values** (Mast 2000, Table 1 — see jwave/LOG.md run
  2026-07-07-05), confirming correct reflection/transmission physics and
  surfacing a real finding: the blood-myocardium impedance contrast is
  genuinely weak (R~-0.0025), consistent with known clinical difficulty of
  endocardial border detection. None of this is the formal Gate 1 GPU
  reproduction (see notebooks/ below) or Phase 2's committed tissue model
  (this is a preview/starting reference for it, pending collaborator
  review).
- `jwave/venv/` — local CPU-only Python venv (jax 0.4.38, jwave 0.2.1) used
  for the toy scripts above. Gitignored (not tracked).
- `jwave/data/` — gitignored (matches root `.gitignore`'s `data/` pattern).
  Will hold simulated acoustic datasets (never raw patient data directly —
  reuses Phase I's ACDC/M&Ms-derived anatomy/motion as *input* to simulation,
  read from `../pilot/data/`).
- `jwave/results/` — gitignored artifacts excluded, figures/metrics/logs from
  this phase's runs go here.
- `jwave/notebooks/` — scratch/exploratory notebooks for this phase.
  Contains `phase1_gate1_reference_repro.ipynb` — Colab-ready notebook that
  reproduces jWave's own documented "Homogeneous Medium" reference example
  (https://ucl-bug.github.io/jwave/notebooks/ivp/homogeneous_medium.html) on
  a GPU runtime, per Gate 1. Not yet run — awaiting report-back of versions/
  GPU/timing/wavefield-sanity to log the Gate 1 result.
- `jwave/LOG.md` — running lab notebook for this phase (Appendix C entries).
- `jwave/requirements.txt` — this phase's pinned dependencies (JAX/jwave or
  k-Wave, decided in Phase 1.1). Independent from `pilot/requirements.txt`.

## Status

Phase 0 (scope/collaboration alignment) done — Gate 0 passed 2026-07-07
(see jwave/LOG.md), with a flagged caveat: physical-correctness ownership
for Gate 2 is shared/per-run rather than one named collaborator, and must
resolve to an actual reviewer before Gate 2. Phase 1 in progress: jWave
chosen as the toolkit; no local GPU on this machine, so Gate 1's reference
simulation runs on a cloud GPU (Colab) via
`jwave/notebooks/phase1_gate1_reference_repro.ipynb`, not yet executed.

**This directory (`jwave/`) is now frozen as the Phase 1 exploratory-scout
record as of 2026-07-07.** It was cloned to `../jwave_test/`, which is
where Phase 2 (acoustic model definition) work proceeds. Nothing here
should be treated as an established result — the scout runs (toy point
source, two-tissue reflection with cited blood/myocardium values, array
source) are informative but uncollaborator-reviewed, per Gate 2's signoff
requirement.

## Relationship to `pilot/`

This phase reuses Phase I's registration-derived anatomy and motion fields
(`pilot/data/processed/ACDC_reg/*.npz`, ground-truth-quality weights in
`pilot/results/phase3_quality_weights.csv`) as the moving-tissue input to the
acoustic forward model (protocol Phase 4.1). Phase I's ground-truth-quality
caveats (see `pilot/LIMITATIONS.md`) carry forward and must not be dropped
when interpreting Phase II results (protocol Gate 4).
