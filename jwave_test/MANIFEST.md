# MANIFEST — jwave_test/ (Phase II, acoustic-simulation layer — Phase 2 active workspace)

**Cloned from `../jwave/` on 2026-07-07** to begin Phase 2 (acoustic model
definition). `../jwave/` is now frozen as the Phase 1 exploratory-scout
record — do not treat its scout-run findings (e.g. the blood/myocardium
weak-contrast result) as established until a named acoustic-physics
collaborator reviews them (Gate 2 requirement). This directory is where
active Phase 2 work happens; content below is inherited from the clone and
is updated as Phase 2 proceeds.

Tracks on-disk state for the acoustic-simulation phase so a new session
doesn't have to re-derive it by exploring the filesystem. Update this when
the directory structure or generated artifacts change materially.

## Layout

- `jwave_test/src/` — simulation, forward-model, and recovery code for this
  phase. Inherited from `jwave/` at clone time: `toy_2d_homogeneous.py`,
  `toy_3d_homogeneous.py`, `toy_2d_array_source.py`,
  `toy_2d_two_tissue_reflection.py` (Phase 1 exploratory scout scripts —
  see `jwave/MANIFEST.md` for details, unchanged here). Phase 2 additions:
  `phase2_config.py` (cited tissue properties for blood/myocardium/
  chest-wall-proxy, anterior-array transducer geometry, grid/CFL — see
  its docstring for full citations) and `phase2_forward_model.py`
  (synthetic ring-phantom single-transmit demo, N=(300,300)/30mm domain,
  ran successfully with a documented stability check — see LOG.md run
  2026-07-07-06). Phase 3 additions: `phase3_config.py` (self-chosen toy
  cardiac-cycle motion parameters, not cited physiological values) and
  `phase3_motion_recovery.py` (per-frame frozen-scene motion injection,
  pitch-catch pulse-echo range recovery, envelope-based first-crossing
  detection, null test — see LOG.md run 2026-07-07-07 for two bugs found
  and fixed during validation, and the null-test result). See
  `PHASE2_TO_PHASE3_DIAGNOSTIC_HANDOFF.md` for the full pre-Phase-3
  diagnostic — item 4 (attenuation) remains an open, confirmed critical
  gap deferred to before Phase 4, per explicit user scoping decision that
  Phase 3's toy proof-of-concept doesn't require physical-realism items
  resolved, only the motion-injection risk (item 5, now null-tested).
  `labels.py` — shared `PENDING_SIGNOFF_BANNER`/`GT_FLOOR_CAPTION`/
  `TOY_EXACT_GT_CAPTION` applied to every figure/result in this codebase
  (see the handoff doc's "Phase 3→4 hard gate" section — Phase 4.2 dataset
  generation is blocked on collaborator signoff; Phase 4.1 prepare is not).
  Phase 4 addition: `phase4_benchmark.py` — Phase 4.1 benchmark-then-
  multiply, **first use of real Phase I anatomy** in this project
  (patient001, nearest-neighbor resampled per CLAUDE.md), timed at 3 small
  grid sizes and extrapolated to the real full-heart FOV (~91s, ~13.8GB
  per transmit at N~900 — infeasible on this CPU machine, confirming the
  protocol's GPU/cluster requirement for Phase 4.2). See LOG.md run
  2026-07-07-09. **Correction (run -11):** N=150/250 crops turned out to
  be entirely inside the LV cavity (no tissue boundary) — only N=350 has
  genuine heterogeneity; timing/memory numbers unaffected, but the "real
  anatomy" framing for those two points was overstated.
  `attenuation_solver.py` + `validate_attenuation.py` — a reimplementation
  of jWave's transient scan loop with per-step exponential absorption
  damping (both density AND velocity — a first version only damped
  density and was validated as ~2x wrong before the fix), validated
  against the analytic exp(-alpha*distance) law to ~0.1%. `calibration.py`
  — proposed (not collaborator-confirmed) source-amplitude calibration via
  FDA Mechanical Index. `phase4_demo_attenuating_real_anatomy.py` —
  combines real anatomy + validated attenuation + calibrated amplitude
  into one working forward model (N=350, stable, 0 NaN, Pa-scale output).
  See `PROXY_AUDIT.md` and LOG.md run 2026-07-07-11 for the full writeup.
- `jwave_test/venv/` — local CPU-only Python venv (jax 0.4.38, jwave 0.2.1),
  recreated fresh at clone time (not copied from `jwave/venv/`). Gitignored.
- `jwave_test/data/` — gitignored (matches root `.gitignore`'s `data/`
  pattern). Reuses Phase I's ACDC/M&Ms-derived anatomy/motion as *input* to
  simulation, read from `../pilot/data/`.
- `jwave_test/results/` — gitignored artifacts; figures/metrics/logs from
  this phase's runs go here.
- `jwave_test/notebooks/` — `phase1_gate1_reference_repro.ipynb` —
  **Gate 1 PASSED** (run 2026-07-07-10): executed on Colab (Tesla T4) via
  the VS Code Google Colab extension. jax 0.4.38, jwave 0.2.1, GPU
  confirmed (`CudaDevice`), timing 53.1ms±7.11ms/loop, wavefront visually
  confirmed circular. Notebook now includes two environment fixes baked
  in (jwave has no `__version__`; jax/CUDA-plugin version skew on Colab
  needing an explicit `jax[cuda12]==0.4.38` reinstall + runtime restart).
- `jwave_test/LOG.md` — running lab notebook for this phase (Appendix C
  entries), continuing from the cloned `jwave/LOG.md` history.
- `jwave_test/requirements.txt` — this phase's pinned dependencies.

## Status

Phase 0 (Gate 0) passed 2026-07-07 (see LOG.md), with a flagged caveat:
physical-correctness ownership for Gate 2 is shared/per-run rather than one
named collaborator — must resolve to an actual reviewer before Gate 2 can
pass. **Phase 1's Gate 1 PASSED 2026-07-07** (run -10, GPU-timed reference
reproduction on Colab Tesla T4: 53.1ms±7.11ms/loop, versions pinned in
`requirements.txt`) — note this is jWave's small 128×128 reference case,
not yet a real-anatomy-scale GPU number (that still needs to happen with
the collaborators, alongside the CPU-based extrapolation from run -09).
Phase 2
(acoustic model definition) is now starting in this directory: tissue
acoustic properties, transducer geometry, grid resolution/timestep, and
2D-first dimensionality decision, per protocol 2.1–2.2. Gate 2 requires a
named acoustic-physics collaborator's signoff — not passable solo.

**Phase 2.1/2.2 done (run 2026-07-07-06): a cited-value ring-phantom
forward model runs successfully.** `PHASE2_TO_PHASE3_DIAGNOSTIC_HANDOFF.md`
documents a confirmed critical gap — jWave's transient time-domain solver
(used throughout this project) does not implement attenuation at all,
despite cited attenuation values being present in the config — plus other
open items (points-per-wavelength margin, unverified PML at this config,
uncalibrated source amplitude). **This still blocks Phase 4** (the real
study) until resolved with the collaborators, independent of Gate 2's
standing signoff requirement.

**Phase 3.1/3.2 done (run 2026-07-07-07): toy simulate→recover loop with
null test.** Per explicit user decision, Phase 3 proceeded without
resolving the physical-realism diagnostic items (attenuation, calibrated
scaling) since a toy proof-of-concept doesn't need them — only the
motion-injection risk mattered, and it was verified via a null test
(zero-motion phantom through the identical pipeline: recovered-radius std
= 0.0000mm at zero noise, far below the 1.9mm true motion amplitude — no
spurious motion-correlated artifact). At zero noise, recovery clearly
beats a naive baseline (RMSE 0.286mm vs 0.740mm), satisfying Gate 3's core
criterion. Recovery is fragile to noise (cliff-edge degradation, not
gradual) with the current simple threshold detector — characterized
honestly, not treated as a blocker; a matched-filter detector would likely
be more robust, flagged as a future refinement. Two real bugs were found
and fixed during validation (truncated-toneburst ringing;
peak-vs-first-crossing echo detection) — see LOG.md for details.

**Phase 4.1 done (run 2026-07-07-09): benchmark-then-multiply on real
anatomy.** First use of real Phase I ACDC registration data
(`patient001.npz`) in this project, resampled to the acoustic grid via
nearest-neighbor (CLAUDE.md rule). Timed at N=(150,250,350): 0.81s/2.74s/
7.85s; extrapolated (log-log fit, exponent ~2.66) to the real full-heart
field of view (N~900): ~91s and ~13.8GB per transmit — confirms real-
anatomy, real-resolution simulation needs GPU/cluster (protocol Appendix
A), not this local CPU machine. A compute-budget formula (not filled in
with invented numbers) is ready to bring to the collaborators, alongside
Gate 1's still-outstanding GPU-timed reference reproduction. **Phase 4.2
(actual dataset generation) remains blocked** on collaborator signoff for
attenuation/scaling/staircasing, per the hard gate above — nothing past
this benchmark should be attempted solo.

**Proxy acoustic-physics audit done (run 2026-07-07-11, see
`PROXY_AUDIT.md`).** Per explicit user request ("proxy audit as an
expert... for this solo dev run") — **this is NOT Gate 2**, which
CLAUDE.md and the protocol both state cannot be passed by Claude or the
user alone. All 6 diagnostic items now have real numbers or fixes: PPW
checked (adequate), PML checked via full domain-crossing run (residual
-75dB, adequate), source scaling calibration proposed (FDA MI anchor,
0.316 MPa peak), **attenuation implemented and validated to ~0.1% against
the analytic law** (was the load-bearing gap; caught and fixed a 2x
under-damping bug along the way), motion injection already resolved
(Phase 3), staircasing quantified on real anatomy (0.59% relative field
difference, small but nonzero). `phase4_demo_attenuating_real_anatomy.py`
demonstrates all of this working together on real anatomy (N=350,
patient001, genuine tissue boundary). **Phase 4.2's real, 150-patient
dataset generation remains blocked on actual collaborator signoff** —
this audit substantially de-risks that eventual review but does not and
cannot substitute for it.

## Relationship to `pilot/`

This phase reuses Phase I's registration-derived anatomy and motion fields
(`pilot/data/processed/ACDC_reg/*.npz`, ground-truth-quality weights in
`pilot/results/phase3_quality_weights.csv`) as the moving-tissue input to the
acoustic forward model (protocol Phase 4.1). Phase I's ground-truth-quality
caveats (see `pilot/LIMITATIONS.md`) carry forward and must not be dropped
when interpreting Phase II results (protocol Gate 4).
