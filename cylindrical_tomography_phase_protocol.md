# Cylindrical (3D) ultrasound tomography — phase protocol

**A lab-notebook-style working protocol, adapted from
`ring_tomography_phase_protocol.md` (the `jwave_ring_tomography/`
protocol), which itself adapted `acoustic_simulation_phase_protocol.md`
(the `jwave_test/` protocol).**
Version 0.1 · Working draft

## Why this is a NEW project, not a continuation of `jwave_ring_tomography/`

`jwave_ring_tomography/` validated a real, working set of mechanisms in
2D (single cross-section): bent-ray/eikonal travel-time correction,
DAS reflectivity imaging (outer-boundary win, best RMSE 0.15mm), a new
backscatter/speckle channel, and a Doppler/motion-timing validation
with near-exact physical-slope agreement. See its `LOG.md` runs -28
through -44 for the full arc, and its final speckle-constrained-
selection result (run -43/-44) as the most recent validated win.

This project changes TWO things at once, each significant enough to
need its own Gate 2 review, not an extension of `jwave_ring_tomography`'s
existing (still-pending) signoff:

1. **Dimensionality**: 2D cross-section -> full 3D volume. A cylindrical
   scanner (a stack of rings, or a helical/distributed array of
   elements around a cylinder surrounding the torso) illuminates and
   receives from a true 3D anatomy, not one frozen slice.
2. **Anatomy**: a single 2D contour per cardiac phase -> a full 3D
   beating heart, reconstructed from real ACDC 3D volumes (10-slice
   short-axis stacks at ED/ES, `pilot/data/processed/ACDC/patientXXX.npz`)
   and Phase I's own 3D registration/displacement field
   (`pilot/data/processed/ACDC_reg/patientXXX.npz`) to interpolate a
   full 3D cardiac cycle, the 3D analog of `jwave_test`'s 2D
   motion-cycle construction.

Reused UNCHANGED from `jwave_ring_tomography/`: cited tissue properties
(`phase2_config.py`'s BLOOD/MYOCARDIUM/CHEST_WALL_PROXY/WATER, verified
directly against Mast 2000's primary source this session), the
established mechanism findings (DAS, speckle, bent-ray, Doppler-timing)
as hypotheses worth re-testing in 3D, not re-derived from scratch.

## Staged escalation (decided 2026-07-10): STATIC 3D before MOTION 3D

Per user: validate a single-timestamp (static) 3D blind reconstruction
on patient023 BEFORE escalating to the full beating (4D) cardiac cycle
-- motion 3D costs roughly (static 3D compute) x (number of cardiac
phases modeled), so proving the pipeline works statically first avoids
paying that multiplier on a still-unvalidated approach.

- **Phase 1 (this stage)**: ONE timestamp (ED, patient023 -- matching
  the rest of this repo's default patient/phase convention), a genuine
  3D volume (RV cavity / myocardium / LV cavity -- NOT a full anatomic
  4-chamber view; ACDC provides no atria segmentation, same 3-tissue
  convention as `jwave_ring_tomography` throughout), built from the
  real 10-slice stack (`pilot/data/processed/ACDC/patient023.npz`).
  Blind 3D reconstruction attempted with whatever channel(s)/methods
  the Phase 0 compute budget allows (DAS-style reflectivity imaging is
  the most directly portable candidate, per `jwave_ring_tomography`
  run -31's 2D result).
- **Phase 2+ (later, not yet started)**: escalate to the full 3D
  beating cycle using the real 3D `displacement_field`
  (`pilot/data/processed/ACDC_reg/patient023.npz`) once Phase 1's
  static case is validated -- NOT before.

## Phase 0 (scoping) — NOT YET PASSED, this is the open, current step

3D full-wave FDTD-style simulation (jWave) scales roughly with total
GRID POINT COUNT. Going from `jwave_ring_tomography`'s 2D domain
(300x300 = 90,000 points) to a matching-resolution 3D domain
(300x300x300 = 27,000,000 points) is a ~300x increase in points, and a
correspondingly large increase in per-timestep compute cost — this
could plausibly turn a few-second 2D simulation into a many-minutes-to-
hours 3D one. Every phase of every prior project in this repo has
required a REAL, empirically-measured compute estimate before
committing to a domain size or element count (e.g., `jwave_test`'s
Gate 1, `jwave_ring_tomography` run -01/-02's own explicit Phase 0.1)
— this project is no exception, arguably more so given the scale jump.

Phase 0 checklist (none checked yet):
- [ ] Empirical timing test: run ONE small 3D forward simulation, time
      it, extrapolate to a realistic domain size before choosing one.
- [ ] Decide domain size / grid resolution given that timing (may need
      a coarser dx than 2D's 0.1mm/cell if 3D is prohibitively
      expensive at that resolution — state the tradeoff honestly if so).
- [ ] Decide element layout: how many rings (z-positions) x how many
      elements per ring, informed by the above cost, not assumed.
- [ ] Confirm 3D data pipeline: build the interpolated 3D beating-heart
      medium from the real ACDC 3D volumes + registration field (the
      3D analog of `jwave_test`'s `mri_motion_cycle_*.npz`), verify
      voxel spacing / orientation are handled correctly (ACDC spacing
      is strongly ANISOTROPIC: 1.5625mm in-plane, 10mm slice thickness
      — a real, non-trivial resampling problem, not a free upgrade from
      the 2D pipeline).
- [ ] GPU vs. CPU: given the scale jump, CPU-only simulation (this
      session's default for 2D work) may not be practical at all for
      3D; budget for Colab-GPU-first, not CPU-first, this time.

## Gate 2 (acoustic-physics signoff) — separately required, does not
carry over from `jwave_ring_tomography`'s (also still-pending) one,
per this repo's standing rule that a new geometry/dimensionality needs
its own review.

## Rules carried over unchanged from `jwave_ring_tomography`/`jwave_test`

- Never invent tissue acoustic property values; every value needs a
  cited, verifiable source (this session verified Mast 2000 directly
  against the primary PDF, not just a secondary citation — same
  standard applies here).
- Fixed seeds everywhere; log the seed in this phase's `LOG.md`.
- Report an upfront compute-time estimate before starting any
  simulation batch, then proceed — but for 3D specifically, get that
  estimate from a REAL small-scale timing test first, not analogy to
  the 2D numbers (the scaling is too different to extrapolate safely
  without a data point).
- Append a LOG.md entry after every run (Appendix-C-style: phase, seed,
  config, dataset/split, result, gate pass/fail, observations, next
  action). Update MANIFEST.md when on-disk state changes materially.
- Not committed/pushed unless explicitly requested, per this repo's
  standing instruction.
