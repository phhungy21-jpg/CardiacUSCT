# MANIFEST — jwave_cylindrical_tomography/

Status tracker for this project's on-disk state. See `LOG.md` for the
full run-by-run reasoning; this file is the quick "what exists right
now" reference for a new session picking this project up.

## Status: Phase 0 CLOSED. Phase 1 (static 3D blind reconstruction) OPEN — first result IN: a real but MODEST positive (runs -10 through -13).

The first real multistatic 3D scan (96 elements x 2 media, run -09) came
back complete (96/96 both media). DAS reconstruction on it shows a
genuine, quantified positive: energy concentrates at the true outer
heart boundary shell (~1.7-2.3x higher than tissue-interior/exterior-
water regions; top-accumulator voxels sit ~4-5x closer to the true
boundary than a random baseline, confirmed at BOTH 60^3 and full N=150
reconstruction resolution, run -13) — the 3D analog of `jwave_ring_
tomography`'s own outer-boundary DAS success (run -31), confirmed to
generalize, not just assumed. Confirmed NOT an artifact of the probe-
ring direct-arrival artifact spotted in run -11.

**The actual fitted-surface RMSE is 10.27mm** (run -14, corrected —
runs -12/-13's original 7.65-7.84mm numbers had a real bug: `argmax`
over a truncated search window was silently clipping to the window
edge for ~20% of directions instead of finding a genuine local peak;
fixed with `scipy.signal.find_peaks`, see run -14). **31.7% of
directions have NO detectable boundary peak at all** — a genuine
failure mode, not just imprecision. Both numbers are MUCH cruder than
`jwave_ring_tomography`'s own real-anatomy result on the same patient
(1.04mm, run -32 there) and nowhere near the often-cited 0.15mm figure
(which was on an IDEALIZED synthetic phantom, NOT real anatomy — a
distinction worth remembering before citing that number again).
Resolution was ruled out as the cause (run -13: full N=150 gives
essentially the same result as the coarse 60^3 grid) — the gap is
structural (likely sparser 3D angular element coverage, more
near-element pairs excluded, the lower 0.8MHz frequency's longer
wavelength, and/or a systematic wrong-peak failure mode at large radii,
run -14), not yet resolved. Still untested: the harder blood/
myocardium INTERNAL boundary.

Phase 0's question (does 3D jWave simulation run in a practical amount
of compute?) is answered: **yes**, at this project's design point --
N=150, domain=130mm, dx=0.867mm, f0=0.8MHz, using jWave's `sensors` API
(NOT the default full-field recording, which was confirmed to blow GPU
memory at any realistic step count) -- confirmed on real Colab GPU
hardware with patient023's real heterogeneous anatomy: ~19s per single
forward simulation. GPU-only (CPU confirmed impractical, run -02).

Gate 2 (acoustic-physics collaborator signoff) for this project's own
setup (3D, cylindrical geometry, f0=0.8MHz) is separately required and
NOT YET DONE -- does not transfer from `jwave_ring_tomography`'s own
still-pending one.

## Design point (decided, runs -05/-07/-08)
- N=150 (isotropic grid, all 3 axes)
- domain = 130mm (heart bbox ~90mm + water-bath margin)
- dx = 0.867mm
- f0 = 0.8MHz (~2.2 points/wavelength -- a real, substantial reduction
  from `jwave_ring_tomography`'s 2.5MHz, justified by jWave/k-Wave's
  k-space-corrected scheme tolerating coarser grids than standard FDTD)
- Sensor-based (not full-field) recording is MANDATORY at this scale --
  receiver positions must be decided before running any simulation.

## Reused UNCHANGED from `jwave_ring_tomography/`
- `phase2_config.py`'s cited tissue properties (BLOOD, MYOCARDIUM,
  CHEST_WALL_PROXY, WATER) — verified this session directly against
  Mast (2000)'s primary source PDF, not re-derived. Frequency-
  independent in this simple model, so the f0 change doesn't need new
  citations (though attenuation IS frequency-dependent and not yet
  addressed for this project).
- The validated 2D mechanism findings (bent-ray/eikonal correction, DAS
  reflectivity imaging, backscatter/speckle channel, Doppler-timing
  validation, speckle-constrained candidate selection) as HYPOTHESES
  worth re-testing in 3D — not assumed to carry over automatically.
- The ACDC_LABEL_TO_TISSUE mapping convention (0=water, 1/3=blood,
  2=myocardium).

## Data pipeline, built and verified working
- `pilot/data/processed/ACDC/patient001.npz`, `patient023.npz`: real
  9/10-slice 3D volumes (`ed_frame`/`ed_mask`/`es_frame`/`es_mask`),
  spacing anisotropic (1.5625mm in-plane, 10mm slice thickness --
  CAUGHT a real axis-order bug here, run -04, fixed + regression-
  guarded with an assertion).
- `pilot/data/processed/ACDC_reg/patient001.npz`, `patient023.npz`: a
  real 3D `displacement_field` from Phase I's own registration work --
  usable for Phase 2+ (motion), NOT used yet (Phase 1 is static-only).
- `src/phase1_static_3d_medium.py`, `src/phase1_build_3d_medium.py`:
  crop-to-heart-bbox + nearest-neighbor resample pipeline, tested and
  working (patient023 ED: cropped 90x131x127mm, resampled to N=150^3,
  tissue fractions water=86.3%/RV=3.3%/MYO=6.6%/LV=3.7%, all physically
  sensible). Known, quantified bias: nearest-neighbor upsampling
  inflates tissue volume 5-13% vs. native resolution -- real, not a bug,
  worth remembering before treating resampled volumes as ground truth.
- `src/phase1_3d_medium_gpu_memory_test.py` + `notebooks/
  phase1_3d_medium_gpu_memory_test.ipynb`: CONFIRMED WORKING on real
  GPU hardware (run -08) -- builds the real heterogeneous medium AND
  runs a full real-duration simulation with sensor-based recording.
  This is the template to extend into an actual multistatic scan.

## NOT yet done
- Gate 2 signoff.
- Isolating whether the INNER blood/myocardium boundary shows any signal
  in this same dataset (only the outer tissue/water shell tested so far).
- Single-ring 36-element test (run -17/-18) DONE: RMSE=5.82mm at 36
  elements vs. 15.26mm at 12 elements (same z=68 slice) — confirms
  element density matters a lot (~2.6x improvement), but a genuine
  ~5.6x gap to 2D's validated 1.04mm remains even at matched element
  density — likely frequency (0.8MHz vs 2D's 2.5MHz), not yet isolated.
  Full 36/ring x 8-ring re-scan (~3hr GPU) not yet decided on.
- Phase 2+ (motion/beating-cycle) — deliberately deferred until Phase 1
  validates, per the user's own staged-escalation decision (run -04).

## Directory structure
```
jwave_cylindrical_tomography/
  LOG.md, MANIFEST.md
  src/
    phase0_3d_timing_test.py               -- CPU timing (impractical, informational)
    phase0_3d_gpu_timing_sweep.py          -- GPU timing sweep (superseded by run -07's finding)
    phase1_static_3d_medium.py             -- resampling pipeline, isolated test
    phase1_build_3d_medium.py              -- full crop+resample+tissue-map build
    phase1_3d_medium_gpu_memory_test.py    -- CONFIRMED WORKING real-GPU test (run -08)
    phase1_cylindrical_scanner.py          -- 96-element cylindrical layout (run -09)
    phase1_cylindrical_multistatic_scan.py -- full multistatic scan script (run -09, completed)
    phase1_das_3d_reconstruction.py        -- 3D DAS energy reconstruction (run -10)
    phase1_das_3d_evaluation.py            -- quantitative ground-truth comparison (run -10)
    phase1_das_3d_visualization.py         -- slice-by-slice comparison figure (run -11)
    phase1_das_3d_boundary_error_map.py    -- directional error map + pred-vs-true scatter (run -14)
    phase1_das_3d_mesh_reconstruction.py   -- lat-long-grid boundary extraction, mesh-ready (run -15)
    phase1_das_perring_2d_reconstruction.py -- same-ring-only 2D test, per run -16's hypothesis
    phase1_single_ring_36_multistatic_scan.py -- 36-element single-ring test scan (run -17)
  results/
    das_3d_reconstruction_patient023.npz   -- N=150 DAS accumulator (gitignored, derived from patient data)
    das_3d_evaluation_patient023.npz       -- ground-truth comparison arrays (gitignored)
    das_3d_boundary_extraction_patient023.npz  -- per-direction pred/true radii (gitignored)
    figures/
      phase1_das_3d_reconstruction_patient023.png  -- ground truth vs. DAS, 4 z-slices (run -11/-13)
      phase1_das_3d_boundary_error_map_patient023.png  -- directional error map (run -14)
  notebooks/        -- gitignored (embeds real ACDC patient023 data as base64)
../cylindrical_tomography_phase_protocol.md  -- protocol doc (root, matching
  jwave_ring_tomography's own root-level protocol convention)
```
