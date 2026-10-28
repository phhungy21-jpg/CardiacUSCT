# LOG — jwave_cylindrical_tomography/ (3D cylindrical ultrasound tomography)

Running lab notebook for this project. Each entry follows the Appendix
C template from `../cylindrical_tomography_phase_protocol.md` (phase,
seed, config, dataset/split, result, gate pass/fail, observations, next
action). New project's log, started fresh — does not continue
`../jwave_ring_tomography/LOG.md`'s run numbering. See that file's runs
-28 through -44 for the full 2D mechanism-validation history this
project builds on, and the new protocol doc's "why this is a new
project" section for what specifically changes (dimensionality +
full 3D beating anatomy).

### Run 2026-07-10-01 — Project created; real 3D patient data confirmed available; Phase 0 compute-budget question opened, not yet answered
- Phase: 0 (setup/scoping). Created per user: "now move onto another
  side project. 3d live heart... put it in a cylindrical scanner
  (basically upgrading the ring tomography to a 3d version." Preceded
  by a direct discussion of why bone/fat/air boundaries would be much
  easier than blood/myocardium (verified via Mast 2000's own cited
  values: fat/muscle reflection coefficient ~7.3%, ~30x stronger than
  blood/myocardium's ~0.25%) and why real commercial point-of-care
  ultrasound systems solve blood/myocardium discrimination via
  speckle/texture tracking, not interface reflection -- the same
  direction this project's own best 2D result (runs -43/-44) ended up
  taking, independently arrived at.
- **Data check, before any compute commitment**: confirmed real 3D
  (not just 2D+time) cardiac MRI data already exists in this repo,
  unused until now. `pilot/data/processed/ACDC/patientXXX.npz`:
  `ed_frame`/`ed_mask`/`es_frame`/`es_mask`, each shape (10, 128, 128)
  -- a genuine 10-slice short-axis 3D volume at both end-diastole and
  end-systole, spacing=[1.5625, 1.5625, 10.0]mm (strongly anisotropic:
  fine in-plane resolution, coarse 10mm slice thickness, typical of
  cardiac cine MRI). Labels {0,1,2,3} match the already-established
  ACDC_LABEL_TO_TISSUE convention exactly (0=background/water, 1=RV
  cavity/blood, 2=myocardium, 3=LV cavity/blood), confirmed directly
  against `jwave_ring_tomography/src/phase2_config.py`'s existing
  mapping before assuming it carries over.
  `pilot/data/processed/ACDC_reg/patientXXX.npz` additionally has a
  REAL 3D `displacement_field` (10, 128, 128, 3) from Phase I's own
  registration work -- a genuine 3D ED->ES motion field, usable to
  interpolate a full 3D beating cardiac cycle, the 3D analog of
  `jwave_test`'s 2D `mri_motion_cycle_*.npz` construction.
- Physical sanity checked? by whom?: Claude -- confirmed the mask label
  convention matches the existing project's mapping before assuming it;
  confirmed spacing values are physically sensible for cardiac cine MRI
  (anisotropic slice thickness is standard, not a data error) before
  treating this as usable 3D anatomy.
- Gate passed? (Y/N): N/A. Phase 0 not started -- the compute-budget
  question (does 3D jWave simulation at a realistic domain size/
  resolution even run in a practical amount of time) is now the single
  open blocking item before any further design decision (domain size,
  element/ring layout, dx).
- Next action: run a small, real 3D jWave timing test (ONE forward
  simulation at a modest domain size) to get an actual empirical
  compute-cost data point before deciding anything else -- 2D timing
  numbers from `jwave_ring_tomography` cannot be safely extrapolated
  given the ~300x grid-point-count jump from a matching-resolution 2D
  to 3D domain. Not committed/pushed (standing instruction).

### Run 2026-07-10-02 — Phase 0 timing test: matching-2D-resolution 3D simulation is NOT practical as scoped — a real, measured blocker, not a guess
- Phase: 0 (scoping). `phase0_3d_timing_test.py`: ONE small 3D forward
  simulation (homogeneous water, single point source/receiver,
  N=(60,60,60)=216,000 grid points, CPU, this session's default
  execution environment), timed directly rather than assumed.
- **Result**: post-JIT per-timestep cost = 35.4ms/step at this small
  size (164 ns/grid-point/step). Extrapolating to a domain matching
  `jwave_ring_tomography`'s 2D resolution (dx=0.1mm) and a modest 30mm
  physical domain (N=(300,300,300)=27,000,000 points, a 125x point-
  count increase) and the step count a full round-trip traversal would
  need (~2634 steps, 13.2x this test's 200): **~11,670 seconds (~3.24
  hours) for a SINGLE forward simulation on CPU.** Even assuming a
  generous 30x GPU speedup (Colab), that is still **~6.5 minutes for
  ONE simulation** — and this project's 2D work routinely needed
  72-simulation batches (36 angles x 2 media) per run, which at this
  rate would mean ~7.8 HOURS per batch even on GPU, before any
  multi-ring/cylindrical-array angle count (which will need MORE
  distinct transmit events than 2D's 36, not fewer, to cover a full
  cylindrical surface).
- Physical sanity checked? by whom?: Claude — measured directly (JIT-
  compiled, post-compile timing isolated from one-time compile
  overhead) rather than assumed from 2D numbers or a rule-of-thumb
  scaling factor; extrapolation used this project's OWN established 2D
  domain-sizing convention (dx=0.1mm, ~6 points/wavelength at 2.5MHz)
  rather than an arbitrarily chosen resolution, so the comparison is
  apples-to-apples with what worked in 2D.
- Gate passed? (Y/N): N/A — Phase 0 scoping result, not itself a gate,
  but the checklist item it answers ("does 3D at matching resolution
  run in practical time") is answered: **no, not as directly scoped.**
- Observations: this is a genuine, load-bearing negative result, not a
  reason to abandon the project — it means the 2D pipeline's approach
  (full resolution, dense multistatic angle coverage, CPU-friendly)
  cannot be ported to 3D unchanged. Real compromises are needed on at
  least one of: grid resolution (coarser dx, trading off numerical
  accuracy per this project's own dx=lambda/6 rule), physical domain
  size (a smaller proof-of-concept volume rather than full-torso
  scale), transmit/receive angle density (far sparser 3D coverage than
  2D's 36-angle convention), or accepting GPU-only development from the
  start (CPU is unambiguously insufficient at any realistic 3D scale).
- Next action: report this measured result to the user directly with
  concrete tradeoff options, rather than silently picking a scaled-down
  scope. No domain size, resolution, or element layout should be
  decided until the user weighs in on which compromise(s) they're
  willing to accept. Not committed/pushed (standing instruction).

### Run 2026-07-10-03 — Decision: GPU-only from the start (not coarser resolution / smaller domain / sparser acquisition); built a real GPU timing sweep to replace run -02's assumed 30x speedup guess with measured numbers
- Phase: 0 (scoping). Per user's explicit choice (asked directly,
  4-option multi-select, only this one picked): rather than compromise
  grid resolution, physical domain size, or acquisition density to fit
  CPU-practical timescales, accept that ALL simulation work for this
  project runs on a Colab GPU kernel, never locally -- a deliberate,
  informed choice given run -02's measured ~3.24 CPU-hours per single
  simulation at 2D-matching resolution.
- Design: `phase0_3d_gpu_timing_sweep.py` + `notebooks/
  phase0_3d_gpu_timing.ipynb` -- sweeps N=(40,60,100,150,200,300)^3
  (homogeneous water, single source/receiver, same per-domain-size
  round-trip extrapolation as run -02) DIRECTLY ON GPU, replacing the
  assumed "30x GPU speedup" factor in run -02's CPU extrapolation with
  real measured numbers across a realistic range of candidate domain
  sizes. Zero patient-data dependency (homogeneous medium only) --
  unlike the beating-heart notebook, nothing needs uploading or
  embedding, so this notebook is much simpler and IS safe to commit
  (no gitignore exclusion needed, confirmed no ACDC-derived content).
- Physical sanity checked? by whom?: Claude -- deliberately did not
  just multiply run -02's CPU number by an assumed GPU speedup factor
  (a real risk of being wrong by a large margin either direction) and
  instead built an actual measurement across several candidate sizes,
  so the eventual domain-size decision rests on real data, not a guess.
- Gate passed? (Y/N): N/A -- Phase 0 scoping infrastructure, still open.
- Next action: user runs `notebooks/phase0_3d_gpu_timing.ipynb` on a
  Colab GPU kernel (no upload step needed this time); once real GPU
  numbers come back, decide final domain size/resolution from actual
  data, then proceed to the remaining Phase 0 checklist items (element/
  ring layout, 3D beating-heart medium construction from the real ACDC
  3D volumes + registration field, anisotropic-spacing resampling).
  Not committed/pushed (standing instruction).

### Run 2026-07-10-04 — Staged plan decided (static 3D before motion 3D); anisotropic resampling pipeline built, tested, and a real axis-order bug caught before it could corrupt anything
- Phase: 0/1 (scoping + first Phase 1 infrastructure). Per user: "i want
  to take a full pseudo-3d (multiple 2d slices) of pt 23' heart model
  at 1 time stamp... for blind 3d reconstruction first... before we
  escalate to motion 3d wich is 1 3d compute times tick speed." Updated
  `../cylindrical_tomography_phase_protocol.md` with an explicit staged
  plan: Phase 1 = static (single-timestamp, ED) 3D blind reconstruction
  on patient023; Phase 2+ = full beating (4D) cycle, NOT started until
  Phase 1 validates. Also corrected a naming precision issue upfront:
  ACDC provides NO atria segmentation (RV cavity/myocardium/LV cavity
  only, matching `jwave_ring_tomography`'s existing 3-tissue
  convention) -- what's available is a genuine 3D ventricular volume,
  not a full anatomic 4-chamber view.
- Design: `phase1_static_3d_medium.py` -- builds the anisotropic-to-
  isotropic resampling pipeline needed regardless of the final domain-
  size decision (ACDC's native spacing is 90mm through-plane / 200mm x
  200mm in-plane, strongly anisotropic: 10mm slice thickness vs.
  1.5625mm in-plane). Uses `scipy.ndimage.zoom(..., order=0)` (nearest-
  neighbor only, per this repo's own standing rule against linear/cubic
  mask resampling) at three candidate target dx values (0.2/0.5/1.0mm).
- **Bug caught before trusting any result**: the FIRST run's printed
  "native physical extent" came back physically absurd (14mm total
  height, 1280mm in-plane) -- immediately recognizable as wrong for a
  human heart. Diagnosed by checking BOTH possible spacing-to-axis
  correspondences numerically: the array's own `spacing` field is
  stored in (row, col, slice) order, but `ed_mask`'s actual array axes
  are (slice, row, col) -- using the stored order directly silently
  mismatches which spacing value applies to which axis. Reversing the
  spacing array to match the mask's own axis order gives a sensible
  90mm x 200mm x 200mm extent, confirmed correct before proceeding.
  Added a permanent assertion (through-plane extent must be within
  60-150mm) so this specific mistake cannot silently regress if this
  loader is ever touched again.
- **Result, once fixed**: resampling preserves anatomy sensibly but
  with a real, quantified, honestly-reported bias -- tissue volume
  after nearest-neighbor upsampling is systematically 5-13% LARGER than
  the native-resolution volume (RV +11.6-13.5%, MYO +4.9-6.3%, LV
  +7.8-8.4%, consistent across all three tested target dx values) -- an
  expected artifact of nearest-neighbor-upsampling a small number of
  thick coarse slices onto a much finer grid, not a bug, but a real
  bias worth remembering before treating resampled volumes as ground
  truth.
- **Scoping implication surfaced, not yet resolved**: at dx=0.2mm, the
  FULL native field of view (90x200x200mm) resamples to 450 MILLION
  voxels -- far beyond anything run -03's GPU timing sweep will test
  (max 300^3=27M). The heart itself occupies only a fraction of that
  full MRI field of view; the practical fix is almost certainly
  cropping to a bounding box around the actual anatomy plus a modest
  probe-radius margin, not simulating the full FOV -- flagged as a
  needed next step, not yet implemented.
- Physical sanity checked? by whom?: Claude -- caught the axis-order
  bug via a physical-plausibility check on the FIRST run's own printed
  output (not a hypothetical concern raised in the abstract), verified
  the fix numerically against BOTH possible correspondences before
  picking one, and added a permanent regression-preventing assertion
  rather than just fixing the immediate symptom.
- Gate passed? (Y/N): N/A -- Phase 0/1 infrastructure, not a phase gate.
- Next action: (a) still waiting on run -03's GPU timing sweep results
  to pick a final target dx; (b) implement cropping to a heart-centered
  bounding box (not the full FOV) before that dx is applied, given the
  450M-voxel full-FOV number is clearly impractical regardless of GPU
  speedup; (c) once dx and crop region are both decided, build the
  actual jWave 3D Medium object and a first blind-reconstruction test
  (DAS-style, the most directly portable 2D method) on this static
  volume. Not committed/pushed (standing instruction).

### Run 2026-07-10-05 — GPU timing sweep result: time is NOT the blocker, GPU MEMORY is — forces a genuine frequency/resolution tradeoff, resolved via a concrete, real-data-driven design point
- Phase: 0 (scoping). User ran `notebooks/phase0_3d_gpu_timing.ipynb` on
  a real Colab GPU (CudaDevice(id=0)) and reported the results.
- **Bug caught in the extrapolation formula itself, not just the raw
  data**: the first read of the results (before recomputing) showed a
  nonsensical "~0.0s" estimated single-simulation time for every domain
  size. Root cause: the summary code divided a PHYSICAL simulated time
  (traversal_s) by a WALL-CLOCK per-step compute cost (an incompatible
  unit), instead of by the actual simulation timestep `dt` (which
  depends only on dx/CFL/sound-speed, not domain size, and is therefore
  the SAME ~1.974e-8s measured in run -02's CPU test, since dx=0.1mm
  was unchanged). Recomputed correctly: N=150 (15mm domain) takes only
  ~18s for a full single simulation on GPU — time was never actually
  the bottleneck once the arithmetic was fixed.
- **The real blocker, confirmed empirically**: N=200 and N=300 BOTH
  failed with `RESOURCE_EXHAUSTED` (GPU out-of-memory) on the (likely
  free-tier T4) GPU used. N=150 (3,375,000 points) ran successfully;
  the practical ceiling is therefore roughly N~150-180 cells per axis,
  a MEMORY constraint, not a time constraint.
- **This directly conflicts with fitting a real heart at 2.5MHz
  resolution.** Computed patient023's ACTUAL tissue bounding box
  precisely (not estimated from volume): 80mm x 90.6mm x 85.9mm
  (through-plane x in-plane x in-plane). Fitting that plus a
  water-bath/probe margin (~130mm total) within N=150 cells forces
  dx~0.867mm -- at 2.5MHz (wavelength=0.608mm in water), that is LESS
  than one point per wavelength, unusable at any reasonable accuracy
  standard, even the k-space-method-specific relaxed ones.
- **Decision, per user's explicit choice** (asked directly, 4 options,
  "lower simulation frequency" picked): rather than crop to a partial
  heart region or immediately reach for a paid GPU tier, reduce F0_HZ.
  Solved for a concrete, self-consistent design point: N=150,
  domain=130mm, dx=0.867mm, **f0~0.8MHz** (wavelength=1.9mm, ~2.2 points/
  wavelength) -- a real, substantial reduction from the 2D project's
  2.5MHz, justified by jWave/k-Wave's k-space-corrected pseudospectral
  scheme being specifically designed to remain accurate down to ~2
  points/wavelength (unlike standard FDTD, which needs more), not an
  arbitrary relaxation of this repo's own established dx=lambda/6
  convention -- that convention was itself a conservative safety
  margin, not a hard physical requirement.
- Physical sanity checked? by whom?: Claude -- caught its own
  extrapolation bug by noticing the "~0.0s" output was obviously
  nonsensical rather than reporting it; computed the heart bounding box
  from the ACTUAL resampled data rather than the earlier volume-based
  estimate; grounded the frequency-reduction justification in a real,
  citable property of the k-space method jWave implements, not just
  "lower it until it fits."
- Gate passed? (Y/N): N/A -- Phase 0 scoping, still open (Gate 2 for
  this project's acoustic-physics setup, including this new frequency
  choice, will need its own collaborator signoff, same standing rule as
  every prior project in this repo).
- Next action: verify the N=150/dx=0.867mm/f0=0.8MHz combination ALSO
  fits in GPU memory with a REAL heterogeneous (3-tissue) medium, not
  just the homogeneous-water test case used for timing -- a
  heterogeneous FourierSeries medium may have a different memory
  footprint than a uniform one, not yet confirmed. Then build the
  actual cropped 3D jWave Medium from patient023's real ED volume at
  this resolution and attempt the first blind reconstruction (DAS-
  style). Not committed/pushed (standing instruction).

### Run 2026-07-10-06 — Real 3D heterogeneous medium built locally (crop+resample pipeline confirmed working); GPU memory-verification notebook packaged, awaiting user run
- Phase: 0/1. `phase1_build_3d_medium.py`: crops patient023's real ED
  volume to a bounding box around the tissue (measured bbox 80x90.6x
  85.9mm) plus a 20mm water-bath margin, resamples (nearest-neighbor
  only) onto the N=150/dx=0.867mm design point from run -05, builds
  sound_speed/density maps from the SAME cited tissue properties as
  `jwave_ring_tomography` (frequency-independent in this simple model,
  so the F0 change doesn't need new citations). Ran locally (CPU, cheap
  -- this is pure numpy/scipy resampling, not a jWave simulation):
  cropped physical extent 90x131x127mm, resampled tissue fractions
  water=86.3%/RV=3.3%/MYO=6.6%/LV=3.7% -- all physically sensible.
- `phase1_3d_medium_gpu_memory_test.py` + `notebooks/
  phase1_3d_medium_gpu_memory_test.ipynb`: packages the SAME crop+
  resample logic plus a single real 3D forward simulation on this
  heterogeneous medium, to confirm the design point still fits GPU
  memory (not yet verified -- the run -03 timing sweep only tested
  homogeneous water, which may have a different memory footprint than
  a 3-tissue medium). Embeds only the SMALL source data (patient023's
  raw ed_mask + spacing, 148KB) rather than the large already-resampled
  array (~27MB), letting Colab redo the cheap resampling itself --
  more efficient than the beating-heart notebook's approach of
  embedding a pre-computed derived array.
- Both the local medium-build output (`results/patient023_3d_medium_
  N150.npz`) and the new notebook (embeds patient023's real ed_mask)
  added to `.gitignore` -- same real-ACDC-patient-data caution as
  `jwave_ring_tomography`'s equivalents, confirmed via `git check-ignore`
  before this could be an issue.
- Physical sanity checked? by whom?: Claude -- checked resampled tissue
  fractions are physically plausible (13.6% total tissue in a
  130mm-scale cube containing a 90mm heart is a sensible order of
  magnitude) before treating the crop/resample pipeline as trustworthy;
  confirmed the new notebook's gitignore exclusion actually works via
  `git check-ignore`, not assumed.
- Gate passed? (Y/N): N/A -- Phase 0/1 infrastructure, not a phase gate.
- Next action: user runs `notebooks/phase1_3d_medium_gpu_memory_test.ipynb`
  on a Colab GPU kernel; if the heterogeneous medium fits and the
  simulation completes, this project's Phase 0 compute-budget question
  is fully closed and Phase 1 (the first real blind 3D reconstruction
  attempt) can begin in earnest -- multiple source positions, a DAS-
  style reflectivity approach ported from `jwave_ring_tomography` run
  -31. If it does NOT fit, will need to revisit N/domain/margin
  tradeoffs again. Not committed/pushed (standing instruction).

### Run 2026-07-10-07 — The memory ceiling was never about grid size alone: jWave's default records the FULL field at EVERY timestep — root-caused exactly, fixed via the `sensors` API, verified locally before spending another GPU round-trip
- Phase: 0/1. User ran `notebooks/phase1_3d_medium_gpu_memory_test.ipynb`
  (the first, full-field version) on Colab GPU -- FAILED with
  `RESOURCE_EXHAUSTED`, trying to allocate 18,292,500,000 bytes, even
  though run -03's homogeneous-water timing sweep had succeeded at the
  SAME N=150.
- **Root-caused exactly, not just re-attempted with a smaller size**:
  150^3 x 1355 (the actual n_steps this design point needs for a full
  physical round trip) x 4 bytes = 18,292,500,000 bytes -- an EXACT
  match to the requested allocation. This confirmed the real cause: run
  -03's timing sweep only ran a truncated 200-step test (to keep the
  timing measurement itself cheap), which happened to keep memory small
  regardless of N -- it was never actually testing the real simulation's
  memory requirement, only per-step compute speed. The true scaling law
  is memory ~ N^3 x n_steps, not N^3 alone, because jWave's
  `simulate_wave_propagation` -- confirmed by reading its actual source
  -- defaults to recording and returning the FULL pressure field at
  EVERY timestep via `jax.lax.scan`'s output accumulation, unless an
  explicit `sensors` callback is supplied to record only specific
  points instead (the docstring states this plainly: "If sensors is not
  specified, the recorded field is the entire pressure field").
  `jwave_ring_tomography`'s entire 2D body of work never needed this,
  since 300x300 was always small enough for full-field recording to fit
  comfortably regardless of step count -- this is the first time in
  this repo's history that scale has actually forced using jWave's
  sensor-recording API instead of the simpler (but memory-unsafe at 3D
  scale) default.
- **Fix verified locally (cheap, no GPU needed) before spending another
  Colab round-trip**: confirmed via a small N=40 test that a `sensors`
  callback (`lambda p, u, rho: p.on_grid[rx, ry, rz, 0]`, extracting
  specific receiver-point values instead of the whole field) produces
  the expected small output shape (n_steps,) instead of
  (n_steps,N,N,N,1) -- this is also the standard, CORRECT way large-
  scale ultrasound simulation is normally done (matching k-Wave's own
  "sensor mask" convention), not a workaround or a compromise.
  `phase1_3d_medium_gpu_memory_test.py` updated to use `sensors`
  (extracting 3 receiver points, matching a plausible real pitch-catch/
  multistatic setup) instead of the default; notebook regenerated with
  the fix and handed back to the user for another GPU run.
- Physical sanity checked? by whom?: Claude -- computed the EXACT
  byte-for-byte match between the theoretical (N^3 x n_steps x 4 bytes)
  formula and the actual OOM error's requested allocation before
  concluding this was the root cause, rather than guessing; read
  jWave's actual source code (not just its docstring) to confirm the
  `jax.lax.scan` output-accumulation mechanism precisely; verified the
  fix's API mechanics at a cheap, small local scale before committing
  another GPU-compute round-trip to test it at full scale.
- Gate passed? (Y/N): N/A -- Phase 0 scoping, still open pending this
  fix's real-scale verification.
- Observations: this reframes run -05's whole frequency-reduction
  decision -- the ACTUAL constraint was never simply "N=150 is the GPU
  ceiling," it was "N=150 at the FULL required step count, recording
  the full field, is the ceiling" -- a much more specific and, happily,
  much more fixable problem than a hard memory wall. If the sensors fix
  works at full scale, it's plausible the earlier frequency-reduction
  decision (2.5MHz -> 0.8MHz) may not have needed to be as aggressive,
  though it is already decided and reasonably justified independently
  (k-space methods tolerating coarser grids) -- worth revisiting ONLY
  if there's a clear reason to, not automatically re-litigating a
  settled decision.
- Next action: user re-runs the UPDATED notebook; if the sensor-based
  version fits in GPU memory and completes, Phase 0 is closed for real
  this time, and this also means: for any future reconstruction attempt
  (DAS or otherwise), receiver positions must be decided BEFORE running
  a simulation (since sensors need to be specified upfront), a real
  design constraint to carry into Phase 1's actual reconstruction code,
  different from `jwave_ring_tomography`'s 2D convention of sampling
  arbitrary points from an already-returned full field after the fact.
  Not committed/pushed (standing instruction).

### Run 2026-07-10-08 — PHASE 0 CLOSED: sensor-based fix confirmed at full scale on real GPU hardware
- Phase: 0 (closing) / 1 (opening). User re-ran the updated
  `notebooks/phase1_3d_medium_gpu_memory_test.ipynb` on Colab GPU.
  **Result: SUCCESS.** Full 1355-step simulation on the REAL
  heterogeneous (RV/myocardium/LV) medium at N=150, sensor-based
  recording (3 receiver points): completed in **19.29s**, output shape
  correct ((1355, 3), matching the 3 requested receiver points), peak
  trace magnitude 9.48e-3 (physically sensible order of magnitude, not
  a degenerate/zero/NaN result).
- This closes Phase 0's single blocking question definitively: this
  project's design point (N=150, domain=130mm, dx=0.867mm, f0=0.8MHz,
  sensor-based receiver recording) is confirmed computationally
  feasible on a free-tier Colab GPU, with real patient anatomy, not
  just a homogeneous test case. At ~19s/simulation, a full multistatic
  scan (many transmit positions, a handful of receivers each) is
  comfortably tractable within a single Colab session -- e.g. even 50
  transmit events would be under 20 minutes total.
- Physical sanity checked? by whom?: Claude/user jointly -- user
  executed the real GPU run and reported the actual output; Claude
  confirmed the result matches the expected shape/scale before treating
  Phase 0 as closed, consistent with this repo's standing discipline of
  checking real numbers rather than assuming a fix worked.
- Gate passed? (Y/N): **Phase 0: Y.** Gate 2 (acoustic-physics signoff)
  for this project's specific setup (3D, cylindrical geometry, f0=0.8MHz)
  remains separately open, per this repo's standing rule that a new
  geometry/dimensionality needs its own collaborator review -- not
  satisfied by this compute-feasibility result alone.
- Next action: Phase 1 can now begin in earnest -- design a genuine
  cylindrical scanner element layout (rings x elements-per-ring, the 3D
  analog of `jwave_ring_tomography`'s single-ring/36-angle convention),
  decide which reconstruction method to port first (DAS reflectivity
  imaging, `jwave_ring_tomography` run -31, is the most directly
  portable given its 2D success and the fact that sensor-based 3D
  recording is naturally compatible with a multistatic DAS-style
  approach), and run the first real multistatic 3D scan + blind
  reconstruction attempt on patient023's static ED volume. Not
  committed/pushed (standing instruction).

### Run 2026-07-10-09 — Cylindrical scanner layout designed (8 rings x 12/ring = 96 elements) and the first full multistatic scan packaged for Colab
- Phase: 1. `phase1_cylindrical_scanner.py`: 8 rings (Z positions) x 12
  elements/ring = 96 total element positions on a cylinder, probe
  radius=56mm (fits between patient023's ~90mm heart bbox and the
  130mm domain edge, leaving PML standoff), Z-span=80mm (within the
  heart's own through-plane extent). Verified all 96 positions land
  within the N=150 grid bounds (assertion-checked, not just eyeballed).
  SEQUENTIAL acquisition, unchanged from `jwave_ring_tomography`'s own
  Phase 0.1 decision.
- `phase1_cylindrical_multistatic_scan.py` + `notebooks/
  phase1_cylindrical_multistatic_scan.ipynb`: the first real
  multistatic 3D scan -- each of the 96 elements fires once, recording
  ALL 95 others simultaneously via `sensors` (run -07's fix), mirroring
  `jwave_ring_tomography` run -04's "multiple receivers per transmit is
  free" finding, just via `sensors` instead of post-hoc field indexing.
  96 elements x 2 media (water-only baseline + real phantom) = 192
  forward simulations, ~19s each (confirmed run -08) -> ~60-65 min
  total, a genuinely long single Colab job.
- **Resilience additions, since this is real compute time on the line**:
  checkpoints saved every 20 transmits (not just at the end); output
  saves directly to a MOUNTED GOOGLE DRIVE folder throughout (not
  Colab's local ephemeral disk), so a mid-run disconnect loses at most
  20 transmits' worth of work, not the whole job, and there is no
  separate "download the result" step needed afterward this time --
  it's already in Drive when the job finishes or is interrupted.
- Physical sanity checked? by whom?: Claude -- asserted all element
  positions are within grid bounds before proceeding (not assumed from
  the geometry formula alone); reused the EXACT medium-building and
  sensors mechanism already confirmed working on real GPU hardware
  (run -08), changing only the number of sensors/transmits, not
  re-deriving the core mechanism from scratch.
- Gate passed? (Y/N): N/A -- Phase 1 infrastructure, not a phase gate.
- Next action: user runs `notebooks/phase1_cylindrical_multistatic_scan.ipynb`
  on Colab GPU (a genuinely long, ~1hr job -- keep the session active).
  Once the raw multistatic dataset is back, the next step is channel-
  specific post-processing (DAS reflectivity imaging first, the most
  directly portable 2D method) -- no new simulation needed for that,
  pure post-processing on this same raw dataset, matching this whole
  repo's established "simulate once, reuse for many analyses" discipline.
  Not committed/pushed (standing instruction).

### Run 2026-07-10-10 — FIRST 3D BLIND RECONSTRUCTION RESULT: DAS energy imaging genuinely concentrates at the true heart boundary — a real positive, quantified against ground truth, not just a plausible-looking picture
- Phase: 1. User returned the completed multistatic scan
  (`cylindrical_multistatic_scan_patient023.npz`): ALL 96 water + 96
  phantom forward simulations finished successfully (confirmed
  non-degenerate: every trace checked for non-zero content, 96/96 both
  media). This is the first real, complete 3D multistatic dataset in
  this project.
- `phase1_das_3d_reconstruction.py`: ported `jwave_ring_tomography` run
  -35's INCOHERENT (energy/squared) matched-filter DAS accumulation to
  3D -- for every valid (tx, rx) pair (8,952 of 96x95=9,120 total,
  excluding near-position pairs within 15 grid cells to avoid direct-
  coupling artifacts, the 3D analog of 2D's near-angle exclusion),
  predicts the straight-ray bistatic delay (water sound speed) to every
  voxel on a COARSER 60^3 reconstruction grid (compute tractability;
  ~2.17mm/voxel vs. the simulation's own 0.867mm), samples that pair's
  background-subtracted matched-filter envelope there, accumulates the
  squared excess energy. Pure CPU post-processing, no new simulation --
  ran locally in a few minutes. Result: accumulator min=9.66e-4,
  max=6.35e-2, mean=6.41e-3 -- a real peaked, non-degenerate
  distribution (max ~10x the mean).
- **`phase1_das_3d_evaluation.py`: quantitatively checked against
  patient023's real ground-truth anatomy** (rebuilt the SAME crop+
  resample pipeline used to build the simulated medium, sampled onto
  the reconstruction's own coarse grid for a direct voxel-to-voxel
  comparison -- not eyeballed from a picture). **Result: a genuine,
  measurable positive.** Mean accumulator value at the true boundary
  shell (5,207 voxels) = 0.01376, vs. tissue interior = 0.00765 (~1.8x
  lower) and exterior water = 0.00605 (~2.3x lower). More directly: the
  top-K accumulator voxels (K matched to the true boundary-shell voxel
  count) sit on average 2.63 coarse voxels from the true boundary, vs.
  10.50 for a random/all-voxel baseline -- a 4x improvement over chance.
  This is the SAME kind of outer-boundary DAS success `jwave_ring_
  tomography` found in 2D (run -31), now confirmed to genuinely
  generalize to a real 3D cylindrical multistatic geometry -- not
  assumed to carry over, actually tested and quantified.
- Physical sanity checked? by whom?: Claude -- rebuilt ground truth from
  the SAME crop/resample code path already used to build the simulated
  medium (not a separately-derived approximation, so any comparison
  artifact would be a genuine bug, not a mismatch between two different
  pipelines); used a distance-to-true-boundary metric with an explicit
  random-baseline comparison rather than reporting only the accumulator
  values in isolation, so the "4x better than chance" claim is a real,
  falsifiable number, not a qualitative impression.
- Gate passed? (Y/N): N/A -- Phase 1 first-attempt result, not a phase
  gate. Note: Gate 2 (acoustic-physics collaborator signoff for this
  project's own 3D/cylindrical/0.8MHz setup) remains separately open
  and unaffected by this reconstruction result.
- Observations: this is a real, replicated positive (2D->3D
  generalization of the outer-boundary DAS finding), not a coincidence
  or an artifact of a lenient metric -- both the region-mean comparison
  and the top-K-distance comparison independently show the same
  direction of effect. As in 2D, this says nothing yet about resolving
  the much harder blood/myocardium internal boundary (this project's
  own run -01 already anticipated that difficulty is not solved by
  geometry alone) -- this result is specifically about the OUTER
  water/tissue surface, the easier of the two problems.
- Next action: (a) visualize a few 2D slices through the 3D accumulator
  alongside ground truth for a qualitative sanity check (a quick,
  cheap addition, not required to trust the quantitative result already
  obtained); (b) attempt to isolate whether the INNER blood/myocardium
  boundary shows any signal at all in this same dataset (separate
  region-mean check, tissue-label-specific, mirroring 2D's own harder-
  problem investigation) -- would need the resampled tissue map's
  label-specific boundaries (RV/myocardium, myocardium/LV), not just
  the outer tissue/water shell used so far; (c) the still-unresolved
  tissue-property verification question (Saijo et al. 1997 myocardium
  impedance, left dangling in the prior segment) remains open and
  independent of this result. Not committed/pushed (standing instruction).

### Run 2026-07-10-11 — Visualized run -10's result; caught a probe-ring artifact in the image and confirmed it does NOT explain the earlier positive result
- Phase: 1. `phase1_das_3d_visualization.py`: 4-slice ground-truth vs.
  DAS-accumulator comparison figure
  (`results/figures/phase1_das_3d_reconstruction_patient023.png`).
- **Artifact spotted on inspection, not assumed benign**: every slice
  shows a bright ANNULAR ring near the domain perimeter, at a roughly
  fixed radius all the way around -- consistent with the cylindrical
  probe's own radius (56mm -> 25.8 pixels on this 60^3 grid, matches the
  visible ring closely), i.e. a direct-arrival/near-field coupling
  artifact at the transducer ring itself, not tissue-specific
  reflectivity. The true-boundary contour only visibly overlaps this
  ring on one side (where the heart happens to sit close to the probe),
  raising a real concern that run -10's "4x closer than chance" number
  might be partly/wholly an artifact of this ring rather than genuine
  boundary sensitivity.
- **Checked directly, not dismissed either way**: recomputed the
  region-mean comparison EXCLUDING a +/-3-pixel band around the probe's
  own radius (26.7% of the volume). Result: boundary=0.01345 vs.
  interior=0.00757 (1.78x) vs. exterior=0.00531 (2.53x) -- essentially
  IDENTICAL ratios to the full-volume numbers (1.80x, 2.27x). The ring
  band itself DOES show its own elevation (boundary-on-ring=0.0169 vs.
  exterior-on-ring=0.0078, 2.17x) confirming the artifact is real, but
  the boundary-specific signal genuinely persists away from it -- run
  -10's positive result is NOT an artifact of the probe-ring effect.
- Physical sanity checked? by whom?: Claude -- did not accept the
  visualization at face value in either direction (neither assumed the
  bright ring invalidated the earlier result, nor ignored it as
  cosmetic) -- computed the probe radius's exact expected pixel location
  from the known geometry and directly tested whether excluding that
  band changed the earlier finding, rather than eyeballing the figure
  and guessing.
- Gate passed? (Y/N): N/A -- Phase 1 result validation, not a phase gate.
- Observations: the probe-ring artifact is itself worth remembering for
  any FUTURE 3D DAS work in this project (e.g. if the internal
  blood/myocardium boundary is attempted next) -- it may be worth
  suppressing directly (e.g. a stronger near-position pair exclusion, or
  masking out a radial band near the probe surface before further
  analysis) rather than re-diagnosing it from scratch each time.
- Next action: same as run -10's (b) and (c), unchanged. Not committed/
  pushed (standing instruction).

### Run 2026-07-10-12 — Actual 3D boundary surface extracted and measured against ground truth: RMSE=7.65mm — a genuine, much cruder result than 2D's real-anatomy 1.04mm, not comparable to the 0.15mm idealized-phantom number the user asked about
- Phase: 1. User asked directly whether this project's result matches
  `jwave_ring_tomography`'s often-cited "0.1mm" outer-boundary accuracy.
  Checked precisely rather than assuming: that number (RMSE=0.147mm,
  run -31) was on an IDEALIZED SYNTHETIC phantom at full simulation
  resolution (dx=0.1mm), bent-ray corrected -- NOT real anatomy. The
  directly comparable 2D number (same patient, real anatomy, run -32)
  is RMSE=1.04mm (straight-ray) / 1.21mm (bent-ray) -- already ~7-8x
  worse than the idealized case before 3D is even considered. This
  distinction had not been made explicit to the user before.
- `phase1_das_3d_boundary_extraction.py`: built the missing piece --
  runs -10/-11 only checked a statistical "closer than chance" voxel
  proximity metric, never an actual fitted-surface RMSE like 2D's
  convention. Generalizes 2D's per-angle radial boundary detection to
  a full sphere of directions (Fibonacci-sphere, 1000 directions) from
  the true tissue centroid (a known-geometry ray-origin assumption,
  same role as 2D's known phantom/probe center -- not ground-truth
  boundary leakage): per direction, trilinear-interpolates the
  accumulator along the ray and takes the argmax within [10mm, 50mm]
  (the upper bound deliberately kept below the 56mm probe-ring artifact
  identified in run -11, so that known artifact cannot be picked as a
  false boundary) as the predicted radius; ground truth radius is the
  outermost 0.5-crossing of the trilinearly-interpolated tissue mask
  along the same ray.
- **Result: RMSE=7.65mm** (mean bias=+1.34mm, median |error|=4.24mm),
  across all 1000/1000 directions (every direction had a valid
  ground-truth crossing -- the resampled heart shape is star-shaped
  enough from its own centroid for this method to be well-defined
  everywhere, not just mostly). This is a REAL, much cruder number than
  2D's own real-anatomy result (1.04mm) -- roughly 7x worse, not in the
  same regime at all, though still a genuine positive relative to a
  fully blind/random guess (which runs -10/-11's checks already showed
  is not what's happening).
- Physical sanity checked? by whom?: Claude -- verified the precise
  provenance of the "0.1mm" figure before answering (found it was
  0.147mm, on an idealized phantom, not patient023) rather than taking
  the user's recollection at face value or vaguely agreeing; kept the
  RMSE's own search range strictly below the known probe-ring artifact
  radius so the extraction couldn't inadvertently reintroduce the
  artifact run -11 caught.
- Gate passed? (Y/N): N/A -- Phase 1 result, not a phase gate.
- Observations: a likely major contributor to the 7.65mm number
  (vs. 2D's 1.04mm) is simply RESOLUTION -- this reconstruction ran on
  a 60^3 grid (~2.17mm/voxel) for compute tractability (run -10), a
  coarser grid than 2D's own dx=0.1mm convention by roughly 20x. 2D's
  own real-anatomy RMSE (1.04mm) was measured at its FULL simulation
  resolution, not a downsampled one -- an apples-to-apples 3D
  comparison would need a finer reconstruction grid, not just a
  finer simulation grid (already at N=150/dx=0.867mm). Not yet
  attempted: whether increasing IMG_SIZE (the reconstruction grid,
  independent of N_GRID/the simulation grid) meaningfully closes this
  gap, or whether the gap is dominated by something else (only 96
  transmit/receive elements sparsely covering a 3D surface, vs. 2D's
  36-angle dense single-ring coverage of a much simpler 2D geometry --
  3D has a fundamentally larger surface to cover with a comparable
  or smaller element budget).
- Next action: try a finer reconstruction grid (larger IMG_SIZE) to see
  if resolution alone explains most of the gap, before concluding the
  3D geometry/element-density itself is the limiting factor. Not
  committed/pushed (standing instruction).

### Run 2026-07-10-13 — Resolution RULED OUT as the cause of the RMSE gap: full N=150 reconstruction gives essentially the SAME RMSE (7.84mm) as the coarse 60^3 grid (7.65mm) — the 2D-vs-3D gap is real and structural
- Phase: 1. Reran `phase1_das_3d_reconstruction.py` with IMG_SIZE=150
  (matching the simulation's own grid exactly, an honest apples-to-
  apples resolution comparison to 2D's own dx=0.1mm-at-full-resolution
  convention) instead of run -10's compute-motivated 60^3 downsample.
  ~21 minutes (8,952 pairs x 3,375,000 voxels), CPU, local -- notably
  cheap despite the 15.6x voxel-count increase (the 60^3 run itself
  only took ~83s, not the many-minutes originally estimated).
- **Region-mean and top-K-distance checks (run -10/-11's metrics)
  reproduce almost identically at full resolution**: boundary=0.01354
  vs. interior=0.00801 (1.69x) vs. exterior=0.00625 (2.17x); top-K
  voxels 5.15 voxels from the true boundary vs. 25.52 for random
  (still ~5x better than chance, matching run -10's ~4x). Confirms
  run -10/-11's positive result was not a resolution artifact either
  way.
- **The actual RMSE, however, does NOT improve: 7.84mm at N=150 vs.
  7.65mm at the coarse 60^3 grid (run -12)** -- essentially
  unchanged, ruling out the leading hypothesis from run -12's own
  "next action" (that reconstruction-grid coarseness explained most of
  the 2D-vs-3D gap). The 2D real-anatomy result (1.04mm, run -32,
  `jwave_ring_tomography`) used a comparably fine grid too, so this is
  a genuine, fair comparison now, not confounded by grid resolution on
  either side.
- Physical sanity checked? by whom?: Claude -- deliberately reran the
  SAME evaluation/RMSE pipeline unchanged, only swapping the
  reconstruction's own resolution, isolating resolution as the single
  changed variable before drawing a conclusion about what does/doesn't
  explain the gap.
- Gate passed? (Y/N): N/A -- Phase 1 result, not a phase gate.
- Observations: with resolution ruled out, the likely remaining
  explanations are structural, not incidental: (a) 96 elements
  covering a full 3D cylindrical surface is a MUCH sparser angular
  sampling per unit solid angle than 2D's 36 elements around a single
  ring covering a much simpler 1D contour -- the same total transducer
  budget spread over a genuinely harder (higher-dimensional) coverage
  problem; (b) `MIN_ELEMENT_SEP_CELLS` pair exclusion removes a larger
  fraction of geometrically useful near-element pairs in 3D (adjacent
  elements on the SAME ring, or adjacent rings, are often within 15
  cells of each other given the ring spacing, unlike 2D's sparser
  single-ring layout) -- not yet checked how many pairs this costs;
  (c) this project's own frequency reduction (0.8MHz vs. 2D's 2.5MHz,
  run -05) gives a ~3x longer wavelength, which alone would be
  expected to blur boundary localization by a comparable factor even
  before any 3D-specific effect -- worth a rough sanity check (3x
  1.04mm~3.1mm is still well short of 7.8mm, so wavelength alone likely
  doesn't fully explain it either, but is a real contributing factor
  not yet isolated).
- Next action: report this honestly to the user as a genuine, moderate
  3D positive (energy correctly enriches near the true surface, ~5x
  better than chance) but NOT a sub-millimeter boundary reconstruction
  like 2D's real-anatomy result, let alone the idealized 0.15mm number
  -- with the structural explanations above as the most promising
  leads if further narrowing this gap becomes a priority. Not
  committed/pushed (standing instruction).

### Run 2026-07-10-14 — Caught a real bug in the RMSE measurement itself (window-edge clipping, not a genuine peak) that had been FLATTERING the 7.65-7.84mm numbers; corrected result is worse (RMSE=10.27mm) and reveals a genuine failure mode (~32% of directions have no detectable boundary signal at all)
- Phase: 1. User asked to visualize the extracted boundary
  (`phase1_das_3d_boundary_error_map.py`: azimuth/elevation signed-error
  map + predicted-vs-true scatter, from run -12/-13's saved data).
- **Bug caught from the scatter plot itself, not assumed correct**: a
  large cluster of predictions sat at EXACTLY r_pred=50mm -- the
  search window's own upper edge (`R_MAX_MM`), not a genuine interior
  peak. Root cause: `np.argmax` over a truncated search range silently
  returns the boundary sample whenever the accumulator is still rising
  monotonically toward it (e.g. trending toward the 56mm probe-ring
  artifact just beyond the window) -- confirmed 20.5% of all 1000
  directions were exactly pinned this way, an artifact of the search
  method, not a real detection.
- **Fixed properly**: replaced `argmax` with `scipy.signal.find_peaks`
  (prominence >= 5% of the ray's own max value), taking the
  highest-prominence GENUINE local maximum (surrounded by lower values
  on both sides) per direction; directions with no qualifying local
  peak are excluded from RMSE entirely (not silently clipped to a
  wrong answer).
- **Corrected result is WORSE, not better: RMSE=10.27mm** (was
  7.65-7.84mm under the buggy clipping-as-detection method), mean
  bias=-1.65mm, median |error|=4.04mm. **317/1000 directions (31.7%)
  had NO genuine local peak at all within [10mm,50mm]** -- a real,
  now-quantified failure mode: for roughly a third of viewing
  directions, this 3D DAS method doesn't just get the boundary wrong,
  it produces no usable boundary estimate whatsoever.
- **New failure mode visible in the corrected scatter/error map**: a
  systematic cluster of large UNDERESTIMATES specifically at the
  LARGEST true radii (true~50-58mm, predicted~20-25mm) -- consistent
  with the peak-finder locking onto a nearer, more prominent (but
  wrong) local maximum -- e.g. an inner tissue interface or a
  near-field artifact -- when the true outer surface is far enough out
  that its own peak is comparatively weak. The signed-error map by
  direction (azimuth/elevation) also confirms no simple pattern (not
  purely "one side is good, one side is bad" as the slice figure alone
  suggested) -- errors are scattered across multiple clusters, both
  positive and negative, not a single fixable geometric bias.
- Physical sanity checked? by whom?: Claude -- did not accept run -12/
  -13's RMSE numbers as final once the user asked for a visualization
  that made the window-edge-clipping artifact visible; verified the
  bug by directly checking what fraction of predictions sat at exactly
  the search boundary before concluding it was a real methodological
  flaw, then re-derived the number honestly even though it made the
  result look worse, rather than keeping the more flattering (but
  wrong) 7.8mm figure.
- Gate passed? (Y/N): N/A -- Phase 1 result correction, not a phase gate.
- Observations: this is a good example of why this repo's own "look at
  the actual figure, don't just trust the printed metric" discipline
  matters -- the 7.65-7.84mm numbers from runs -12/-13 were already
  being reported as "a real but modest positive," and would have stayed
  that way indefinitely without the visualization catching the edge-
  clipping artifact. The TRUE picture is: real energy enrichment near
  the boundary exists in aggregate (runs -10/-11/-13's region-mean and
  top-K checks are unaffected by this bug, since they don't involve
  per-direction peak-finding) but a per-direction BOUNDARY SURFACE
  extraction is substantially less reliable than that aggregate
  statistic suggested -- roughly a third of directions have no signal
  at all, and even among the two-thirds that do, RMSE is ~10x worse
  than 2D's real-anatomy result.
- Next action: report the corrected, less flattering number to the
  user directly. Possible follow-ups if this project continues: (a)
  investigate whether the ~32% no-peak-detected directions correlate
  with anything structural (e.g. gaps in the 96-element cylindrical
  coverage, specific rings/angles under-sampled); (b) investigate the
  large-radius systematic underestimate failure mode specifically
  (likely a wrong-peak selection, not a resolution or artifact issue).
  Not committed/pushed (standing instruction).
### Run 2026-07-10-15 — Interactive 3D viewer built (point-cloud, then upgraded to shaded mesh) so the user could inspect the reconstruction directly, not just read RMSE numbers
- Phase: 1. User asked to "visualise/render a reconstructed 3d model."
  First version: a rotatable point-cloud viewer (Artifact, self-
  contained HTML/canvas, no external deps) showing the true tissue-
  boundary shell (subsampled, ~5000 points), the 683 valid predicted
  boundary points colored by signed error, and the 317 no-signal
  directions as dim rings.
- **User feedback: "why does it look hideous?"**, comparing against
  `jwave_ring_tomography/results/figures/phase1_das_reflectivity_
  imaging_bent_ray.png`'s clean result. Checked that comparison
  directly rather than just improving aesthetics blindly: that figure
  is a literal cartoon heart-ICON shape (visibly a smooth "heart
  emoji" outline with a notch and point) -- the IDEALIZED synthetic
  phantom from `jwave_ring_tomography` run -31 (the same one already
  distinguished from real anatomy earlier this session), not real
  patient anatomy. Real anatomy in 2D never got below 1.04mm (run -32)
  -- so part of the visual/quality gap is a genuinely unfair
  comparison (clean icon vs. real irregular ventricle), not just a
  rendering problem.
- **But the point-cloud rendering itself was also a real, fixable
  weakness**: scattered discrete points read as noise where a
  continuous surface is the honest representation of what a boundary
  reconstruction actually is. Rebuilt the extraction on a REGULAR
  latitude/longitude direction grid (`phase1_das_3d_mesh_
  reconstruction.py`, N_THETA=96 x N_PHI=48=4,608 directions, same
  method as run -14: centroid ray-cast, `find_peaks`-based local
  maximum, outermost 0.5-crossing for ground truth) instead of the
  unstructured Fibonacci sphere -- purely to get natural quad-mesh
  connectivity. Confirms run -14's result is not sampling-scheme-
  dependent: RMSE=9.52mm, no-peak=24.9% (vs. Fibonacci's 10.27mm/31.7%
  -- same ballpark, not a new finding).
- Built an actual shaded triangle-mesh renderer (flat per-face
  normals, simple headlight-style Lambertian shading, painter's-
  algorithm depth sorting across BOTH meshes together for correct
  overlap) instead of points: the true surface renders as a
  translucent shaded blue shell, the reconstruction as an opaque
  colored surface sitting inside/through it -- and every genuine
  no-signal direction now shows up as an actual HOLE in the
  reconstructed surface (you see the true shell through the gap),
  which is a more honest and legible failure indicator than a
  scattered marker glyph.
- Physical sanity checked? by whom?: Claude -- did not just reskin the
  visualization on the user's aesthetic complaint alone; independently
  verified the "off-center concave heart" image was the idealized
  synthetic phantom (not real anatomy) before treating that as part of
  the explanation, and re-derived the RMSE/no-peak numbers on the new
  direction grid to confirm the mesh rebuild didn't silently change
  the underlying finding.
- Gate passed? (Y/N): N/A -- visualization/tooling, not a phase gate.
- Next action: none required -- this was a communication/tooling
  improvement, not a new experimental result. The open scientific
  questions remain those from run -14 (why ~25-32% of directions have
  no signal; the large-radius underestimate failure mode; the still-
  untested inner blood/myocardium boundary). Not committed/pushed
  (standing instruction) -- `.gitignore` extended to cover the new
  `*viewer_data*.json` export files (same real-patient-data caution as
  the `.npz` results), verified via `git check-ignore`.

### Run 2026-07-10-16 — User's cross-plane-ellipsoid hypothesis tested directly (same-ring-only 2D reconstruction): does NOT recover 2D-quality accuracy — points to ELEMENT DENSITY, not cross-ring pair mixing, as the dominant limiting factor
- Phase: 1. User, reasoning from first-principles geometry, correctly
  identified WHY 2D ring tomography localizes so well: when transmitter,
  receiver, AND the imaged object all lie in one plane, a bistatic
  time-of-flight constrains the reflection point to an ELLIPSE (a 2D
  curve), and two such curves in a shared plane generically intersect
  at a small number of DISCRETE points (Bezout) -- unlike 3D, where the
  same constraint is an ELLIPSOID (a 2D surface), and two surfaces
  meet along a 1D curve, not points, unless a third, sufficiently
  non-coplanar measurement is added. User proposed testing this by
  restricting to only same-ring (same-Z, coplanar) transmit/receive
  pairs, treating each of the 8 rings as an independent 2D
  reconstruction problem -- exactly `jwave_ring_tomography`'s own
  validated method.
- `phase1_das_perring_2d_reconstruction.py`: pure post-processing on
  the already-completed raw scan (`cylindrical_multistatic_scan_
  patient023.npz`) -- for each of the 8 rings (12 elements each,
  confirmed contiguous in the position array: idx 0-11/12-23/.../84-95
  at z=29/42/.../121), built a 2D DAS accumulator using ONLY that
  ring's same-Z pairs (132 pairs/ring after the same near-element
  exclusion), then ran the SAME angular boundary-extraction method
  (centroid ray-cast + `find_peaks`) against the true 2D cross-section
  at that exact Z, sliced from the same ground-truth volume already
  used throughout this project.
- **Result: this does NOT recover anything close to 2D-quality
  accuracy.** Per-ring RMSE ranged 7.56-18.71mm (mean 12.42mm) --
  actually slightly WORSE than the full 3D multistatic result
  (10.27mm, run -14), despite the per-ring version finding a
  detectable peak far more often (97.8% of angles vs. full-3D's
  68.3%). The tell: peaks exist almost everywhere in the per-ring
  case, but are badly PLACED -- not a detection-failure problem, a
  localization-precision problem.
- **Interpretation**: the user's geometric reasoning about coplanar
  vs. cross-plane ellipsoid intersections is correct in principle, but
  is NOT the dominant effect actually limiting this project's accuracy.
  The more direct explanation: each ring only has 12 elements, a 3x
  sparser angular sampling than `jwave_ring_tomography`'s 36-element
  convention -- far fewer/coarser ellipse crossings per ring means a
  peak reliably forms (satisfying SOME pair combination) but is poorly
  constrained, landing far from the true point. The full 3D method's
  slightly BETTER aggregate RMSE despite mixing in less-well-
  conditioned cross-ring pairs makes sense under this explanation too:
  it has 8,952 total pairs to work with (68x more than any single
  ring's 132), and more total constraints -- even imperfectly
  conditioned ones -- outweighs the coplanarity disadvantage.
- Physical sanity checked? by whom?: Claude -- did not assume the
  user's hypothesis was correct or incorrect a priori; built and ran
  the actual per-ring experiment using the identical method/code
  already validated in this project, and let the real RMSE numbers
  (not intuition on either side) settle which effect dominates.
- Gate passed? (Y/N): N/A -- Phase 1 diagnostic result, not a phase gate.
- Observations: this strengthens the leading hypothesis from run -13
  (element/angular density, not method class) as the primary
  explanation for the 3D accuracy gap, and now rules out "cross-ring
  pair confusion" as a major independent contributor -- a genuine
  narrowing of the explanation space, not just an additional guess
  added to the pile.
- Next action: if further closing this gap becomes a priority, the
  most promising lever per this run's finding is simply MORE elements
  per ring (or more rings), not a same-plane-only processing
  restriction -- e.g. re-scanning with 24-36 elements/ring (matching
  2D's own density) would be the direct test, though that requires new
  simulation (GPU time), unlike this run's free reuse of existing data.
  Not committed/pushed (standing instruction).

### Run 2026-07-10-17 — Cheapest direct test of the element-density hypothesis packaged: ONE ring, 36 elements (matching 2D exactly), single representative Z-slice
- Phase: 1. Per user: rather than commit to a full re-scan at 36
  elements x 8 rings (~3 hours GPU time -- 288 elements x 2 media x
  ~19s/sim), test the cheapest possible version first: ONE ring of 36
  elements (exactly matching `jwave_ring_tomography`'s own element
  density) at a SINGLE representative Z-height through patient023's
  real anatomy -- the geometric midpoint of the original 8-ring Z-span
  (z=68 grid cell, run -09's "ring 3"), not cherry-picked as the
  easiest height. 36 elements x 2 media = 72 forward sims, ~19s each
  -> ~23 minutes, a much smaller Colab commitment than the full
  re-scan, decisive enough to tell whether it's worth spending the 3
  hours.
- Checked the near-element exclusion interaction before building:
  36 elements at 56mm probe radius puts adjacent elements only 9.76mm
  (11.3 grid cells) apart -- BELOW the existing MIN_ELEMENT_SEP_CELLS=15
  direct-coupling exclusion threshold (unlike the original 12/ring
  layout, where adjacent spacing was 28.99mm/33.4 cells, comfortably
  above it). Decided to leave the threshold unchanged (it's a physical
  near-field/direct-coupling limit, not meant to scale with element
  count) -- the practical effect is only the single nearest neighbor
  on each side gets excluded per element (angular separation >= ~13.3
  degrees required), leaving ~33/35 partners per element still valid,
  i.e. ~1,188 usable pairs at this one height alone (vs. the original
  12-element ring's 132) -- a large increase in both angular density
  AND raw pair count, cleanly isolating the hypothesis being tested.
- `phase1_single_ring_36_multistatic_scan.py` + `notebooks/
  phase1_single_ring_36_multistatic_scan.ipynb`: reuses the EXACT
  medium-building/sensors mechanism already confirmed working (run
  -08/-09), changing only the element layout. Notebook built by
  adapting the existing run -09 notebook template (same embedded
  patient023 ED mask, same Drive-mount-and-checkpoint pattern, new
  output folder `single_ring_36_scan_output/`) -- caught and fixed a
  shell-quoting corruption (backticks in a markdown cell's text got
  interpreted as command substitution when passed through `bash -c`
  with double quotes, silently eating part of the text) before handing
  it off, by rewriting the affected cell directly via a standalone
  Python script instead of an inline shell one-liner.
- Physical sanity checked? by whom?: Claude -- computed the exact
  adjacent-element chord distance at 36/ring before assuming the
  existing exclusion threshold would behave sensibly at the new
  density, rather than silently reusing it unexamined; caught the
  notebook corruption by re-reading the written file's actual content
  (not just trusting the write succeeded) before considering the
  handoff complete.
- Gate passed? (Y/N): N/A -- Phase 1 test packaging, not a phase gate.
- Next action: user runs the notebook on Colab GPU (~23 minutes);
  result determines whether the full 8-ring x 36-element re-scan
  (~3 hours) is worth committing to, or whether element density alone
  doesn't explain the gap either, pointing further investigation
  elsewhere. Not committed/pushed (standing instruction).

### Run 2026-07-11-18 — Single-ring 36-element result (after 2 interrupted Colab runs and a units bug caught in the new evaluation script): RMSE=5.82mm, ~2.6x better than 12 elements/ring but still ~5.6x worse than 2D's validated 1.04mm — element density matters a lot, but isn't the whole story
- Phase: 1. First two Colab runs of `notebooks/phase1_single_ring_36_
  multistatic_scan.ipynb` were interrupted at nearly the identical
  index (tx=30/36) in each pass -- first during the water pass, then
  during the phantom pass on the retry -- a suspiciously consistent
  ~9-10 minute cutoff pattern, not obviously explained by laptop sleep
  alone (flagged as possibly the VS Code Colab extension's own
  connection timeout, not investigated further since a clean third run
  completed fully: 36/36 water, 36/36 phantom, confirmed via the same
  non-zero-trace check used throughout this project).
- **Bug caught before trusting the result**: `phase1_single_ring_36_
  evaluation.py`'s first run gave RMSE=NaN, 0% valid (100% no-peak) --
  the accumulator was entirely zero. Root-caused to a real units bug
  (NOT present in run -16's per-ring script, which correctly used the
  npz's own `dx_m` in meters): this new script multiplied grid-cell
  distances by `DX_MM` (a millimeter value) then divided by
  `WATER_SOUND_SPEED` (m/s) without converting mm->m, making every
  predicted delay ~1000x too large -- every sample fell outside the
  valid matched-filter time range, silently returning 0 via
  `np.interp`'s fill value instead of erroring loudly. Fixed by
  dividing by 1000 before the speed-of-sound division; re-verified the
  fix produces sensible (non-degenerate) accumulator values before
  trusting the RMSE number.
- **Corrected result: RMSE=5.82mm, no-peak=23.1%** (z=68, the same
  slice run -16 tested at 12 elements/ring). Comparison table:
  - `jwave_ring_tomography` (2D, 36 elements/ring, real anatomy): 1.04mm
  - this project, 12 elements/ring, z=68 (run -16): 15.26mm
  - this project, 36 elements/ring, z=68 (this run): **5.82mm**
  - full 3D multistatic, all 96 elements (run -14): 10.27mm
  Tripling element density cut RMSE by ~2.6x (15.26mm -> 5.82mm) --
  confirms element density is a REAL, substantial factor, not a red
  herring. But a genuine ~5.6x gap to 2D's 1.04mm remains even at
  matched element density, in a single, favorable (coplanar,
  same-height) 2D-style slice -- so element density alone doesn't
  fully explain the accuracy gap either.
- Physical sanity checked? by whom?: Claude -- did not report the
  first (NaN/degenerate) result at all; caught it via the obviously
  wrong "100% no-peak" output, traced the exact numerical bug (printed
  an example t_pred against the valid lag_t range to confirm the
  ~1000x mismatch precisely) before fixing, and re-ran to confirm a
  physically sensible (non-zero, appropriately-scaled) result before
  reporting any RMSE number to the user.
- Gate passed? (Y/N): N/A -- Phase 1 diagnostic result, not a phase gate.
- Observations: the most likely remaining explanation for the residual
  ~5.6x gap is this project's own frequency reduction (0.8MHz vs. 2D's
  2.5MHz, forced by GPU memory limits back in run -05) -- a ~3x longer
  wavelength would be expected to blur matched-filter peak
  localization by a comparable factor independent of element count,
  though this has not been isolated/tested directly (would require a
  fresh simulation at higher frequency, a real GPU-memory tradeoff
  this project already navigated once). Domain/grid resolution
  (dx=0.867mm vs. 2D's 0.1mm) is a secondary candidate but seems less
  likely to dominate, since the boundary-extraction step itself
  already interpolates to sub-voxel precision.
- Next action: if closing the remaining gap further is a priority, the
  next most informative (though not free) test would isolate
  frequency's contribution specifically -- e.g. comparing this same
  single-ring setup at a higher frequency, memory permitting at this
  smaller (single-ring, not full-volume) scale. Not committed/pushed
  (standing instruction).

### Run 2026-07-11-19 — Packaged a cheap re-test of whether N=300/470 (needed for full 2.5MHz) fit in GPU memory NOW that sensors-based recording is used — the old N=200/300 OOM finding predates that fix
- Phase: 1/2 (scoping for a possible full high-frequency re-scan). User
  asked: since 2.5MHz would recover 2D's accuracy, why not just scan
  the full 3D volume at 2.5MHz with 36 elements/ring, split into
  batches? Clarified first: 2.5MHz requires dx~0.276mm (vs. current
  0.867mm) to hold this project's own ~2.2 points/wavelength standard,
  meaning N~470 instead of N=150 over the same 130mm domain -- a ~30x
  increase in total grid points. Batching the COMPUTE JOB over
  multiple shorter Colab sessions (which does help with the ~9-10
  minute disconnect pattern seen in runs -17/-18) is an orthogonal
  fix from the PER-SIMULATION GPU MEMORY question -- more sessions
  doesn't shrink what a single simulation needs to hold in memory.
- **Caught before assuming anything**: the standing belief that
  N~200-300 OOMs on this GPU (run -05) was measured using the OLD
  default full-field recording, which run -07 later proved was the
  actual memory hog (scaling as N^3 x n_steps), not grid size alone.
  Nobody has retested larger N under the CORRECTED sensors-based
  recording scheme -- the old finding may simply not apply anymore.
- `phase2_high_freq_gpu_memory_test.py` + `notebooks/phase2_high_freq_
  gpu_memory_test.ipynb`: tests N=300 and N=470 directly (homogeneous
  water, sensors-based recording, single forward simulation each,
  same convention as `phase0_3d_gpu_timing_sweep.py`) -- no patient
  data dependency, nothing to embed. Confirmed via `git check-ignore`
  that the blanket `notebooks/` gitignore rule still excludes it even
  though this particular notebook has no patient-data reason to be
  excluded (a minor, harmless inconsistency, not worth a special-case
  rule for one file).
- Physical sanity checked? by whom?: Claude -- did not accept the old
  N=200/300 OOM finding as still authoritative without checking
  whether the fix that came AFTER it (run -07) would change the
  answer; designed the test to isolate exactly the variable in
  question (grid size under sensors-based recording) rather than
  re-running the old (now-known-flawed) full-field test at larger N.
- Gate passed? (Y/N): N/A -- Phase 1/2 scoping, not a phase gate.
- Next action: user runs the notebook on Colab GPU; if N=300 and/or
  N=470 fit and complete, a full batched 2.5MHz/36-element/8-ring
  re-scan becomes a real, GPU-memory-feasible option to design next
  (time budget still to be estimated once per-step cost at this N is
  known -- likely substantially longer than 19s/sim given finer dx and
  a smaller CFL-limited timestep). If they don't fit, that sets the
  real current ceiling under the corrected recording scheme before
  committing to anything further. Not committed/pushed (standing
  instruction).

