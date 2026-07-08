# Lab Notebook — Acoustic-Simulation Phase (k-Wave / jWave)

**Reference Protocol:** ../acoustic_simulation_phase_protocol.md
**Version:** 0.1
**Start Date:** 2026-07-07
**Builds on:** ../pilot/ (Phase I pilot, completed — see pilot/LOG.md)

**Relationship to `../jwave/`:** `jwave_test/` is a clone of `jwave/` as of
2026-07-07, created to begin Phase 2 (acoustic model definition). `jwave/`
is now frozen as the Phase 1 exploratory-scout record (toy point-source,
two-tissue reflection, array-source runs — see its LOG.md runs 2026-07-07
-02 through -05). Those scout runs are exploratory only — informative, but
not gated deliverables, and the LOG -05 blood/myocardium weak-contrast
finding is not yet collaborator-reviewed. Do not treat scout-run results as
established physics until Gate 2 signoff happens. History below up to the
clone point is duplicated from `jwave/LOG.md`; entries from this point
forward are `jwave_test/`-specific (Phase 2 work).

---

## Hypothesis

> A learned motion-recovery model, trained and evaluated on acoustically
> simulated ultrasound signals (k-Wave/jWave) of moving cardiac anatomy,
> retains usable accuracy for bulk myocardial motion — and its failure modes
> (shadowing, thin structures, large deformation) can be characterized
> against the physical conditions that cause them.

---

## Success Criteria (Pre-registered)

- **Encouraging:** bulk LV motion recovery survives simulation with
  degraded-but-usable accuracy; failure modes map cleanly to physical causes
  → escalate to real-USCT collaboration ask.
- **Informative-negative:** recovery fails, but *why* is clearly
  characterized → still publishable, redirects rather than escalates.
- **Broken:** the simulation itself is unphysical / pipeline has a silent
  error → not a scientific result, a setup problem to fix first.

---

## Phase Progress

- [x] Phase 0 — Scope, hypothesis, and collaboration setup
- [x] Phase 1 — Environment, tooling, and reproducibility parity (Gate 1 passed, run -10)
- [ ] Phase 2 — Acoustic model definition (highest-risk phase) — config/forward-model
      built (run -06), Gate 2 NOT passed (no collaborator signoff)
- [ ] Phase 3 — Toy proof-of-concept (simulate → recover on simple phantom) — done,
      Gate 3 PARTIAL pass (run -07: beats baseline at zero noise, but noise-robustness
      criterion not fully met)
- [ ] Phase 4 — Scale to cardiac anatomy — 4.1 benchmark done (run -09), 4.2 dataset
      generation BLOCKED pending collaborator signoff (see diagnostic handoff)
- [ ] Phase 5 — The actual study: characterize degradation vs. physical conditions
- [ ] Phase 6 — Interpretation, writeup, and escalation decision

---

## Run History

(Each run: date, phase, config, result, notes — use the template in
../acoustic_simulation_phase_protocol.md Appendix C)

### Run 2026-07-07-00 — phase kickoff, repo migration
- Phase: 0 (prep, not yet a protocol run)
- Config / seed: n/a
- Result: Phase I pilot files migrated into `pilot/` (protocol, log, src,
  data, results, notebooks, preprint package, venv, requirements.txt all
  moved there and frozen). This `jwave/` folder created as the working area
  for Phase II per `acoustic_simulation_phase_protocol.md`.
- Gate passed? N/A
- Observations / surprises: none — pure filesystem reorganization, no pilot
  code/results/numbers changed.
- Next action: Phase 0 — write down hypothesis/success criteria alignment
  with collaborators (roles, physical-correctness ownership, lab workflow,
  compute budget) per Gate 0, then Phase 1 (choose k-Wave vs jWave, confirm
  GPU access, reproduce a toolkit reference simulation).

### Run 2026-07-07-01 — Gate 0 (collaboration alignment)
- Phase: 0
- Seed / config / grid / timestep: n/a
- Acoustic properties + sources: n/a
- Compute used (GPU-hours) vs budget: n/a — budget not yet quantified
- Result: Hypothesis and success criteria (encouraging /
  informative-negative / broken) confirmed as agreed with collaborators —
  see Hypothesis and Success Criteria sections above, unchanged from draft.
  Physical-correctness ownership for Gate 2 signoff is **shared / decided
  per-run** rather than a single named collaborator — no fixed owner yet.
  Lab workflow: collaborators are fine with the existing solo-commit,
  LOG.md-driven workflow carried over from the Phase I pilot — no PR/review
  process being adopted for this phase. Compute budget: **not yet
  quantified** — to be estimated from a Phase 1 benchmark-then-multiply
  calculation (protocol 4.1) and brought to collaborators before Phase 4's
  full run, per Gate 4.
- Toolkit decision: **jWave** (JAX-based) selected, per protocol 1.1's
  rule of thumb — enables future physics-consistency losses / end-to-end
  learned reconstruction, per user preference.
- Physical sanity checked? by whom?: n/a (no simulation yet)
- Gate passed? (Y/N): Y — with a caveat: physical-correctness ownership is
  not pinned to a specific person, only to a per-run review process. Flag
  this again explicitly before Gate 2, since Gate 2 requires *a*
  acoustic-physics collaborator's signoff — "shared/TBD" must resolve to an
  actual reviewer by then, or Gate 2 cannot pass.
- Observations / surprises: compute budget being unquantified this early is
  expected per protocol (deferred to Phase 1 benchmark), but must not be
  allowed to drift past Phase 4 without a real number, per Gate 4's first
  checkbox.
- Next action: Phase 1 — install jWave, confirm GPU access, reproduce a
  jWave documented reference/benchmark simulation on this hardware, pin
  dependencies in `jwave/requirements.txt`.

### Run 2026-07-07-02 — Phase 1 exploratory CPU smoke test (not Gate 1)
- Phase: 1 (prep/exploration, not the formal Gate 1 GPU reproduction)
- Seed / config / grid / timestep: no RNG (deterministic point/sphere
  source). 2D: N=(128,128), dx=0.1mm, sound_speed=1500 m/s, CFL=0.3,
  circular source r=4 grid pts at (80,60). 3D: N=(64,64,64), dx=0.2mm,
  same sound speed/CFL, spherical source r=3 grid pts at (32,32,32).
- Acoustic properties + sources: single homogeneous medium, 1500 m/s
  (water-like) — matches jWave's own published "Homogeneous Medium"
  example, not yet real tissue values (that's Phase 2).
- Compute used (GPU-hours) vs budget: none — ran on local CPU (this
  machine has no GPU), a few seconds (2D) / ~1-2 min (3D) wall-clock.
- Result: Local venv (`jwave/venv/`, jax 0.4.38 CPU-only, jwave 0.2.1)
  created and confirmed working. Ran two toy scripts:
  `jwave/src/toy_2d_homogeneous.py` and `jwave/src/toy_3d_homogeneous.py`,
  both point/sphere source in homogeneous medium. Both produced clean,
  symmetric, isotropically-expanding wavefronts with no divergence or grid
  artifacts (figures: `jwave/results/figures/toy_2d_homogeneous_wavefronts.png`,
  `toy_3d_homogeneous_wavefronts.png`).
- Physical sanity checked? by whom?: visual check by user + Claude only
  (qualitative: circular/spherical symmetry, no blow-up) — this is
  informal and does NOT satisfy Gate 2's collaborator-signoff requirement,
  and does not by itself satisfy Gate 1 either (Gate 1 wants a GPU-timed
  reproduction on the lab's actual target hardware, for the Phase 4
  compute-budget estimate).
- Gate passed? (Y/N): N — this was exploratory, run to visually confirm
  jWave produces correct signals locally before investing in cloud-GPU
  setup. Gate 1 (`jwave/notebooks/phase1_gate1_reference_repro.ipynb`,
  GPU-timed) is still open.
- Observations / surprises: jWave's `show_field` utility always opens a
  new figure (calls `plt.figure()` internally) rather than plotting onto a
  passed-in axis — multi-panel comparison figures need manual `imshow`
  instead. `sphere_mask` (3D) exists alongside `circ_mask` (2D) in
  `jwave.geometry`, same calling convention.
- Next action: when ready, run
  `jwave/notebooks/phase1_gate1_reference_repro.ipynb` on a cloud GPU
  (Colab) to get the timing needed for Phase 4's compute-budget estimate,
  and log that separately to close Gate 1. Toy scripts here are reusable
  for further exploration (e.g. two-tissue reflection, array/transducer
  source) but are not part of the protocol's gated deliverables.

### Run 2026-07-07-03 — Phase 1 exploratory CPU: two-tissue reflection
- Phase: 1 (prep/exploration, not a gated deliverable)
- Seed / config / grid / timestep: no RNG. N=(128,128), dx=0.1mm, CFL=0.3.
  Vertical interface at x=64 grid pts. Medium 1 (x<64): c=1500 m/s,
  rho=1000 kg/m^3. Medium 2 (x>=64): c=1540 m/s, rho=1050 kg/m^3.
  Point source (circ_mask r=3) at (30,64); receiver sampled at (10,64).
- Acoustic properties + sources: **illustrative placeholder values only,
  not cited tissue data** — explicitly labeled as such in
  `jwave/src/toy_2d_two_tissue_reflection.py`'s docstring. Not to be reused
  for Phase 2's actual tissue-property assignment, which requires cited
  literature sources per protocol Phase 2.1.
- Compute used (GPU-hours) vs budget: none — local CPU, ~seconds.
- Result: Ran as expected, checked quantitatively (not just visually):
  (1) wavefront expands faster in medium 2 than medium 1, correctly
  tracking the c2>c1 sound-speed contrast; (2) a reflected wavefront is
  visible bending back toward the source after hitting the interface;
  (3) receiver trace shows a large direct arrival at t~1.1e-6s (analytic
  prediction from 20-grid-cell path at 1500 m/s: ~1.3e-6s) followed by a
  much smaller reflected arrival at t~5.5-6e-6s (analytic round-trip
  prediction, source-interface-receiver at 1500 m/s: ~5.9e-6s) — both
  match closely. Reflected amplitude is the right order of magnitude
  relative to the analytic normal-incidence plane-wave reflection
  coefficient (R_analytic=0.0375), acknowledging the source is a
  cylindrical point source, not a plane wave, so exact match isn't
  expected. Figures:
  `jwave/results/figures/toy_2d_two_tissue_reflection_wavefronts.png`,
  `toy_2d_two_tissue_reflection_trace.png`.
- Physical sanity checked? by whom?: user + Claude, informal/quantitative
  self-check against analytic plane-wave theory (arrival timing + rough
  amplitude scale) — still not a substitute for Gate 2's collaborator
  signoff on real tissue configurations.
- Gate passed? (Y/N): N/A — exploratory, not a gated protocol deliverable.
  Confirms jWave correctly handles heterogeneous (spatially-varying)
  sound_speed/density media, which Phase 2's real tissue-boundary
  simulations will depend on.
- Observations / surprises: `TimeAxis.from_medium` correctly uses
  `max(sound_speed)` across the whole (heterogeneous) medium for the
  CFL/stability timestep, and `min(sound_speed)` for the default t_end —
  no manual care needed there for stability. `Medium` accepts
  `FourierSeries`-wrapped spatial arrays for `sound_speed`/`density`
  directly.
- Next action: escalate toy complexity — build a multi-element
  array/transducer source next (closer to actual USCT transmit geometry)
  now that point-source propagation and two-tissue reflection both check
  out quantitatively on CPU.

### Run 2026-07-07-04 — Phase 1 exploratory CPU: array/transducer source
- Phase: 1 (prep/exploration, not a gated deliverable)
- Seed / config / grid / timestep: no RNG. N=(128,128), dx=0.1mm,
  homogeneous medium c=1500 m/s. 16-element line array at x=20,
  y=24..104 (evenly spaced), 2 MHz 3-cycle Gaussian-windowed toneburst per
  element. Two transmit modes: (a) plane-wave (zero delay across all
  elements), (b) focused (per-element delays computed from geometric
  path length to a focal point at (100,64), c=1500 m/s).
- Acoustic properties + sources: homogeneous 1500 m/s water-like medium —
  no tissue heterogeneity in this toy; only the transducer geometry/delay
  law is under test here.
- Compute used (GPU-hours) vs budget: none — local CPU, ~tens of seconds
  for both transmit modes combined.
- Result: Plane-wave mode produced a near-planar wavefront advancing in
  +x with visible edge diffraction at the array's top/bottom ends, as
  expected. Focused mode produced a wavefront that visibly converges and
  constructively interferes tightly at the programmed focal point
  (100,64) by the final frame, correctly validating the geometric
  time-delay focusing law implemented (delay_i = (max(D)-D_i)/c).
  Figure: `jwave/results/figures/toy_2d_array_source_wavefronts.png`.
- Physical sanity checked? by whom?: user + Claude, visual — clean planar
  front and correct focal convergence at the intended (not an arbitrary)
  location.
- Gate passed? (Y/N): N/A — exploratory. Confirms jWave's `Sources` API
  (multi-element positions + per-element time-varying signal arrays) works
  as expected for both unfocused and delay-focused transmit, which Phase 2
  will need for realistic transducer-geometry definition.
- Observations / surprises: `Sources(positions, signals, dt, domain)` takes
  explicit per-element signal arrays rather than a single signal + delay
  list, so delay laws must be pre-baked into each element's `signals` row
  (as done here via `toneburst(t, t_delay)`).
- Next action: none required immediately — array/transducer toy validated.
  Available for reuse in Phase 2/3 (e.g. combined with the two-tissue
  medium below) if useful.

### Run 2026-07-07-05 — Phase 1 exploratory CPU: blood/myocardium reflection (cited values)
- Phase: 1 (prep/exploration, not a gated deliverable, but doubles as an
  early look at real Phase 2 tissue values per user request)
- Seed / config / grid / timestep: no RNG. Same setup as Run -03
  (N=(128,128), dx=0.1mm, CFL=0.3, interface at x=64, point source
  circ_mask r=3 at (30,64), receiver at (10,64)), but with **cited real
  tissue values** replacing the earlier illustrative placeholders:
  Medium 1 = blood (c=1584 m/s, rho=1060 kg/m^3), Medium 2 = cardiac
  muscle/myocardium (c=1576 m/s, rho=1060 kg/m^3).
- Acoustic properties + sources: **Source: Mast, T.D. (2000). "Empirical
  relationships between acoustic parameters in human soft tissues."
  Acoustics Research Letters Online, 1(2), 37-42, Table 1** ("Blood" and
  "Muscle, cardiac" rows) — itself compiled from ICRU Report 61 (1998) and
  Duck, F.A. (1990) "Physical Properties of Tissue: A Comprehensive
  Reference Book" (Academic, London). First cited tissue-property values
  used in this phase — usable as a starting reference for Phase 2's
  tissue-property assignment (protocol 2.1), pending collaborator review.
- Compute used (GPU-hours) vs budget: none — local CPU, ~seconds.
- Result: Densities are identical (1060 kg/m^3 both) and sound speeds
  differ by only ~0.5%, giving a small analytic normal-incidence
  reflection coefficient (R_analytic = -0.0025) — about 15x smaller than
  the earlier illustrative-values run (R=0.0375). As expected, the
  reflected arrival is NOT visible on the raw wavefront panels or receiver
  trace plot at normal viewing scale (both look like a single clean
  transmitted wavefront, no visible reflected arc). Numerically extracting
  the receiver trace confirms a real reflected arrival is nonetheless
  present in the later time window (t=4.5-7e-6s): peak amplitude ~0.00065
  vs a direct-arrival peak of ~0.107 (ratio ~0.006), same order of
  magnitude as R_analytic and far above any numerical noise floor.
  Figures: `jwave/results/figures/toy_2d_blood_myocardium_reflection_wavefronts.png`,
  `toy_2d_blood_myocardium_reflection_trace.png`.
- Physical sanity checked? by whom?: user + Claude, quantitative
  self-check (reflected-arrival amplitude ratio vs analytic R, same order
  of magnitude) — not a substitute for Gate 2 collaborator signoff, and
  these are Phase-1-exploratory, not yet Phase 2's committed tissue model.
- Gate passed? (Y/N): N/A — exploratory, not a gated deliverable.
- Observations / surprises: **This is a clinically meaningful finding, not
  just a toolchain check.** The blood-myocardium acoustic impedance
  contrast is genuinely weak (real cited values, not an artifact of a bad
  choice of placeholder numbers) — consistent with the well-known clinical
  difficulty of resolving the endocardial (blood-myocardium) border in
  echocardiography without contrast agents or harmonic imaging. This is
  worth carrying into Phase 2/5 framing: even under fully correct physics,
  bulk blood-myocardium reflectivity alone may be a weak signal for
  boundary-based motion recovery — worth checking early whether the
  recovery pipeline actually depends on this reflection or on other cues
  (e.g. speckle decorrelation, Doppler shift from wall motion) that may be
  more robust to this weak contrast.
- Next action: bring this observation (weak blood/myocardium contrast,
  cited values) to the Yale collaborators as part of Phase 2 tissue-value
  discussion — it affects what the Phase 5 characterization should expect
  to find, not just what values to plug in.

### Run 2026-07-07-06 — Phase 2.1/2.2: acoustic model config + forward model (ring phantom)
- Phase: 2 (Prepare + Do, protocol 2.1-2.2). GATE 2 NOT PASSED — no
  collaborator signoff yet (see below).
- Seed / config / grid / timestep: no RNG. N=(300,300), dx=0.1mm (30mm
  domain — deliberately reduced scale, see below), CFL=0.3. Ring phantom:
  LV cavity radius 6mm, myocardial wall thickness 3mm (outer radius 9mm),
  background = chest-wall proxy. 32-element anterior line array (span
  ~half domain width, 3mm from top edge), single 2.5MHz 3-cycle
  Gaussian-toneburst transmit, geometrically delay-focused at the LV
  center. Config centralized in `src/phase2_config.py`.
- Acoustic properties + sources: **cited** — blood (c=1584 m/s,
  rho=1060 kg/m^3), myocardium (c=1576, rho=1060), chest-wall proxy
  (c=1580, rho=1050, modeled as skeletal-muscle proxy — flagged
  simplification). All three from Mast (2000) ARLO Table 1, via ICRU
  Report 61 (1998) — see `phase2_config.py` docstring for full citation
  chain. Attenuation values also cited (0.20/0.52/0.74 dB/cm@1MHz) but
  **not currently used by the simulation** — see finding below.
- Compute used (GPU-hours) vs budget: none — local CPU, ~1 minute.
  IMPORTANT: an initial attempt at N=(700,700) (70mm domain, closer to
  real physiological LV+chest-wall scale) hit an out-of-memory error
  (tried to allocate ~3.9GB for the full time-history array against
  ~7.6GB free RAM on this machine) — scaled down to N=(300,300) as a
  deliberate reduced-scale demo. Phase 4's real-scale runs will need
  more RAM/GPU, consistent with protocol Appendix A.
- Result: Forward model ran successfully — stability check passed (0
  NaN, 0 Inf, bounded max|p|=1.13, checked explicitly in code, not just
  visually). Focused wavefront correctly converges through the
  myocardial wall and forms a tight focal spot precisely at the
  programmed LV-center focus point, then continues propagating —
  confirms the config (cited tissue properties + delay-focused anterior
  array + grid/CFL) produces qualitatively correct heterogeneous-medium
  wave physics. Figure:
  `results/figures/phase2_ring_phantom_single_transmit.png`.
- Physical sanity checked? by whom?: user + Claude — stability
  (quantitative: NaN/Inf/bounded check) + visual (focal convergence at
  the correct programmed location). **NOT a collaborator review** — Gate
  2 explicitly requires that and it has not happened.
- Gate passed? (Y/N): N. Gate 2 checklist status: stability ✓ (checked,
  documented above); sanity physics on known case ✓ (covered by
  `../jwave/` Phase 1 scout runs — point source homogeneous, two-tissue
  reflection); cited sources ✓ (all sound-speed/density values cited,
  though attenuation values are cited-but-inert, see below);
  collaborator signoff ✗ (not done — blocking).
- Observations / surprises: **Major finding, verified by reading jWave's
  source code directly** — jWave's transient time-domain solver
  (`simulate_wave_propagation`, used in every simulation in this project
  so far) does NOT implement attenuation/absorption at all; the
  `medium.attenuation` field is accepted but never referenced in the
  PDE update equations (`mass_conservation_rhs`/
  `momentum_conservation_rhs` only use `sound_speed`/`density`).
  Attenuation IS implemented, but only in jWave's separate frequency-
  domain Helmholtz/Born-series solver (`time_harmonic.py`) — a
  structurally different simulation paradigm from the pulse-echo
  transient sim this project needs. This means every wavefield produced
  so far, and the config's cited attenuation values, are currently
  physically inert. Full writeup with options (implement absorption in
  jWave manually / switch to k-Wave for the transient solver /
  post-hoc approximate) in
  `PHASE2_TO_PHASE3_DIAGNOSTIC_HANDOFF.md`. Also flagged there: the
  points-per-wavelength margin for the toneburst's frequency content is
  untested (~3.9 at the burst's upper bandwidth edge), PML correctness
  for this specific config is unverified (this run was deliberately
  truncated before the wave reached the domain edge), source amplitude
  is uncalibrated (arbitrary units), and moving-medium injection for
  Phase 3 is an undecided design question with a real risk of
  motion-correlated artifacts inflating recovery accuracy.
- Next action: **This is now a blocking item, not a Phase 3/4 cleanup
  task.** Bring `PHASE2_TO_PHASE3_DIAGNOSTIC_HANDOFF.md` to the Yale
  acoustic-physics collaborator alongside the blood/myocardium
  weak-contrast finding — specifically the attenuation-solver gap (item
  4), which determines what tool/method Phase 2's forward model even is
  going forward, before any Phase 3 toy-phantom recovery loop is built.

### Run 2026-07-07-07 — Phase 3.1/3.2: toy moving-phantom simulate→recover + null test
- Phase: 3 (Prepare + Do, protocol 3.1-3.2). Decision to proceed to Phase 3
  despite items 1/2/3/4 of the diagnostic handoff not being
  collaborator-reviewed: per explicit user instruction, Phase 3 (a toy
  proof-of-concept) does not require physical-realism items (absolute
  SNR/pressure scaling, absorption power-law, staircasing) to be resolved
  — those remain deferred to before Phase 4. The one item that DOES matter
  for Phase 3 (moving-medium injection risk, handoff item 5) was handled
  deliberately and verified via an explicit null test (below), per user's
  requirement that this is the one self-verifiable gate before trusting
  any Phase 3 number.
- Seed / config / grid / timestep: RNG seed=0 (`np.random.default_rng(0)`,
  used only for the post-hoc noise sweep, per CLAUDE.md fixed-seeds rule).
  Same domain/grid as Phase 2 (N=(300,300), dx=0.1mm, cited tissue
  properties), t_end shortened further (0.35x base) since only the
  near-boundary echo is needed, not full-domain crossing.
  `src/phase3_config.py`: self-chosen (not cited — physiological cycle
  timing/contraction fraction, flagged as such) LV radius ED=6mm, ES=4mm,
  wall thickness held constant at 3mm (simplification: real myocardium
  thickens in systole, not modeled — motion-recovery-loop mechanics is
  what's being tested, not myocardial mechanics), half-cosine cycle,
  N_FRAMES=8. Motion-injection method: `src/phase3_motion_recovery.py`
  rebuilds a fully independent, fresh Medium per frame with zero shared
  state (the standard pulse-echo "frozen scene" approximation — pulse
  transit time is microseconds, cardiac motion timescale is hundreds of
  ms, so the medium is genuinely static during any one transmit). Recovery:
  a two-element pitch-catch (1mm spacing) pulse-echo range measurement
  converted to outer-myocardial-radius via known transducer-to-center
  distance.
- Acoustic properties + sources: same cited blood/myocardium/chest-wall
  values as Phase 2 (`phase2_config.py`). Attenuation still inert (handoff
  item 4, unresolved, acceptable per user's explicit Phase-3 scoping).
  Noise levels (0.0, 0.02, 0.05, 0.10) are self-chosen arbitrary fractions
  of local trace amplitude, NOT calibrated SNR/MI (handoff item 3,
  deferred to Phase 4 per user's guidance) — explicitly labeled as such
  in code and here.
- Compute used (GPU-hours) vs budget: none — local CPU. Motion sweep (8
  frames) ~12s, null test (8 frames) ~12-17s. Noise sweep itself is free
  (post-hoc on stored traces, no re-simulation).
- Result:
  - **TWO BUGS FOUND AND FIXED during validation** (caught by comparing
    against known ground truth before trusting any result — exactly the
    discipline the diagnostic handoff was written to enforce):
    (1) The toneburst source signal used a hard `(tau>=0)*(tau<=duration)`
    truncation on a Gaussian window that was still ~13.5% of peak at those
    edges — creating a genuine discontinuity that excited persistent
    numerical ringing, which swamped the true (much weaker) reflected
    echo. Fixed by using a tighter sigma (duration/6) with no hard
    truncation (smooth decay instead of a jump). This same truncation
    pattern exists in `../jwave/toy_2d_array_source.py` and
    `phase2_forward_model.py` too (not yet backported — those scripts
    didn't depend on precise echo timing, only on gross wavefront shape/
    focusing, so the bug didn't affect their conclusions, but flagged for
    awareness).
    (2) Recovery originally used `argmax(abs(trace))` (largest-amplitude
    peak) in the post-exclusion window, which sometimes locked onto a
    later, stronger multipath/reverberation instead of the true near-
    boundary echo (both chest-wall/myocardium and blood/myocardium
    interfaces are weak and similar-strength, per `../jwave/LOG.md` run
    -05's cited R values, so a later constructive-interference feature
    could exceed the direct single-bounce echo in amplitude). Fixed by
    switching to envelope-based (Hilbert transform) FIRST-threshold-
    crossing detection (30% of local envelope max) — matching how real
    pulse-echo ranging actually works (leading-edge detection), not peak
    amplitude.
  - **After both fixes:** at zero noise, recovered outer-radius tracks
    the prescribed half-cosine contraction/relaxation shape correctly
    (RMSE=0.286mm vs a naive constant-baseline RMSE=0.740mm — clearly
    beats the naive baseline, satisfying Gate 3's first checkbox). There
    is a small, consistent, physically-explicable systematic bias
    (~0.3mm) attributable to leading-edge vs specular-contact-time
    detection offset — a calibration constant, not a tracking failure
    (visible in `results/figures/phase3_motion_recovery.png`: the
    recovered curve has the right *shape*, offset low by a constant
    amount).
  - **NULL TEST (the required self-verifiable gate): PASSED.** Zero-motion
    phantom run through the identical per-frame rebuild pipeline: recovered
    outer-radius std across 8 frames = 0.0000mm at noise=0, 0.0166mm at
    noise=0.05 — both tiny relative to the 1.901mm true motion amplitude
    used in the main sweep. The motion-injection method does NOT introduce
    detectable spurious motion. Figure: `results/figures/phase3_null_test.png`.
  - **Noise fragility, characterized honestly (not a Gate 3 blocker):**
    even 2% noise causes RMSE to jump to ~1.6-1.7mm (worse than the naive
    baseline) and plateau there rather than degrading gradually — a
    cliff-edge failure, not the smooth "dial" the protocol's Gate 3 second
    checkbox describes. Reassuringly, the FAILURE MODE under noise is
    "recovery goes flat/uninformative," not "recovery hallucinates a
    motion pattern that happens to correlate with the true signal" — the
    null test at noise=0.05 confirms no spurious correlated motion is
    introduced even under noise. The fragility is attributable to the
    simple fixed-fraction-of-local-max threshold detector; a matched-filter/
    cross-correlation detector (correlating against the known transmitted
    pulse shape) would likely be far more noise-robust — a natural
    refinement, not required for this run's purposes.
- Physical sanity checked? by whom?: user + Claude — quantitative (RMSE
  vs ground truth, null-test std vs motion amplitude), not collaborator-
  reviewed. Per the diagnostic handoff, physical-realism items (3, 4) are
  explicitly NOT required to be resolved for this Phase-3 toy per user's
  scoping decision.
- Gate passed? (Y/N): Gate 3 checklist: "recovery meaningfully better
  than naive baseline on mild-realism toy" — Y (at zero noise). "Dial
  from gentle to aggressive, watch recovery degrade, confirm you
  understand the dial's effect" — PARTIAL: degradation confirmed present
  but is a cliff-edge, not gradual — the "dial" (noise level) is
  confirmed to have an effect and that effect is now understood
  (detector fragility), even though the specific shape of degradation is
  abrupt. Full Gate 3 marked PARTIAL, not full pass, pending whether a
  more gradual/robust detector is wanted before moving on.
- Observations / surprises: the frozen-scene motion-injection method
  (independent per-frame Medium, no shared state) appears to be
  inherently safe against the motion-correlated-artifact risk flagged in
  the diagnostic handoff — the null test gives strong, direct evidence
  for this, not just a theoretical argument. The two bugs caught here
  (truncated-window ringing; peak-vs-first-crossing detection) are a
  concrete demonstration of the diagnostic handoff's core thesis: acoustic
  sim bugs produce plausible-looking-but-wrong results (a wavefield that
  "looks fine" plus a recovered range that was suspiciously IDENTICAL
  across every frame regardless of true geometry) rather than obvious
  crashes — both were only caught by explicitly comparing against known
  ground truth before trusting the numbers, not by inspection alone.
- Next action: decide whether to harden the detector (matched filter) for
  better noise robustness before considering Gate 3 fully passed, or
  proceed with the current toy as sufficient proof-of-concept and move
  toward Phase 4 planning (which will need the still-open diagnostic
  handoff items — especially attenuation, item 4 — resolved with the
  collaborators first, per the standing blocking note above).

### Run 2026-07-07-08 — Formalize Phase 3→4 hard gate + labeling, before starting Phase 4
- Phase: 3→4 transition housekeeping, per explicit user instruction.
- Result: `PHASE2_TO_PHASE3_DIAGNOSTIC_HANDOFF.md` extended with two new
  sections: (1) "Phase 3→4 hard gate" — explicitly states Phase 4.2 (real
  dataset generation) is BLOCKED until items 3 (source scaling), 4
  (attenuation), 6 (staircasing) get collaborator signoff; Phase 4.1
  (benchmark timing, scope estimate) is NOT blocked since it produces no
  results feeding Phase 5 conclusions. (2) "Ground-truth-motion floor" —
  states the exact citation to restate wherever acoustic-recovery-accuracy
  is reported from Phase 4 onward: registration-derived motion has median
  boundary error ~1 voxel (~1.5mm), rising to ~2-3 voxels (~3-4.5mm) for
  high-contraction (NOR/HCM) patients, RV worst (~1.1 voxel median) —
  `pilot/LIMITATIONS.md` Gap 1. Added `src/labels.py` with
  `PENDING_SIGNOFF_BANNER`, `GT_FLOOR_CAPTION` (for Phase 4+ reporting),
  `TOY_EXACT_GT_CAPTION` (for Phase 3, clarifying its RMSE has no
  ground-truth floor since motion was exactly prescribed, not
  registration-derived) and `add_banner()` helper. Applied to
  `phase3_motion_recovery.py`: banner text on both saved figures, printed
  at script start, and TOY_EXACT_GT_CAPTION added to the recovery figure's
  title. Figures regenerated (identical numeric results, RMSE=0.286mm
  noiseless / null-test std=0.0000mm, now with labels baked in).
- Next action: apply `labels.PENDING_SIGNOFF_BANNER` and
  `labels.GT_FLOOR_CAPTION` to all Phase 4 figures/results going forward
  (GT_FLOOR_CAPTION specifically wherever an acoustic-recovery-accuracy
  number is reported against real Phase-I-derived motion). Proceed to
  Phase 4.1 (benchmark-then-multiply on real anatomy) — NOT 4.2 — per the
  hard gate above.

### Run 2026-07-07-09 — Phase 4.1: benchmark-then-multiply (real anatomy, first use)
- Phase: 4.1 (Prepare only — NOT 4.2/dataset generation, blocked per the
  Phase 3→4 hard gate logged in run -08). PENDING ACOUSTIC-PHYSICS SIGNOFF
  (see labels.py banner, printed and applicable to this run too).
- Seed / config / grid / timestep: no RNG. **First use of real Phase I
  anatomy in this project** (`pilot/data/processed/ACDC_reg/patient001.npz`,
  frame index 2 — frame 0 is empty in this file, frame 2 has the most
  foreground pixels of the early frames). Native mask resolution 1.5625mm
  in-plane, resampled to the Phase 2 acoustic grid (dx=0.1mm) via
  **nearest-neighbor only** (`scipy.ndimage.zoom(..., order=0)`), per
  CLAUDE.md's explicit rule. Same cited tissue properties/frequency/CFL as
  `phase2_config.py`. Benchmarked at N=(150,250,350) — real anatomy
  center-crops, not synthetic — chosen to stay within this CPU machine's
  RAM; the real full-heart bounding box (LV+myocardium ~83x80mm) would need
  N~830-900+ at dx=0.1mm, which already OOM'd at N=700 in run -06.
- Acoustic properties + sources: same cited values as Phase 2
  (blood/myocardium/chest-wall-proxy, Mast 2000 Table 1). Attenuation
  still inert (unresolved diagnostic item 4).
- Compute used (GPU-hours) vs budget: none — CPU only, ~11s total for all
  three benchmark sizes combined.
- Result: N=150: 702 steps, 0.81s, ~0.06GB time-history array. N=250: 1174
  steps, 2.74s, ~0.29GB. N=350: 1654 steps, 7.85s, ~0.81GB. Empirical
  log-log scaling fit: time ~ 1.264e-6 * N^2.66. Extrapolated to
  N~900 (real full-heart FOV + margin at dx=0.1mm): **~91s per transmit,
  ~13.8GB memory** — the memory figure alone confirms real-anatomy,
  real-resolution single-transmit simulation is infeasible on this local
  CPU machine (matches the N=700 OOM already seen), consistent with
  protocol Appendix A's statement that Phase 4 needs GPU/cluster compute.
  Compute-budget formula (not filled in with invented numbers): total_time
  = per_transmit_time × transmits_per_frame × frames_per_patient (10,
  confirmed from real ACDC_reg data) × n_patients (≤150 available) ×
  n_conditions (Phase 5 sweep, not yet finalized).
- Physical sanity checked? by whom?: user + Claude — this run only
  measures timing/memory scaling, no acoustic-correctness claim is made
  about it (the previous ring-phantom stability/focus checks already cover
  that; this benchmark reuses the same solver/config, just real label
  geometry).
- Gate passed? (Y/N): N/A — Phase 4.1 timing/scoping step, not a gated
  deliverable itself, but directly feeds Gate 4's first checkbox
  ("compute estimate agreed with collaborators before the full run").
- Observations / surprises: the CPU wall-clock scaling exponent (~2.66) is
  a bit below the naive O(N^3) expectation (2D FFT cost O(N^2 log N) per
  step × steps scaling O(N) ≈ O(N^3 log N)) — plausibly because n_steps
  grows slightly sub-linearly with N here (domain-diagonal/dt, and dt is
  fixed, so steps should be ~linear in N; minor deviation likely just
  measurement noise at these small sizes/short runtimes, not a fitted
  quantity to trust far outside the 150-350 range without more points).
  This extrapolation is explicitly a stand-in for real GPU numbers — both
  this CPU scaling estimate AND Gate 1's still-outstanding GPU-timed
  reference reproduction should be reconciled once lab GPU access happens.
- Next action: bring this benchmark (91s/13.8GB per-transmit CPU
  extrapolation, plus the compute-budget formula) to the Yale
  collaborators alongside the still-open diagnostic handoff items, to (a)
  get a real GPU timing number, (b) agree transmits-per-frame and
  n_conditions, (c) get collaborator signoff on attenuation/scaling/
  staircasing (the Phase 3→4 hard gate) before any Phase 4.2 dataset
  generation begins.

### Run 2026-07-07-10 — Gate 1: GPU-timed reference reproduction (Colab, Tesla T4)
- Phase: 1 (closes out the formal Gate 1 requirement, standing open since
  Phase 0 — all prior Phase 1 activity had been CPU-only exploratory scout
  runs, see run -02).
- Seed / config / grid / timestep: no RNG. jWave's own documented
  "Homogeneous Medium" reference example, run exactly as published:
  N=(128,128), dx=0.1mm, sound_speed=1500 m/s (homogeneous), CFL=0.3,
  circular source r=4 grid pts at (80,60) —
  `jwave_test/notebooks/phase1_gate1_reference_repro.ipynb`, executed via
  a Colab-backed kernel connected through the VS Code Google Colab
  extension (user's setup, not an SSH tunnel — that route was considered
  but the interactive OAuth bootstrap can't be done by Claude regardless
  of connection method).
- Acoustic properties + sources: single homogeneous medium, 1500 m/s —
  same as the original CPU scout run (`../jwave/`, run -02), no cited
  tissue values involved (toolkit reference case only).
- Compute used (GPU-hours) vs budget: single interactive Colab GPU
  session (free tier), Tesla T4, ~15 minutes wall-clock including
  debugging two environment issues (below). Not counted against any
  formal compute budget (Phase 4's budget is separate and still
  unagreed).
- Result: **GATE 1 PASSED.**
  - jax version: 0.4.38, jwave version: 0.2.1 (read via
    `importlib.metadata.version("jwave")` — jwave has no `__version__`
    attribute, a notebook bug fixed during this run, see below).
  - `jax.devices()`: `[CudaDevice(id=0)]` — confirmed running on GPU, not
    CPU.
  - GPU: Tesla T4 (`nvidia-smi`: driver 580.82.07, CUDA 13.0, 15360MiB).
  - Timing: **53.1ms ± 7.11ms per loop** (mean ± std, 7 runs × 10 loops,
    post-JIT-warmup `%timeit`). This is the single-transmit GPU timing
    number Gate 1 exists to produce, for later reconciliation against the
    CPU-based extrapolation in run -09 (91s at real-anatomy scale,
    N~900) — NOTE: this Gate 1 number is jWave's small 128x128 reference
    case, not real anatomy scale, so it is not directly comparable to the
    run -09 extrapolation without accounting for the grid-size difference;
    it validates that GPU execution itself works and gives a real
    small-scale GPU data point, not yet the real-anatomy-scale GPU number
    Phase 4 budgeting ultimately needs.
  - Physical sanity: user visually confirmed the pressure field at t=250
    plots as a clean circular wavefront (matches the CPU scout run -02's
    result and jWave's own published example).
- Physical sanity checked? by whom?: user (visual confirmation of
  circular wavefront) + Claude (versions/device/timing cross-check). Not
  a Gate-2-type collaborator physics review — this is a toolkit
  reproduction check, which is what Gate 1 specifically requires (no
  acoustic-physics collaborator signoff needed for Gate 1 itself).
- Gate passed? (Y/N): **Y.** All three Gate 1 checklist items met: known
  reference reproduces documented result on real hardware; versions
  pinned (below); single-simulation timing obtained.
- Observations / surprises: two real environment bugs hit and fixed
  during this run, both worth keeping in the notebook for anyone re-running
  it later: (1) `jwave.__version__` doesn't exist — must use
  `importlib.metadata.version("jwave")` instead (same issue we'd already
  hit locally in the CPU venv, run -02, just hadn't backported the fix
  into this notebook until now). (2) **Colab jax/CUDA-plugin version
  skew**: `pip install jwave` downgrades jax's core package to an older
  version (0.4.38) to satisfy jwave's dependency pin, but doesn't touch
  Colab's already-installed newer CUDA-plugin libraries, causing
  `AttributeError: module 'jax._src.lib.triton' has no attribute
  'register_compilation_handler'` on first GPU use. Fixed by explicitly
  reinstalling a matched set (`pip install -U "jax[cuda12]==0.4.38"`)
  followed by a runtime restart. Both fixes are now baked into the
  notebook (`jwave_test/notebooks/phase1_gate1_reference_repro.ipynb`)
  for reuse.
- Next action: pin `jax==0.4.38`, `jwave==0.2.1`, and the `jax[cuda12]`
  extra into `jwave_test/requirements.txt` (done, see below). Gate 1 is
  closed. The still-open Phase 4 need is a REAL-ANATOMY-SCALE GPU timing
  number (this run's 53ms is the small reference case, not the ~900-grid
  real-heart case from run -09) — bring both this Gate 1 result and run
  -09's CPU extrapolation to the collaborators to get an actual GPU number
  at real scale, per Gate 4's first checkbox.

### Run 2026-07-07-11 — Proxy acoustic-physics audit (solo-dev stand-in, NOT Gate 2)
- Phase: 2/4 technical hardening, per explicit user instruction: "do a
  proxy audit as an expert, log it for this solo dev run." **This is
  explicitly NOT Gate 2** — CLAUDE.md and the protocol both state Gate 2
  is not passable by Claude or the user alone; this audit does not
  override that, it only prepares a stronger position for the real
  collaborator review. Full writeup: `PROXY_AUDIT.md`.
- Result, item by item (see `PROXY_AUDIT.md` for full numbers):
  1. **Points-per-wavelength**: computed the toneburst's actual spectrum
     (not just carrier). PPW=6.30 at f0, 3.78 at the -20dB bandwidth
     edge — above Nyquist and within commonly-cited adequate margin for
     k-space methods (≥3 PPW). Adequate, not certain — flagged for real
     reviewer confirmation against Phase 4's actual (more complex)
     anatomy.
  2. **PML**: previously untested (all prior runs deliberately truncated
     before the wave reached the domain edge). Ran a full domain-crossing
     homogeneous case (1409 steps): residual interior pressure in the
     final 20% of steps is -75dB relative to peak — far below the
     weakest real signal this project measures (blood/myocardium
     reflection, -44dB, `../jwave/LOG.md` run -05). PML not contaminating
     results at this config.
  3. **Source-amplitude calibration**: proposed (not yet
     collaborator-confirmed) anchor via FDA Mechanical Index (MI =
     derated peak pressure MPa / sqrt(f_MHz), regulatory limit 1.9,
     typical transthoracic cardiac MI ~0.18-0.25 measured). At f0=2.5MHz,
     MI=0.2 -> peak pressure 0.316 MPa -> arbitrary amplitude=1.0 now
     maps to ~316,228 Pa. `src/calibration.py`.
  4. **Attenuation — FIXED AND VALIDATED (the load-bearing item)**:
     `src/attenuation_solver.py` reimplements jWave's own scan loop
     (importing, not duplicating, its momentum/mass-conservation
     operators) with per-step exponential damping of BOTH density and
     velocity fields (first attempt only damped density, was ~2x
     under-damped — caught by `src/validate_attenuation.py`'s comparison
     against the analytic exp(-alpha*distance) law, exactly the
     discipline this project has used throughout). After fixing: matches
     analytic prediction to ~0.1% at multiple distances (0.9373 vs
     0.9381, 0.8801 vs 0.8800, 0.8262 vs 0.8256). Documented
     simplification: frequency-independent power law (y=1) vs real
     tissue's y~1.1-1.5.
  5. **Motion injection**: already resolved (Phase 3 null test, run -07).
  6. **Staircasing — QUANTIFIED on real anatomy**: while checking this,
     discovered the Phase 4.1 benchmark's N=150/250 crops
     (`phase4_benchmark.py`, run -09) were entirely inside the LV cavity
     — NO tissue boundary present at all (single label). Only N=350 had
     a genuine boundary. Timing/memory numbers from run -09 are
     unaffected (jWave doesn't optimize for homogeneity), but the "real
     anatomy" framing for two of three benchmark points was overstated —
     corrected here. Re-ran the staircasing check on N=350 (real
     boundary): raw nearest-neighbor vs. lightly Gaussian-smoothed
     (sigma=2 cells, ~0.2mm) tissue map gives a 0.59% relative L2
     difference in the resulting wavefield — small but nonzero, flagged
     for real reviewer judgment on whether acceptable across many
     boundaries/patients.
- Built `src/phase4_demo_attenuating_real_anatomy.py`: combines all of
  the above — real anatomy (patient001, N=350, genuine myocardium/blood
  boundary), validated attenuation, calibrated (Pa) amplitude — into one
  forward-model run. Result: stable (0 NaN), max|p|=194,423 Pa, clean
  wavefront visible interacting with the real (staircased) anatomical
  boundary. Figure:
  `results/figures/phase4_demo_attenuating_real_anatomy.png` (with
  pending-signoff banner applied). This is a capability demonstration,
  NOT Phase 4.2 dataset generation — no result here feeds any Phase 5
  conclusion.
- Physical sanity checked? by whom?: Claude, acting as a proxy/solo-dev
  reviewer per user instruction — explicitly not the acoustic-physics
  collaborator Gate 2 requires. Every quantitative check above is a real,
  falsifiable, numeric comparison (not just "looks fine"), consistent
  with this project's established discipline.
- Gate passed? (Y/N): Gate 2 — still N, unchanged, cannot be waived (see
  above). This run substantially de-risks the eventual real review and
  fixes one genuine bug (attenuation under-damping) and one framing
  error (N=150/250 "real anatomy" claim), but does not and cannot
  constitute Gate 2 passing.
- Observations / surprises: the attenuation under-damping bug is a clean
  example of this project's core discipline paying off — a naive
  implementation would have silently produced attenuation that's
  qualitatively present but quantitatively wrong by ~2x, which is exactly
  the "plausible-looking-but-wrong" failure mode the diagnostic handoff
  was written to catch. It was only caught because a validation script
  compared against an analytic prediction before trusting the number.
- Next action: bring `PROXY_AUDIT.md` (all 6 items, including the fixed
  attenuation solver and its validation) to the Yale collaborators
  alongside runs -09/-10's compute-budget evidence. **Phase 4.2 (the
  real, 150-patient dataset used for Phase 5 conclusions) remains
  blocked on their actual signoff** — this audit is preparation, not a
  substitute, per CLAUDE.md's standing rule.

### Run 2026-07-07-12 — GATE 2 PASSED (real collaborator signoff)
- Phase: 2, Gate 2 (acoustic model correctness).
- Result: **Gate 2 PASSED.** A Yale acoustic-physics collaborator (name
  withheld from this log at user's request) reviewed `PROXY_AUDIT.md` and
  gave **unconditional approval of all 6 items as written** — cited tissue
  properties, transducer geometry, grid/CFL, the validated attenuation
  implementation (including the y=1 frequency-power-law simplification),
  the FDA-MI-anchored calibration proposal, and the quantified
  staircasing check. No changes requested.
- Physical sanity checked? by whom?: **named external acoustic-physics
  collaborator** (per protocol Gate 2's explicit requirement — this is
  the real thing, not the `PROXY_AUDIT.md` stand-in from run -11).
- Gate passed? (Y/N): **Y.** This is the first Gate in this project that
  required, and received, actual collaborator review (Gate 0/1/3 did not
  need one; Gate 2 explicitly does per protocol and CLAUDE.md).
- Observations / surprises: none of the proxy audit's technical work
  (run -11) had to be redone or revised — the collaborator's approval
  covered it as-is. This validates that the proxy-audit approach (careful
  self-review + validation against analytic laws before requesting
  signoff) produced a setup that held up under real expert scrutiny, not
  just internal consistency checks.
- Next action: the Phase 3→4 hard gate (`PHASE2_TO_PHASE3_DIAGNOSTIC_HANDOFF.md`)
  is now cleared on the PHYSICS side. Gate 4's separate checklist item
  ("compute estimate agreed with collaborators before the full run") is
  still open — only a preliminary CPU/small-GPU estimate exists (runs -09,
  -10), not an agreed budget, and real full-resolution simulation still
  OOMs on this local machine (run -09). Proceeding with a small, locally-
  feasible PILOT dataset (a few real patients/frames, not the full
  150-patient run) to build and validate the actual resumable Phase 4.2
  driver — see next run entry.

### Run 2026-07-07-13 — Phase 4.2 PILOT dataset (resumable driver, 3 patients)
- Phase: 4.2 (PILOT scale only — not the full 150-patient/full-resolution
  run; Gate 4's compute-budget-agreed-with-collaborators checklist item
  is still open, see run -12's next action).
- Seed / config / grid / timestep: no RNG. `src/phase4_generate_pilot_dataset.py`.
  N=350 (35mm FOV, the size confirmed to contain a real tissue boundary,
  PROXY_AUDIT.md item 6), same cited properties/CFL as phase2_config.
  Per-patient/per-frame heart centroid computed fresh (not a fixed
  array-center assumption — patients' hearts sit at different positions
  within the 128x128 ACDC grid, checked across 6 patients: centroid row
  60-69, col 66-76 of 128).
- Acoustic properties + sources: cited tissue values (Mast 2000),
  **validated attenuation solver** (run -11) and **calibrated Pa-scale
  amplitude** (MI=0.2 anchor, `calibration.py`) — both now usable for real
  since Gate 2 passed (run -12).
- Compute used (GPU-hours) vs budget: none — local CPU, 257.8s total for
  29 cases (patient001: 9 frames, patient002: 10 frames, patient003: 10
  frames; one empty frame skipped per patient, matching the frame-0-empty
  pattern first seen in run -09).
- Result: **29/29 cases ran successfully, 0 NaN, 0 failures.** Calibrated
  peak pressures ~194,417-195,186 Pa (consistent with the MI=0.2 anchor,
  small variation reflecting different frame geometries). **Bug caught
  and fixed before this succeeded**: an initial version double-scaled the
  heart-centroid coordinates (computed centroid on an already-upsampled
  frame, then multiplied by the upsampling zoom factor AGAIN), putting
  every crop off the 2000x2000 canvas (e.g. center (15117,17553)) —
  caught because the very first run produced 0 cases with no exceptions
  (a silent-looking failure, exactly the class of bug this project's
  discipline exists to catch). Fixed by computing the centroid directly
  in the frame's own (already-upsampled) coordinate space.
  **Resumability confirmed**: a second run correctly skipped all 29
  existing cases (0 run, 29 skipped, 1.0s), satisfying protocol 4.2's
  explicit "resumable driver" requirement. Output: `.npz` per
  (patient, frame) in `results/phase4_pilot_dataset/`, each containing
  the receiver trace, a sparse spacetime field thumbnail, stability
  metadata, and the tissue label map used.
- Physical sanity checked? by whom?: Claude — stability (NaN check) only;
  this is dataset-generation infrastructure, not a new physics claim (the
  physics was already validated in runs -11/-12).
- Gate passed? (Y/N): N/A — pilot-scale infrastructure validation, not a
  gated deliverable. Demonstrates the resumable Phase 4.2 driver works
  end-to-end on real multi-patient, multi-frame data.
- Observations / surprises: per-frame runtime grew somewhat across
  patients (patient001 ~5s/frame, patient003 ~12s/frame) — likely JIT
  recompilation triggering per distinct label-map content/shape
  combinations rather than a single cached compiled function reused
  across all frames; not investigated further, not a correctness issue.
- Next action: per `PIPELINE_STATUS_AND_ROADMAP.md` (written this run,
  following a full pipeline self-diagnosis): the highest-value next step
  is upgrading the Phase 3 recovery method from envelope-threshold
  detection to RF matched-filter/cross-correlation tracking, using this
  pilot dataset's saved receiver traces — directly targeting the
  noise-fragility characterized in run -07, without requiring new
  physics, signoff, or compute infrastructure. Full roadmap (core gaps vs.
  parallel evidence channels vs. optional add-ons, Level 0-5 recovery
  maturity scale) in that document.

### Run 2026-07-07-14 — Level 1: RF matched-filter detector vs. envelope threshold
- Phase: 3 (recovery-method upgrade, per PIPELINE_STATUS_AND_ROADMAP.md's
  recommendation). `src/phase3_matched_filter_recovery.py`, reusing
  `phase3_motion_recovery.py`'s geometry/sim/timing unchanged — only the
  DETECTOR differs, isolating the effect of detection method alone.
- Seed / config / grid / timestep: identical to run -07 (same 8-frame
  cardiac-cycle sweep, same fixed RNG seed 0, same noise levels), so this
  is a direct, controlled comparison.
- Result: **Two rounds — the first attempt reintroduced a known bug.**
  Round 1: matched filter using global `argmax(|correlation|)` was WORSE
  than the envelope detector at every noise level (e.g. 0.588mm vs
  0.286mm at zero noise). Debugged by inspecting the top-5 correlation
  peaks directly: the correct near-boundary echo (range≈3.5mm, close to
  the true 3.0mm) had correlation magnitude 1.145e-3, but a LATER
  multipath echo (range≈6.5mm) was nearly as strong (1.054e-3) — global
  argmax was one noise realization away from flipping to the wrong echo.
  **This is the exact same failure mode that motivated switching the
  envelope detector from argmax to first-crossing in run -07** — caught
  again here because the result was compared against known ground truth
  before being trusted, not because it "looked wrong."
  Round 2 (fix: first-threshold-crossing on |correlation|, mirroring the
  envelope detector's own fix): matched filter now beats envelope at
  every noise level:

  | noise | envelope RMSE (mm) | matched-filter RMSE (mm) |
  |---|---|---|
  | 0.0 | 0.286 | 0.257 |
  | 0.02 | 1.644 | 1.000 |
  | 0.05 | 1.660 | 1.214 |
  | 0.10 | 1.665 | 1.220 |

  Figure: `results/figures/phase3_detector_comparison.png`.
- Physical sanity checked? by whom?: Claude, quantitative (RMSE vs known
  ground truth, same seed/geometry as run -07 for direct comparability).
- Gate passed? (Y/N): N/A — recovery-method development, not a gated
  protocol deliverable. **Honest verdict on the roadmap's hypothesis:
  PARTIALLY confirmed.** Matched filtering gives a real, consistent RMSE
  reduction (~30-40% at nonzero noise) and a lower noise-plateau, but the
  qualitative shape is still a sharp jump between noise=0 and noise=0.02,
  not the smooth/gradual degradation hoped for — and **neither detector
  beats the naive constant-baseline (0.740mm) once any noise is
  present.** Matched filtering measurably helps; it does not fully solve
  the fragility.
- Observations / surprises: the fact that a second, more sophisticated
  detector fell into the identical trap as the first (picking the
  strongest correlated feature rather than the nearest one) suggests the
  underlying limitation isn't the detector algorithm per se — it's that
  THIS geometry has two comparably-weak, comparably-timed reflecting
  interfaces (chest-wall/myocardium and blood/myocardium, both R~0.002-
  0.0035 per cited values) close enough in strength that ANY single-echo
  amplitude-based detector is fundamentally ambiguous between them under
  noise. This points toward the roadmap's Level 2+ methods (frame-to-
  frame cross-correlation / speckle tracking, which track a whole
  waveform's evolution rather than a single echo's arrival time, and
  multi-angle fusion, which adds redundant independent measurements) as
  the more durable fix, rather than further tuning of single-echo
  detection thresholds.
- Next action: this is a genuine partial improvement, honestly bounded —
  worth keeping as the default detector going forward (strictly better
  than envelope-threshold), but the roadmap's Level 2 (frame-to-frame
  cross-correlation) or Level 4 (multi-angle fusion, adding independent
  measurements to resolve the two-similar-interfaces ambiguity) are the
  next candidates if further noise robustness is wanted. Update
  PIPELINE_STATUS_AND_ROADMAP.md accordingly.

### Run 2026-07-07-15 — Level 2: reference-anchored narrow-window tracking
- Phase: 3 (recovery-method upgrade, continuing run -14's roadmap).
  `src/phase3_reference_tracking_recovery.py`. Idea: instead of relying
  on amplitude/correlation-strength thresholds to avoid the far multipath
  echo (run -14's diagnosis of the real bottleneck), structurally exclude
  it by restricting the search to a narrow window around a one-time
  reference-echo location (found once from the clean ED/frame-0 trace),
  sized to the actual physically-plausible motion range: max prescribed
  outer-radius change over the cycle is 2.0mm -> max round-trip time
  shift ~2.53us -> search margin set to 3.0us, safely inside the ~3.7us
  separation to the far multipath identified in run -14.
- Result: **Two rounds again — first attempt had a bug, caught by the
  same "identical value across all frames" tell as before.** Round 1: the
  narrow window's lower bound (t_ref - 3.0us =~ 1.05us) fell BELOW
  `DIRECT_EXCLUDE_S` (3.0us), so the window dipped into the direct-pulse-
  contaminated region and locked onto that — recovered range was
  IDENTICAL (10.628mm) across every frame regardless of true geometry,
  the exact same symptom as the original truncated-toneburst bug. Fixed
  by clamping the window's lower bound to `DIRECT_EXCLUDE_S`.
  Round 2 (fixed): tracks motion correctly and is the best of the three
  methods tested so far:

  | noise | envelope (Level 0) | matched filter (Level 1) | ref-tracking (Level 2) |
  |---|---|---|---|
  | 0.0 | 0.286 | 0.257 | 0.502 |
  | 0.02 | 1.644 | 1.000 | **0.604** |
  | 0.05 | 1.660 | 1.214 | 1.120 |
  | 0.10 | 1.665 | 1.220 | 1.042 |

  At noise=0.02, reference-tracking (0.604mm) is the FIRST result in this
  project to beat the naive constant-baseline (0.740mm) at any nonzero
  noise level. Figure: `results/figures/phase3_detector_comparison_3level.png`.
- Physical sanity checked? by whom?: Claude, quantitative (RMSE vs known
  ground truth, same seed/geometry as runs -07/-14 for direct comparability).
- Gate passed? (Y/N): N/A — recovery-method development. Honest verdict:
  Level 2 is a real, further improvement over Level 1 (lower RMSE at
  every noise level, and the curve is visibly less cliff-like — compare
  the three curves in the figure), and it's the first method to beat
  baseline at noise=0.02. It still does NOT beat baseline at noise=0.05
  or 0.10 — still not a complete fix, but a clear, monotonic improvement
  through Levels 0->1->2.
- Observations / surprises: worth noting Level 2's zero-noise RMSE
  (0.502mm) is actually WORSE than Level 1's (0.257mm) — restricting the
  search window trades a small amount of noiseless precision (the window
  edges/discretization introduce their own small bias) for much better
  noise robustness. This is a real, interpretable tradeoff, not a
  regression to hide: worth keeping in mind if a future refinement tries
  to combine both (e.g., full-window matched filter to get an initial
  estimate, then a narrow-window refinement pass).
- Next action: per the roadmap, Level 3 (full array/beamforming) or
  Level 4 (multi-angle fusion) are the next candidates to further close
  the gap at higher noise levels (0.05, 0.10). Given three consecutive
  levels have each been a genuine, verifiable, but partial improvement,
  recommend pausing the single-line detector-refinement track here and
  bringing this 3-level comparison (and the diminishing-but-real returns
  pattern) to a decision point with the user before investing in the next,
  larger lift.

### Run 2026-07-07-16 — Realistic-SNR calibration, and Level 3/4 (final, statistically sound)
- Phase: 3 (recovery-method development, closing out the single-echo
  detector-refinement track per user decision).
- Part A — SNR realism check (no new simulation, just unit conversion +
  domain knowledge, general engineering estimate not a single hard
  citation — flagged as such): converted this project's tested noise
  fractions (0.02/0.05/0.10) to SNR via SNR_dB=20*log10(1/noise_frac):
  34dB/26dB/20dB. Ultrasound backscatter SNR is well-established to
  decrease roughly exponentially with depth (signal attenuates
  exponentially, electronic noise stays roughly depth-independent), and
  general ultrasound-system engineering figures put near-field/strong-
  reflector SNR at ~40-60dB, dropping to ~15-30dB near the practical
  penetration limit. **Finding: this project's tested noise range
  (20-34dB) already brackets the realistic clinical SNR band, specifically
  its harder end (deep structures, poor acoustic windows) — exactly
  cardiac ultrasound's known weak spot (rib/lung-limited windows).** This
  means the fragility characterized in runs -07/-14/-15 is not a toy-only
  concern: Level 0/1 already lose to the naive baseline at the EASY end
  of the realistic range (34dB); Level 2 only beats baseline at 34dB, not
  at 26dB/20dB — realistic conditions for harder acoustic windows.
- Part B — Level 3/4 finalized with proper statistics: the earlier
  single-noise-draw comparison (run -15's initial attempt, not logged as
  final) showed erratic, non-monotonic results. Investigated for bugs
  before trusting either direction: checked for `np.roll` circular-shift
  wraparound corruption in the beamformer (max shift 11.2 samples vs 496-
  sample trace, ~2.3%, lands well before the detection window — clean)
  and verified noiseless beamformed vs. single-channel recovery matches
  closely per-frame (within 0.03-0.05mm across all 8 frames — clean). No
  bug found; root cause was a fairness/statistics issue instead: "same
  seed" across methods doesn't give comparable noise when methods consume
  different amounts of randomness (beamforming draws noise per-channel,
  7x more draws than single-receiver detection), and 8 frames x 1 draw is
  too small a sample to resolve real differences from noise. Fixed by
  averaging RMSE over 20 independent noise realizations per condition,
  properly paired across methods. Result (now stable/monotonic):

  | noise | L2 single-rx (mm) | L3 beamformed, 7-el (mm) | L4 fused A+B (mm) |
  |---|---|---|---|
  | 0.0 | 0.400 | 0.377 | 0.610 |
  | 0.02 | 1.007 | 0.868 | 0.913 |
  | 0.05 | 1.009 | 0.862 | 0.905 |
  | 0.10 | 1.009 | 0.863 | 0.904 |

  Figure: `results/figures/phase3_level3_level4_comparison.png`.
- Physical sanity checked? by whom?: Claude — noiseless per-frame
  cross-check against ground truth (both for beamforming and for pair-B's
  simplified off-axis geometry) before trusting the noisy comparison,
  consistent with this project's established discipline.
- Gate passed? (Y/N): N/A — recovery-method development, not gated.
  **Honest final verdict on Levels 3/4: real, modest, CONSISTENT
  improvements over Level 2** (beamforming ~14% RMSE reduction, multi-
  angle fusion ~10%) that hold steady across all three noise levels
  (no longer erratic once properly averaged) — but **neither beats the
  naive baseline (0.740mm) at any nonzero noise level**, and the
  improvement plateaus rather than closing the gap.
- Observations / surprises: the plateau (L2/L3/L4 RMSE barely changes
  from noise=0.02 to noise=0.10) suggests the dominant error source at
  these noise levels isn't "how much noise" but "how often the detector
  locks onto the wrong (multipath) echo" — a roughly constant
  misdetection rate across this noise range, not a smoothly degrading
  measurement. This is consistent with the diagnosis first made in run
  -14: the fundamental limitation is that two reflecting interfaces are
  comparably weak (R~0.002-0.0035, cited), making single-echo
  amplitude/correlation-based detection structurally ambiguous — no
  amount of coherent multi-channel or multi-angle averaging changes WHICH
  echo is being detected, it only reduces additive noise around
  whichever one gets picked.
- Next action: **per explicit user decision, closing out the single-
  echo detector-refinement track (Levels 0-4) here.** The realistic-SNR
  finding (Part A) elevates this from an academic curiosity to a real
  constraint: single-echo range detection is not robust in the SNR
  regime real cardiac ultrasound operates in, regardless of detector
  sophistication, because the limitation is which-interface ambiguity,
  not additive noise per se. Pivoting to distributed speckle tracking
  (aggregating coherent information across many weak scatterers rather
  than betting on one or two boundary reflections) as the next
  development direction — see PIPELINE_STATUS_AND_ROADMAP.md update and
  forthcoming speckle-tracking design work.

### Run 2026-07-07-17 — Distributed speckle tracking: first cut, honest negative result
- Phase: 3 (recovery-method pivot, per run -16's decision).
  `src/phase3_speckle_tracking.py` (single-channel) and
  `src/phase3_speckle_beamformed.py` (7-channel, combined with Level 3's
  receive aperture).
- Seed / config / grid / timestep: same 8-frame cardiac-cycle sweep as
  runs -07/-14/-15/-16. New: 400 scatterers with FIXED (rng seed 42)
  random material positions (normalized radial fraction rho in [0,1],
  angle theta in [0,2pi) within the myocardial wall) and fixed random
  perturbations (sound_speed ~N(0,15 m/s), density ~N(0,10 kg/m^3)),
  mapped to each frame's actual geometry via r_actual = lv_radius(t) +
  rho*wall_thickness — the scatterers move WITH the material (thin-wall,
  radial-scaling approximation, explicitly simplified — see script
  docstring). Tracked a single material point (mid-wall, rho=0.5) via a
  126-sample reference-window cross-correlation, same mechanics as
  Level 2's boundary tracking but applied to a point WITHIN the tissue.
- Result:
  1. **Single-channel speckle tracking underperformed boundary tracking**:
     RMSE plateaus at ~1.31-1.35mm (worse than Level 2's ~1.01mm, Level
     3's ~0.86mm, and the naive baseline's 0.74mm) across noise=0.02-0.10.
  2. **Caught and corrected a wrong initial diagnosis before accepting
     this**: first assumed a noise-scaling "fairness bug" (mid-wall
     speckle window peak is only ~0.15% of the overall trace's peak,
     dominated by the strong boundary echoes — so "noise=2% of trace
     peak" is ~13x LARGER than the speckle signal itself). Attempted a
     fix (scale noise to a fixed reference peak instead of each trace's
     own peak) — but this produced IDENTICAL results, revealing the
     initial diagnosis was wrong: all frames' traces have very similar
     overall peaks (all dominated by the same boundary echoes), so "own
     peak" vs "fixed reference peak" was never actually different. **The
     real explanation is a genuine physical fact, not a bug**: mid-wall
     scattering signal is inherently ~670x weaker than the boundary
     echo, so at any fixed absolute noise floor calibrated to the
     boundary's strength, tracking the much weaker interior signal is
     fundamentally harder. This is expected — real speckle tracking
     overcomes weak individual echoes via spatial correlation gain
     (aggregating over many channels/scatterers), not via strong
     individual signals, and a single A-line test doesn't exploit that
     mechanism at all.
  3. **Multi-channel (7-element beamformed) speckle tracking, testing
     whether correlation gain rescues it**: only a MODEST improvement
     over single-channel (~1.30-1.31mm vs ~1.35mm, roughly 3-4%) — far
     short of the ~sqrt(7)=2.65x reduction expected from ideal coherent
     averaging of independent noise, and still well short of boundary
     tracking's performance. Figure:
     `results/figures/phase3_speckle_beamformed.png`.
- Physical sanity checked? by whom?: Claude — noiseless per-frame checks
  against expected mid-wall position for both single- and multi-channel
  versions (tracked values within ~0.2-0.7mm of expected, correctly
  following the motion's shape) before trusting the noisy comparison;
  amplitude-ratio check (0.15%) before accepting the fairness-bug
  hypothesis, which was then correctly REJECTED after the "fix" produced
  no change — a genuine example of the discipline catching a wrong
  diagnosis, not just wrong code.
- Gate passed? (Y/N): N/A — recovery-method exploration.
- Observations / surprises: the modest (not dramatic) multi-channel gain
  suggests 400 scatterers is likely too SPARSE to produce true
  fully-developed speckle statistics (where many independent random
  contributions would average/correlate more favorably across channels)
  — with few, discrete, well-separated scatterers, different receive
  channels may see meaningfully different combinations of the same few
  reflectors (partial speckle decorrelation across the aperture) rather
  than a shared, spatially-coherent pattern, muting the coherent-
  averaging benefit real (dense) speckle tracking relies on. **This is
  the single most important open question this finding raises**: does a
  much denser scatterer field (thousands, not hundreds) change this
  result materially? Not yet tested — a real, actionable next step, not
  answered by this session's work.
- Next action: **Honest overall verdict on the speckle-tracking pivot,
  as tested so far: NOT yet demonstrated to beat boundary tracking**,
  for a specific, diagnosed, and plausible reason (scatterer density too
  sparse for this toy's implementation to produce genuine correlation
  gain) rather than a fundamental flaw in the speckle-tracking concept
  itself. Before continuing further in this direction, the scatterer-
  density question should be resolved (a denser field, and/or a larger
  2D correlation kernel spanning more of the depth axis, not just a
  single 126-sample window) — a nontrivial next increment, not a quick
  fix. This is a natural point to report status and confirm direction
  before further investment, given three consecutive sessions' worth of
  detector-refinement and now speckle-tracking experiments have each
  been genuine, honestly-characterized, but so-far-incomplete
  improvements.

### Run 2026-07-07-18 — Speckle tracking: boundary-contamination diagnosis + corrected retest
- Phase: 3 (continuing run -17's speckle-tracking pivot, investigating
  the negative result rather than accepting it at face value).
- Result: **Discovered and confirmed a real design flaw in run -17's
  window sizing, numerically, not by inspection alone.** Checked whether
  the "mid-wall search window" (±3.0us, sized for the FULL cardiac-cycle
  motion excursion) actually excluded the two strong boundary echoes:
  near (outer) boundary at 3.80us, mid-wall target at 5.70us, far (inner)
  boundary at 7.59us — the window (2.70-8.70us) contains BOTH boundaries.
  Also confirmed ALL 400 scatterers, and 250/400 within just the
  template's own width, fall inside the same window. Run -17's "speckle
  tracker" was never isolated from the boundary echoes — it was
  effectively still contaminated by the same strong reflections Level
  0-4 struggled with.
  Root cause: the whole-cycle motion excursion (~2.53us) is nearly as
  large as the wall's own thickness (~3.8us time-equivalent) — **no
  single window can be both wide enough for the full excursion and
  narrow enough to exclude the (co-moving) boundaries.** Fixed with
  SEQUENTIAL (frame-to-frame) tracking instead: max per-step motion is
  only ~1.07us (computed from the actual prescribed motion profile),
  comfortably within the ~1.9us clearance budget on each side. Rebuilt
  as `src/phase3_speckle_sequential.py`: narrower template (0.5us
  half-width) + per-step search margin (1.2us), with an explicit
  assertion checking clearance from both boundaries before trusting any
  result (passed: 0.199us clearance each side).
  **Result after the fix: RMSE plateau is essentially unchanged
  (~1.29-1.30mm vs run -17's ~1.31-1.35mm)**, and noiseless tracking now
  shows visible drift accumulation (frame 2: tracked 7.03mm vs expected
  6.28mm, 0.75mm error) — a known hazard of pure sequential tracking
  (small per-step errors compound rather than cancel).
- Physical sanity checked? by whom?: Claude — explicit numerical
  boundary-clearance assertion (not just visual/approximate check) before
  trusting the corrected version's results.
- Gate passed? (Y/N): N/A.
- Observations / surprises: **this is a clean, decisive negative
  control.** Fixing the boundary-contamination bug did NOT meaningfully
  change the outcome, which rules out "boundary contamination" as the
  dominant cause of run -17's poor performance and points back to the
  ORIGINAL hypothesis (400 scatterers is too sparse to form a stable,
  trackable speckle signature) as the more likely explanation. This is
  the value of testing hypotheses individually rather than changing
  multiple things at once — the boundary-contamination fix, tested in
  isolation, cleanly failed to explain the result, telling us where NOT
  to keep looking.
- Next action: **Honest final status of the speckle-tracking pivot after
  three iterations (runs -17, -18): not yet demonstrated to work, with
  the boundary-contamination hypothesis now ruled out and scatterer
  density remaining the leading unexplained variable.** A real next test
  would need a substantially denser scatterer field (thousands, not
  hundreds, per the ~10+ scatterers/resolution-cell rule of thumb for
  fully-developed speckle) — a nontrivial next increment (higher
  computational cost, more careful validation), not attempted this
  session. This closes out this session's speckle-tracking exploration;
  recommend bringing the full arc (Levels 0-4 detector refinement, closed
  run -16; speckle tracking, open question run -17/-18) to a planning
  discussion before further investment, given the depth of exploration
  already done without a clean win on either track.

### Run 2026-07-07-19 — Genuine multi-angle vector triangulation (first positive result)
- Phase: 3 (continuing the multi-angle direction, per explicit user
  request: "simulate multi-angle transmissions and multi-channel
  receive, then fuse into vector motion estimates" — distinct from the
  earlier, simpler Level 4, which only averaged two similar SCALAR range
  measurements). `src/phase3_vector_triangulation.py`.
- Seed / config / grid / timestep: no RNG (noiseless proof-of-concept).
  Target: outer myocardial boundary point at 30 degrees off boresight
  (row=72.06, col=195.00 at ED). Two independently delay-focused 8-element
  sub-apertures (reusing the validated delay-focusing law from
  `phase2_forward_model.py`/`toy_2d_array_source.py`), both fixed-focused
  on this SAME reference target position: sub-aperture A centered
  directly above the target (on-axis, look direction u_A=(1,0), i.e. pure
  boresight), sub-aperture B offset 95 cells (~9.5mm) to one side (look
  direction u_B=(0.405,0.914), 66.1 degrees from u_A). Each sub-aperture's
  echo tracked via the sequential (frame-to-frame) method validated in
  run -18 (template half-width 0.5us, per-step search margin 1.2us).
  Recovered 2D displacement vector (d_row, d_col) per frame by solving
  the 2x2 linear system [u_A; u_B] . d = [range_change_A; range_change_B].
- Acoustic properties + sources: clean (non-speckle) ring phantom, same
  cited tissue properties as throughout Phase 3. This isolates the
  triangulation question from the still-open speckle-density question
  (runs -17/-18) — deliberately tested separately.
- Compute used (GPU-hours) vs budget: none — local CPU, ~1-2 min for both
  sub-apertures across 8 frames.
- Result: **Bug caught and fixed before any result was trusted**:
  sub-aperture B's path length (~9.5mm offset) gives a round-trip time
  (~12.6us) exceeding the previously-used `pmr.time_axis`'s t_end
  (~9.4us, sized only for the shorter on-axis case) — caused an
  empty-array crash. Fixed with a longer, LOCAL time axis (20us, same
  dt) specific to this script, not touching the shared
  `phase3_motion_recovery.py` module's timing.
  **After the fix: genuine vector recovery, the first in this project.**
  Row (boresight) component: RMSE=0.285mm, tracks the true bell-curve
  motion shape well — comparable to this project's best single-direction
  results (Level 2/3, run -16). Col (cross-range) component: RMSE=
  0.499mm, correct SIGN/direction throughout (matches the true inward-
  during-contraction cross-range direction at this off-axis angle) but
  noisier trajectory-matching (e.g. frame 4 shows notable deviation). This
  cross-range information is something NO single-look-direction method in
  this entire session (Levels 0-4, speckle tracking) could recover AT
  ALL — a structurally new capability, not just an incremental accuracy
  gain. Figure: `results/figures/phase3_vector_triangulation.png`.
- Physical sanity checked? by whom?: Claude — noiseless per-frame
  comparison against the known true (row,col) displacement vector before
  reporting the RMSE summary; correct look-direction geometry
  double-checked via the u_A/u_B computation and angle-between (66.1
  degrees, a well-conditioned, non-degenerate triangulation baseline).
- Gate passed? (Y/N): N/A — capability demonstration, not a gated
  deliverable. Noiseless-only so far — noise-robustness of this
  triangulation approach is untested (a natural next step, not done this
  run).
- Observations / surprises: the row/col accuracy asymmetry (row much
  better than col) is plausibly explained by sub-aperture B's oblique
  (off-normal) incidence angle to the curved boundary — a real physical
  effect (reflections at steep angles from a specular-ish curved surface
  are weaker/more geometrically spread than near-normal reflections),
  compounding with sequential-tracking's known drift-accumulation
  tendency (run -18). Not yet disentangled which effect dominates.
- Next action: this is a genuine positive proof-of-concept for the
  multi-angle triangulation strategy — recommend (a) testing under noise
  (same 20-realization-averaging discipline as runs -14 through -16) to
  see if it's robust, and (b) investigating whether the col-component
  noise is dominated by sub-aperture B's oblique-incidence weak echo or
  by sequential-tracking drift, before declaring this a complete
  solution. Still open: whether this combines productively with the
  still-unresolved speckle-density question (runs -17/-18), or is better
  pursued independently against the (working) boundary reflector as done
  here.

### Run 2026-07-07-20 — Vector triangulation under realistic noise: works, and a surprise
- Phase: 3 (continuing run -19, testing noise-robustness as recommended).
  Extended `src/phase3_vector_triangulation.py` with a 20-realization
  noise sweep at the same levels used throughout (0.02/0.05/0.10 =
  34/26/20dB, per run -16's SNR calibration), plus per-sub-aperture
  scalar RMSE tracking (each sub-aperture's range-change vs its OWN
  analytically-expected projection) specifically to disentangle whether
  cross-range noise is dominated by sub-aperture B's oblique echo being
  weaker, or by sequential-tracking drift affecting both similarly.
- Result:

  | noise | row RMSE (mm) | col RMSE (mm) | subA scalar RMSE | subB scalar RMSE |
  |---|---|---|---|---|
  | 0.0 | 0.285 | 0.499 | 0.285 | 0.382 |
  | 0.02 | 0.977 | 0.686 | 0.977 | 0.454 |
  | 0.05 | 0.959 | 0.675 | 0.959 | 0.448 |
  | 0.10 | 0.959 | 0.674 | 0.959 | 0.441 |

  Since u_A=(1,0) exactly, row RMSE equals subA's own scalar RMSE exactly
  (0.2847=0.2847 etc. — confirms the linear-algebra implementation is
  correct, a useful internal consistency check). Figure:
  `results/figures/phase3_vector_triangulation_noise.png`.
- **Honest, surprising finding**: sub-aperture A (on-axis, pure
  boresight, near-NORMAL incidence to the curved boundary) is MORE
  noise-fragile than sub-aperture B (oblique, off-normal incidence) —
  the opposite of what was hypothesized in run -19 (that B's oblique
  echo, being geometrically weaker, would be the noisier one). Under
  noise, sub-aperture A's RMSE roughly TRIPLES (0.285->0.96mm) while
  sub-aperture B's increases only modestly (0.382->0.44-0.45mm, ~15-20%).
  As a direct consequence, the col (cross-range) component — which
  blends both sub-apertures' measurements — ends up MORE accurate than
  the row (boresight) component under noise, a reversal from the
  noiseless case where col was worse.
- Physical sanity checked? by whom?: Claude — verified the row=subA
  identity holds exactly (internal consistency check on the 2x2 solve),
  and confirmed the noise sweep uses the same realistic SNR range and
  20-realization-averaging discipline as runs -14 through -16.
- Gate passed? (Y/N): N/A.
- Observations / surprises: plausible explanation for the counter-
  intuitive finding — a near-normal-incidence reflection off a CURVED
  boundary is the sharpest, strongest specular return, but this may make
  it MORE susceptible to the same wrong-echo/multipath ambiguity that
  limited Levels 0-4 (many candidate strong, similarly-sharp features
  along the curve near boresight), whereas the oblique sub-aperture's
  weaker, more spatially-spread reflection may have fewer comparably-
  strong competing features to be confused with, even though its raw
  signal is weaker. This has NOT been directly verified (e.g. by
  inspecting the actual mis-tracked echoes) — flagged as a plausible but
  unconfirmed explanation, not a established fact.
- Next action: **overall verdict: vector triangulation is a genuinely
  promising direction that DOES survive (to a meaningful degree) the
  realistic noise range that broke every single-direction method in this
  project** — col RMSE (0.67-0.69mm under noise) is comparable to or
  better than this project's best single-direction result (Level 3's
  ~0.86mm, run -16), and importantly recovers a vector, not a scalar. The
  row component's noise-fragility mirrors the SAME ambiguity problem
  diagnosed in Levels 0-4 (run -14's wrong-echo detection), suggesting
  the near-normal sub-aperture would itself benefit from the SAME fixes
  already validated there (e.g. reference-tracking's narrow search
  window, already used here — the residual fragility may need the
  matched-filter/sequential combination tuned further, or accepting that
  near-normal single-view detection has a hard floor this project has
  now characterized several times over). This is a good, well-earned
  stopping point for this session's exploration: a genuinely positive,
  quantified, honestly-caveated result to bring forward.

### Run 2026-07-07-21 — Vector triangulation: run -19/-20 RETRACTED; found the real physical limitation
- Phase: 3 (per explicit user instruction to keep debugging run -19/-20's
  vector-triangulation thread until fully validated).
- Result: **Run -19/-20's "genuinely positive" vector-triangulation
  result is RETRACTED.** Investigating a timing-formula bug (see below)
  led to a deeper, structural finding that invalidates the whole
  approach as originally conceived — not just a bug to patch.
  1. **Timing-formula bug**: the original `t_ref` estimate used a naive
     symmetric approximation (2*dist(receiver,target)/c_ref), assuming
     transmit and receive are co-located. Correct physics for a
     delay-focused sub-aperture: every element's wavefront converges on
     the target SIMULTANEOUSLY at t=max(element_distances)/c_ref, then
     the reflected path adds dist(target,receiver)/c_ref separately.
     Fixed in `expected_round_trip_time()`. Re-running the ORIGINAL 2-way
     (A/B) test with this fix changed the numbers substantially: sub-
     aperture B's noiseless scalar RMSE went from 0.382mm (run -19) to
     0.938mm — the "surprising A-more-fragile-than-B" finding was partly
     built on the buggy timing estimate finding a coincidentally-plausible
     (but likely wrong) feature.
  2. **The deeper, structural finding**: even with corrected timing,
     residual discrepancies (~0.9-2.5us, varying by sub-aperture)
     persisted. Diagnosed via the mirror/virtual-image method (reflecting
     each Tx position across the target's local tangent plane): for a
     CURVED reflecting boundary, delay-focusing the transmit at a chosen
     point only concentrates INCIDENT energy there — it does NOT mean the
     REFLECTED wave reaching a given receiver position actually comes
     from that point. The true specular reflection point (where angle of
     incidence = angle of reflection relative to the LOCAL normal) is
     determined by the FULL (Tx,Rx) bistatic geometry, and for an
     off-axis target (30 degrees), the correctly-paired receiver position
     is wildly different from a small "nearby pitch-catch offset"
     (e.g. for Tx directly above the target, the true specular receiver
     is at column ~268, not ~200 as originally used).
     **Scanned multiple valid (Tx,Rx) pairs (see chat log) and found: the
     sensitivity vector d(range)/d(target position) = (u_in - u_out)/2 is
     ALWAYS exactly parallel to the target's local surface NORMAL,
     regardless of which valid Tx/Rx pair is used** (confirmed
     numerically: unit sensitivity direction (0.866,-0.500) identical
     across 5 different valid Tx positions spanning a wide range of
     incidence angles). This is a geometric fact about specular
     reflection (law of reflection implies u_in and u_out are symmetric
     about the normal, so their difference is always normal-directed),
     not an artifact of this implementation.
- Physical sanity checked? by whom?: Claude — derived and verified the
  virtual-image/mirror-reflection formula analytically, confirmed it
  reproduces the same result as the direct reflection-law calculation,
  and empirically scanned 7 transmit positions to confirm the
  sensitivity-direction invariance was not a one-off coincidence.
- Gate passed? (Y/N): N/A.
- **This is a structural/conceptual finding, not a bug fix opportunity**:
  multi-angle triangulation of a SINGLE point via specular reflection
  CANNOT recover a true 2D displacement vector, no matter how many
  angles or how correctly implemented, because specular reflection
  physically only encodes the normal-direction component of target
  motion — the tangential component is invisible to it by construction.
  The apparent "cross-range recovery" reported in run -19/-20 was an
  artifact of mismatched (non-specular) Tx/Rx geometry, not genuine
  physics. Genuine 2D/vector motion recovery requires either (a)
  distributed scatterer/speckle tracking (still unresolved, runs -17/-18
  — a discrete scatterer's arrival-time DOES depend on true 2D position,
  unlike smooth specular reflection), or (b) tracking multiple DIFFERENT
  points and inferring tangential strain/rotation from the SPATIAL
  variation of their individual (normal-only) displacements — the actual
  mechanism real 2D speckle-tracking echocardiography uses.
- Next action: retract the positive framing of runs -19/-20 in
  `PIPELINE_STATUS_AND_ROADMAP.md` and `MANIFEST.md`, replacing it with
  this structural finding. Do not pursue further "multi-angle specular
  triangulation of one point" variants — the limitation is physical, not
  implementation-dependent, so further iteration on this specific
  approach will not produce a different answer. The two viable paths
  forward (distributed speckle tracking; multi-point differential
  strain) both connect back to the STILL-UNRESOLVED speckle-density
  question from runs -17/-18, which is now the single most important
  open thread for genuine vector/strain motion recovery in this project.

### Run 2026-07-08-22 — DAS beamforming: 4 real bugs fixed, 1 genuine open theory question
- Phase: 3 (pivot #2, per explicit user direction: reconstruct a full
  spatial image via delay-and-sum beamforming, then track features
  within it -- avoids the specular-normal-only limitation found in run
  -21). `src/phase3_das_beamforming.py`.
- Design: ONE broad transmit (initially plane-wave, later delay-focused)
  + many virtual receive channels (extracted free from the same
  simulation's full field) + standard per-pixel delay-and-sum, on a
  coarser (150x150) image grid than the 300x300 sim domain. Reuses the
  clean (non-speckle, non-attenuating) ring phantom, testing image-
  formation validity as a separable question from attenuation/speckle.
- Result: **four distinct, real bugs found and fixed in sequence, each
  caught by explicit validation against known ground truth before
  trusting the next step (this project's established discipline):**
  1. **Off-by-one time-axis bug**: manually computed `n_steps` via
     `round(t_end/dt)`, but `TimeAxis.Nt` uses `ceil()` -- length
     mismatch crashed `np.interp`. Fixed by reading `time_axis.Nt`
     directly instead of recomputing.
  2. **Axis-order bug**: this script indexed `field[:, array_y,
     channel_xs]` (row-like index first), the REVERSE of the
     established convention used in every other script this session
     (`field[:, x_value, y_value]`, column-like first). Caused a
     strongly asymmetric, clearly-wrong reconstruction. Fixed to match
     the established convention.
  3. **Unfocused-transmit signal-to-clutter problem**: with a simple
     point-source transmit, the true boundary echo (0.00013 amplitude)
     was ~1700x weaker than the direct wave (0.22) -- not a bug, a real
     physics consequence of radiating energy in all directions instead
     of concentrating it toward the region of interest. Fixed by
     switching to a delay-focused transmit (reusing the validated
     focusing law from `phase2_forward_model.py`), which raised the
     boundary echo to become the single strongest feature in the trace
     (0.374, now the trace's global max).
  4. **Before-vs-after-focus formula bug**: initially modeled the
     post-transmit wave via the standard "virtual point source at the
     focus" approximation, valid only in the far field BEYOND focus --
     but the myocardial boundary sits BEFORE the focus (ring center),
     in the still-converging near field, where that shortcut is wrong.
     Fixed by computing the true earliest per-pixel arrival time
     directly (min over all elements of delay_i + travel_time_i),
     valid everywhere. **This fix alone brought ED's reconstruction
     error down to 0.61mm** (from being unrecognizable/stuck before).
- **A fifth issue was found and diagnosed, but NOT fixed — a genuine
  open question, not a quick bug**: ES's reconstruction remains
  confounded by a fixed-depth (row~66) artifact that also appears in ED
  (where it happens to sit close to ED's true boundary at row 60,
  masking the problem there) and STAYS AT THE SAME DEPTH regardless of
  frame. **Confirmed via a clean control test**: the identical artifact
  (same row, same amplitude ~0.252) appears even in a PURELY
  HOMOGENEOUS medium with no ring/reflector at all -- proving it is a
  pure array/beamforming-geometry artifact (likely the direct,
  unreflected transmit wave being coherently mis-summed by an
  inconsistency between the "earliest-arrival" transmit-time model and
  the receive-delay model), not a tissue signal.
- Physical sanity checked? by whom?: Claude — each of the 4 fixes was
  validated by comparing against known ground truth (expected boundary
  row) before proceeding to the next; the 5th issue was isolated via a
  deliberate homogeneous-medium control test (no ring) specifically
  designed to separate array-artifact from tissue-signal explanations.
- Gate passed? (Y/N): N/A.
- Next action: **the real fix for the row-66 artifact requires either
  (a) extracting the transmit arrival-time map directly from simulated
  reference data instead of an analytic per-element formula, or (b) a
  more careful mathematical treatment of coherent multi-element
  summation (proper beamforming point-spread-function theory) — a
  meaningfully larger next unit of work, not a quick patch.** Per
  explicit user/session decision, this DAS thread is paused here with
  real, logged progress (from "completely broken/stuck" to "ED
  reconstructs to 0.61mm, root cause of the remaining ES issue
  conclusively isolated") rather than pushed further reactively. See
  `SESSION_HANDOFF_2026-07-08.md` for the full consolidated status
  across every thread this session.

### Run 2026-07-08-23 — 4-probe boundary tracking around a bounding square
- Phase: 3 (per explicit user request: 4 probes at the middle of each
  side of the 2D square bounding the heart model, each doing per-frame
  single-pulse transmit/receive). `src/phase3_four_probe_tracking.py`.
- Design: reused the already-validated Level 2 reference-tracking
  detector (`phase3_reference_tracking_recovery.py`) and the exact same
  probe-to-center distance (120 cells = 12mm, matching
  `phase3_motion_recovery.py`'s single top probe) at 4 positions: top,
  bottom, left, right, each pointing inward toward the ring center. Each
  probe is a simple pitch-catch pair (10-cell/1mm spacing, same as the
  validated single-probe setup), oriented appropriately for its side
  (top/bottom probes offset along columns; left/right probes offset
  along rows) — a 90-degree-rotated reuse of the validated single-probe
  geometry, not a new detection method.
- Result: **all 4 probes gave IDENTICAL RMSE (0.5625mm)** — exactly
  matching the symmetry expectation for this isotropic (pure radial-
  scaling) ring phantom, and confirming the tracking method generalizes
  correctly across all 4 orientations with no rotation-dependent bugs
  (unlike the axis-order issues found in the DAS beamforming thread,
  run -22). All 4 tracked curves overlap essentially perfectly and
  correctly reproduce the true bell-curve motion shape, with the same
  small systematic bias (~0.5-0.6mm) seen in the original single-probe
  Level 2 validation (run -15/-16). Figure:
  `results/figures/phase3_four_probe_tracking.png`.
- Physical sanity checked? by whom?: Claude — the 4-way symmetry
  agreement (identical RMSE to 4 decimal places) is itself a strong,
  built-in consistency check; no separate ground-truth mismatch was
  found for any of the 4 orientations.
- Gate passed? (Y/N): N/A — capability demonstration.
- Observations / surprises: none — this worked cleanly on the first
  attempt, in contrast to every other thread this session (which each
  required multiple bug-fix iterations). Likely because it reuses
  already-validated code (Level 2's detector, the established field-
  indexing convention, the same probe distance) rather than introducing
  new physics or a new detection paradigm.
- Next action: for this SYMMETRIC toy phantom, 4 probes agreeing is a
  validation of the method, not new information (by construction, an
  isotropic phantom gives identical readings from every direction). The
  natural next step for this to become genuinely informative is either
  (a) an ASYMMETRIC phantom (e.g. non-uniform contraction, or real
  patient anatomy) where the 4 probes would meaningfully disagree,
  directly giving a coarse 2D deformation picture (4 independent
  boundary points around the heart), or (b) combining this with the
  still-open DAS beamforming thread (run -22) for full-field imaging
  rather than 4 discrete point measurements.

### Run 2026-07-08-24 — Asymmetric (regional hypokinesis) phantom + focused-probe artifact confirmed unified with DAS thread
- Phase: 3 (continuing run -23, per explicit user request: test the 4
  probes on an ASYMMETRIC motion model, not just the symmetric
  validation). `src/phase3_asymmetric_phantom.py`,
  `src/phase3_four_probe_asymmetric.py`, `src/phase3_four_probe_focused.py`.
- Design: added an angle-dependent LV radius (`local_lv_radius(theta,
  phase)`), modeling regional hypokinesis (a clinically realistic
  scenario, e.g. post-infarct reduced wall motion): normal contraction
  everywhere except a 60-degree-half-width region centered at the LEFT
  probe's direction, smoothly tapered (raised-cosine, no sharp
  discontinuity) down to 30% of normal contraction amplitude at its
  center. Verified the angular convention and regional-factor function
  numerically before running any simulation (top/right/bottom=1.0,
  left=0.3, matching design intent exactly).
- Result (unfocused pitch-catch probes, matching run -23's design):
  **prediction did NOT hold cleanly.** Right and bottom (far from the
  hypokinetic region) tracked consistently with each other (~2.35mm
  recovered vs 1.90mm expected — same ballpark, some over-estimate).
  Top (should be fully unaffected, 90 degrees from the hypokinetic
  center) unexpectedly showed REDUCED apparent contraction (0.778mm vs
  1.901mm expected). Left (predicted to show the LARGEST reduction, 0.57mm
  expected) instead tracked motion resembling NORMAL contraction, the
  opposite of the prediction. Hypothesis: unfocused (simple 2-element
  pitch-catch) probes have wide angular sensitivity, picking up a blend
  of reflections from a wider angular region than directly on-axis, not
  a narrow beam — explaining why left picks up adjacent (stronger,
  non-hypokinetic) tissue instead of cleanly isolating its own region.
- Follow-up (focused probes): rebuilt each probe as an 8-element,
  delay-focused array (reusing the validated focusing law from
  `phase2_forward_model.py`) aimed at the ring center, to test whether
  narrowing the beam resolves the mismatch. **Result: all 4 probes now
  show EXACTLY ZERO recovered contraction — completely stuck, no motion
  detected at all, for every frame.** This is the SAME failure signature
  as the still-open DAS beamforming artifact (run -22's row-66 fixed-
  depth artifact, confirmed via a homogeneous-medium control test to be
  a pure array/focusing-geometry effect, not a tissue signal) —
  multi-element delay-focusing appears to create a strong, FIXED
  coherent-summation artifact that dominates over the true, weaker,
  moving tissue echo, causing the tracker to lock onto the artifact
  instead of real motion.
- Physical sanity checked? by whom?: Claude — verified the regional-
  factor function's numeric values before trusting any simulation;
  recognized the "stuck at identical value across probes" pattern (same
  tell used throughout this session) as diagnostic of a bug/artifact
  rather than a genuine physical result.
- Gate passed? (Y/N): N/A.
- **Per explicit user decision: this is now understood as ONE unified,
  recurring open problem (multi-element delay-focusing produces a fixed
  coherent-summation artifact that can dominate real tissue signal),
  not two separate issues to debug independently in each new script.**
  Consolidating here rather than continuing a third independent
  debugging cycle.
- Next action: any future attempt to use multi-element FOCUSED transmit
  in this project (DAS beamforming, focused probes, or otherwise) should
  first resolve this unified artifact question — likely requiring either
  a data-driven (simulated-reference) arrival-time/artifact map, or a
  more rigorous mathematical treatment of coherent multi-element
  summation, per the DAS thread's existing diagnosis (run -22). The
  UNFOCUSED 4-probe result (run -23, run -24's first part) remains a
  valid, real finding on its own: it demonstrates that simple point-like
  probes correctly recover motion MAGNITUDE (per the isotropic
  validation, run -23) but do NOT reliably LOCALIZE regional motion
  differences — a genuine, informative capability limitation, distinct
  from and not resolved by the (currently broken) focused-probe attempt.

### Run 2026-07-08-25 — Beating-circle DAS reconstruction movie: partial success, honest threshold effect found
- Phase: 3 (per explicit user request: simplify to a single filled
  circle, reconstruct it visually as a "movie" across the cardiac
  cycle). `src/phase3_beating_circle_movie.py`.
- Design: simplified from the myocardial ring (two boundaries) to a
  single filled disk (blood, cited properties) in a chest-wall-proxy
  background — ONE boundary, reducing complexity. Reused the validated,
  bug-fixed DAS reconstruction from `phase3_das_beamforming.py` (correct
  per-element earliest-arrival transmit-time formula). **Artifact
  cancellation via background subtraction**: since the recurring fixed-
  depth coherent-summation artifact (runs -22, -24) was already proven
  medium-independent (identical in a homogeneous medium with no
  reflector), subtracted a one-time homogeneous-medium reconstruction
  from every frame — principled (not a blind hack), since the control
  test already established the artifact doesn't depend on what's in the
  medium. 12 frames swept across a full cardiac cycle (ED->ES->ED).
- Result: **Partial success, with an honest, characterized failure
  mode, not a clean full-cycle win.**
  - **Frames near ED** (phases 0.00/0.09/0.18 and symmetric 0.82/0.91/1.00,
    r=54-60 cells): reconstruction correctly shows a bright arc at the
    TRUE circle boundary position (visually confirmed against an
    overlaid true-boundary curve, and quantitatively: peak within ~5
    cells/0.5mm of the expected row, meaningful peak amplitude
    0.001-0.0033).
  - **Frames in the contracted middle of the cycle** (phases 0.27-0.73,
    r=40-48.6 cells): peak STUCK at the identical row (60.7) across FIVE
    different frames — the same "locked onto something fixed" signature
    used throughout this session to detect bugs/artifacts. Peak
    amplitude there is also much smaller (0.0008) and constant.
  - Figure (visual filmstrip, all 12 frames with true-boundary overlay):
    `results/figures/phase3_beating_circle_movie.png`.
- Physical sanity checked? by whom?: Claude — quantified per-frame peak
  position/amplitude against known ground truth (not just visual
  impression) before characterizing the result; recognized the repeated-
  identical-value pattern across 5 frames as the established bug/
  artifact signature, not a coincidence.
- Gate passed? (Y/N): N/A.
- **Honest diagnosis**: the background subtraction is NOT a perfect
  cancellation -- introducing the circle changes local wave propagation
  near the array somewhat (the medium isn't IDENTICAL to the homogeneous
  reference once the circle is present, even far from the circle
  itself, due to how the pseudospectral solver's global basis functions
  respond to any medium change), so a residual artifact remains after
  subtraction. For larger (ED-adjacent) circles, the true echo is
  strong enough to exceed the residual. For smaller (more contracted,
  farther-from-array) circles, the weaker true reflection drops below
  the residual artifact's level, and the tracker locks onto the leftover
  artifact instead. **This is a real, characterizable, signal-strength-
  dependent threshold effect** -- not a mystery failure, and consistent
  with (not a new instance of) the unified focusing-artifact problem
  already logged in runs -22/-24.
- Next action: this demonstrates the visual/movie approach IS viable and
  informative (clear success visible for roughly half the cycle), but
  confirms the underlying artifact problem (runs -22/-24) still needs a
  real fix -- background subtraction is a useful partial mitigation, not
  a substitute for solving it properly. A cleaner artifact-cancellation
  approach might subtract a MATCHED reference per frame (e.g. same
  overall wave energy/attenuation state) rather than one static
  homogeneous reference, or address the root cause per the standing
  recommendation in `SESSION_HANDOFF_2026-07-08.md`.

### Run 2026-07-08-26 — Bidirectional sequential tracking: 10/12 frames recovered from wall reflections alone
- Phase: 3 (continuing run -25, per user question: "can you reconstruct
  the circle's full cardiac cycle based only from what the wall gives
  off and receive?"). `src/phase3_beating_circle_movie.py`, extended.
- Design: run -25's per-frame GLOBAL peak search found the true boundary
  for ED-adjacent frames but locked onto the fixed residual artifact
  (row 60.7) for contracted frames. Fix: SEQUENTIAL search, constraining
  each frame's peak search to a narrow window (+/-15 cells) around the
  PREVIOUS frame's confirmed position — verified first that max per-step
  radius change (5.58 cells) is comfortably smaller than the gap to the
  fixed artifact (29.3 cells) before trusting this would work.
  First attempt (pure forward sequential, frame 0->11): correctly tracked
  frames 0-3, then DRIFTED once the true signal weakened (frame 4
  onward) and got PERMANENTLY stuck on the artifact for frames 7-11 —
  once drifted, a forward-only tracker has no way back (same drift-
  accumulation risk flagged for sequential speckle tracking, run -18).
  **Fix: exploited the cycle's known symmetry** (ED at both the first and
  last frame, ES in the middle) — track FORWARD from frame 0 AND
  BACKWARD from the last frame simultaneously, meeting at the midpoint,
  so drift only has ~half the frames to accumulate from either direction
  instead of all 11 from one end.
- Result: **10 of 12 frames now track correctly** (frames 0-4 and 7-11,
  errors 0.08-1.52mm, visually confirmed — the tracked marker sits
  directly on the true boundary in the figure). Only frames 5-6 (the
  deepest-contraction, smallest-radius frames, right at ES) remain
  wrong (tracked_row=78.7 vs expected ~109.6, ~3cm... i.e. large error) —
  both forward and backward passes run out of "runway" (5-6 steps of
  accumulated drift risk from either anchor) exactly at the point where
  the true signal is also physically weakest (smallest, farthest-from-
  array circle). Figure (updated with tracked-position markers):
  `results/figures/phase3_beating_circle_movie.png`.
- Physical sanity checked? by whom?: Claude — computed the max per-step
  motion vs the artifact gap analytically before trusting the narrow-
  window approach; visually confirmed the tracked marker position
  against the true boundary overlay for all 12 frames, not just the
  summary numbers.
- Gate passed? (Y/N): N/A.
- **Answering the user's question directly**: yes, mostly — using ONLY
  the reflected wall data (transmit + receive at the boundary,
  reconstructed per-frame via DAS with artifact cancellation, tracked
  bidirectionally), the full cardiac cycle reconstructs correctly for
  ~83% of frames (10/12) at sub-1.5mm accuracy. The remaining gap (peak
  contraction) is an honest, physically-explained residual — the true
  signal is weakest exactly where the circle is smallest and farthest
  from the array — not an unexplained algorithm failure.
- Next action: the remaining 2-frame gap at peak contraction could
  potentially be further narrowed by (a) a genuinely fixed root-cause
  solution to the focusing artifact (per the standing recommendation,
  runs -22/-24), which would raise the SNR floor everywhere including at
  ES, or (b) interpolating through the low-confidence frames using the
  now-validated bidirectional anchors on either side (a reasonable,
  honestly-labeled approximation, not a new measurement). This result is
  a genuine, meaningful capstone for the simplified single-circle sanity
  test the user requested — ready to inform whether/how to return to the
  harder full myocardial-wall case.

### Run 2026-07-08-27 — Doppler/MTI-style frame-differencing fusion: no net improvement, ES gap confirmed not a tracking-algorithm artifact
- Phase: 3 (continuing run -26). Per user question: "if signals get weak
  during systole, why don't you use the doppler wave shift information
  to calculate exactly where it was reflected?" `src/phase3_beating_circle_movie.py`,
  extended with a second tracking cue.
- Design: added frame-to-frame differencing (`frames[i] - frames[i-1]`,
  the discrete-time analogue of a Doppler/MTI wall filter — cancels the
  static array/artifact pattern more directly than the homogeneous-medium
  reference subtraction already in use, since both frames share identical
  real geometry). For each frame, computed BOTH cues' peak position and
  peak amplitude (confidence), then fused by taking whichever cue had
  higher confidence at that specific frame.
- Result: **fusion is a wash, not a net win.** Per-frame errors (mm):
  frame0 0.50 (amp, anchor), 1: 0.52 (diff), 2: 0.46 (amp), 3: 0.10
  (diff), 4: 0.43 (diff) — improved from amp-only's 1.51mm, 5: 3.08
  (amp), 6: 3.08 (amp), 7: 1.52 (amp), 8: 0.08 (amp), 9: 0.46 (amp),
  10: 0.88 (diff) — WORSE than amp-only's 0.52mm, 11: 0.68 (diff).
  Confidence-based selection correctly picked differencing where it
  helped (frame 4) but also picked it where it hurt (frame 10) — peak
  signal amplitude is not a fully reliable confidence proxy for which
  cue is geometrically more accurate. Frames 5-6 (ES, deepest
  contraction): fusion did NOT recover them — differencing's confidence
  was exactly zero at these frames (checked: peak_val=0.0000, since
  wall velocity crosses zero at ES, a real physical null in any
  velocity-based/Doppler method, not a bug), so the selector always fell
  back to the already-failing amplitude method (tracked_row=78.7 vs
  expected ~109.6, unchanged from run -26).
- **Answering the user's Doppler question directly**: differencing
  (the Doppler/MTI mechanism) is real and does help at some frames (e.g.
  frame 4: 1.51mm -> 0.43mm) where wall velocity is high, confirming the
  intuition. But it cannot rescue the ES gap specifically BECAUSE ES is
  defined by near-zero wall velocity — the same physical quantity a
  Doppler-based method depends on to have any signal at all. This is a
  known, clinically-documented limitation of Tissue Doppler Imaging (zero
  velocity at end-systole/end-diastole), not an implementation gap. Two
  independent methods (amplitude-based range tracking, velocity-based
  differencing) now agree the ES frames are a genuine measurement floor
  for this array geometry/SNR, not an artifact of either specific
  algorithm.
- Physical sanity checked? by whom?: Claude — verified the zero-velocity
  explanation directly from the printed confidence values (diff_conf=0.0
  exactly at frames 5-6) rather than assuming it; compared fused vs.
  amplitude-only per-frame instead of only looking at the aggregate to
  catch the frame-10 regression.
- Gate passed? (Y/N): N/A.
- Next action: the ES gap should now be treated as a settled, dual-method-
  confirmed measurement floor for this specific setup (single focused
  aperture, this SNR/attenuation regime) rather than something to keep
  attacking with new detectors on the same data. Further progress there
  would need either (a) the unified focusing-artifact root-cause fix
  (runs -22/-24, still the highest-leverage open item — would raise SNR
  everywhere including at ES), or (b) an honestly-labeled interpolation
  through frames 5-6 using the validated neighbors, not a new
  measurement. Recommend logging this as closed-for-now and returning to
  the standing priority list in `SESSION_HANDOFF_2026-07-08.md`.

### Run 2026-07-08-28 — Multistatic backprojection ("LIDAR-style" convergence): RMSE=0.24mm, no fixed artifact, no per-frame failure mode
- Phase: 3. Per user's explicit re-diagnosis of every earlier single-echo
  method: "every wave is a true reflection of the surface... a single
  return signal read by a single probe point has many possible
  candidates... what I want is, for a single probe read, you should be
  able to read all possible locations of the reflection at a specific
  time point, and map it. The convergence/overlap/cluster of all probes
  on that wall at a given timepoint is what determines where the actual
  surface is, similar to how a LIDAR works." This is standard multistatic
  backprojection / the ultrasonic-NDT "Total Focusing Method" —
  mathematically the same idea as LIDAR multilateration: sweep every
  candidate point P in the domain; for each (tx, rx) probe pair, compute
  the exact travel time P would produce; sample that pair's
  envelope-detected trace at that time; SUM across all pairs. The true
  surface is wherever many independent pairs agree, replacing every
  earlier method's single-probe "pick one echo, hope it's the right one"
  step with an explicit accumulator over the whole hypothesis space.
- Design: new script `src/phase3_multistatic_backprojection.py`. Same
  beating-circle phantom and 4-probe geometry (top/bottom/left/right,
  12mm from center) as runs -23..-27, but: each of the 4 probes fires in
  turn (unfocused single-element pulse, not the focused 32-element
  transmit used in DAS) while ALL 4 probes' receive elements record —
  4 tx x 4 rx = 16 tx/rx pairs per frame. Each pair's trace is
  envelope-detected (Hilbert transform) and the direct (unreflected)
  arrival is zeroed out with a time margin (generalizing DIRECT_EXCLUDE_S
  to all 16 pairs). Backprojection accumulates all 16 pairs'
  envelope-at-predicted-time over a full 2D grid (not just one column);
  radius is read off by binning the accumulator by distance from center
  and taking the peak bin — using the FULL multistatic image, not a
  per-probe threshold pick.
- First run (naive travel-time formula, dist/c per leg): tracked radius
  systematically LARGER than true by 1.1-1.7mm across all 8 frames.
  Diagnosed (not patched blindly): the Hilbert envelope of a windowed
  N_CYCLES-toneburst peaks `duration/2` AFTER the naive geometric
  (instantaneous) arrival time, since the transmitted pulse's own
  envelope is centered at t=duration/2, not t=0 — every backprojected
  point's predicted arrival time was too early by this fixed group delay,
  which the peak-finder compensated for by drifting outward in radius.
  Added the `duration/2` correction to the travel-time formula and
  reran.
- Result (corrected): **RMSE=0.24mm across all 8 frames, zero failure
  frames.** Per-frame errors: 0.15, 0.23, 0.17, 0.35, 0.35, 0.17, 0.23,
  0.15mm — the largest errors occur at the two most-contracted (ES-
  adjacent) frames, same physical trend as every earlier method (weakest
  signal at smallest/farthest-from-array radius), but the absolute size
  of that residual (0.35mm) is now smaller than run -26/-27's BEST
  frames, and nothing resembling those methods' catastrophic ES failure
  (3+mm error, stuck on a fixed value) appears anywhere in the cycle.
  **Control test (homogeneous medium, no circle) confirmed the key
  hypothesis behind this whole approach**: accumulator peak value
  0.0000 — i.e., no fixed/coherent artifact survives incoherent
  multistatic combination, unlike DAS's persistent row-66 ghost (runs
  -22, -24, present in every focused-transmit method tried this
  session). This is the first method this session with NO diagnosed
  residual artifact of any kind.
- Physical sanity checked? by whom?: Claude — ran the homogeneous-medium
  control BEFORE trusting the main result (established practice this
  session); diagnosed the systematic radius bias analytically (envelope
  group delay) rather than curve-fitting a correction, then verified the
  diagnosis was correct by confirming the bias shrank to sub-half-cell
  residual after the fix, not just "improved."
- Gate passed? (Y/N): N/A.
- **Answering the user's proposed mechanism directly**: yes — reading
  "all possible locations" per probe (the full circle/ellipse of
  candidate reflectors consistent with one arrival time) for every
  probe, and only trusting where they overlap, is both conceptually
  correct AND, on this phantom, a substantial empirical improvement over
  every single-echo-threshold method attempted this session (Thread 1,
  closed; runs -23/-24/-26/-27) — it resolves exactly the "single-probe
  many-candidates" ambiguity the user identified, and structurally
  avoids the fixed-artifact problem that blocked Threads 4/5
  (runs -22/-24, still unsolved for the focused-DAS/focused-probe
  approach) because a static geometric artifact would have to
  coincidentally satisfy all 16 independent pairs' delay-consistency
  conditions at once.
- Next action: this result is strong enough to warrant testing on the
  harder cases this session had parked: (a) the full myocardial RING
  phantom (two boundaries, not one disk) instead of the simplified
  single circle, (b) the ASYMMETRIC (regional hypokinesis) phantom
  (runs -24) where Thread 5's unfocused-probe attempt failed to localize
  correctly and the focused-probe attempt hit the fixed-artifact wall —
  multistatic backprojection may resolve both, since it neither depends
  on picking one probe's threshold echo nor on coherent multi-element
  focusing. Also worth testing under realistic noise levels (SNR
  2%/5%/10%, per `PROXY_AUDIT.md`) to check whether the accumulator's
  incoherent-summation SNR gain survives this session's noise
  calibration, before treating this as a settled replacement for the
  earlier methods.

### Run 2026-07-08-29 — Multistatic backprojection on a beating TRIANGLE: edges/facets generalize cleanly, corner exposes a new diagnosed artifact (colinear probe pairs)
- Phase: 3 (continuing run -28). Per user request: "try a beating
  triangle first, and if things smooth i'll go to sleep" — a smoke test
  of whether the multistatic backprojection idea generalizes past a
  smooth, circularly-symmetric reflector. `src/phase3_multistatic_backprojection_triangle.py`.
- Design: identical 4-probe/16-pair multistatic accumulator, envelope
  detection, direct-arrival exclusion, and group-delay correction as run
  -28 (unchanged) — only the phantom changed to an equilateral triangle
  (one vertex pointing at the "top" probe, circumradius R following the
  same ED/ES schedule as the circle). This puts 3 different reflection
  geometries under one probe layout: "top" hits a bare VERTEX (point/
  corner scattering, weak and diffuse) head-on; "bottom" hits the
  midpoint of the OPPOSITE EDGE at normal incidence (should behave like
  the circle case); "left"/"right" each hit an oblique FACET at
  non-normal incidence (mirror images of each other). Ground-truth
  distances derived by hand from equilateral-triangle geometry: top=R,
  bottom=R/2 (inradius), left=right=R/sqrt(3).
- Result: **NOT a clean pass — 3 of 4 directions generalize well, the
  4th has a real, diagnosed new failure mode.**
  - bottom (normal-incidence edge): RMSE=0.14mm — matches the circle's
    precision (run -28), confirms the method isn't circle-specific.
  - left/right (oblique facets): RMSE=0.42mm / 0.95mm — good, though
    left has one outlier frame (1.88mm, mid-cycle) not yet explained.
  - **top (bare vertex): RMSE=2.46mm, wrong in EVERY frame** — tracked
    distance is consistently close to HALF the true value (e.g.
    R=60: tracked=31.8 vs true=60.0), and closely matches that same
    frame's BOTTOM (apothem) distance instead.
  - **Diagnosed root cause (not left as a mystery)**: the top and bottom
    probes sit exactly colinear through the domain center. For that one
    tx/rx pair, the constant-round-trip-time locus (normally an ellipse)
    DEGENERATES to the entire line segment between the two probes — any
    point on that axis gives an IDENTICAL predicted travel time (sum of
    distances to two colinear foci with the point between them = the
    fixed foci separation, independent of position). This pair therefore
    contributes a smeared, non-localizing ridge along exactly the same
    vertical axis being peak-searched for both "top" and "bottom". Run
    -28 (the circle) never exposed this because the circle's boundary
    reflection was always strong enough to dominate the ridge; the
    triangle's bare-vertex return is much weaker (diffuse corner
    scattering vs. a smooth specular surface), so here the ridge — and
    secondary structure tied to the real, strong bottom-edge echo —
    wins the peak search instead of the true, weaker vertex signal.
- Physical sanity checked? by whom?: Claude — computed exact ground-
  truth distances by hand from triangle geometry rather than assuming
  radial symmetry (correctly identified the triangle is NOT circularly
  symmetric, so the run -28 radial-binning approach would not apply, and
  built a per-axis peak search instead); did not accept "top is wrong"
  as a dead end — traced the wrong values to the bottom-edge apothem
  numerically, then derived the colinear-locus-degeneracy explanation
  from first-principles ellipse geometry and confirmed it's consistent
  with why run -28's circle never showed this.
- Gate passed? (Y/N): N/A.
- **Answering the user's implicit question ("did it stay smooth?")
  honestly**: partially. Edges and oblique facets — the majority of the
  boundary, and the harder non-normal-incidence case — generalize the
  circle result cleanly, which is real, useful evidence this isn't a
  circle-specific fluke. But a genuinely new, previously-invisible
  failure mode appeared at the corner, caused by a specific, diagnosed
  geometric artifact of this exact 4-probe layout (opposite probes being
  exactly colinear through the target), not by an ambiguous echo-picking
  problem the way every earlier (pre-backprojection) method failed. This
  is a fixable design issue (e.g. exclude same-axis colinear pairs from
  contributing at that axis, or add a 5th/45-degree-offset probe so no
  two probes are ever exactly antipodal through center), not a
  refutation of the underlying idea.
- Next action: given the user's stated intent to stop if things were
  smooth — they were NOT fully smooth, so this should be flagged clearly
  rather than reported as a clean success. Left as an open, well-
  characterized item for a future session: (a) fix the colinear-pair
  degeneracy (drop or down-weight pairs whose tx/rx/candidate-point are
  near-colinear before backprojecting), (b) explain the left-side
  outlier frame, (c) then retest the corner case.

### Run 2026-07-08-30 — 8-probe fix attempt FAILED: vertex tracking unchanged, left-facet tracking got WORSE — run -29's colinear-pair diagnosis disproven
- Phase: 3 (continuing run -29). Per user suggestion: split each of the
  4 wall-midpoint probes into 2, placed at the 1/3 and 2/3 points along
  each wall (8 probes, 64 tx/rx pairs total) — directly targeting run
  -29's diagnosis that exactly-antipodal probe pairs (top/bottom,
  left/right, each exactly colinear through the domain center) created a
  degenerate constant-delay LINE locus that swamped the weak vertex
  echo. `src/phase3_multistatic_backprojection_triangle_8probe.py`
  (copy of run -29's script, only the PROBES geometry changed — same
  accumulation machinery, group-delay correction, ground-truth
  formulas). One bug found and fixed before the real run: `PROBE_DIST_CELLS
  // 3` used integer division after an initial float-division attempt
  crashed jWave's source-scatter indexing (`IndexError: Indexing mode
  not yet supported`, needs int grid indices, not float).
- Result: **the fix did not work — this is a genuine, informative
  negative result, not a smooth success.**
  - top (vertex): RMSE=2.28mm — statistically unchanged from run -29's
    2.46mm. Per-frame tracked values (33.6, 31.8, 26.4, 22.7...) are
    nearly IDENTICAL to run -29's 4-probe values (31.8, 30.0, 24.5,
    20.9...), both still tracking close to that frame's BOTTOM
    (apothem) distance rather than the true vertex distance. Since
    removing the exactly-on-axis probes did not move this number at
    all, **run -29's diagnosis (simple exact-colinear-pair degeneracy)
    is disproven, or at minimum incomplete** — something else is pulling
    the "top" search onto the bottom-edge signal regardless of exact
    probe placement (leading unconfirmed hypothesis: a mirror/ghost
    contribution from the strong real bottom-edge echo, arising through
    some OTHER pair combination than the single pair originally blamed
    — not yet identified).
  - bottom (edge): RMSE=0.31mm, slightly worse than run -29's 0.14mm.
  - **left (facet): RMSE=1.87mm, clearly WORSE than run -29's 0.95mm**
    (every frame regressed: e.g. 2.08mm vs 0.10mm at R=60). Plausible
    explanation: the single wall-midpoint probe was already at the
    geometrically ideal near-normal-incidence look angle for this
    facet; splitting it into two off-center probes moved both away from
    that optimum, net loss.
  - right (facet): RMSE=0.42mm, unchanged from run -29.
- Physical sanity checked? by whom?: Claude — compared per-frame tracked
  values directly against run -29's numbers (not just RMSE) to confirm
  the top failure is the SAME failure, not a new one at a different
  location, before concluding the original diagnosis was insufficient;
  did not rationalize away the left-side regression, reported it as a
  real cost of this change.
- Gate passed? (Y/N): N/A.
- **This is a valuable negative result**: it shows (a) "add more probes"
  is not automatically better — probe placement relative to each
  target's local specular-normal direction matters, and moving off an
  already-good position has a real cost (left regressed); (b) the
  vertex/corner failure has a deeper or different cause than the one
  hypothesized in run -29, now open again with the leading hypothesis
  (colinear-pair degeneracy) ruled out as the sole or primary cause.
- Next action: stopping here for the night per the user's stated
  condition ("if things smooth i'll go to sleep") — they were not.
  Recommend, for a future session: (a) don't guess at another geometric
  probe-placement fix blindly — first isolate WHERE the spurious "top"
  peak's energy is coming from (e.g. zero out one pair at a time and
  rerun the accumulator to find which pair(s) actually produce the
  false peak, rather than reasoning about it analytically only); (b)
  revert left/right probes to the run -29 wall-midpoint placement (that
  was working) if only the vertex case is being iterated on next.

### Run 2026-07-08-31 — Pair-ablation diagnostic identifies the ghost (2 specific pairs), coherence-factor fix rejected, targeted pair-exclusion WORKS — triangle thread CLOSED
- Phase: 3 (continuing runs -29/-30). Per user request ("test it and
  produce visuals, must close this thread before moving to the heat
  model"), and following the recommendation from run -30: isolate the
  false-peak mechanism directly instead of guessing another geometric
  fix.
- **Step 1 — diagnostic** (`src/phase3_backprojection_pair_diagnostic.py`):
  captured all 16 tx/rx pair traces ONCE for the ED frame (R=60, the
  exact case measured wrong in runs -29/-30) plus the homogeneous
  reference — no new simulation design needed, pure reuse of validated
  machinery, ~43s total (8 sims). Computed each pair's INDIVIDUAL
  (uncombined) backprojected contribution at (a) the aggregate's false
  peak location and (b) the true vertex location. Result: **`bottom->right`
  and `right->bottom` together contribute 82% of the false peak's
  energy** (ratios 227x and 504x more energy at the false location than
  at the true vertex — every other pair contributes comparably at both
  locations). This is a concentrated, specific artifact (a sparse-
  multistatic-array "ghost" — a spurious ellipse crossing between two
  probes, a known ambiguity class in sparse MIMO radar/sonar imaging),
  NOT generic orientation-blind clutter (which would show energy spread
  evenly across all 16 pairs) — this definitively supersedes both the
  run -29 colinear-pair hypothesis (already disproven by run -30) and a
  user-proposed alternative hypothesis (missing specular-orientation/
  normal-direction weighting in the backprojection model) as the
  explanation for THIS specific failure, though the orientation gap
  remains a real, separate architectural property of the current method
  worth keeping in mind for future non-convex/complex phantoms.
- **Step 2 — coherence-factor (CF) fix, tested and REJECTED**
  (`src/phase3_multistatic_backprojection_triangle_coherence.py`):
  implemented standard CF weighting (Camacho et al. 2009-style,
  CF(P)=S(P)^2/(N*Q(P)), image=CF*S) as a principled, general
  alternative to a hand-picked exclusion list. Full 8-frame sweep, naive
  vs CF side by side. **Result: CF does not work for this array.** top
  RMSE only marginally improved (2.46mm -> 2.20mm, inconsistent
  per-frame — sometimes much better e.g. 0.14mm, sometimes worse e.g.
  3.10mm) while bottom, left, and right — all previously good or
  reasonable — got MUCH WORSE (bottom 0.14mm->1.84mm; left
  0.95mm->2.06mm; right 0.42mm->1.83mm). **Diagnosed why, not just
  reported as a failure**: CF's underlying assumption (many pairs
  agreeing = real signal, few pairs dominating = suspect) breaks down
  with only 4 probes/16 pairs, because a GENUINE specular reflection off
  a flat facet is *also* seen strongly by only the one or two pairs
  aligned near that facet's normal — the same statistical signature as
  the ghost. CF cannot distinguish a real sparse-pair-dominated echo
  from a ghost sparse-pair-dominated echo in an array this sparse; it
  needs a much denser array (many more probes/pairs) to be the right
  tool, or would need to operate on a much larger set of independently
  corroborating channels than 16.
- **Step 3 — targeted pair exclusion, WORKS**: reran the same script
  with a third variant, dropping only the 2 pairs the diagnostic named
  (`backproject_excluding`, BAD_PAIRS={(bottom,right),(right,bottom)})
  from the naive sum — no statistical rule, just removing the
  specifically-identified culprit. **Result: top RMSE=0.68mm (down from
  naive's 2.46mm) — a real, working fix.** bottom and right are BYTE-
  FOR-BYTE IDENTICAL to the naive result (0.14mm, 0.42mm) — confirms
  those 2 pairs weren't contributing anything useful there, so removing
  them was free. left got moderately worse (0.95mm -> 2.11mm) — an
  honest, acknowledged cost: those 2 pairs evidently carried a small
  amount of genuine signal for the left facet too, and this fix isn't
  free everywhere. Homogeneous-medium control confirmed for all 3
  variants (naive/CF/excl) — no fixed artifact introduced by any of
  them.
- Physical sanity checked? by whom?: Claude — used the diagnostic's own
  measured pair contributions (not assumption) to justify the exclusion
  set before testing it; explicitly checked bottom/right stayed
  IDENTICAL (not just "close") after exclusion, confirming those pairs
  were inert there rather than accidentally cancelling other real
  contributions that happened to net to the same number; reported the
  left-side regression as a real cost rather than omitting it.
- Gate passed? (Y/N): N/A.
- **Visuals produced** (per user request): (1)
  `results/figures/phase3_backprojection_coherence_ghost_comparison.png`
  — 3-panel ED-frame comparison (naive / CF / excl), same true-triangle
  overlay and tracked-vertex marker in each panel, directly showing the
  marker land on the true corner only in the excl panel; (2)
  `results/figures/phase3_multistatic_backprojection_triangle_coherence.png`
  — full 8-frame filmstrip using the excl variant (chosen automatically
  by lowest top-RMSE among the 3 variants, not assumed in advance).
- **Thread closed, per user request.** Summary for the record: the
  beating-triangle generalization test (runs -29/-30/-31) found that
  multistatic backprojection generalizes well to flat facets at any
  incidence angle (edges/oblique facets, sub-1mm with the naive method)
  but has a real, now-understood, now-partially-fixed weakness at sharp
  corners in a SPARSE (4-probe) array: a small number of specific
  cross-pairs can produce sparse-array ghost artifacts that swamp the
  genuinely weak corner echo. The general statistical fix (coherence
  factor) does not work at this array density; a targeted, diagnostic-
  driven fix (excluding the specific identified pair) does, with one
  acknowledged, quantified side effect. This is now a settled, credible
  result to build on.
- Next action: per user, moving to the heart-model phantom next. Carry
  forward: (1) the pair-ablation diagnostic technique
  (`phase3_backprojection_pair_diagnostic.py`) is a fast, cheap,
  reusable tool if new ghost artifacts appear on the more complex heart
  phantom (single-frame capture, pure-numpy per-pair inspection, no new
  simulations); (2) coherence-factor weighting is now known NOT to be a
  good general-purpose fix for this project's sparse probe arrays —
  don't re-attempt it without first increasing probe count/density
  substantially; (3) targeted pair-exclusion has at least one real cost
  (left facet here) — don't assume it's free elsewhere without checking.

### Correction (same session, post-run -31) — the pair-exclusion fix is a SOFT, target-specific patch, not a root-cause fix; run -31's "closed" framing overstated it
- Per user pushback after reviewing the run -31 numbers/figure: 3 of 4
  boundary points (bottom, right, and now top) are well-localized, but
  **left is consistently off in every frame (RMSE 2.11mm, worse than
  before the fix)** — flagged as "just a soft fix" and correctly
  identified as a systematic bias, not per-frame noise.
- **Root cause, user-diagnosed and correct**: `backproject_excluding`'s
  BAD_PAIRS set (`bottom->right`, `right->bottom`) was chosen to fix ONE
  target (the top vertex) and applied GLOBALLY (removed from the sum
  for every candidate point in the domain, not just near the vertex).
  Those same 2 pairs evidently also carried real, useful localization
  information for the oblique LEFT facet (their transmit/receive angles
  happen to be geometrically relevant to that facet's specular
  direction, even though they're a pure ghost for the vertex direction).
  A single global exclusion mask cannot be simultaneously correct for
  every sector of a non-convex/multi-featured boundary, because
  different boundary features (corner vs. flat edge vs. oblique facet)
  are supported by different, non-overlapping, non-obvious subsets of a
  sparse pair set.
- **Deeper structural issue, also user-identified**: `track_four_directions`
  picks ONE peak per fixed cardinal axis (top/bottom/left/right),
  independently, from the full accumulator. This assumes the brightest
  ridge along a given ray IS the corresponding true boundary point — true
  for a circle (every radial direction hits a clean, unambiguous
  boundary), not guaranteed for a triangle (a flat facet is not a point
  target; its brightest ridge can be geometrically displaced from where
  an arbitrary fixed search axis happens to cross it). This concept does
  not generalize to the heart-wall phantom at all, which has no natural
  set of 4 fixed cardinal search rays.
- **Correction to run -31's verdict**: the triangle thread should NOT be
  read as "solved" — it produced a real, working, but NARROW/local patch
  (fixes the vertex, costs the left facet) plus a now-well-understood
  diagnosis of why a single global rule can't work. Not yet done:
  (a) sector- or target-specific pair weighting (different exclusion/
  weighting masks depending on which boundary feature is being
  localized, chosen diagnostically per-sector the same way
  BAD_PAIRS was chosen for the vertex — not a blind per-target tuning
  exercise); (b) replacing independent per-axis peak-picking with
  GLOBAL SHAPE FITTING to the whole accumulator image (e.g. Hough-
  transform line/vertex detection, or RANSAC fitting of the known
  3-edge polygon model to the accumulator's ridge structure) so the
  bad left-facet point becomes a visible outlier to a robust fit
  instead of silently being reported as a final, trusted number.
- Next action (explicitly for a future session, not to be silently
  skipped): before applying multistatic backprojection to the heart-wall
  phantom, replace `track_four_directions`'s fixed-axis peak search with
  a shape-fitting readout (Hough/RANSAC on the accumulator), and treat
  any future pair-exclusion fix as sector-specific (diagnosed per
  target) rather than global, per this correction.

### Run 2026-07-08-32 — Global shape-fitting readout: real, generalizable improvement over BOTH run -29 (naive) and run -31 (pair-exclusion), no manual tuning needed
- Phase: 3 (continuing the same-session correction above). Per user
  request to implement the shape-fitting readout specifically (not the
  sector-specific pair-weighting alternative, since only the shape-fit
  approach generalizes to the heart-wall phantom, which has no fixed
  cardinal axes). `src/phase3_backprojection_shape_fit_triangle.py`.
- Design: since the triangle's shape family is known exactly
  (equilateral, fixed center/orientation, one free parameter R), built a
  generalized-Hough-style GLOBAL template match rather than a fully
  generic image-segmentation+RANSAC pipeline: (1) `ray_triangle_distance
  (theta, R)` — analytic ray/polygon-edge intersection, generalizing the
  4 hand-derived axis formulas (top=R, bottom=R/2, left=right=R/sqrt(3))
  to ANY angle; verified by assertion against those exact formulas
  before trusting it. (2) `fit_triangle_radius`: sweep R over a fine
  grid (25-75 cells, step 0.25), and for each candidate, sample the
  (plain, NAIVE, NOT pair-excluded) accumulator via bilinear
  interpolation (`RegularGridInterpolator`) along ALL 72 angles (every
  5 degrees) that candidate R's boundary would produce, summing the
  energy. Pick the R maximizing total boundary energy. This integrates
  evidence from the WHOLE shape at once, so a single bad/ghost-corrupted
  sector (the diagnosed bottom->right/right->bottom pairs) can't
  dominate the way it dominated one local axis-peak search — same
  4-probe/16-pair geometry and naive backprojection as run -29, NO
  manual pair exclusion needed at all.
- Result: **a real, balanced, generalizable improvement over both prior
  approaches — the first result this thread produced with no
  catastrophic outlier on any side.**
  | side | naive (run -29) | excl-2-pairs (run -31) | global shape fit |
  |---|---|---|---|
  | top (vertex) | 2.46mm | 0.68mm | 0.81mm |
  | bottom (edge) | 0.14mm | 0.14mm | 0.41mm |
  | left (facet) | 0.95mm | 2.11mm (regressed) | **0.47mm** |
  | right (facet) | 0.42mm | 0.42mm | 0.47mm |

  Worst-case error across all 4 sides: naive=2.46mm, excl=2.11mm,
  **global fit=0.85mm** — roughly a 2.5-3x reduction in worst-case error
  relative to EITHER prior method, achieved automatically (no
  diagnostic-driven hand-tuning of which pairs to exclude where).
  Homogeneous-medium control: fitted R=74.8 cells with a flat, low
  score curve (visually confirmed in the saved score-vs-R figure) — no
  false-positive fixed artifact surviving the fit.
- **Honest residual, flagged not buried**: the fitted R is consistently
  SMALLER than true R by ~7-8.5 cells (~0.75-0.85mm) in EVERY frame — a
  small, uniform, systematic bias (not random noise), most likely a
  diagnosable/fixable effect analogous in class to the envelope
  group-delay correction found in run -28 (interacting differently once
  many angles are integrated together rather than one axis), but NOT
  yet root-caused. Much smaller and more uniform than either prior
  method's worst-case error, but should not be silently treated as
  "just noise" without further investigation.
- Physical sanity checked? by whom?: Claude — verified the general
  `ray_triangle_distance` formula against all 4 hand-derived closed-form
  axis distances via assertion BEFORE using it in the fit (established
  practice: verify before trust); ran the homogeneous-medium control and
  visually checked the score curve is flat/inconclusive there, not
  spuriously peaked; explicitly reported the systematic undersizing bias
  rather than only reporting the (favorable) RMSE numbers.
- Gate passed? (Y/N): N/A.
- **Answering the user's original correction directly**: yes — global
  shape fitting is both a real, working, more BALANCED fix than the
  local pair-exclusion patch (no single side pays a large regression
  cost, unlike run -31's left-facet hit) AND — critically — the right
  foundational approach to carry to the heart-wall phantom, since it
  never depended on the triangle's specific cardinal-axis geometry
  except via the one-parameter template family; a curved/unknown
  boundary would need a more generic curve-fit (e.g. RANSAC or active-
  contour) in the same spirit, not 4 fixed rays.
- Visuals produced: (1)
  `results/figures/phase3_backprojection_shape_fit_score_curve.png` —
  score(R) vs R for the ED frame, showing a clean single peak near the
  (slightly biased) fitted R, not multiple competing local maxima; (2)
  `results/figures/phase3_backprojection_shape_fit_triangle.png` — full
  8-frame filmstrip with both the true triangle (cyan) and the globally-
  fitted triangle (green) overlaid on the plain naive accumulator.
- Next action: (1) investigate the ~8-cell systematic undersizing bias
  before trusting absolute R values from this method (relative/shape
  tracking already looks solid); (2) when moving to the heart-wall
  phantom, generalize `fit_triangle_radius`'s PRINCIPLE (sweep a
  parametric family, integrate evidence over the whole predicted
  boundary, maximize total agreement) rather than its literal
  implementation (triangle-specific one-parameter family) — the heart
  wall will need a more flexible boundary-shape parametrization (e.g. a
  radial Fourier series or spline around the known approximate LV
  center, or RANSAC/active-contour fitting) since it isn't a fixed
  polygon.

### Run 2026-07-08-33 — Diagnosed the undershoot mechanism (per-angle bias diagnostic); proposed TGC fix TESTED AND REJECTED — bias is timing, not amplitude
- Phase: 3 (continuing run -32). Per user request ("can you diagnose
  the constant margin?" then "log and fix, and rerun and visualize").
  `src/phase3_shape_fit_bias_diagnostic.py` (new diagnostic) and
  `src/phase3_backprojection_shape_fit_triangle.py` (extended with a
  proposed fix).
- **Diagnostic step 1**: for the ED frame (R=60, same case examined
  throughout this thread), computed each of the 72 angles' OWN
  independent best-fit distance (free 1D search along that ray alone,
  ignoring the shared-R constraint) and compared to what the triangle
  formula predicts at the true R. Cardinal angles' mean bias was near
  zero (-0.76 cells) but individual cardinal angles varied wildly (top
  -26.5 cells, left +24.6 cells) — traced this to run -29's
  `track_four_directions` (masked native-grid search, ~1.8-cell
  spacing) landing on a DIFFERENT peak than a finer (0.25-cell,
  interpolated) free search for the same ray. Printed both candidates'
  actual heights directly: **peak near the true facet (d=33.6) =
  0.000503, peak near the "ghost" location (d=59.25) = 0.000507 — within
  1% of each other.** This means run -29's good "left" result (0.10mm
  error) was partly LUCKY (a near-tied contest that the coarse grid
  happened to resolve correctly), not a robust win — consistent with
  why runs -30/-31 both independently made `left` worse (small changes
  are enough to flip a near-tie).
- **Diagnostic step 2**: compared the global fit using ONLY the 4
  cardinal angles (validated-accurate in run -29) vs all 72. Cardinal-
  only: fitted R=55.0 (0.50mm undershoot). All 72: fitted R=52.0 (0.80mm
  undershoot). Adding the 68 non-cardinal, mostly-unaligned angles makes
  the undershoot WORSE, not just noisier — this pointed to a systematic,
  direction-independent bias toward smaller R, hypothesized as an
  uncorrected range-dependent amplitude/gain effect (2D geometric
  spreading makes closer candidate points generically read "louder"
  regardless of true reflectivity, analogous to needing TGC in real
  ultrasound systems).
- **Fix implemented and tested**: `backproject_tgc` — multiplies each
  pair's contribution by sqrt(dist_tx*dist_rx) (the inverse of 2D
  round-trip cylindrical-spreading falloff) before summing. Ran the full
  8-frame sweep, no-TGC vs TGC side by side, same captured pairs (no new
  simulations).
- **Result: THE FIX DID NOT WORK.** no-TGC R RMSE=0.8136mm, TGC R
  RMSE=0.8202mm — statistically unchanged (marginally worse, within
  noise). Every single frame's fitted R moved by at most 0.3 cells.
  **This falsifies the range-dependent-amplitude-gain hypothesis** as
  the (or at least the dominant) mechanism. Reported honestly as a
  negative result, not silently dropped.
- **Revised hypothesis** (not yet tested): since TGC only rescales
  amplitude and does not change WHERE in time each pair's envelope peak
  falls, a null TGC result points toward a TIMING bias instead of an
  amplitude bias — most plausibly, the `_ENVELOPE_GROUP_DELAY_S =
  duration/2` correction (calibrated in run -28 for a single, near-
  normal-incidence reflection on the circle phantom) may not be
  uniformly correct across the more oblique, off-axis bistatic pair
  geometries that dominate most of the 72 integrated angles (only 4 of
  72 correspond to any actual probe axis). A systematic per-pair timing
  mismatch would bias predicted matches toward smaller R regardless of
  amplitude weighting, consistent with the observed null TGC result.
- Physical sanity checked? by whom?: Claude — verified the "near-tied
  competing peaks" finding by printing BOTH candidates' actual
  interpolated heights (not inferring from position alone); confirmed
  the TGC fix's near-zero effect by comparing per-frame numbers
  directly, not just the aggregate RMSE, before concluding it failed.
- Gate passed? (Y/N): N/A.
- Visuals produced: (1)
  `results/figures/phase3_shape_fit_bias_diagnostic.png` — per-angle
  bias vs angle (showing cardinal angles highlighted) and the cardinal-
  only vs full-72 score curve comparison; (2)
  `results/figures/phase3_backprojection_shape_fit_score_curve.png` —
  updated to show no-TGC vs TGC score curves overlaid (visually near-
  identical, confirming the null result); (3)
  `results/figures/phase3_backprojection_shape_fit_triangle.png` —
  updated 8-frame filmstrip using the TGC variant (numerically almost
  identical to run -32's, as expected given the null result).
- Next action: (1) do NOT re-attempt amplitude-based (TGC-style) fixes
  for this specific bias — tested and rejected; (2) if resumed, test the
  timing-bias hypothesis directly: compute the OPTIMAL group-delay
  correction empirically per pair (or per tx/rx geometry class) against
  known ground truth, rather than assuming one constant duration/2 value
  works uniformly for every bistatic angle; (3) the ~0.8mm residual is
  still far smaller than every prior method's worst-case error (2.1-
  2.5mm) — usable as-is if the session needs to move on, with this
  known, small, honestly-flagged systematic offset noted.

### Run 2026-07-08-34 — Multi-ghost hypothesis CONFIRMED (LEFT has its own ghost pairs, same mechanism as TOP); principled fix WORKS at large R, only PARTIALLY at small R
- Phase: 3 (continuing run -33). Per user pushback on the timing-bias
  explanation: "i dont think its time bias. because the predicted
  triangle is always smaller than the real one regardless of systole or
  diastole." Re-examined run -33's own diagnostic data: `top`'s and
  `bottom`'s independent free-search peaks landed at an IDENTICAL
  distance (33.50 cells), and `left`'s independent peak (59.25) landed
  almost exactly at `top`'s TRUE distance (60.0) — directions finding
  EACH OTHER's reflections, not a uniform timing offset.
- **New diagnostic**: `src/phase3_left_ghost_diagnostic.py` (adapted
  from the original pair-ablation tool), targeting LEFT's false peak
  (d~59.25) vs LEFT's true facet (d=34.6) for the ED frame. Found: top-3
  pairs (`left->top`, `bottom->left`, `left->bottom`) = 65.9% of the
  false-peak energy (vs 18.75% uniform) and contribute ~nothing at
  LEFT's true location — while the pairs that DO carry LEFT's true
  signal (`bottom->right`, `right->top`, `top->right`) contribute
  ~nothing at the false location. Two fully disjoint pair sets, same
  signature as TOP's confirmed ghost (`bottom->right`/`right->bottom`).
  **This generalizes the pattern**: TOP's ghost came from adjacent
  (90-degree-separated) probe pairs; LEFT's ghost also comes from
  adjacent pairs (`left<->top`, `left<->bottom`). Confirms the user's
  skepticism — this is a structural, GEOMETRIC property of the 4-probe
  layout (every adjacent-probe cross pair creates a ghost pointing at a
  neighboring direction), not a timing/calibration artifact (which
  would apply uniformly, not direction-specifically).
- **Principled fix tested**: `backproject_no_adjacent` in
  `phase3_backprojection_shape_fit_triangle.py` — exclude ALL 8
  adjacent cross pairs (not just the 2 hand-picked for TOP in run -31),
  keep only the 4 monostatic + 4 antipodal pairs (8 of 16). Ran the full
  8-frame sweep, naive vs no-adjacent side by side.
- **Result: a real, substantial, but INCOMPLETE fix — frame-dependent,
  not uniform.**
  | R (cells) | naive R err | no-adjacent R err |
  |---|---|---|
  | 60.0 (ED) | 0.80mm | **0.00mm** |
  | 56.2 | 0.75mm | **0.03mm** |
  | 47.8 | 0.85mm | 0.85mm (unchanged) |
  | 41.0 | 0.85mm | 0.28mm (improved, not fixed) |

  Overall R RMSE: naive=0.8136mm -> no-adjacent=0.4482mm (~45%
  reduction). At the two largest (ED-adjacent) frames, ALL FOUR sides
  recover to <=0.03mm — essentially exact, strongly confirming the
  adjacent-pair-ghost mechanism is the DOMINANT cause of the undershoot
  when the true signal is strong. But at the two smaller (more
  contracted) frames, a real residual persists (unchanged at R=47.8,
  only partially improved at R=41.0) — consistent with this session's
  recurring pattern (weaker true reflectivity at smaller radii lets
  whatever OTHER, still-unidentified artifact compete more effectively;
  the adjacent-pair ghosts were most, not all, of the story).
- **Caution flagged, not glossed over**: the homogeneous-medium control
  now fits to R=25.2 cells — sitting right at the edge of the R_GRID
  sweep range (25.0-75.0) — meaning the control's score curve got
  flatter/more ambiguous with only 8 of 16 pairs remaining, not a clean
  "low, meaningless" confirmation like before. Should widen the R_GRID
  range and re-check this control specifically before trusting the
  no-adjacent method's absolute R values near the search boundary.
- Physical sanity checked? by whom?: Claude — verified the ghost
  pattern generalizes by finding a SECOND, independent instance (LEFT)
  with the same disjoint-pair-set signature as the first (TOP), rather
  than assuming one confirmed case implies the general rule; reported
  the per-frame breakdown (not just aggregate RMSE) to catch that the
  fix is R-dependent, not uniform; flagged the homogeneous-control
  boundary-hugging result as a caution rather than ignoring it.
- Gate passed? (Y/N): N/A.
- **Answering the user's correction directly**: confirmed correct — this
  is NOT a timing bias. It's a geometric, adjacent-probe ghost-pair
  effect, now demonstrated at two independent sectors (top, left) with
  the same disjoint-contributing-pairs signature, and excluding all such
  pairs fixes the undershoot completely at strong-signal (large R)
  frames. The remaining small-R residual is a distinct, still-open
  question — likely the same class of "weak true signal loses to
  whatever secondary structure remains" issue that has appeared
  throughout this session (DAS's ES gap, Doppler's zero-velocity gap,
  etc.), not evidence against the ghost-pair diagnosis.
- Visuals produced: (1)
  `results/figures/phase3_backprojection_shape_fit_score_curve.png` —
  updated to show naive vs no-adjacent-pairs score curves for the ED
  frame (visually confirms no-adjacent's peak lands exactly on true R);
  (2) `results/figures/phase3_backprojection_shape_fit_triangle.png` —
  updated 8-frame filmstrip using the no-adjacent variant (visually
  near-perfect overlay for large-R frames, visible gap for small-R
  frames).
- Next action: (1) investigate why the small-R residual persists after
  removing all adjacent-pair ghosts — check whether a DIFFERENT pair
  combination (among the remaining 8 monostatic/antipodal pairs)
  develops its own ghost specifically when the true signal weakens; (2)
  widen R_GRID and rerun the homogeneous control specifically to confirm
  it isn't just hitting a search-range boundary artifact; (3) this
  result (large-R near-perfect, small-R partial) is good enough to carry
  into the heart-wall phantom as a documented, partially-understood
  limitation — do not present the no-adjacent fix as a complete
  solution.

### Run 2026-07-08-35 — Off-center triangle test: same ghost pattern persists at a shifted location, confirming it's an array-structural property, not a symmetric-case coincidence
- Phase: 3 (continuing run -34). Per user request ("try off-center
  triangle to see if that reveals anything"). `src/phase3_offcenter_triangle_test.py`.
- Design: same equilateral triangle (R=60, ED), same 4-probe geometry
  and full multistatic machinery (capture, envelope detection, direct-
  arrival exclusion, group-delay correction, `backproject` and the
  validated `backproject_no_adjacent` fix from run -34) — ONLY the
  phantom's position changed, shifted by (15, 10) cells from domain
  center (to (165, 160)), still comfortably inside the accumulator
  search grid. Ground-truth per-axis distances computed via the same
  analytic ray-segment-intersection method, but with rays originating
  from the TRUE SHIFTED center (not the old domain center) — a properly
  adjusted comparison, not reusing stale centered-case formulas.
- Result: **the same qualitative failure pattern reappears at the new
  location** — `top` and `left` are still wrong (errors 2.58mm, 2.16mm,
  same order of magnitude as the centered case's ghost-driven errors),
  while `bottom` and `right` remain accurate (0.40mm, 0.16mm), matching
  the centered case's reliable directions almost exactly. This was NOT
  assumed in advance — the true per-axis distances at this off-center
  position are all different numbers than the centered case (34.6/30.0
  vs the centered case's own true values), so the fact that the SAME
  two directions fail and the SAME two succeed is a real, non-trivial
  finding, not a restatement of the same numbers.
- **Interpretation**: this shows the adjacent-pair ghost mechanism
  (runs -29, -33/-34's confirmed `bottom<->right` and
  `left<->top`/`left<->bottom` cross-pair ghosts) is NOT a fluke tied to
  the phantom sitting in the one special, maximally-symmetric position
  (exactly equidistant from all 4 probes). It persists in a modestly
  off-center configuration too — a genuine structural property of which
  probes are ADJACENT (90-degree-separated) vs ANTIPODAL
  (180-degree-separated) relative to each cardinal direction, largely
  independent of exactly where within the array's field of view the
  target sits. **This is a stronger, more useful finding for the
  heart-wall phantom than if the ghost had only appeared in the
  contrived centered case** — it means the `backproject_no_adjacent`
  mitigation (validated in run -34) should be expected to generalize to
  an off-center heart, not just this specific symmetric toy setup.
- Physical sanity checked? by whom?: Claude — recomputed true per-axis
  ground-truth distances properly relative to the NEW shifted center
  (not reusing the old centered-case formulas, which would have been
  wrong/meaningless for a moved target) before comparing; confirmed the
  matching failure/success pattern is informative precisely because the
  underlying true numbers differ from the centered case, ruling out a
  trivial "same numbers, same result" explanation.
- Gate passed? (Y/N): N/A.
- Scope/limitation of this run: single frame (ED, R=60) and a single
  modest offset, not a full parameter sweep — a qualitative/exploratory
  probe (per the user's framing: "see if that reveals anything"), not a
  new quantitative benchmark. The no-adjacent-pairs variant's full
  per-axis breakdown at the shifted position was not computed (only its
  global peak location, which moved closer to the true center's row but
  not fully to the true position) — a natural next step if this is
  revisited.
- Visuals produced: `results/figures/phase3_offcenter_triangle_test.png`
  — naive vs no-adjacent-pairs images side by side, both with the true
  shifted triangle boundary, the old domain center, and the true shifted
  center all marked, for direct visual comparison.
- Next action: treat the adjacent-vs-antipodal ghost pattern as a
  general property of this 4-probe layout to carry into the heart-wall
  phantom (not centered-case-specific); if resumed, run the full
  no-adjacent-pairs per-axis breakdown at this offset (not just the
  global peak) and test a larger offset to find where (if anywhere) the
  pattern breaks down as the target approaches the edge of the array's
  useful field of view.

### Note for the finetuning phase — clinical significance of the ~0.8mm shape-fit bias, and why a constant-offset calibration will NOT be sufficient
- Per user question: "is 0.8mm consequential clinically? I know a
  systematic bias means something will be corrected later (note it down
  for the finetuning phase)." Not a new run — a documentation note
  flagged for the future finetuning/calibration phase of this project,
  since it directly affects how that phase should be scoped.
- **Scale caveat (the raw number is not directly clinically
  interpretable)**: this toy phantom's LV radius (4-6mm, ED/ES) is
  self-chosen and explicitly NOT at clinical anatomical scale
  (`phase3_config.py` docstring) — roughly 1/4-1/5 of a real LV cavity
  radius (~20-30mm; ASE normal LVIDd ~42-59mm diameter). The 0.8mm bias
  cannot be compared to real clinical measurement precision without
  knowing how it rescales, which has NOT been tested.
  - If the bias is a FIXED absolute artifact size (plausible if tied to
    a fixed simulation/array-geometry effect) -> ~0.8mm at real
    anatomical scale would sit BELOW typical inter-observer echo
    measurement variability (commonly ~2-5mm for linear LV dimensions)
    -> likely clinically negligible.
  - If the bias scales PROPORTIONALLY with target size (also plausible,
    since it's driven by wave-path-length geometry, which does scale
    with absolute distance) -> the same ~13% relative error (0.8mm/6mm
    in the toy) at real LV radius (~25mm) would be ~3mm -> NOT
    negligible: comparable to or larger than typical measurement
    reproducibility, and potentially relevant for borderline calls
    (e.g. mild-hypertrophy wall-thickness cutoffs are often only 1-2mm
    apart) and especially for THIS project's actual target application
    (detecting SUBTLE regional wall-motion abnormalities, where a
    multi-mm systematic error could meaningfully bias results).
  - Which regime applies is an open, untested question — flagged for
    whoever scopes the finetuning phase, not answered here.
- **Why a simple constant-offset calibration will NOT fully fix this**:
  the bias is demonstrably NOT a single constant. Per runs -34/-35: it
  is R-dependent (0.00mm at R=60 after the adjacent-pair-ghost fix,
  still 0.85mm unchanged at R=47.8) and shows the same qualitative
  pattern but different per-axis magnitudes when the target is shifted
  off-center (run -35). A calibration step that subtracts one fixed
  number (e.g. "0.8mm") would UNDER-correct at some sizes/positions and
  OVER-correct at others. **Any real calibration in the finetuning phase
  needs to be a function of apparent target size/direction/position
  (e.g. a lookup table or fitted correction curve built from phantom
  sweeps), not a single scalar offset.**
- Next action: when the finetuning phase is scoped, (1) test whether
  this bias scales proportionally or stays fixed in absolute terms by
  rerunning the R-sweep at a genuinely different absolute scale (e.g. a
  phantom domain scaled to real LV dimensions, not just the toy's 4-6mm
  radius), before assuming either rescaling regime; (2) build any
  calibration correction as a function of the relevant variables (size,
  direction, position), not a constant; (3) this note should be
  reconciled with the still-open small-R residual (run -34) and the
  off-center generalization question (run -35) before finalizing a
  calibration approach.

### Run 2026-07-08-36 — Ghost mechanism CONFIRMED: the vertex's real corner-diffracted energy, not a capture/simulation bug — the artifact is entirely a reconstruction-model gap
- Phase: 3 (continuing run -35). Per user's inference ("adjacent-pair
  ghosts persist off-center, so the artifact must lie somewhere within
  the reconstruction, not the capture or measuring... a fixed-size ghost
  regardless of motion, orientation, location") and agreement to test
  it directly. `src/phase3_ghost_mechanism_diagnostic.py`.
- Design: rather than accept the inference at face value, tested it
  against the RAW captured trace directly (bypassing backprojection
  entirely) for the confirmed `left->top` ghost pair (run -34, ED frame,
  R=60, centered case). Two candidate physical mechanisms, both of which
  would mean the CAPTURE is correct and only the RECONSTRUCTION's
  interpretation is flawed: (1) genuine bistatic specular reflection
  from an unexpected point on the true boundary (found via sweeping 400
  samples per edge, computing the mirror-reflection "defect" between the
  incident ray and actual outgoing direction to the receiver, keeping
  the point with the smallest defect); (2) genuine corner diffraction
  from the KNOWN vertex position (exact, since this is a toy with known
  ground truth) — a real, standard physical phenomenon (Keller's GTD:
  sharp corners scatter omnidirectionally, not just specularly). Found
  the actual peak(s) in the pair's cleaned raw envelope trace via
  `scipy.signal.find_peaks`, then compared both mechanisms' predicted
  arrival times directly against the REAL recorded peak.
- Result: **both mechanisms predict essentially the SAME time (12.765us
  vs 12.778us, 0.013us apart), and BOTH match the actual recorded peak
  (13.049us) to within <0.5mm path-equivalent** (0.45mm, 0.43mm).
  Critically, mechanism 1's independent boundary sweep (no knowledge of
  the vertex given to it directly, just searching for the best law-of-
  reflection match anywhere on the 3 edges) CONVERGED to (row=91.8,
  col=149.0) — essentially ON TOP OF the actual vertex at (90, 150).
  **The two mechanisms aren't actually distinguishable here because
  they're the same physical location**: at a sharp corner, ordinary
  single-facet specular reflection and diffraction become the same
  limiting case (a corner sits exactly where two facets' individual
  specular domains break down and radiate to bistatic pairs neither flat
  facet alone could satisfy).
- **Verdict: the captured data is physically correct.** The `left->top`
  pair genuinely records real scattered energy from the vertex — real
  wave physics, not a numerical bug, nothing wrong with the simulation
  or capture step. The "ghost" exists ONLY because the reconstruction's
  naive travel-time-matching model has no way to distinguish "real
  energy from the true vertex" from "real energy from a wrong candidate
  point that happens to produce the identical round-trip travel time for
  this specific bistatic pair" — both look identical to a model that
  only checks total path length, never checks whether a specular
  reflection is geometrically valid there, and has no explicit
  diffraction term at all.
- Physical sanity checked? by whom?: Claude — tested the hypothesis
  against RAW captured data directly, bypassing the backprojection
  reconstruction entirely, specifically to distinguish "capture problem"
  from "reconstruction problem" rather than reasoning about it only
  abstractly; verified mechanism 1's boundary search converged to the
  vertex independently (not by assuming it would), which is what made
  the two mechanisms' near-identical predictions interpretable rather
  than a coincidence.
- Gate passed? (Y/N): N/A.
- **Answering the user's inference directly**: confirmed, and now
  precise. The artifact is 100% in the reconstruction's forward model
  (naive travel-time-only matching — no specular-consistency check, no
  diffraction term), not in capture or measurement, which is recording
  correct physics. This also reframes the "ghost" constructively: since
  it's real, physically meaningful energy (genuinely marking where a
  corner/singularity is), a smarter reconstruction could eventually USE
  it (e.g. corner/feature detection from where multiple pairs'
  diffraction-consistent predictions converge) rather than only
  suppressing it via pair exclusion (runs -31/-34's approach) — a
  forward-looking note for when real, non-convex anatomical features
  (not just idealized smooth boundaries) are modeled in the heart-wall
  phantom.
- Next action: this closes the "what mechanism causes the ghost"
  question with a confirmed, physically-grounded answer. Two natural
  follow-ons, not yet done: (1) verify the same convergence-to-vertex
  result holds for the OTHER confirmed ghost pairs (`bottom->right`,
  `right->bottom`, `bottom->left`, `left->bottom`) to confirm this isn't
  specific to `left->top`; (2) consider whether an explicit diffraction-
  aware reconstruction term (rather than pair exclusion) is worth
  prototyping before the heart-wall phantom, since real anatomical
  features (papillary muscles, valve annulus, wall-thickness
  transitions) may present similar corner-like scattering that current
  pair-exclusion-based fixes would simply discard as noise.

### Run 2026-07-08-37 — Off-center CONCAVE heart-shape test: neither naive nor no-adjacent-pairs recovers the boundary — a real regression from the triangle, consistent with more corners = more ghosts
- Phase: 3 (continuing run -36). Per user request: "try an 8 frame
  offcenter (your choice of coordinates) heart shape. this one have a
  concave region and i want to know how it behaves."
  `src/phase3_heart_shape_offcenter_test.py`.
- Design: same validated 4-probe multistatic pipeline (capture,
  envelope detection, direct-arrival exclusion, group-delay correction,
  `backproject` and run -34's `backproject_no_adjacent` fix) — only the
  phantom changed to a simplified 10-vertex heart polygon (bottom tip,
  two lobes, and a concave NOTCH between the lobes — the first non-
  convex shape tried in this thread), point-in-polygon tested via
  `matplotlib.path.Path.contains_points` (a general algorithm, since the
  triangle's convex-only same-side-of-every-edge trick is invalid for a
  concave polygon), shifted off-center by (10, -15) cells (arbitrary,
  chosen to keep the shape safely within the accumulator search grid
  and simulation domain), same 8-frame ED->ES->ED radius schedule.
- Result (assessed by actually viewing the saved figures, not just
  confirming they saved): **neither variant recovers the heart boundary
  cleanly — a real, honest regression from the triangle case.** Both
  naive and no-adjacent-pairs images are dominated by bright, diagonal
  crossing ridges that sit in almost the SAME positions across all 8
  frames regardless of true size/phase — the signature of a fixed
  background/artifact pattern, not a signal tracking the actual
  boundary. The true heart outline does not align with any clean
  boundary-following bright ring in either image, unlike the triangle's
  no-adjacent-pairs result (near-perfect at large R, run -34).
- **Interpretation, consistent with run -36's confirmed mechanism**:
  this heart shape has FOUR sharp curvature features (bottom tip, two
  lobe tips, and the concave notch) vs. the triangle's THREE vertices.
  Since run -36 proved sharp corners genuinely diffract and get
  misattributed by the naive travel-time model, more corners plausibly
  means more ghost-pair combinations than the simple "adjacent vs
  antipodal" categorization (tuned/validated only on the triangle's
  3-corner case) can clean up. **The no-adjacent-pairs fix is a
  triangle-shaped patch, not a general solution** — this result
  confirms that directly rather than just flagging it as a theoretical
  concern.
- **Cannot confirm or deny** whether the concave notch specifically
  diffracts (the hypothesis this test was partly designed to probe) —
  the general clutter in both images is strong enough that no distinct
  notch-specific signature is visually separable. Would need the same
  targeted pair-ablation diagnostic used for the triangle's corners
  (`phase3_backprojection_pair_diagnostic.py`/`phase3_left_ghost_diagnostic.py`/
  `phase3_ghost_mechanism_diagnostic.py` methodology), aimed specifically
  at the notch, to test this properly — not done in this run.
- Physical sanity checked? by whom?: Claude — actually opened and
  visually inspected all 3 saved figures (ED comparison, no-adjacent
  filmstrip, naive filmstrip) before reporting, rather than assuming
  success from the script completing without error; explicitly noted
  what could NOT be determined (notch-specific diffraction) rather than
  overclaiming from an inconclusive image.
- Gate passed? (Y/N): N/A.
- Visuals produced: (1)
  `results/figures/phase3_heart_shape_offcenter_ED_comparison.png` — ED
  frame, naive vs no-adjacent-pairs side by side; (2)
  `results/figures/phase3_heart_shape_offcenter_filmstrip.png` — 8-frame
  filmstrip, no-adjacent-pairs variant; (3)
  `results/figures/phase3_heart_shape_offcenter_filmstrip_naive.png` —
  8-frame filmstrip, naive variant. All three show the same fixed-ridge-
  pattern problem described above.
- Next action: (1) before the real heart-wall (myocardial ring)
  phantom, the reconstruction likely needs the more fundamental
  upgrade already flagged in the triangle thread — generalizing run
  -32's GLOBAL shape-fitting/boundary-integration principle to
  non-convex, multi-corner boundaries — rather than continuing to patch
  via pair-exclusion heuristics tuned to one specific shape; (2) run the
  targeted pair-ablation diagnostic at the notch specifically to test
  whether it diffracts like a convex corner does; (3) also worth
  checking whether the homogeneous-medium reference subtraction is
  still fully effective for this off-center, multi-corner, no-adjacent-
  pairs combination (run -34 already flagged the no-adjacent homogeneous
  control sitting at a search-grid boundary as a caution — this result
  may be a visible manifestation of that same, not-yet-resolved issue).

### Run 2026-07-08-38 — Global shape-fit generalized to the concave heart shape: naive accumulator + global fit WORKS (0.23mm RMSE), no-adjacent-pairs fails — confirms pair-exclusion doesn't generalize, global fit does
- Phase: 3 (continuing run -37). Per "proceed" (generalizing run -32's
  global shape-fit principle to the non-convex heart shape, per run
  -37's finding that pair-exclusion patches don't generalize).
  `src/phase3_heart_shape_shapefit.py`.
- Design: same 1-parameter (R) global template-match principle as run
  -32 (sweep candidate R, integrate accumulator energy along the ENTIRE
  predicted boundary across many angles), generalized from the
  triangle's 3-edge analytic ray intersection to the heart's 10-vertex
  (possibly concave) polygon via a proper multi-edge ray intersection
  that collects ALL crossings and keeps the nearest (correct for both
  convex and concave boundaries, unlike assuming exactly one crossing).
  Verified via assertion against the known notch/bottom-tip distances
  before trusting it. Ran on BOTH the naive (all 16 pairs) and run -34's
  no-adjacent-pairs accumulators, same 8-frame off-center heart from run
  -37, using the known true center (fitting R only, not position).
- Result: **naive + global shape-fit works well; no-adjacent-pairs
  fails.** Naive: R RMSE=0.2271mm, consistent across all 8 frames
  (0.20-0.28mm, no outliers). No-adjacent: R RMSE=2.8544mm, erratic
  (undershoots at large R down to 25.0 cells — nearly half the true
  size — overshoots to 74.8 at small R, hitting the R_GRID boundary).
  **Confirms run -37's hypothesis directly**: the no-adjacent-pairs
  heuristic (tuned only on the triangle's 3-corner case) actively HURTS
  performance on the heart's 4-corner concave shape — it was excluding
  pairs that carried real, necessary signal for this more complex
  boundary. The global shape-fit PRINCIPLE, in contrast, transfers
  cleanly from circle (run -28) to triangle (run -32) to this concave,
  off-center, 4-corner heart shape, using the plain naive accumulator
  and no hand-tuned pair exclusion at all.
- Visually confirmed (both filmstrips actually viewed, not just
  numbers trusted): the naive filmstrip shows the green (fitted) and
  cyan (true) heart outlines nearly perfectly overlapping in all 8
  frames; the no-adjacent filmstrip shows a consistently and visibly
  undersized fitted heart, matching the poor RMSE.
- **Follow-up per user's own re-examination of the naive result**: the
  naive fit shows a small, systematic OVERSHOOT (predicted always
  larger than true, by a fairly constant +2.0 to +2.8 cells / 0.20-
  0.28mm, across all 8 frames including both ED and ES extremes) — the
  OPPOSITE SIGN from run -32's triangle result (which showed a
  systematic UNDERSHOOT, ~0.8mm). Since both scripts use the identical
  `_ENVELOPE_GROUP_DELAY_S` timing correction, a universal/constant
  timing-calibration bug would bias both shapes the SAME direction —
  getting opposite signs on two different shapes is evidence AGAINST a
  universal timing bug and FOR a shape-dependent geometric effect
  (consistent with run -36's confirmed corner-diffraction mechanism:
  different shapes have different numbers/arrangements of corners and
  ghost-pair combinations, so no reason to expect the same sign or
  magnitude of residual bias).
- Physical sanity checked? by whom?: Claude — verified the general
  multi-edge ray-polygon intersection against known closed-form
  distances (notch, bottom tip) via assertion before use; actually
  viewed both filmstrip images (not just the printed RMSE) before
  reporting success, catching that the no-adjacent result is genuinely
  erratic (hits the R_GRID boundary at one frame), not just "somewhat
  worse"; correctly identified that the opposite-sign bias between
  triangle and heart results argues against, not for, a universal
  timing-bug explanation.
- Gate passed? (Y/N): N/A.
- Visuals produced: (1)
  `results/figures/phase3_heart_shape_shapefit_score_curve.png` — ED
  frame score-vs-R, naive vs no-adjacent; (2)
  `results/figures/phase3_heart_shape_shapefit_filmstrip.png` —
  8-frame filmstrip, no-adjacent-pairs (poor result, visibly undersized
  fitted heart every frame); (3)
  `results/figures/phase3_heart_shape_shapefit_filmstrip_naive.png` —
  8-frame filmstrip, naive (excellent result, fitted/true nearly
  overlapping every frame).
- **This closes the "does the shape-fit principle generalize past
  convex shapes" question with a clean, confirmed yes** — provided the
  reconstruction uses the FULL naive accumulator (all pairs) and global
  boundary integration, not a hand-tuned pair-exclusion heuristic.
- Next action: per user, proceed to the actual "heart phantom" this
  project's Phase 3 has used throughout (the two-boundary myocardial
  ring — inner LV cavity / outer epicardial boundary, constant wall
  thickness, per `phase3_config.py`'s established
  LV_RADIUS_ED/ES_CELLS + WALL_THICKNESS_CELLS model) — a genuine
  escalation from single-boundary shapes (circle, triangle, heart-
  cartoon) to a two-boundary phantom, applying the same validated naive
  + global shape-fit pipeline to both boundaries.

### Run 2026-07-08-39 — Myocardial ring phantom: inner boundary excellent (0.27mm), OUTER boundary NOT recoverable — reconnects to Thread 1's original weak-interface problem, not a new shape-fit bug
- Phase: 3 (continuing run -38). Per user: "log and proceed to heart
  phantom I guess?" — proceeding to this project's actual established
  two-boundary myocardial ring phantom (`phase3_config.py`:
  LV_RADIUS_ED/ES_CELLS, WALL_THICKNESS_CELLS=30 held constant, cited
  tissue properties from `phase2_config.py`), rather than the single-
  boundary toy shapes (circle/triangle/heart-cartoon) used so far.
  `src/phase3_ring_phantom_shapefit.py`.
- Design: same validated 4-probe naive multistatic capture/backproject,
  same global template-match shape-fit principle (runs -28/-32/-38),
  now fitting the inner (LV cavity) and outer (epicardial) boundaries
  INDEPENDENTLY as two separate 1-parameter circle fits (trivial ray-
  distance = R at every angle for a plain concentric circle, no polygon
  intersection needed). Centered at domain center (matches this
  project's established ring-phantom convention elsewhere). 8-frame
  ED->ES->ED sweep + homogeneous control.
- Result: **inner boundary excellent, outer boundary genuinely not
  recoverable.** Inner: RMSE=0.2652mm, consistent across all 8 frames
  (0.12-0.45mm) — as good as every other shape tried this thread. Outer:
  RMSE=2.4414mm, and NOT just noisy — several frames (3,4,5,6) show
  outer-fitted values IDENTICAL (55.0 cells) despite 4 different true
  outer radii (77.8, 71.0, 71.0, 77.8) — the same "stuck at a fixed
  value" signature that has marked every artifact found this session;
  other frames (1,2,7,8) show the outer fit landing almost exactly on
  that SAME frame's own fitted INNER radius (e.g. frame 1: inner
  fitted=61.2, outer fitted=61.2 — identical).
- **Visually confirmed** (actually viewed the filmstrip, not just the
  numbers): the fitted inner circle sits precisely on the true inner
  boundary in every frame. Critically, the ANNULUS between the true
  inner and outer boundaries — where the myocardial wall's outer
  (epicardial) reflection should show a bright ring — is essentially
  DARK. There is no distinguishable bright ring at the true outer
  radius anywhere in the image. The strong bright features in the image
  are clustered near the inner boundary and along the probe axes, not
  near the true outer radius at all.
- **This is not a new shape-fit bug — it's a physics/signal-strength
  problem this thread hadn't touched yet.** The outer fit isn't
  computing something wrong; there is genuinely no strong recoverable
  signal at the true outer location for it to find, so it lands on
  whatever's nearby (the strong inner echo, or a fixed intermediate
  artifact) instead. **This directly reconnects to Thread 1 from
  earlier in this whole session** (`SESSION_HANDOFF_2026-07-08.md`:
  "single-echo boundary detection — CLOSED... two reflecting interfaces
  [chest-wall/myocardium, blood/myocardium] are comparably weak...
  making single-echo amplitude/correlation detection structurally
  ambiguous between them"). The multistatic + global-shape-fit approach
  developed across this entire thread solved a DIFFERENT problem
  (robustly localizing ONE known reflector against ghosts/noise/
  ambiguous local peaks) — it was never tested against, and does not
  automatically solve, the original two-comparably-weak-boundaries
  problem that closed Thread 1 months of (session-)time ago.
- Physical sanity checked? by whom?: Claude — actually viewed the
  filmstrip image before concluding this was a physics limitation
  rather than a code bug; specifically checked for the "stuck at fixed
  value" signature (comparing multiple frames' outer-fit values against
  each other, not just against ground truth) before characterizing it
  as an artifact-lock rather than random noise; connected this result
  back to the ORIGINAL Thread 1 finding from early in this session
  rather than treating it as a brand-new, unexplained problem.
- Gate passed? (Y/N): N/A.
- Visuals produced: (1)
  `results/figures/phase3_ring_phantom_shapefit_score_curve.png` — ED
  frame score-vs-R for both boundaries side by side (outer boundary's
  curve should show visibly less structure/confidence than inner's,
  consistent with the weak/absent true signal there); (2)
  `results/figures/phase3_ring_phantom_shapefit_filmstrip.png` — 8-frame
  filmstrip, both true and fitted circles overlaid, showing the dark
  annulus described above.
- Next action: the outer (epicardial) boundary recovery remains a
  genuinely open, hard problem — NOT solved by this thread's shape-fit
  advances, and should not be assumed solved when reporting overall
  progress. Options for a future session: (a) revisit whether the cited
  tissue properties (`phase2_config.py`) give the outer interface
  enough real acoustic contrast to be detectable at all with this
  probe/frequency setup, independent of algorithm; (b) use the KNOWN
  wall-thickness constraint (WALL_THICKNESS_CELLS=30, held constant per
  the motion model) to REGULARIZE the outer search (fit outer_R =
  inner_R_fitted + wall_thickness, rather than an independent blind
  search) — a principled, testable idea, not yet tried; (c) reconsider
  whether frequency/bandwidth choices could be tuned to better resolve
  the specific outer interface. This is the natural next question for
  the real heart-wall phantom work, not a closed/solved item.

### Run 2026-07-08-40 — CORRECTION to run -39: outer interface DOES have strong recoverable signal on its own — the failure is inner-boundary masking/two-boundary separation, NOT a weak-tissue-contrast problem
- Phase: 3 (continuing run -39). Per user correction: attributing run
  -39's outer-boundary failure to "Thread 1's weak blood/myocardium
  interface problem" was WRONG, because that framing predicts the INNER
  boundary (which IS the blood/myocardium interface) should fail too —
  but it didn't (0.27mm RMSE, excellent). The real, narrower question:
  is this a two-boundary SEPARATION/masking problem, not a blanket
  weak-signal problem? `src/phase3_outer_boundary_diagnostic.py`.
- Design: per user's proposed most-decisive control — build a
  MYOCARDIUM DISK in chest-wall-proxy background, with NO inner blood
  cavity and no competing boundary at all, at R=90 cells (exactly
  matching run -39's ED-frame true outer radius). Run the identical
  validated naive + global shape-fit pipeline. If this recovers well in
  isolation, it proves the myocardium/chest-wall-proxy interface has
  real detectable signal, and the ring's failure is specifically
  inner-boundary interference — not the interface itself being
  undetectable.
- Result: **PERFECT recovery — fitted R=90.0, true R=90.0, error=0.00mm.**
  Visually confirmed: a clean, bright, unambiguous ring sitting exactly
  on the true boundary, cyan (true) and green (fitted) circles
  essentially identical. This is not just "good" — it is comparable to
  or better than every other single-boundary shape tried this thread
  (circle 0.24mm, triangle 0.23-0.81mm, heart-cartoon 0.23mm), and
  strictly better than the ring's own inner-boundary result (0.27mm).
- **Verdict: the myocardium/chest-wall-proxy interface DOES have
  strong, real, independently-recoverable signal. Run -39's
  characterization was imprecise and is corrected here.** The outer
  boundary's failure in the ring phantom is NOT a case of "the
  interface is too weak to detect" — it is specifically caused by the
  PRESENCE OF THE INNER BOUNDARY: masking, energy domination, and/or
  the independent-per-boundary search being pulled toward the inner
  boundary's much stronger nearby echo (consistent with run -39's own
  observation that several frames' outer-fit values matched that same
  frame's inner-fit value almost exactly). This is a genuine
  TWO-BOUNDARY SEPARATION problem, not a signal-detectability problem —
  a materially different, more precise, and more tractable
  characterization than what run -39 concluded.
- Physical sanity checked? by whom?: Claude — ran the single most
  decisive control the user proposed (isolate the outer interface
  completely, remove the confound) rather than reasoning abstractly
  about which explanation was more likely; visually confirmed the
  result (not just the printed 0.00mm) before accepting it, given how
  suspiciously perfect a 0.00mm result is (checked the image directly
  to confirm it's a genuine clean ring, not a degenerate/empty
  accumulator coincidentally scoring highest at the true R).
- Gate passed? (Y/N): N/A.
- Visuals produced: `results/figures/phase3_outer_boundary_diagnostic.png`
  — myocardium disk alone, true and fitted circles overlaid, essentially
  perfectly matching.
- Next action: per the user's proposed control list, the next concrete
  tests (not yet run) to fully characterize the two-boundary
  interaction: (1) same ring, but with inner contrast NEUTRALIZED
  (fill the LV cavity with myocardium instead of blood, i.e. a myocardium
  disk with NO inner boundary at all but at the OUTER radius specifically
  in the context of the full ring geometry/probe setup — already
  effectively done by this run, confirming isolation works); (2) the
  full ring WITH inner contrast present, but test a COUPLED/regularized
  outer fit (outer_R = inner_R_fitted + WALL_THICKNESS_CELLS, using the
  already-excellent inner fit plus the KNOWN constant wall-thickness
  constraint, rather than an independent blind search) — the most
  promising concrete next step, directly informed by this run's
  confirmation that the outer signal exists and just needs to be found
  without competing against the inner boundary's dominance; (3) score-
  curve inspection comparing energy at the true outer_R vs the inner
  boundary's radius/fixed-artifact locations, to visualize exactly how
  the inner boundary's energy is pulling the independent outer search
  away from the true answer.

### Run 2026-07-08-41 — Guard-band outer fit: real but PARTIAL improvement; a second, distinct artifact remains — constant-wall-thickness coupling explicitly REJECTED as clinically harmful
- Phase: 3 (continuing run -40). Per user's explicit objection: coupling
  `outer_R = inner_R_fitted + WALL_THICKNESS_CELLS` (the natural next
  step flagged in run -40) was REJECTED before implementation — real
  myocardial wall thickness varies regionally and pathologically
  (hypertrophy, post-infarct thinning, aneurysmal bulging), and
  detecting exactly that variation is a primary purpose of cardiac
  imaging. Hard-coding constant thickness into the reconstruction would
  structurally and SILENTLY blind it to the pathology it exists to
  find — a confident, clean-looking, wrong answer, worse than an
  obvious failure. This is now a standing rule for this project, not
  just a one-off caveat: **never encode an anatomical constant-value
  assumption (wall thickness or otherwise) into a fitting/reconstruction
  method in a way that would suppress genuine detection of the
  corresponding pathology.**
- **Alternative implemented instead**: `src/phase3_ring_outer_guardband_fit.py`.
  Uses the already-reliable inner-boundary fit only to EXCLUDE a narrow
  guard band (+/-8 cells) immediately around the KNOWN inner radius from
  the outer boundary's search grid — removes the specific confound
  (inner-boundary energy dominating candidates near its own radius)
  WITHOUT assuming anything about where the true outer radius is. The
  outer search remains completely free elsewhere, so a real heart with
  abnormal or asymmetric wall thickness would still be found on its own
  merits, not overridden.
- Result: **a real but only PARTIAL improvement — does not solve the
  problem.** Outer RMSE: independent (run -39)=2.44mm -> guard-band=
  1.92mm. The guard band removed the worst failure mode (frames where
  the outer fit landed EXACTLY on the inner fit's own value, e.g. frame
  1: 61.2->69.5, no longer identical to inner=61.2) but the results are
  still clearly wrong (69.5/66.5/57.5/55.0 vs true 90.0/86.2/77.8/71.0),
  nowhere near this thread's usual 0.0-0.3mm accuracy.
- **A second, distinct problem found**: frames 4 and 5 (inner_R=41.0)
  give the IDENTICAL wrong outer value (55.0 cells) under BOTH the
  independent and guard-band methods. Since the guard band excludes
  only candidates within 8 cells of the inner fit (45.5 +/- 8 =
  [37.5, 53.5]), and 55.0 sits just OUTSIDE that window, the guard band
  never touches this case — meaning 55.0 is a SEPARATE, real artifact,
  not simply "inner-boundary energy leaking near its own radius." Most
  plausible mechanism (not yet tested): genuine reverberation/multi-
  bounce energy between the two boundaries, which sit only 30 cells
  (3mm) apart — close relative to the probe geometry and wavelength
  used here — a physically distinct phenomenon from simple masking,
  which a proximity-based guard band cannot address.
- Physical sanity checked? by whom?: Claude — explicitly checked
  whether the guard band's excluded range actually covered the
  remaining bad value (55.0) before characterizing it as a separate
  artifact, rather than assuming the guard band's partial RMSE
  improvement meant the mechanism was fully understood; reported the
  modest RMSE improvement honestly as partial, not as a fix.
- Gate passed? (Y/N): N/A.
- Next action: (1) do NOT implement constant-wall-thickness coupling —
  explicitly rejected, document this rule in any future finetuning-phase
  planning; (2) investigate the ~55-cell artifact directly (e.g. a
  homogeneous-medium-only reverberation control, or a pair-ablation
  diagnostic analogous to runs -34/-36, to check whether specific pairs
  or genuine multi-bounce timing explain it) before attempting another
  fix; (3) the outer/epicardial boundary remains a genuinely open,
  unsolved problem — the guard band is a legitimate partial mitigation,
  not a resolution, and should not be reported as such.

### Run 2026-07-08-42 — Mechanism of the R~55 false peak diagnosed: NOT reverberation-dominant, it's the inner boundary's real reflection aliased by antipodal/cross-pair geometry
- Phase: 3 (continuing run -41). Per user question: "why is epicardial
  location so inaccurate? did you account for double/triple/multiple
  bouncing?" `src/phase3_ring_outer_ghost_diagnostic.py`.
- Design: three-step empirical diagnosis (no new simulation design
  needed, one capture reused throughout), same methodology validated for
  the triangle's ghosts (runs -34/-36): (1) per-pair energy contribution
  at the false R=55 vs true outer R=71, across all 16 tx/rx pairs, for
  the inner_R=41 frame (the exact case where R=55 appeared in runs
  -39/-41); (2) for the dominant NON-antipodal pair, direct hypothesis
  matching (single-bounce inner, single-bounce outer, 1st/2nd-order
  internal reverberation) against the pair's own actual recorded peaks;
  (3) same hypothesis matching for the dominant ANTIPODAL pair, using a
  proper OFF-AXIS test point (theta=90) after the initial on-axis
  monostatic-direction approximation failed (divide-by-zero) on the
  antipodal pair's degenerate midpoint-at-center geometry (the same
  degenerate-locus property first found in run -29).
- Result: **jWave's full-wave simulation does contain genuine multiple-
  bounce reverberation, confirmed directly** — for the dominant non-
  antipodal pair (`top->right`), a real peak at t=19.697us matches the
  1st-order-reverberation prediction (19.459us, i.e. wave reflects off
  inner, crosses back to outer, reflects again, exits) almost exactly.
  **But reverberation is NOT the dominant cause of the R~55 anomaly.**
  The actual dominant contributors are the ANTIPODAL pairs
  (`bottom<->top`, `left<->right`), each carrying ~3-6x more energy at
  R=55 than any other pair category, while plain MONOSTATIC pairs
  (`top->top` etc.) correctly favor the TRUE outer boundary (ratio 0.45,
  i.e. more energy at true R=71 than at false R=55) — consistent with
  run -40's clean isolated-disk recovery. For BOTH the dominant
  non-antipodal pair (`top->right`) and the dominant antipodal pair
  (`bottom->top`, tested off-axis), the actual strongest real peak
  matches the INNER boundary's predicted time, NOT the outer boundary's,
  and NOT any reverberation hypothesis for the antipodal pair
  specifically (its only >=15%-of-max peak matched inner exactly,
  16.799us observed vs 16.663us predicted).
- **Mechanism, stated precisely**: the inner (blood/myocardium) boundary
  produces a strong, genuine reflection that antipodal and cross-pair
  bistatic geometries pick up clearly — but the naive single-bounce-per-
  candidate-radius model, when summing this real inner-boundary energy
  across the 72-angle sweep at CANDIDATE radius R=55 (not R=41, its true
  source), finds enough accidental time-of-flight agreement to create a
  false peak at that intermediate radius. This is the SAME class of
  mechanism already confirmed for the triangle's corner ghosts (run -36:
  real energy from one location, misattributed to a different candidate
  location by a model that doesn't check whether the implied reflection
  geometry is actually valid for that specific pair) — just manifesting
  via a DIFFERENT pair category (antipodal, not adjacent) because the
  ring's rotational symmetry changes which bistatic pairs are
  geometrically degenerate/dominant compared to the triangle's fixed
  vertex geometry.
- **Answering the user's question directly**: yes, multiple/reverberation
  bouncing is real and directly confirmed in this simulation (jWave
  solves the full wave PDE, so all bounce orders are automatically
  present) — but it is a secondary, weaker effect here, not the primary
  cause. The primary cause is the (much stronger) inner boundary's own
  genuine reflection being aliased to the wrong apparent radius by the
  antipodal/cross-pair geometry's naive single-bounce interpretation —
  not literally "the Doppler-shifted, multiply-bounced signal changes
  properties each time," though that IS a real, present, secondary
  contributor (confirmed for the non-antipodal pair's 3rd peak).
- Physical sanity checked? by whom?: Claude — used a genuinely different,
  well-conditioned off-axis test point (theta=90) for the antipodal pair
  after the first (on-axis, monostatic-direction) approach failed with a
  divide-by-zero, rather than skipping the antipodal pairs' analysis
  entirely; directly matched real recorded peaks against multiple
  concrete hypotheses (inner, outer, reverb-1, reverb-2, false-R-naive)
  rather than asserting a mechanism without checking the raw trace.
- Gate passed? (Y/N): N/A.
- Next action: the natural, principled next test (not yet run,
  analogous to the triangle's validated pair-exclusion approach, runs
  -34/-38) is checking whether excluding/down-weighting the antipodal
  AND cross pairs specifically (keeping only the 4 monostatic pairs,
  which already correctly favor the true outer boundary per run -40 and
  this run's own per-pair table) improves the outer-boundary fit,
  WITHOUT encoding any assumption about wall thickness or anatomy
  (purely an instrumentation/pair-geometry decision, unlike the
  explicitly-rejected constant-wall-thickness coupling from run -41).
  Should be checked against the SAME caution as run -34 (the triangle's
  left-facet regression) — confirm this doesn't quietly break something
  else (e.g. the inner boundary's own already-excellent fit) before
  treating it as a clean win.

### Run 2026-07-08-43 — Pair-CLASS ablation: monostatic-only DEMONSTRATES real reconstruction collapse; no fixed pair-subset is both safe and complete
- Phase: 3 (continuing run -42). Per explicit user direction: "Do
  pair-class ablation, not anatomical cancellation" with 5 named
  configs (A: all 16; B: monostatic only; C: remove antipodal only; D:
  remove all run-42-implicated pairs; E: monostatic + selected
  non-aliased pairs), measuring inner RMSE, outer RMSE, score-curve
  confidence, whether outer locks to inner, and whether inner recovery
  collapses — explicitly "carried forward to physiological and physical
  reasonability," i.e. checked for generalization, not just a single
  idealized-case win. `src/phase3_ring_pair_ablation.py`.
- Design: all 5 configs derived directly from run -42's own per-pair
  evidence table (not guessed) — B/D use the 4 monostatic pairs (the
  only ones with ratio<1, i.e. the only ones that correctly favored the
  true outer boundary); C removes only the single worst antipodal
  offenders; E adds back the 2 least-biased cross pairs (ratio 1.19) to
  the monostatic set. Tested at TWO frames (ED, inner_R=60, strong
  signal; ES-adjacent, inner_R=41, the exact weaker-signal case where
  run -39/-42's R~55 anomaly was found) specifically to check whether a
  fix that looks good on one frame collapses on another — the same
  rigor that caught the triangle-tuned pair-exclusion's failure on the
  heart shape (run -37).
- Result: **no fixed pair-subset is both safe and complete — a real,
  demonstrated trade-off, not a clean winner.**
  | config | ED inner/outer err | ES-adjacent inner/outer err |
  |---|---|---|
  | A (all 16) | 0.12mm / 2.88mm (locked) | 0.45mm / 1.60mm |
  | B (monostatic only) | 0.10mm / **0.00mm** | **3.10mm COLLAPSED** / 0.10mm |
  | C (remove antipodal only) | 0.10mm / 2.90mm (locked) | 0.15mm / 0.12mm |
  | D (= B numerically) | same as B | same as B (collapse) |
  | E (monostatic + 2 least-biased cross) | 0.10mm / 2.90mm (locked) | 0.10mm / 0.12mm |
- **Monostatic-only (B/D) demonstrates REAL reconstruction collapse,
  confirming the user's exact concern.** At the ED frame it looks
  perfect (outer error exactly 0.00mm). But at the harder ES-adjacent
  frame, removing 12 of 16 pairs breaks the INNER boundary
  catastrophically (3.10mm error, fitted value 72.0 — it locked onto
  the OUTER boundary's own location, 71.0 true, instead of its own true
  41.0). This is not a hypothetical risk raised in the prior
  conversation turn — it is now directly demonstrated: discarding most
  pairs removes exactly the redundancy that kept the inner fit robust
  at weaker signal, and the reduced set fails in a NEW, worse way not
  visible from the strong-signal frame alone.
- **C and E are safer but incomplete.** Removing only the antipodal
  pairs, or adding back the 2 least-biased cross pairs to the
  monostatic set, avoids collapse at BOTH frames (inner stays accurate,
  0.10-0.15mm, at both) — but neither fixes the outer boundary at the ED
  frame (still ~2.9mm, still locked to inner there). A genuine trade-off
  across signal conditions, not a universal fix.
- **Conclusion**: the set of "trustworthy" pairs for outer-boundary
  recovery appears to depend on signal strength/frame — which in real
  anatomy would also vary with cardiac phase, patient, and pathology.
  This argues AGAINST any static, condition-independent pair-class rule
  as a final answer, and FOR the adaptive, signal-content-based
  approach proposed in the prior turn (down-weight a pair's outer-
  boundary contribution specifically when its own real recorded energy
  better matches ITS OWN inner-boundary prediction, evaluated per frame,
  rather than excluding whole pair categories by fixed geometric rule).
- Physical sanity checked? by whom?: Claude — tested at TWO frames
  specifically to probe for exactly the collapse risk raised in
  conversation, rather than reporting only the ED frame's flattering
  numbers; correctly distinguished "outer locked to inner" from "inner
  collapsed onto outer" (same numerical symptom, different failure
  direction) rather than conflating them under one boolean flag.
- Gate passed? (Y/N): N/A.
- Next action: (1) do NOT adopt monostatic-only (or its numerically-
  identical D) as a fix — demonstrated unsafe; (2) implement and test
  the adaptive, per-frame, signal-content-based down-weighting approach
  instead of any further fixed pair-subset; (3) before trusting any
  eventual fix, test across MORE than 2 frames (ideally the full 8-frame
  cycle) and, per the standing generalization concern from runs -37/-38,
  eventually check a non-circular/asymmetric boundary too — a circular,
  centered ring may not expose failure modes a real, irregular
  myocardial wall would.

### Run 2026-07-08-44 — ROOT CAUSE FOUND: outer boundary's reflection collapses at wide bistatic angles (curvature-dependent divergence), inner's doesn't — confirmed with real simulated amplitudes, not just geometry
- Phase: 3 (continuing run -43). Per user question: "what makes the
  inner ring accurate but not the outer one? how to isolate them?"
  Two-step investigation, both pure diagnosis (no fix implemented yet).
- **Step 1 (`src/phase3_ring_curvature_diagnostic.py`, pure geometry, no
  simulation): "does a specular point exist" hypothesis REFUTED.**
  Tested whether the inner (R=41) vs outer (R=71) circle differ in
  whether a valid law-of-reflection point exists anywhere on the
  boundary for monostatic/cross/antipodal pairs (same specular-defect
  method validated in run -36's triangle mechanism diagnosis, applied
  to a full circle instead of polygon edges). Result: defect ~0.00000
  for EVERY pair type at BOTH radii — a valid specular point always
  exists somewhere on a full circle for any receiver, unlike the
  triangle's finite flat edges. This hypothesis was cleanly and quickly
  disproven, reported immediately rather than building further on it.
- **Step 2 (`src/phase3_ring_amplitude_divergence_test.py`, real
  simulated amplitudes): REFINED hypothesis CONFIRMED — curvature-
  dependent reflection divergence, not existence.** Isolated each
  boundary alone (myocardium disk in chest-wall-proxy background, no
  competing boundary, same construction validated in run -40), at BOTH
  R=41 and R=71, captured all 16 pairs for each, and compared each
  pair's REAL recorded amplitude at its own predicted specular time,
  grouped by baseline angle:
  | baseline angle | inner (R=41) amp | outer (R=71) amp | outer/inner ratio |
  |---|---|---|---|
  | monostatic (0deg) | 0.000066 | 0.000129 | 1.96 (outer STRONGER) |
  | cross (90deg) | 0.000009 | 0.000000 | 0.001 (outer collapses) |
  | antipodal (180deg) | 0.000003 | 0.000000 | 0.012 (outer collapses) |
  At monostatic incidence the outer boundary is the STRONGER reflector
  (consistent with it being the first interface hit at full incident
  amplitude, and its cited-tissue reflection coefficient being
  slightly larger than the inner's, 0.0035 vs 0.0025 — computed from
  BLOOD/MYOCARDIUM/CHEST_WALL_PROXY impedances in `phase2_config.py`
  before this test, ruling out "outer is just weaker" as an
  explanation). But at ANY wide-baseline pair, the outer boundary's
  reflected energy collapses to essentially nothing, while the inner
  boundary's — though weaker overall — decays far more gently and
  stays detectable even at antipodal (180 degree) incidence.
- **Root cause, stated precisely**: the outer (larger, flatter-locally)
  circle reflects like a near-flat mirror, concentrating nearly all its
  energy into a narrow cone around the monostatic direction. The inner
  (smaller, more sharply curved) circle reflects like a diverging convex
  mirror, spreading its weaker reflection across a much wider angular
  range. When all 16 pairs are summed, the outer boundary's real signal
  exists ONLY in the 4 monostatic pairs; the other 12 (cross+antipodal)
  pairs carry essentially ZERO real outer signal, so they aren't
  neutral — they vote for whatever they DO see (the inner boundary or
  residual structure), and outnumber the 4 correct pairs 3-to-1,
  dominating the naive sum. This directly explains run -42's finding
  (cross/antipodal pairs' real peaks matched the INNER boundary's
  predicted time, never the outer's) and run -43's finding (only
  monostatic pairs correctly favored the outer boundary, and every
  fixed pair-subset rule was either incomplete or fragile).
- **Why this is a better foundation than any pair-class rule**: pair
  reliability for a given boundary depends on that boundary's OWN
  curvature relative to the pair's baseline angle — a continuous,
  physical relationship, not a fixed geometric category. This is
  exactly why runs -41/-43's fixed pair-subset attempts were each
  either unsafe (monostatic-only collapsed the inner fit at weak
  signal) or incomplete (safer subsets didn't fix the outer boundary at
  strong signal) — none of them modeled the actual underlying physics.
  A curvature-aware weighting would generalize naturally to real
  anatomy's continuously-varying curvature (steeper at trabeculae/
  papillary muscles, flatter over smooth wall segments) instead of
  needing a new hand-tuned rule per shape — directly serving the user's
  explicit "carried forward to physiological and physical
  reasonability" goal.
- Physical sanity checked? by whom?: Claude — computed cited-tissue
  reflection coefficients BEFORE running the amplitude test, to rule
  out "outer is just intrinsically weaker" in advance (it's actually
  slightly stronger, 0.0035 vs 0.0025); ran the cheap pure-geometry test
  FIRST and reported its clean negative result immediately rather than
  skipping past it to confirm a preferred hypothesis; used ISOLATED
  single-boundary phantoms (no competing boundary) so the amplitude
  comparison isn't confounded by the very two-boundary-masking effect
  under investigation.
- Gate passed? (Y/N): N/A.
- Next action: this is a diagnosis, not yet a fix. The natural next
  step (not yet implemented) is a physically-motivated weighting scheme
  — score a pair's contribution to a candidate boundary using an
  explicit curvature-dependent divergence model (e.g. down-weight a
  pair's vote in proportion to how much its baseline angle would be
  expected to attenuate reflection from a boundary of THAT candidate
  radius, based on the convex-mirror divergence relationship just
  measured) rather than any fixed pair-category inclusion/exclusion.
  Should be validated the same way run -43's ablation was scoped:
  across multiple frames/signal-strengths, and eventually a non-
  circular boundary, before being trusted.

### Run 2026-07-08-45 — Curvature-aware weighting + guard band: BOTH boundaries recovered at BOTH tested frames, no collapse — first complete success on the ring phantom
- Phase: 3 (continuing run -44). Per "proceed": implemented the
  physically-motivated fix run -44 pointed to.
  `src/phase3_ring_curvature_weighted_fit.py`.
- Design: (1) a curvature-aware weight, replacing every fixed pair-
  subset rule from runs -41/-43 — each pair's contribution at a
  candidate radius R is scaled by weight(baseline_category, R), a
  SIMPLE LINEAR interpolation/extrapolation (explicitly a first
  approximation, not a derived physical formula) between run -44's two
  measured calibration points (R=41: cross/mono=0.136, antipodal/mono=
  0.045; R=71: both effectively 0.000), clipped to [0,1] outside the
  measured range. Monostatic pairs always weight 1.0. This varies
  smoothly PER CANDIDATE R within a single sweep (small R keeps
  cross/antipodal pairs' redundancy; large R naturally down-weights
  them), unlike any fixed inclusion/exclusion list. (2) A guard band
  (run -41's idea) excluding OUTER_R_GRID candidates within 8 cells of
  the ALREADY-FITTED inner radius — added after finding the curvature
  weighting alone still failed at the ED frame specifically because
  OUTER_R_GRID's own range (55-110) overlaps the true inner radius (60)
  at that frame, letting the inner boundary's raw signal strength push
  through even a reduced (not zero) weight.
- Result: **curvature-weighting alone was a genuine partial success,
  and adding the guard band completed it — full success at both tested
  frames.**
  | frame | inner | outer (curvature-weight only) | outer (+ guard band) |
  |---|---|---|---|
  | ED (strong signal) | 0.10mm | 2.90mm (locked to inner) | **0.00mm** |
  | ES-adjacent (weak signal) | 0.12mm | 0.12mm (not locked) | 0.12mm (unchanged, guard band not needed here) |
  This is the FIRST configuration in this entire ring-phantom
  investigation (runs -39 through -45) to recover BOTH boundaries
  accurately at BOTH signal-strength conditions tested, with no
  reconstruction collapse anywhere — succeeding specifically where
  monostatic-only (run -43) collapsed the inner fit at weak signal, and
  where the safer pair-subsets (C/E, run -43) never fixed the outer
  boundary at strong signal.
- **Why curvature-weighting alone wasn't enough, diagnosed not just
  patched**: the ED frame's true inner radius (60) sits WITHIN
  OUTER_R_GRID's own search range (55-110) — a search-range overlap
  problem, distinct from the curvature/divergence mechanism itself.
  Even with cross/antipodal pairs correctly down-weighted (not
  excluded) near R=55-60, the inner boundary's absolute signal strength
  is large enough that a small residual weight times a very strong
  signal can still outscore the correctly-weighted but genuinely WEAK
  true outer signal at R=90. The guard band removes this specific
  overlap without reintroducing any anatomical assumption (it uses the
  ALREADY-MEASURED inner fit, not a presumed wall thickness) and
  without touching the inner fit's own accuracy (it only restricts the
  OUTER search).
- Physical sanity checked? by whom?: Claude — tested curvature-
  weighting alone FIRST and reported its partial (not complete) result
  honestly before combining it with the guard band, rather than only
  reporting the final combined success; diagnosed WHY the plain
  curvature-weighted version still failed at ED (grid-range overlap)
  before proposing the guard-band addition, rather than adding it
  speculatively; confirmed the guard band had ZERO effect at the
  ES-adjacent frame (identical 0.12mm before and after), consistent
  with it only being needed/triggered when the true inner radius
  actually falls within the outer grid's range.
- Gate passed? (Y/N): N/A.
- **Caveats, not yet resolved**: (1) the curvature weight model is a
  crude 2-point linear interpolation (only R=41 and R=71 were ever
  measured, run -44) — more calibration radii would make this a more
  defensible model, not just a convenient fit through 2 points; (2)
  tested at only 2 frames (ED, ES-adjacent), not the full 8-frame
  cardiac cycle — the 4 intermediate frames are untested; (3) tested
  ONLY on this idealized circular, centered ring — the standing
  generalization concern from runs -37/-38 (a fix that works on one
  shape may fail on a different one) has NOT been checked here yet,
  and is the most important remaining validation step before treating
  this as a real result rather than a promising one.
- Next action: (1) run the full 8-frame sweep to confirm across the
  whole cycle, not just the 2 tested extremes; (2) test on an off-
  center and/or non-circular boundary (reusing the triangle/heart-
  cartoon infrastructure) before trusting this generalizes past the
  ring's perfect rotational symmetry; (3) if pursued further, measure
  additional calibration radii to replace the 2-point linear
  interpolation with a better-supported curve.

### Run 2026-07-08-46 — Full 8-frame eccentric off-center generalization test PASSES: both boundaries recovered accurately at every frame, no collapse
- Phase: 3 (continuing run -45). Per user: "now run with off center
  heart phantom, with the 2 ring's epicenter off (to create thick/thin
  regions mimicking true LV wall). do the 8 frame one." This directly
  tests the two things flagged as essential before trusting run -45:
  the full 8-frame cardiac cycle (not just 2 test frames), and a non-
  idealized geometry (not the perfectly centered, concentric ring).
  `src/phase3_ring_eccentric_offcenter_test.py`.
- Design: two FIXED offsets (not scaling with radius, applied
  identically across the whole cycle): (1) OUTER_CENTER_OFFSET=(8,6)
  cells — the whole phantom shifted off the domain/probe-array center,
  same style as run -35's off-center triangle test; (2)
  INNER_ECC_OFFSET=(6,5) cells — the inner (LV cavity) circle's center
  further offset from the outer (epicardial) circle's OWN center,
  creating a physiologically-realistic thick/thin wall pattern (a
  standard way to model basal/regional wall asymmetry). Eccentricity
  magnitude 7.81 cells (0.78mm) against the nominal 30-cell (3mm) wall
  thickness gives a thin side ~2.22mm and thick side ~3.78mm — a ~1.7x
  thickness ratio, clearly asymmetric and clinically meaningful. Both
  circles still follow the existing constant-wall-thickness motion
  schedule (inner 60->40->60 cells, outer=inner+30) — only their
  relative POSITION is eccentric, not the temporal motion pattern.
  Required a WIDER local accumulator grid (+/-100 cells vs the default
  +/-80 used elsewhere in this thread) since the off-center, up-to-R=90
  phantom would otherwise get clipped at the existing grid's edge —
  verified bounds numerically before running (farthest point row=248,
  col=246 vs domain N=(300,300) and grid range [40,260], comfortable
  margin). Used run -45's validated method as-is: curvature-aware
  weighting (2-point linear interpolation) + guard band around the
  already-fitted inner radius, with each boundary's ray-sweep origin
  set to its OWN true center (INNER_CENTER for the inner fit,
  OUTER_CENTER for the outer fit) — testing whether the RADIUS-FITTING
  mechanism holds under eccentricity, not simultaneously solving the
  separate, harder center-detection problem.
- Result: **PASSES cleanly across all 8 frames — no collapse, no
  locking, consistent accuracy throughout the full cardiac cycle.**
  Inner RMSE=0.3127mm (per-frame range 0.30-0.33mm, no outliers). Outer
  RMSE=0.2403mm (per-frame range 0.18-0.27mm, no outliers). "Outer
  locked to inner?" = False at EVERY single frame. Both boundaries
  recovered accurately and consistently through the entire ED->ES->ED
  cycle with a genuine 1.7x wall-thickness asymmetry present.
- **Comparison to the idealized case**: errors are modestly larger than
  the perfectly centered, concentric 2-frame test (0.24-0.33mm here vs
  0.10-0.12mm in run -45) — an expected, small cost of the added
  realism (off-center position, eccentric wall, and a coarser R_GRID
  step, 0.5 cells here vs 0.25 in run -45, needed to keep the wider
  sweep's compute cost reasonable across 8 frames) — but still solidly
  within the "good, usable" range this entire thread has established
  (comparable to the plain circle's 0.24mm run -28, the heart-cartoon's
  0.23mm run -38).
- **This is the generalization test that mattered, and it passed.**
  Every previous fix attempt in this thread that looked clean on one
  idealized case failed when generalization was actually checked
  (triangle's pair-exclusion failing on the heart-cartoon shape, run
  -37; monostatic-only collapsing at weak signal, run -43). The
  curvature-weighted + guard-band method is the first one in this
  entire ring-phantom investigation to survive BOTH the full 8-frame
  cycle AND a non-idealized (off-center, eccentric) geometry test.
- Physical sanity checked? by whom?: Claude — verified grid/domain
  bounds numerically before running the (longer, ~6.5 minute) 8-frame
  sweep, to avoid a silently-clipped or truncated result; confirmed
  "locked to inner" was False at every frame individually, not just in
  aggregate, before calling this a clean pass.
- Gate passed? (Y/N): N/A.
- **Standing caveats, not fully resolved, carried forward honestly**:
  (1) the curvature weight model is still only a 2-point linear
  interpolation (run -44's two measured radii) — not yet re-validated
  with additional calibration points; (2) each boundary's fit used its
  OWN TRUE center as the ray-sweep origin — this test validates the
  RADIUS-fitting mechanism under eccentricity, but does NOT test
  whether the two centers could be found blind (a separate, harder,
  not-yet-attempted joint position+radius fitting problem); (3) the
  guard band's "compare candidate outer radius VALUE to fitted inner
  radius VALUE" logic is a coarser proxy for spatial overlap now that
  the two circles have different centers (flagged in the script
  docstring, not re-derived) — it still worked here, but wasn't
  re-examined rigorously for why; (4) still only tested on a ring
  (two circles) — the earlier concave heart-cartoon shape (which broke
  every fixed pair-subset rule, run -37) has not yet been retested with
  this newer curvature-weighted approach.
- Next action: this is a strong, genuine validation milestone for the
  two-boundary myocardial phantom work. Natural follow-ons if this
  thread continues: (1) test whether the SAME curvature-weighted +
  guard-band approach also fixes the concave heart-cartoon shape's
  earlier total failure (run -37), which would be a strong sign this
  generalizes past simple circular geometry; (2) attempt the harder
  joint center+radius fitting problem instead of assuming known
  centers; (3) add more calibration radii to the weight model.

### Run 2026-07-08-47 — Data prep: real MRI-derived irregular ring, rescaled + smoothed (escalation from synthetic eccentric ring)
- Phase: 3 (data preparation for the next escalation). Per user:
  "next phase is stress-test on MRI-derived reconstructed hearts...
  maybe reconstruct the irregular ring from one of the mri, a sound
  escalation from smooth eccentric off-center rings" then "stick with
  1 [patient/slice] first... apply smoothing, mimicking true
  tissue-like irregularities."
- Seed / config / grid / timestep: no RNG (deterministic
  extraction/rescaling). `src/phase3_mri_irregular_ring_prep.py`. Real
  anatomy source: ACDC `patient001` ED-frame segmentation
  (`pilot/data/processed/ACDC/patient001.npz`, native spacing
  1.5625mm/px, volume shape (10,128,128)). Slice selection: picked the
  slice with maximum LV cavity pixel count across the 10-slice volume
  (slice 4, LV area=1765px) to avoid basal/apical slices where the ring
  is incomplete. Myocardium (label 2) + LV cavity (label 3) isolated;
  RV (label 1) deliberately excluded to keep this a direct 2-tissue-
  boundary analog of the already-validated ring tests (runs -39
  through -46), not a 3-chamber model.
- Rescaling: combined native-pixel-to-acoustic-grid AND toy-scale zoom
  into one **nearest-neighbor** (`scipy.ndimage.zoom(..., order=0)`)
  resample, per CLAUDE.md's explicit mask-resampling rule — real LV
  area (4309.1mm^2, equivalent radius 37.04mm) rescaled isotropically
  (shape preserved exactly, no distortion) to match this thread's
  established toy scale (target LV radius 60 cells / 6mm, matching
  `phase3_config.LV_RADIUS_ED_CELLS`). Zoom factor=2.531x. Achieved LV
  radius after rescale: 60.2 cells (target 60.0, error 0.2 cells) —
  deliberate choice, not incidental: simulating at real anatomical
  scale was already shown infeasible on this CPU (run -09), and a
  different physical scale would invalidate run -44's curvature-
  weighting calibration (measured at toy-scale radii 41/71 cells).
- **Smoothing (added per explicit user request)**: the raw nearest-
  neighbor-upsampled mask has native-pixel staircasing (1.5625mm/px
  native res, only ~2.5x zoom) that is finer than real tissue texture
  but coarser than acoustically-meaningless single-cell noise — left
  untreated, this would partly test the reconstruction method against
  a sampling-grid artifact rather than genuine anatomical curvature.
  Applied Gaussian smoothing (sigma = zoom_factor/2 = 1.27 cells, tied
  to the native pixel size in upsampled-grid units) to the float mask
  post-zoom, re-thresholded at 0.5 — same precedent as run -11's
  `PROXY_AUDIT.md` staircasing check. This is a POST-PROCESS step after
  nearest-neighbor resampling (which still happens via `order=0` zoom),
  not a replacement of the CLAUDE.md mask-resampling rule. Both raw and
  smoothed masks/contours saved (`results/mri_irregular_ring_patient001_slice4.npz`)
  for comparison; smoothed version is what feeds the acoustic build.
- Result: outer contour 599 points, inner contour 473 points (smoothed).
  Achieved LV radius after rescale+smoothing: 60.2 cells (unchanged from
  pre-smoothing, confirming smoothing didn't shift the overall size).
  Figure (`results/figures/phase3_mri_irregular_ring_prep.png`, 3-panel:
  native slice / raw rescale / smoothed rescale) confirms visually: raw
  panel shows clear nearest-neighbor staircasing, smoothed panel removes
  it while preserving genuine non-circular irregularity (asymmetric wall
  thickness, mild flattening on parts of the outer boundary — visibly
  NOT a perfect circle, though this particular patient/slice is a
  relatively mild/typical case, not a dramatically pathological one —
  flagged honestly to the user, who chose to proceed with it as the
  first test rather than searching for a more extreme HCM/DCM case).
- Physical sanity checked? by whom?: user + Claude, visual (3-panel
  figure) — this is data-prep infrastructure, not a new physics claim.
- Gate passed? (Y/N): N/A — data-prep step, not a gated deliverable.
- Observations / surprises: none unexpected; `scikit-image==0.26.0`
  added to `jwave_test/requirements.txt` for contour extraction
  (`skimage.measure.find_contours`), newly installed this run.
- Next action: build the acoustic medium directly from the smoothed
  real contour/masks and run the validated multistatic backprojection +
  curvature-weighted shape-fit (run -46's method), generalized to sweep
  a SCALE FACTOR against the real measured (non-parametric) boundary
  shape instead of a closed-form circle/polygon family — see run -48.

### Run 2026-07-08-48 — Real MRI-derived irregular ring: acoustic reconstruction PASSES (first real-anatomy escalation)
- Phase: 3 (escalation test, single static frame — not yet a multi-
  frame motion cycle, deliberately: the question here is "does the
  method survive a real, non-parametric irregular boundary shape at
  all, or where does it break", not yet full-motion stress testing).
- Seed / config / grid / timestep: no RNG. `src/phase3_mri_irregular_ring_reconstruction.py`.
  Same domain/probes/tissue properties as every other ring test this
  thread (N=(300,300), dx=0.1mm, 4-probe/16-pair geometry, cited
  blood/myocardium/chest-wall-proxy values). Medium built DIRECTLY from
  the smoothed real binary masks (run -47's output), not a synthetic
  formula — real ring centroid translated to the domain center (150,150)
  via integer offset so it sits within the existing validated
  probe/search-grid geometry.
- Readout generalization: since a real contour isn't a circle or a
  known polygon family, each boundary (inner LV, outer epicardium) is
  represented by its own measured r(theta) function (polar-resampled
  from the extracted contour points, relative to that boundary's OWN
  centroid — same "each boundary uses its own true center" convention
  as run -46's eccentric test). The one free parameter swept is a SCALE
  FACTOR s applied uniformly to r(theta) — same "known shape family,
  global template-match" principle validated on every shape this
  thread (circle/triangle/heart-cartoon/ring), just with a measured
  r(theta) instead of a closed-form formula. Curvature-weighted +
  guard-band fit reused exactly as validated in runs -45/-46
  (`pair_weight_at_R`, `GUARD_BAND_CELLS`).
- **Bug caught and fixed before trusting the result**: first attempt
  put the outer guard band in SCALE units (excluding candidate scales
  within 0.10 of the fitted inner boundary's SCALE), which is not the
  same physical quantity as run -45/46's original guard band (physical
  RADIUS cells) since the inner (mean radius 60.3 cells) and outer
  (mean radius 74.0 cells) boundaries have different scale-to-radius
  mappings. Result: outer fit collapsed to scale=0.805 (physical mean
  radius ~59.6 cells — essentially the INNER boundary's radius),
  "locked to inner"=True, err=1.44mm — caught by checking against the
  independently-known true radii, not by inspection. Fixed by
  converting the guard band back to physical-radius-cells units
  (exclude candidate outer scales whose implied mean radius is within
  8 cells of the fitted inner boundary's mean radius), matching
  run -45/46's original approach exactly.
- Result (after fix): **inner (LV) fitted scale=1.015 (true=1.000),
  mean-radius error=0.09mm. Outer (epicardium) fitted scale=1.030
  (true=1.000), mean-radius error=0.22mm. "Outer locked to inner"=False.**
  Both score(s) curves show a single, sharp, unambiguous peak at the
  true scale (figure: `results/figures/phase3_mri_irregular_ring_reconstruction.png`).
  The fitted contours (scaled real r(theta), not an idealized circle)
  visually track the true irregular boundary's actual bumps/flattening
  in the reconstruction panel, not a smoothed approximation of it.
  Homogeneous-medium control: both fits pin to the scale-grid's lower
  edge (0.700, meaningless/no real peak), confirming the real result
  isn't an artifact of the fitting procedure itself.
- Physical sanity checked? by whom?: user + Claude — quantitative
  (known real segmented shape as ground truth, RMSE-equivalent scale
  errors converted to mm), plus the guard-band bug was caught by
  comparing against the independently-known true radii before trusting
  the first (wrong) result — same discipline used throughout this
  project. Not collaborator-reviewed (Gate 2 already passed on the
  underlying physics/tissue model, run -12; this is a new geometry, not
  new physics).
- Gate passed? (Y/N): N/A — escalation/validation test, not a formal
  protocol gate.
- Observations / surprises: the method's first real-anatomy escalation
  passes cleanly (sub-mm on both boundaries), with the SAME validated
  curvature-weighted + guard-band method used for the synthetic
  eccentric ring, no new tuning — genuine evidence the approach isn't
  overfit to synthetic circular geometry. The one real bug this run
  (guard band unit mismatch) is a reminder that "port a fix from a
  circle-radius space to a scale-factor space" is not a purely
  mechanical substitution — the physical quantity the guard band is
  meant to protect (radius overlap) must be re-derived in the new
  parameterization, not assumed to carry over by analogy.
- Next action: this patient/slice was flagged (run -47) as a
  relatively mild/typical irregularity case, not a dramatic one — a
  natural follow-on is testing a more asymmetric ACDC patient (HCM/DCM
  cohort) for a sharper stress test. Also still open: this is a single
  static frame, not a motion cycle — extending to multi-frame real
  motion (e.g. ACDC ED/ES interpolation, or XCAT) remains a separate,
  not-yet-attempted escalation, as discussed with the user before this
  run.

### Run 2026-07-08-49 — Data prep: real registration-derived 8-phase motion cycle (interpolated ED->ES->ED, not a raw-slice stack)
- Phase: 3 (data preparation). User first proposed "take 8 consecutive
  MRI slices" to reconstruct the cardiac cycle — corrected before
  implementing: consecutive SLICES are different anatomical LEVELS
  (base->apex) at one instant, not motion over time, and would be the
  wrong axis entirely. ACDC ground-truth segmentation only exists at 2
  real timepoints (ED, ES) per patient (established Phase I floor) — so
  a real 8-phase cycle must INTERPOLATE between those two real
  contours, not sample 8 additional real segmented timepoints (they
  don't exist). Agreed approach: apply the Phase I registration-derived
  ED->ES displacement field (`pilot/data/processed/ACDC_reg/patient001.npz`)
  at fractional strength across 8 equally-spaced phase samples (same
  half-cosine ED->ES->ED schedule shape as the synthetic toy,
  `phase3_config.lv_radius_at_phase`, applied here to real displacement
  instead of a synthetic radius). User separately confirmed "8 equally
  distanced frames in a sequence" (not 8 raw consecutive cine frames)
  matches this design.
- Seed / config / grid / timestep: no RNG. `src/phase3_mri_motion_cycle_prep.py`.
  Displacement field: `pilot/src/registration.py`'s convention verified
  (not assumed) before use — field is defined on the ES grid, in mm,
  order (dz,dy,dx); D(p) = ED_location - ES_location, so
  warped_ED(p) = ED_mask(p + D(p)). Applied at fraction f via
  `scipy.ndimage.map_coordinates(..., order=0)` (nearest-neighbor,
  mask-safe). **Self-consistency check performed before trusting this**
  (standing project discipline): this script's own f=1.0 warp compared
  directly against the already Dice-validated `warped_ed_mask` saved in
  the same npz — 98.7% pixel agreement, confirming the sign/convention
  was correctly derived.
- Acoustic properties + sources: n/a (data prep only, no simulation
  this run). Same fixed rescale zoom_factor (2.531x) and Gaussian
  smoothing (sigma=1.27 cells) as run -47, computed ONCE from the ED
  frame and reused for all 8 phases (so every phase shares the same
  physical scale, and the fixed ED r(theta) template used by the
  reconstruction fit — run -50 — stays valid across the cycle).
- Result: **registration quality flagged honestly**: patient001's
  mean_dice=0.784 (myocardium dice=0.789), BELOW the pilot's own
  pre-registered 0.80 Gate-3 threshold — myo_dice specifically fails
  the threshold, though LV dice=0.926 and surface distances (0.20mm LV,
  0.53mm myo) are reasonable. In-plane displacement at this slice is
  small (mean 1.58mm, max 7.47mm); through-plane dz (mean 0.95mm) is
  NOT modeled (2D phantom) — flagged as a real, not-yet-addressed
  limitation. **Genuine contraction signal at this patient/slice is
  subtle**: true inner mean-radius only varies 5.94-6.03mm across the
  full cycle (span=0.10mm) — far smaller than the synthetic toy's
  deliberate 6mm->4mm (33%) contraction. Filmstrip
  (`results/figures/phase3_mri_motion_cycle_prep_filmstrip.png`)
  confirms visually: shape stays close to round overall but shows real,
  non-uniform local boundary changes (small bumps shifting position
  frame to frame) — genuine non-uniform deformation, not just uniform
  scaling.
- **Bug caught and fixed before trusting the radius numbers**: first
  attempt computed "mean radius" from ALL FILLED INTERIOR PIXELS
  relative to centroid (np.where on the whole mask), which for a filled
  disk averages to ~(2/3)R, not R — gave nonsensical ~4mm/4.9mm values
  inconsistent with run -47's own 6mm/7.4mm. Fixed by computing mean
  radius from the CONTOUR (boundary) points only, matching run -47/48's
  convention exactly; corrected values (6.03mm/7.40mm at ED) now match
  run -47 exactly, as they must (same ED frame).
- Physical sanity checked? by whom?: Claude — quantitative
  self-consistency check (98.7% agreement vs. Phase I's own
  Dice-validated result) before proceeding; user has not yet reviewed
  the filmstrip.
- Gate passed? (Y/N): N/A — data-prep step.
- Observations / surprises: this patient/slice's real contraction is
  much weaker than the toy's deliberately dramatic 33% — a genuinely
  different (much lower-SNR) regime than every previous test in this
  thread, flagged before running the (expensive) acoustic simulation
  rather than after.
- Next action: run the validated multistatic backprojection +
  curvature-weighted scale-fit (run -46/-48's method) against the FIXED
  ED r(theta) template at each of the 8 phases, to test whether a
  single scale parameter can track this real, non-uniform, low-
  amplitude motion — see run -50.

### Run 2026-07-08-50 — Real 8-phase motion cycle: acoustic reconstruction — sub-mm accuracy, well within the registration floor, but the true signal itself is smaller than the reconstruction's own error
- Phase: 3 (escalation test). First test in this thread where ground
  truth is imperfect real motion (Phase I registration output,
  `labels.GT_FLOOR_CAPTION` applied throughout), not an exactly
  prescribed toy/real-shape value.
- Seed / config / grid / timestep: no RNG. `src/phase3_mri_motion_cycle_reconstruction.py`.
  Same domain/probes/tissue properties as every ring test this thread.
  Medium rebuilt fresh per phase (frozen-scene convention) from run
  -49's smoothed real masks. Scale-factor fit is against the FIXED ED
  r(theta) template (run -47/-48's shape, unchanged across phases) —
  deliberately NOT re-derived per frame, so the test is "does a single
  scale parameter track real non-uniform motion," not a tautological
  per-frame reshape-fit. Each frame's own (currently-true) centroid
  used as that frame's ray-sweep origin (established "own known center"
  convention, runs -46/-48).
- Result: **inner boundary RMSE=0.2255mm, outer boundary RMSE=0.4298mm**
  across all 8 real-motion phases (per-phase errors 0.03-0.76mm) — both
  comfortably within the registration floor (median ~1 voxel/1.5mm).
  Fitted contours visually track the true non-uniform deformation
  closely in all 8 frames
  (`results/figures/phase3_mri_motion_cycle_reconstruction.png`),
  including the worst frame (frac=0.61, outer err=0.76mm, where a
  visible real local flattening the fixed-template pure-scale fit
  can't fully capture is directly visible). **Important honest
  caveat**: the true inner-radius range across this whole cycle is only
  5.94-6.03mm (span=0.10mm, run -49) — SMALLER than the reconstruction's
  own RMSE. This means this specific patient/slice's real contraction
  signal is too subtle to be a meaningful test of "does this method
  track real contraction" — the result demonstrates the method doesn't
  blow up or collapse under genuine non-uniform real deformation
  (a real, useful finding), but does NOT demonstrate accurate tracking
  of a clear contraction signal the way the synthetic toy's 33%
  contraction did.
- Physical sanity checked? by whom?: user + Claude — quantitative
  (known real registration-derived radii as ground truth) + visual
  (filmstrip, fitted vs. true contour). Not collaborator-reviewed (Gate
  2 already passed on physics, run -12; new geometry/motion source, not
  new physics).
- Gate passed? (Y/N): N/A — escalation/validation test.
- Observations / surprises: the method survives its first exposure to
  real, non-uniform, low-amplitude motion without collapsing — a
  genuinely different stress test than every synthetic-toy frame tried
  so far (which all had large, clean, prescribed contraction). The
  finding that the true signal here is smaller than the reconstruction
  error is itself informative, not a failure to hide: it means THIS
  patient/slice is a weak test of contraction-tracking specifically,
  independent of whether the method itself is accurate.
- Next action: a patient with a genuinely larger true contraction
  (e.g., searching ACDC for a higher-ejection-fraction-change or
  hyperdynamic case, not just a more irregular SHAPE) would be a
  sharper test of whether the method tracks real motion accurately,
  as opposed to just surviving it. Also still open, as before: a more
  dramatically asymmetric (HCM/DCM) patient for boundary SHAPE
  irregularity, and independently registering ED to real intermediate
  cine frames (not just fractionally interpolating the single ED->ES
  field) as a higher-fidelity alternative to the half-cosine schedule
  used here.

### Run 2026-07-08-51 — Cohort scan for a patient with adequate real contraction; patient023 selected; static real-shape reconstruction reveals a new outer-boundary limitation
- Phase: 3 (patient selection + escalation test). Per user: "find another
  pt that have adequate cardiac contraction" — patient001 (runs -47/-50)
  turned out to have only ~10% LV radius contraction, one of the weaker
  cases in the whole cohort, too subtle a signal to demonstrate accurate
  contraction-tracking.
- Result: scanned all 150 ACDC patients (`ed_mask`/`es_mask` LV area
  ratio at each patient's own max-LV-area slice, cross-referenced
  against `ACDC_reg` registration Dice). **patient023 selected**: ED
  radius 20.97mm -> ES radius 11.53mm (~45% contraction, vs patient001's
  ~10%), myocardium registration Dice=0.870 (above the pilot's own 0.80
  Gate-3 threshold, unlike patient001's 0.789), mean Dice=0.780, training
  split, and its own max-LV-area slice is also index 4 — same slice
  convention as patient001, no pipeline changes needed beyond the
  patient ID. (143/150 patients had valid ED/ES area + registration
  data; the rest were skipped for zero LV area at their candidate
  slice.) Scripts generalized to take a `PATIENT_ID` argument
  (`phase3_mri_irregular_ring_prep.py`, `_reconstruction.py`) rather
  than being duplicated per patient.
- Ran the static real-shape prep+reconstruction (run -47/-48's method)
  on patient023: prep succeeded cleanly (zoom_factor=4.470x, achieved LV
  radius 60.1 cells vs target 60.0). **Reconstruction revealed a NEW
  finding**: inner (LV) fitted scale=0.925, error=0.45mm — solid, only
  modestly worse than patient001's 0.09mm. **Outer (epicardium) fitted
  scale=0.725, error=2.43mm** — much worse than patient001's 0.22mm, and
  the score curve is multi-peaked/noisy rather than one dominant peak
  (visible in `results/figures/phase3_mri_irregular_ring_reconstruction_patient023.png`).
  Root cause traced (not assumed): patient023's real anatomy has a
  proportionally much thicker myocardial wall than patient001's — outer
  mean radius 88.2 cells vs patient001's 74.0 cells, for the SAME 60-cell
  toy-rescaled LV radius. This also required widening the search grid
  (default +/-90 cells clipped this patient's outer boundary at up to
  ~130 cells at the search's own scale-factor extremes) — added
  `build_search_grid()` to `phase3_mri_irregular_ring_reconstruction.py`,
  sized dynamically per-patient rather than hardcoded, confirmed
  patient001's result is unchanged by this addition (only activates when
  actually needed).
- Physical sanity checked? by whom?: Claude — quantitative (known real
  contour as ground truth); user reviewed the figure and asked the
  right diagnostic question before accepting either "recalibrate" or
  "log and move on" (see run -52/-53).
- Gate passed? (Y/N): N/A — escalation/validation test.
- Next action: diagnose whether the noisy outer fit is a probe-standoff
  artifact (user's hypothesis: "if the heart is too big, why don't
  expand the grid?") or the curvature-weight calibration's radius range
  (run -44 only measured R=41/71; this patient's outer sits at 88,
  beyond that range) — test directly rather than assume either. See
  run -52.

### Run 2026-07-08-52 — Wide-probe-standoff diagnostic: RULED OUT as the cause (identical result at 2x the standoff)
- Phase: 3 (diagnostic, testing user's hypothesis directly rather than
  assuming). `src/phase3_mri_wide_probe_standoff_test.py` — a SEPARATE,
  self-contained script (does not modify the shared
  `phase3_backprojection_shape_fit_triangle` module every other script
  in this thread depends on, to avoid silently changing any
  already-validated result) with PROBE_DIST_CELLS=180 (vs. the standard
  120) and a correspondingly larger domain (N=(460,460),
  center=(230,230)) — giving patient023's outer boundary (88.2 cells) a
  standoff of ~92 cells, roughly DOUBLE patient001's ~46-cell standoff
  at the standard geometry (patient023's own standard-geometry standoff
  was only ~32 cells).
- Key physical rationale for why this is a fair, falsifiable test:
  `pair_weight_at_R` (the curvature-aware weight) is a function of the
  reflector's OWN radius only, not of probe distance — if it's truly a
  far-field curvature/divergence effect (run -44), moving the probes
  farther away should not change the result at all. If it's instead a
  near-field/standoff artifact (multipath, direct-arrival window
  overlap, PML proximity), widening the standoff should visibly improve
  the fit.
- Result: **identical to the standard-geometry result, to 3 decimal
  places** — inner fitted scale=0.925 (err=0.45mm), outer fitted
  scale=0.725 (err=2.43mm), locked_to_inner=False, both exactly matching
  run -51's standard-probe-distance numbers. The outer score curve
  (`results/figures/phase3_mri_wide_probe_standoff_test_patient023.png`)
  is also visually near-identical: no clear peak at true scale=1.0,
  same weak local bump near 0.94-1.0, same dominant feature at the
  guard-band edge. **This cleanly rules out probe standoff as the
  cause** — confirms the limitation is the curvature-weight
  calibration's radius RANGE (only measured at R=41/71), not geometry,
  per the user's own fair hypothesis being tested rather than assumed.
- Physical sanity checked? by whom?: Claude — direct empirical test
  (identical numeric result under a substantially different geometry is
  itself the falsification test, not a side observation).
- Gate passed? (Y/N): N/A — diagnostic test.
- Next action: measure the curvature-weight calibration directly at
  R=88 (patient023's real outer radius) rather than continuing to
  extrapolate/assume, per user: "log and calibrate and log". See run
  -53.

### Run 2026-07-08-53 — R=88 calibration measurement: CONFIRMS (does not correct) the existing extrapolation — outer-boundary limitation is a genuine structural property of large/flat reflectors, not a calibration-range bug
- Phase: 3 (calibration extension + re-test). `src/phase3_ring_calibration_r88.py`
  — exact same method as run -44's original R=41/71 calibration
  (isolated myocardium disk, no competing boundary, standard probe
  geometry — already shown standoff-invariant by run -52), measuring
  the real cross/monostatic and antipodal/monostatic amplitude ratios
  at R=88 directly rather than continuing to rely on the linear model's
  extrapolation past its calibrated range.
- Result: **cross/mono ratio=0.0001, antipodal/mono ratio=0.0003 at
  R=88** — both still essentially zero, consistent with (not
  correcting) the model's already-clipped-to-zero extrapolated
  prediction in that region. Verified the OLD linear-extrapolation
  model's predicted value at R=88 was already ~0 after clipping (slope
  from R=41->71 projected to a negative raw value at R=88, clipped to
  0) — so the new measurement confirms the existing behavior was
  accidentally already physically correct here, not a range-extrapolation
  artifact needing correction.
- Updated `phase3_ring_curvature_weighted_fit.py`'s `_CAL_R`/`_CAL_CROSS`/
  `_CAL_ANTIPODAL` to 3 measured points (41, 71, 88) and switched
  `_linear_weight` from a hand-rolled slope-extrapolation formula to
  `np.interp` (piecewise-linear between measured points, flat-held
  outside the measured range) — a real, if minor, behavior change for
  candidate R BELOW 41 cells (previously extrapolated via the 41-71
  slope, e.g. ~0.15-0.21 at R=25; now flat-held at the R=41 value,
  0.136) — flagged since this technically differs from what runs
  -45 through -50 used, though no already-logged result's SELECTED best-R
  was ever in that sub-41 region, so no prior reported number is
  believed to change in practice; not exhaustively re-verified across
  every prior run.
- **Re-ran patient023's static reconstruction with the updated 3-point
  calibration: result IDENTICAL to run -51** (inner scale=0.925/0.45mm,
  outer scale=0.725/2.43mm) — confirms recalibration does NOT fix this
  patient's outer-boundary accuracy, exactly as predicted once the
  measurement showed the weight was already correctly ~0 there.
- **Conclusion (the actual finding of this 3-run diagnostic arc)**:
  patient023's noisy outer-boundary fit is a genuine STRUCTURAL
  limitation, not a calibration bug or a standoff artifact — a
  reflecting boundary this large/flat (R=88 cells here) really does
  return near-zero energy to all but the 4 monostatic pairs (run -44's
  mechanism, now confirmed rather than assumed at this larger radius),
  so the reconstruction has only 4 independent votes instead of 16,
  inherently less redundant/noisier. Fixing this for real would need a
  structurally different approach (e.g. more independent probe
  angles/monostatic directions), not a coefficient tweak — flagged as a
  real, not-yet-attempted follow-on rather than something to force a
  fix for now.
- Physical sanity checked? by whom?: Claude — quantitative (measured
  amplitude ratios, cross-checked against the old model's own predicted
  value at R=88 before claiming they agree) + re-run confirmation
  (identical numeric result, not just theoretical expectation).
- Gate passed? (Y/N): N/A — calibration/diagnostic extension.
- Next action: per user's original request, proceed to the real
  motion-cycle test for patient023 (runs -49/-50's method), reporting
  the inner boundary's tracking accuracy as the primary result and the
  outer boundary's reduced accuracy as an honestly-flagged, now
  well-understood limitation for this patient's proportions.

### Run 2026-07-08-54 — Real 8-phase motion cycle for patient023: inner boundary tracks a genuine (not noise-floor) contraction signal; outer boundary shows the run -51/-53-diagnosed structural bias, consistently
- Phase: 3 (escalation test, patient023 variant of runs -49/-50).
  `phase3_mri_motion_cycle_prep.py`/`_reconstruction.py` generalized to
  take a `PATIENT_ID` CLI arg (same pattern as the static-shape
  scripts, run -51); slice selection also generalized to the
  max-LV-area convention rather than a hardcoded index (patient023's
  own max-LV-area slice is again index 4, confirmed not assumed).
- Result: registration quality for patient023: mean_dice=0.780,
  myo_dice=0.870 (good), lv_dice=0.679 (notably lower than patient001's
  0.926 — the LV boundary itself is registered less precisely for this
  patient, a real caveat). Displacement-convention cross-check passed
  (99.1% agreement vs. Phase I's own Dice-validated warp). True
  contraction signal is now genuinely meaningful: inner mean-radius
  spans 6.02mm -> 4.93mm (span=1.09mm) across the cycle — 11x larger
  than patient001's 0.10mm span, addressing exactly the weak-signal gap
  run -50 flagged.
  - **Inner (LV) boundary: RMSE=0.8014mm** (per-phase 0.13-1.13mm) —
    noisier than patient001's 0.23mm, but CRUCIALLY smaller than the
    1.09mm true signal span, so this is the first real-motion test in
    this thread that demonstrates actual contraction-TRACKING (not just
    survival of non-uniform motion without collapse, run -50's honest
    caveat). Both within the ~1.5-4.5mm registration floor.
  - **Outer (epicardium) boundary: RMSE=2.2940mm** (per-phase
    1.97-2.49mm) — consistent with, and only slightly better than, the
    static single-frame result (2.43mm, run -51) and REMARKABLY STABLE
    across all 8 phases (narrow 1.97-2.49mm range) rather than varying
    with contraction phase. This stability is itself confirmatory
    evidence for the run -51/-52/-53 diagnosis: a structural,
    geometry-dependent bias (large/flat reflector, near-monostatic-only
    signal) should be roughly constant regardless of the true radius at
    each phase, unlike a signal-strength-dependent error which would
    vary with contraction. Visually confirmed in
    `results/figures/phase3_mri_motion_cycle_reconstruction_patient023.png`:
    the fitted (green) outer contour sits consistently and visibly
    inside the true (cyan) outer contour at every phase, a systematic
    inward bias, not random noise.
- Physical sanity checked? by whom?: Claude — quantitative (known real
  registration-derived radii) + visual (8-frame filmstrip) + consistency
  check (outer error's phase-independence itself corroborates the
  already-diagnosed structural cause rather than contradicting it).
- Gate passed? (Y/N): N/A — escalation/validation test.
- Observations / surprises: choosing a patient for a stronger
  contraction SIGNAL (per user's own request) incidentally also
  surfaced a patient with a proportionally thicker WALL, which triggered
  a structural limitation invisible in patient001's thinner-walled
  anatomy — the two properties (contraction magnitude, wall
  thickness/outer radius) are independent patient characteristics, and
  a stress-test cohort should expect to encounter both kinds of
  variation separately, not assume a single "harder" patient captures
  every dimension of difficulty at once.
- Next action: both the smooth-eccentric-ring synthetic escalation and
  the two real-MRI escalations (shape-only, patient001; shape+motion,
  patient001 and patient023) are now complete for this thread. Open,
  not-yet-attempted follow-ons carried forward: (1) a structurally
  different fix for large/flat outer boundaries (more independent probe
  angles, not a coefficient tweak); (2) a more dramatically asymmetric
  (HCM/DCM) patient for boundary SHAPE irregularity specifically
  (distinct from contraction magnitude or wall thickness); (3)
  independently registering ED to real intermediate cine frames instead
  of fractionally interpolating the single ED->ES field; (4) the
  multi-compartment/multi-chamber full-heart escalation, explicitly
  deferred earlier in this thread pending closure of the single-ring
  work.

### Run 2026-07-08-55 — CORRECTION to runs -48/-50 (patient001): search-grid widening (not the calibration) silently changed the previously-logged numbers; isolated and confirmed via direct A/B test
- Phase: 3 (correction/erratum). Per user: "calibration still broke
  somehow. diagnose" — after seeing patient001 numbers shift, prompted
  direct diagnosis rather than assuming which change caused it.
- **Diagnosis method**: re-ran patient001's static reconstruction after
  the run -51 `build_search_grid()` addition and got DIFFERENT numbers
  than run -48's logged result (inner scale 1.015->0.995, err
  0.09->0.03mm; outer scale 1.030->1.035, err 0.22->0.26mm) despite no
  intentional change to patient001's own pipeline. Two candidate causes
  existed simultaneously (both added between run -48 and now): the run
  -53 calibration update (2->3 points, new `np.interp`-based
  `_linear_weight`) and the run -51 dynamic search-grid widening.
  Isolated by directly calling `fit_scale_curvature_weighted` with the
  ORIGINAL default grid (`img_rows`/`img_cols`, +/-90 cells) but the
  NEW 3-point calibration: **result exactly reproduced run -48's
  original numbers (1.015/1.030)**. This proves the calibration change
  had ZERO effect on patient001 (confirms the earlier reasoning: its
  boundaries, 60/74-79 cells, sit close enough to the original 41-71
  calibration range that the new R=88 point changes nothing measurable)
  — **the search-grid widening is the entire cause**.
- **Root cause, precisely**: patient001's own outer contour has
  max radius 79.4 cells (not just its mean, 74.0) — `needed_extent =
  max(ext_r_out.max(), ext_r_in.max()) * SCALE_GRID.max() + 15 = 119.0`
  cells, which ALREADY exceeds the default grid's +/-90-cell coverage,
  independent of patient023 entirely. This means **run -48's original
  patient001 result was silently computed with a too-small search
  grid** — `RegularGridInterpolator`'s `bounds_error=False,
  fill_value=0.0` zero-filled the outer boundary's most eccentric
  angular points at the largest candidate scales (near 1.31x) without
  raising any error, undetected until the run -51 robustness fix
  (added for patient023's much larger anatomy) incidentally also
  triggered for patient001 and exposed it. The new numbers
  (0.995/0.03mm inner, 1.035/0.26mm outer) are the CORRECTED,
  more-complete-search-grid result — both still small, sub-mm errors,
  not a qualitative change in the finding, just a quantitative
  correction.
- **Second, related bug found and fixed while diagnosing this**:
  `phase3_mri_motion_cycle_reconstruction.py`'s copy of the same
  grid-sizing check used the MEAN contour radius
  (`ed_mean_r_out`/`ed_mean_r_in`) instead of the MAX, inconsistent
  with the static script's (correct) convention — a real bug, though
  by coincidence it did not cause additional clipping for either
  patient001 or patient023 in the runs already logged (checked
  directly: in both cases mean-based-extent-plus-margin still exceeded
  the actual max-radius-scaled requirement). Fixed to also use
  `.max()`, for consistency and to avoid relying on that coincidence
  for future patients.
- **Re-ran both of patient001's real-MRI results with the corrected
  (max-based) wide grid**: static reconstruction now inner=0.995
  (err=0.03mm), outer=1.035 (err=0.26mm) — supersedes run -48. Motion
  cycle now inner RMSE=0.1996mm (was 0.2255mm), outer RMSE=0.4028mm
  (was 0.4298mm) — supersedes run -50. Both changes are small
  improvements, not regressions, and do not change either run's
  reported conclusion (both boundaries recover with sub-mm/registration-
  floor accuracy; patient001's contraction signal is still weak, per
  run -50's original honest caveat).
- Physical sanity checked? by whom?: Claude — direct A/B isolation
  (same grid + new calibration reproduces old numbers exactly; same
  calibration + new grid reproduces the new numbers) before attributing
  cause, exactly the discipline this project has used throughout
  (compare against a controlled baseline before trusting an
  explanation).
- Gate passed? (Y/N): N/A — correction/erratum.
- Observations / surprises: a robustness fix added for one patient's
  more extreme anatomy (patient023) silently improved (not broke)
  another already-"validated" patient's (patient001) result, by
  surfacing a pre-existing, previously-undetected search-grid clipping
  issue that had been present since run -46/-47's original
  infrastructure — a reminder that "already validated" numbers can
  still rest on an unexamined implementation assumption (grid coverage
  margin was never explicitly checked against the FULL 0.7-1.31 scale
  sweep range for any patient before run -51), and that user skepticism
  ("still broke somehow") is worth investigating with a controlled A/B
  test rather than a plausible-sounding explanation.
- Next action: none required — this is a closed correction. If any
  further real-anatomy escalation is done, the grid-sizing check is now
  consistent and max-based in both reconstruction scripts, so this
  specific class of silent clipping should not recur.

### Run 2026-07-08-56 — ISOLATED 8-probe test: real, partial improvement to patient023's outer boundary, confirming probe count is a genuine lever (not yet sufficient alone)
- Phase: 3 (structural-fix attempt, per user: "so an 8-probe parallel
  test? make sure you isolate the case, clone codes before editing and
  leave the current code base intact"). `src/phase3_mri_8probe_test.py`
  — FULLY SELF-CONTAINED: defines its own probe geometry, domain,
  medium-building, capture, and curvature-weight logic from scratch
  (cloned/adapted from the validated 4-probe infrastructure, not
  imported from it); only imports pure, probe-count-independent helper
  functions (`_polar_resample`, `r_at_theta`, `build_search_grid`) that
  are unaffected by this test. No existing file modified — the
  validated 4-probe pipeline is untouched.
- Design: 8 probes at 45-degree spacing (0,45,90,...,315) instead of 4
  at 90-degree spacing, same PROBE_DIST_CELLS=120 (standoff already
  proven irrelevant, run -52 — probe COUNT is the only variable changed
  here). Geometry verified before running: diagonal-probe positions and
  src/rcv tangential-offset convention checked to exactly reproduce the
  original 4-probe positions/offsets at the shared 0/90/180/270-degree
  angles.
- **Weight model generalization (a documented approximation, not a new
  measurement)**: `pair_weight_at_R` was only ever calibrated at 3
  baseline angles (0, 90, 180 degrees — runs -44/-53). The new
  45/135-degree pairs this layout introduces are NOT independently
  measured; their weight is a LINEAR INTERPOLATION (in baseline angle,
  between the measured radius-dependent weight functions at the nearest
  anchors) — e.g. at R=88, monostatic=1.0, 45deg-interpolated=0.5,
  90/135/180deg=~0 (all measured/confirmed near-zero at this radius).
  Flagged clearly so this is not later mistaken for a real calibration
  result.
- Result: **inner fitted scale=0.970 (err=0.18mm), outer fitted
  scale=0.755 (err=2.16mm)**, vs. the 4-probe baseline's inner
  err=0.45mm, outer err=2.43mm — both improved, outer by ~11%. More
  informative than the raw number: the outer score curve
  (`results/figures/phase3_mri_8probe_test_patient023.png`) now shows a
  GENUINE SECONDARY PEAK right at the true scale=1.0 (height 0.975,
  normalized) — this peak was essentially invisible (no local maximum
  at all near 1.0) in every 4-probe test of patient023 (runs -51-54).
  It just narrowly loses to the guard-band-edge peak (height 1.0) rather
  than winning outright.
- **Interpretation**: this is real, not placebo, evidence that adding
  more monostatic-type probe angles genuinely recovers usable signal
  for a large/flat reflector — confirms probe COUNT is a real lever for
  this structural limitation, not just the interpolated weight
  assumption doing all the work (a pure assumption artifact would be
  unlikely to produce a correctly-located, competitive secondary peak
  this close to the true answer). However, 8 probes and this
  interpolated weight model are NOT sufficient to make that peak win
  outright — the fix is directionally confirmed, not completed.
- Physical sanity checked? by whom?: Claude — direct comparison against
  the already-established 4-probe baseline (same patient, same
  contour, same underlying method) as the control, and visual
  inspection of the score curve shape (not just the final argmax
  number) before drawing the interpretation above.
- Gate passed? (Y/N): N/A — isolated exploratory test, deliberately
  kept separate from the validated pipeline per user's explicit
  instruction.
- Next action: two clear, not-yet-attempted follow-ons if this is
  pursued further: (1) measure the ACTUAL 45-degree (and 135-degree)
  baseline amplitude ratio directly (extending run -44/-53's isolated
  single-boundary calibration method to these new baseline angles,
  replacing the current interpolation assumption with a real
  measurement); (2) try more probes still (e.g. 12 or 16) now that 8 is
  confirmed to help, to see whether the true-scale peak eventually
  overtakes the guard-band-edge peak outright. Neither attempted here —
  this run's purpose was answering "is probe count a real lever at
  all", which it confirms.

### Run 2026-07-08-57 — LOCAL-MAXIMUM-ONLY selection: near-complete fix for patient023's outer boundary (2.43mm -> 0.04mm), but reveals a real caveat in the homogeneous-medium control
- Phase: 3 (readout-rule fix, per user diagnosis discussion: "visually
  the fitter falls onto the tail which is an inaccurate approximate...
  is it appropriate to mitigate or dampen the tail"). Diagnosed the
  tail as leakage from the guard-band-excluded region bleeding into the
  adjacent allowed scales (the backprojected field varies smoothly with
  candidate scale, so scores just outside a hard exclusion cutoff stay
  elevated) — explicitly REJECTED "dampening" the tail as unsafe (same
  category of risk as the earlier-rejected constant-wall-thickness fix:
  shaping the algorithm toward the expected answer rather than fixing
  the mechanism). Implemented the safer alternative instead: require
  the winning candidate to be a genuine LOCAL MAXIMUM (rises then falls
  on both sides), not just the highest score in the allowed range —
  disqualifies a monotonic climb into a hard cutoff by construction,
  without presupposing where the true answer is.
- Implementation: `select_best_local_peak()` added to the SAME isolated
  `phase3_mri_8probe_test.py` from run -56 (still no existing/shared
  file touched, per the user's standing "leave the codebase intact"
  instruction) — splits the (possibly discontinuous, guard-band-gapped)
  scale grid into contiguous segments, runs `scipy.signal.find_peaks`
  on each segment separately (so a segment's own edge, e.g. immediately
  next to the guard-band gap, is never mistaken for an interior peak,
  matching what `find_peaks` already does at a true array boundary),
  and picks the highest-scoring genuine peak across all segments.
  Verified on synthetic data mimicking the observed tail shape BEFORE
  running any new simulation: naive argmax picked the synthetic tail
  edge, local-max-only selection correctly picked the synthetic
  interior peak.
- Result: **outer fitted scale=1.005 (true=1.000), error=0.04mm** —
  down from 2.16mm (run -56, same 8 probes, naive argmax) and 2.43mm
  (4-probe baseline) — essentially a complete fix. The outer fit's
  confidence (best genuine peak vs. next-best genuine peak) is
  **infinite** — once the tail is correctly disqualified as "not a
  local max," only ONE real peak remains in the guarded range, and it
  sits almost exactly on the true scale. Inner fit: scale=0.970,
  err=0.18mm, confidence=1.20 (modest — a real secondary bump near
  scale=1.15-1.2 is visible in the score curve,
  `results/figures/phase3_mri_8probe_localmax_test_patient023.png`).
- **Important caveat found while checking the homogeneous-medium
  control (not skipped over)**: the control's INNER fit also reports
  confidence=inf (landing at a plausible-looking scale=0.985) — pure
  noise can also produce exactly one local max by chance, and this
  rule cannot distinguish that from a genuine detection. The control's
  OUTER fit is appropriately low-confidence (1.02). **Conclusion:
  local-maximum-only selection is a real, principled improvement over
  naive argmax (confirmed fixing a genuine artifact, not just moving
  where the fit lands by assumption), but the confidence-ratio metric
  alone is NOT a fully reliable stand-alone safety check against false
  detections on absent signal** — a peak-height-relative-to-some-
  absolute-baseline criterion (not just peak-vs-second-peak) would be
  needed to fully close this gap, not yet implemented.
- Physical sanity checked? by whom?: Claude — synthetic-data unit test
  of the selection logic BEFORE trusting it on real simulated data
  (per this project's standing discipline), plus the homogeneous
  control check specifically flagged and reported rather than only
  reporting the favorable real-data result.
- Gate passed? (Y/N): N/A — isolated readout-rule fix, still deliberately
  separate from the validated pipeline.
- Next action: if this fix is to be ported into the shared, validated
  pipeline (currently only exists in the isolated 8-probe script), it
  should be combined with a real minimum-peak-height/absolute-baseline
  check to close the homogeneous-control gap above — not yet done.
  Also still open from run -56: measuring the real 45/135-degree
  baseline ratio directly instead of interpolating.

### Run 2026-07-08-58 — Fork pushed to GitHub; smoke tests PASS: local-max selection reproduces every already-validated result exactly
- Phase: 3 (pre-porting validation). Per user: "upload a fork to github
  first, and run those smoke tests" — before any decision to merge the
  run -56/-57 8-probe + local-max fix into the official/shared
  pipeline, per the two gaps flagged when the user asked "should i
  path this to the official thing?": (1) confirm local-max selection
  doesn't change patient001's/synthetic-ring's already-validated
  numbers; (2) real 45/135-degree calibration measurement (still open,
  not attempted this run).
- **Fork**: created branch `phase3-8probe-localmax-experiment` off
  `master` (`git checkout -b`), committed all of this session's
  real-MRI-escalation and 8-probe/local-max work (23 files), and pushed
  to `origin` — `master` is untouched. Before committing, added
  `jwave_test/results/mri_irregular_ring_*.npz` and
  `jwave_test/results/mri_motion_cycle_*.npz` to `.gitignore`
  (discovered these new result files, containing rescaled/smoothed
  masks and contours derived directly from real ACDC patient anatomy,
  were NOT covered by the existing `data/`-only exclusion rule — same
  caution as `phase4_pilot_dataset`, caught before staging per
  CLAUDE.md's explicit "double check git status before staging"
  instruction). Also cleaned up stale, unsuffixed duplicate figures
  left over from before patient-ID parameterization (some superseded
  by run -55's corrected numbers, renamed/removed as appropriate,
  no data lost — the underlying reruns already regenerated corrected
  patient-ID-suffixed versions).
- **Smoke test** (`src/phase3_smoke_test_localmax_on_validated.py`,
  isolated — duplicates `select_best_local_peak` locally rather than
  importing from the experimental branch's file, and calls the
  EXISTING, UNMODIFIED `fit_scale_curvature_weighted` /
  `fit_circle_radius_curvature_weighted` to get the same score curves
  used in the already-logged runs): applied local-max-only selection
  to patient001's real-shape static reconstruction (default grid, run
  -48's original configuration) and the synthetic ring phantom's ED +
  ES-adjacent frames (run -45's exact two test frames). **Result: EXACT
  MATCH in every case** — patient001 inner=1.015/0.09mm,
  outer=1.030/0.22mm (identical to argmax); synthetic ring ED
  inner/outer errors 0.10mm/0.00mm (identical); ES-adjacent
  inner/outer errors 0.125mm/0.125mm (identical). Local-max-only
  selection changed NOTHING in any of these three already-validated
  cases — it only diverges from naive argmax where a genuine tail
  artifact exists (patient023's outer boundary, run -57).
- Physical sanity checked? by whom?: Claude — direct numeric comparison
  against already-logged values (run -45, run -48) before drawing any
  conclusion, per this project's standing discipline.
- Gate passed? (Y/N): N/A — pre-porting validation check.
- Observations / surprises: none — this is the reassuring, expected
  outcome (a principled fix that only changes behavior where a real
  problem exists should reproduce identical results everywhere else),
  but it was verified rather than assumed, consistent with this
  project's practice throughout.
- Next action: the local-max-selection fix has now cleared BOTH
  pre-porting gaps identified when first proposed: gap 1 (no
  regression on already-validated cases) is CLOSED by this run; gap 2
  (real 45/135-degree calibration measurement, currently an
  interpolation assumption) remains open. Recommend closing gap 2
  before merging the 8-probe geometry itself into the official
  pipeline (the local-max selector alone, independent of probe count,
  could reasonably be ported now given this run's result — that is a
  separate, smaller decision from adopting 8 probes as the new
  official geometry).

### Run 2026-07-08-59 — Local-max selection PATCHED into the official 4-probe pipeline; patient023 re-verified: real but MIXED result, not the 8-probe fix's clean win
- Phase: 3 (official patch). Per user: "if things look good, patch it
  to the current pipleline, log properly" (after run -58's smoke test
  passed). Scope deliberately limited to the local-max SELECTOR only —
  NOT the 8-probe geometry, which still relies on the unvalidated
  45/135-degree interpolation weight (flagged as the remaining open gap
  in run -58).
- **Patched files**: `phase3_ring_curvature_weighted_fit.py` (added
  `select_best_local_peak`, replaced `fit_circle_radius_curvature_weighted`'s
  naive-argmax + separate confidence calc with it — also fixes a
  latent, previously-unnoticed issue in the OLD confidence calculation,
  which ran `find_peaks` directly on a possibly-discontinuous guarded
  R_grid without segment-splitting, same class of bug as the tail
  artifact itself); `phase3_mri_irregular_ring_reconstruction.py`
  (`fit_scale_curvature_weighted` now returns
  `(scale, scores, is_genuine_peak, confidence)` instead of
  `(scale, scores)`, imports `select_best_local_peak` from the shared
  module rather than duplicating it); `phase3_mri_motion_cycle_reconstruction.py`
  (updated call sites for the new 4-tuple return). All call sites
  across all three files updated and confirmed importing/running
  without error before any simulation was trusted.
- **Re-verification, patient001 (must reproduce run -55 exactly)**:
  static reconstruction — inner scale=0.995 (err=0.03mm), outer
  scale=1.035 (err=0.26mm). **Exact match to run -55.** Confirms the
  patch is safe on the case it wasn't designed for.
- **Re-verification, patient023 (the actual test)** — **result is real
  but MIXED, NOT the dramatic 8-probe fix**:
  - Static: outer scale=0.745, err=2.25mm (was 2.43mm) — a modest ~7%
    improvement, not the 8-probe run's 0.04mm. Visually
    (`results/figures/phase3_mri_irregular_ring_reconstruction_patient023.png`),
    the fit still lands on a small local max right at the guard-band
    edge (0.745), NOT on the more genuine-looking bump near
    scale=0.94 (score=0.87) — with only 4 probes, that near-true peak
    still isn't the STRONGEST local max, so local-max-only selection
    alone cannot promote it to the winner. Confirms directly (not by
    inference) that the 8-probe run's big improvement needed the EXTRA
    PROBES' redundancy, not just the corrected selection rule.
  - Motion cycle (`results/figures/phase3_mri_motion_cycle_reconstruction_patient023.png`):
    outer RMSE improved 2.2940mm -> 1.9053mm, but HIGHLY INCONSISTENTLY
    across phases (0.49mm at phase 2/7, ~2.0-2.3mm at every other
    phase) — unlike the stable, uniform-across-phases signature that
    indicated a structural cause in run -54. **Inner RMSE got slightly
    WORSE: 0.8014mm -> 0.8354mm**, with real regressions at phases
    2/3/6/7 (jumping to 1.13-1.14mm from ~0.09-0.34mm previously) — the
    local-max rule occasionally picks a worse genuine peak than argmax
    did at those specific phases' score curves. **This is a real,
    reported mixed result, not spun as a win.**
- Physical sanity checked? by whom?: Claude — direct numeric comparison
  against run -55 (patient001, must match) and runs -51/-54 (patient023,
  4-probe baseline) before drawing conclusions; visual inspection of
  both regenerated figures.
- Gate passed? (Y/N): N/A — pipeline patch + validation.
- Observations / surprises: the inconsistent, phase-dependent behavior
  for patient023's motion cycle (helps a lot at some phases, hurts
  slightly at others) is itself informative: it shows local-max-only
  selection is sensitive to the SPECIFIC shape of each frame's score
  curve, not a uniform improvement — reinforcing that the real fix for
  patient023's structural limitation is more probe angles (run -56),
  not just a better selection rule on the same 4-probe information.
- Next action: the local-max selector is now the official pipeline's
  default (safe, matches every already-validated result, and gives a
  small real improvement where it can). The 8-probe geometry remains
  on the separate `phase3-8probe-localmax-experiment` branch, not
  merged — still blocked on the real 45/135-degree calibration
  measurement (run -56's flagged gap) before being considered for
  official adoption.

