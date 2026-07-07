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
