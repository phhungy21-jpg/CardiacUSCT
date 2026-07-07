# Lab Notebook — Acoustic-Simulation Phase (k-Wave / jWave)

**Reference Protocol:** ../acoustic_simulation_phase_protocol.md
**Version:** 0.1
**Start Date:** 2026-07-07
**Builds on:** ../pilot/ (Phase I pilot, completed — see pilot/LOG.md)

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
- [ ] Phase 1 — Environment, tooling, and reproducibility parity
- [ ] Phase 2 — Acoustic model definition (highest-risk phase)
- [ ] Phase 3 — Toy proof-of-concept (simulate → recover on simple phantom)
- [ ] Phase 4 — Scale to cardiac anatomy
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
