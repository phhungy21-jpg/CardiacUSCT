# LOG — jwave_ring_tomography/ (ring/water-bath ultrasound tomography)

Running lab notebook for this project. Each entry follows the Appendix
C template from `../ring_tomography_phase_protocol.md` (phase, seed,
config, dataset/split, result, gate pass/fail, observations, next
action). This is a NEW project's log, started fresh on 2026-07-09 — it
does not continue `../jwave_test/LOG.md`'s run numbering. See that
file's "PROJECT CLOSURE" entry for the full history and reasoning that
led here, and `MANIFEST.md` in this directory for what was reused vs.
deliberately left behind.

### Run 2026-07-09-01 — Project created; access geometry decided; scaffolding only, no simulation yet
- Phase: 0 (setup/scoping). Created in direct response to
  `jwave_test/`'s PROJECT CLOSURE finding and a parallel ChatGPT-
  consulted exploration of denser acquisition geometries (two-opposite-
  probe scanning, rotating/ring-array transmission+reflection
  tomography). Per explicit user request ("new project folder for
  this... we can use heart constructed data and tissue information from
  old projects").
- Decision made this run (via AskUserQuestion, user selected the
  recommended option): **water-bath / full-surround acquisition
  geometry** (Butterfly/Midjourney-style whole-body scanner model),
  not the clinically standard anterior-arc-only handheld transthoracic
  probe. This resolves a real physical conflict: `jwave_test`'s tissue
  config explicitly carried over Phase I's "no posterior acoustic
  window, anterior access only" rationale, which a literal 360° ring
  around a standard chest-probe setup would have violated without
  addressing it. The honest cost of this choice, to carry into any
  future writeup: this is now a research/investigational acquisition
  mode, not standard point-of-care cardiac ultrasound.
- Work done: `ring_tomography_phase_protocol.md` written (root),
  adapting `acoustic_simulation_phase_protocol.md`'s phase-gate
  structure for the 3 things this project changes at once (acquisition
  geometry, acquisition mode [transmission+reflection, not reflection-
  only], and explicit motion-during-acquisition modeling from the
  start — the one lesson most directly acted on from `jwave_test`'s
  closure). `src/phase2_config.py` written: clones `jwave_test`'s cited
  tissue properties (BLOOD/MYOCARDIUM/CHEST_WALL_PROXY) unchanged, adds
  a NEW water-bath coupling-medium property (proposed citation: Duck
  1990, NOT yet collaborator-confirmed), and leaves `N_ELEMENTS`/
  `CAPTURE_MODE` as explicit placeholders pending Phase 0.1's
  compute-budget and motion-handling decisions. `MANIFEST.md` written,
  documenting what's reused (tissue properties, the Phase I/`jwave_test`
  real-anatomy pathway via `pilot/data/processed/`) vs. deliberately
  NOT reused (the closed sparse-probe backprojection/blind-discovery
  code stays in `jwave_test/`, frozen).
- Physical sanity checked? by whom?: N/A — no simulation run yet, pure
  scaffolding.
- Gate passed? (Y/N): N/A. Phase 0 not yet complete — the
  simultaneous-vs-sequential capture-mode decision (protocol 0.1) is
  still open.
- Next action: decide simultaneous-ring-capture vs. sequential/rotating
  acquisition mode (Phase 0.1) BEFORE writing any Phase 1 scout code;
  get a real per-transmit compute estimate at the new, larger
  (full-torso, not one anterior arc) domain size before committing to
  an element count; then proceed to Phase 1 scout smoke tests (point
  source in water, two-tissue water/chest-wall reflection, full-ring
  source/receive geometry sanity check), mirroring `jwave/`'s original
  Phase 1 approach.

### Run 2026-07-09-02 — Motion-capture mode decided: SEQUENTIAL
- Phase: 0 (setup/scoping). Per explicit user choice: "lets go with
  sequential for data clarity." Resolves the one open Phase 0.1 decision
  flagged in run -01.
- Decision: each ring element fires and records in its own turn (not a
  simultaneous multi-element snapshot). Chosen for data clarity — every
  measurement is cleanly attributable to one transmit angle at one
  moment, unlike a simultaneous capture where many elements' returns are
  entangled in time. The explicit, accepted tradeoff: this reintroduces
  a CT-style motion-during-acquisition problem that `jwave_test`'s
  frozen-scene sparse setup never had to face, since a full sequential
  scan takes real time and the heart moves during it.
- Consequence, now a hard Phase 2 requirement rather than optional:
  the forward model needs an explicit scan-time -> cardiac-phase
  mapping (a cardiac cycle length and total scan duration), so each
  element's capture samples the medium at ITS OWN moment in the cardiac
  cycle, not one static scene reused for every element. Added
  `CAPTURE_MODE="sequential"` (decided) and `CARDIAC_CYCLE_S`/
  `SCAN_DURATION_S` (TBD placeholders, depend on `N_ELEMENTS`/timing
  still pending a compute-budget estimate) to `src/phase2_config.py`.
  Updated `ring_tomography_phase_protocol.md`'s Phase 0.1/2.1 checkboxes
  and `MANIFEST.md` accordingly.
- Physical sanity checked? by whom?: N/A — decision/scoping only, no
  simulation run yet.
- Gate passed? (Y/N): N/A. Phase 0 now complete (both open decisions —
  access geometry, capture mode — are resolved).
- Next action: get a real per-transmit compute estimate at the ring/
  water-bath domain size, decide `N_ELEMENTS` against that budget, then
  set `CARDIAC_CYCLE_S`/`SCAN_DURATION_S` from the resulting scan
  timing before writing Phase 1's scout smoke tests. Per updated
  standing instruction, this and future work in this project is NOT
  committed/pushed unless explicitly requested.

### Run 2026-07-09-03 — Phase 1 scout: rotating-probe transmission tomography, STATIC centered phantom — geometry/timing PASSES its own analytic check (4.6% match) after catching a sign bug in the prediction formula itself
- Phase: 1 (scout). Per user: "test it first on a static model. try it
  on a 2d cross section first. a rotating probe transmitter beaming
  through the tissue and project to the opposite side, record it, move
  the probe a bit, project, record. until the probe completes a full
  Perimeter." Built `src/phase1_rotating_transmission_scout.py` and
  `src/labels.py` (cloned from `jwave_test`, own Gate 2 banner text).
- Design: STATIC (no motion yet), 2D, water-bath background, a single
  CENTERED circular myocardium-like disk phantom (radius 60 cells).
  SEQUENTIAL acquisition (per run -02's decision): one transmitter/
  receiver pair at a time, receiver diametrically opposite at the same
  probe radius (120 cells), both rotating together in 10-degree steps
  around a full 360-degree perimeter (36 angles). Captures TRANSMISSION
  only (through-tissue arrival), not reflection — matches exactly what
  was asked. Two independent, analytically-predictable checks set up
  BEFORE trusting any result: (1) a water-only control should show
  ~flat arrival time across all angles (pure rotational symmetry); (2)
  the phantom's arrival time should ALSO be flat (every diametrically-
  opposite ray passes through the same tissue chord, the phantom's full
  diameter, by construction), shifted from the water-only baseline by
  an analytically computable amount. Attenuation NOT modeled in this
  test (flagged explicitly, matches `jwave_test`'s own early-phase gap
  before its later attenuation-solver work) — amplitude numbers here
  are informational only, not a validated attenuation map. Compute:
  72 total forward simulations (36 angles x 2 media), estimated
  ~15-20 minutes based on comparable-grid-size runs in `jwave_test`;
  actual run completed within that estimate.
- **Result**: water-only control: mean=16.3816us, std=13.16ns across 36
  angles (~0.08% relative spread — flat, as required by symmetry; the
  small residual is attributable to tx/rx positions rounding to integer
  grid cells at different angles, not a geometry bug). Phantom: mean=
  16.1140us, std=21.03ns (also flat, slightly more jitter than the
  water-only control for the same reason plus the tissue boundary's own
  grid quantization). Measured excess delay (phantom - water-only) =
  **-267.54ns** (phantom arrives EARLIER, correctly signed: myocardium's
  cited sound speed, 1576 m/s, is faster than the water bath's, 1520
  m/s). First comparison against the analytic straight-ray prediction
  FAILED (195% relative error) — but the failure was in the ANALYTIC
  FORMULA's own sign convention (`chord*(1/c_water - 1/c_tissue)`,
  backwards), not the simulation: the correct formula is
  `chord*(1/c_tissue - 1/c_water)` = **-280.52ns**, which the measured
  value matches to **4.6% relative error** — a good, verifiable pass
  for a first scout test at this grid resolution. Caught and fixed
  before reporting (same discipline as every analytic-vs-simulated
  comparison throughout this project's history): re-derived the sign
  from first principles (mixed_time - water_only_time =
  chord/c_tissue - chord/c_water) rather than assuming either number
  was right, confirmed the SIGN of the measured value already matched
  physical intuition (faster tissue -> earlier arrival) before
  concluding the formula, not the simulation, was wrong.
- Physical sanity checked? by whom?: Claude — set up both control
  checks (water-only flatness, phantom flatness) BEFORE running
  anything, per this project's established pre-registration discipline;
  when the first comparison failed, re-derived the analytic prediction
  from first principles rather than adjusting the simulation to match a
  possibly-wrong number.
- Gate passed? (Y/N): this is Phase 1 (scout), not a formal ⛔ gate, but
  its purpose (does the ring/water-bath geometry produce physically
  sane, quantitatively verifiable results before building anything more
  complex on top) is satisfied.
- Next action: this validates the STATIC, no-motion, centered-phantom
  case only. Natural next scout steps: (a) an OFF-CENTER or non-
  circular phantom, to confirm the transmission-timing signal actually
  carries shape/position information (a centered circular phantom is a
  necessary but not sufficient first test — every angle gives the same
  answer by construction, so this test alone can't show the method
  distinguishes angles from each other); (b) a two-tissue case (blood +
  myocardium, matching real LV+wall anatomy) to check multi-boundary
  transmission timing; (c) only once those pass, move to Phase 2's
  hard requirement — explicit motion-during-acquisition modeling, now
  that sequential capture's static-case geometry is confirmed sound.
  Not committed/pushed (standing instruction).

### Run 2026-07-09-04 — Full multistatic transmission-tomography reconstruction: circular sequential scan DOES carry real spatial information (visual + quantitative confirmation), with an expected unfiltered-backprojection artifact
- Phase: 1 (scout). Per user: "why not visualise the blind
  reconstruction from the circular scan and overlap it with the cross
  section so i can see if circular sequential scan works?"
- **Caught before building anything**: checked run -03's actual
  geometry first — confirmed numerically that every diametrically-
  opposite tx/rx pair's straight-line path passes through the EXACT
  SAME center point (150,150) at every single angle. That design has
  only ONE degree of freedom per transmit angle (a single line integral
  through the center), which is mathematically insufficient to
  reconstruct any 2D image regardless of phantom shape — explains why
  run -03's phantom curve was flat as a necessary consequence of the
  geometry, not a success signal. Flagged this to the user before
  building the requested visualization, rather than producing a
  visualization from data that couldn't support it.
- **Fix, at no extra simulation cost**: jWave computes the full
  pressure field for each transmit event, so instead of sampling only
  the single opposite receiver, sampled EVERY other probe angle's
  position from that same field (`phase1_transmission_tomography_reconstruction.py`).
  Still only one transmitter fires at a time (sequential, unchanged from
  run -02's decision) — multiple receivers per transmit is standard
  practice (same pattern as `jwave_test`'s `capture_all_pairs`), not a
  reversal of that decision. Excluded pairs within 20 degrees of each
  other (near-field/direct-coupling, not a clean through-tissue
  transmission). Compute: same 72 forward simulations as run -03 (36
  angles x 2 media) — estimate given upfront (~15-20 min, per new
  standing instruction), actual run completed within it. 1,188 total
  ray paths captured.
- Reconstruction: simple UNFILTERED straight-ray backprojection —
  smear each pair's excess delay (phantom arrival - water-only arrival,
  same ray) uniformly along its path in a 150x150 image grid, no
  filtering/normalization for ray density (same "naive first" spirit as
  this whole project's very first backprojection sanity check).
- **Result: the reconstructed image's bright/fast-tissue-like region
  visually and quantitatively matches the true phantom boundary** —
  a clear central blob whose size and position track the true 60-cell-
  radius disk closely (see
  `results/figures/phase1_transmission_tomography_reconstruction.png`).
  Per-ray excess delay ranged -276.3 to +19.7ns (mean -72.8ns) — rays
  crossing the phantom show large negative delay (faster, as expected,
  myocardium > water sound speed) and rays missing it cluster near 0ns,
  consistent with the underlying physics. **This is a materially
  different, positive result compared to run -03**: real spatial
  information exists in this scan design, not just a single aggregate
  number.
- **Artifact, named not hidden**: the image shows a 36-pointed star/
  gear pattern radiating from the probe ring — the well-known signature
  of UNFILTERED backprojection with a small number of discrete view
  angles (the same star artifact that motivated the ramp filter in
  X-ray CT's filtered backprojection). Not a bug in this geometry/data;
  the expected cost of the simplest possible reconstruction method.
- Physical sanity checked? by whom?: Claude — verified the single-ray-
  per-angle geometry's fundamental information-content limitation
  analytically BEFORE building any visualization, rather than producing
  a possibly-misleading image from insufficient data; named the star
  artifact explicitly as a known method limitation rather than an
  unexplained residual.
- Gate passed? (Y/N): N/A — scout-phase visual/quantitative check, not
  a formal gate.
- Next action: (a) filtered backprojection or a proper iterative
  inversion (ART/SIRT) to remove the star artifact and get a cleaner,
  more quantitatively trustworthy image, before trusting absolute
  boundary measurements from this method; (b) the previously-planned
  off-center/non-circular phantom test, now well-motivated since this
  design is confirmed to carry real spatial information worth testing
  against a harder shape; (c) two-tissue (blood+myocardium) case. Not
  committed/pushed (standing instruction).

### Run 2026-07-09-05 — Real MRI anatomy (patient001): reconstruction tracks genuine irregular shape, not just a circle
- Phase: 1 (scout). Per user, correcting a real gap in run -04: "why is
  it just a circle? i thought you put a crossection of the mri
  reconstructed heart?" — accurate catch: run -04 used a synthetic
  circular disk (deliberately, as the simplest first sanity check,
  matching how `jwave_test` itself escalated circle -> eccentric ring
  -> real MRI), but nothing in this project had actually loaded real
  anatomy yet, despite `MANIFEST.md` documenting the intent to reuse it.
- Design: `phase1_real_mri_transmission_tomography.py` loads patient001's
  already-prepped real contour
  (`jwave_test/results/mri_irregular_ring_patient001_slice4.npz` — the
  SAME file `jwave_test`'s own real-anatomy runs, e.g. runs -47/-48,
  used), places it at this project's water-bath domain center via the
  same offset/canvas convention as those earlier scripts. Kept as a
  SINGLE-TISSUE test (the whole outer/ring_mask interior — myocardium +
  LV cavity combined — treated as one MYOCARDIUM-property region, not
  yet split into two boundaries) for a fair, isolated comparison to run
  -04: isolates the effect of REAL, IRREGULAR shape from run -04's
  perfect circle, without also introducing a second boundary at the
  same time. STATIC still (no motion) — only the real shape is reused
  here, not yet real motion. Same full multistatic transmission +
  unfiltered straight-ray backprojection pipeline as run -04, unchanged.
  Real contour's max radius=79.5 cells, safely inside the 120-cell
  probe radius (no overlap warning triggered). Compute: same 72 forward
  simulations as runs -03/-04, ~15-20 min estimate given upfront,
  completed within it.
- **Result: the reconstructed bright/fast-tissue-like region visually
  tracks patient001's actual IRREGULAR boundary, not a clean circle** —
  see `results/figures/phase1_real_mri_transmission_tomography_patient001.png`.
  The reconstruction bulges and narrows roughly where the true contour
  does, a materially stronger confirmation than run -04's circle (which
  couldn't distinguish "correct shape" from "any radially-symmetric
  blob of the right size"). 1,188 ray paths captured; excess delay
  range -355.3 to +39.5ns (mean -120.2ns), larger in magnitude than run
  -04's synthetic circle (-276.3 to +19.7ns) consistent with this
  patient's larger real anatomy (79.5 vs 60 cells radius). Same
  36-point unfiltered-backprojection star artifact as run -04, expected
  and not investigated further here (same known cause).
- Physical sanity checked? by whom?: Claude — reused the exact same
  offset/placement convention already validated in `jwave_test`'s own
  real-anatomy scripts rather than re-deriving it; checked the real
  contour's max radius against the probe radius before trusting the
  simulation (would have flagged a warning if too close, per the script's
  built-in check).
- Gate passed? (Y/N): N/A — scout-phase visual/quantitative check.
- Next action: same as run -04's still-open items (filtered/iterative
  reconstruction to remove the star artifact; two-tissue blood+myocardium
  case), now on a path toward using patient023 as well (the harder,
  strong-contraction patient) once the static single-tissue case is
  fully characterized. Not committed/pushed (standing instruction).

### Run 2026-07-09-06 — SIRT reconstruction: residual converges well but the star artifact PERSISTS — SIRT alone does not fix a sparse-angle problem
- Phase: 1 (scout). Per user: "do them both" (both the star-artifact fix
  and the two-tissue test, run -05's two open items). This entry covers
  the SIRT half. Added `src/tomography_recon.py` (shared reconstruction
  module: `unfiltered_backprojection`, matching runs -04/-05's method
  for direct comparison, and `sirt_reconstruct`, a new Simultaneous
  Iterative Reconstructive Technique solver). Re-ran patient001's
  single-tissue case (identical medium to run -05) via
  `phase1_sirt_reconstruction.py`, this time SAVING the raw per-ray
  excess-delay dataset to `results/patient001_single_tissue_rays.npz`
  so future reconstruction-algorithm experiments don't need to
  resimulate. Compute: same 72 forward simulations as prior runs,
  ~15-20 min estimate given upfront, completed within it (run in
  parallel with run -07 below).
- **Result: SIRT's residual RMS converged cleanly (188.6ns -> 28.9ns
  over 30 iterations) — a real, working iterative fit — but the
  36-point star/gear artifact from runs -04/-05 is STILL clearly
  present in the reconstructed image, visually almost indistinguishable
  in shape from the unfiltered result** (see
  `results/figures/phase1_sirt_reconstruction_patient001.png`). **This
  corrects an implicit overclaim from the prior turn's framing** ("a
  proper iterative solve... would substantially clean this up") — SIRT
  minimizes measurement residual, it does not inherently suppress a
  sparse-angle streak artifact, because the streaks are a genuine
  null-space consequence of having only 36 discrete view angles, not
  noise that a better fit removes. Both the unfiltered sum and SIRT are
  equally consistent with the same under-sampled ray data; neither has
  any reason to prefer a smooth answer over a streaky one without an
  explicit regularizer (e.g. total-variation/smoothness prior) or,
  more fundamentally, more view angles.
- Physical sanity checked? by whom?: Claude — compared SIRT and
  unfiltered images side by side against the same true boundary rather
  than trusting the residual-convergence number alone; caught that a
  well-converged residual does not imply a visually/qualitatively
  improved image before reporting SIRT as a fix.
- Gate passed? (Y/N): N/A — scout-phase method comparison.
- Next action: if the star artifact needs fixing before trusting
  absolute reconstructed values, the real levers are (a) more transmit/
  receive angles (this project's own ring/water-bath design already
  points that way vs. `jwave_test`'s sparse probes) or (b) an explicit
  regularization term (TV-minimization or similar) added to the SIRT
  update, neither yet implemented. Not committed/pushed (standing
  instruction).

### Run 2026-07-09-07 — Two-tissue reconstruction: outer (epicardial) boundary clearly visible, inner (blood/myocardium) boundary correctly predicted and confirmed INVISIBLE — same weak-contrast limitation as jwave/'s very first scout finding, now reproduced under a completely different acquisition method
- Phase: 1 (scout). Per user: "do them both" (this is the two-tissue
  half). Verified before building: `myo_mask`/`lv_mask` in the prepped
  npz are disjoint and their union equals `ring_mask` (myo_mask = wall
  only, lv_mask = cavity only) — confirmed numerically, not assumed.
  Built `phase1_two_tissue_reconstruction.py`: patient001's real
  anatomy with BOTH boundaries present (myocardium wall + LV/blood
  cavity), using the SIRT reconstruction from run -06 (not the
  unfiltered method, since -06 showed SIRT is at least as good a
  starting point). **Prediction stated BEFORE running**: the inner
  (blood/myocardium) boundary should be much harder to see than the
  outer (myocardium/water) boundary — blood (1584 m/s) vs. myocardium
  (1576 m/s) differ by only ~0.5%, myocardium (1576 m/s) vs. water
  (1520 m/s) differ by ~3.6%, roughly 7x more contrast. Compute: same
  72 forward simulations as prior runs, ~15-20 min estimate given
  upfront, completed within it (run in parallel with run -06). Saved
  raw ray dataset to `results/patient001_two_tissue_rays.npz`.
- **Result: prediction confirmed cleanly.** The reconstruction shows a
  clear boundary matching the TRUE OUTER (epicardial) contour closely.
  The radial/horizontal profile through the domain center is smooth
  and monotonic straight through the inner (endocardial) contour's
  location — no step, no plateau, no visible second ring at the
  predicted inner-boundary radius (see
  `results/figures/phase1_two_tissue_reconstruction_patient001.png`).
  **This is a pre-registered confirmation, not a post-hoc
  rationalization** — the prediction was written into the script's
  docstring and printed before the simulation ran. Notably, this
  reproduces `jwave/LOG.md`'s very first scout-phase finding (the
  blood/myocardium weak-contrast result) under a COMPLETELY DIFFERENT
  acquisition method (transmission tomography, not pulse-echo
  reflection) — independent cross-confirmation from a different
  physical mechanism is stronger evidence this is a real tissue-
  property limitation, not an artifact specific to either acquisition
  method.
- Physical sanity checked? by whom?: Claude — stated the quantitative
  prediction (7x contrast ratio) before running, not after seeing the
  result; cross-checked the finding against the unrelated, much
  earlier `jwave/` pulse-echo scout result rather than treating it as a
  new, isolated observation.
- Gate passed? (Y/N): N/A — scout-phase visual/quantitative check.
- Next action: the weak blood/myocardium contrast is now confirmed
  across two independent acquisition methods and two different
  projects — treat it as a standing physical constraint for this
  tissue set, not a fixable artifact, going forward. Combined with run
  -06's finding, both of run -05's open items are now closed for the
  static single-tissue/two-tissue cases; remaining open items: extend
  to patient023 (harder anatomy), and either more view angles or an
  explicit regularizer if the star artifact needs to be resolved before
  trusting absolute reconstructed values. Not committed/pushed
  (standing instruction).

### Run 2026-07-09-08 — CHANNEL 1 (reflection): detects the inner blood/myocardium boundary that transmission (run -07) completely missed
- Phase: 1 (scout). Per user, after stating the multi-channel goal
  ("only using all of that information can we precisely reconstruct
  the tissue" -- reflection + refraction + dispersion + absorption +
  beam divergence): "note those channel down and proceed with the
  first one." Documented the full 5-channel roadmap and tractability
  order in `ring_tomography_phase_protocol.md`'s new "Multi-channel
  information roadmap" section before building anything (channel 1
  reflection first: cheapest, reuses data this project had been
  discarding; channel 2 absorption needs a real physics addition
  (jwave_test's attenuation solver, not yet ported); channel 3
  refraction needs bent-ray/FWI reconstruction, a much bigger lift;
  channel 4 dispersion needs broadband spectral analysis, not started;
  channel 5 beam/echo divergence connects back to `jwave_test` run -44's
  already-characterized curvature-dependent reflection mechanism).
- **Key realization, stated before building**: every prior run (-03
  through -07) excluded near-angle receiver pairs as a "near-field
  artifact." That exclusion discarded real reflection data — a
  near-side receiver has no straight transmission path through tissue,
  so anything it picks up beyond the direct src-rcv coupling IS
  reflected/scattered energy.
- Design: `phase1_reflection_channel_scout.py` — classic MONOSTATIC-
  style pitch-catch A-scan (closely-spaced src/rcv pair, `jwave_test`'s
  proven pulse-echo convention: direct-arrival exclusion, envelope
  detection) at each of 36 ring positions, on patient001's two-tissue
  phantom (same medium as run -07). Predicted BOTH boundaries'
  reflection arrival times geometrically (water leg + tissue leg, each
  at its own tissue's sound speed) before running, then searched a
  narrow window around each prediction for the actual peak — testing
  whether reflection detects the inner boundary transmission could not,
  not assuming it either way. Compute: 72 forward simulations (36
  angles x 2 media), ~15-20 min estimate given upfront, completed
  within it.
- **Result: YES — reflection detects the inner boundary at 25/36 (69%)
  angles, using a conservative relative threshold, and the signal is
  actually nonzero at every single angle** (see
  `results/figures/phase1_reflection_channel_scout.png`) — a dramatic
  contrast to run -07's transmission result, which showed a perfectly
  smooth, monotonic profile with NO detectable step at the inner
  boundary at all. The representative A-scan visually shows a clear
  secondary bump at the predicted inner-reflection time, well above the
  water-only control's flat noise floor. Outer/inner mean excess
  amplitude ratio: 0.001448 / 0.0003404 -> inner is ~24% of outer's
  magnitude, notably HIGHER than the ~14% (0.5%/3.6%) sound-speed-
  contrast ratio would suggest — consistent with reflection coefficient
  being a genuinely different physical quantity (impedance mismatch,
  not sound speed directly) rather than the same weak signal read a
  different way.
- **Honest caveats, not glossed over**: (1) the search window (+/-0.4us)
  is wide enough that some measured "inner" signal could include
  spillover from nearby reverberation/multipath, not a perfectly
  isolated single-boundary measurement — the headline finding (real,
  detectable signal exists) is solid, but the exact 0.235 ratio number
  should not be over-trusted; (2) the inner signal's large angle-to-
  angle variability (0.0001 to 0.001, ~10x spread) is consistent with
  `jwave_test` run -44's curvature-dependent reflection mechanism, not
  yet confirmed as the specific cause here — worth remembering before
  assuming this channel is uniformly reliable across an irregular
  boundary.
- Physical sanity checked? by whom?: Claude — predicted both boundaries'
  reflection times analytically before running (not fit after the
  fact); confirmed the water-only control shows near-zero signal at
  both predicted times (ruling out a geometry/artifact false positive)
  before treating the phantom's peaks as real reflected signal.
- Gate passed? (Y/N): N/A — scout-phase channel validation.
- Next action: channel 1 (reflection) is validated as carrying real,
  usable information the transmission channel lacks — the natural next
  step is COMBINING both channels (transmission + reflection) into a
  single reconstruction, rather than adding channel 2 (absorption) yet,
  since fusing two validated channels is more valuable next than adding
  a third, unvalidated one. Also worth: checking whether the
  curvature-dependent variability hypothesis for the inner signal holds
  up under direct testing (not yet done). Not committed/pushed
  (standing instruction).

### Run 2026-07-09-09 — FUSED transmission + reflection: both boundaries recovered BLIND, with a real (and corrected) systematic bias on the inner one
- Phase: 1 (scout). Per user: "proceed, and do visualise this one for
  me to see" — fuses channel 0 (transmission) and channel 1
  (reflection) into one visualization, and critically, does GENUINE
  BLIND per-angle peak detection for reflection this time (run -08
  only checked envelope amplitude AT the already-known true boundary
  locations; this run finds peaks with no knowledge of where the true
  boundary is, the fair test of whether reflection alone could locate
  it in practice).
- Design: `phase1_fused_channel_reconstruction.py`. Transmission channel
  reused run -07's saved dataset (`results/patient001_two_tissue_rays.npz`,
  no resimulation) and reconstructed via SIRT (run -06's method).
  Reflection channel re-simulated the same pitch-catch A-scan as run
  -08 (72 forward simulations, ~15-20 min estimate given upfront,
  completed within it), this time with `scipy.signal.find_peaks`
  picking the two most prominent post-direct-arrival envelope peaks
  per angle (thresholded against that angle's own water-only baseline,
  no true-location information used), converting arrival time to
  radius via an ALL-WATER round-trip assumption (the simplest
  conversion a real device without prior tissue knowledge would use).
- **Result: both boundaries recovered BLIND, with excellent outer
  accuracy and a real, quantified bias on the inner one** (see
  `results/figures/phase1_fused_channel_reconstruction.png`). Outer
  boundary candidate found at 36/36 angles, mean error +0.1 cells
  (excellent — that leg is genuinely all-water, exactly as expected).
  Inner boundary candidate ALSO found at 36/36 angles (a stronger
  result than run -08's 25/36 "detectable" count, since this is a
  cleaner blind-peak-detection method rather than a narrow known-
  location search window), but with mean error +7.8 cells (a real
  systematic overshoot, visible in the figure as the green markers
  sitting consistently just outside the true inner contour).
- **Self-correction, caught by checking the actual sign of the data**:
  the script's own docstring predicted an UNDERSHOOT (inner boundary
  appearing too close), reasoning that a water-speed conversion of a
  partly-faster-tissue path would underestimate distance traveled.
  Re-deriving the algebra properly after seeing the result showed this
  was backwards: a path partly through faster myocardium takes LESS
  time to cover the same distance, so converting that shorter time
  using water's slower speed makes the reconstructed point look
  FARTHER away, not closer — an OVERSHOOT, matching the measured
  +7.8 cells. Flagged and corrected rather than silently reported as
  if predicted correctly (same discipline as run -03's analytic-sign
  bug and run -74's direction checks).
- **Second honest check, not glossed over**: computed the sound-speed-
  substitution mechanism's OWN predicted magnitude for a representative
  ray (~40 cells water leg, ~20 cells myocardium leg) — only ~0.7
  cells, roughly 10x SMALLER than the observed 7.8-cell bias. So the
  sound-speed mechanism is correctly SIGNED but does not by itself
  explain most of the error's SIZE — the larger contributor is more
  likely peak-detection timing precision and/or the true specular
  reflection point not lying exactly on the assumed straight radial
  line (a real reflection's specular point on an irregular boundary can
  sit off that line) — not yet isolated which dominates.
- Physical sanity checked? by whom?: Claude — worked through the
  round-trip-time algebra explicitly after seeing the result's sign,
  rather than accepting either the original (wrong) prediction or the
  data at face value without reconciling them; quantified the
  sound-speed mechanism's own predicted magnitude before attributing
  the full observed bias to it, catching that it only explains a small
  fraction.
- Gate passed? (Y/N): N/A — scout-phase channel-fusion visualization.
- Next action: isolate whether peak-detection timing or off-radial
  specular-point geometry dominates the remaining ~7-cell inner-
  boundary bias (not yet done) — matters for deciding whether a simple
  correction (e.g. assuming a nominal tissue sound speed for the second
  leg once an outer boundary is already known) would fix most of it,
  or whether the specular-point assumption itself needs revisiting.
  Also still open: patient023 (harder anatomy), and channel 2
  (absorption) once the transmission+reflection fusion is fully
  characterized. Not committed/pushed (standing instruction).

### Run 2026-07-09-10 — CHANNEL 2 (absorption): solver ported and validated to ~0.1% in this project's own context — correcting the user's hypothesis about run -09's bias, since attenuation was entirely ABSENT from every prior run
- Phase: 1 (scout), moving to Phase 2 (channel 2 on the multi-channel
  roadmap). Per user: "absorption might influence your prior
  prediction about the inner wall... expected." Correction made before
  building anything: absorption was not merely weak in runs -03 through
  -09, it was COMPLETELY ABSENT — jWave's base transient solver has no
  attenuation term at all, so it could not have caused run -09's +7.8
  cell bias. The broader instinct (attenuation is a real, expected
  future noise/detectability factor, especially for the inner boundary,
  which travels the longest total path through the most tissue) is
  sound and worth testing directly, once actually added.
- Ported `attenuation_solver.py` from `jwave_test` unchanged (CLONED,
  not re-derived — this is core jWave physics infrastructure, not
  reconstruction methodology, same reuse rationale as this project's
  cited tissue properties; `jwave_test`'s own validation of this code
  does NOT carry over to this project's different domain/background
  medium, so it needed a fresh check here). Wrote
  `validate_attenuation.py` (homogeneous MYOCARDIUM medium, point
  source, 4 receivers at increasing distance, lossless vs. attenuating
  run, ratio cancels geometric spreading). Compute: 2 simulations
  (much smaller than the 72-sim batches), ~2-5 min estimate given
  upfront, completed within it.
- **Result: validated to ~0.1% against the analytic exp(-alpha*distance)
  law** — obs.ratio/ref vs. expected: 1.0000/1.0000, 0.9555/0.9561,
  0.9133/0.9141, 0.8737/0.8740 (0.06%, 0.09%, 0.03% error) — matching
  `jwave_test`'s own original validation quality. Attenuation physics
  is now genuinely available in this project, not just cited as a
  tissue property.
- Physical sanity checked? by whom?: Claude — corrected the specific
  causal claim (absorption could not have caused an already-observed
  bias if it wasn't in the model at all) before agreeing with the
  broader hypothesis, rather than accepting the suggestion at face
  value; re-validated the ported solver fresh in this project's own
  domain/tissue context rather than assuming `jwave_test`'s validation
  transfers.
- Gate passed? (Y/N): N/A — infrastructure validation, not a phase gate
  (this project's own Gate 2 for the water-bath/transmission setup is
  still separately unresolved, per the protocol).
- Next action: re-run the two-tissue reflection/fusion test (runs
  -08/-09) with attenuation now switched on, to directly test whether
  it (a) further weakens inner-boundary detectability and/or (b)
  changes the +7.8 cell timing bias, rather than reasoning about it in
  the abstract. Not committed/pushed (standing instruction).

### Run 2026-07-09-11 — Channel 2 direct test: attenuation makes NO measurable difference at this toy scale — a clean null result on the user's hypothesis, not confirmation
- Phase: 1/2. Direct test of the hypothesis behind run -10 ("absorption
  might influence your prior prediction about the inner wall... the
  fact that tissue-affected sound speed/strength/frequency attenuation
  cause the noise... is expected"). Re-ran runs -08/-09's exact
  pitch-catch reflection scan and blind two-peak detection method on
  the SAME two-tissue patient001 phantom, changing ONLY the solver
  (`simulate_wave_propagation_attenuating`, validated run -10, instead
  of the lossless `simulate_wave_propagation`) so any difference is
  attributable to attenuation alone, not a confounded methodology
  change. `phase1_reflection_with_attenuation.py`. Compute: 72 forward
  simulations (same as prior batches), ~15-20 min estimate given
  upfront, completed within it.
- **Result: a clean null — every measured quantity matched run -09's
  no-attenuation result exactly at the reported precision.** Detection
  rate: 36/36 both boundaries (identical). Outer bias: +0.1 cells
  (identical). Inner bias: +7.8 cells (identical). Inner/outer
  amplitude ratio: 0.235 (identical to run -08). **Absorption is NOT
  the explanation for the +7.8 cell bias or for inner-boundary
  detectability limits in this specific toy phantom.**
- **Why, quantified rather than asserted**: the actual tissue path
  length here is short (~2mm myocardium wall thickness); at 2.5MHz,
  myocardium's cited attenuation predicts only ~5-6% amplitude
  reduction over that distance — real, but far too small to move a
  ~24% inner/outer amplitude ratio or a 7-8 cell timing bias
  detectably. **This is a SCALE-DEPENDENT finding, not a universal
  one** — over real full-chest distances (centimeters, not
  millimeters), the same per-cm attenuation coefficient would compound
  multiplicatively and become far more significant. The instinct that
  attenuation matters was reasonable to test; it just isn't the
  explanation at this toy phantom's scale.
- Physical sanity checked? by whom?: Claude — changed only the solver,
  not the method, isolating attenuation as the single variable;
  computed the expected attenuation magnitude analytically (~5-6% over
  the actual tissue path length) to explain WHY the null result makes
  physical sense, rather than reporting "no difference" without
  accounting for it.
- Gate passed? (Y/N): N/A — hypothesis test, not a phase gate.
- Next action: the remaining +7.8 cell inner-boundary bias is still
  unexplained by anything tested so far (sound-speed substitution:
  ~0.7 cells predicted, too small; attenuation: negligible at this
  scale) — peak-detection timing precision and off-radial specular-
  point geometry remain the leading untested candidates. Worth
  re-testing this same absorption question once real (cm-scale) full-
  anatomy distances are in play, since the scale-dependence found here
  predicts it WOULD matter there. Not committed/pushed (standing
  instruction).

### Run 2026-07-09-12 — Positive control isolates the +7.8 cell bias to the DETECTION METHOD, not real-anatomy shape irregularity — and the bias is actually WORSE for a perfect circle
- Phase: 1/2. Per user: "isolate those and diagnose the bias" — the
  two remaining candidates from run -09/-11 (peak-detection/group-delay
  timing precision, a property of the method; vs. off-radial specular-
  point geometry, a property of real anatomy's irregularity).
- Design: `phase1_circular_positive_control.py` — a PERFECTLY CIRCULAR,
  CENTERED two-tissue phantom (blood core R=60, myocardium ring R=80,
  matched to patient001's real scale), same tissue contrast, same
  pitch-catch + blind two-peak detection pipeline as runs -09/-11
  exactly. For a centered circle, a radially-placed monostatic pair's
  specular reflection point MUST lie exactly on that radial line, by
  symmetry — mechanism (2) (off-radial geometry) is STRUCTURALLY
  IMPOSSIBLE here. If the bias vanishes, mechanism (2) explains the
  real-anatomy result; if it persists, mechanism (1) (detection method)
  does. Compute: 72 forward simulations, ~15-20 min estimate given
  upfront, completed within it.
- **Result: the bias did NOT vanish — it got LARGER: +11.79 cells
  (std=4.94) for the perfect circle, vs. patient001's +7.8 cells.**
  Outer boundary remained excellent (-0.17 cells, std=0.40), same as
  every prior run. **This decisively rules out off-radial specular-
  point geometry as the (or even a major) cause — real anatomy's
  irregularity is not the problem; the detection method itself
  produces this bias even under perfectly ideal, symmetric conditions,
  and real anatomy's actual bias (+7.8) is milder than the idealized
  circular case's (+11.8), the opposite of what shape-irregularity-as-
  cause would predict.**
- **Leading hypothesis for the mechanism (not yet directly confirmed at
  the raw-waveform level)**: the sound-speed-substitution effect
  (~0.7 cells, same tissue thickness as before) is far too small to
  explain an 11.8-cell bias — roughly 16x too small. The most likely
  candidate is REVERBERATION between the two closely-spaced boundaries
  (a thin, high-impedance-contrast wall bounces the wave back and forth
  between inner and outer interfaces multiple times) — `jwave_test`'s
  own earlier project (run -42, a different geometry entirely) already
  confirmed this kind of multi-bounce reverberation IS real and present
  in these simulations, not hypothetical. The blind two-peak detector
  takes "the second most prominent peak" as the inner-boundary echo;
  if a later reverberation echo is more prominent than the true
  first-bounce inner reflection, it would be mistaken for the boundary
  itself — a shape-independent, detection-method artifact, consistent
  with everything observed.
- Physical sanity checked? by whom?: Claude — designed the isolating
  experiment specifically so its OWN symmetry would make one hypothesis
  structurally impossible, rather than just re-testing on another
  irregular case; computed the sound-speed-substitution mechanism's own
  predicted magnitude again for this geometry (~0.7 cells, unchanged)
  before ruling it out as insufficient to explain the observed 11.8
  cells.
- Gate passed? (Y/N): N/A — diagnostic isolating test.
- Next action: NOT yet done — direct confirmation at the raw-waveform
  level (inspect a full A-scan trace for additional peaks beyond the
  two currently extracted, which would directly show reverberation
  rather than inferring it). The reverberation hypothesis, if
  confirmed, suggests a concrete fix: prefer the FIRST peak after the
  outer-boundary reflection that exceeds a threshold (true first-bounce
  echo), rather than "second most prominent" (which can grab a later,
  stronger reverberation echo instead). Not committed/pushed (standing
  instruction).

### Run 2026-07-09-13 — Reverberation mechanism CONFIRMED directly: the true inner echo sits at peak position #4, not #2 — bias drops from +11.79 to +1.31 cells once correctly matched, but a real blind detector still can't find it reliably
- Phase: 1/2. Per user: "thats why we need a time-of-impact to measure
  which hits first, second, third etc. to differentiate first-order
  bounce vs echoes" — clarified before building that run -09/-12's
  detector already sorts peaks chronologically (first=outer,
  second=inner); the actual gap is that a reverberation echo can
  legitimately occupy that same "position #2" for a thin, high-contrast
  wall, so ordinal position alone doesn't identify the right PHYSICAL
  echo — what's needed is matching against the analytically PREDICTED
  true single-bounce arrival time, not just counting peaks in order.
- Design: `phase1_diagnose_reverberation.py` — same circular positive-
  control phantom as run -12, but finds ALL peaks per angle (not just
  the first two) and identifies which chronological peak position is
  closest to the analytically predicted true inner-reflection time
  (same `predicted_reflection_times` formula validated in run -08).
  Compute: 72 forward simulations, ~15-20 min estimate given upfront,
  completed within it.
- **Result: mechanism directly confirmed.** A peak matching the true
  predicted time was found at SOME position in 12/36 angles; of those,
  8/12 (67%) matched at chronological POSITION #4, not #2 — confirming
  2 extra (reverberation) peaks arrive before the genuine first-bounce
  inner echo. Accuracy using the correctly-matched peak: mean error
  **+1.31 cells** (vs. the naive position-#2 method's +11.79 cells,
  matching run -12 exactly) — an order-of-magnitude improvement, and
  right in line with the ~0.7-1.3 cell scale the sound-speed-
  substitution mechanism alone would predict (run -09). **This confirms
  the reflection channel's underlying physics is fine — the bias was
  entirely a peak-MISIDENTIFICATION artifact (counting ordinal position
  instead of matching expected arrival time), not a fundamental
  limitation of the channel itself.**
- **Honest limit, not glossed over**: a matching peak was found at ANY
  position in only 12/36 (33%) of angles — at the other 24, nothing
  landed close enough to the predicted time to count as a confident
  match (genuinely absent, too weak, or shifted beyond the matching
  window). This confirms the MECHANISM and shows the accuracy CEILING
  once the true echo is correctly identified — it does not hand over a
  robust standalone blind detector. A real fix needs something like
  matched filtering against the expected reverberation PATTERN (not
  just the single target echo), to reliably identify the right peak
  without already knowing the true boundary location — a real,
  nontrivial next step, not solved by this diagnosis alone.
- Physical sanity checked? by whom?: Claude — distinguished "detector
  ignores arrival order" (false — it already sorts chronologically)
  from the actual mechanism ("ordinal position doesn't guarantee the
  right physical echo when reverberation can occupy the same position")
  before building the diagnostic, rather than accepting the surface-
  level framing of the hypothesis; reported the 33% match-rate
  limitation explicitly rather than letting the order-of-magnitude
  accuracy improvement stand as an unqualified win.
- Gate passed? (Y/N): N/A — mechanism-diagnosis test.
- Next action: build a proper matched-filter or multi-hypothesis
  reverberation model (predict the FULL expected echo train — direct
  inner reflection plus specific reverberation paths — and fit observed
  peaks against that whole pattern) rather than either naive ordinal
  counting or single-prediction matching, to get a robust BLIND
  detector rather than one that only works when the true answer is
  already known. Also still open: patient023 (harder anatomy), the
  real irregular-shape version of this same reverberation diagnosis
  (this run used the circular positive control only), and testing this
  at real full-scale (cm) anatomy distances where run -11 predicted
  attenuation would start to matter. Not committed/pushed (standing
  instruction).

### Run 2026-07-09-14 — Matched filter echo extraction: a real scipy indexing bug caught and fixed via a synthetic self-test, then a nuanced (not clean) result — MORE peaks found, not fewer, arguing FOR genuine multi-bounce reverberation rather than simple ringing artifacts
- Phase: 1/2. Per user's proposed direction (link every echo in a
  single pulse-fire back to the specific surface/bounce-count that
  produced it -- a "complete-picture inverse equation"): before
  attempting that, tested whether run -13's extra pre-echo peaks were
  genuine distinct bounces or just ringing/sidelobes of the strong
  outer reflection's own waveform (which raw envelope detection can't
  distinguish from real echoes). Built
  `phase1_matched_filter_echo_extraction.py`: cross-correlate the
  received trace against the KNOWN transmitted toneburst (standard
  radar/sonar pulse compression), which should collapse ringing while
  preserving genuinely separate echoes.
- **Bug caught before trusting any result**: the FIRST run's sanity
  check (matched-filter outer-boundary timing, should be ~0 like run
  -12's -0.17 cells) came back at **-73.69 cells** — clearly wrong.
  Diagnosed via a cheap SYNTHETIC self-test (no jWave, pure numpy: a
  known, exactly-injected delay through the same correlate/hilbert
  pipeline) before spending another compute cycle: `scipy.signal.
  correlate(..., mode="same")`'s raw output index does NOT correspond
  to actual delay time -- confirmed the correct approach is
  `mode="full"` with `lag = index - (n-1)`, `delay_time = lag * dt`.
  Also found a second, more subtle issue while deriving this: matched
  filtering against the FULL known pulse shape recovers the TRUE
  ballistic delay directly, with NO group-delay correction needed
  (unlike raw envelope-peak detection, which peaks at the pulse's
  CENTER and needs the `_ENVELOPE_GROUP_DELAY_S` correction) --
  applying that correction a second time would have been a new,
  self-inflicted bias. Both fixes verified together on the synthetic
  test (exact match to the injected delay) before rerunning the real
  72-simulation batch.
- **Result, corrected: sanity check now passes cleanly** (-0.25 cells,
  matching run -12's -0.17). **But the main result is a genuine
  complication, not a clean fix — reported honestly rather than spun.**
  Matched filtering found MORE peaks per angle (mean 9.2) than raw
  envelope detection (~4) — the OPPOSITE of what "these are just
  ringing/sidelobe artifacts" would predict (a proper matched filter
  suppresses a single echo's own ringing while preserving genuinely
  separate arrivals, so finding MORE distinct peaks with better
  temporal resolution argues these are more likely REAL, separate
  multi-bounce echoes, not artifacts of crude envelope smoothing).
  Match rate against the true predicted inner-echo time actually
  DECREASED (4/36, down from run -13's 12/36) — a real complication.
  Where matched, accuracy improved further (-0.30 cells, vs. run -13's
  +1.31), consistently at chronological position #3 (not #2, not #4).
  Naive position-#2 accuracy remains badly biased regardless of
  detection method (+9.57 cells here vs. run -13's +11.79 with raw
  envelope) — confirming the fix needed is WHICH peak to trust
  (physics-informed), not which signal-processing method extracts
  peaks.
- Physical sanity checked? by whom?: Claude — caught the correlation-
  alignment bug via its own dedicated sanity check (not by comparing
  final numbers against expectation after the fact); diagnosed and
  fixed it using a cheap synthetic test with a known exact answer
  BEFORE spending another 15-20 min compute cycle on the real
  simulation; reported the match-rate decrease honestly as a
  complication rather than omitting it once the headline accuracy
  number looked good.
- Gate passed? (Y/N): N/A — diagnostic/method-validation test.
- Next action: this result argues FOR building the user's proposed
  multi-bounce path-enumeration model (predict the FULL expected echo
  train -- direct outer, direct inner, N-round-trip internal
  reverberations -- and fit ALL observed peaks against that whole
  pattern at once) rather than continuing to refine single-echo
  detection methods, since the evidence now points toward genuinely
  multiple real echoes rather than a signal-processing artifact to be
  filtered away. Not committed/pushed (standing instruction).

### Run 2026-07-09-15 — Multi-bounce cascade model: an informative NEGATIVE — simple radial reverberation explains only 13% of observed peaks, isolating the real cause to the pitch-catch pair's un-modeled bistatic geometry
- Phase: 1/2, moving toward Phase 2. Per user's "complete-picture
  inverse" proposal (proceed with building it): implemented the full
  physically-derived multi-bounce cascade for this two-interface
  geometry -- outer reflection (never enters tissue), direct inner
  reflection (1 bounce), and inner reflection with 1/2/3 EXTRA internal
  round-trips inside the myocardium wall before finally exiting. Each
  extra round-trip is purely additive (+2*wall_thickness/c_myo), so
  ALL reverberation terms are analytically guaranteed to arrive LATER
  than the direct inner echo, never earlier -- a concrete, falsifiable
  prediction, not an assumption. `phase1_multibounce_cascade_model.py`,
  reusing run -14's validated matched-filter extraction (more genuine
  distinct echoes resolved than raw envelope). Compute: 72 forward
  simulations, ~15-20 min estimate given upfront, completed within it.
- **Result: the model mostly FAILS to explain the data -- a real,
  informative negative, not a confirmation.** Only 44/332 total
  detected peaks (13%) matched ANY cascade entry within the matching
  window; 288/332 (87%) are unexplained by simple radial reverberation.
  Of the matches: outer dominated (32/44, as expected -- this timing
  was already validated in runs -12/-14). Direct inner echo (inner_k0):
  only 4 matches, but excellent accuracy where matched (-0.30 cells,
  consistent with runs -13/-14). Reverberation terms got weak,
  INCONSISTENT support: inner_k1 (1 extra round-trip) got ZERO matches,
  while inner_k2 (2 extra round-trips, a later and naively "less
  likely" prediction) got 8 -- a non-monotonic pattern more consistent
  with coincidental alignment than genuine confirmation of that
  specific bounce count. Critically, 64/288 unexplained peaks arrive
  BEFORE the direct inner echo -- something the radial cascade model
  CANNOT produce by construction (every reverberation term only adds
  time), directly confirming these "mystery" early peaks (first
  flagged runs -12/-13) are NOT radial in-tissue reverberation at all.
- **Diagnosis: the dominant remaining candidate is the ONE geometric
  approximation running through every prediction in this whole
  investigation (runs -08 through -15) -- the pitch-catch pair is not
  perfectly monostatic.** The transmitter and receiver sit 5 cells
  apart tangentially (`OFFSET_CELLS`, run -08's convention, reused
  unquestioned since), not at the same point. For a target along the
  exact radial line this is a small correction, but for reflection off
  a CURVED interface, the true specular point and path length for a
  slightly-separated bistatic pair genuinely differ from the monostatic
  approximation every `predicted_*_times` function in this project has
  used -- and unlike the radial cascade, a bistatic correction could
  plausibly produce arrivals both BEFORE and after the direct inner
  echo, matching what's actually observed (the radial model only
  explains "after").
- Physical sanity checked? by whom?: Claude -- built the cascade from
  first-principles path-length algebra (not fit to the data after the
  fact) so the model's failure is a genuine test result, not a
  tuning failure; specifically checked whether unexplained peaks
  clustered before or after the direct inner echo (rather than only
  reporting an aggregate match rate) since the radial model's own
  structure makes a clear, checkable prediction about WHICH side any
  real gap should appear on.
- Gate passed? (Y/N): N/A -- model-validation test, informative negative.
- Next action: do NOT extend the radial reverberation cascade further
  (k=5, k=6, ...) -- the data does not support it. Instead, properly
  model the BISTATIC specular geometry for the actual pitch-catch
  offset (compute the true reflection point and path length for a
  tangentially-separated tx/rx pair off a circular arc, rather than
  assuming monostatic) before attempting any further "complete-picture"
  echo-train modeling -- this has been an unexamined approximation
  since run -08 and is now the leading candidate for most of the
  unexplained peak structure. Not committed/pushed (standing
  instruction).

### Run 2026-07-09-16 — Off-axis outer-boundary bounce model CONFIRMED: most of the "mystery gap" is wider-angle first-order bounces off the SAME outer wall, not reverberation or inner-boundary echoes
- Phase: 1/2. Per user's proposed mechanism: run -15's unexplained
  peaks arriving between the direct outer and direct inner echoes are
  wider-angle, first-order bounces off the SAME outer boundary (a real
  point source illuminates the whole nearby arc, not just the exact
  radial point) -- longer water-only path (later arrival) and weaker
  amplitude (reduced reflection efficiency at larger incidence angle)
  as the off-axis angle increases, not tissue reverberation or the
  inner boundary at all.
- Design: `phase1_offaxis_outer_bounce_model.py`. Derived the round-
  trip time as a function of off-axis angle phi analytically (law of
  cosines, monostatic approximation: dist(phi)^2 = R_probe^2 +
  R_outer^2 - 2*R_probe*R_outer*cos(phi)), checked BEFORE running that
  this model could even reach the mystery gap using ordinary angles:
  phi~26 degrees already reaches the direct-inner-echo time (7.80us)
  using pure water-path geometry alone -- a strong a priori plausibility
  check, not an assumption. For every detected peak in the mystery gap
  (between the direct outer and direct inner echo times), inverted the
  model to get an IMPLIED off-axis angle, and tested whether implied
  angle vs. amplitude follows the predicted monotonically-decreasing
  relationship -- the "equation to normalize" the user asked for.
  Compute: 72 forward simulations, ~15-20 min estimate given upfront;
  completed successfully, though a trivial plotting-only bug (unrelated
  to the physics, an empty-placeholder scatter call) crashed the script
  AFTER all numeric results printed -- fixed for future runs.
- **Result: CONFIRMED, cleanly.** 92 peaks in the mystery gap were
  explained by some phi in [0,75] degrees. Correlation(implied angle,
  amplitude) = **-0.832** -- a strong negative relationship, exactly as
  predicted (wider angle, weaker echo). Fitted power-law falloff:
  amplitude ~ cos(phi)^20.4 -- a steep but physically sensible falloff
  for a smooth, large-radius-of-curvature reflector (a tight specular
  lobe, not broad diffuse scattering). **This gives a concrete,
  physically-grounded normalization equation**: predicted arrival time
  from phi via the law-of-cosines formula above, predicted relative
  amplitude via cos(phi)^20.4 -- usable to predict and down-weight/
  subtract off-axis outer-wall contributions before searching for the
  genuine inner-boundary echo, replacing the ad hoc peak-position
  heuristics used in every run since -08.
- Physical sanity checked? by whom?: Claude -- checked the geometric
  model's reach into the mystery gap analytically BEFORE running any
  simulation (falsifiable a priori check, not fit after seeing data);
  reported the strong correlation number directly rather than only a
  qualitative "peaks form clusters" observation, giving a testable,
  reusable quantitative relationship.
- Gate passed? (Y/N): N/A -- mechanism-confirmation test.
- Next action: this substantially resolves the "mystery peak" question
  raised across runs -12 through -16 -- most of the previously-
  unexplained peak structure is now attributed to a real, quantified,
  physically-grounded mechanism (off-axis outer-wall bounces), not
  reverberation and not a detection-method artifact. Remaining open:
  (a) build a proper composite model (outer direct + off-axis outer
  family + direct inner + reverberation cascade) and re-test overall
  peak coverage, to see how much of the 87%-unexplained figure from run
  -15 this single mechanism accounts for on its own; (b) the same
  question for the INNER boundary's own off-axis family (this run only
  modeled off-axis bounces off the OUTER wall); (c) patient023/real
  irregular anatomy, still not yet re-examined with any of this
  reverberation/off-axis machinery. Not committed/pushed (standing
  instruction).

### Run 2026-07-09-17 — Separated the two amplitude-loss mechanisms the user flagged: geometric spreading (mild) vs. angular reflection efficiency (dramatic), previously conflated into one fitted exponent
- Phase: 1/2. Per user's direct question: "both travel time (mild) and
  bounce (dramatic) cause a drop in strength, did you account for
  that?" — correct catch: run -16's single `amplitude ~ cos(phi)^n`
  fit (n=20.42) conflated BOTH effects into one number, since implied
  angle and travel distance both increase together in this geometry.
- Fix: added `dist_cells(phi, r_outer)` (the same law-of-cosines
  formula already used for arrival time, applied to distance), computed
  the 2D monostatic round-trip geometric-spreading correction
  (~1/distance, from two legs of ~1/sqrt(distance) cylindrical
  spreading each), applied it to UNDO the mild distance-based loss
  BEFORE refitting the angular falloff, isolating the angle-only
  effect. Re-ran the same 72-simulation experiment (no new physics,
  same data, corrected analysis) — this run's compute estimate (~15-20
  min) given upfront as usual.
- **Result: both effects are real, and now separated rather than
  conflated.** Raw (conflated) fit: n=20.42, correlation=-0.832.
  Spreading-corrected (angle-only) fit: **n=16.51**, correlation=-0.796.
  Distance in this geometry ranged only 1.51x across the tested angles
  (0-27 degrees) — the "mild" spreading effect accounts for a real but
  modest fraction of the total apparent falloff (n dropping by ~4, about
  19% of the raw exponent), while the angular reflection-efficiency
  effect remains dominant (n=16.51, still a genuinely steep, tight
  specular lobe) — confirming the user's own characterization
  ("travel time mild, bounce dramatic") quantitatively, not just
  qualitatively.
- Physical sanity checked? by whom?: Claude — did not accept the raw
  fit's steepness as a clean measurement of the angular effect once the
  conflation was pointed out; computed the actual distance RANGE (1.51x)
  to show explicitly how much of the correction's effect is plausible
  before reporting the corrected exponent, rather than asserting the
  fix mattered without quantifying it.
- Gate passed? (Y/N): N/A — model-refinement test.
- Next action: this two-parameter model (arrival-time -> implied angle
  via the law-of-cosines geometry, amplitude -> corrected
  reflection-efficiency via cos(phi)^16.5) is now a complete, physically
  separated normalization the user asked for, ready to reuse for
  down-weighting/subtracting off-axis outer-wall contributions in any
  future composite echo-train model. Raw data saved to
  `results/offaxis_outer_bounce_data.npz` (phi, amp, amp_corrected,
  dist, t) for reuse without resimulating. Remaining open items
  unchanged from run -16 (composite model coverage check, inner-
  boundary's own off-axis family, patient023/real anatomy). Not
  committed/pushed (standing instruction).

### Run 2026-07-09-18 — Composite coverage check: 13% -> 40% explained, a real improvement, but with a genuine methodology flaw caught before trusting the number
- Phase: 1/2. Per user: "composite coverage check" -- combined the
  discrete cascade (outer/inner/reverberation, run -15) with the
  off-axis outer-wall family (runs -16/-17), capped at 30 degrees (near
  the empirically-validated ~27-degree range those runs actually
  tested, not extended to the full 75-degree search range, which would
  encroach further into inner_k1/k2 territory). `phase1_composite_coverage_check.py`:
  classifies every detected peak against whichever candidate
  (discrete cascade entry, or best-fit off-axis angle) has the smallest
  timing residual. Compute: 72 forward simulations, ~15-20 min estimate
  given upfront, completed within it.
- **Result: coverage improved substantially -- 132/332 (40%) explained,
  up from run -15's 44/332 (13%).** Off-axis outer family alone
  explained 116 peaks, more than 7x the entire discrete cascade
  combined (16). Real, meaningful progress on run -15's original
  87%-unexplained finding.
- **Caught before trusting the number: `inner_k0` (the direct inner-
  boundary echo) got ZERO matches this run, down from run -16's 4 —
  a red flag, not a clean improvement.** The off-axis-outer
  continuum's capped range (30 degrees) reaches almost exactly to
  where the direct inner echo is predicted (~26 degrees, per run -16's
  analytic check) -- the two candidate explanations OVERLAP in time
  near that boundary. Since the classifier picks whichever has the
  smallest residual, and a continuous 300-point angular search will
  almost always find SOME nearby angle with a tiny residual, it can
  out-compete the discrete inner-boundary hypothesis by chance
  closeness rather than genuine correctness. **This means the 40%
  figure likely OVERSTATES true coverage -- it is probably absorbing
  some genuine inner-boundary echoes into the off-axis-outer bucket**,
  not because they are actually outer-wall reflections, but because the
  two models were not given clean, non-overlapping territory to compete
  over.
- Physical sanity checked? by whom?: Claude -- checked the per-category
  breakdown (not just the aggregate 40%) specifically because the
  aggregate improvement alone could hide a redistribution-not-genuine-
  explanation problem; noticed inner_k0's count dropping to zero
  (a comparison against run -16's own result, not just an isolated
  number) before accepting the 40% figure as clean progress.
- Gate passed? (Y/N): N/A -- model-validation test, real improvement
  with a caught caveat.
- Next action: fix the classifier before trusting the 40% figure
  precisely -- either cap the off-axis-outer range more conservatively
  (well short of inner_k0's predicted time, e.g. 15-20 degrees instead
  of 30) or give discrete cascade matches priority over off-axis
  matches whenever both are plausible, rather than pure smallest-
  residual competition. Re-run the coverage check with that fix before
  reporting a trusted final coverage number. Not committed/pushed
  (standing instruction).

### Run 2026-07-09-19 — Composite coverage check v2: hierarchy-corrected, honest number is 33% (down from the flawed 40%), with inner_k0 correctly recovered
- Phase: 1/2. Per user's precise diagnosis of run -18's flaw ("classic
  statistical phenomenon known as overfitting via continuity... a
  discrete prediction occupies only a single point in time [while the
  continuous off-axis model] creates a near-unbroken time-of-flight
  continuum... cannibalizing genuine internal reflections") and
  specified fix: (1) hard-gate the off-axis search at
  phi_critical - 2.5 degrees (phi_critical = angle where off-axis
  outer time exactly equals the direct inner-echo time); (2) give
  discrete cascade matches priority over off-axis matches; (3) for
  genuine grey-zone cases (both plausible -- per user, "further travels
  outer and fast travel inner CAN land at the same time but with
  distinct energy"), use amplitude as a tiebreaker rather than
  defaulting either way blindly.
- Design: `phase1_composite_coverage_check_v2.py`. Computed
  phi_critical=25.81 degrees (via the same `implied_phi` root-find
  already validated in runs -16/-17), gated the off-axis search to
  [0, 23.31] degrees. Calibrated an amplitude model from run -17's
  saved data (`amplitude = 0.01261*cos(phi)^20.42`, the RAW conflated
  exponent, appropriate here since we want to predict actually-observed
  amplitude, not the spreading-corrected one) -- if an ambiguous peak's
  observed amplitude exceeds 2x this prediction, classify it as the
  discrete (genuine) echo instead of off-axis. Compute: 72 forward
  simulations, ~15-20 min estimate given upfront, completed within it.
- **Result: honest coverage is 33% (108/332), down from run -18's
  flawed 40% -- but `inner_k0` is now correctly recovered at 4 matches**
  (matching run -16's original, un-cannibalized result), confirming the
  hierarchy fix worked as intended. Breakdown: outer=0 (now correctly
  absorbed into the off-axis family's own phi=0 case, not double-
  counted), inner_k0=4, inner_k2=8, off-axis outer (gated)=96.
  32 peaks were genuinely ambiguous (grey zone) -- of these, **0 were
  resolved to discrete via the amplitude-anomaly check**, all
  classified as off-axis by default.
- **Honest caveat on the tiebreaker itself, not glossed over**: zero
  grey-zone reclassifications could mean all 32 really are weak
  off-axis bounces -- or it could mean the 2x anomaly threshold, or the
  underlying assumption that a genuine inner echo should look
  dramatically stronger than a comparable-delay off-axis bounce, isn't
  well-calibrated. The inner echo itself travels through two real
  tissue legs with its own transmission/reflection losses, so it may
  not always be dramatically stronger than a similar-delay off-axis
  bounce -- this specific part of the fix is unresolved, not confirmed
  working correctly, and shouldn't be assumed clean just because it ran
  without error.
- Physical sanity checked? by whom?: Claude -- verified inner_k0's
  count specifically returned to run -16's original value (4) as the
  direct confirmation the hierarchy fix worked, rather than only
  checking the aggregate percentage; reported the grey-zone
  tiebreaker's zero-reclassification result as an open question rather
  than silently treating "it ran" as "it works."
- Gate passed? (Y/N): N/A -- model-validation test, corrected.
- Next action: the amplitude-tiebreaker's calibration is the most
  valuable remaining open item -- worth checking against known cases
  (e.g. does it correctly classify the run -16-confirmed inner_k0
  matches as discrete when deliberately placed in the grey zone, a
  sanity check not yet done) before trusting it on genuinely ambiguous
  peaks. Also still open: the inner boundary's own off-axis family
  (not yet modeled), and patient023/real irregular anatomy. Not
  committed/pushed (standing instruction).

### Run 2026-07-09-20 — Inner boundary's own off-axis family: geometry verified, but the mechanism does NOT confirm cleanly (honest negative, unlike the outer case)
- Phase: 1/2. Per user: "model the inner boundary's own off-axis
  family." Built `phase1_inner_offaxis_bounce_model.py`: for an
  off-axis angle psi on the inner circle, found where the straight
  line from probe to that point crosses the OUTER circle (line/circle
  intersection), splitting the path into a water leg (at c_water) and
  a myocardium leg (at c_myo) -- a straight-line/no-refraction
  approximation (documented, not hidden: the wave should genuinely bend
  via Snell's law at the oblique outer crossing, which this ignores).
  **Verified analytically before running anything**: at psi=0, this
  exactly reproduces the already-validated direct inner-echo leg
  lengths (40.0/20.0 cells) to machine precision -- PASS. Gated the
  search against the next cascade term (inner_k1); psi_critical was not
  reached within the 40-degree search range, so no gating was needed
  here (unlike the outer model against inner_k0). Compute: 72 forward
  simulations, ~15-20 min estimate given upfront, completed within it.
- **Result: the mechanism does NOT confirm cleanly here — an honest
  negative, in contrast to run -16's strong outer-boundary
  confirmation.** 104/236 peaks (at/after the direct inner echo time)
  matched SOME psi in the model's range, but correlation(implied psi,
  amplitude) = -0.049 (essentially zero, vs. the outer model's -0.832)
  and the fitted power-law exponent is flat (n=0.15, vs. the outer
  model's n=20.42). The geometry is correct (sanity check passed) but
  the matched peaks do not show the same coherent physical signature
  the outer family did — most likely these 104 matches are a mix of
  unrelated things (noise, other mechanisms) coincidentally aligning in
  TIME, not one real physical family, plausibly because inner-boundary
  signals are already much weaker to start with (see below) before any
  angular falloff even applies.
- **Separately, a major reframing surfaced in the same discussion**:
  computed the REAL reflection/transmission coefficients from this
  project's own cited tissue properties (acoustic impedance Z=density*
  sound_speed): R_outer(water/myo)=5.02%, R_inner(myo/blood)=0.253%
  (confirming the long-standing "weak blood/myocardium contrast"
  finding, now quantified exactly), giving direct-inner-echo amplitude
  (T*R_inner*T) at only ~20x weaker than direct-outer, and ONE-EXTRA-
  ROUND-TRIP reverberation (k1) at a further ~8000x weaker than THAT --
  i.e., ~160,000x weaker than the direct outer echo, almost certainly
  below any realistic simulation noise floor. **This casts serious
  doubt on runs -15/-18/-19's "inner_k2" matches (8 peaks each run)** --
  if k1 alone is already far too weak to detect, k2 (two extra round
  trips) is weaker still; those matches were almost certainly
  coincidental time-alignment with something else (off-axis
  contributions or numerical noise), not genuine second-order
  reverberation. Also found a real, UNRESOLVED discrepancy: the
  coefficient theory predicts direct-inner/direct-outer amplitude ratio
  ~5%, but run -08's actual measured ratio was ~23.5% (~4.7x higher) --
  most likely because these are flat-interface, normal-incidence
  coefficients, and the inner boundary's smaller radius of curvature
  means it doesn't behave like a flat reflector (the same mechanism
  `jwave_test` diagnosed in its own run -44) -- not explained away,
  left as an open, flagged gap.
- Physical sanity checked? by whom?: Claude -- verified the leg-split
  geometry analytically at psi=0 BEFORE running any simulation (not
  after); computed the coefficient-based amplitude strata from first
  principles (cited tissue properties) rather than asserting the
  "reverberation is negligible" conclusion without a number; explicitly
  compared the theoretical ratio against run -08's actually-measured
  ratio rather than treating the simple theory as automatically
  correct.
- Gate passed? (Y/N): N/A -- mechanism test (negative) + independent
  theoretical reframing.
- Next action: per user's proposed fix (amplitude strata should make
  bounce-order classification "an easy job"), build a proper
  coefficient-based amplitude-strata classifier (order-1 paths at
  either boundary within roughly an order of magnitude of each other;
  ANY reverberation order treated as essentially undetectable given the
  ~160,000x predicted gap) to REPLACE the ad hoc "2x anomaly" tiebreaker
  from run -19, and re-examine whether the "inner_k2" matches from runs
  -15/-18/-19 survive this stricter, physically-grounded test (strong
  prior expectation: they will not). Also still open: the curvature-
  vs-flat-interface amplitude discrepancy (~4.7x, unresolved), and
  patient023/real irregular anatomy. Not committed/pushed (standing
  instruction).

### Run 2026-07-09-21 — Amplitude-strata veto: decisive confirmation — "inner_k2" was never real, "outer" and "inner_k0" both cleanly validated
- Phase: 1/2. Direct test of the user's amplitude-strata classifier
  idea, applied to the most decisive available case: are runs
  -15/-18/-19's "inner_k2" matches (8 peaks, consistently found across
  three separate runs) amplitude-consistent with genuine 2nd-order
  reverberation, given run -20's coefficient-derived prediction that
  k2 should be ~160,000x weaker than the direct outer echo?
  `phase1_amplitude_strata_veto.py`: calibrated the coefficient-
  predicted amplitude scale for each cascade category to the measured
  direct-outer amplitude (median 0.01438), then vetoed any match whose
  OBSERVED amplitude exceeded 10x its category's coefficient-predicted
  value. Compute: 72 forward simulations, ~15-20 min estimate given
  upfront, completed within it.
- **Result: unanimous, decisive.** `outer` (32 matches) and `inner_k0`
  (4 matches) BOTH survived cleanly -- every observed amplitude landed
  within the predicted order-of-magnitude range, zero vetoes. **All
  8/8 `inner_k2` matches were VETOED** -- observed amplitude
  (7.48e-4) was ~64,000x STRONGER than the coefficient-predicted scale
  for genuine k2 reverberation (1.17e-11), an utterly decisive
  mismatch, not a close call. **This confirms run -20's prediction and
  corrects three earlier runs**: the "inner_k2" label in runs -15, -18,
  and -19 was wrong every time -- those 8 peaks were never genuine
  second-order reverberation. They are something else (most likely
  off-axis contribution or another still-unidentified mechanism) that
  happened to coincide in TIME with the k2 prediction. The
  reverberation-cascade concept beyond the direct (k0) echo contributes
  NOTHING real in this phantom -- a clean elimination, confirmed by
  amplitude, not just argued from theory.
- Physical sanity checked? by whom?: Claude -- calibrated the
  prediction to a REAL measured baseline (direct-outer amplitude, the
  most validated quantity in this whole investigation) rather than
  relying on uncalibrated theoretical units alone; checked that the
  categories expected to survive (outer, inner_k0) actually DID survive
  cleanly, not just that the expected-to-fail category failed --
  a two-sided check, not a test rigged to only look for the negative.
- Gate passed? (Y/N): N/A -- decisive mechanism-elimination test.
- Next action: the amplitude-strata veto should now be built into the
  composite classifier as a standing filter (any reverberation-order
  match gets vetoed by default given the ~160,000x gap, rather than
  ever being offered as a live hypothesis) -- this would clean up
  run -19's 33% coverage figure by removing the 8 spurious inner_k2
  "explanations" and reclassifying them as unexplained (or matchable
  against a still-undetermined alternative mechanism). Also still open:
  what the vetoed 8 peaks actually ARE (their amplitude, ~7.5e-4, is
  only ~3x weaker than inner_k0's own 2.4e-3 -- much closer in scale to
  a genuine order-1 echo than to noise, worth investigating rather than
  leaving as an unexplained residual); the curvature-vs-flat-interface
  amplitude discrepancy (~4.7x, unresolved); patient023/real irregular
  anatomy. Not committed/pushed (standing instruction).

### Run 2026-07-09-22 — "Further-travelled k0" hypothesis tested directly and RULED OUT (geometry, not just weak correlation) -- vetoed peaks remain genuinely unexplained
- Phase: 1/2. Per user's specific hypothesis for run -21's 8 vetoed
  "inner_k2" peaks: "the vetoed slightly weaker k0 is most likely a
  further-travelled k0... if you can somehow also normalize it then go
  ahead." Diagnosed why run -20's inner off-axis test may have been
  under-powered: unlike the outer model (run -17), the raw fit was
  never corrected for geometric spreading before fitting the angular
  relationship -- redid this properly using run -20's ALREADY-SAVED
  data (`results/inner_offaxis_bounce_data.npz`, no new simulation
  needed), normalizing by the total physical round-trip distance at
  each angle (analogous to run -17's outer-model correction).
- **Result: the hypothesis does not hold up, and fails in TWO
  independent ways.** (1) Spreading correction did not rescue a
  physical relationship -- correlation went from -0.049 (raw) to
  +0.242 (corrected), the WRONG SIGN (a genuine reflection-efficiency
  falloff must be negative; the fitted exponent, n=-1.06, is also
  unphysical, implying amplitude increasing with angle). (2) More
  decisively: NONE of run -20's 104 saved data points land anywhere
  near the vetoed peaks' actual arrival time (17.95us), and solving
  for what off-axis angle WOULD explain that time (extended search to
  60 degrees) returns NO SOLUTION -- a single-bounce, off-axis
  inner-boundary echo is GEOMETRICALLY INCAPABLE of arriving that
  late, at any reasonable angle. **This is a clean, decisive
  elimination by geometry, not an inconclusive normalization
  failure.**
- Physical sanity checked? by whom?: Claude -- reused already-saved
  data rather than re-simulating, to test the hypothesis cheaply before
  committing more compute; checked BOTH the correlation-sign test and
  the direct geometric reachability test, rather than stopping at the
  first (weaker) piece of negative evidence.
- Gate passed? (Y/N): N/A -- hypothesis test, decisive negative.
- Next action: per user, moving to patient023 (real anatomy) to
  validate the established mechanisms (direct outer/inner echoes,
  off-axis outer family with its spreading/angular-efficiency
  separation, amplitude-strata veto for reverberation-order matches) on
  real irregular anatomy rather than the synthetic circular positive
  control. The vetoed peaks' true identity remains OPEN and unsolved --
  best remaining untested candidate is the one geometric simplification
  present in every model so far (treating the pitch-catch pair as
  perfectly monostatic despite the real 5-cell tangential separation),
  not pursued further per explicit user instruction to move on. Not
  committed/pushed (standing instruction).

### Run 2026-07-09-23 — Patient023 (real anatomy) validation: core physics transfers, but detection reliability drops substantially -- consistent with this whole project's (and jwave_test's) established real-vs-idealized finding
- Phase: 1/2. Per user: "move to patient023 to validate the
  established mechanism we noted so far." Built
  `phase1_patient023_validation.py`: loads patient023's real, prepped
  contour (`jwave_test/results/mri_irregular_ring_patient023_slice4.npz`
  -- same file `jwave_test`'s own extensive sparse-probe investigation
  used as its hardest real case, ~45% real contraction), builds a
  two-tissue medium (myocardium wall + LV/blood core) in the water-bath
  domain, runs the same 36-angle pitch-catch reflection + matched-
  filter pipeline validated on the circular positive control. Unlike
  the circular case, each angle's predicted echo times use that angle's
  OWN measured r_outer(theta)/r_inner(theta) (polar-resampled from the
  real contour), not a single global constant -- real anatomy has no
  single "R_outer". Compute: 72 forward simulations, ~15-20 min
  estimate given upfront, completed within it. Outer boundary max
  radius=98.7 cells, safely inside the 120-cell probe radius.
- **Result: the core physics transfers, but reliability drops
  substantially on real, irregular anatomy.** Outer boundary detected
  at only 9/36 (25%) angles -- far below the circular phantom's
  near-100% -- with a real -4.71 cell radius bias where detected (vs.
  near-zero bias on the circle). Inner boundary detected at 11/36
  (31%). **This drop is consistent with everything else this whole
  project (and `jwave_test` before it) has found: real, irregular,
  off-center anatomy is fundamentally harder to localize than an
  idealized centered circle, for the same reasons already diagnosed
  throughout this investigation (off-radial specular points, local
  curvature variation) -- not a new or surprising finding, a
  confirming one.**
- **A genuine cross-validation, not just a repeat**: the inner/outer
  amplitude ratio on real anatomy (0.195) landed close to the circular
  phantom's measured ratio (0.235, run -08) -- both notably ABOVE the
  naive coefficient-only theoretical prediction (0.050) by a similar
  ~4-5x factor. The SAME discrepancy appearing independently in both
  the idealized circle and real irregular anatomy is real evidence it
  reflects genuine physics (most likely the curvature-dependent
  reflection mechanism flagged in run -20/-22, echoing `jwave_test`'s
  own run -44 finding), not a coincidence specific to one geometry.
- **Honest gap, not glossed over**: the amplitude-strata veto (runs
  -21/-22) was NOT applied to these real-anatomy matches. With only
  11/36 inner detections on an irregular boundary (versus a
  rotationally-symmetric circle where misattribution is much less
  likely), some fraction could be genuine weak echoes and some could
  be misattributed artifacts -- this validation checks detection RATE
  and amplitude SCALE, not yet per-match amplitude-consistency.
- Physical sanity checked? by whom?: Claude -- checked the outer-
  boundary max radius against the probe radius before trusting the
  simulation; compared the real-anatomy amplitude ratio against BOTH
  the circular phantom's measured value AND the pure coefficient
  prediction, rather than only checking one reference point, to
  properly characterize the discrepancy as reproducible rather than
  circderived from a single case.
- Gate passed? (Y/N): N/A -- real-anatomy validation test.
- Next action: apply the amplitude-strata veto to patient023's matches
  specifically (checking whether the 11 "inner" detections survive an
  amplitude-consistency check, analogous to run -21); the -4.71 cell
  outer-boundary bias on real anatomy is itself worth diagnosing
  (likely related to local curvature/off-radial effects at the specific
  angles where detection succeeded); patient001 (the milder real
  anatomy case) not yet tested with this same reflection pipeline for
  comparison. Not committed/pushed (standing instruction).

### Run 2026-07-09-24 — Amplitude-strata veto applied to patient023's real-anatomy matches: all 11 inner detections SURVIVE, strengthening (not undermining) run -23's validation
- Phase: 1/2. Per user: "applying the strata veto to patient023's
  specific matches." Extended `phase1_patient023_validation.py` to
  calibrate the coefficient-predicted inner_k0 amplitude to
  patient023's OWN measured outer baseline (median observed, same
  methodology as run -21), and veto any inner match whose amplitude
  exceeds 10x that prediction. Compute: 72 forward simulations, ~15-20
  min estimate given upfront, completed within it.
- **Result: all 11/11 inner matches SURVIVE the veto -- a clean,
  positive confirmation, not a correction.** Calibrated outer baseline
  (median): 0.00785; predicted inner_k0 amplitude: 0.000395; veto
  threshold: 0.003949. Every one of the 11 detected inner amplitudes
  (range 0.00099-0.00279) fell comfortably within this range. This
  strengthens run -23's real-anatomy validation: the detected inner-
  boundary echoes are amplitude-consistent with genuine order-1 direct
  reflections, not misattributed artifacts -- the same veto that
  decisively eliminated 8 spurious "inner_k2" matches on the circular
  phantom (run -21) found nothing to reject here.
- Physical sanity checked? by whom?: Claude -- calibrated the
  prediction to THIS patient's own measured baseline (not reusing the
  circular phantom's calibration), consistent with the standing rule
  that calibration doesn't transfer across geometries/setups without
  re-measurement.
- Gate passed? (Y/N): N/A -- validation test, positive result.
- Next action: the -4.71 cell outer-boundary radius bias on real
  anatomy remains undiagnosed; patient001 (the milder real-anatomy
  case) not yet tested with this same reflection pipeline for
  comparison; the vetoed circular-phantom peaks' true identity (run -22)
  remains unsolved. Committed locally this session (not pushed,
  standing instruction).

### Run 2026-07-09-25 — Patient023 fused reconstruction (all mechanisms): outer excellent, inner FAILS spatial validation despite passing every consistency check
- Phase: 1/2. Per user: "now i want the 023 scan that looks like
  phase1_fused_channel_reconstruction.png," then, after an initial
  naive-blind attempt was rejected: "make sure you incoporated all
  mechanisms we discovered." Built
  `phase1_patient023_fused_reconstruction.py`: TRANSMISSION channel
  (full multistatic + SIRT, runs -04/-06, unchanged) fused with a
  REFLECTION channel using a genuinely comprehensive BLIND per-trace
  classifier -- matched filtering (runs -14+), off-axis-outer-family
  exclusion calibrated LOCALLY per trace (generalizing runs -16/-17's
  circular geometry to real irregular anatomy via a local-radius
  approximation), and the coefficient-derived amplitude-strata veto
  (runs -21/-22/-24) applied in sequence, with NO true-contour
  information used anywhere in the classification. Compute: 144 total
  forward simulations (72 transmission + 72 reflection), ~30-40 min
  estimate given upfront, completed within it.
- **Result: a genuinely mixed outcome, not a clean win — reported
  precisely rather than only citing the high detection-rate numbers.**
  Outer boundary: 36/36 (100%) detected, and VISUALLY EXCELLENT --
  both the transmission SIRT background and the reflection-derived
  points track patient023's true irregular epicardial contour closely
  all the way around (see
  `results/figures/phase1_patient023_fused_reconstruction.png`).
  Inner boundary: 31/36 (86%) passed the full classifier chain
  (matched filter + off-axis exclusion + strata veto) -- a much higher
  count than run -23/-24's stricter known-time-matched validation
  (11/36) -- but **these points do NOT trace the true inner contour at
  all; they cluster systematically well inside it, toward the
  center.** This is an important, decisive finding: passing the
  amplitude-strata veto is NECESSARY but NOT SUFFICIENT to guarantee a
  genuine inner-boundary echo. Something is producing peaks with
  plausible order-1 amplitude that do not spatially correspond to the
  real anatomical inner boundary.
- **Connects directly to run -22's still-unsolved mystery**: the
  circular-phantom's vetoed peaks (amplitude ~7.5e-4, ~3x weaker than
  genuine inner_k0, ruled out as reverberation but never identified)
  are the most likely same-category culprit here -- a real, systematic
  mechanism (plausibly the un-modeled bistatic pitch-catch offset
  geometry, flagged as the leading untested candidate since run -15)
  that produces amplitude-plausible-but-spatially-wrong echoes,
  now shown to survive the FULL validated classifier chain, not just
  a simple amplitude check.
- Physical sanity checked? by whom?: Claude -- did not report the
  31/36 figure as a validated result without checking the actual
  SPATIAL positions against the true contour; caught the systematic
  inward clustering visually before treating the high pass-rate as
  progress, and explicitly connected it back to the still-open run -22
  mystery rather than treating it as a new, unrelated problem.
- Gate passed? (Y/N): N/A -- integrated validation test, mixed result.
- Next action: the inner-boundary detection problem is NOT solved by
  combining the established mechanisms -- a genuinely new investigation
  is needed into what specifically produces these amplitude-plausible,
  spatially-wrong peaks, most plausibly starting with the bistatic
  pitch-catch geometry correction flagged repeatedly since run -15 but
  never built. The outer-boundary result is solid and could reasonably
  be reported as a real positive on its own. Committed locally this
  session (not pushed, standing instruction).

### Run 2026-07-09-26 — Raw peak scatter reveals a flat, persistent noise floor -- likely explanation for run -25's spatially-wrong "inner" detections
- Phase: 1/2. Per user: "do a quick xy plot for incoming amplitude and
  time for every signal." Built `phase1_patient023_raw_peak_scatter.py`:
  plots every detected (time, amplitude) peak across all 36 angles on
  patient023's real anatomy, reflection channel, with NO classification
  applied -- the raw population. Compute: 72 forward simulations
  (reflection only), ~15-20 min estimate given upfront, completed
  within it. 306 total peaks found.
- **Result: a clear, expected strong cluster (the direct outer echo,
  3-5us, amplitude 0.01-0.03) decays rapidly, then settles onto a FLAT,
  roughly constant floor (~0.001-0.003) that persists essentially
  unchanged from ~9us all the way to 20us, across every angle** (see
  `results/figures/phase1_patient023_raw_peak_scatter.png`). This does
  NOT look like decaying physical echo structure from a specific
  geometric mechanism -- it looks like a generic reverberation/
  numerical noise floor that happens to sit in the same amplitude range
  the strata veto expects for a genuine weak inner echo. **This
  revises the leading hypothesis for run -25's spatially-wrong "inner"
  detections**: rather than a specific unmodeled geometric effect (the
  bistatic pitch-catch offset), a persistent noise/reverberation floor
  that coincidentally overlaps the expected inner-echo amplitude scale
  is now at least as plausible, and simpler.
- Physical sanity checked? by whom?: Claude -- plotted the FULL raw
  population (not just classified/surviving peaks) specifically to
  look for structure vs. noise-floor characteristics, rather than only
  inspecting summary statistics.
- Gate passed? (Y/N): N/A -- diagnostic visualization.
- Next action (identified but not yet run, per user's next instruction
  redirecting to a different test): confirm the floor's origin by
  checking whether the water-only control shows the same flat floor at
  a similar level (would confirm numerical/reverberation origin,
  unrelated to real tissue) -- not yet done.

### Run 2026-07-09-27 — BLIND off-center concave heart (exact jwave_test replica): RMSE=1.30mm, modestly better than jwave_test's 1.54-1.67mm, and VISUALLY tracks both the concave notch and sharp tip cleanly -- no ghost-artifact corruption
- Phase: 1/2. Per user: "try a blinded off center heart shape for me,
  to test that this ray-theory approach can bypass our last project
  failure mode." Built `phase1_offcenter_heart_blind_test.py`: an EXACT
  replication of `jwave_test`'s off-center concave heart phantom (same
  10-vertex polygon, same OFFSET=(10,-15), same HEART_R=50 -- the exact
  shape that broke sparse-probe blind reconstruction in runs -70/-72/-73),
  built in this project's water-bath domain, single-tissue (myocardium
  in water, matching jwave_test's own single-boundary test convention
  for a fair comparison). Tested BOTH established channels, fully BLIND
  (no true-contour information used): transmission (full multistatic +
  SIRT) and reflection (pitch-catch + matched filter + strata veto,
  36-probe dense coverage instead of jwave_test's sparse 4/8/16).
  Compute: 144 forward simulations (72 transmission + 72 reflection),
  ~30-40 min estimate given upfront, completed within it.
- **Result: a real, modest quantitative improvement, and a much
  clearer QUALITATIVE one.** RMSE=1.3047mm across all 36/36 detected
  angles -- vs. jwave_test's 8-probe RMSE=1.544mm (run -72) and
  16-probe RMSE=1.674mm (run -73, WORSE than 8-probe -- the decisive
  finding that more sparse probes did not help irregular anatomy). A
  ~15-22% RMSE reduction is real but not dramatic. **Visually, the
  improvement is much clearer**: the reflection-derived boundary (see
  `results/figures/phase1_offcenter_heart_blind_test.png`) tracks the
  true heart shape at BOTH the concave notch and the sharp convex tip
  reasonably well, with NO ghost-spike/chaotic-artifact corruption --
  a recognizable heart shape, not the chaotic reconstructions
  jwave_test's sparse-probe approach produced at 8 or 16 probes.
- **Scope, stated precisely, not overclaimed**: this is the
  single-tissue case only (myocardium in water, no LV/blood cavity) --
  the harder two-tissue confound found in run -25 (spatially-wrong
  "inner" detections surviving every established check) has NOT been
  re-tested on this irregular shape. For the SPECIFIC question asked
  (does dense-ring coverage + established signal-processing mechanisms
  bypass the ghost-cone corruption that broke sparse-probe blind
  reconstruction), the answer is a real, qualified YES -- for the
  single-boundary case.
- Physical sanity checked? by whom?: Claude -- used the EXACT same
  phantom geometry, offset, and scale as jwave_test's own runs (not an
  approximately-similar shape) for a valid apples-to-apples comparison;
  reported both the modest quantitative RMSE improvement AND the more
  meaningful qualitative (no ghost-artifact) difference, rather than
  only citing whichever number looked better.
- Gate passed? (Y/N): N/A -- direct comparative validation test,
  positive result.
- Next action: re-test the two-tissue (myocardium+blood) version of
  this SAME off-center concave heart shape, to see whether the
  spatially-wrong inner-detection problem (run -25) also appears on
  irregular anatomy, not just patient023's real contour; confirm the
  noise-floor hypothesis (run -26) via the water-only control check.
  Not committed/pushed (standing instruction).

### Run 2026-07-09-28 — Bent-ray (eikonal) correction: Dijkstra-graph attempt fails (-230%), then fixed with scikit-fmm (+48.6%)
- Phase: 1 (methodology/mechanism validation, transmission channel).
  Prompted directly by the user's question: "this should be simple
  physics... how do you infer a medium's unknown properties just by
  looking at its ultrasonic residuals" — i.e., resolve the straight-ray
  (no-refraction) approximation used everywhere so far, WITHOUT
  circularly assuming the medium is already known. Method used
  (standard bent-ray/iterative travel-time tomography): take the
  already-built straight-ray SIRT reconstruction as a real, data-derived
  (not assumed) estimate of the sound-speed field, then solve for the
  TRUE travel time through that estimated field via Fermat's principle,
  and compare to the real observed data. No new jWave simulation —
  reuses `results/patient001_single_tissue_rays.npz` throughout (pure
  post-processing, cheap to iterate).
- Attempt 1 (`scipy.sparse.csgraph.dijkstra` on an 8-connected grid
  graph, edge weight = distance x avg slowness): **-230.2%** ("improvement"
  is negative — bent-ray prediction 3.3x WORSE than straight-ray;
  straight-ray RMS=163.9ns vs. bent-ray RMS=601.1ns). Leading hypothesis:
  SIRT's known star-artifact (isolated spurious "tissue" pixels)
  contaminating the sound-speed field.
- Diagnostic (per user: "proceed"): added connected-component cleaning
  (`scipy.ndimage.label`, keep only the largest connected "tissue"
  component, discard the rest). Found 60 components, discarded 59
  islands (5011px kept out of a larger raw threshold). Result:
  **-227.8%** (RMS=618.3ns) — essentially unchanged. This RULES OUT the
  star-artifact hypothesis as the (sole) cause.
- Re-diagnosis: naive 8-connected graph-Dijkstra is a known-bad
  stand-in for the eikonal equation — it can find unphysical
  "shortcuts" through a fast (water) region that no real, continuously-
  refracting wavefront could actually take, because it's restricted to
  8 fixed bend directions per grid step with no true angular/continuity
  constraint. This is a discretization artifact, not a data or
  star-artifact problem.
- Fix (per user: "go ahead"): installed `scikit-fmm` (v2025.6.23) and
  replaced the Dijkstra call with `skfmm.travel_time(phi, sound_speed,
  dx=cell_size_m)` — a proper upwind fast-marching solver for the actual
  eikonal equation `|grad T| = 1/c(x)`. Everything else (SIRT
  reconstruction, component cleaning, sound-speed field construction,
  comparison against real observed data) unchanged.
- Result: **+48.6% improvement** — straight-ray mean error=158.865ns
  (RMS=170.280ns) vs. bent-ray mean error=81.584ns (RMS=96.430ns).
  Confirmed visually (`results/figures/phase1_bent_ray_correction.png`):
  scatter of per-ray error (straight vs. bent) shows the majority of
  points below the "no improvement" diagonal.
- Physical sanity checked? by whom?: Claude, via (a) a real physical
  mechanism (Fermat/eikonal solve through a data-derived field, not a
  circular medium assumption), (b) a decisive ablation (component
  cleaning) that isolated the true cause of the initial failure, and
  (c) both a numeric and visual confirmation of the fix. NOT yet
  collaborator-reviewed (Gate 2 for this project is still open).
- Gate passed? (Y/N): N/A — this is a methodology-validation result
  within the still-open Phase 1, not a phase-gate item itself.
- Observations: this closes out, with a genuine physics-based positive
  result, the "why is straight-ray so inaccurate for simple physics"
  question the user raised right before this — the bent thing IS
  fixable, and fixing it does NOT require already knowing the medium,
  because the SIRT reconstruction itself supplies the (imperfect but
  real) starting estimate that the eikonal solve refines against.
- Next action: none yet specified by the user. Natural candidates
  flagged but NOT started: (a) fold this bent-ray correction into an
  iterative refine loop (re-run SIRT using bent-ray predicted times,
  re-solve eikonal, repeat); (b) re-test on the off-center heart phantom
  (run -27) or patient023 to see if the improvement holds on harder/
  irregular anatomy; (c) the earlier-flagged Doppler/motion-channel
  question (whether a proxy blood/myocardium motion field is needed for
  blood-vs-myocardium discrimination, which would require a live 3D
  cardiac model). Not committed/pushed (standing instruction).

### Run 2026-07-09-29 — Bent-ray correction retested on harder anatomy: off-center concave heart (+55.1%) and real patient023 (+63.1%) — BOTH beat patient001's original +48.6%
- Phase: 1 (methodology validation, transmission channel). Per user
  direction ("off-center heart or patient023") after run -28's
  patient001 result — retested the same straight-ray-vs-bent-ray
  (scikit-fmm eikonal) methodology on two harder cases, to check whether
  the fix generalizes or was specific to the easy centered-circle case.
  Refactored `phase1_bent_ray_correction.py`'s core into a reusable
  `evaluate_bent_ray_correction()` function (SIRT rebuild -> star-
  artifact cleaning -> single-guessed-tissue-speed field -> scikit-fmm
  solve -> straight vs. bent comparison against real simulated data),
  callable from new per-phantom driver scripts. Neither phantom had
  cached transmission-channel data, so both required a FRESH 72-sim
  (36 angles x 2 media) transmission-only capture
  (`phase1_offcenter_heart_bent_ray.py`, `phase1_patient023_bent_ray.py`)
  — no reflection-channel work repeated. Compute estimate (~15-20 min
  each, both run in parallel) given upfront per standing instruction.
- **Caught and fixed a path bug before trusting results**: both driver
  scripts were first launched with cwd set to `src/` (matching how the
  session had been invoking other one-off scripts), but this project's
  scripts assume cwd = `jwave_ring_tomography/` (relative paths like
  `../jwave_test/results/...` and `results/...`). Patient023's run
  failed loudly (`FileNotFoundError` on the MRI npz) and was caught
  immediately; the off-center heart run hadn't yet failed but would
  have silently written its outputs to the wrong directory, so it was
  stopped and restarted with the correct working directory before
  either result was trusted.
- Off-center concave heart (exact run -27 phantom: HEART_R=50, offset
  (10,-15), single-tissue myocardium-in-water): straight-ray RMS=
  176.958ns, bent-ray RMS=87.863ns, **+55.1% improvement**. Recovered
  sound-speed field visually resembles the true heart silhouette (notch
  and tip both visible) despite the off-center placement.
- Patient023 (real irregular anatomy, two-tissue myocardium+blood, ~45%
  contraction — same patient as this project's reflection-channel
  validation, runs -21/-26, and jwave_test's hardest real case):
  straight-ray RMS=194.975ns, bent-ray RMS=79.526ns, **+63.1%
  improvement**. Uses a single guessed tissue sound speed (MYOCARDIUM's)
  for the whole "tissue" region since the straight-ray SIRT image alone
  can't distinguish blood from myocardium — same simplification as the
  other two tests, so all three results are a fair apples-to-apples
  comparison of the SAME simplified method.
- Both retests beat patient001's original +48.6%, and both scatter
  plots show the same qualitative pattern confirmed in run -28 (majority
  of points below the "no improvement" diagonal) — i.e., this isn't a
  fluke specific to the easy centered-circle geometry; it holds up on an
  off-center concave shape AND on real, irregular, two-tissue anatomy.
- Physical sanity checked? by whom?: Claude — via the same reasoning as
  run -28 (real Fermat/eikonal solve through a data-derived field) plus
  a direct visual check of both recovered sound-speed fields and both
  scatter plots. NOT yet collaborator-reviewed (Gate 2 for this project
  is still open).
- Gate passed? (Y/N): N/A — methodology validation within the still-open
  Phase 1, not a phase-gate item itself.
- Observations: the bent-ray/eikonal correction is now validated on
  three phantoms of increasing difficulty (centered circle, off-center
  concave shape, real irregular two-tissue anatomy), with the
  improvement percentage actually GROWING (not shrinking) as anatomy
  gets harder (+48.6% -> +55.1% -> +63.1%). This is the opposite pattern
  from `jwave_test`'s closed investigation (where accuracy degraded as
  anatomy got harder/more irregular) — a meaningfully positive signal
  for this project's fundamentally different (dense-ring, transmission-
  based) approach.
- Next action: none yet specified by the user. Same candidates as run
  -28 remain open (iterative refine loop; re-derive an actual boundary
  estimate FROM the bent-ray-corrected field rather than only measuring
  travel-time-prediction error; the Doppler/motion-channel question).
  Not committed/pushed (standing instruction).

### Run 2026-07-09-30 — Iterative bent-ray refinement + blind boundary extraction: data-fit converges cleanly, but the derived boundary is WORSE than the reflection channel and even worse than jwave_test's rejected sparse-probe benchmark — a genuine, informative negative on image-based extraction, not a clean win
- Phase: 1 (methodology validation). Per user: fold the bent-ray
  correction (runs -28/-29) into an iterative refinement loop, derive an
  actual boundary estimate from the corrected field (not just travel-
  time-prediction error), and evaluate a detailed physics framing they
  laid out: sequential acquisition solves "which transmitter produced
  this waveform" but NOT "which surface/path produced this peak" — a
  single pitch-catch firing can still return on-axis outer, off-axis
  outer, inner, concavity, and reverberation echoes all in one waveform,
  and the reflection channel's existing mitigations (amplitude-strata
  veto, off-axis-outer exclusion) only partially address this.
- Implementation, no new jWave simulation (reuses
  `results/offcenter_heart_rays.npz` from run -29): added
  `iterative_bent_ray_refinement()` to `phase1_bent_ray_correction.py`
  — at each outer iteration, solves the eikonal equation through the
  CURRENT sound-speed estimate (scikit-fmm, exact forward model for
  that estimate), computes the residual between observed and bent-ray-
  predicted arrival time, and backprojects that residual via straight-
  ray SIRT to update the image. **Limitation stated up front in the
  code, not discovered after the fact**: the correct update would
  backproject each ray's residual along its ACTUAL bent path (a
  Frechet-derivative/adjoint-state update, the standard approach in
  real iterative travel-time tomography); scikit-fmm gives travel TIMES
  only, not explicit ray paths, so straight-ray backprojection is used
  as a practical stand-in. Also added `extract_boundary_radius_per_angle()`
  — BLIND per-angle ray-casting into the reconstructed tissue mask, using
  the SAME "origin known, shape blind" convention as this project's
  established reflection-channel RMSE metric (run -27), so the two
  channels' numbers are directly, fairly comparable.
- **Data-fit result: converges cleanly.** Bent-ray data-residual RMS:
  87.863ns (iter 1, matching run -29's single-shot result exactly — a
  good consistency check) -> 72.314ns (iter 2) -> 69.564ns (iter 3) ->
  69.493ns (iter 4, plateaued) — a genuine further ~21% reduction beyond
  the single-shot correction.
- **Boundary result: a real but small improvement, and still clearly
  worse than the alternative channel.** Straight-ray SIRT boundary
  RMSE=3.9559mm (iteration 0) -> iterative bent-ray boundary RMSE=
  3.5502mm (after 4 outer iters) — about a 10% improvement, genuine but
  modest. **Both numbers are substantially worse than this project's own
  REFLECTION channel on the IDENTICAL phantom (run -27: RMSE=1.3047mm),
  and worse even than `jwave_test`'s own already-REJECTED sparse-probe
  benchmark on this same shape (1.544mm/1.674mm, runs -72/-73)** — the
  transmission-channel image-based boundary is currently the WORST of
  the three numbers available for this exact phantom, not a new best.
- **Visual finding, not glossed over**: the iteration-4 tissue mask
  (`results/figures/phase1_offcenter_heart_iterative_refinement.png`) is
  visibly FRAGMENTED and noisy/speckled compared to iteration-0's clean
  solid blob, despite sitting close to the true heart outline. This is
  consistent with the stated limitation: straight-ray backprojection of
  a residual re-smears correction along the WRONG (unbent) path,
  plausibly reintroducing streak-type artifacts with each additive
  update rather than cleanly sharpening the boundary.
- **Diagnosis for why transmission-image extraction underperforms
  reflection-timing extraction — a resolution bottleneck, not the same
  ambiguity problem**: image-based (tomographic) boundary detection is
  fundamentally limited by the sparse view count (36 angles) and 2D
  pixel-grid discretization of the reconstructed image; reflection's
  per-angle timing instead gives ONE direct, high-precision distance
  measurement per angle, with no image-reconstruction blur in between.
  This is a genuinely different limitation from the reflection channel's
  path-ambiguity problem, not a restatement of it.
- **Direct answer to the user's physics framing, addressed explicitly,
  not just implicitly through the numbers**: "which transmitter" vs.
  "which path" is the right distinction, and sequential acquisition
  only solves the first. For the REFLECTION channel specifically, the
  ambiguity is real and UNSOLVED by anything built in this project so
  far — the amplitude-strata veto and off-axis-outer exclusion (runs
  -18/-19/-21/-22) rule out some candidate families (implausible
  amplitude, geometrically-unreachable angle) but do not implement a
  real specular-geometry/peak-identity model (transmit/receive
  directivity, reflection-law-matched candidate surface points,
  concavity multi-candidate handling) — this is the likely cause of run
  -27's reflection boundary being "too smooth and displaced... especially
  where the true contour is concave/off-center," exactly as diagnosed.
  The TRANSMISSION/bent-ray channel tested this run is structurally
  DIFFERENT and does NOT have this same ambiguity: each tx/rx pair's
  signal is governed by exactly one physical quantity (the Fermat
  least-time path), computed exactly by the eikonal solve, not a family
  of competing specular candidates. Its own analogous simplification is
  diffraction/multipath (a real wavefront can split around a concave
  notch and arrive via more than one near-equal-time path; first-
  arrival-only FMM captures just the fastest) — a real, different, and
  NOT YET TESTED effect, not the same problem restated.
- Physical sanity checked? by whom?: Claude — stated the straight-ray-
  backprojection-as-update limitation in the code BEFORE running, not
  after seeing a disappointing result; reported the boundary-RMSE
  comparison honestly even though it is unfavorable to this run's own
  method, rather than only reporting the (favorable-looking) data-fit
  convergence number.
- Gate passed? (Y/N): N/A — methodology validation within the still-open
  Phase 1, not a phase-gate item itself.
- Next action: the reflection channel's real physical gap (peak-to-
  surface-point identity: on-axis vs. off-axis vs. inner vs. concavity
  vs. reverberation, all within one firing) remains the more
  consequential unsolved problem and is NOT addressed by anything built
  this run — building a real specular-geometry/peak-identity model
  (directivity, reflection-law-matched candidate points, concavity
  handling) is the larger, not-yet-started undertaking the user's
  framing correctly points toward. For the transmission channel
  specifically, if pursued further: (a) a full bent-ray/adjoint-state
  update (backprojecting along the ACTUAL bent path, not a straight-
  line stand-in) instead of the current approximation; (b) more view
  angles or an explicit regularizer (flagged since runs -06/-15,
  never implemented); (c) direct testing of the diffraction/multipath
  simplification identified above (not yet done). The Doppler/motion-
  channel question (runs earlier this session) also remains open. Not
  committed/pushed (standing instruction).

### Run 2026-07-09-31 — DAS (delay-and-sum) reflectivity imaging REPLACES the single-peak reflection detector: RMSE drops from 1.30mm to 0.68mm (straight-ray) to 0.15mm (bent-ray) — the biggest single accuracy jump in this project's history, and a direct, principled answer to the peak-identity problem
- Phase: 1 (methodology, reflection channel). Per user: "do all of them
  but whats most impactful first," ranking the open items from run -30
  (reflection channel's peak-to-surface-point identity gap; full
  bent-ray backprojection; more view angles; diffraction/multipath
  test). Argued DAS reflectivity imaging is the highest-impact item
  because it doesn't patch the single-peak classifier — it REPLACES it
  with a method that sidesteps the identity question entirely: for
  EVERY candidate pixel and EVERY one of the 36 pitch-catch firings,
  compute the predicted bistatic (src->pixel->rcv) round-trip delay and
  sample that shot's matched-filter envelope there, accumulating across
  all firings. A true reflector is consistent with genuine echoes from
  MANY different firing angles (including off-axis ones) at once and
  builds up strong signal; a non-reflector point only matches a few
  angles by chance and stays low. This is the standard synthetic-
  aperture-ultrasound/seismic-migration answer to exactly the ambiguity
  the user described, and it naturally folds in directivity, off-axis
  echo families, and concavity handling WITHOUT an explicit per-shot
  classifier, unlike every reflection-channel method built in this
  project since run -08.
- Design: `phase1_das_reflectivity_imaging.py` (straight-ray, homogeneous
  water travel-time model). Required a FRESH pitch-catch simulation on
  the off-center concave heart phantom (this project had never saved raw
  reflection traces for this specific phantom before) — 72 forward sims,
  ~15-20 min estimate given upfront, completed within it. Saved raw
  traces to `results/offcenter_heart_reflection_raw_traces.npz` for
  reuse. Background-subtracted (phantom minus water-only matched-filter
  envelope, clipped at 0) before accumulation, consistent with this
  project's established "excess signal" convention. Boundary extracted
  BLIND per-angle as the radius of PEAK accumulated intensity (not a
  threshold crossing, since a reflectivity image's true boundary should
  be the dominant peak) — same "origin known, shape blind" convention as
  run -27's benchmark, for a fair comparison.
- **Straight-ray DAS result: RMSE=0.6809mm** — essentially a 2x
  improvement over this project's own established single-peak
  reflection method (run -27: RMSE=1.3047mm) on the IDENTICAL phantom.
  Visually
  (`results/figures/phase1_das_reflectivity_imaging_straight_ray.png`):
  the heart silhouette, INCLUDING the concave notch, glows clearly
  against a background of circular sidelobe arcs (the expected,
  well-known synthetic-aperture artifact from a finite/sparse aperture
  — not a bug), and the extracted boundary tracks the notch correctly —
  direct visual confirmation that cross-angle accumulation handles
  concavity without needing an explicit specular-geometry classifier.
- **Bent-ray upgrade result: RMSE=0.1472mm** (`phase1_das_reflectivity_
  imaging_bent_ray.py`) — swapped the straight-ray Euclidean travel-time
  model for scikit-fmm-solved travel times through the CLEAN (non-
  fragmented) straight-ray SIRT sound-speed field from the transmission
  channel (deliberately NOT run -30's noisier iteratively-refined field,
  since that one's fragmentation would inject spurious speed variation
  into the delay prediction). No new jWave simulation — reused both the
  cached transmission rays (`results/offcenter_heart_rays.npz`) and the
  cached reflection traces from this run's straight-ray step; only 72
  additional scikit-fmm solves (one per pitch-catch src/rcv position).
  This is a **further ~4.6x improvement over straight-ray DAS, and ~9x
  improvement over the original single-peak method** — the single
  largest accuracy jump recorded in this project's history for any one
  change. This operationalizes "fold the bent-ray correction into the
  reflection channel," a natural convergence of this session's two
  separate threads (transmission-channel bent-ray correction, runs
  -28/-29/-30, and reflection-channel peak-identity, this run).
- **View-angle-count sensitivity** (`phase1_das_view_angle_sensitivity.py`,
  no new simulation — subsampled the already-simulated 36-angle dataset
  down to 18/12/9 angles): RMSE degrades sharply and roughly
  monotonically as angle count drops — 36 angles: 0.68mm, 18: 2.03mm,
  12: 2.48mm, 9: 1.96mm (minor non-monotonicity at the lowest counts,
  likely specific-subsample geometry, not a real reversal). **DAS needs
  roughly 27+ angles to match run -27's old 1.30mm benchmark** — more
  view angles is confirmed as a real, quantified lever (previously only
  flagged qualitatively since runs -06/-15), not just a plausible guess.
- **Diffraction/multipath, qualitative check only (not a rigorous test)**:
  visual inspection of the bent-ray DAS image shows the concave notch
  resolved just as sharply and with boundary points just as tight as the
  convex regions — no visible evidence of a diffraction-related dropout
  at the concave feature specifically, in this toy phantom at this
  resolution. This is an observational check, NOT a validated absence-
  of-diffraction result — no explicit multi-arrival/diffraction model
  was built or tested; a rigorous test remains open.
- Physical sanity checked? by whom?: Claude — grounded the method in a
  standard, well-established imaging technique (delay-and-sum/Kirchhoff
  migration) rather than an ad-hoc heuristic; named the sidelobe arcs in
  the DAS image explicitly as a known aperture artifact rather than an
  unexplained residual; used the CLEAN (not the fragmented, run -30)
  sound-speed field for the bent-ray upgrade with the reason stated
  before running, not after seeing a result.
- Gate passed? (Y/N): N/A — methodology validation within the still-open
  Phase 1, not a phase-gate item itself.
- Observations: this is the best boundary-accuracy result of any method
  in this ENTIRE project (0.15mm, vs. the previous best of 0.68mm
  straight-ray DAS, vs. 1.30mm single-peak reflection, vs. 3.55mm
  transmission-channel bent-ray image extraction, vs. jwave_test's
  rejected 1.54-1.67mm sparse-probe benchmark) — and it was obtained by
  replacing the peak-CLASSIFICATION approach with a peak-AGGREGATION
  approach, exactly matching the user's own diagnosis that the deeper
  problem was never really solvable per-shot without more structure.
  Both the DAS method itself and the confirmed view-angle sensitivity
  are strong evidence that THIS project's dense-ring geometry (as
  opposed to `jwave_test`'s sparse arc) is what makes this kind of
  imaging viable at all.
- Next action: (a) full bent-ray/adjoint-state backprojection for the
  TRANSMISSION channel (run -30's still-open item) is now lower priority
  relative to this result, but still open; (b) a rigorous diffraction/
  multipath test (not just the qualitative notch-sharpness check done
  here) remains undone; (c) re-test DAS (both straight-ray and bent-ray)
  on patient023's real anatomy, to confirm this result generalizes
  beyond the synthetic off-center heart phantom, the same escalation
  pattern used for the bent-ray correction itself (runs -28 -> -29); (d)
  the Doppler/motion-channel question remains open. Not committed/pushed
  (standing instruction).

### Run 2026-07-09-32 — DAS on patient023 real anatomy: OUTER boundary reaches this project's best-ever real-anatomy accuracy (1.04mm) after catching a real extraction bug, but the INNER (blood/myocardium) wall is STILL not genuinely detected — the apparent "36/36" was a false positive, confirmed by near-zero correlation with the true contour
- Phase: 1 (methodology, reflection channel). Per user: "go ahead. see
  if this time inner wall is good" — direct escalation of run -31's DAS
  result to patient023's real, irregular, two-tissue anatomy, testing
  head-on whether cross-angle accumulation (many weak-but-consistent
  echoes reinforcing each other) can succeed where every single-peak
  method in this project (runs -07 through -26) has struggled: the
  inner (blood/myocardium) boundary's genuinely weak ~0.25% reflection
  coefficient (run -21's coefficient-derived strata).
- Design: `phase1_das_patient023.py`. Required a FRESH pitch-catch
  simulation (patient023 had never had raw reflection traces saved in
  this project before) — 72 forward sims, ~15-20 min estimate given
  upfront, completed within it. Saved raw traces to `results/
  patient023_reflection_raw_traces.npz`. Built BOTH straight-ray and
  bent-ray DAS images (bent-ray reused the ALREADY-cached transmission
  data, `results/patient023_transmission_rays.npz` from run -29, no new
  transmission sim needed). Extended the single-boundary extraction
  (run -31) to a genuinely blind DUAL-boundary search: find all local
  radial-intensity maxima per angle, take the outermost sufficiently-
  prominent one as "outer," and the strongest peak inward of that (if
  any clears a minimum prominence threshold) as "inner."
- **First result, BEFORE the bug was caught: outer RMSE=3.71-3.82mm,
  inner "detected" 36/36 with RMSE~3.0mm.** Both numbers looked
  mediocre-to-reasonable at first glance, but were WRONG in a specific,
  catchable way.
- **Bug caught by inspecting the actual figure, not just the printed
  numbers** (`results/figures/phase1_das_patient023.png`): the "DAS
  inner" markers sat almost exactly ON the TRUE OUTER contour, while
  "DAS outer" markers were scattered out in the visible near-probe
  sidelobe halo (the same rosette/arc artifact family as run -31's
  off-center heart image, just closer to the probe ring here). The
  "outermost peak = outer wall" heuristic had grabbed a near-probe
  self-artifact as "outer," which pushed the TRUE outer-wall peak down
  to the "next peak inward" slot and mislabeled it "inner." Fixed
  (`phase1_das_patient023_boundary_fix.py`, no new simulation --
  reused the cached raw traces) by constraining the radial search to
  `r_max_cells=108` (0.9x the 120-cell probe radius, a generic physical
  constraint, NOT informed by patient023's own specific true contour --
  near-probe self-artifacts cluster closest to the ring itself).
- **Fixed result: outer boundary reaches this project's best-ever
  real-anatomy accuracy.** Straight-ray DAS: outer RMSE=1.0425mm (was
  3.71mm before the fix); bent-ray DAS: outer RMSE=1.2107mm (was 3.82mm
  before the fix; the bent-ray upgrade did not help here, unlike the
  synthetic heart test — plausibly because the sound-speed field used
  is still a single-guessed-tissue-speed binary approximation that
  doesn't capture the blood cavity's own distinct speed, diluting the
  benefit relative to the off-center heart's genuinely single-tissue
  case).
- **Inner boundary: the apparent "36/36 detected" is a FALSE POSITIVE,
  confirmed quantitatively, not just suspected.** Checked directly
  (correlation between the DAS-detected inner radius and the true inner
  radius, across all 36 angles): **correlation = 0.25** (near-noise).
  The true inner contour is nearly circular (radius range 51.6-65.9
  cells, std=4.6 cells); the DAS "inner" detections range wildly from
  23.5 to 97 cells (std=25.6 cells, 5.6x more scattered than the truth
  itself) and visually cluster mostly near the domain's CENTER rather
  than tracing the true inner contour's actual shape at all (see
  `results/figures/phase1_das_patient023_fixed.png`). The extraction
  heuristic's "second peak inward, if prominent enough" is picking up
  interference/crossing artifacts from the DAS accumulator's own
  central region (where many bistatic ellipses from different angles
  cross), not genuine weak inner-wall echo. **Honest, direct answer to
  the user's question: no, the inner wall is not good this time either
  — DAS's cross-angle SNR gain was not enough to pull a ~0.25%
  reflection-coefficient signal above the noise floor on real,
  irregular anatomy with this simple radial-peak extraction method.**
- Physical sanity checked? by whom?: Claude — caught the labeling bug by
  actually looking at the figure rather than trusting the printed RMSE
  numbers alone (a plausible-looking 3.0mm inner "RMSE" would have been
  reported as a genuine result if not visually inspected); after fixing
  the bug, did NOT stop at the still-plausible-looking fixed inner RMSE
  either — ran a direct correlation/scatter check specifically because
  the visual pattern (points clustered near center, not tracing a ring)
  looked suspicious, and reported the resulting near-zero correlation
  honestly rather than the more flattering RMSE-in-mm number alone.
- Gate passed? (Y/N): N/A — methodology validation within the still-open
  Phase 1, not a phase-gate item itself.
- Observations: this project now has its cleanest-ever real-anatomy
  outer-boundary number (1.04mm, DAS straight-ray, patient023) alongside
  its clearest-ever DIRECT disproof that a detection method is not
  really working for the inner wall (correlation=0.25, not just a
  vague "unreliable" characterization). The inner-wall problem remains
  exactly what it has been since run -07 — a genuine, unsolved, tissue-
  contrast-limited detectability problem, not a peak-identity/geometry
  problem DAS could route around the way it did for outer-boundary
  concavity (run -31). This is an important asymmetry: DAS fixed the
  problem that WAS about peak identity/geometry (outer boundary,
  concave shapes) but did not fix the problem that is fundamentally
  about SIGNAL STRENGTH (inner boundary, weak reflection coefficient) —
  consistent with the mechanism DAS actually offers (cross-angle
  consistency), which only helps if there is real, if weak, signal
  to accumulate, not if the signal is genuinely buried in structured
  interference at this resolution/angle-count.
- Next action: if the inner wall is to be recovered, the more promising
  levers (per this run's own diagnosis) are: (a) more view angles (run
  -31 already confirmed this matters substantially for the outer
  boundary; likely even more consequential for a signal this much
  weaker); (b) a smarter inner-boundary extraction that suppresses the
  DAS accumulator's own central-crossing interference pattern
  specifically (not yet attempted); (c) accepting the inner wall may
  need the Doppler/motion channel (this session's earlier discussion)
  rather than any refinement of the reflection/transmission channels
  tested so far. Full bent-ray/adjoint-state backprojection and the
  rigorous diffraction test (run -31's other open items) remain
  untouched. Not committed/pushed (standing instruction).

### Run 2026-07-09-33 — CHANNEL 2-alternate (backscatter/speckle): a distinct, previously-untested physical channel, confirmed to produce a clear, consistent, physically-explicable signal (3.93x elevation, positive at all 36/36 angles)
- Phase: 1 (new channel validation). Per user, after run -32's honest
  "no" on inner-wall reflection-coefficient detection: "but static echo
  can still diff myo and blood?" — correctly pointing at a gap: real
  clinical echocardiography sees the endocardial border mainly via
  VOLUME BACKSCATTER CONTRAST (myocardium's fibrous microstructure
  scatters diffusely; blood is comparatively anechoic), not primarily
  the smooth-interface reflection coefficient (~0.25%) that run -32
  showed conclusively is too weak. Every phantom built in this project
  so far (heart, patient001, patient023, the circular positive control)
  models myocardium and blood as perfectly homogeneous regions, so
  speckle CANNOT appear in any of them by construction — this is a
  genuinely untested channel, not a variant of anything tried before.
- Design: `phase1_backscatter_speckle_channel.py`. Reused the run -12
  circular positive control's exact geometry (R_outer=80, R_inner=60,
  centered) as the HOMOGENEOUS baseline (unchanged, no new work needed
  for that half), and built a new SPECKLE variant: per-grid-cell random
  sound-speed/density fluctuations (3% relative, Gaussian, independent
  per cell — a standard coarse proxy for unresolved tissue microstructure)
  within the myocardium ring only; blood left perfectly homogeneous
  (matching the real near-anechoic-blood assumption). Fixed seed=42,
  logged per this project's standing seed-discipline. Compared echo
  energy WITHIN THE MYOCARDIUM-WALL TIME WINDOW specifically (between
  predicted outer- and inner-boundary arrival times, trimmed 0.3us at
  each edge to reduce contamination from the two boundary echoes'
  own matched-filter width) — energy in this window can ONLY come from
  inside the wall's volume, never from either boundary, so it isolates
  the backscatter question directly. PREDICTION stated before running:
  homogeneous phantom's within-wall window should sit near the noise
  floor (nothing to scatter from except its two boundaries, both
  excluded); speckle phantom should show detectably elevated energy if
  volume backscatter is simulable at this grid resolution (dx=0.1mm,
  ~6 cells/wavelength at 2.5MHz). Compute: 72 forward simulations (36
  angles x speckle/homogeneous), ~15-20 min estimate given upfront,
  completed within it.
- **Result: prediction confirmed cleanly and decisively.** Mean
  within-wall RMS energy: homogeneous=0.004369, speckle=0.01715 —
  **3.93x elevation**. Confirmed visually two ways
  (`results/figures/phase1_backscatter_speckle_channel.png`): (1) the
  representative A-scan shows the speckle trace's envelope filling the
  within-wall window AND continuing with real oscillatory structure
  well beyond it (consistent with distributed volume scattering,
  decaying gradually), while the homogeneous trace decays smoothly with
  no internal content, exactly as predicted; (2) the per-angle energy
  comparison shows the speckle phantom ABOVE the homogeneous phantom at
  **all 36/36 angles**, not just on average — a consistent, physically
  robust effect, not an artifact of a few outlier angles.
- Physical sanity checked? by whom?: Claude — stated the quantitative
  prediction and its physical basis (homogeneous = noise floor, speckle
  = elevated if resolvable) before running, not after; used the SAME
  circular phantom geometry validated many times before (runs -12
  through -15) for the homogeneous half, isolating microstructure
  scattering as the only variable; checked per-angle consistency (all
  36/36), not just the aggregate ratio, before calling the result robust.
- Gate passed? (Y/N): N/A — new-channel validation test, not a phase
  gate (and this project's own Gate 2 remains open regardless).
- Observations: this is a genuinely new, physically distinct, POSITIVE
  channel result for exactly the blood/myocardium discrimination problem
  that reflection (run -32), transmission (run -07), and every method in
  between have been unable to solve via impedance/sound-speed contrast
  alone. It directly validates the physical mechanism the user's own
  channel ranking (this session, earlier) identified as #2 in importance
  (right behind Doppler/motion) — and does so with a real simulation,
  not just a plausibility argument. IMPORTANT SCOPE CAVEAT, stated
  honestly: this test shows the effect EXISTS and is detectable in a
  raw A-scan's energy within a known time window — it does NOT yet show
  that speckle can be turned into a BLIND boundary-detection method
  (no boundary-extraction/imaging step was attempted here, unlike every
  reflection/transmission channel test in this project). The fluctuation
  magnitude (3%) and correlation length (single-grid-cell, no spatial
  smoothing) were both chosen as a first, simple, coarse proxy, not
  calibrated against any real measured myocardial backscatter coefficient
  — the qualitative conclusion (a real, sizeable, resolvable effect at
  this grid resolution) is solid, but the specific 3.93x number should
  not be over-trusted as a calibrated physical prediction.
- Next action: (a) turn this into a genuine BLIND detector — e.g., a
  DAS-style or simple windowed-energy image that maps per-pixel
  backscatter intensity across the full 2D domain (not just a known
  time window at a known angle), and see whether the myocardium ring
  vs. blood core distinction is visible/extractable as an actual
  boundary, the same standard this project holds every other channel to;
  (b) test with a more realistic (spatially-correlated, sub-wavelength)
  scatterer field instead of pure per-cell white noise, and/or calibrate
  the fluctuation magnitude against a cited real myocardial backscatter
  value rather than an assumed 3%; (c) test on patient023's real,
  irregular two-tissue anatomy, the same escalation pattern used for
  every other validated channel in this project. Not committed/pushed
  (standing instruction).

### Run 2026-07-09-34 — Backscatter/speckle DAS imaging: a genuinely nuanced result — naive bulk metric shows NO localization (contaminated by the known central DAS artifact), but the radial profile reveals real, physically-correct structure (a peak at the annulus midpoint, a sharp cutoff exactly at the true outer radius)
- Phase: 1 (new channel imaging). Per user ("sure"), turning run -33's
  raw energy-elevation finding into an actual image, the same bar every
  other channel in this project has been held to. `phase1_backscatter_
  das_image.py` — reused the DAS accumulation machinery (run -31)
  UNCHANGED, passing the SPECKLE trace as "phantom" and the HOMOGENEOUS
  trace as "background" (both from run -33's cached raw traces, no new
  jWave simulation), so DAS's existing background-subtraction isolates
  specifically the extra scattering contribution speckle adds, since
  the two shared boundary reflections are identical in both phantoms
  and cancel out.
- **First metric, naive bulk region averages: NO localization
  (wall/core ratio = 1.00x)** — looked like a clean negative on first
  read. **Caught before accepting that at face value**: inspected the
  actual image and a proper radial profile rather than trusting the two
  bulk numbers alone (same discipline as run -32's bug catch). The
  image shows a bright hot spot concentrated at the exact domain CENTER
  — the same central DAS crossing-artifact already diagnosed as a false
  positive in run -32 (many bistatic ellipses from different angles
  intersect there regardless of any real signal). That artifact
  dominates the naive "core" average (r<57), masking whatever weaker,
  real signal exists in the annulus.
- **Radial profile (mean intensity vs. radius from center) tells a
  different, more honest story**: after the central-artifact region,
  intensity dips to a local MINIMUM around r=45-55 (~0.13), rises to a
  local MAXIMUM around r=70 (~0.15) — almost exactly the myocardium
  annulus's own midpoint, (60+80)/2=70 — and then falls in a SHARP,
  CLEAN cutoff precisely at r=80, the TRUE outer boundary (see
  `results/figures/phase1_backscatter_das_image.png`). This is real,
  physically-meaningful, correctly-located structure — just much
  weaker in magnitude than the dominant central artifact, and much
  weaker than the reflection-channel DAS imaging of a genuine specular
  boundary (run -31, 0.15mm-scale precision).
- **Diagnosis, stated plainly**: DAS's constructive-accumulation logic
  is built for a single well-defined specular reflecting point — a
  boundary. It excels there (run -31). Speckle is the physical opposite:
  many random, spatially-incoherent micro-scatterers spread through a
  volume. Simple delay-and-sum does not localize that nearly as cleanly,
  because there is no single "correct" point for many different angles'
  echoes to constructively agree on — each angle's excess energy comes
  from many different actual scatterer locations within the wall, not
  one shared point. The genuine localization signal seen here (peak at
  the annulus midpoint, sharp cutoff at the true outer radius) shows the
  effect IS spatially present and DAS-detectable in aggregate, but not
  cleanly resolvable as a crisp image with this method at this angle
  count, unlike the specular-reflection case.
- Physical sanity checked? by whom?: Claude — did not stop at the
  flattering-looking-clean-but-wrong naive bulk metric (1.00x, which
  would have been reported as "no localization" and closed the question)
  and instead inspected the actual spatial pattern, catching the same
  known central-artifact contamination diagnosed in run -32 before
  drawing a conclusion; distinguished "the effect is real but weakly
  localized by this method" from "the effect doesn't localize at all,"
  a materially different and more accurate characterization.
- Gate passed? (Y/N): N/A — new-channel imaging test, not a phase gate.
- Observations: this refines, rather than overturns, run -33's positive
  finding. Backscatter/speckle contrast is real (run -33) AND
  spatially localized in aggregate (this run's radial profile) — but
  turning it into a clean, standalone blind image needs something
  beyond simple DAS (e.g., an incoherent/energy-based imaging method
  designed for diffuse scattering, more view angles, or an explicit
  central-artifact suppression step), not a solved problem yet. The
  asymmetry from run -32 (DAS solves peak-identity/geometry problems,
  not signal-strength problems) generalizes further here: DAS also
  does not automatically solve INCOHERENT-SOURCE localization problems
  the way it does coherent specular-reflection ones — a third, distinct
  category of limitation, now identified across three channels.
- Next action: (a) suppress or explicitly exclude the central DAS
  artifact region before computing any bulk statistic for this or
  future speckle-imaging tests (a fixable methodological gap, now that
  it's identified twice); (b) try an incoherent/intensity-based
  imaging method (e.g., accumulate squared/energy contributions rather
  than raw envelope amplitude, or a proper speckle-tracking-style
  approach) instead of standard coherent DAS, since the physics here is
  fundamentally different from specular reflection; (c) run -33's own
  still-open items (spatially-correlated scatterer field, calibrated
  fluctuation magnitude, patient023 real anatomy) remain untouched.
  Not committed/pushed (standing instruction).

### Run 2026-07-09-35 — Incoherent (energy) backscatter imaging FIXES run -34's localization problem: a genuine, clean annular signature now visible, wall/clean-core ratio improves from 1.00x to 1.69x
- Phase: 1 (new-channel imaging, follow-up fix). Per user ("the first
  one," choosing between the two levers proposed at the end of run -34):
  implemented the incoherent/energy-based accumulation fix, combined
  with explicit central-artifact exclusion. Also directly answered a
  clarifying question first ("so why speckle still fail here? blood is
  so different from myocardium") — the physical CONTRAST is genuinely
  the starkest found all session (myocardium scatters, blood essentially
  doesn't), but coherent delay-and-sum (built for a single specular
  point that every viewing angle agrees on) has no mechanism to localize
  an INCOHERENT source, where each angle sees a genuinely different,
  uncorrelated realization of the same underlying random microstructure
  — there is no single delay for 36 angles to constructively agree on
  the way there is for a real boundary.
- Design: `phase1_backscatter_das_image_energy.py`. Two changes from
  run -34, both flagged as the fix at the time: (1) accumulate SQUARED
  (energy) excess envelope per angle instead of raw amplitude — the
  physically appropriate accumulator for a spatially-extended, randomly-
  phased scattering source, and one that disproportionately suppresses
  small/inconsistent per-angle contributions relative to genuinely
  elevated ones; (2) explicitly exclude a GENERIC central region
  (r<20 cells, not informed by patient023's or this phantom's own true
  contour) from all bulk statistics, since run -34 already diagnosed
  that the naive bulk metric was contaminated by a known central DAS
  crossing-artifact. Compared the wall annulus (R_inner+5 < r <
  R_outer-5) against a CLEAN core region (central-exclusion radius < r
  < R_inner-5, deliberately avoiding both the artifact zone and the
  wall) rather than run -34's contaminated "everything inside R_inner"
  region. No new jWave simulation — reused the exact same cached raw
  traces as run -34 (`results/speckle_channel_raw_traces.npz`, run -33),
  pure post-processing, for a direct, apples-to-apples comparison.
- **Result: a real, substantial improvement.** Wall/clean-core ratio:
  **1.69x** (vs. run -34's 1.00x — no localization at all). Wall/outside
  ratio: 1.98x (vs. run -34's 1.81x). The radial profile
  (`results/figures/phase1_backscatter_das_image_energy.png`) now shows
  a clean, well-shaped rise starting almost exactly AT the true inner
  radius (r=60), peaking around r=70 (the annulus midpoint, same
  location run -34's much noisier profile also weakly suggested),
  staying elevated through the true outer radius (r=80), then falling
  sharply — a genuinely convincing, well-localized annular signature,
  not just a marginal bump buried in noise. The image itself shows a
  visibly brighter halo between the two true contours, distinguishable
  by eye from both the (now excluded) central artifact and the exterior
  water region.
- Physical sanity checked? by whom?: Claude — explained the underlying
  mechanism (coherent vs. incoherent source imaging) BEFORE building the
  fix, so the fix follows directly from a stated physical diagnosis
  rather than trial-and-error; reused the identical cached data as run
  -34 so the comparison isolates the METHOD change as the only variable.
- Gate passed? (Y/N): N/A — new-channel imaging test, not a phase gate.
- Observations: this closes out run -34's open methodological gap with
  a real, working fix — backscatter/speckle contrast is now not just
  detectable in aggregate (run -33) but genuinely spatially localizable
  as a real annular image (this run), once the accumulation method
  matches the physical character of the source (incoherent energy, not
  coherent amplitude) and the known central artifact is excluded rather
  than left to contaminate bulk statistics. This is the first genuinely
  positive, spatially-resolved result for blood-vs-myocardium
  discrimination in this entire project, via a channel (backscatter)
  completely different from the interface-reflection approach that
  every earlier attempt (runs -07 through -32) relied on and that was
  ultimately shown to be too weak.
- Next action: (a) run -33's still-open items remain (spatially-
  correlated/sub-wavelength scatterer field instead of pure per-cell
  white noise; calibrate the fluctuation magnitude against a cited real
  myocardial backscatter value rather than an assumed 3%); (b) test this
  incoherent-imaging fix on patient023's real, irregular anatomy — the
  standard escalation pattern for every validated channel/method in
  this project, and the natural next step now that the synthetic
  circular case is working; (c) test whether MORE view angles further
  sharpens this annular signature (per run -31's finding that view
  count matters substantially even for the easier coherent case, it
  likely matters at least as much here). Not committed/pushed (standing
  instruction).

### Run 2026-07-09-36 — Speckle-informed surface selection on patient023 real anatomy: NEGATIVE result on this implementation, but diagnosed to a pre-existing, already-known candidate-radius bias, not a flaw in the rescoring concept itself
- Phase: 1 (channel integration test). Per the user's detailed proposed
  integration framing: "Reflection/transmission propose surfaces.
  Speckle validates the tissue region between surfaces... use it as a
  soft anatomical prior / likelihood term, not as a peak-picking
  boundary mode," with a specific immediate experiment: "Take your
  existing reflection-derived candidate boundaries and rescore them
  with a speckle wall-likelihood term. If the correct inner contour
  gets better ranking than the false reflection candidates, then
  speckle has become useful."
- Design: `phase1_speckle_patient023_sim.py` (36 NEW forward
  simulations — patient023's real two-tissue anatomy with randomized
  myocardium microstructure, same seed=42/std_frac=0.03 as runs -33/-35
  — reusing the ALREADY-cached homogeneous patient023 traces from run
  -32 as the background half, so only 36 sims were needed here, not 72;
  ~7-10 min estimate given upfront, completed within it) and
  `phase1_speckle_informed_surface_selection.py`: (1) generated MULTIPLE
  inner-boundary candidates per angle (mean 6.0/angle) from the classic
  single-shot pitch-catch matched-filter trace — not just the single
  strongest peak, since the user's framing specifically calls for
  rescoring among several candidates; (2) built the incoherent (energy)
  speckle field for patient023's real anatomy (run -35's fix, applied
  here for the first time to real irregular anatomy rather than the
  synthetic circle); (3) scored each candidate by a RISING-EDGE
  likelihood (speckle energy just outside the candidate radius, minus
  just inside — a genuine inner boundary should show low-inside/
  high-outside, since blood doesn't scatter and myocardium does) and
  picked the top-scoring candidate per angle; (4) compared against the
  naive "strongest-amplitude peak" baseline, per the user's exact
  proposed test.
- **Result: no improvement — correlation with the true inner contour
  actually got slightly WORSE (naive: 0.262 -> speckle-rescored: 0.106),
  though RMSE improved somewhat (naive: 1.839mm -> speckle-rescored:
  1.409mm) — a genuinely mixed, inconclusive signal, reported honestly
  rather than cherry-picking the favorable RMSE number.**
- **Diagnosis, not just the raw negative result**: inspected the actual
  per-angle scatter plot
  (`results/figures/phase1_speckle_informed_surface_selection.png`)
  rather than stopping at the correlation numbers. BOTH methods'
  candidates cluster systematically 10-20 cells ABOVE the true inner
  contour, in the SAME direction, not scattered independently — i.e.,
  naive and speckle-rescored are not failing differently, they are
  failing THE SAME WAY. This points directly at a cause already
  established in this project back in run -09: converting a peak's
  arrival TIME to a candidate RADIUS via the all-water round-trip
  assumption systematically OVERSHOOTS for any path that partly crosses
  faster myocardium (the ~+7.8 cell bias characterized there). Every
  candidate radius fed into the rising-edge scorer here used that same
  biased conversion, so the "just inside/just outside" sample points
  were evaluated at a location shifted outward from where the true
  wall/blood transition physically sits in the speckle field --
  contaminating the rescoring step with a bias inherited from candidate
  GENERATION, not a defect in the rescoring LOGIC itself. The visible
  speckle field for patient023 itself still looks reasonably localized
  (a bright halo roughly tracking the annulus between the true outer
  and inner contours, similar in character to run -35's synthetic-
  circle success) -- the failure is specifically in evaluating that
  field at systematically mislocated candidate positions, not in the
  field's own quality.
- Physical sanity checked? by whom?: Claude — did not stop at the
  aggregate correlation numbers (which alone could be read as "speckle
  rescoring doesn't work" and closed as a negative result); inspected
  the actual per-angle candidate positions, noticed BOTH methods share
  the same directional bias rather than independent scatter, and traced
  that pattern back to a specific, already-documented mechanism (run
  -09's all-water conversion overshoot) rather than leaving the result
  as an unexplained null.
- Gate passed? (Y/N): N/A — channel-integration test, not a phase gate.
- Observations: this is a genuine test of a real, well-motivated
  integration idea that came back negative on its FIRST implementation,
  for an identifiable and fixable reason — not evidence the underlying
  concept (speckle as a soft validator/rescorer for candidate surfaces,
  rather than a standalone detector) is wrong. The correct next test is
  to rescore candidates using a BIAS-CORRECTED radius (or, better,
  sample the speckle field directly from candidate TIME via a corrected
  time-to-radius model, or search over an assumed intermediate sound
  speed for the tissue-crossing leg) rather than the naive all-water
  conversion this project has used as its baseline convention since run
  -09 — this has never actually been fixed, only diagnosed, across the
  whole history of this project's inner-boundary work.
- Next action: (a) MOST PROMISING — apply a bias-corrected radius
  conversion (e.g., assume the correction magnitude already measured in
  run -09/-13's circular positive control, or iteratively solve for the
  tissue-crossing sound speed given the already-known outer boundary)
  to the candidate set BEFORE speckle rescoring, then re-run this exact
  comparison; (b) run -35's still-open items (spatially-correlated
  scatterer field, calibrated fluctuation magnitude, more view angles)
  remain untouched; (c) consider evaluating the rising-edge score
  directly in matched-filter TIME/delay space (sampling the speckle
  field via each candidate's own predicted bistatic delay through an
  assumed medium) rather than via an intermediate radius conversion, to
  avoid re-introducing the same conversion bias in a different form.
  Not committed/pushed (standing instruction).

### Run 2026-07-09-37 — Bias-corrected candidate conversion works exactly as diagnosed (systematic bias eliminated), but this EXPOSES a separate, deeper problem: neither naive amplitude nor speckle rescoring reliably picks the right candidate — a precision/identification gap, not a bias gap
- Phase: 1 (channel integration test, follow-up). Per user ("sure, run
  and diagnose"): implemented run -36's proposed fix directly —
  `phase1_speckle_informed_surface_selection_v2.py` replaces the naive
  all-water time-to-radius conversion with a proper TWO-LEG conversion
  (water leg from probe to the ALREADY-VALIDATED outer boundary, run
  -32's DAS estimate; myocardium leg from there inward to the
  candidate), correctly crediting the myocardium leg's faster sound
  speed instead of assuming the whole round trip is water. No new jWave
  simulation — reused all three cached trace sets (water/homogeneous
  from run -32, speckle from run -36) plus the cached DAS image for the
  outer-boundary reference, pure post-processing.
- **Result: the fix worked EXACTLY as diagnosed, on the specific thing
  it targeted.** Speckle-rescored candidates' mean bias dropped from a
  clear systematic overshoot to **+0.211mm** (essentially unbiased) —
  direct confirmation that run -36's diagnosis (all-water conversion
  overshoot, the mechanism characterized back in run -09) was the
  correct explanation for why both methods drifted the same direction.
- **But this did NOT translate into better boundary-tracking accuracy —
  if anything, per-angle precision got WORSE.** Correlation with the
  true inner contour: naive 0.272 (run -36 uncorrected: 0.262, no
  change), speckle-rescored 0.098 (run -36 uncorrected: 0.106, no
  change) — both still effectively noise-level. RMSE: naive 2.146mm
  (run -36: 1.839mm, WORSE), speckle-rescored 1.740mm (run -36:
  1.409mm, also WORSE), despite the much-improved mean bias. Inspecting
  the actual scatter
  (`results/figures/phase1_speckle_informed_surface_selection_v2.png`)
  explains why: the naive candidates still cluster mostly above the
  true contour (the fix helped speckle-rescored's mean bias much more
  than naive's), while the speckle-rescored candidates are now
  scattered on BOTH sides of the true contour — some far below
  (r~20-30 cells, clearly wrong) and some still far above. Removing the
  systematic one-directional offset did not improve precision; it
  converted a CONSISTENT error into UNPREDICTABLE scatter in either
  direction, with the mean now coincidentally near zero.
- **Correct, honest reframing**: run -36's fix addressed exactly the
  bias/offset component of the error (confirmed, worked). It did
  nothing for a SEPARATE, deeper problem: among the ~6 candidate peaks
  per angle (multiple genuinely plausible echo sources — reverberation,
  off-axis spillover, the true inner echo, etc.), neither "pick the
  strongest amplitude" nor "pick the best speckle rising-edge score" is
  a reliable enough criterion, ON ITS OWN, to consistently identify
  WHICH of those ~6 candidates is the real one. This is a precision/
  identification gap, not a bias gap — fixing the coordinate conversion
  cannot fix an underlying ambiguous multi-candidate selection problem.
- Physical sanity checked? by whom?: Claude — verified the fix's own
  claimed effect (bias) directly via the printed bias statistic before
  looking at whether it solved the BROADER problem (correlation/RMSE),
  correctly separating "did the specific diagnosed mechanism get fixed"
  (yes) from "did overall accuracy improve" (no) rather than conflating
  the two; inspected the actual scatter pattern to characterize HOW the
  error changed (consistent bias -> unpredictable scatter) rather than
  reporting only the aggregate correlation/RMSE numbers.
- Gate passed? (Y/N): N/A — channel-integration test, not a phase gate.
- Observations: this is a clean methodological lesson, independent of
  this project's specific numbers — a correctly-diagnosed and correctly
  -fixed systematic bias does not automatically fix a coexisting
  precision/variance problem, and the two should always be checked
  separately (bias statistic AND correlation/scatter pattern), not
  inferred from a single aggregate error metric. For THIS project, it
  means the real bottleneck for inner-boundary detection on real
  anatomy is not (or not only) the coordinate-conversion bias
  characterized since run -09 — it is a genuine multi-candidate
  IDENTIFICATION problem that neither of the two criteria tested so far
  (amplitude, speckle rising-edge) solves alone.
- Next action: the user's own original framing (a JOINT score combining
  surface-fit evidence AND speckle-wall-likelihood evidence, rather than
  either criterion alone or a simple max-score pick per candidate) is
  now the clearly-motivated next thing to try, since neither criterion
  alone is sufficient — e.g., score = surface-fit-quality + lambda *
  speckle-wall-likelihood - mu * speckle-in-cavity-penalty, jointly
  optimized across candidates AND angles (a smooth-contour constraint
  across neighboring angles, not independent per-angle picks), rather
  than the independent-per-angle single-criterion approach tested in
  both run -36 and this run. Also still open: run -35's items (more
  view angles, spatially-correlated scatterer field, calibrated
  fluctuation magnitude). Not committed/pushed (standing instruction).

### Run 2026-07-09-38 — Joint optimization (surface-fit + speckle + cross-angle smoothness) gives the best inner-boundary result yet on real anatomy (corr=0.416, RMSE=1.565mm) — but a parameter sweep then reveals the improvement is NOT actually coming from speckle
- Phase: 1 (channel integration test, full implementation of the user's
  original framing). Per user ("sure"): implemented the joint scoring
  scheme exactly as proposed — score = surface-fit (z-scored amplitude)
  + lambda * speckle rising-edge likelihood, jointly optimized across
  BOTH candidates and neighboring angles (a smoothness/continuity
  constraint — real anatomy contours don't jump wildly angle to angle),
  solved via EXACT dynamic programming over the cyclic 36-angle sequence
  (`phase1_speckle_joint_optimization.py`). No new jWave simulation —
  reused the identical bias-corrected candidates and cached data as runs
  -36/-37.
- **First result (lambda=1.0, kappa=0.5, both picked as a reasonable
  default): correlation=0.416, RMSE=1.565mm, mean bias=+1.411mm — the
  best inner-boundary result on real anatomy in this project's entire
  history** (vs. run -37's independent methods: naive corr=0.272/
  RMSE=2.146mm, speckle-rescored corr=0.098/RMSE=1.740mm). Visually
  (`results/figures/phase1_speckle_joint_optimization.png`), the
  joint-optimized contour now tracks the true inner contour's
  UNDULATION PATTERN qualitatively (e.g., both dip over roughly the same
  angular range), a real, if partial, shape-tracking result — the first
  time any method in this project has shown genuine (if weak) angular
  correlation with patient023's real inner-contour shape.
- **Caught before accepting the improvement as due to speckle**: ran a
  parameter sweep (`phase1_speckle_joint_sweep.py`, 25 combinations of
  lambda_speckle in [0, 0.5, 1, 2, 4] x kappa_smooth in [0.1, 0.5, 1, 2,
  4], all free/cached, no new simulation) INCLUDING a lambda=0 control
  (smoothness + amplitude ONLY, zero speckle weight) specifically to
  isolate whether speckle was doing real work or whether the smoothness/
  DP constraint alone explained the gain. **Result: the single BEST
  correlation in the entire 25-point sweep (0.566, at kappa=2.0) occurs
  at lambda=0.0 — i.e., with NO speckle term at all.** No lambda>0
  configuration robustly or clearly exceeded the best lambda=0 result
  (closest: lambda=4.0/kappa=0.5 at corr=0.551, still below 0.566).
- **Honest, corrected conclusion: the real driver of run -38's
  improvement over runs -36/-37 is the CROSS-ANGLE SMOOTHNESS/
  CONTINUITY constraint (exploiting that a real anatomical contour is a
  connected curve, not 36 independent measurements), not the speckle
  rising-edge term.** This is a materially different, more important
  finding than the first result's framing suggested, and is reported
  here explicitly rather than left standing on the more flattering
  initial number. Every reflection-channel method in this project
  before this run (runs -08 through -37) selected candidates
  INDEPENDENTLY per angle — this is the first time enforcing physical
  continuity across the whole contour was tried at all, and it appears
  to be the more powerful, previously-untapped lever, independent of
  whatever channel (speckle or otherwise) supplies the per-candidate
  score.
- Physical sanity checked? by whom?: Claude — did not stop at the first,
  favorable-looking result (which would have been reported as "speckle
  fixed the problem"); designed and ran a specific ablation (lambda=0)
  BEFORE attributing the improvement to speckle, and let that ablation
  override the initial, more flattering framing once it showed the
  opposite of what was assumed.
- Gate passed? (Y/N): N/A — channel-integration test, not a phase gate.
- Observations: even the best smoothness-only result (corr=0.566,
  RMSE=1.921mm, or the separately-best RMSE=1.181mm at lambda=4/
  kappa=0.5) is a real, substantial improvement over every independent-
  per-angle method tried in this project's history, but still far from
  solved (moderate correlation, ~1.2-2mm RMSE, and every configuration
  in the sweep shows a persistent positive bias of roughly +1-1.9mm,
  suggesting the run -09-era conversion/geometry issues are still only
  partially resolved even with the two-leg correction). Whether speckle
  contributes ANYTHING on top of smoothness remains genuinely
  unresolved by this sweep — it neither clearly helps nor clearly hurts
  across most settings, it just isn't the dominant effect. This doesn't
  invalidate runs -33/-35's earlier findings (speckle IS a real,
  detectable, spatially-associated signal) — it means turning that
  signal into a per-candidate discriminative score strong enough to beat
  a well-tuned smoothness prior, on this specific real-anatomy test,
  has not yet been achieved.
- Next action: (a) properly separate the two contributions with a
  cleaner ablation design (e.g., cross-validate lambda/kappa rather than
  reading off a single sweep table, since 36 angles is a small dataset
  and some of this sweep's variation could be overfitting noise); (b)
  the smoothness-based joint optimization itself, independent of
  speckle, is now the more promising lead for future inner-boundary work
  and could be tried on patient023's OUTER boundary or the off-center
  heart phantom as a further validation; (c) run -35's items (more view
  angles, spatially-correlated scatterer field, calibrated fluctuation
  magnitude) remain untouched and could still change whether speckle
  contributes more once its own signal quality is improved. Not
  committed/pushed (standing instruction).

### Run 2026-07-09-39 — CHANNEL 3 (Doppler/motion, oracle-validated): echo-timing shift tracks TRUE anatomical motion with near-exact physical slope — the strongest, most quantitatively precise result of the entire session
- Phase: 1 (new channel validation). Per user ("doppler"): tests the
  Doppler/motion channel -- ranked #1 in this session's own channel
  ranking for blood-vs-myocardium discrimination, never touched until
  now (every prior channel test used a single static frame). Preceded
  by a separate, larger detour: building an end-to-end CNN on real
  "beating" ACDC cardiac-cycle data (jwave_test's already-prepped
  8-phase motion cycles for patient001/patient023), including a full
  Colab GPU simulation pipeline (packaged and debugged through several
  rounds: a hung `files.upload()`/`files.download()` widget, traced to
  Colab's browser-JS bridge not working through VS Code's Colab
  extension, fixed by embedding the package as base64 and using a
  Google Drive save instead) -- the CNN itself showed a real but weak,
  non-obvious generalization result (train correlation 0.68 on
  patient001; test correlation 0.23 on held-out patient023, with an
  unexpected split: WORSE on the "normal-range" test subset (corr=0.11)
  than on the physically extreme, strongly-contracted subset (corr=0.51)
  -- opposite of a naive "out-of-distribution collapse" story). That
  CNN's best subset-correlation (0.51) landed close to run -38's best
  static physics correlation (0.57), a suggestive (not proven, given
  small-sample noise) convergence worth remembering as context for this
  run.
- Design: `phase1_doppler_motion_channel.py`. SCOPE STATED HONESTLY UP
  FRONT: true continuous Doppler measures a frequency shift accumulated
  WITHIN one pulse from a continuously-moving scatterer -- nothing in
  this project's infrastructure simulates a medium that moves DURING
  wave propagation. What this tests instead (zero new jWave simulation
  -- reuses the ALREADY-CACHED beating-heart traces from the Colab run
  plus the cached water baseline): the discrete analog -- does the
  inner-boundary echo's arrival TIME shift between ADJACENT cardiac
  phases track the real, KNOWN anatomical motion? Uses an ORACLE-
  assisted peak match (the analytically-predicted true inner-echo time,
  from the already-known true contours, identifies which observed peak
  is the genuine inner echo at each phase/angle) -- validating the
  MECHANISM exists before attempting a blind detector, this project's
  established discipline since the circular positive control (run -12).
- **Result: a clean, strong, quantitatively PRECISE confirmation.**
  Correlation between true frame-to-frame radial motion and observed
  echo-time shift: patient001 (mild, ~1.6% contraction) corr=-0.517
  (n=48/126 valid pairs); patient023 (strong, ~18% contraction)
  corr=-0.932 (n=40/126) -- an extremely tight fit, visually confirmed
  (`results/figures/phase1_doppler_motion_channel.png`: points sit
  almost exactly on the fitted line). Pooled correlation=-0.885.
  **The fitted SLOPES for both patients are nearly identical AND match
  the analytically-predicted physical magnitude (2/c_myo=1269.04 ns/mm)
  to within 1.5% (patient001: -1288.11 ns/mm measured) and 0.03%
  (patient023: -1268.62 ns/mm measured, essentially exact).** This is
  not just a directional correlation -- the actual physical slope of
  the relationship was independently predicted from cited tissue sound
  speed alone and then confirmed by simulation to within noise.
- **Self-caught sign error, corrected before reporting**: the script's
  own first-draft comment predicted the WRONG sign (assumed contraction
  -> earlier echo). Re-derived carefully: the probe sits OUTSIDE at
  r=PROBE_RADIUS; a SHRINKING inner radius (contraction) moves the
  boundary AWAY from the probe (toward the domain center), LENGTHENING
  the round trip, so the echo arrives LATER, not earlier -- a NEGATIVE
  correlation between radial motion and echo-time shift is the
  physically CORRECT prediction, exactly matching the measured result.
  This also meant the script's own automatic "verdict" logic (which
  only checked for positive correlation) had mislabeled this excellent
  result as "weak/none" -- caught and fixed (checks `abs(corr)` now)
  before reporting, rather than letting a strong result be reported as
  a failure due to a sign-convention bug.
- Physical sanity checked? by whom?: Claude -- predicted the physical
  slope magnitude (2/c_myo) analytically BEFORE comparing to the
  measured slope, not fit after the fact; caught and corrected its own
  sign error by re-deriving the geometry from scratch rather than
  accepting the first (wrong) intuition; caught that the wrong sign
  had propagated into an automatic verdict-classification bug that
  would have mischaracterized a strong positive result as a negative
  one, and fixed it before reporting.
- Gate passed? (Y/N): N/A -- new-channel validation, not a phase gate.
- Observations: this is the single strongest, most quantitatively
  precise result of the entire session -- stronger and more exactly
  physics-matched than the static reflection channel (run -32, weak/
  uncorrelated), the DAS-based speckle imaging (run -35, real but
  partial localization), or the joint-optimization static method (run
  -38, corr 0.42-0.57 at best). It directly validates the premise
  behind this session's original channel ranking (motion/Doppler > 
  backscatter/speckle > reflection): motion genuinely carries by far
  the cleanest, most physically well-behaved signal of anything tested
  for blood/myocardium-adjacent boundary information in this project.
- Next action: this validates the MECHANISM (oracle-assisted) but not
  yet a BLIND detector -- the natural next step, attempted immediately
  after this run (see run -40), is combining this with the best
  validated static method (run -38) into an actual blind full-cardiac-
  cycle reconstruction via temporal tracking. Not committed/pushed
  (standing instruction).

### Run 2026-07-09-40 — Motion-tracked BLIND reconstruction (patient023, full cycle): naive nearest-time tracking locks onto a static, non-moving artifact instead of the true boundary — a real, honestly-diagnosed distinct failure mode, not a re-run of run -39's problem
- Phase: 1 (reconstruction attempt, combining runs -38 and -39). Per
  user ("no can u reconstruct using that on pt23? use colab if compute
  more than 10 mins" -- flagged as unnecessary since this needed zero
  new jWave simulation, pure post-processing on already-cached data,
  so it stayed local): `phase1_motion_tracked_reconstruction.py`
  combines the two best validated results into a genuine BLIND
  reconstruction across patient023's full cardiac cycle: (1) phase 0
  (ED) initial boundary via run -38's best STATIC method (bias-
  corrected candidates + cross-angle smoothness DP, lambda=0); (2)
  phases 1-7 via BLIND temporal TRACKING -- at each phase, pick the
  peak closest in TIME to the previous phase's own tracked peak, using
  NO true-contour information past phase 0, exploiting run -39's
  validated near-exact timing-to-motion relationship.
- **First attempt (TRACK_WINDOW_S=400ns): only 83/144 tracked, corr=
  0.353, RMSE=1.571mm.** Diagnosed immediately (not accepted at face
  value): run -39 measured real frame-to-frame shifts up to ~1268ns at
  peak patient023 contraction -- a 400ns window was too narrow and was
  killing tracking specifically during the largest, most clinically
  interesting motion excursions (visible in the figure as missing/NaN
  cells clustered right where contraction is strongest).
- **Second attempt (TRACK_WINDOW_S=1600ns, comfortably above the
  validated max): coverage improved (126/144 tracked) but accuracy did
  NOT (corr=0.353, unchanged; RMSE=1.540mm, barely moved).** Inspecting
  the actual reconstructed image
  (`results/figures/phase1_motion_tracked_reconstruction.png`) revealed
  why: nearly every column (angle) is FLAT across all 8 phases -- i.e.,
  the "tracker" is finding essentially the SAME echo time at every
  single phase, not following genuine motion at all. **This is
  DIFFERENT from and NOT explained by the tracking-window fix**: it
  means the tracker locked onto a STABLE, NON-MOVING feature (most
  likely a persistent strong echo -- the outer boundary or a
  reverberation artifact) rather than the true, weaker, genuinely
  moving inner-boundary echo, and then correctly (but uselessly)
  re-found that same static feature every subsequent phase.
- **Honest diagnosis of the real, distinct failure mode**: naive
  nearest-time-neighbor tracking has no way to distinguish "the true
  target barely moved this frame" from "I locked onto the wrong,
  non-moving thing" -- a strong, always-reliably-present static echo is
  the "safest" (lowest-uncertainty) match for a memoryless nearest-
  neighbor tracker, so it wins by default, especially since the phase-0
  seed itself (from run -38's method) was already weak on its own
  (correlation=0.219 alone, using only 18 of the original 36 angles).
  This compounds TWO still-unsolved sub-problems on top of each other:
  blind SEEDING (which peak is the genuine inner echo at phase 0) and
  blind TRACKING (avoiding lock-on to a stable artifact) -- neither
  solved individually, and naively chaining them does not average out
  to something better.
- Physical sanity checked? by whom?: Claude -- did not accept the
  first negative-looking result (83/144, corr=0.353) as final without
  first checking whether a specific, identifiable parameter (the
  tracking window) explained it; after fixing that and finding accuracy
  STILL unchanged, inspected the actual reconstructed image rather than
  the aggregate correlation number alone, which revealed a completely
  different, more fundamental failure mode (static lock-on) that a
  parameter tweak could not have fixed.
- Gate passed? (Y/N): N/A -- reconstruction attempt, not a phase gate.
- Observations: run -39's core finding is NOT undermined by this
  result -- the oracle-assisted mechanism validation (echo timing
  tracks true motion with near-exact physical slope) remains solid.
  What this run shows is that naive blind tracking is a genuinely
  harder problem than the oracle-assisted validation suggested, because
  it requires solving blind seeding and lock-on-avoidance simultaneously,
  neither of which any method in this project has solved yet. This is
  consistent with (and extends) the pattern from runs -36/-37 (bias-
  correction fixed one specific diagnosed problem but exposed a
  separate, deeper identification problem underneath it).
- Next action: (a) a smarter tracker is needed -- e.g., requiring the
  tracked peak's AMPLITUDE/character to remain roughly consistent
  frame-to-frame (not just nearest in time), or a proper motion-model-
  based tracker (Kalman-filter-style, predicting an EXPECTED shift from
  the validated ~1269 ns/mm physical slope and penalizing candidates far
  from that prediction, rather than naive nearest-time-neighbor); (b)
  a better phase-0 seed (the current one, run -38's method restricted to
  18 angles, was already weak at corr=0.219 -- worth re-deriving from
  the full 36-angle version instead); (c) run -35's and run -38's other
  still-open items remain untouched. Not committed/pushed (standing
  instruction).

### Run 2026-07-09-41 — Motion-tracked reconstruction v2 (better seed + velocity tracker): the seed fix worked exactly as intended, the tracker fix did NOT — precisely diagnosed to a zero-velocity bootstrap vulnerability, not a new mystery
- Phase: 1 (reconstruction attempt, continuing runs -39/-40). Per user
  ("proceed"): implemented both fixes run -40 flagged as needed.
  `phase1_motion_tracked_reconstruction_v2.py`: (1) re-seeded phase 0
  using the FULL 36-angle static joint-optimization result (run -38's
  method, same setting lambda=0/kappa=2.0) instead of the impoverished
  18-angle-only version run -40 used, then subsampled the result to the
  18 angles the beating-heart simulation actually captured; (2) replaced
  naive nearest-time-to-previous-frame tracking with a constant-
  VELOCITY predictor (predicts each phase's expected echo time as
  previous time + an online exponential-moving-average velocity
  estimate, not just the previous time itself). No new jWave
  simulation -- reuses all cached data (beating-heart traces, water
  baseline, patient023 speckle field, cached DAS outer-boundary
  estimate).
- **Result: the seed fix worked exactly as intended, in isolation.**
  Reproduced run -38's 36-angle result exactly (corr=0.566, confirming
  no regression), and phase-0-alone accuracy on the 18-angle subsample
  improved dramatically: corr=0.535 (run -40's 18-angle-only seed:
  corr=0.219) -- more than double.
- **But the FULL-CYCLE result got WORSE, not better: corr=0.219
  (run -40: corr=0.353), RMSE=1.795mm (run -40: 1.540mm).** Inspecting
  the actual reconstructed image
  (`results/figures/phase1_motion_tracked_reconstruction_v2.png`)
  showed the SAME failure signature as run -40 -- nearly every column
  (angle) still flat across all 8 phases, static lock-on persisting
  despite the much better starting seed.
- **Precisely diagnosed, not left as an unexplained regression**: the
  velocity-based tracker's protection against static lock-on can only
  work AFTER velocity becomes nonzero, but velocity is initialized at
  ZERO, so the very first tracking transition (phase 0->1) is exactly
  as vulnerable to static lock-on as naive nearest-time tracking was --
  no protection at the one moment it's needed to bootstrap correctly.
  Worse, if lock-on happens on that first step, the observed shift is
  ~0 by construction (a static echo doesn't move), which the
  exponential-moving-average update reads as confirmation of "zero
  velocity" -- reinforcing rather than escaping the bad state, a
  self-sustaining attractor that then persists for the rest of the
  cycle. The better seed didn't help because the FAILURE happens at the
  tracking step immediately after the seed, not at the seed itself.
- Physical sanity checked? by whom?: Claude -- verified the seed fix's
  claimed effect in isolation (reproduced run -38's number exactly,
  confirmed the 18-angle subsample's improvement) BEFORE looking at
  full-cycle accuracy, correctly separating "did fix #1 work" (yes)
  from "did the overall problem get fixed" (no); traced the counter-
  intuitive worse-full-cycle-despite-better-seed result to a specific,
  identifiable mechanism (the zero-velocity bootstrap vulnerability)
  rather than reporting an unexplained regression.
- Gate passed? (Y/N): N/A -- reconstruction attempt, not a phase gate.
- Observations: two consecutive, honest attempts (runs -40 and -41)
  at blind motion tracking have now hit the SAME core failure mode
  (static lock-on) via two different specific mechanisms (naive
  nearest-time matching; zero-velocity-initialized constant-velocity
  prediction) -- suggesting the failure is structural to "track via
  time-proximity alone, seeded from one static detection," not merely a
  parameter-tuning problem. Real clinical tissue-Doppler/wall-tracking
  imaging solves this with much richer information than this project's
  discrete 8-phase, 18-angle snapshots provide (continuous, densely-
  sampled RF lines at high pulse-repetition-frequency, dedicated
  cross-correlation-based wall-tracking algorithms) -- the gap here may
  be as much about acquisition density as tracking-algorithm cleverness.
- Next action: a fix would need to break the zero-velocity bootstrap
  problem directly -- e.g., bootstrap velocity from the FIRST TWO
  phases using an amplitude-consistency check (only accept the phase-0
  -to-phase-1 match if the candidate's amplitude is also close to the
  phase-0 seed's amplitude, not just time-close) before trusting any
  velocity estimate, or abandon frame-to-frame tracking in favor of a
  single JOINT optimization across ALL 8 phases at once (extending run
  -38's cyclic-DP-across-angles idea to a combined DP across BOTH
  angles and phases simultaneously, with a physically-motivated
  transition cost based on the validated ~1269 ns/mm slope rather than
  greedy frame-to-frame propagation). Given this is the third
  consecutive attempt hitting the same underlying failure mode, this is
  a natural point to pause this specific thread and consolidate/take
  stock of the session rather than continue ad-hoc tracker tweaks. Not
  committed/pushed (standing instruction).

### Run 2026-07-09-42 — Multi-modal CNN fusion (reflection + previous-phase/motion, 2-channel): a clean, honest NULL result — no generalization improvement from naive channel concatenation
- Phase: 1 (ML channel-fusion test). Per user: "put all live echo
  measured information we did so far into cnn to see if any multi-
  model influence" -- the original CNN (earlier this session) only
  ever saw a single phase's reflection envelope in isolation, with NO
  motion information at all, despite run -39 showing timing-shift is by
  far the strongest single signal found this entire session. Tests
  whether fusing that motion information IN at the input (not hand-
  engineering a timing-shift feature, but giving the raw previous-phase
  envelope as a second channel and letting the network find whatever
  relationship helps) improves generalization to the held-out patient.
- Design: `phase1_multimodal_prepare_dataset.py` builds a 2-channel
  dataset -- channel 1: current phase's matched-filter envelope
  (background-subtracted, same as the original CNN); channel 2: the
  PREVIOUS phase's envelope, using CYCLIC indexing (phase 0's
  "previous" is phase 7, since the 8-phase cycle already closes
  ED->ES->ED, fractions[0]==fractions[7]==0). `phase1_multimodal_cnn.py`
  trains the IDENTICAL architecture as the original single-channel CNN
  (same seed, epochs, learning rate, weight decay), changing only
  Cin=2 instead of 1 for the first conv layer -- an apples-to-apples
  comparison isolating the input-fusion change as the only variable.
  No new jWave simulation -- reuses the same cached beating-heart
  traces and water baseline as every CNN/tracking experiment this
  session.
- **Result: no improvement, a clean null.** TRAIN (patient001, seen):
  corr=0.677 (single-channel: 0.684 -- essentially identical). TEST
  (patient023, unleaked): corr=0.177 (single-channel: 0.233 -- slightly
  worse, within noise given n=144). Breaking down by range: in-range
  subset corr=0.139 (single-channel: 0.114, marginally better, within
  noise); out-of-range (strong-contraction) subset corr=0.384 (single-
  channel: 0.509, WORSE). Visually
  (`results/figures/phase1_multimodal_cnn.png`): the SAME collapse
  pattern as the original single-channel CNN -- test predictions
  compressed into a narrow band (~55-63) regardless of the true value's
  much wider range (35-66) -- adding the second channel did not change
  the qualitative failure mode at all.
- Physical sanity checked? by whom?: Claude -- ran the IDENTICAL
  architecture/hyperparameters as the original single-channel CNN
  (only changing Cin) specifically so the comparison isolates the
  fusion change as the only variable, rather than confounding it with
  an incidental architecture/training difference; reported the small
  in-range improvement honestly as "within noise" rather than
  cherry-picking it as a win when the out-of-range and overall numbers
  both went the other way.
- Gate passed? (Y/N): N/A -- ML channel-fusion test, not a phase gate.
- Observations: this extends the exact pattern from runs -40/-41 (naive
  combination of two independently-validated signals does not
  automatically inherit either one's strength) into the ML domain --
  naively concatenating raw channels and hoping a small (144-example)
  CNN discovers the useful cross-channel relationship on its own is not
  enough. This is consistent with WHY runs -40/-41's explicit,
  hand-engineered motion-tracking attempts also struggled: the
  cross-phase relationship that made run -39's oracle-assisted result
  so strong (an exact, analytically-derived physical slope) is not
  something a tiny, generically-initialized 2-layer CNN is likely to
  discover from 144 raw-signal examples without either far more data or
  an architecture/loss that explicitly encodes the differencing
  operation (e.g., feeding the EXPLICIT timing-shift or envelope
  DIFFERENCE as an input feature, rather than the two raw channels
  side by side for the network to difference itself).
- Next action: if pursuing fusion further, feed the network an
  EXPLICIT difference/cross-correlation feature between adjacent phases
  (not just two raw channels) -- closer to how run -39's actual
  validated mechanism works (a timing shift, not raw amplitude
  co-presentation); otherwise, this is a natural point to STOP this
  specific ML-fusion sub-thread and consolidate the session, per the
  same reasoning as run -41's own next-action note (three consecutive
  attempts at combining validated pieces -- tracking-with-better-seed,
  velocity-aware-tracking, multi-modal CNN fusion -- have now each
  landed as a null or negative result, a consistent, honestly-reported
  pattern rather than three isolated failures). Not committed/pushed
  (standing instruction).

### Run 2026-07-09-43 — Speckle as a GEOMETRIC PATH CONSTRAINT (not a pointwise score): breaks the three-run losing streak decisively — best inner-boundary result of the entire session (corr=0.697, RMSE=0.903mm)
- Phase: 1 (channel integration, new idea). Per user, correctly
  identifying the actual gap in every prior speckle-integration attempt
  (runs -36/-37/-38): "since speckle have located the object, cant it
  localise the sound path?" -- every earlier attempt used speckle only
  as a POINTWISE rising-edge score evaluated AT each reflection
  candidate's own (bias-prone) implied location, never as an
  independent geometric constraint on its own. This tests the more
  direct version of the idea: extract the speckle field's OWN per-angle
  radial localization (where backscatter intensity rises from blood-
  like/low to wall-like/high along each angle's own ray) as a
  standalone estimate, then use THAT location -- not amplitude -- to
  pick which reflection-channel candidate is real.
- Design: `phase1_speckle_path_constraint.py`. Three comparisons, all on
  patient023 real anatomy, ALL REUSING CACHED DATA (no new jWave
  simulation -- confirmed unnecessary before running, so this stayed
  local rather than using Colab): (1) speckle-alone: per-angle radial
  profile's half-max rising-edge crossing, as a standalone boundary
  estimate; (2) speckle-CONSTRAINED candidate selection: among the
  SAME multi-candidate reflection-channel peaks used throughout runs
  -36/-37/-38, pick whichever candidate's radius is closest to the
  speckle-derived location (replacing "pick by amplitude" entirely,
  not just rescoring on top of it); (3) compared against run -38's best
  prior result (amplitude+smoothness joint optimization, corr=0.566).
- **Result: BOTH new methods beat everything tried before them, and
  the combination is the clear best.** Speckle-alone: corr=0.503,
  RMSE=1.041mm -- already competitive with run -38's much more complex
  joint-optimization pipeline, from geometry alone. Speckle-CONSTRAINED
  candidate selection: **corr=0.697, RMSE=0.903mm** -- decisively the
  best inner-boundary result of the ENTIRE session, beating run -38
  (0.566/~1.9mm), the naive amplitude baseline (0.272/2.146mm), and the
  speckle-alone estimate on its own (0.503/1.041mm). Confirmed visually
  (`results/figures/phase1_speckle_path_constraint.png`): the speckle
  rising-edge points sit close to the true inner contour around most of
  the boundary in the 2D field image, and the per-angle comparison plot
  shows the speckle-constrained curve tracking the true contour's
  undulation shape far better than the wildly erratic naive-amplitude
  curve (which swings from near 0 to ~95 cells in places).
- Physical sanity checked? by whom?: Claude -- structured the test as
  three explicit, ordered comparisons (speckle-alone, speckle-
  constrained, vs. prior best) rather than a single number, so the
  source of improvement is attributable; confirmed the result visually
  (both the 2D field image and the per-angle trace) before reporting it
  as a genuine win, not just from the aggregate correlation number.
- Gate passed? (Y/N): N/A -- channel-integration test, not a phase
  gate.
- Observations: this breaks a three-run losing streak (runs -40/-41/-42,
  all null or negative attempts at combining validated pieces) with a
  clean, decisive, correctly-targeted fix -- the difference was using
  speckle as an independent GEOMETRIC constraint (a location derived
  from the speckle field alone, that the reflection candidate must be
  consistent with) rather than a pointwise SCORE (evaluating speckle
  AT an already-computed, possibly-biased candidate location, which is
  what runs -36/-37/-38 did and which run -38's own ablation showed
  contributed nothing beyond smoothness alone). This is a meaningful
  conceptual distinction the user's question isolated precisely:
  speckle doesn't need to validate a candidate's amplitude plausibility
  -- it can directly supply an alternative, independent LOCATION
  estimate that competing candidates are then filtered against.
- Next action: (a) test this same speckle-constrained selection
  approach on the off-center heart phantom and/or patient001, to check
  it generalizes beyond this one patient; (b) try combining this with
  run -38's smoothness-DP (speckle-constrained candidate selection AS
  the unary term feeding into the cyclic DP, rather than amplitude) --
  plausibly the best-of-both-worlds combination, not yet tested; (c)
  the same idea could apply to the OUTER boundary and to patient023's
  bent-ray/transmission-channel work. Not committed/pushed (standing
  instruction).

### Run 2026-07-09-44 — Combining speckle-constrained selection with smoothness DP pushes patient023 to a new best (corr=0.738), but the SAME approach does NOT generalize to patient001 — an important, self-corrected negative result
- Phase: 1 (channel integration, both follow-ups from run -43). Per
  user ("proceed with both"): (a) combined run -43's speckle-constrained
  candidate selection with run -38's smoothness DP; (b) tested the same
  approach on patient001, a genuinely different, much more mildly-
  contracting patient (~1.6% real contraction vs. patient023's ~18%),
  requiring a FRESH reflection-channel simulation (patient001 had never
  had reflection-channel data in this project before, only
  transmission, runs -05/-07) -- 72 forward sims (homogeneous +
  speckle-injected two-tissue phantom, 36 angles each), ~15-20 min,
  water-only baseline reused from the cached geometry-only patient023
  trace (no re-simulation needed for that half).
- **Part (a) result, patient023 (all cached data, no new simulation):
  a genuine further improvement.** `phase1_speckle_constrained_dp.py`
  uses proximity to the speckle-derived location (not amplitude) as
  the DP's unary term. Built-in consistency check passed exactly:
  kappa_smooth=0 reproduces run -43's result exactly (corr=0.697,
  RMSE=0.903mm), confirming the DP correctly reduces to independent
  per-angle nearest-candidate selection with no smoothness. Modest
  smoothness (kappa=0.5-2.0) pushed further to **corr=0.738,
  RMSE=0.70-0.80mm -- a new best for this session** -- while high
  smoothness (kappa=4-8) collapsed the result (over-regularization
  destroying the real per-angle signal, the same over-smoothing failure
  mode as run -38's own sweep). Confirmed visually
  (`results/figures/phase1_speckle_constrained_dp.png`): the resulting
  curve tracks the true contour's undulation shape well, including the
  dip, with a modest consistent positive offset (a bias, not a
  shape-tracking failure).
- **Part (b) result, patient001: does NOT generalize -- corrected from
  an initially wrong automatic verdict.** Naive baseline: corr=0.047
  (noise-level). Speckle-alone: corr=-0.089 (WORSE than noise, negative).
  Speckle-constrained selection (kappa=0): corr=-0.108 (negative).
  Sweeping smoothness: kappa=0.5/1.0/2.0 all stayed negative (-0.10 to
  -0.15); kappa=4.0 alone produced corr=0.471 (the script's own
  automatic verdict logic flagged this as "GENERALIZES"); kappa=8.0
  collapsed to corr=-0.640. **Caught and corrected before accepting the
  script's own verdict**: a single positive result at ONE specific
  kappa value, bracketed on both sides by negative results (kappa=2.0:
  -0.10, kappa=8.0: -0.64), is a textbook multiple-comparison/cherry-
  picking red flag, not evidence of a robust effect, especially with
  only 36 data points feeding the correlation estimate. Inspecting the
  actual per-angle curve
  (`results/figures/phase1_patient001_speckle_constrained.png`)
  confirmed this: the "best" (kappa=4.0) reconstruction swings far more
  widely (41-63 cells) than patient001's genuinely narrow true range
  (56.5-63.8 cells, reflecting its much milder contraction), and does
  not track the true contour's shape at all -- consistent with noise
  correlating weakly, by chance, with a low-variance true signal, not
  genuine shape recovery.
- Physical sanity checked? by whom?: Claude -- for part (a), verified
  the kappa=0 consistency check EXACTLY matched run -43 before trusting
  any further sweep result; for part (b), did NOT accept the script's
  own automatic "corr > 0.4 -> generalizes" verdict at face value --
  noticed the surrounding kappa values were negative, correctly
  identified this as a cherry-picking risk, and confirmed the negative
  read by inspecting the actual reconstructed curve rather than the
  correlation number alone. This is a real, self-caught correction, not
  a hypothetical caveat -- the tool's own printed output said
  "GENERALIZES" and that claim is explicitly retracted here.
- Gate passed? (Y/N): N/A -- channel-integration test, not a phase gate.
- Observations: run -43's speckle-constrained selection idea is
  confirmed as a genuine, robust win specifically on patient023 (now
  further improved to corr=0.738 with modest smoothness), but does NOT
  straightforwardly generalize to patient001. The most likely
  explanation, not yet confirmed: patient001's much milder contraction
  means its TRUE inner-radius variation across angles is small (a
  narrow ~7-cell range) relative to whatever noise floor this method
  operates at, making genuine shape-tracking much harder to distinguish
  from chance correlation -- patient023's much larger, more irregular
  anatomy and its own stronger true signal may be why the method worked
  there. This is an important boundary condition for the whole
  approach, not a reason to discard it -- but it means run -43/-44's
  results should NOT yet be described as "validated across patients,"
  only "validated on patient023, not yet replicated on patient001."
- Next action: (a) investigate WHY patient001 behaves so differently --
  compare the two patients' actual speckle-field quality/localization
  sharpness directly (patient001's outer radius estimate, 102.1 cells,
  is notably larger/different geometry than patient023's ~88-90 cells,
  worth checking if this affects the rising-edge extraction itself);
  (b) do NOT keep sweeping kappa and reporting the best result as if
  pre-registered -- if kappa needs tuning per-patient, that itself is a
  finding (the method is not turn-key), and any future kappa choice
  should be fixed BEFORE looking at accuracy, not selected after; (c)
  the off-center heart phantom test (still not done) remains a cleaner,
  synthetic third data point free of real-anatomy confounds. Not
  committed/pushed (standing instruction).
