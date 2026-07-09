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
