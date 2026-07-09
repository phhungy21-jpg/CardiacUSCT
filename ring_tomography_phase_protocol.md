# Ring/dense-angular ultrasound tomography — phase protocol

**A lab-notebook-style working protocol, adapted from
`acoustic_simulation_phase_protocol.md` (the `jwave_test/` protocol).**
Version 0.1 · Working draft

## Multi-channel information roadmap (2026-07-09)

Per user: "only using all of that information can we precisely
reconstruct the tissue" — reflection + refraction + dispersion +
absorption + beam/echo divergence, not transmission time-of-flight
alone. Motivated directly by run -07's finding: transmission data
cannot see the blood/myocardium boundary (~0.5% sound-speed contrast);
a different physical channel (reflection amplitude depends on
impedance mismatch, not sound speed) might see it where transmission
can't. Standing discipline: add and validate ONE channel at a time
against a known case before fusing any of them — do not repeat
`jwave_test`'s early mistake of combining untested mechanisms at once.

Tractability order, and why:
1. **Reflection** — cheapest, most immediately actionable. Every
   near-angle receiver pair (within `_MIN_ANGLE_SEP_DEG` of the
   transmitter) has been EXCLUDED from every run so far as a
   "near-field artifact." That's wrong for the wider end of that
   window: a near-side receiver can't receive anything by straight
   transmission (no clear path through tissue to it), so anything it
   picks up is reflected/scattered energy — real reflection data this
   project has been discarding, not noise. Reuses `jwave_test`'s
   entire proven pulse-echo methodology (pitch-catch pairs, direct-
   arrival exclusion, envelope detection), just in the new geometry.
2. **Absorption** — NOT physically modeled at all yet (jWave's base
   solver has no attenuation term; `jwave_test` had to write a custom
   per-step damping solver, `attenuation_solver.py`, to add it,
   validated to ~0.1% against the analytic law). Until that's ported
   here, amplitude differences are geometric/transmission-coefficient
   noise, not real attenuation signal — a real physics addition
   needed, not just better data processing.
3. **Refraction** — the full-wave jWave simulation already includes it
   physically (Snell's-law bending happens automatically in a real
   wave-equation solve); the gap is on the RECONSTRUCTION side, since
   straight-ray backprojection/SIRT assumes rays travel in straight
   lines. Properly exploiting it means bent-ray tracing or full-
   waveform inversion (FWI) — a substantially bigger lift than anything
   built so far in this project.
4. **Dispersion** — frequency-dependent effects. Current arrival-time
   detection only uses envelope peak time, discarding the pulse's
   spectral shape entirely. Needs broadband spectral analysis (e.g.
   frequency-dependent attenuation, spectral centroid shift) — real
   but nontrivial signal-processing work, not yet started.
5. **Beam/echo divergence off curved surfaces** — this is exactly the
   "curvature-dependent reflection divergence" mechanism `jwave_test`
   diagnosed in its own run -44 (flat/large reflectors return energy
   almost only to the monostatic direction; curved/small reflectors
   scatter it widely) — directly relevant to how much weight reflection
   amplitude from different receiver angles should get once channel 1
   is built. Connects channel 1 to an already-characterized mechanism
   rather than a new one.

Currently only transmission time-of-flight (channel 0, already built,
runs -03 through -07) has been tested. Channel 1 (reflection) is next.

## Why this is a NEW project, not a continuation of `jwave_test/`

`jwave_test/` closed with a well-diagnosed finding: fine blind boundary
reconstruction from a sparse (4-16 probe), anterior-arc-only,
pulse-echo multistatic setup is a dead end for irregular real cardiac
anatomy. Bulk/aggregate contraction recovery survived, narrowly and
anatomy-dependently. See `jwave_test/LOG.md`'s "PROJECT CLOSURE" entry
for the full reasoning — read it before assuming anything here
supersedes it; it doesn't, it's a different acquisition geometry built
in response to what that closure diagnosed.

This project changes THREE things at once, each individually
significant enough to require fresh Gate 2 review, not an extension of
`jwave_test/`'s existing signoff:

1. **Acquisition geometry**: dense angular / ring coverage instead of
   4-16 sparse fixed probes. Per user decision (2026-07-09): assumes a
   **water-bath / full-surround setup** (patient immersed, probes
   effectively surround the torso near-360°, modeled on the real
   Butterfly/Midjourney-style whole-body USCT scanner) — NOT the
   clinically standard handheld transthoracic probe, which only has
   anterior acoustic access (no posterior window through lung/bone).
   This is a deliberate, explicit choice: it resolves the physical
   access-geometry conflict that a literal "ring around the chest"
   would otherwise have with `jwave_test/phase2_config.py`'s carried-
   over anterior-arc-only rationale, at the cost of this being a
   research/investigational acquisition mode, not standard
   point-of-care cardiac ultrasound. State this plainly in any writeup.
2. **Acquisition mode**: transmission tomography (time-of-flight →
   sound-speed map, amplitude loss → attenuation map) in addition to
   pulse-echo reflection — a genuinely different inverse problem
   (travel-time/full-waveform inversion) than backprojection.
3. **Motion handling**: modeled EXPLICITLY from Phase 2 onward, not
   discovered as an afterthought (the lesson from `jwave_test/`, where
   motion-during-acquisition was never modeled at all because the
   sparse setup was implicitly simultaneous-capture). A dense angular
   scan takes real time; if it's sequential (a literally rotating
   probe), different angles see different cardiac phases — a CT-style
   motion artifact. **Decided (2026-07-09, per explicit user choice
   "for data clarity"): SEQUENTIAL/rotating acquisition, not
   simultaneous multi-element capture.** Each element/angle fires and
   records in its own turn, keeping each measurement's data cleanly
   attributable to one transmit angle at one moment — the tradeoff
   being that this reintroduces the CT-style motion problem
   `jwave_test` never had to face (its sparse 4/8/16-probe setup
   captured all pairs against one frozen scene). Because of this
   choice, **explicit motion-during-acquisition modeling is now a hard
   Phase 2 requirement, not optional**: the forward model must inject
   the correct cardiac phase for each element's capture time (a
   time-varying medium across the scan, not one static medium sampled
   many times), and any Phase 3+ recovery method must account for the
   fact that different angles observed different instants of the
   cardiac cycle.

## Phase 0 — Setup and honest scoping

### 0.1 Prepare
- [x] Confirm the water-bath/full-surround geometry decision is durable
  (recorded above; revisit only with an explicit reason).
- [x] Decide simultaneous-ring-capture vs. sequential/rotating
  acquisition BEFORE writing the Phase 2 forward model (see point 3
  above) — **DECIDED: sequential**, per explicit user choice. This is a
  first-principles modeling choice, not a detail to discover
  empirically.
- [ ] Compute budget. Ring/dense-angular geometries mean far more
  tx/rx pairs than `jwave_test`'s 4-16-probe setup (N probes -> up to
  N² pairs) — get a real per-transmit cost estimate early and multiply
  honestly before committing to element count.

### 0.2 Do
- Write this protocol's Phase 2 config (tissue properties, water
  coupling medium, ring/arc geometry, grid/timestep) before any
  simulation code.

### 0.3 Gate 0
- [ ] Access geometry, acquisition mode, and motion-handling strategy
  are all decided and written down (not implicit).
- [ ] Reused assets identified (see `jwave_ring_tomography/MANIFEST.md`)
  and NOT-reused assets identified (the closed sparse-probe
  backprojection code stays in `jwave_test/`, frozen).

---

## Phase 1 — Scout: does the ring/water-bath forward model even run?

Same spirit as `jwave/`'s Phase 1 scout (point source, two-tissue
reflection, array source) but for the new geometry: water as the
homogeneous background/coupling medium (not `CHEST_WALL_PROXY`), a
ring or dense arc of source/receiver elements around it, and — new —
a transmission (straight-through) capture in addition to pulse-echo.

### 1.1 Do
- Point source in a homogeneous water bath — sanity-check wave speed,
  stability, PML at this new geometry's larger domain (a ring/water-
  bath domain is bigger than `jwave_test`'s anterior-arc domain).
- Two-tissue reflection (water/chest-wall-proxy interface) — same
  purpose as the original `jwave/` scout: is the reflection physically
  sensible before adding real anatomy.
- Full ring array source/receive smoke test — confirm every element's
  geometry (position, look direction) is correct BEFORE any inversion
  code is written (this project's predecessor found and fixed several
  geometry bugs the hard way; verify analytically this time first).

### 1.2 ⛔ Gate 1
- [ ] Ring/water-bath forward model runs without instability (NaN,
  divergence) at the target grid resolution.
- [ ] Wave speed and basic reflection physics confirmed sane (same
  checks as `jwave/` Phase 1, adapted for water background).

---

## Phase 2 — Acoustic model definition (water tissue properties, ring geometry, motion-capture mode)

### 2.1 Prepare
- **Cited tissue properties.** Reuse `jwave_test/src/phase2_config.py`'s
  BLOOD/MYOCARDIUM/CHEST_WALL_PROXY (already cited, already
  Gate-2-reviewed once for the pulse-echo setup — the values themselves
  don't change, only the geometry around them does). ADD a WATER
  coupling-medium property (new, not previously needed) — must be
  cited, not invented, per the standing project rule. A candidate
  citation: Duck, F.A. (1990) "Physical Properties of Tissue," water
  entry (sound speed ~1480-1520 m/s depending on temperature,
  density ~993-1000 kg/m^3 at body-adjacent bath temperature,
  attenuation ~0.0022 dB/cm/MHz^2 — negligible relative to tissue).
  **This is a PROPOSAL pending real collaborator confirmation, exactly
  like every other cited value in this project — do not treat as
  final until Gate 2.**
- **Geometry.** N elements (TBD, budget-dependent) around a ring/arc
  approximating full-surround water-bath access; explicit
  transmission (opposite-side) AND reflection (near-side) pair capture,
  not reflection-only.
- **Motion-capture mode: SEQUENTIAL (decided, Phase 0.1).** Each
  element/angle's transmit-receive event happens at its own moment in
  the scan, so the forward model must inject the correct cardiac phase
  for that element's capture time — a time-varying medium sampled once
  per element as the scan proceeds, NOT one frozen scene reused for
  every element the way `jwave_test`'s simultaneous multistatic capture
  worked. This means the medium-building step needs a scan-time ->
  cardiac-phase mapping (e.g. a scan rate and a cardiac cycle length)
  decided explicitly here, not left implicit.

### 2.2 Do
- Build the ring/water-bath medium + full tx/rx pair capture, reusing
  `jwave_test`'s `capture_all_pairs`-style pattern generalized to many
  more elements and both reflection and transmission legs.

### 2.3 ⛔ Gate 2
- [ ] Every acoustic property (including the NEW water value) has a
  cited source.
- [ ] **A collaborator with acoustic-physics expertise has reviewed and
  signed off on THIS setup specifically** — `jwave_test`'s Gate 2
  signoff (run -12) covered the anterior-arc pulse-echo setup, not
  this water-bath/transmission-tomography one. Not passable solo, same
  as before.

---

## Phase 3 — Toy proof-of-concept: does tomographic inversion recover a simple, MOVING phantom

Mirrors `jwave_test`'s Phase 3, but the recovery method is now a real
tomographic inversion (travel-time/sound-speed reconstruction, and/or
full-waveform inversion), not a backprojection-based shape-fit or blind
per-angle discovery — those were characterized and closed for the
sparse pulse-echo setup; this is a different inverse problem with
different, better-established literature (X-ray-CT-like travel-time
tomography, FWI) to draw on.

### 3.1 Prepare
- A simple 2D moving phantom, reusing `jwave_test`'s toy ring/heart-
  cartoon phantoms where useful, but now inside a water bath with full
  angular tx/rx coverage.

### 3.2 Do
1. Simulate transmission + reflection data across the full ring/arc.
2. Recover a sound-speed (and/or attenuation, and/or boundary) map via
   an actual tomographic inversion — not a scale-sweep template fit.
3. Compare recovered vs. prescribed motion, dialing acoustic realism
   from gentle to aggressive (Gate 3's explicit requirement, which
   `jwave_test`'s bulk-contraction channel never got to close).

### 3.3 ⛔ Gate 3
- [ ] Recovery meaningfully beats a naive baseline on the toy phantom.
- [ ] Realism dialed gentle → aggressive with observed, understood
  degradation (this is the checkbox `jwave_test`'s most promising
  result — bulk contraction — never actually closed; close it properly
  here from the start).

---

## Phases 4-6

Unchanged in spirit from `acoustic_simulation_phase_protocol.md`'s
Phase 4 (scale to cardiac anatomy, compute-heavy, budget-scoped —
reuse Phase I ACDC/M&Ms anatomy exactly as before), Phase 5
(characterize degradation vs. physical conditions — this project's
extra physical conditions now include ring coverage density, water-bath
temperature/sound-speed uncertainty, and motion-capture mode), and
Phase 6 (interpretation/writeup/escalation decision). Do not copy
those sections verbatim without re-reading them against this project's
actual acquisition model — the compute multiplier in particular will
differ substantially (many more tx/rx pairs than the 4-16-probe case).
