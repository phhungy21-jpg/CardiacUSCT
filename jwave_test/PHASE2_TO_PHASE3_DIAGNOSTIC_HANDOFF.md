# Phase 2 → Phase 3 diagnostic handoff

**Purpose:** a pre-Phase-3 safety checklist for the specific silent-failure
modes that make acoustic simulation dangerous (a wrong setup produces a
clean, plausible-looking wavefield, not an obvious error). This is not a
generic warning list — each item below was checked against *this* repo's
actual Phase 2 config and code, not assumed. Status: CONFIRMED items are
verified by reading source or running numbers; OPEN items are unresolved
design questions, not yet checked; OK items have a documented reason for
being acceptable at this stage.

**This handoff blocks Phase 3 the same way Gate 2 does** — it requires
collaborator engagement, not just a solo read-through. Two items below
(#4, and to a lesser extent #5) are load-bearing for what Phase 5
ultimately claims to characterize.

---

## 1. Numerical dispersion / points-per-wavelength — OPEN, needs collaborator input

jWave's time-domain solver is a Fourier pseudo-spectral method, not a
finite-difference stencil — it doesn't have classical FD numerical
dispersion, but it does alias/distort content above its resolvable
bandwidth. Current config (`phase2_config.py`): dx=0.1mm, f0=2.5MHz,
c≈1576–1584 m/s → λ≈0.63mm → **6.3 points/wavelength at the carrier**.
But the 3-cycle Gaussian toneburst is not monochromatic — its fractional
bandwidth is roughly ±1/T_burst ≈ ±0.8MHz around 2.5MHz, so energy extends
toward ~4MHz, where points/wavelength drops to **~3.9**. That's still
above the Nyquist floor (2 pts/wavelength) but below the ≥6–8 pts/wavelength
margin often recommended for heterogeneous media with sharp interfaces.
**Not yet determined whether this margin is adequate** — this is exactly
the kind of check that needs a specialist's judgment, not a rule of thumb
found by us. Bring this specific number (6.3 pts/wavelength at carrier,
~3.9 at burst edge) to the collaborator.

## 2. PML / absorbing boundary — OK for now, not rigorously verified for this config

jWave's default PML (`pml.py`, `td_pml_on_grid`) is a standard exponential-
damping layer, `pml_size=20` grid cells (2mm at our resolution), default
`alpha_max=2.0`. In the Phase 1 scout runs (`../jwave/`, homogeneous
128×128 domain), the wavefront was tracked crossing nearly the full domain
and faded cleanly with no visible reflected artifact returning — weak
evidence the default PML behaves reasonably at this dx/frequency. **Not
yet quantitatively tested**: we have not measured the actual reflection
coefficient off the domain edge for this specific config (dx=0.1mm,
f0=2.5MHz, 300×300 domain), and the Phase 2 forward-model run
(`phase2_forward_model.py`) was deliberately truncated (`t_end` cut to 60%
of full domain-crossing time) specifically so the wave *wouldn't* reach the
edge yet — meaning **this run provides zero evidence about PML behavior at
this config**. Recommended before Phase 3: rerun with full `t_end` and
check the trailing wavefield for reflected energy re-entering the domain.

## 3. Source scaling / normalization — CONFIRMED, not calibrated (acceptable for now)

All source amplitudes (`p0`, toneburst signals) in every script so far use
arbitrary unit amplitude (1.0), not calibrated to a physical transducer
output (Pa, MI, or acoustic power). This is fine for *within-config*
qualitative/relative comparisons (e.g. the blood/myocardium reflection
ratio check in `../jwave/`, which only needed self-consistent relative
amplitudes) but **is not valid** for anything requiring absolute physical
scale — e.g. claiming a specific SNR, comparing absolute pressure across
different configs, or computing mechanical index / safety limits. Must be
addressed before any quantitative attenuation/SNR claim in Phase 5.

## 4. Attenuation / absorption modeling — CONFIRMED CRITICAL GAP

**This is the load-bearing finding of this handoff.** Verified by reading
jWave's source directly (`venv/Lib/site-packages/jwave/acoustics/`):

- `time_varying.py`'s `simulate_wave_propagation` — the transient,
  pulse-echo, time-domain solver used in **every single simulation in this
  project so far** (all `../jwave/` scout runs, and `phase2_forward_model.py`)
  — calls `mass_conservation_rhs` and `momentum_conservation_rhs`. Neither
  function references `medium.attenuation` anywhere (grep-confirmed, and
  read in full — only `medium.density` and `medium.sound_speed` appear in
  the PDE update).
- `medium.attenuation` **is** used, but only in `operators.py`'s
  `wavevector()` function, which is exclusively called by
  `time_harmonic.py`'s Helmholtz/Born-series/angular-spectrum solvers — a
  **continuous-wave, frequency-domain** simulation paradigm, structurally
  different from the transient pulse-echo simulation this project needs.

**Consequence:** `phase2_config.py`'s cited attenuation values (blood
0.20, myocardium 0.52, chest-wall-proxy 0.74 dB/cm@1MHz) are currently
inert — assigned to the config, cited correctly, but **not affecting any
simulation we've run**. Every wavefield produced so far, including the
blood/myocardium reflection-coefficient check, was computed in a
perfectly lossless medium. This matters most because **attenuation-driven
shadowing/dropout is explicitly named in the protocol's Phase 5 sweep
conditions** — the planned characterization cannot be produced by the
current time-domain solver as used, regardless of how correct the cited
values are.

**Options, for collaborator discussion (not a decision to make solo):**
- (a) Implement power-law absorption manually in jWave's time-domain PDE
  (nontrivial numerics — a fractional-Laplacian or relaxation term needs
  to be added to `mass_conservation_rhs`/`momentum_conservation_rhs`, or
  the underlying `jwave`/`jaxdf` library extended).
- (b) Switch the transient forward model to **k-Wave**, which implements
  power-law absorption natively in its time-domain solver — this may
  reopen the Phase 1.1 k-Wave-vs-jWave decision specifically for this
  reason (jWave's differentiability was the reason jWave was chosen; if
  attenuation is a hard requirement and difficult to add to jWave, that
  tradeoff needs to be re-weighed with the collaborators).
- (c) Approximate attenuation post-hoc (e.g. apply a frequency-dependent
  filter to the lossless output) — a common approximation in some
  pipelines, but needs explicit collaborator sign-off that it's adequate
  for this study's claims, since it's physically different from
  attenuation acting during propagation (e.g. it won't naturally produce
  correct shadowing behind attenuating structures).

This should be treated as a **blocking item for Gate 2**, not a Phase 3/4
cleanup task — it changes what tool/method Phase 2's forward model even
is.

## 5. Moving-medium representation — OPEN, explicit design decision needed before Phase 3

Not yet addressed at all (Phase 2 so far is entirely static geometry).
Phase 3 will need to inject the phantom's prescribed motion into the
acoustic sim somehow — e.g. re-drawing `medium.sound_speed`/`density`
maps at each transmit event (medium "jumps" between static frames), vs.
a continuously-deforming boundary/mesh. **The risk flagged is real and
specific to this project**: if motion is injected in a way that leaves any
detectable systematic signature (e.g. per-frame grid-aliasing artifacts
that correlate with the direction of motion), Phase 3's recovery model
could partially lock onto that artifact rather than genuine wave-physics
motion encoding — inflating recovery accuracy in exactly the direction
that would look like success. **Recommendation:** pre-register the
motion-injection method and, if possible, a null test (e.g. recover
"motion" from a static/non-moving control run through the same injection
pipeline — recovered motion should be ~zero) before trusting any Phase 3
recovery number.

## 6. Grid staircasing at tissue boundaries — OK for the current idealized phantom, OPEN for Phase 4

The Phase 2 ring phantom uses hard circular masks (binary tissue-property
assignment per grid cell) at r=60 and r=90 grid cells (6mm/9mm) on a
0.1mm grid — staircase roughness is ~1 grid cell against a 60–90 cell
radius (~1–1.7%), and the wavefront panels
(`results/figures/phase2_ring_phantom_single_transmit.png`) show no
visible spurious scattering at the myocardium boundary. **Likely fine for
this idealized circular case.** Real ACDC/M&Ms segmentation boundaries
(Phase 4) are irregular and pixel-level jagged at their native ~1.56mm
resolution, and will need upsampling to the ~0.1mm acoustic grid —
per CLAUDE.md, that resampling **must use nearest-neighbor**, which
means real anatomical boundaries will be more staircased than this
synthetic circle, not less. Reassess when Phase 4 anatomy is used; not
blocking now.

---

## Phase 3→4 hard gate (added after Phase 3 ran — see LOG.md run -07)

Phase 3's toy proof-of-concept correctly did NOT need items 3 (source
scaling), 4 (attenuation), or 6 (staircasing) resolved — a controlled toy
with prescribed, exactly-known motion doesn't require physical realism to
validly test whether the recovery *mechanism* works. That reasoning does
NOT extend to Phase 4. This is now a **hard gate, not a soft intention**:

**Phase 4.2 ("Do" — generating the real simulated dataset) is BLOCKED
until items 3, 4, and 6 have collaborator signoff.** Specifically:
- Item 4 (attenuation) — jWave's transient solver doesn't implement it at
  all; Phase 5's shadowing/dropout characterization is meaningless without
  it. Must be resolved (implement in jWave / switch to k-Wave / approved
  approximation) before generating data that Phase 5 will analyze.
- Item 3 (uncalibrated source amplitude) — Phase 5's SNR/noise-level
  sweeps need a physically anchored scale, not arbitrary units, or "noise
  level X" is meaningless outside this codebase.
- Item 6 (staircasing) — untested at real ACDC/M&Ms segmentation
  resolution (Phase 2/3 only used an idealized smooth circle); must be
  checked once real anatomy is used, before trusting any boundary-adjacent
  result.

**Phase 4.1 (Prepare — benchmarking timing, scoping compute against
budget) is NOT blocked** by this gate, since it doesn't produce results
that feed Phase 5's conclusions — only a timing number and a scope
decision. Proceeding with 4.1 now, explicitly labeled preliminary/pending
signoff throughout (see below), is consistent with the standing gate.

**Labeling requirement, going forward:** every figure, log entry, and
result produced by this codebase prior to collaborator signoff — Phase 3's
outputs included — must carry an explicit "PENDING ACOUSTIC-PHYSICS
SIGNOFF" label so no later phase transition silently treats a toy/
preliminary result as an established one. See `src/labels.py`.

## Ground-truth-motion floor — restate wherever acoustic recovery accuracy is reported

Phase 3's ground truth was a toy's exactly-prescribed motion (no
registration involved) — its RMSE numbers have no ground-truth floor and
are not comparable across phases on that basis alone.

**Starting at Phase 4**, ground truth switches to Phase I's
registration-derived motion fields (`pilot/data/processed/ACDC_reg/*.npz`),
which are NOT exact. Per `pilot/LIMITATIONS.md` Gap 1: registration-derived
motion has a **median boundary error of ~1 voxel (~1.5mm)**, rising to
**~2-3 voxels (~3-4.5mm) for patients with large ED→ES contraction**
(predominantly NOR/HCM pathology groups), with **RV consistently the least
accurate structure** (median ~1.1 voxel, worse tail than LV/myocardium).
Through-plane (base-apex) motion is also under-resolved by the cohort's
10mm slice thickness.

**Every reported acoustic-recovery-accuracy number from Phase 4 onward
must restate this floor in the same breath** (e.g., "acoustic recovery
endpoint error = X mm, against a ground truth with its own ~1.5mm median
floor (rising to ~3-4.5mm for high-contraction patients) — X should not be
read as absolute accuracy, since it's relative to an already-imperfect
ground truth"). This prevents "acoustic recovery is X% accurate" from
being misread as "the method recovers true cardiac motion to X% accuracy"
— a distinct, stronger, and unsupported claim.

## Bottom line

Items 3 and 6 are acceptable as-is with documented caveats. Item 1 and 2
are open technical checks that need either collaborator judgment (1) or a
rerun we haven't done yet (2). **Item 4 is a confirmed, structural gap**
that must be resolved — and item 5 is a design decision that must be made
deliberately — before Phase 3's recovery loop is built, because both
affect whether Phase 3–5's results would even be measuring what the
protocol says they measure. This is now the concrete blocking item to
bring to the Yale acoustic-physics collaborator, alongside the earlier
blood/myocardium weak-contrast finding.
