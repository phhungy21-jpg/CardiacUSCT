# Session handoff — 2026-07-08

Consolidated status across an extended, multi-thread exploration session
following Gate 2's real signoff (see `PROXY_AUDIT.md`, `LOG.md` run -12).
Written to let a fresh session resume with full context rather than
re-deriving any of this. Every claim below is backed by a `LOG.md` run
entry — check those for full numeric detail; this document is the map.

## What's solid and settled (don't re-litigate)

- **Gate 1 PASSED** (run -10): jWave GPU-timed reference reproduction,
  Colab Tesla T4.
- **Gate 2 PASSED** (run -12): real Yale acoustic-physics collaborator,
  unconditional approval of `PROXY_AUDIT.md`'s 6 items.
- **Attenuation solver implemented and validated** to ~0.1% against the
  analytic law (`attenuation_solver.py`, runs -11).
- **Phase 4.1 pilot dataset built and working**: 29 real cases, 3
  patients, resumable driver confirmed (run -13).
- **Realistic SNR calibration**: this project's tested noise levels
  (2%/5%/10% = 34/26/20dB) bracket real clinical ultrasound SNR,
  specifically its harder end (run -16) — the fragility found below is
  not a toy-only concern.
- **Phase 4.2 (the real, 150-patient dataset) remains blocked** on Gate
  4's compute-budget-agreement checklist — separate from Gate 2, still
  open.

## Thread 1: single-echo boundary detection — CLOSED, modest partial gains

Progression: envelope-threshold (Level 0) → matched filter (Level 1) →
reference-anchored narrow-window tracking (Level 2) → 7-channel
beamforming (Level 3) → 2-pair naive fusion (Level 4). Each step gave a
real, verified RMSE improvement (runs -07, -14, -15, -16), but **none
beat the naive constant-baseline once realistic noise (>=2%) is
present**, and the improvement plateaus. Root cause, confirmed: two
reflecting interfaces (chest-wall/myocardium, blood/myocardium) are
comparably weak (R~0.002-0.0035, cited values), making single-echo
amplitude/correlation detection structurally ambiguous between them.

**Verdict: this track is closed.** Better single-echo detectors will not
fix a structural ambiguity between two similarly-weak features. Do not
resume without a new idea (e.g. a fundamentally different discriminating
feature between the two interfaces).

## Thread 2: distributed speckle tracking — OPEN, unresolved

Three iterations (runs -17, -18), each ruling out one hypothesis without
producing a working result:
1. Single-channel tracking of a mid-wall material point: worse than
   boundary tracking. Root cause investigated: NOT a noise-scaling
   fairness bug (checked and rejected).
2. Multi-channel (7-element) beamformed version: only modest ~3-4%
   improvement, far short of the ~sqrt(7) expected from ideal coherent
   averaging.
3. Sequential (frame-to-frame) tracking with corrected boundary
   exclusion: RMSE plateau essentially unchanged after the fix — proving
   boundary contamination was NOT the dominant cause.

**Leading unresolved hypothesis**: 400 scatterers is too sparse for
fully-developed speckle statistics (rule of thumb wants ~10+ scatterers
per resolution cell; this project's phantom is well short of that). NOT
tested. This is the single most important open question for genuine
vector/strain motion recovery in this project (see Thread 3).

**If resuming**: the next concrete test is a much denser scatterer field
(thousands, not hundreds) in `phase3_speckle_tracking.py`'s framework —
a real compute/validation increment, not a quick tweak.

## Thread 3: multi-angle vector triangulation — CLOSED, real physical limitation found (RETRACTED positive result)

Initially reported as a genuine positive result (run -19: recovered a 2D
displacement vector via two focused sub-apertures, cross-range RMSE
0.499mm noiseless, seemingly robust to noise in run -20). **This result
was RETRACTED (run -21)** after continued debugging (per explicit user
instruction) revealed:

1. A real timing-formula bug (naive symmetric transmit-time
   approximation, wrong for asymmetric sub-apertures) that partly
   explained the original "surprising" A-vs-B finding.
2. **A deeper, structural, physically-fundamental limitation**: for a
   curved reflecting boundary, the sensitivity vector of ANY valid
   specular (Tx,Rx) pair is always exactly parallel to the target's
   local surface normal — confirmed analytically (law of reflection: u_in
   and u_out are symmetric about the normal, so their difference is
   always normal-directed) and numerically (identical unit sensitivity
   direction across 7 scanned transmit positions spanning a wide angle
   range).

**Verdict: multi-angle triangulation of a SINGLE point via specular
reflection cannot recover a true 2D displacement vector, structurally.**
No implementation fix changes this — it's a property of specular
reflection physics, not this project's code. Do not attempt further
single-point multi-angle specular variants. Genuine 2D/vector recovery
needs Thread 2 (speckle, where a discrete scatterer's arrival time DOES
depend on true 2D position) or multi-POINT differential strain tracking
(infer tangential motion from how multiple points' individual
normal-displacements vary spatially — the actual mechanism real 2D
speckle-tracking echocardiography uses).

## Thread 4: DAS (delay-and-sum) beamformed imaging — OPEN, real progress, one theory question left

Per explicit user direction, following Thread 3's finding: reconstruct a
full spatial image (many receive channels, one broad transmit, per-pixel
coherent summation) and track features WITHIN it, rather than trying to
triangulate a single point's vector directly. This is literally how
clinical B-mode/synthetic-aperture ultrasound works, and structurally
avoids Thread 3's normal-only limitation.

`phase3_das_beamforming.py`. **Four distinct, real bugs found and fixed
in sequence** (run -22), each caught by validating against known ground
truth before moving to the next:
1. Off-by-one time-axis length mismatch (`round()` vs `TimeAxis.Nt`'s
   `ceil()`).
2. Axis-order inconsistency vs. every other script this session
   (`field[:, row, col]` vs the established `field[:, col, row]`).
3. Unfocused point-source transmit: real echo ~1700x weaker than direct
   wave (physics, not a bug) — fixed by switching to a focused transmit.
4. "Virtual point source at focus" approximation, wrong for targets
   BEFORE the focus (our boundary sits before the ring-center focus, in
   the still-converging near field) — fixed with a true per-element
   earliest-arrival calculation.

**After these 4 fixes: ED frame reconstructs correctly (0.61mm error).**

**One issue remains, diagnosed but NOT fixed**: ES's reconstruction is
confounded by a fixed-depth (row~66) artifact. **Confirmed via a clean
control test (homogeneous medium, no ring/reflector at all) that this
artifact is a pure array/beamforming-geometry effect, not a tissue
signal** — likely the direct (unreflected) transmit wave being
coherently mis-summed due to an inconsistency between the "earliest
per-element arrival" transmit-time model and the receive-delay model.

**This is a genuine open theory question, not a quick patch.** The real
fix likely requires either: (a) extracting the transmit arrival-time map
directly from simulated reference data (a separate reference simulation)
instead of an analytic per-element formula, or (b) more careful
mathematical treatment of coherent multi-element summation (proper
beamforming point-spread-function theory).

**If resuming**: start from `phase3_das_beamforming.py` as-is (working
for ED). Investigate the row~66 artifact using the homogeneous-medium
control test already built into the debugging history (see LOG.md run
-22) as a starting point — any fix should be verified to leave the
homogeneous-medium case artifact-free before declaring success.

## Thread 5: 4-probe boundary tracking around a bounding square — split result (unfocused: real success; focused: same open artifact as Thread 4)

Per user request: 4 probes (top/bottom/left/right, each 12mm from the
ring center, matching the exact validated single-probe distance) doing
per-frame single-pulse transmit/receive with the already-proven Level 2
detector.

**Symmetric (isotropic) phantom** (`phase3_four_probe_tracking.py`, run
-23): worked cleanly on the FIRST attempt — all 4 probes gave IDENTICAL
RMSE (0.5625mm), confirming the method generalizes correctly across
orientations with no rotation-dependent bugs. A validation result (an
isotropic phantom SHOULD look identical from every direction), not yet
new information.

**Asymmetric (regional hypokinesis) phantom** (`phase3_asymmetric_phantom.py`,
`phase3_four_probe_asymmetric.py`, run -24): a clinically-realistic test
(one region contracting to only 30% of normal amplitude, smoothly
tapered, centered on the LEFT probe). **With unfocused (simple 2-element
pitch-catch) probes, the prediction did NOT hold cleanly**: right/bottom
tracked consistently (as expected, far from the affected region), but
top (should be unaffected) showed unexpectedly reduced motion, and left
(should show the LARGEST reduction) instead tracked near-NORMAL motion —
the opposite of the prediction. Hypothesis: unfocused probes have wide
angular sensitivity and pick up a blend of reflections from a region
wider than directly on-axis, so left ends up sensing adjacent
(stronger, non-hypokinetic) tissue instead of its own region.

**Attempted fix: focused (8-element delay-steered) probes**
(`phase3_four_probe_focused.py`) — **hit the exact same artifact class
as Thread 4**: all 4 probes showed EXACTLY ZERO recovered contraction,
completely stuck. This confirms Thread 4's row-66 artifact is not a
one-off DAS-specific bug — **multi-element delay-focusing in this
codebase has a recurring, unified, unsolved artifact problem** that
shows up wherever coherent multi-element summation is used, not
specific to any one script.

**Standing, valid result from this thread**: the UNFOCUSED 4-probe test
demonstrates that simple point-like probes correctly recover motion
MAGNITUDE (isotropic case) but do NOT reliably LOCALIZE regional motion
differences — a real, informative capability limitation, independent of
the (currently broken) focused-probe attempt.

## Recommended priority for a fresh session

1. **Threads 4 and 5's focused-probe failure are ONE unified open
   problem, not two.** Whenever this codebase uses multi-element
   delay-focused transmit (DAS beamforming or focused point-tracking
   probes), a strong fixed/coherent-summation artifact can dominate the
   real, weaker, moving tissue signal. Confirmed via TWO independent
   control tests (DAS's homogeneous-medium test; the focused-probe
   all-zero-contraction result). **Solving this once, properly — likely
   via a data-driven transmit arrival-time map or rigorous beamforming
   point-spread-function treatment — unblocks both Thread 4 (DAS) and
   Thread 5's focused-probe localization problem simultaneously.** This
   is the single highest-value thing to fix next.
2. **Thread 2 (speckle density)** is the other genuinely open question,
   and is likely a prerequisite for DAS-based tracking to work on
   anything other than the strong specular boundary (real cardiac
   ultrasound relies heavily on myocardial speckle, not just boundary
   echoes).
3. **Threads 1 and 3 are closed** — their limitations are structural/
   physical, not implementation gaps. Do not resume without a
   genuinely new idea.
4. **Thread 5's unfocused-probe result stands on its own** as a real,
   usable finding (motion magnitude: yes; regional localization: no,
   without focusing) — usable for further work even before the unified
   focusing-artifact problem is solved.

## Standing reminders (do not forget across a context reset)

- Every result in `jwave_test/` carries `labels.PENDING_SIGNOFF_BANNER` —
  none of this is collaborator-reviewed physics (that's `PROXY_AUDIT.md`,
  which IS reviewed — don't confuse the two).
- Ground truth in every Phase 3 script is TOY exact prescribed motion,
  not Phase I's imperfect registration-derived motion — `GT_FLOOR_CAPTION`
  applies starting at Phase 4, not here.
- Attenuation, calibration, and staircasing are proxy-audited and
  Gate-2-approved (`PROXY_AUDIT.md`) — safe to build on.
- Phase 4.2 (real 150-patient dataset) stays blocked on Gate 4's
  compute-budget agreement — a separate, still-open item.
