# MANIFEST — jwave_test/ (Phase II, acoustic-simulation layer — Phase 2 active workspace)

**Cloned from `../jwave/` on 2026-07-07** to begin Phase 2 (acoustic model
definition). `../jwave/` is now frozen as the Phase 1 exploratory-scout
record — do not treat its scout-run findings (e.g. the blood/myocardium
weak-contrast result) as established until a named acoustic-physics
collaborator reviews them (Gate 2 requirement). This directory is where
active Phase 2 work happens; content below is inherited from the clone and
is updated as Phase 2 proceeds.

Tracks on-disk state for the acoustic-simulation phase so a new session
doesn't have to re-derive it by exploring the filesystem. Update this when
the directory structure or generated artifacts change materially.

## Layout

- `jwave_test/src/` — simulation, forward-model, and recovery code for this
  phase. Inherited from `jwave/` at clone time: `toy_2d_homogeneous.py`,
  `toy_3d_homogeneous.py`, `toy_2d_array_source.py`,
  `toy_2d_two_tissue_reflection.py` (Phase 1 exploratory scout scripts —
  see `jwave/MANIFEST.md` for details, unchanged here). Phase 2 additions:
  `phase2_config.py` (cited tissue properties for blood/myocardium/
  chest-wall-proxy, anterior-array transducer geometry, grid/CFL — see
  its docstring for full citations) and `phase2_forward_model.py`
  (synthetic ring-phantom single-transmit demo, N=(300,300)/30mm domain,
  ran successfully with a documented stability check — see LOG.md run
  2026-07-07-06). Phase 3 additions: `phase3_config.py` (self-chosen toy
  cardiac-cycle motion parameters, not cited physiological values) and
  `phase3_motion_recovery.py` (per-frame frozen-scene motion injection,
  pitch-catch pulse-echo range recovery, envelope-based first-crossing
  detection, null test — see LOG.md run 2026-07-07-07 for two bugs found
  and fixed during validation, and the null-test result). See
  `PHASE2_TO_PHASE3_DIAGNOSTIC_HANDOFF.md` for the full pre-Phase-3
  diagnostic — item 4 (attenuation) remains an open, confirmed critical
  gap deferred to before Phase 4, per explicit user scoping decision that
  Phase 3's toy proof-of-concept doesn't require physical-realism items
  resolved, only the motion-injection risk (item 5, now null-tested).
  `labels.py` — shared `PENDING_SIGNOFF_BANNER`/`GT_FLOOR_CAPTION`/
  `TOY_EXACT_GT_CAPTION` applied to every figure/result in this codebase
  (see the handoff doc's "Phase 3→4 hard gate" section — Phase 4.2 dataset
  generation is blocked on collaborator signoff; Phase 4.1 prepare is not).
  Phase 4 addition: `phase4_benchmark.py` — Phase 4.1 benchmark-then-
  multiply, **first use of real Phase I anatomy** in this project
  (patient001, nearest-neighbor resampled per CLAUDE.md), timed at 3 small
  grid sizes and extrapolated to the real full-heart FOV (~91s, ~13.8GB
  per transmit at N~900 — infeasible on this CPU machine, confirming the
  protocol's GPU/cluster requirement for Phase 4.2). See LOG.md run
  2026-07-07-09. **Correction (run -11):** N=150/250 crops turned out to
  be entirely inside the LV cavity (no tissue boundary) — only N=350 has
  genuine heterogeneity; timing/memory numbers unaffected, but the "real
  anatomy" framing for those two points was overstated.
  `attenuation_solver.py` + `validate_attenuation.py` — a reimplementation
  of jWave's transient scan loop with per-step exponential absorption
  damping (both density AND velocity — a first version only damped
  density and was validated as ~2x wrong before the fix), validated
  against the analytic exp(-alpha*distance) law to ~0.1%. `calibration.py`
  — proposed (not collaborator-confirmed) source-amplitude calibration via
  FDA Mechanical Index. `phase4_demo_attenuating_real_anatomy.py` —
  combines real anatomy + validated attenuation + calibrated amplitude
  into one working forward model (N=350, stable, 0 NaN, Pa-scale output).
  See `PROXY_AUDIT.md` and LOG.md run 2026-07-07-11 for the full writeup.
- `jwave_test/venv/` — local CPU-only Python venv (jax 0.4.38, jwave 0.2.1),
  recreated fresh at clone time (not copied from `jwave/venv/`). Gitignored.
- `jwave_test/data/` — gitignored (matches root `.gitignore`'s `data/`
  pattern). Reuses Phase I's ACDC/M&Ms-derived anatomy/motion as *input* to
  simulation, read from `../pilot/data/`.
- `jwave_test/results/` — gitignored artifacts; figures/metrics/logs from
  this phase's runs go here.
- `jwave_test/notebooks/` — `phase1_gate1_reference_repro.ipynb` —
  **Gate 1 PASSED** (run 2026-07-07-10): executed on Colab (Tesla T4) via
  the VS Code Google Colab extension. jax 0.4.38, jwave 0.2.1, GPU
  confirmed (`CudaDevice`), timing 53.1ms±7.11ms/loop, wavefront visually
  confirmed circular. Notebook now includes two environment fixes baked
  in (jwave has no `__version__`; jax/CUDA-plugin version skew on Colab
  needing an explicit `jax[cuda12]==0.4.38` reinstall + runtime restart).
- `jwave_test/LOG.md` — running lab notebook for this phase (Appendix C
  entries), continuing from the cloned `jwave/LOG.md` history.
- `jwave_test/requirements.txt` — this phase's pinned dependencies.

## Status

Phase 0 (Gate 0) passed 2026-07-07 (see LOG.md), with a flagged caveat:
physical-correctness ownership for Gate 2 is shared/per-run rather than one
named collaborator — must resolve to an actual reviewer before Gate 2 can
pass. **Phase 1's Gate 1 PASSED 2026-07-07** (run -10, GPU-timed reference
reproduction on Colab Tesla T4: 53.1ms±7.11ms/loop, versions pinned in
`requirements.txt`) — note this is jWave's small 128×128 reference case,
not yet a real-anatomy-scale GPU number (that still needs to happen with
the collaborators, alongside the CPU-based extrapolation from run -09).
Phase 2
(acoustic model definition) is now starting in this directory: tissue
acoustic properties, transducer geometry, grid resolution/timestep, and
2D-first dimensionality decision, per protocol 2.1–2.2. Gate 2 requires a
named acoustic-physics collaborator's signoff — not passable solo.

**Phase 2.1/2.2 done (run 2026-07-07-06): a cited-value ring-phantom
forward model runs successfully.** `PHASE2_TO_PHASE3_DIAGNOSTIC_HANDOFF.md`
documents a confirmed critical gap — jWave's transient time-domain solver
(used throughout this project) does not implement attenuation at all,
despite cited attenuation values being present in the config — plus other
open items (points-per-wavelength margin, unverified PML at this config,
uncalibrated source amplitude). **This still blocks Phase 4** (the real
study) until resolved with the collaborators, independent of Gate 2's
standing signoff requirement.

**Phase 3.1/3.2 done (run 2026-07-07-07): toy simulate→recover loop with
null test.** Per explicit user decision, Phase 3 proceeded without
resolving the physical-realism diagnostic items (attenuation, calibrated
scaling) since a toy proof-of-concept doesn't need them — only the
motion-injection risk mattered, and it was verified via a null test
(zero-motion phantom through the identical pipeline: recovered-radius std
= 0.0000mm at zero noise, far below the 1.9mm true motion amplitude — no
spurious motion-correlated artifact). At zero noise, recovery clearly
beats a naive baseline (RMSE 0.286mm vs 0.740mm), satisfying Gate 3's core
criterion. Recovery is fragile to noise (cliff-edge degradation, not
gradual) with the current simple threshold detector — characterized
honestly, not treated as a blocker; a matched-filter detector would likely
be more robust, flagged as a future refinement. Two real bugs were found
and fixed during validation (truncated-toneburst ringing;
peak-vs-first-crossing echo detection) — see LOG.md for details.

**Phase 4.1 done (run 2026-07-07-09): benchmark-then-multiply on real
anatomy.** First use of real Phase I ACDC registration data
(`patient001.npz`) in this project, resampled to the acoustic grid via
nearest-neighbor (CLAUDE.md rule). Timed at N=(150,250,350): 0.81s/2.74s/
7.85s; extrapolated (log-log fit, exponent ~2.66) to the real full-heart
field of view (N~900): ~91s and ~13.8GB per transmit — confirms real-
anatomy, real-resolution simulation needs GPU/cluster (protocol Appendix
A), not this local CPU machine. A compute-budget formula (not filled in
with invented numbers) is ready to bring to the collaborators, alongside
Gate 1's still-outstanding GPU-timed reference reproduction. **Phase 4.2
(actual dataset generation) remains blocked** on collaborator signoff for
attenuation/scaling/staircasing, per the hard gate above — nothing past
this benchmark should be attempted solo.

**Proxy acoustic-physics audit done (run 2026-07-07-11, see
`PROXY_AUDIT.md`).** Per explicit user request ("proxy audit as an
expert... for this solo dev run") — **this is NOT Gate 2**, which
CLAUDE.md and the protocol both state cannot be passed by Claude or the
user alone. All 6 diagnostic items now have real numbers or fixes: PPW
checked (adequate), PML checked via full domain-crossing run (residual
-75dB, adequate), source scaling calibration proposed (FDA MI anchor,
0.316 MPa peak), **attenuation implemented and validated to ~0.1% against
the analytic law** (was the load-bearing gap; caught and fixed a 2x
under-damping bug along the way), motion injection already resolved
(Phase 3), staircasing quantified on real anatomy (0.59% relative field
difference, small but nonzero). `phase4_demo_attenuating_real_anatomy.py`
demonstrates all of this working together on real anatomy (N=350,
patient001, genuine tissue boundary). **Phase 4.2's real, 150-patient
dataset generation remains blocked on actual collaborator signoff** —
this audit substantially de-risks that eventual review but does not and
cannot substitute for it.

**GATE 2 PASSED 2026-07-07 (run -12)**: a Yale acoustic-physics
collaborator reviewed `PROXY_AUDIT.md` and gave unconditional approval of
all 6 items. This is the real Gate 2, not the proxy-audit stand-in.

**Phase 4.2 PILOT dataset generated (run -13, `phase4_generate_pilot_dataset.py`,
`results/phase4_pilot_dataset/`)**: 29 real cases (3 patients, real ACDC
frames, N=350, validated attenuation + calibrated amplitude), all
succeeded (0 NaN), resumability confirmed. **Still NOT the full
150-patient/full-resolution Phase 4.2 run** — that remains gated on Gate
4's compute-budget-agreement checklist item, separate from Gate 2.

**`PIPELINE_STATUS_AND_ROADMAP.md` written (run -13)**: a full
self-diagnosis of the project (Phase I proxy pipeline + Phase II jWave
layer) — inventories what exists, categorizes what's missing (core gaps:
RF matched-filtering, calibrated noise, speckle modeling, beamforming,
multi-angle/vector estimation; parallel evidence channels for an eventual
fusion model: Doppler, RF cross-correlation, speckle tracking, boundary
timing, transmission ToF, biomechanical priors; optional: bigger
models/more patients, deferred). Recommends RF matched-filter/cross-
correlation tracking as the next concrete step, using the pilot dataset's
saved receiver traces, to address the noise-fragility characterized in
Phase 3 (run -07).

**Detector upgrade attempted (run -14, `phase3_matched_filter_recovery.py`)**:
RF matched-filter detection beats envelope-threshold at every noise level
(~30-40% RMSE reduction) but does NOT fully fix the noise fragility —
neither detector beats the naive baseline once any noise is present.
Diagnosis: two comparably-weak reflecting interfaces make single-echo
detection fundamentally ambiguous under noise.

**Detector upgrade continued (run -15, `phase3_reference_tracking_recovery.py`)**:
Level 2 (reference-anchored narrow-window tracking) beats Level 1 at
every noise level and is the first method to beat the naive baseline at
any nonzero noise level (0.604mm vs 0.740mm at noise=0.02) — still not a
complete fix (baseline still wins at noise=0.05/0.10). Three consecutive
levels (0->1->2) have each been a genuine, partial improvement;
`PIPELINE_STATUS_AND_ROADMAP.md` recommends a deliberate decision point
before the larger lifts of Level 3 (beamforming) or Level 4 (multi-angle
fusion).

**Distributed speckle tracking explored (runs -17/-18): open question,
not resolved.** `phase3_speckle_tracking.py` (single-channel),
`phase3_speckle_beamformed.py` (multi-channel), `phase3_speckle_sequential.py`
(sequential, boundary-excluded) — three iterations, each ruling out a
hypothesis (noise-scaling fairness, boundary contamination) without
producing a result that beats boundary-echo tracking. Leading remaining
hypothesis: 400 scatterers is too sparse for fully-developed speckle
statistics — untested.

**Genuine multi-angle vector triangulation (runs -19/-20): strongest
result of the session.** `phase3_vector_triangulation.py` — two
independently-focused 8-element sub-apertures (66.1 degree look-direction
separation) triangulate a full 2D displacement vector, not just a scalar
range. Under realistic noise (20-34dB SNR, run -20): cross-range RMSE
0.67-0.69mm, comparable to or better than this project's best single-
direction result (Level 3, ~0.86mm) — while also recovering information
(cross-range motion) no single-direction method can see at all. Surprising
finding: the on-axis (near-normal) sub-aperture is MORE noise-fragile than
the oblique one (opposite of expectation) — plausibly the same wrong-echo
ambiguity that limited Levels 0-4, unconfirmed.

**RETRACTED (run -21).** Continued debugging found: (1) a timing-formula
bug that partly explains the "surprising" finding above, and (2) a
deeper, structural, PHYSICAL limitation — for a curved reflector, the
sensitivity vector of any valid specular (Tx,Rx) pair is always exactly
parallel to the target's local surface normal, confirmed analytically and
across 7 scanned transmit positions. Multi-angle triangulation of ONE
point via specular reflection cannot recover a true 2D vector — no
implementation fix changes this. The apparent cross-range recovery above
was a geometry-mismatch artifact, not real physics. Genuine vector/strain
recovery needs distributed speckle tracking (runs -17/-18, still
unresolved) or multi-point differential tracking — not single-point
multi-angle triangulation. See `PIPELINE_STATUS_AND_ROADMAP.md` run -21
update for full details.

## Relationship to `pilot/`

This phase reuses Phase I's registration-derived anatomy and motion fields
(`pilot/data/processed/ACDC_reg/*.npz`, ground-truth-quality weights in
`pilot/results/phase3_quality_weights.csv`) as the moving-tissue input to the
acoustic forward model (protocol Phase 4.1). Phase I's ground-truth-quality
caveats (see `pilot/LIMITATIONS.md`) carry forward and must not be dropped
when interpreting Phase II results (protocol Gate 4).

## Session handoff (2026-07-08)

**Read `SESSION_HANDOFF_2026-07-08.md` before resuming any Phase 3
recovery-method work.** It consolidates four exploration threads: single-
echo detection (closed, structural ambiguity), speckle tracking (open,
scatterer density untested), vector triangulation (closed, RETRACTED —
real physical limitation: specular reflection sensitivity is always
normal-only, confirmed analytically), and DAS beamforming
(`phase3_das_beamforming.py`, open — 4 real bugs fixed, ED reconstructs
correctly at 0.61mm error, one genuine open theory question remains for
ES, confirmed via a clean homogeneous-medium control test to be an array/
beamforming-geometry artifact, not a tissue signal). DAS beamforming is
the most promising, closest-to-working thread to resume first.

**5th thread added (runs -23/-24): 4-probe boundary tracking**
(`phase3_four_probe_tracking.py`) — top/bottom/left/right probes around
a bounding square, reusing the validated Level 2 detector. Symmetric
phantom: worked cleanly first attempt (all 4 probes identical
RMSE=0.5625mm). Asymmetric (regional hypokinesis) phantom
(`phase3_asymmetric_phantom.py`, run -24): unfocused probes did NOT
cleanly localize the affected region (left showed near-normal motion
instead of the predicted reduction) — real, standing finding: motion
magnitude recovers correctly, regional localization does not, without
focusing. Attempted fix (focused 8-element probes,
`phase3_four_probe_focused.py`) hit the EXACT SAME artifact class as
Thread 4/DAS (all 4 probes stuck at zero contraction) — **confirmed this
is one unified, unsolved multi-element-focusing artifact problem
spanning both DAS beamforming and focused point-tracking, not two
separate bugs.** See `SESSION_HANDOFF_2026-07-08.md` (updated) for the
consolidated priority: solving the focusing-artifact problem once
unblocks both threads simultaneously.

**Beating-circle DAS movie (run -25, `phase3_beating_circle_movie.py`)**:
simplified to a single filled disk, reconstructed a 12-frame filmstrip
across the cardiac cycle with artifact cancellation (subtracting a
homogeneous-medium reference, principled given the artifact's proven
medium-independence). Partial success: ED-adjacent frames (larger
circle) correctly show the true boundary (~0.5mm accuracy); contracted
mid-cycle frames (smaller circle, weaker echo) fall below the residual
(imperfectly-cancelled) artifact and lock onto it instead — a real,
characterized signal-strength threshold effect, consistent with (not
separate from) the unified focusing-artifact problem above.

**Improved (run -26): bidirectional sequential tracking recovers 10/12
frames.** Narrow-window sequential search (forward from ED, backward
from ED, meeting at ES) avoids the fixed artifact by construction (its
row is far outside any plausible per-step motion range) — 10 of 12
frames now track the true boundary correctly (0.08-1.52mm error,
visually confirmed). Only the 2 deepest-contraction (ES) frames remain
wrong, an honest residual where the true signal is physically weakest.
**This answers the user's question ("can the full cycle be reconstructed
from wall reflections alone?") with: mostly yes (~83% of frames,
sub-1.5mm), with a well-characterized, physically-sensible gap at peak
contraction** — a genuine capstone result for the simplified single-
circle sanity test.

**Doppler/MTI fusion tested (run -27): a wash, but confirms the ES gap
is a real physical floor.** Frame-differencing (`frame[i]-frame[i-1]`,
the discrete Doppler/MTI mechanism) helps at some frames, hurts at
others when fused with the amplitude method by confidence — net RMSE
roughly unchanged. Crucially, differencing's confidence is exactly zero
at the ES frames (wall velocity crosses zero there), so it cannot rescue
them — two independent methods (amplitude-range, velocity-differencing)
now agree ES is a genuine measurement floor for this setup, not an
artifact of either specific algorithm.

**BREAKTHROUGH (run -28): multistatic backprojection ("LIDAR-style"
convergence) — RMSE=0.24mm, zero failure frames, no fixed artifact.**
`src/phase3_multistatic_backprojection.py`, per the user's explicit
re-diagnosis: every earlier method's core flaw was picking ONE echo per
probe from many physically-valid candidates (every point on the
constant-range locus for that arrival time); the fix is to sweep every
candidate point in the domain, compute each of the 4 probes' 16 tx/rx
pairs' predicted travel time to that point, sample each pair's envelope
trace there, and sum — the true surface is wherever independent pairs
agree (mathematically the ultrasonic-NDT "Total Focusing Method" /
LIDAR multilateration). After diagnosing and fixing one bug (Hilbert
envelope group-delay offset, `duration/2`, missing from the naive
travel-time formula — caused a systematic 1.1-1.7mm outward radius
bias, confirmed and corrected, not curve-fit): **RMSE=0.24mm across all
8 cardiac-cycle frames, with NO frame failing catastrophically** (worst
frames, at ES-equivalent contraction, are only 0.35mm — smaller than
every earlier method's BEST frames). **Homogeneous-medium control
confirms the key hypothesis: accumulator peak value 0.0000 — no fixed
coherent-summation artifact survives**, unlike DAS's persistent row-66
ghost (runs -22/-24) that blocked Threads 4/5. This is the first method
this session with no diagnosed residual artifact of any kind, and the
strongest standing result on the beating-circle phantom. Next candidate
tests (not yet run): the full myocardial ring (two boundaries), the
asymmetric/regional-hypokinesis phantom (where Thread 5's focused/
unfocused probes both struggled), and realistic noise levels
(2%/5%/10% SNR).

**Triangle generalization test (run -29, `phase3_multistatic_backprojection_triangle.py`):
NOT a clean pass — real partial success, one new diagnosed failure
mode.** Same phantom shape change only (equilateral triangle, one
vertex at "top"), same accumulator machinery as run -28. bottom
(normal-incidence edge) RMSE=0.14mm and left/right (oblique facets)
RMSE=0.42/0.95mm all generalize the circle result cleanly — real
evidence the method isn't circle-specific. **top (bare vertex/corner)
RMSE=2.46mm, wrong in every frame** — diagnosed root cause: the top and
bottom probes are exactly colinear through the domain center, so that
one tx/rx pair's constant-delay locus degenerates to a line (not an
ellipse), contaminating peak search along the very axis being
evaluated; this was always present but invisible in run -28 because the
circle's strong boundary echo dominated it — the triangle's much weaker
corner scattering does not. Fixable (exclude near-colinear pairs, or
avoid antipodal probe placement) but NOT yet fixed — open item.

**8-probe fix attempted and FAILED (run -30,
`phase3_multistatic_backprojection_triangle_8probe.py`).** Splitting
each wall's probe into 2 (at the 1/3/2/3 points, 8 probes/64 pairs, no
probe left exactly on the vertex axis) did NOT fix the vertex tracking
(RMSE 2.28mm, essentially unchanged from run -29's 2.46mm, same wrong
value each frame) and made the left-facet case WORSE (0.95mm ->
1.87mm, likely because the single centered probe was already at the
ideal near-normal look angle for that facet and splitting it moved both
copies away from that optimum). **Disproves run -29's colinear-pair-
degeneracy diagnosis as the sole cause** — the real mechanism behind the
vertex failure is still unidentified. Next session should isolate which
specific tx/rx pair(s) produce the false "top" peak (e.g. zero out pairs
one at a time) rather than guessing at another geometric fix.

**TRIANGLE THREAD CLOSED (run -31).** `phase3_backprojection_pair_diagnostic.py`
isolated the exact cause cheaply (one ED-frame capture, ~43s, pure
numpy per-pair inspection): **2 of 16 pairs (`bottom->right`,
`right->bottom`) contribute 82% of the false peak's energy** — a
specific sparse-multistatic-array ghost, not general orientation-blind
clutter as an alternative user hypothesis suggested (that would show
energy spread evenly across pairs, not concentrated 200-500x in two of
them). Tested coherence-factor (CF) weighting as a general fix
(`phase3_multistatic_backprojection_triangle_coherence.py`) — **CF
FAILED**: only marginal/inconsistent help at the vertex, but broke the
previously-good bottom/left/right cases (all RMSE roughly 4-13x worse),
because with only 4 probes a genuine specular echo is *also*
few-pair-dominated — CF can't tell real sparse signal from ghost sparse
signal at this array density. **Targeted fix WORKS**: excluding only
the 2 diagnostically-identified pairs from the sum gives top
RMSE=0.68mm (down from 2.46mm), with bottom/right byte-for-byte
unchanged (confirms those pairs were inert there) and left moderately
worse (0.95mm->2.11mm, an honest acknowledged cost, not free). Visuals:
`results/figures/phase3_backprojection_coherence_ghost_comparison.png`
(3-panel naive/CF/excl comparison) and
`results/figures/phase3_multistatic_backprojection_triangle_coherence.png`
(full 8-frame filmstrip, excl variant). **Carry forward**: the
pair-ablation diagnostic technique is fast/cheap/reusable if new ghosts
appear on the heart-model phantom; CF is not a good general fix for
this project's sparse arrays; targeted exclusion has a real,
non-zero cost elsewhere (don't assume it's free without checking).

**CORRECTION (user pushback, same session): run -31 was a SOFT,
target-specific patch, not a root-cause fix — "closed" overstated it.**
The excluded pairs (`bottom->right`/`right->bottom`) were chosen to fix
ONE target (vertex) and applied globally, which is why `left` regressed
(those same pairs carried real localization info for that oblique
facet). Deeper issue: `track_four_directions` picks one peak per fixed
cardinal axis independently, which assumes every direction hits a clean
point target — true for a circle, not guaranteed for a flat facet, and
meaningless for the heart-wall phantom (no natural 4 cardinal rays).
**Before applying this to the heart model**: (1) any future pair-
exclusion should be sector/target-specific, diagnosed per boundary
feature, not global; (2) replace fixed-axis peak-picking with GLOBAL
SHAPE FITTING to the full accumulator (Hough-transform or RANSAC fit of
the known polygon/boundary model), so a bad localized point becomes a
visible outlier to a robust fit rather than a silently-trusted number.
See `jwave_test/LOG.md`, "Correction" entry immediately after run -31.

**GLOBAL SHAPE-FIT READOUT IMPLEMENTED AND WORKS (run -32,
`phase3_backprojection_shape_fit_triangle.py`).** Generalized-Hough-
style 1-parameter template match: sweep candidate R, score each by
summing the (plain, NAIVE, no pair-exclusion) accumulator's energy
along ALL 72 angles of that R's predicted boundary, pick the best R.
Real, balanced improvement over BOTH prior approaches — no catastrophic
outlier on any side: top=0.81mm, bottom=0.41mm, left=0.47mm (vs run
-31's regressed 2.11mm), right=0.47mm. Worst-case error across all 4
sides: 0.85mm, vs 2.46mm (naive) / 2.11mm (excl) — achieved
automatically, no hand-tuned pair exclusion needed at all. **Honest
residual**: fitted R consistently undershoots true R by ~7-8.5 cells
(~0.8mm) in every frame — small, uniform, likely diagnosable (not yet
root-caused). Visuals:
`results/figures/phase3_backprojection_shape_fit_score_curve.png`
(score-vs-R curve, ED frame) and
`results/figures/phase3_backprojection_shape_fit_triangle.png`
(8-frame filmstrip, true vs fitted triangle overlaid). **This is now
the recommended readout approach to generalize (in principle, not
literal triangle-specific code) to the heart-wall phantom** — sweep a
parametric boundary family and integrate evidence over the whole
predicted boundary, rather than picking independent local peaks along
fixed axes.

**Undershoot bias diagnosed; proposed TGC fix TESTED AND REJECTED (run
-33, `phase3_shape_fit_bias_diagnostic.py` +
`phase3_backprojection_shape_fit_triangle.py`'s `backproject_tgc`).**
Diagnostic found: (1) run -29's "good" left-facet result was partly
LUCKY — the true-facet peak and a competing ghost peak are within 1% of
each other in height (0.000503 vs 0.000507), so which one a naive
argmax finds depends on sub-percent numerical detail, not robust
dominance; (2) restricting the global fit to only the 4 (validated)
cardinal angles gives a smaller undershoot (0.50mm) than using all 72
(0.80mm) — pointing to a systematic, direction-independent bias, not
just noise. Hypothesized cause: uncorrected range-dependent amplitude
falloff (2D geometric spreading makes closer candidate points read
"louder" regardless of true reflectivity) — implemented a TGC
(depth-gain-compensation) fix, `backproject_tgc`, multiplying by
sqrt(dist_tx*dist_rx). **Tested: NO EFFECT** (R RMSE 0.8136mm ->
0.8202mm, statistically unchanged) — this FALSIFIES the amplitude-gain
hypothesis. Revised hypothesis (untested): a TIMING bias, not amplitude
— the `_ENVELOPE_GROUP_DELAY_S=duration/2` correction (calibrated for
one normal-incidence case in run -28) may not be uniformly correct
across the mostly-oblique, off-axis bistatic angles making up 68 of the
72 integrated directions. **Do not re-attempt amplitude-based fixes for
this bias** — next step, if resumed, is testing the timing-bias
hypothesis directly (fit the group-delay correction empirically rather
than assuming one constant value). The ~0.8mm residual remains usable
as-is (far smaller than every prior method's worst case) if the
project needs to move on without resolving this further.

**REVISED: the "timing bias" framing was WRONG (user correctly pushed
back); it's a MULTI-GHOST geometric effect, confirmed at a 2nd sector,
fix WORKS at large R only (run -34).** Per user's observation ("the
predicted triangle is always smaller regardless of systole or
diastole"), re-examined run -33's data: `top` and `bottom`'s
independent free-search peaks were IDENTICAL (33.50 cells), and
`left`'s independent peak (59.25) landed almost exactly at `top`'s TRUE
distance — directions finding each other's reflections, not a uniform
offset. New diagnostic (`phase3_left_ghost_diagnostic.py`) confirmed
LEFT has its OWN ghost pairs (`left->top`, `bottom->left`,
`left->bottom` = 65.9% of its false-peak energy, ~0% at its true
location — same disjoint-pair-set signature as TOP's confirmed ghost).
**Generalized fix**: `backproject_no_adjacent` excludes ALL 8 adjacent
(90-degree-separated) probe-pair combinations, keeping only 4
monostatic + 4 antipodal pairs. **Result: works essentially perfectly
at large R (ED-adjacent frames: 0.00mm, 0.03mm errors, all 4 sides) but
only partially at small R (unchanged 0.85mm at R=47.8, improved-not-
fixed 0.28mm at R=41.0)** — overall R RMSE 0.8136mm -> 0.4482mm (~45%
reduction), NOT a complete fix. Confirms the ghost-pair mechanism is
real and dominant at strong signal, but a SECOND, still-unidentified
effect persists at weak signal (smaller/more-contracted radii) —
consistent with this session's recurring "weak-signal residual" theme.
Caution: homogeneous control now fits to R=25.2, right at the R_GRID
search boundary — needs a wider grid to confirm it isn't a boundary
artifact. **Do not present this as a solved problem** — carry the
large-R success and small-R open residual both forward to the heart
model. See `jwave_test/LOG.md` run -34.

**Off-center triangle test (run -35, `phase3_offcenter_triangle_test.py`):
ghost pattern persists at a shifted location — confirms it's an
array-structural property, not a symmetric-case coincidence.** Same
phantom shifted (15,10) cells off domain center: `top` and `left`
still wrong (2.58mm, 2.16mm errors), `bottom` and `right` still
accurate (0.40mm, 0.16mm) — same qualitative pattern as the centered
case, at genuinely different true distances (not a trivial restatement).
This means the adjacent-vs-antipodal ghost mechanism (runs -29/-33/-34)
is a general property of the 4-probe layout, not tied to the one
maximally-symmetric centered position — the `backproject_no_adjacent`
mitigation should be expected to generalize to an off-center heart
phantom. Qualitative/single-frame probe, not a full sweep — full
per-axis breakdown of the no-adjacent variant at this offset, and a
larger-offset test, are natural next steps if revisited.

**NOTE FOR THE FINETUNING PHASE (see `jwave_test/LOG.md`'s "Note for the
finetuning phase" entry, right after run -35):** the ~0.8mm shape-fit
bias's clinical significance is UNKNOWN, not negligible-by-default —
this toy phantom (4-6mm LV radius) is far from clinical scale (~20-30mm
real LV radius), and whether the bias is a fixed absolute artifact
(-> likely negligible at real scale) or scales proportionally with
target size (-> could be ~3mm at real scale, clinically relevant) has
NOT been tested. Also: the bias is demonstrably NOT a single constant
(R-dependent, position-dependent per runs -34/-35) — **any calibration
built during finetuning must be a function of apparent size/direction/
position, not a single scalar offset subtraction.** Test the scaling
question (real-scale phantom sweep) before scoping a calibration
approach.

**GHOST MECHANISM CONFIRMED (run -36, `phase3_ghost_mechanism_diagnostic.py`):
it's the vertex's real corner-diffracted energy, not a capture/
simulation bug — the artifact is 100% in the reconstruction model.**
Tested the `left->top` ghost pair's RAW captured trace directly
(bypassing backprojection): both candidate mechanisms (genuine bistatic
specular reflection from an unexpected boundary point; genuine corner
diffraction from the known vertex) predict essentially the SAME arrival
time, both matching the actual recorded peak to <0.5mm. Mechanism 1's
independent boundary sweep (no vertex location given to it) converged
to a point essentially ON the actual vertex — confirming the two
mechanisms are the same physical location (ordinary specular reflection
and diffraction become the same limiting case at a sharp corner). **The
captured data is physically correct** — the simulation is doing real
wave physics; the "ghost" exists purely because the reconstruction's
naive travel-time-only matching has no specular-consistency check and
no diffraction term, so it can't distinguish real vertex-diffracted
energy from a wrong candidate point that happens to share the same
round-trip path length for that specific pair. **Reframes the ghost
constructively**: since it's real, meaningful signal (marking a genuine
corner/singularity), a future reconstruction could use it for feature
detection rather than only suppressing it via pair exclusion — relevant
for the heart-wall phantom, where real anatomical corner-like features
(papillary muscles, valve annulus, wall-thickness transitions) may
produce the same effect. Not yet verified for the other 3 confirmed
ghost pairs (`bottom->right`, `right->bottom`, `bottom->left`,
`left->bottom`) — a natural follow-on.

**CONCAVE HEART-SHAPE TEST (run -37, `phase3_heart_shape_offcenter_test.py`):
neither naive nor no-adjacent-pairs recovers the boundary — a real
regression from the triangle, confirming the fix doesn't generalize.**
First non-convex shape tried (10-vertex heart polygon: bottom tip, 2
lobes, concave notch between them), off-center by (10,-15) cells, same
8-frame ED/ES schedule and validated pipeline. **Visually inspected the
saved figures directly** (not just confirmed they saved): both variants
show the same dominant, roughly FIXED diagonal ridge pattern across all
8 frames regardless of true size/phase — not tracking the actual heart
boundary at all, unlike the triangle's near-perfect no-adjacent-pairs
result at large R. Consistent with run -36's confirmed corner-diffraction
mechanism: 4 sharp features (tip, 2 lobes, notch) vs. the triangle's 3
plausibly creates more ghost-pair combinations than the simple
"adjacent vs antipodal" categorization (tuned only on the triangle) can
clean up. **The no-adjacent-pairs fix is a triangle-shaped patch, not a
general solution.** Could not determine whether the concave notch
specifically diffracts (general clutter too strong to isolate it
visually) — would need the same targeted pair-ablation diagnostic aimed
at the notch. **Before the real heart-wall phantom, this pipeline likely
needs the global shape-fitting/boundary-integration principle (run -32)
generalized to non-convex boundaries, not more pair-exclusion patches.**

**Global shape-fit CONFIRMED to generalize past convex shapes (run -38,
`phase3_heart_shape_shapefit.py`).** Same 1-parameter global template-
match principle, generalized to the concave heart's 10-vertex polygon
via proper multi-edge ray intersection. Naive accumulator + global fit:
R RMSE=0.23mm, consistent across all 8 frames — no outliers. The
no-adjacent-pairs heuristic (tuned only on the triangle): R RMSE=2.85mm,
erratic, actively worse — confirms it's a triangle-specific patch, not
a general fix. Visually confirmed both results directly. Small
systematic OVERSHOOT found in the naive result (opposite sign from the
triangle's run -32 UNDERSHOOT) — evidence against a universal timing
bug, for a shape-dependent geometric effect (consistent with run -36).

**MYOCARDIAL RING PHANTOM (run -39, `phase3_ring_phantom_shapefit.py`):
inner boundary excellent, OUTER boundary NOT recoverable — reconnects
to Thread 1's original weak-interface problem from early in this
session.** First test of this project's actual two-boundary "heart
phantom" (`phase3_config.py`'s LV_RADIUS_ED/ES_CELLS +
WALL_THICKNESS_CELLS ring model). Inner (LV cavity) boundary:
RMSE=0.27mm, matches every other shape tried this thread. Outer
(epicardial) boundary: RMSE=2.44mm, NOT just noisy — several frames
show IDENTICAL fitted values (55.0 cells) despite different true radii
(the "stuck at a fixed value" artifact signature), others land almost
exactly on that frame's own inner-boundary fit.

**CORRECTED (run -40, `phase3_outer_boundary_diagnostic.py`): the outer
interface is NOT weak — it's a two-boundary separation problem, not a
weak-tissue-contrast problem.** Run -39's attribution to "Thread 1's
weak blood/myocardium problem" was imprecise (that framing predicts the
INNER boundary should also fail, but it didn't). Decisive test: a
myocardium disk in chest-wall-proxy background, NO inner boundary at
all, at R=90 (matching the ring's true ED outer radius) — **recovers
PERFECTLY (0.00mm error)**, visually confirmed as a clean, unambiguous
ring, as good as or better than every other single-boundary shape this
thread. **This proves the myocardium/chest-wall-proxy interface has
strong, real, independently-recoverable signal** — the ring's outer-
boundary failure is caused specifically by the INNER boundary's
presence (masking/domination/search ambiguity), not by the outer
interface being undetectable. **The outer/epicardial boundary is a
genuinely open, unsolved TWO-BOUNDARY SEPARATION problem — do not
report overall pipeline progress as complete, and do not re-attribute
this to weak tissue contrast.**

**STANDING RULE (established run -41, per explicit user objection —
apply to all future work, not just the ring phantom): never encode an
anatomical constant-value assumption (e.g. constant wall thickness)
into a fitting/reconstruction method in a way that would suppress
genuine detection of the corresponding pathology.** The natural-seeming
"couple outer_R = inner_R + WALL_THICKNESS_CELLS" fix was explicitly
REJECTED before implementation: real myocardial wall thickness varies
regionally and pathologically (hypertrophy, post-infarct thinning,
aneurysmal bulging), and detecting exactly that variation is a primary
purpose of cardiac imaging. Hard-coding a constant into the
reconstruction would SILENTLY blind it to the pathology it exists to
find — a confident, clean-looking, wrong answer, which is worse than an
obvious failure.

**Alternative implemented instead (run -41,
`phase3_ring_outer_guardband_fit.py`): a guard-band exclusion — real but
only PARTIAL improvement, does not solve the problem.** Uses the
already-reliable inner fit only to exclude a narrow region (+/-8 cells)
immediately around it from the outer search, without assuming anything
about the true outer radius (a real heart with abnormal/asymmetric wall
thickness would still be found on its own signal). Result: outer RMSE
2.44mm -> 1.92mm — removed the worst failure mode (outer fit landing
exactly on the inner fit) but still far from this thread's usual
0.0-0.3mm accuracy. **A second, distinct artifact found**: two frames
give an identical wrong value (55.0 cells) that sits OUTSIDE the guard
band's excluded range, meaning it's not simple inner-boundary-adjacency
leakage — most likely genuine reverberation/multi-bounce energy between
the two boundaries (only 30 cells/3mm apart). **Outer/epicardial
boundary recovery remains open and unsolved** — next step is
diagnosing the ~55-cell artifact specifically (reverberation control or
pair-ablation diagnostic) before attempting another fix.

**MECHANISM DIAGNOSED (run -42, `phase3_ring_outer_ghost_diagnostic.py`):
NOT reverberation-dominant — it's the inner boundary's real reflection
aliased by antipodal/cross-pair geometry.** Per-pair energy analysis:
the R=55 false peak is dominated by ANTIPODAL pairs (`bottom<->top`,
`left<->right`, 3-6x more energy there than other pair categories),
while MONOSTATIC pairs correctly favor the true outer boundary
(consistent with run -40). Direct hypothesis matching against actual
recorded peaks: genuine 1st-order internal reverberation IS confirmed
present (a real peak matches its predicted time almost exactly for a
non-antipodal pair) — but it's a secondary, weaker effect. The DOMINANT
mechanism: both the dominant non-antipodal and antipodal pairs' actual
strongest peaks match the INNER boundary's predicted time, not the
outer boundary's — the inner boundary's strong, genuine reflection gets
aliased to the wrong intermediate apparent radius (R=55) by the naive
single-bounce model summed across many angles. **Same class of
mechanism as the triangle's confirmed corner ghosts (run -36)** — real
energy from one location misattributed to a different candidate
location — manifesting via a different pair category (antipodal, not
adjacent) because of the ring's rotational symmetry. Next principled
test (not yet run): exclude/down-weight antipodal + cross pairs, keep
only monostatic pairs (which already correctly favor the true outer
boundary) — a pure instrumentation/geometry decision, NOT an anatomical
assumption (distinct from the rejected wall-thickness coupling). Must
check this doesn't regress the inner boundary's own excellent fit
first (same caution as the triangle's run -34 left-facet regression).

**PAIR-CLASS ABLATION DONE (run -43, `phase3_ring_pair_ablation.py`):
monostatic-only DEMONSTRATES real reconstruction collapse — REJECTED as
a fix; no fixed pair-subset is both safe and complete.** Tested 5
configs (all-16, monostatic-only, remove-antipodal-only,
remove-run42-implicated [=monostatic-only numerically], monostatic+2-
least-biased-cross) at 2 frames (ED strong-signal, ES-adjacent weak-
signal). **Monostatic-only looks perfect at ED (outer error 0.00mm) but
CATASTROPHICALLY COLLAPSES the inner fit at the harder ES-adjacent
frame** (3.10mm error — locked onto the OUTER boundary's own location
instead of its own true value) — discarding 12 of 16 pairs removes
exactly the redundancy that kept inner recovery robust at weaker
signal. **This directly confirms the user's reconstruction-collapse
concern and rules out monostatic-only (and its numerical twin, config
D) as a fix.** Removing only antipodal pairs, or monostatic+2-least-
biased-cross, avoid collapse at both frames (inner stays ~0.1-0.15mm)
but neither fixes the outer boundary at ED (still ~2.9mm, still locked
to inner). **Conclusion: no static, condition-independent pair-subset
rule solves this — the set of "trustworthy" pairs depends on signal
strength/frame, which in real anatomy would also vary with phase/
pathology.** Recommends an adaptive, per-frame, signal-content-based
down-weighting (reject a pair's outer-boundary vote specifically when
its own real recorded energy better matches its own inner-boundary
prediction) instead of any fixed pair-class exclusion. Not yet
implemented. Also flagged: only tested on this idealized circular,
centered ring — a non-circular/asymmetric boundary test remains a
standing follow-on per the runs -37/-38 generalization concern.

**ROOT CAUSE FOUND (run -44, `phase3_ring_curvature_diagnostic.py` +
`phase3_ring_amplitude_divergence_test.py`): curvature-dependent
reflection divergence, confirmed with real simulated amplitudes.**
Step 1 (pure geometry): "does a specular point exist" hypothesis
REFUTED — a valid specular point always exists on a full circle for
any pair/radius (unlike the triangle's finite edges); defect~0 for
every pair type at both R=41 and R=71. Step 2 (isolated single-boundary
phantoms, real amplitudes): confirmed a DIFFERENT, correct mechanism —
at monostatic incidence, outer (R=71) is the STRONGER reflector (ratio
1.96, consistent with it being the first interface + a slightly larger
cited-tissue reflection coefficient, 0.0035 vs inner's 0.0025 — ruling
out "outer is just weak"). But at ANY wide-baseline pair, outer's
energy COLLAPSES to ~0 (cross ratio 0.001, antipodal ratio 0.012) while
inner's decays far more gently and stays detectable even at 180
degrees. **Mechanism**: the outer (larger/flatter) circle reflects like
a near-flat mirror, concentrating energy in a narrow monostatic cone;
the inner (smaller/more curved) circle reflects like a diverging convex
mirror, spreading weaker energy across a wide angular range. With all
16 pairs summed, outer's real signal exists ONLY in the 4 monostatic
pairs — the other 12 carry ~zero real outer signal and vote for
whatever they DO see instead (inner boundary), outnumbering the correct
votes 3-to-1. **This explains why every fixed pair-subset rule (runs
-41/-43) was either unsafe or incomplete** — none of them modeled the
actual physics (pair reliability is a continuous function of boundary
curvature vs. baseline angle, not a fixed category). **Next step (not
yet implemented)**: a physically-motivated weighting — score each
pair's vote using an explicit curvature-dependent divergence model
rather than any fixed inclusion/exclusion list. This generalizes
properly to real anatomy's continuously-varying curvature (trabeculae,
papillary muscles, smooth wall segments) instead of needing a new
hand-tuned rule per shape.

**FIRST COMPLETE SUCCESS (run -45, `phase3_ring_curvature_weighted_fit.py`):
curvature-aware weighting + guard band recovers BOTH boundaries at BOTH
tested frames, no collapse.** Implemented run -44's proposed fix: each
pair's contribution at candidate radius R scaled by a curvature-aware
weight (simple linear interpolation between run -44's 2 measured
calibration points, clipped [0,1] — an explicit first approximation).
This alone fixed the ES-adjacent (weak-signal) frame's outer boundary
(0.12mm, not locked) while preserving inner accuracy at both frames
(no collapse) — but still failed at the ED frame (2.90mm, locked)
because OUTER_R_GRID's range (55-110) overlaps the true inner radius
(60) at that frame, letting inner's raw strength push through even a
reduced weight. Adding a guard band (run -41's idea, now informed by
the already-fitted inner radius rather than an anatomical assumption)
around the outer search fixed this: **ED outer=0.00mm, ES-adjacent
outer=0.12mm (unchanged) — both boundaries, both frames, no collapse,
no locking, for the first time in this entire investigation (runs
-39 through -45).** Caveats not yet resolved: (1) the weight model is a
crude 2-point linear fit, not a fully derived physical formula; (2)
only 2 frames tested, not the full 8-frame cycle; (3) only tested on
this idealized circular, centered ring — the standing generalization
concern (does this survive a non-circular/off-center boundary, per
runs -37/-38) is the most important remaining validation step before
treating this as a real result rather than a promising one.

**GENERALIZATION TEST PASSED (run -46,
`phase3_ring_eccentric_offcenter_test.py`): full 8-frame eccentric,
off-center phantom recovers both boundaries accurately at every
frame, no collapse.** This closes the two gaps run -45 flagged. Whole
phantom shifted off the domain/probe center (offset (8,6) cells) AND
the inner (LV cavity) circle's center further offset from the outer
(epicardial) circle's own center by (6,5) cells (fixed, not scaling
with radius) — creating a genuine ~1.7x thick/thin wall-thickness
asymmetry (2.22mm vs 3.78mm), mimicking real regional LV wall
variation. Required a wider local search grid (+/-100 cells) to avoid
clipping the off-center phantom. Result across all 8 frames of the
full ED->ES->ED cycle: **inner RMSE=0.31mm, outer RMSE=0.24mm, "locked
to inner"=False at every single frame** — modestly worse than the
idealized centered/concentric case (0.10-0.12mm) but solidly within
this thread's established "good" range (comparable to the plain
circle's 0.24mm, the heart-cartoon's 0.23mm). **This is the first
method in the entire ring-phantom investigation to survive both the
full cardiac cycle and a non-idealized (eccentric, off-center)
geometry** — every previous fix that looked clean on one idealized case
(triangle's pair-exclusion, monostatic-only) failed when generalization
was actually tested. Standing caveats: each boundary's fit used its own
TRUE center as the ray-sweep origin (tests radius-fitting under
eccentricity, not blind joint center+radius fitting — a separate,
harder, not-yet-attempted problem); the curvature weight model is still
only a 2-point linear fit; the concave heart-cartoon shape (run -37)
has not yet been retested with this newer curvature-weighted approach.

**FIRST REAL-ANATOMY ESCALATION PASSES (runs -47/-48): MRI-derived
irregular ring (ACDC patient001, slice 4) reconstructed with sub-mm
accuracy on both boundaries.** `src/phase3_mri_irregular_ring_prep.py`
extracts a real myocardium+LV segmentation from ACDC patient001
(mid-ventricular slice, max LV area), rescales it isotropically
(shape-preserving) to this thread's toy scale (60 cells / 6mm LV
radius, matching the synthetic tests) via nearest-neighbor resampling,
then applies a light Gaussian smoothing pass (sigma=1.27 cells, tied to
native pixel size) to remove nearest-neighbor staircasing while
preserving genuine anatomical irregularity — output:
`results/mri_irregular_ring_patient001_slice{4}.npz` (raw + smoothed
masks/contours), figure `results/figures/phase3_mri_irregular_ring_prep.png`.
Note: this patient/slice is a relatively mild/typical irregularity
case (fairly round LV, roughly uniform wall thickness), not a
dramatically pathological one — flagged to the user, who chose to
proceed with it as the first test.
`src/phase3_mri_irregular_ring_reconstruction.py` builds the acoustic
medium DIRECTLY from the smoothed real masks (not a synthetic formula)
and reuses run -46's validated curvature-weighted + guard-band fit,
generalized so the "known shape family" is the real measured r(theta)
per boundary (polar-resampled from the extracted contour, each
boundary using its own centroid) with a SCALE FACTOR as the one free
parameter, instead of a closed-form circle/polygon. Result: **inner
(LV) fitted scale=1.015 (true=1.0), error=0.09mm; outer (epicardium)
fitted scale=1.030 (true=1.0), error=0.22mm; not locked to inner**;
fitted contours visually track the real irregular boundary shape (not
a smoothed circle approximation) in
`results/figures/phase3_mri_irregular_ring_reconstruction.png`. One bug
caught and fixed during this run: an initial guard-band implementation
operated in SCALE units instead of physical-radius-cells units (the
quantity run -45/46's guard band was actually designed to protect),
causing a false "outer locks to inner" failure (err=1.44mm) until
converted back to radius-cells units, matching run -45/46 exactly.
Single static frame only (real segmentation is one ED timepoint, not a
motion cycle) — multi-frame real motion and a more dramatically
asymmetric (HCM/DCM) patient/slice remain open, not-yet-attempted
follow-ons.

New files this run: `jwave_test/src/phase3_mri_irregular_ring_prep.py`,
`jwave_test/src/phase3_mri_irregular_ring_reconstruction.py`,
`jwave_test/results/mri_irregular_ring_patient001_slice4.npz`,
`jwave_test/results/figures/phase3_mri_irregular_ring_prep.png`,
`jwave_test/results/figures/phase3_mri_irregular_ring_reconstruction.png`.
`requirements.txt` gained `scikit-image==0.26.0` (contour extraction).

**FIRST REAL-MOTION ESCALATION (runs -49/-50): 8-phase cycle built by
interpolating the Phase I registration-derived ED->ES displacement
field (not raw consecutive slices, not raw consecutive cine frames) —
acoustic reconstruction survives it, sub-mm accuracy, but this
patient/slice's true contraction signal is weaker than that error.**
`src/phase3_mri_motion_cycle_prep.py` applies
`pilot/data/processed/ACDC_reg/patient001.npz`'s ED->ES displacement
field at fractional strength (half-cosine schedule, 8 equally-spaced
phases) via nearest-neighbor `map_coordinates` warping of the ED
myo/LV masks — convention cross-checked against Phase I's own
Dice-validated `warped_ed_mask` (98.7% agreement) before trusting it.
Registration quality for this patient is flagged honestly:
mean_dice=0.784, myo_dice=0.789 (below the pilot's own 0.80 Gate-3
threshold). Real contraction here is subtle: true inner mean-radius
only spans 6.03mm->5.94mm (0.10mm) across the whole cycle, far smaller
than the synthetic toy's deliberate 6mm->4mm. Output:
`results/mri_motion_cycle_patient001_slice4.npz`, figure
`results/figures/phase3_mri_motion_cycle_prep_filmstrip.png`.
`src/phase3_mri_motion_cycle_reconstruction.py` reruns the validated
curvature-weighted + guard-band multistatic backprojection at each of
the 8 phases, fitting a SCALE FACTOR against the FIXED ED r(theta)
template (not re-derived per frame) — testing whether one scale
parameter tracks genuinely non-uniform real wall motion. Result: inner
RMSE=0.2255mm, outer RMSE=0.4298mm (both within the ~1.5mm registration
floor), fitted contours visually track the true non-uniform
deformation in `results/figures/phase3_mri_motion_cycle_reconstruction.png`.
**Honest caveat**: the true signal (0.10mm radius span) is smaller than
the reconstruction's own RMSE — this patient/slice demonstrates the
method survives real non-uniform motion without collapsing, but is too
weak a contraction signal to demonstrate accurate contraction-tracking.
A higher-ejection-fraction-change patient would be a sharper test.
`labels.GT_FLOOR_CAPTION` applied (first use of real, imperfect ground
truth in this thread, rather than exact toy/segmentation values).

New files this run: `jwave_test/src/phase3_mri_motion_cycle_prep.py`,
`jwave_test/src/phase3_mri_motion_cycle_reconstruction.py`,
`jwave_test/results/mri_motion_cycle_patient001_slice4.npz`,
`jwave_test/results/figures/phase3_mri_motion_cycle_prep_filmstrip.png`,
`jwave_test/results/figures/phase3_mri_motion_cycle_reconstruction.png`.

**PATIENT023 SELECTED FOR ADEQUATE CONTRACTION (runs -51/-52/-53): scan
of all 150 ACDC patients found patient001 (used so far) has only ~10%
LV contraction — patient023 has ~45% (ED 20.97mm -> ES 11.53mm),
myo_dice=0.870 (above pilot's 0.80 threshold), same slice index 4.**
`phase3_mri_irregular_ring_prep.py` and `_reconstruction.py` generalized
to take a `PATIENT_ID` CLI arg (default patient001) instead of being
duplicated per patient; output files now include the patient ID
(`results/mri_irregular_ring_{PATIENT_ID}_slice{z}.npz`,
`results/figures/phase3_mri_irregular_ring_{prep,reconstruction}_{PATIENT_ID}.png`).

Static reconstruction on patient023 found a NEW limitation: inner
fit is solid (scale=0.925, err=0.45mm) but outer fit is poor
(scale=0.725, err=2.43mm, noisy multi-peaked score curve) — traced to
patient023's proportionally thicker real wall (outer mean radius=88.2
cells vs patient001's 74.0, same 60-cell toy LV radius). Also required
a dynamically-sized search grid (`build_search_grid()` added to
`phase3_mri_irregular_ring_reconstruction.py`) since the default
+/-90-cell grid clipped this patient's larger anatomy.

Two hypotheses tested directly (not assumed) per user's questions:
1. **Probe standoff** (`src/phase3_mri_wide_probe_standoff_test.py`,
   self-contained, does not modify the shared probe-geometry module):
   doubling patient023's standoff (46->92 cells) gave an IDENTICAL
   result — ruled out.
2. **Curvature-weight calibration range** (`src/phase3_ring_calibration_r88.py`):
   measured cross/monostatic and antipodal/monostatic amplitude ratios
   directly at R=88 (patient023's real outer radius, beyond run -44's
   original R=41/71 calibration) — both ~0, CONFIRMING (not correcting)
   the model's already-clipped-to-zero extrapolated value there.
   `phase3_ring_curvature_weighted_fit.py`'s `_CAL_R/_CAL_CROSS/_CAL_ANTIPODAL`
   extended to 3 measured points (41, 71, 88); `_linear_weight` switched
   from hand-rolled slope-extrapolation to `np.interp` (flagged: a minor
   behavior change for candidate R<41, not believed to affect any prior
   logged result's selected best-R). Re-running patient023 with the
   updated calibration gave an IDENTICAL result to before recalibration.

**Conclusion**: patient023's noisy outer fit is a genuine STRUCTURAL
limitation (a large/flat reflector genuinely returns near-zero energy
to all but the 4 monostatic pairs, confirmed with real measurements at
R=88, not assumed) — not a calibration bug, not a standoff artifact.
Fixing it for real would need more independent probe angles, not a
coefficient tweak. Carried forward as an open, well-understood
limitation rather than force-fixed.

New/changed files: `jwave_test/src/phase3_mri_wide_probe_standoff_test.py`,
`jwave_test/src/phase3_ring_calibration_r88.py`,
`jwave_test/results/mri_irregular_ring_patient023_slice4.npz`,
`jwave_test/results/figures/phase3_mri_irregular_ring_prep_patient023.png`,
`jwave_test/results/figures/phase3_mri_irregular_ring_reconstruction_patient023.png`,
`jwave_test/results/figures/phase3_mri_wide_probe_standoff_test_patient023.png`.

**PATIENT023 MOTION-CYCLE RESULT (run -54): inner boundary demonstrates
real contraction-tracking (not just survival); outer boundary shows
the structural bias consistently, confirming the diagnosis.**
`phase3_mri_motion_cycle_prep.py`/`_reconstruction.py` generalized to
`PATIENT_ID` CLI arg + max-LV-area slice auto-selection. True signal:
inner mean-radius spans 6.02->4.93mm (1.09mm span, 11x patient001's
0.10mm). Inner RMSE=0.8014mm (smaller than the true signal span — real
tracking demonstrated, not just non-collapse). Outer RMSE=2.2940mm,
remarkably stable across all 8 phases (1.97-2.49mm) — phase-independence
itself corroborates the run -51/-53 structural-limitation diagnosis
(a geometry-dependent bias should be constant regardless of contraction
phase, unlike a signal-strength-dependent error). Visually, fitted
outer contour sits consistently inside the true outer contour at every
phase (systematic inward bias, not noise) —
`results/figures/phase3_mri_motion_cycle_reconstruction_patient023.png`.

This closes out the real-MRI escalation arc for this thread (synthetic
eccentric ring -> real static shape -> real motion cycle, both
patient001 and patient023). Open follow-ons: (1) more independent probe
angles to fix the structural outer-boundary limit; (2) a more
dramatically asymmetric (HCM/DCM) patient for boundary SHAPE
irregularity specifically; (3) independently registering ED to real
intermediate cine frames; (4) the multi-compartment/multi-chamber
full-heart escalation (previously deferred).

New files: `jwave_test/results/mri_motion_cycle_patient023_slice4.npz`,
`jwave_test/results/figures/phase3_mri_motion_cycle_prep_filmstrip_patient023.png`,
`jwave_test/results/figures/phase3_mri_motion_cycle_reconstruction_patient023.png`.

**CORRECTION (run -55): patient001's runs -48/-50 numbers superseded —
cause was the search-grid widening (run -51), NOT the calibration
update (run -53).** Confirmed via direct A/B isolation: default grid +
new calibration exactly reproduces run -48's original numbers;
default calibration + new grid reproduces the new ones. Root cause:
patient001's own outer contour has max radius 79.4 cells (needed_extent
119.0 cells at the sweep's largest scale), which already exceeded the
default +/-90-cell grid *before* patient023 was ever introduced —
`RegularGridInterpolator`'s `fill_value=0.0` silently zero-filled the
most eccentric angular points at large candidate scales, undetected
until run -51's robustness fix (added for patient023) incidentally
also triggered for patient001. Also fixed a real (if so-far-inert)
inconsistency: `phase3_mri_motion_cycle_reconstruction.py`'s grid-sizing
check used MEAN contour radius instead of MAX (like the static script),
now corrected to `.max()` in both.
**Corrected numbers (supersede runs -48/-50):** static reconstruction
inner scale=0.995 (err=0.03mm), outer scale=1.035 (err=0.26mm); motion
cycle inner RMSE=0.1996mm (was 0.2255mm), outer RMSE=0.4028mm (was
0.4298mm). Both are small improvements, not regressions — no
conclusion from runs -48/-50 changes qualitatively.

**ISOLATED 8-PROBE TEST (run -56): real, partial fix for the structural
outer-boundary limitation.** `src/phase3_mri_8probe_test.py` — a fully
self-contained script (own probe geometry/domain/capture/weight logic,
no existing file modified) testing 8 probes at 45-degree spacing (vs.
the standard 4 at 90-degree spacing) on patient023. Weight model for
the new 45/135-degree baseline pairs is a documented INTERPOLATION
approximation (not a new measurement) between the existing 0/90/180-
degree calibration anchors. Result: outer error improved 2.43mm->2.16mm
(~11%); more tellingly, the outer score curve now shows a genuine
secondary peak AT the true scale=1.0 (previously invisible in every
4-probe test) that narrowly loses to the guard-band-edge peak rather
than winning outright. Confirms probe count is a real lever for this
limitation, not fully sufficient at 8 probes alone. Open follow-ons:
measure the real 45-degree baseline ratio directly (replacing the
interpolation assumption); try more probes (12-16).

New files: `jwave_test/src/phase3_mri_8probe_test.py`,
`jwave_test/results/figures/phase3_mri_8probe_test_patient023.png`.

**LOCAL-MAXIMUM-ONLY SELECTION (run -57): outer error 2.43mm -> 0.04mm,
essentially a complete fix.** Diagnosed the "tail" (naive argmax landing
on the guard-band cutoff edge) as leakage from the excluded region, not
genuine signal — REJECTED dampening it (unsafe, shapes the algorithm
toward the expected answer) in favor of `select_best_local_peak()`
(added to the same isolated `phase3_mri_8probe_test.py`): require the
winning candidate to be a genuine local maximum (rises then falls),
splitting the guard-band-gapped scale grid into contiguous segments
first so a segment's own edge is never mistaken for an interior peak.
Verified on synthetic data before trusting it on real simulation.
Result: outer scale=1.005 (err=0.04mm), confidence=inf (only one
genuine peak survives once the tail is disqualified). Inner
scale=0.970 (err=0.18mm), confidence=1.20.
**Caveat found, not hidden**: the homogeneous-medium control's INNER
fit also reports confidence=inf (pure noise can have exactly one local
max by chance) — local-max-only selection is a real, principled
improvement, but the confidence-ratio metric alone is not a fully
reliable stand-alone safety check against false detections on absent
signal; a peak-height-vs-absolute-baseline criterion would be needed
to close this gap, not yet implemented.

New/updated: `jwave_test/src/phase3_mri_8probe_test.py` (extended with
`select_best_local_peak`), `jwave_test/results/figures/phase3_mri_8probe_localmax_test_patient023.png`.

**FORK PUSHED + SMOKE TESTS PASS (run -58): local-max selection is safe
to consider porting — reproduces every already-validated result
exactly.** Per user: "upload a fork to github first, and run those
smoke tests". Branch `phase3-8probe-localmax-experiment` created off
`master` and pushed to origin (master untouched); all of this session's
work committed (23 files). Before committing: added
`jwave_test/results/mri_irregular_ring_*.npz` and
`mri_motion_cycle_*.npz` to `.gitignore` (new result files derived
directly from real ACDC patient anatomy, not covered by the existing
`data/`-only rule — caught before staging); cleaned up stale unsuffixed
duplicate figures superseded by run -55's corrected numbers.
Smoke test (`src/phase3_smoke_test_localmax_on_validated.py`, isolated):
applied local-max-only selection to patient001's real-shape result and
the synthetic ring's ED/ES-adjacent frames (run -45) using the
EXISTING, unmodified scoring functions — **exact match in every case**,
confirming the fix changes nothing where no tail artifact exists and
only diverges where one does (patient023's outer boundary).
**Status**: gap 1 (no regression) CLOSED. Gap 2 (real 45/135-degree
calibration measurement, still an interpolation assumption) remains
open — recommended before merging the 8-probe GEOMETRY itself as the
new official probe layout. The local-max SELECTOR alone (independent
of probe count) could reasonably be ported now.

New files: `jwave_test/src/phase3_smoke_test_localmax_on_validated.py`.

**OFFICIAL PATCH (run -59): local-max selection is now the DEFAULT in
the shared 4-probe pipeline. patient023 result is real but MIXED, not
the 8-probe run's clean win.** Patched
`phase3_ring_curvature_weighted_fit.py` (added `select_best_local_peak`,
also fixed a latent segment-splitting bug in the old confidence calc),
`phase3_mri_irregular_ring_reconstruction.py` and
`phase3_mri_motion_cycle_reconstruction.py` (`fit_scale_curvature_weighted`
now returns `(scale, scores, is_genuine_peak, confidence)`). 8-probe
geometry NOT ported — stays on the `phase3-8probe-localmax-experiment`
branch pending the real 45/135-degree calibration measurement.
Re-verified: patient001 static reconstruction EXACTLY matches run -55
(0.995/0.03mm inner, 1.035/0.26mm outer) — patch is safe. patient023
static: outer err 2.43mm->2.25mm (modest ~7%, still lands on a small
peak at the guard-band edge, NOT the near-true bump at scale=0.94 —
confirms the 8-probe run's big win needed the extra probes, not just
better selection). patient023 motion cycle: outer RMSE 2.2940->1.9053mm
but HIGHLY INCONSISTENT across phases (0.49mm at phase 2/7, ~2.0-2.3mm
elsewhere); inner RMSE got slightly WORSE (0.8014->0.8354mm, real
regressions at phases 2/3/6/7). Reported honestly as a mixed result,
not spun as a win.

Figures regenerated: `jwave_test/results/figures/phase3_mri_irregular_ring_reconstruction_patient001.png`,
`..._patient023.png`, `jwave_test/results/figures/phase3_mri_motion_cycle_reconstruction_patient023.png`.

**REAL 45/135-DEGREE CALIBRATION MEASURED (run -60), REPLACING THE
INTERPOLATION ASSUMPTION — discovered old 4-probe 90/180 values don't
transfer to the 8-probe geometry.** `src/phase3_8probe_calibration_45_135.py`
measured all 4 non-monostatic baseline categories (45/90/135/180
degrees) self-consistently within the 8-probe geometry at R=41/71/88,
after finding the 8-probe geometry's own 90-degree measurement (0.238
at R=41) does NOT match the original 4-probe calibration (0.136) —
cross-geometry calibration mixing is invalid. `phase3_mri_8probe_test.py`'s
weight model updated to an exact per-category lookup (no baseline-angle
interpolation needed — 8 probes at 45-degree spacing only ever produce
separations of exactly 0/45/90/135/180 degrees).

**Re-verified with real calibration (run -61): patient023 outer error
2.43mm->0.62mm** (real, ~75% reduction — less dramatic than the
assumption-based 0.04mm, which is now understood to have been partly
an artifact of the unvalidated interpolation). Homogeneous control's
outer channel shows confidence=inf (same known false-positive risk as
before, now on a different channel) — not resolved.

**Breadth-of-validation (runs -62/-63)**: applied to patient001 static
(run -62) — confirms clean, matches/slightly betters the 4-probe
pipeline (0.03mm/0.18mm), homogeneous control fully low-confidence
here. Applied to patient023's FULL real motion cycle (run -63,
`src/phase3_mri_8probe_motion_cycle_test.py`, new isolated script) —
**MIXED result, does NOT uniformly outperform 4 probes**: outer RMSE
improved (1.7482mm vs 4-probe's 1.9053mm) but inner RMSE got WORSE
(0.9978mm vs 0.8354mm), and phases 3/6 show a genuine NEW failure mode
— a confidently-wrong (confidence=inf) inner overshoot (fitted 7.10mm
vs true 5.40mm) not seen in any static test. **Recommendation: do NOT
promote 8 probes to the official pipeline** without first diagnosing
why phases 3/6 fail this way — an open, not-yet-investigated problem.

New files: `jwave_test/src/phase3_8probe_calibration_45_135.py`,
`jwave_test/src/phase3_mri_8probe_motion_cycle_test.py`,
`jwave_test/results/figures/phase3_mri_8probe_localmax_test_patient001.png`,
`jwave_test/results/figures/phase3_mri_8probe_motion_cycle_test_patient023.png`.

**ROOT CAUSE DIAGNOSED + CONFIDENCE-METRIC FIX (runs -64/-65): phase
3/6's failure is a "no real peak at all" problem, not an artifact-vs-
true-peak competition — confirmed by plotting the raw score curve
directly.** `src/phase3_diagnose_8probe_phase3_failure.py` showed the
inner score curve at frac=0.61 is a smooth monotonic decline with NO
bump at the true scale — the only "genuine local max" found is a
noise-level wiggle (height 0.58 vs the curve's excluded edge-max of
1.0), which is why `confidence=inf` was reported despite being wrong
(zero competing peaks makes best/second-best undefined).

Per user's proposed fix, added to `phase3_mri_8probe_test.py`'s
`select_best_local_peak`: **prominence** = absolute peak height
relative to the curve's own dynamic range (independent of whether a
competitor exists), plus **SNR vs. homogeneous control** (real peak
score / homogeneous score at the same candidate, reusing the
already-captured homogeneous pairs, no extra simulation) and a
post-hoc **temporal-consistency** outlier check (MAD-based, flags
frames whose fit deviates from neighbor-smoothed trajectory — a
diagnostic flag only, not a forced correction).

**Validated on patient023's full cycle: prominence (0.18 vs 1.00
everywhere else) and temporal-consistency (outlier_score 3.9/3.3, only
frames crossing 3.0) BOTH independently and correctly flag phase 2/5
(frac=0.61) with no false negatives.** Miscalibration found and
reported: the printed prominence threshold (0.7) was tuned for inner
and too strict for outer, false-flagging some genuinely good outer
frames — conclusion: flag RELATIVE (within-cycle) outliers or use
channel-specific thresholds, not one fixed universal cutoff. Not yet
implemented as the final rule; raw metrics are reported regardless.

This closes the SAFETY half of the diagnosis (detecting confident
wrongness). The ACCURACY half — user's proposed scale+translation+
low-order-deformation-mode model (replacing pure global scale) — is
scoped as a separate, larger undertaking, not yet implemented.

New files: `jwave_test/src/phase3_diagnose_8probe_phase3_failure.py`,
`jwave_test/results/figures/phase3_diagnose_8probe_phase3_failure_patient023.png`.

**DEFORMATION-MODEL HYPOTHESIS TESTED AND REFUTED (run -66), before
building it.** Before committing to the larger scale+translation+
deformation-mode undertaking (scoped in run -65), tested its core
premise directly and cheaply: `src/phase3_diagnose_true_shape_signal.py`
refit phase 2's (frac=0.61) inner boundary against that SAME frame's
own TRUE contour (the best possible template — literally the correct
answer). If shape mismatch were the cause of the overshoot failure,
this should produce a sharp peak at scale=1.0. **It did not** — same
featureless, monotonic-decline curve as the fixed-ED-template case (run
-64), just relocated within a too-narrow scale window on the first
attempt (a false-positive "peak" was caught and corrected via visual
re-inspection before being reported). **Conclusion: do NOT build the
deformation-mode model on the strength of this failure** — the phase
3/6 problem is a signal-availability issue (too weak/incoherent to
detect with ANY shape hypothesis), not a template-shape problem. If
deformation modes are still worth pursuing, they need a different,
verified motivating case. Open question, not investigated further: why
frac=0.61 specifically produces weak inner-boundary signal for
patient023 when frac=0.95 (phase 4/5) works excellently.

New files: `jwave_test/src/phase3_diagnose_true_shape_signal.py`,
`jwave_test/results/figures/phase3_diagnose_true_shape_signal_patient023.png`.

**LITERATURE GROUNDING + COHERENCE FACTOR IMPLEMENTED (run -67).** Web
research mapped this session's ad hoc fixes onto established fields:
(1) "more probes" = Full Matrix Capture/Total Focusing Method (FMC-TFM)
in ultrasonic NDT; (2) "confidence=inf but wrong" = a documented,
known limitation of Coherence Factor (CF) adaptive beamforming (weak
real signals get suppressed by a small CF, exactly matching run -64's
finding); (3) "safe temporal borrowing" = robust/outlier-aware Kalman
filtering + spatiotemporal-regularized cardiac speckle-tracking echo
(confidence-gated + temporal-smoothness, a validated combination for
this exact clinical application).

Implemented a proper Mallart-Fink-style CF in
`phase3_mri_8probe_test.py`'s `fit_scale_curvature_weighted` (now a
6-tuple return, pairs treated as "channels"). Result on patient023's
motion cycle: **inner channel — CF cleanly flags phase 2/5 as worst
(0.270 vs 0.428-0.507 elsewhere).** **Outer channel — CF does NOT
cleanly rank accuracy**: phase 2/5 still gets lowest CF (correctly),
but phase 1/6 (a genuinely bad result) has the HIGHEST outer CF,
while phase 3/4 (the best result) has one of the lowest — same
miscalibration pattern as `prominence` (run -65), now confirmed with a
literature-standard formula too. Conclusion: CF/prominence/SNR/temporal
-consistency all agree on catching the ONE catastrophic case but
disagree on ranking moderate frames — likely reflects the outer
boundary's structurally lower signal redundancy (runs -44/-53), not a
fixable confidence-formula issue. Robust/outlier-aware temporal
filtering (the third literature-grounded direction) remains
unimplemented as an actual estimator — still just a diagnostic flag.

**ROBUST TEMPORAL ESTIMATOR IMPLEMENTED + TESTED (run -68): real RMSE
win on patient023, but safety validation is only a PARTIAL pass.**
`src/phase3_robust_temporal_estimator.py` — precision-weighted
(Kalman-style) fusion of each frame's own CF-derived precision against
a neighbor-based prior (never discards the raw value). On patient023's
already-computed results: inner RMSE 0.9963->0.4363mm, outer
1.7488->0.9453mm, mostly from fixing the one catastrophic frame
(phase 2/5); a few already-decent frames get slightly worse (honest
tradeoff, not hidden). A `sharpness` exponent tuning attempt was tested
and REJECTED (made things worse — all CFs in this cycle are moderate,
so a power>1 shrank precision separation instead of sharpening it).
**Safety validation** (`src/phase3_validate_temporal_estimator_synthetic.py`,
a fast synthetic ring cycle with one deliberate genuine jump modeling
an ectopic beat): direction is safe (a real change is never erased or
inverted), but the fusion still meaningfully degrades an otherwise-
perfect measurement (0.09mm raw error -> 0.58mm posterior error) even
when that frame's own evidence is correctly identified as more
reliable than its neighbors. Caught the script's own auto-generated
"PRESERVED (safe)" verdict being a borderline pass by construction
(loose 0.5mm threshold) before accepting it. **Conclusion: NOT yet
validated/safe to rely on as-is** — likely fix is a hard gate (trust
own value fully above some absolute CF floor, rather than always
continuously blending), not yet designed. Per user instruction, this
run's commits are LOCAL ONLY on `phase3-8probe-localmax-experiment`,
not pushed to origin.

New files: `jwave_test/src/phase3_robust_temporal_estimator.py`,
`jwave_test/src/phase3_validate_temporal_estimator_synthetic.py`.

**FIRST 2-COMPONENT (LV+RV) RECONSTRUCTION TEST (run -69), standard
4-probe model.** Every prior real-MRI test (runs -47 onward)
deliberately excluded RV to keep a direct 2-tissue analog of the
synthetic ring phantoms. `src/phase3_lv_rv_twocomponent_test.py` adds
RV back in as a spatially-separate third tissue region (ACDC has no
LA/appendage label — confirmed and clarified with the user, who chose
RV as the real second chamber). Same zoom_factor/smoothing/placement
as run -47's LV prep, applied consistently to RV too (0px overlap
confirmed after smoothing; RV centroid 88.9 cells from the ring
centroid, matching real anatomy's RV-adjacent-to-LV-free-wall
position). Needed a much wider search grid (256x256, +/-229 cells).
**Result: LV inner (1.005/0.03mm) and epicardium (1.045/0.33mm) match
run -55's RV-excluded baseline closely — RV's presence does NOT
contaminate the primary chamber's fit.** **RV's OWN fit undershoots
noticeably (scale=0.705, error=1.28mm)** — visually confirmed shrunken
inward from the true crescent shape. Plausible (not yet tested)
explanation: RV's crescentic/elongated shape has locally-varying
(concave-then-convex) curvature unlike every circular boundary the
curvature-weight calibration was measured against. Not yet diagnosed
further (curvature recalibration vs. search-grid/placement artifact
both untested). Commit is LOCAL ONLY, not pushed per user instruction.

New files: `jwave_test/src/phase3_lv_rv_twocomponent_test.py`,
`jwave_test/results/figures/phase3_lv_rv_twocomponent_test_patient001.png`.

**LANDMARK (run -70): first GENUINE BLIND shape reconstruction test —
no shape family assumed, only a known center.** Every prior
"reconstruction" in this thread (circle/triangle/heart-cartoon/ring/
real-MRI/RV) swept one scalar (radius or scale) against an
ALREADY-KNOWN shape — the backprojection image was blind, the readout
never was. `src/phase3_blind_shape_reconstruction_test.py` fixes this:
per-angle (144 angles) independent radius discovery using the same
validated curvature-weighting + local-max selection, no cross-angle
shape assumption. Tested on the synthetic ring's inner boundary (known
exact circle, standard 4-probe). **Result: perfect along the 4 probe
axes (0.00-0.10mm), but severe structured corruption between probes
(diagonal angles balloon to 96 cells vs true 60, 3.6mm error)** —
visually confirmed locking onto the accumulator's own diagonal "ghost"
streaks (the same adjacent-probe-pair mechanism diagnosed for the
triangle vertex, runs -29/-31/-36). This is the first direct
measurement of the method's genuine shape-blind angular resolution,
and explains mechanistically why this thread pivoted to global
template-matching in the first place. Gives a precise case for more
probes specifically for BLIND reconstruction (stronger than the
justification for scale-only estimation). Commit is LOCAL ONLY, not
pushed.

New files: `jwave_test/src/phase3_blind_shape_reconstruction_test.py`,
`jwave_test/results/figures/phase3_blind_shape_reconstruction_test_ring.png`.

**8-PROBE BLIND TEST CONFIRMS THE MECHANISM (run -71).**
`src/phase3_blind_shape_reconstruction_test_8probe.py` — identical
per-angle blind method as run -70, same synthetic ring, 8-probe
geometry instead of 4. RMSE 1.3816mm -> 0.3249mm (~4.25x), ghost-
affected angles dropped from ~half the circle to 8/144 (~5.5%),
visually confirmed as narrower/smaller residual spikes exactly between
the now-denser probe set. A cleaner, more directly-explained
improvement than the earlier probe-count tests gave the scale-only
method. Commit is LOCAL ONLY, not pushed.

New files: `jwave_test/src/phase3_blind_shape_reconstruction_test_8probe.py`,
`jwave_test/results/figures/phase3_blind_shape_reconstruction_test_8probe_ring.png`.

**BLIND RECONSTRUCTION ON OFF-CENTER CONCAVE HEART, 8 PROBES (run
-72): substantially harder than the circle, genuine surprise about
WHERE it breaks.** `src/phase3_blind_shape_reconstruction_test_8probe_heart.py`
— same blind method, same 8 probes, on the off-center 10-vertex
concave heart phantom from runs -37/-38 (only known center given, no
shape info). RMSE=1.544mm — worse than even the 4-probe CIRCLE result
(1.38mm), 51% of angles >1.0mm error, visually confirmed as wild
ghost-artifact spikes (not smooth bias). **Surprise**: the CONCAVE
notch (predicted by run -36's mechanism to be a likely ghost source)
shows EXCELLENT accuracy (0.10-0.18mm); the sharp CONVEX tip is where
it breaks down (0.87-1.90mm) — the opposite of the pre-run hypothesis,
reported honestly rather than omitted. Conclusion: "more probes helps"
does not automatically generalize from a circle to genuinely irregular
multi-featured shapes for BLIND reconstruction — ghost mechanisms
multiply with shape complexity. Not yet tested: whether the non-blind
global shape-fit method (already validated on this same phantom,
runs -37/-38) also struggles with 8 probes, to isolate whether this is
blind-discovery-specific. Commit is LOCAL ONLY, not pushed.

New files: `jwave_test/src/phase3_blind_shape_reconstruction_test_8probe_heart.py`,
`jwave_test/results/figures/phase3_blind_shape_reconstruction_test_8probe_heart.png`.

**BLIND RECONSTRUCTION, 16 PROBES (run -73): circle keeps improving
cleanly, but the irregular heart phantom gets ZERO net benefit from
doubling probes again — decisive, non-monotonic result.** Per user:
"try the 16 probe model (4x4), or if you have a mathematically more
favourable observer field (a hex or circular vs square field)" —
answered directly (uniform circular spacing minimizes max angular gap;
grids are for tiling an area, a different problem) before building.
New self-contained 16-probe module `src/phase3_mri_16probe_test.py`
(22.5-degree spacing) and fresh self-consistent calibration
`src/phase3_16probe_calibration.py` (measures all 8 new non-monostatic
separation categories from scratch, per run -60's finding that
calibration doesn't transfer across geometries; saved to
`results/mri_16probe_calibration.npz`). Same per-angle blind method as
runs -70/-71/-72 in `src/phase3_blind_shape_reconstruction_test_16probe.py`
(parameterized: `circle` or `heart` via argv). Circle: RMSE=0.0986mm,
0/144 angles >1mm — continues the clean scaling trend (4->8->16
probes: 1.38->0.32->0.10mm). Heart (off-center, concave, 10-vertex):
RMSE=1.6742mm — slightly WORSE than 8-probe's 1.544mm (run -72),
67/144 angles >1mm (vs 74/144 — not a real improvement once RMSE is
weighed), failure pattern shifted location rather than shrinking.
**Conclusion, now decisive: "more probes" is a clean fix specifically
for smooth/convex boundaries and does NOT generalize to real,
multi-featured anatomical irregularity for blind reconstruction.**
Next most valuable steps: (1) test whether the non-blind global
shape-fit method also fails to improve with more probes on this same
heart phantom, to isolate whether the limitation is physics-
fundamental or blind-discovery-specific; (2) the real, irregular MRI
shape (patient001) blind test, still not attempted. Commit is LOCAL
ONLY, not pushed.

New files: `jwave_test/src/phase3_mri_16probe_test.py`,
`jwave_test/src/phase3_16probe_calibration.py`,
`jwave_test/src/phase3_blind_shape_reconstruction_test_16probe.py`,
`jwave_test/results/mri_16probe_calibration.npz`,
`jwave_test/results/figures/phase3_blind_shape_reconstruction_test_16probe_circle.png`,
`jwave_test/results/figures/phase3_blind_shape_reconstruction_test_16probe_heart.png`.

**INJECTIVITY PROBE, TIP VS. NOTCH (run -74): neither pure hallucination-
hazard nor pure readout bug — a third, harder case.** Direct test of
whether a learned (U-Net) prior for blind reconstruction would be
recovering real information or hallucinating, run BEFORE training
anything: `src/phase3_tip_notch_sensitivity_test.py` perturbs ONE true
vertex of the heart phantom (tip or notch) by +5 cells radially,
resimulates, and compares the RAW backprojection score curve along that
ray (no peak-selection algorithm involved) before vs. after. Notch:
raw argmax tracks 3.5 of the true 5-cell shift, and the sign of change
at both the old and new true-R positions is physically correct (score
drops where the boundary left, rises where it arrived) — information is
genuinely present and cleanly exploitable, matching run -72's already-
good classical result. Tip: raw argmax is stuck on a dominant, fixed
ghost peak (R~17 cells, nowhere near the true 50-55 range) that barely
moves with the true shift — confirming a geometry-driven artifact,
consistent with runs -42/-44/-72 — but the score AT the true tip
location does shift measurably (+11 to +27%), ruling out zero
information; the shift's DIRECTION is physically inconsistent (score
rose, not fell, where the boundary moved away from), the signature of a
noise/crosstalk-dominated residual rather than a clean specular return.
**Conclusion: a learned prior could plausibly improve tip accuracy by
learning to discount the dominant ghost, but any LARGE accuracy win
specifically at the tip should be treated as suspect (hallucination)
rather than trusted recovery — a win at the notch is much more
credible.** Gives a concrete, falsifiable pre-registered check to apply
to any future trained model. Commit is LOCAL ONLY, not pushed.

New files: `jwave_test/src/phase3_tip_notch_sensitivity_test.py`,
`jwave_test/results/figures/phase3_tip_notch_sensitivity_test.png`.
