# Pipeline self-diagnosis and roadmap

Written 2026-07-07 following a self-diagnosis review of the full project
(Phase I proxy pipeline + Phase II jWave acoustic layer). Captures: what
exists, what's missing (and why some gaps are core vs. parallel evidence
channels vs. optional), and the recommended path forward. Supersedes
nothing — `PHASE2_TO_PHASE3_DIAGNOSTIC_HANDOFF.md` and `PROXY_AUDIT.md`
remain the authoritative record of the acoustic-physics gates; this
document is about recovery-method/measurement-mode strategy.

## What the pipeline currently includes

### Phase I — proxy learning pipeline (`pilot/`, frozen, complete)
- ACDC (train/CV) + M&Ms (external held-out) cine MRI cohorts
- ED→ES motion field via diffeomorphic Demons registration (pseudo-ground-truth)
- Registration QC / per-patient quality weighting (Dice was too harsh alone;
  surface-distance + quality weights used instead)
- Synthetic Doppler-*like* projections at 3 probe angles (-45°/0°/+45° from
  an anterior-left acoustic window) — **projected ED→ES displacement, NOT
  continuous Doppler velocity** (only one displacement field exists per
  patient; no true velocity signal was ever available)
- Small CNN recovering displacement from the synthetic projections
- External-cohort (M&Ms) generalization test
- Explicit, repeated caveats: no acoustic wave physics, no real Doppler
  hardware, no clinical claim

**What Phase I tests:** can a model learn from simplified multi-angle
projected cardiac motion. **What it does NOT test:** can ultrasound
actually measure that motion (no wave physics in the loop at all).

### Phase II — jWave acoustic simulation (`jwave/` frozen scout + `jwave_test/` active)
- jWave GPU reference reproduction (Gate 1, Colab/T4, versions pinned)
- Homogeneous + two-tissue scout simulations (point source, reflection)
- Cited tissue properties (blood, myocardium, chest-wall proxy — Mast 2000)
- Basic anterior line-array / pitch-catch transducer geometry
- Ring-phantom forward model (synthetic LV cavity + myocardial wall)
- Frozen-scene per-frame motion injection (new medium rebuilt per frame),
  **null-tested** — no spurious motion-correlated artifact
- Toy recovery: two-element pitch-catch pulse-echo range → outer
  myocardial radius, via **envelope first-threshold-crossing detection**
- Noise sweep showing the detector is fragile (cliff-edge failure, not
  gradual) — characterized honestly, not hidden
- **Attenuation solver implemented and validated to ~0.1%** against the
  analytic exp(-alpha*distance) law (proxy audit, run -11)
- Source-amplitude calibration proposed (FDA Mechanical Index anchor)
- Real-anatomy benchmark and a small resumable **pilot dataset** (3
  patients, 29 real frames, N=350, real tissue boundary, validated
  attenuation + calibrated amplitude — run -13)
- **Gate 2 (acoustic-physics collaborator signoff): PASSED**, unconditional,
  all 6 proxy-audit items approved as-is (run -12)

**What Phase II tests so far:** can an acoustic simulator produce
plausible, physically-validated wavefields, and can a *toy* detector
recover *known toy* motion. **What it does NOT yet test:** can a
*realistic* ultrasound acquisition robustly recover cardiac motion.

## What's missing — three categories

### Category 1: Core (not optional — the estimation is fragile without these)

1. **RF matched-filtering / cross-correlation tracking.** Current
   recovery throws away phase/waveform information (envelope threshold
   only). This is *why* Phase 3's noise sweep collapsed at just 2% noise.
   Upgrade path: raw RF trace → matched filter (correlate against the
   known transmitted pulse shape) → cross-correlation / phase-shift
   displacement estimate. This is the single highest-value, most tractable
   fix available right now — it operates on RF data the pilot dataset
   already produces (`receiver_trace` in each `.npz`), no new physics
   needed.
2. **Calibrated, realistic noise model.** Current noise levels (0.0, 0.02,
   0.05, 0.10) are arbitrary fractions of trace amplitude, not calibrated
   SNR/MI. Needs: electronic noise, speckle, clutter, reverberation,
   attenuation-dependent loss, aperture-limited artifacts.
3. **Speckle / distributed-scatterer model.** The blood-myocardium
   boundary reflection is weak by cited physics (R≈-0.0025,
   `../jwave/LOG.md` run -05) — tracking a clean boundary echo alone is
   fragile by construction. Real echocardiography leans on myocardial
   speckle texture, not just boundary reflections.
4. **Beamforming / multi-element array reconstruction.** Current recovery
   is single-pair pitch-catch range, not delay-and-sum or model-based
   beamforming across a real receive aperture.
5. **Multi-angle / vector displacement estimation.** One-directional range
   or Doppler is a *projection* of motion. Cardiac motion is radial,
   longitudinal, circumferential, torsional, and through-plane — a single
   beam direction is fundamentally underdetermined for that.
6. **Explicit 2D-vs-3D scoping decision.** Current work is 2D-only,
   reasonable for feasibility, but through-plane motion is invisible to it
   — must be stated as a limitation, not silently assumed away.
7. **Acoustic-physics signoff** — **DONE** (Gate 2 passed, run -12). Listed
   here for completeness of the original diagnosis; no longer open.

### Category 2: Parallel evidence channels for an eventual fusion model

These aren't small add-ons — they're different partial observations of
one latent motion field u(x,t), each with a distinct role:

| Channel | Informs | Role |
|---|---|---|
| Doppler / tissue velocity | axial velocity projection | prior / direction cue |
| RF cross-correlation | local time-shift / displacement | strongest near-term displacement cue |
| Speckle tracking | 2D/3D deformation pattern | myocardial motion cue, less angle-dependent |
| Pulse-echo boundary timing | interface range | geometric constraint (current method) |
| Transmission time-of-flight | speed-of-sound path change | USCT/tomographic constraint |
| Attenuation map | path loss | physical confidence weighting |
| Beamformed B-mode | anatomy/texture | tracking + segmentation prior |
| Phase I MRI-derived motion | pseudo-ground-truth | training target / weak supervision |
| Biomechanical constraint | smoothness/near-incompressibility | regularization |
| Cardiac-phase gating | temporal alignment | phase prior |

The strategic reframe: **Doppler is one observation of the hidden motion
field, not the backbone.** RF phase shift, speckle, boundary timing,
transmission ToF, and biomechanical priors are parallel channels that
should feed one aggregate estimator, not compete as alternatives.

### Category 3: Optional (helps, not urgent, don't do before Category 1)

Larger models (CNN/Transformer), fancy uncertainty quantification, more
patients/pathologies, full differentiable FWI, clinical echo comparison.
**Risk if done early:** a bigger model can hide a broken signal
representation rather than fix it — Category 1 comes first.

## Recommended roadmap (levels, current position marked)

- **Level 0 — sanity** (known phantom, known motion, no noise): **DONE**,
  this is where Phase 3's toy recovery sits.
- **Level 1 — RF matched filtering**: template-correlate against the known
  transmit pulse instead of envelope-thresholding. **NOT DONE — recommended
  next step.**
- **Level 2 — RF cross-correlation tracking**: local-window time-delay/
  phase-shift estimation between frames. Natural follow-on to Level 1.
- **Level 3 — beamformed speckle tracking**: full receive aperture,
  image-domain block matching.
- **Level 4 — multi-angle fusion**: combine multiple transmit-receive
  paths into vector (not scalar) displacement.
- **Level 5 — USCT-style inversion**: joint speed-of-sound/attenuation/
  impedance reconstruction with motion compensation (differentiable
  jWave/FWI-style).

Current pipeline sits at **Level 0-1**. The pilot dataset (run -13, 3
patients, 29 real frames) is exactly the right substrate to build Level 1
on — it already has real RF traces from real anatomy with validated
attenuation and calibrated amplitude.

## Update (run -14): Level 1 implemented — partial win, not a full fix

Implemented and tested the recommendation above
(`src/phase3_matched_filter_recovery.py`). **Result: matched filtering
beats envelope-threshold detection at every noise level (RMSE 1.00/1.21/
1.22mm vs 1.64/1.66/1.66mm at 2%/5%/10% noise — a ~30-40% reduction), but
neither detector beats the naive constant-baseline (0.740mm) once any
noise is present, and the degradation is still a sharp jump between
noise=0 and noise=0.02, not the smooth/gradual curve hoped for.**

Along the way, a second detector (global `argmax` of correlation
magnitude) fell into the *exact same trap* that motivated the envelope
detector's own earlier fix (run -07): picking the strongest-correlated
echo rather than the nearest one, because a later multipath was nearly
as strong as the true near-boundary echo. Fixed with first-threshold-
crossing on correlation magnitude, mirroring the envelope fix.

**Diagnosis: the bottleneck isn't the detector algorithm, it's the
geometry.** This phantom has two comparably-weak, comparably-timed
reflecting interfaces (chest-wall/myocardium and blood/myocardium, both
R~0.002-0.0035 per cited values) — close enough in strength that *any*
single-echo, single-measurement detector is fundamentally ambiguous
between them under noise. Better detection algorithms alone have a
ceiling here.

**Revised recommendation — go to Level 2/4, not further Level-1 tuning:**
- **Level 2 (frame-to-frame cross-correlation / speckle-style tracking):**
  track how the whole waveform evolves between adjacent frames rather
  than re-detecting a single echo's arrival time each frame from scratch
  — uses temporal continuity as extra information single-frame detection
  discards entirely.
- **Level 4 (multi-angle fusion):** add independent measurements (a
  second transmit-receive path at a different angle) so the
  chest-wall/myocardium vs. blood/myocardium ambiguity can be resolved by
  redundancy rather than by a sharper single-path threshold.

Both are bigger lifts than the matched filter was — worth scoping
deliberately (which one first, what it costs) rather than defaulting to
whichever is easiest, since the diagnosis above suggests either could
plausibly break the ceiling, but neither is guaranteed to.

## Update (run -15): Level 2 implemented — another real, partial improvement

Implemented reference-anchored narrow-window tracking
(`src/phase3_reference_tracking_recovery.py`): locate the near-boundary
echo once from a clean reference (ED/frame-0) trace, then for every other
frame, restrict the search to a narrow window sized to the actual
physically-plausible motion range (~3.0us, computed from the prescribed
2.0mm max radius change) — structurally excluding the far multipath
rather than relying on thresholds to avoid it.

**Result: Level 2 beats Level 1 at every noise level** (0.60/1.12/1.04mm
vs 1.00/1.21/1.22mm at 2%/5%/10% noise), and is the **first method in this
project to beat the naive baseline (0.740mm) at any nonzero noise level**
(0.604mm at noise=0.02). It does not yet beat baseline at 0.05 or 0.10.
Zero-noise RMSE is slightly worse than Level 1 (0.502mm vs 0.257mm) — a
real, interpretable tradeoff (window-edge discretization costs a little
noiseless precision in exchange for much better noise robustness), not a
hidden regression.

Same validation discipline caught another bug here: round 1 had the
window's lower bound fall below `DIRECT_EXCLUDE_S`, causing it to lock
onto the direct-pulse tail (identical recovered value across all frames —
the same tell as every previous truncation-window bug in this project).

**Pattern across Levels 0->1->2: each is a genuine, verifiable, but
partial improvement — RMSE keeps dropping, but the "beats baseline at all
noise levels" goal keeps receding.** Recommend a deliberate decision point
before Level 3 (beamforming) or Level 4 (multi-angle fusion), both larger
lifts than anything tried so far, rather than continuing to iterate on
single-line detector refinements.

## Update (run -16): realistic-SNR calibration + Levels 3/4 finalized — track closed

**The noise levels tested throughout Levels 0-4 (2%/5%/10% -> 34dB/26dB/
20dB) turn out to bracket the realistic clinical ultrasound SNR range**,
specifically its harder end (deep structures, poor acoustic windows —
cardiac's known weak spot). This reframes everything above: the fragility
characterized isn't an academic toy artifact, it's telling us single-echo
range detection is not robust in the SNR regime real cardiac ultrasound
must operate in.

Levels 3 (7-element delay-and-sum beamforming) and 4 (2-pair multi-angle
fusion, simplified) were implemented and validated (no wraparound/
implementation bugs found; an initial erratic-looking result traced to a
statistics/fairness issue — different methods consuming different
amounts of randomness against too small a sample — fixed by averaging
over 20 independent noise realizations). Final, stable result:

| noise | L2 (mm) | L3 beamformed (mm) | L4 fused (mm) |
|---|---|---|---|
| 0.02 | 1.007 | 0.868 | 0.913 |
| 0.05 | 1.009 | 0.862 | 0.905 |
| 0.10 | 1.009 | 0.863 | 0.904 |

Real, modest, consistent gains (beamforming ~14%, multi-angle ~10%) that
**plateau rather than closing the gap** — neither beats the naive
baseline at any nonzero noise level. The plateau itself is informative:
RMSE barely changes across a 5x noise range, meaning the error is
dominated by an occasional wrong-echo misdetection (a roughly constant
rate), not smoothly-scaling additive noise. Multi-channel/multi-angle
averaging reduces noise around whichever echo gets picked — it doesn't
change *which* echo that is.

**Conclusion: the single-echo detector-refinement track (Levels 0-4) is
closed.** The structural limitation — two comparably-weak reflecting
interfaces make single-echo detection fundamentally ambiguous — is not
fixable by better detection algorithms on the same signal type. Per
explicit decision: **pivoting to distributed speckle tracking** —
aggregating coherent information across many weak scatterers throughout
the myocardium, rather than betting everything on one or two boundary
reflections. This matches the original strategic diagnosis at the top of
this document (Doppler/boundary-echo range tracking was never meant to be
the backbone) and is now backed by a concrete, quantified reason why.

## Update (run -17): first speckle-tracking cut — honest negative result, with a diagnosed cause

Implemented distributed speckle tracking (400 fixed random scatterers
embedded in the myocardial wall, moving coherently with a thin-wall
radial-scaling motion model): tracked a mid-wall material point via
reference-window cross-correlation, both single-channel and 7-channel
beamformed (focused at mid-wall depth).

**Result: speckle tracking did NOT beat boundary tracking in this first
cut** — single-channel RMSE plateaus at ~1.31-1.35mm (worse than Level
2's ~1.01mm and Level 3's ~0.86mm, and worse than the naive baseline).
Multi-channel beamforming gave only a modest ~3-4% improvement, far short
of the ~2.65x (sqrt(7)) expected from ideal coherent noise averaging.

**Diagnosed cause, not dismissed as a dead end**: the mid-wall speckle
signal is genuinely ~670x weaker than the boundary echo (measured directly
— not a bug; caught and correctly REJECTED an initial "noise-scaling
fairness bug" hypothesis when a proposed fix produced identical results).
The likely reason the expected correlation gain didn't materialize: 400
scatterers is probably too SPARSE for true fully-developed speckle
statistics — with few, discrete, well-separated reflectors, different
receive channels can see meaningfully different combinations of the same
few scatterers (partial decorrelation across the aperture) rather than a
shared, coherent pattern, muting the multi-channel averaging benefit real
(dense) speckle tracking relies on.

**Status: open, not closed.** The speckle-tracking hypothesis has NOT
been fairly tested yet — this first cut used too sparse a scatterer field
and too narrow a correlation kernel (a single 126-sample window, not a
larger 2D-style kernel) to exercise the actual mechanism that makes real
speckle tracking work. The concrete open question for next time: does a
much denser scatterer field (thousands, not hundreds) and/or a wider
correlation kernel change this result materially? Genuinely unknown —
worth resolving before concluding anything definitive about whether
distributed speckle tracking can outperform boundary-echo methods in this
simulation framework.

## Update (run -18): boundary-contamination bug found, fixed, and RULED OUT as the cause

Investigated the run -17 negative result rather than accepting it: found
that the "mid-wall search window" (sized for the full cardiac-cycle
motion excursion, ~2.53us) actually CONTAINED both strong boundary echoes
(near boundary 3.80us, target 5.70us, far boundary 7.59us, window
2.70-8.70us) — the speckle tracker was never isolated from the same
strong reflections Levels 0-4 struggled with. Root geometric cause: the
whole-cycle excursion is nearly as large as the wall's own thickness, so
no single fixed window can be both wide enough for the full excursion and
narrow enough to exclude the co-moving boundaries.

Fixed with sequential (frame-to-frame) tracking — real elastography's
actual approach — since per-step motion (~1.07us) is much smaller and
fits comfortably within the boundary-clearance budget. Explicitly
verified boundary exclusion numerically (0.199us clearance each side)
before trusting the result.

**Result: RMSE plateau essentially unchanged (~1.29-1.30mm vs. ~1.31-
1.35mm before the fix).** This is a clean negative control: fixing the
boundary-contamination bug did NOT change the outcome, ruling it out as
the dominant cause and pointing back to the original scatterer-sparsity
hypothesis (400 scatterers, likely far below the ~10+/resolution-cell
rule of thumb for fully-developed speckle) as the more likely explanation.

**Session conclusion**: three iterations (single-channel, multi-channel
beamformed, sequential boundary-excluded) have each ruled something out
without producing a working result. Speckle tracking remains an open,
unresolved question — not refuted, since scatterer density hasn't been
tested yet, but also not yet demonstrated. The natural next test (much
denser scatterer field) is a nontrivial increment, not a quick follow-up.
Recommend a planning discussion on both tracks (single-echo detector
refinement, closed at run -16 with modest partial gains; speckle
tracking, open at run -18) before further investment.

## Update (run -19): genuine multi-angle vector triangulation — first positive result

Per explicit user direction ("simulate multi-angle transmissions and
multi-channel receive, then fuse into vector motion estimates"), built a
proper test distinct from the earlier Level 4 (which only averaged two
similar SCALAR range measurements): two independently delay-focused
8-element sub-apertures, both fixed-focused on the SAME off-axis target
point (30 degrees off boresight, outer myocardial boundary), with look
directions 66.1 degrees apart. Each sub-aperture's echo tracked
(sequential method, run -18), and the two scalar range-change
measurements combined via a 2x2 linear system to recover the full 2D
displacement vector.

**Result: genuine vector recovery — the first in this project.** Row
(boresight) component: RMSE=0.285mm, comparable to the best single-
direction results (Level 2/3). Col (cross-range) component: RMSE=0.499mm,
correct sign/direction throughout but a noisier trajectory — plausibly
because sub-aperture B's oblique incidence angle gives a weaker,
more-spread reflection (a real physical effect) compounding with known
sequential-tracking drift. **Critically, the cross-range component is
information no single-look-direction method in this entire project could
recover at all** — a structurally new capability, not an incremental
accuracy improvement.

This is a genuine positive proof-of-concept for the strategic direction
proposed (multi-angle > single-view for motion, since real cardiac motion
has components no single beam direction can see). Still noiseless-only —
robustness under realistic noise (20-34dB SNR, per run -16's calibration)
is the natural next test, not yet done.

## Update (run -20): tested under realistic noise — survives, with a surprise

Noise sweep (same 20-34dB SNR range that broke every single-direction
method in this project) plus per-sub-aperture RMSE breakdown to
disentangle causes. **Result: col (cross-range) RMSE under noise is
0.67-0.69mm — comparable to or better than this project's best single-
direction result (Level 3, ~0.86mm, run -16) — while also recovering
information no single-direction method can see at all.** Vector
triangulation is the strongest result of the whole session.

**Surprising, honestly-flagged finding**: the on-axis (near-normal-
incidence) sub-aperture is MORE noise-fragile than the oblique one — the
opposite of the run -19 hypothesis. Plausible explanation (unconfirmed):
near-normal reflection off a curved boundary is sharp and strong, but may
be more susceptible to the same wrong-echo ambiguity that limited Levels
0-4, while the oblique, weaker, more spread reflection has fewer
similarly-strong competing features nearby. Since the col (cross-range)
component blends both sub-apertures, it ends up MORE robust than the
row (boresight) component under noise — a reversal from the noiseless
case.

**Overall session verdict**: three strategic directions explored —
single-echo detection (Levels 0-4, closed, modest partial gains, capped
by a structural two-interface ambiguity); distributed speckle tracking
(open, unresolved, scatterer density untested); multi-angle vector
triangulation (genuinely promising, survives realistic noise, recovers
information other methods structurally cannot). Vector triangulation is
the strongest lead to build on going forward.

## Update (run -21): runs -19/-20 RETRACTED — found the real physical limitation

**Continued debugging (per explicit user instruction) uncovered that the
above "genuinely positive" result does not hold up, and found something
more fundamental than a bug.** Two layers:

1. A timing-formula bug (naive symmetric approximation for the
   delay-focused transmit's convergence time) meant sub-aperture B's
   noiseless accuracy was substantially worse (0.938mm, not 0.382mm) once
   corrected — the earlier "A more fragile than B" finding was partly an
   artifact of this bug finding a coincidentally-plausible wrong feature.

2. **The deeper finding, confirmed analytically and numerically**: for a
   curved reflecting boundary, delay-focusing the transmit at a chosen
   point does not guarantee the reflected wave reaching a given receiver
   actually comes from that point — the true specular point depends on
   the full bistatic (Tx,Rx) geometry. Scanning multiple valid (Tx,Rx)
   pairs (via the mirror/virtual-image reflection method) showed: **the
   sensitivity vector d(range)/d(target position) is ALWAYS exactly
   parallel to the target's local surface normal, regardless of viewing
   angle** — a geometric consequence of the law of reflection (incident
   and reflected rays are symmetric about the normal, so their
   difference is always normal-directed), not an implementation artifact.

**Conclusion: multi-angle triangulation of a single point via specular
reflection cannot recover a true 2D displacement vector, structurally —
no amount of correct implementation changes this.** It only ever measures
the normal-direction component. The apparent cross-range recovery
reported in runs -19/-20 was an artifact of mismatched (non-specular)
receiver geometry, not genuine physics.

**This connects the vector-triangulation thread back to the still-
unresolved speckle-tracking question (runs -17/-18).** Genuine 2D/vector
motion recovery needs either: (a) distributed scatterer/speckle tracking
(a discrete scatterer's arrival time DOES depend on true 2D position,
unlike smooth specular reflection — untested at adequate scatterer
density), or (b) tracking multiple DIFFERENT points and inferring
tangential strain/rotation from the spatial variation of their individual
normal-only displacements — the actual mechanism real 2D speckle-tracking
echocardiography uses. Do not pursue further single-point multi-angle
specular variants; the limitation is physical, not fixable by more
careful geometry.

**Revised overall session verdict**: single-echo detection (Levels 0-4)
closed with modest partial gains, capped by structural ambiguity.
Multi-angle specular triangulation of one point: closed, a real physical
limitation identified (not just inconclusive). Distributed speckle
tracking: still the open, most important unresolved thread — and now
understood to be necessary (not just one option among several) for any
genuine vector/strain motion recovery in this framework.
