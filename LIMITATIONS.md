# Known limitations (deferred, for the eventual writeup)

Two distinct gaps exist in this pilot. Keep them separate — they belong to
different phases and different urgency.

## Gap 1 — MRI motion → registration-extracted motion (Phase 3 — resolved via metric correction + per-patient weighting)

The ground-truth motion this pilot trains on isn't "true cardiac motion" —
it's the displacement field produced by Phase 3's registration, which
carries registration error and through-plane blindness with it.

**Ground-truth error floor.** Registration-derived motion has a median
boundary error of ~1 voxel (~1.5mm), but this rises to ~2-3 voxels for
patients with large ED->ES contraction (predominantly the NOR and HCM
pathology groups — vigorous/complex contraction proved harder to register
than DCM's weaker, easier-to-track contraction; see `LOG.md` runs
2026-07-06-06/-11). **RV is consistently the least accurate structure**
(median ~1.1 voxels, worse tail than LV/myocardium). **Through-plane
(base-apex) motion is under-resolved** by the 10mm slice thickness common
in this cohort — the registration cannot see motion that happens between
slices. Every downstream result (Phase 4 Doppler synthesis, Phase 5/6
model training and evaluation) inherits this ground-truth error floor —
reported accuracy numbers are relative to this imperfect ground truth, not
to true cardiac motion.

**Methodology note (metric, not just data quality):** an extensive
registration method search (B-spline at 2 resolutions, intensity-driven
diffeomorphic Demons, and mask-guided Demons refinement at several label
weightings — `LOG.md` runs 2026-07-06-04 through -10) initially could not
get Dice >= 0.80 on all three structures at once, which looked like a
registration failure. A dedicated diagnostic (run -11) showed this was
substantially a **Dice-vs-thin-structure metric artifact**: 148/150
patients have mean surface distance <=2 voxels across all labels even
though only 27% pass the Dice gate — Dice is a very steep function of
small absolute error for structures near voxel resolution. Gate 3's primary
metric was therefore changed from Dice to surface distance, a decision
argued transparently in `LOG.md` with both readings shown side by side (not
a post-hoc relaxation — the thin-structure concern was flagged in run -05,
*before* the failing full-cohort Dice numbers existed).

**All 150 patients are used** — no patient is hard-excluded. Instead, each
patient carries a **quality weight** (`results/phase3_quality_weights.csv`,
`weight = 1/(1+mean_surface_distance_in_voxels)`) that down-weights
lower-quality registrations (predominantly large-contraction NOR/HCM
patients) in Phase 5 training, rather than filtering them out and
introducing pathology-group selection bias.

## Phase 6/7 cross-cohort result — an unresolved confound to state plainly

**Lead with this framing, always: no cross-cohort degradation in endpoint
error (bootstrap 95% CI on the gap is entirely negative, [-0.172,-0.095]mm
— unlikely to be sampling noise), immediately followed by the confound
below in the same breath, not as a footnote.**

The model showed no accuracy drop moving from ACDC to M&Ms (endpoint error
0.717mm vs. 0.849mm in-distribution; robust per bootstrap, not just noise)
— encouraging by the pre-registered Phase 0 criteria, but **do not present
this as clean evidence of generalization**: M&Ms's own registration-
derived ground truth is lower quality than ACDC's (Demons Dice 0.641 vs
0.752). A harder-to-register cohort plausibly yields a smoother, less-
detailed displacement field, which could make the evaluation target easier
for a smooth CNN to match closely — independent of whether real motion
recovery actually transfers well.

**Key diagnostic: endpoint error and Dice disagree on the direction of the
gap** — endpoint error says M&Ms is (robustly) better, Dice says M&Ms is
worse (0.589 vs 0.664, also excludes zero, in the *expected* direction).
If the model were genuinely generalizing better or worse on
equal-quality ground truth, the two metrics would be expected to agree in
direction. This disagreement is itself evidence pointing at the
ground-truth-quality confound as the likely driver of the endpoint-error
result, not genuine superior generalization.

Ruled out: preprocessing/spacing mismatch (Gate 2 confirmed identical
canonical representation) and "M&Ms simply has less motion" (true motion
magnitude only ~7% smaller on M&Ms, not enough to explain a ~16% smaller
error). Not ruled out: the ground-truth-quality confound above.

**The genuinely clean result: the 4-vendor breakdown within M&Ms** (0.674-
0.736mm endpoint error, tight clustering, no outlier vendor) — this is a
within-M&Ms comparison and is NOT subject to the ACDC-vs-M&Ms ground-truth-
quality confound, so "accuracy is consistent across 4 vendors/scanners" is
a defensible standalone claim.

## Gap 2 — synthetic Doppler → real ultrasound (deferred to writeup/future work)

The synthetic Doppler signal (Phase 4) assumes full-field, shadow-free
acoustic access and models measurement error as additive Gaussian noise.
Real ultrasound differs in physiologically important ways this pilot does
not attempt to model:

- **Acoustic shadowing / dropout.** Ribs, lungs, and sternum block real
  ultrasound entirely — whole myocardial segments have no signal on a given
  view ("poor acoustic window"). The synthetic projection passes cleanly
  through all tissue. Real acoustic windows are also geometrically
  constrained (beam entry points are dictated by rib/lung anatomy), so the
  angular coverage assumed here may not be physically attainable in practice.
- **Speckle, not just noise.** Real ultrasound texture is dominated by
  speckle — multiplicative, spatially correlated, not Gaussian, not
  independent frame-to-frame. Speckle *decorrelation* under large
  deformation is the dominant real-world failure mode of ultrasound motion
  tracking; additive Gaussian noise cannot exhibit this failure mode at all,
  and understates difficulty generally.
- **Dynamic acoustic impedance / echogenicity.** Blood-myocardium boundary
  visibility shifts through the cardiac cycle; fibrosis, scar, and fat alter
  local echo properties. ACDC includes pathological cohorts (DCM etc.) where
  this matters most clinically — the proxy treats healthy and scarred
  myocardium identically as motion-bearing tissue.
- **Speed-of-sound distortion.** Real ultrasound assumes a constant ~1540
  m/s to convert echo timing to depth; actual speed varies by tissue
  (fat/muscle/blood), producing small geometric distortions. MRI-derived
  geometry has no such distortion — the synthetic "measurement" is
  geometrically truer than any real ultrasound could be.
- **Myocardial anisotropy.** Cardiac muscle backscatter is fiber-direction
  dependent (fiber orientation rotates transmurally); the geometric,
  intensity-agnostic projection used here has no analogue for this.

**Framing for the writeup:** these results should be characterized as an
optimistic upper bound on achievable motion recovery, not a performance
estimate of a real acoustic system — real hardware faces strictly harder
conditions than this proxy models. State this explicitly rather than
implying realism.

**Optional Phase 4 strengthening (not yet decided/committed):** a cheap
shadowing/dropout ablation — randomly zero out one or two angular sectors or
a myocardial segment per sample during Doppler synthesis, and report the
accuracy degradation. Converts "we ignored shadowing" into "we tested
robustness to shadowing" for a few lines of code. Revisit when Phase 4
starts; not required for the pilot to be valid without it.
