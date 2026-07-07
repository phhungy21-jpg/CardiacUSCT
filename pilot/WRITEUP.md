# Cross-Cohort Generalization of Cardiac Tissue Motion Recovery from Synthetic Doppler Projections: A Pilot Study

*Draft v0.1 — target venue: STACOM (MICCAI workshop) or similar cardiac-motion/computational-modeling workshop. Not yet submitted anywhere.*

**Authors:** [TBD — fill in before submission]
**Affiliations:** [TBD — fill in before submission]

## Abstract

We test the feasibility of a specific idea: that a small model trained to recover
cardiac tissue displacement from synthetic, multi-angle, Doppler-like radial
projections of MRI-derived motion generalizes across cardiac MRI cohorts collected
at different centers with different scanner vendors. Using ACDC (150 patients, 1
center) for training and 5-fold in-distribution cross-validation, and M&Ms (342
usable patients, 4 vendors) for a single held-out evaluation, a 3,058-parameter CNN
achieves an in-distribution endpoint error of 0.849mm (95% CI [0.814, 0.887]) and
shows **no cross-cohort degradation** on the held-out cohort (0.717mm, 95% CI
[0.704, 0.731]; gap CI [-0.172, -0.095]mm, excluding zero). We report this
result honestly alongside an unresolved confound — the held-out cohort's own
registration-derived ground truth is of measurably lower quality than the training
cohort's — that plausibly explains part or all of the apparent improvement, and we
show a second metric (Dice) that disagrees with the first on the direction of the
gap, which we treat as evidence for that confound rather than evidence of genuine
superior generalization. Accuracy is consistent across all 4 vendors in the
held-out cohort (0.674–0.736mm), which is not subject to the same confound and is
the cleanest positive result in this study. This is explicitly a feasibility/proxy
study: no acoustic wave physics, no real Doppler hardware, and no clinical claim
is made.

## 1. Scope, hypothesis, and what this is not

**Hypothesis:** a model trained to recover cardiac tissue displacement from
synthetic multi-angle Doppler-like velocity projections — using ground-truth
motion derived from single-center cine MRI — retains accuracy when evaluated on a
different center/vendor's MRI-derived motion.

**This is not:** a solution to motion-compensated USCT; a use of acoustic wave
physics or simulation; a clinical validation; or a claim about real Doppler
hardware. It is a feasibility/proxy study testing whether the underlying
learning-and-generalization idea holds before investing in an acoustic-simulation
layer.

**Pre-registered success criteria** (Phase 0, before any result was seen):
encouraging = modest cross-cohort drop, model beats a naive baseline; null-but-
useful = large drop, still informative; broken = in-distribution failure. The
result obtained is closest to "encouraging," reported with the caveats below.

## 2. Data

- **ACDC**: 150 patients (100 training + 50 testing splits, both with ED/ES
  ground-truth masks — this local release includes GT for the "testing" split,
  unlike the original challenge). Single center.
- **M&Ms**: 345 patients, 4 vendors (A: Siemens, B: Philips, C: GE, D: Canon —
  confirmed against the dataset's own metadata CSV), 342 usable after data-quality fixes (below).
  Held out entirely from training; touched exactly once, in the final evaluation.

Both datasets provide cine short-axis MRI with expert segmentation masks at
end-diastole (ED) and end-systole (ES) only — not every cardiac phase.

## 3. Methods

[Figure 1: pipeline overview — `results/figures/pipeline_schematic.png`]

### 3.1 Scope decision: ED→ES only

Ground-truth motion is extracted for the single ED→ES transition per patient, not
the full ~30-frame cardiac cycle. Rationale: the only independently checkable
transition is ED→ES, since that's the only pair with expert segmentation on both
ends. Registering the full cycle would mean the majority of "ground truth" motion
is unverifiable — a pilot's job is a trustworthy small result, not an impressive
fragile one. This means the model recovers a single contraction displacement
field, not full cardiac dynamics; a natural, credible future-work extension.

### 3.2 Preprocessing

Both cohorts are mapped into an identical canonical representation: resampled to
(1.5625, 1.5625, 10.0)mm spacing (the ACDC mode, chosen to minimize interpolation
for the majority of patients), LPS orientation, cropped to a 128×128 in-plane
window centered on the whole-heart mask centroid, per-frame z-scored intensity
normalization. Verified identical spacing/shape/orientation across both cohorts
before any downstream step (Gate 2).

M&Ms required two data-quality fixes before use: (1) its label convention is the
reverse of ACDC's (1=LV/2=myo/3=RV vs. ACDC's 1=RV/2=myo/3=LV) — confirmed by
geometry and remapped explicitly; (2) 25 patients' metadata had a placeholder
ED=ES=0 instead of real frame indices — fixed by detecting the true labelled
frames directly from the mask and assigning ED/ES by LV cavity size (larger =
ED). A further 3 patients (all one vendor/centre) were excluded for genuinely
invalid image headers (non-orthonormal direction cosines). Final usable M&Ms
cohort: 342/345 (99.1%), with vendor representation essentially unaffected (A
95/95, B 125/125, D 50/50, C 72/75).

### 3.3 Ground-truth motion extraction

Displacement fields are extracted via diffeomorphic Demons registration (ED→ES,
per-patient), chosen over B-spline after a method comparison showed it was both
faster and more accurate. All 150 ACDC and 342 M&Ms fields are diffeomorphic
(zero folding voxels).

Gate 3 (validating this ground truth) surfaced an important methodological
finding: the pre-registered Dice≥0.80 threshold, applied per-structure (RV,
myocardium, LV), failed for the majority of the ACDC cohort (27% pass) — but a
diagnostic comparing Dice against surface distance showed 98.7% of patients have
mean boundary error ≤2 voxels (~3mm), and the Dice-vs-distance relationship is
smooth and continuous, not bimodal. This is a case of Dice being a very steep,
demanding function of small absolute error for structures near voxel resolution,
not evidence the motion fields are physically wrong. Gate 3's primary metric was
therefore corrected to surface distance (with both readings reported
transparently), and each patient carries a continuous quality weight
(1/(1+mean surface distance in voxels)) into training rather than a hard
pass/fail filter — all 150 ACDC patients are retained, weighted by registration
quality (mean weight 0.599±0.134; range [0.32, 0.90]; only 2 patients >2 voxels,
both from the hardest-to-register pathology group, kept and down-weighted, not
excluded). [Figure 2: Dice-vs-surface-distance, all 3 labels —
`results/gate3_diagnostic/dice_vs_surfdist_scatter.png`] [Figure 4: qualitative
best/median/worst registration — `results/figures/warp_panel_best_median_worst.png`]
[Figure 5: boundary error vs. real cardiac wall dimensions —
`results/figures/clinical_units_error.png`] [Figure 7: quality-weight
distribution — `results/figures/quality_weight_distribution.png`]

Registration quality correlates with contraction magnitude, not disease
severity per se: patients with weak/reduced contraction (dilated cardiomyopathy)
register best (mean Dice 0.847); patients with vigorous or complex contraction
(healthy, hypertrophic cardiomyopathy) register worst (0.705, 0.667) — large
deformation, not pathology, is what defeats the registration (r = -0.777
between contraction magnitude and Dice). [Figure 3:
`results/figures/quality_vs_contraction.png`] [Table 5: `RESULTS_TABLES.md`]

### 3.4 Synthetic Doppler projection

Three in-plane probe positions at [-45°, 0°, 45°] from an anterior reference,
restricted to the anterior-left arc to match real transthoracic acoustic windows
(parasternal/apical/subcostal; the posterior chest is not a real acoustic window).
The synthetic signal is the projected *displacement* (mm) along the probe-to-voxel
direction — not velocity, since only a single ED→ES displacement field exists per
patient, not a continuous cine. Additive Gaussian noise, SD=1.0mm, calibrated from
tissue Doppler myocardial displacement reproducibility literature (reported SDs of
0.6–1.3mm across spectral/colour TDI vs. anatomic M-mode comparison methods), not
invented. A closed-form per-voxel least-squares recovery on the noiseless signal
confirmed the projection geometry is exactly invertible (recovery error at machine
precision, ≤1e-6mm) before any noisy or learned evaluation.

### 3.5 Model

A deliberately small 2D CNN (3 conv layers, 3→16→16→2 channels, 3×3 kernels,
**3,058 parameters**), operating per-slice (through-plane motion is excluded from
the target, consistent with the documented 10mm-slice-thickness resolution
limit). Trained with weighted MSE loss (per-voxel heart mask × per-patient
quality weight), Adam, 30 epochs, full-batch. All architecture/hyperparameter
decisions were frozen based on 5-fold ACDC-only cross-validation before M&Ms was
touched at all.

## 4. Results

### 4.1 In-distribution (ACDC, 5-fold CV)

The model beats both a zero-motion baseline and a per-voxel mean-motion-field
baseline on **100% of 150 patients**: endpoint error 0.849±0.227mm vs. 3.773±1.047mm
(zero-motion) and 3.151±0.908mm (mean-motion). Train/validation loss curves track
closely across all 5 folds with no divergence — no overfitting signature, expected
given the model's small size relative to ~1,200+ training slices per fold.
[Table 2: `RESULTS_TABLES.md`; curves: `results/phase5_train_val_curves.png`]

### 4.2 Cross-cohort evaluation (M&Ms, one clean pass)

**No cross-cohort degradation in endpoint error.** ACDC (in-distribution) 0.849mm
[95% CI 0.814, 0.887]; M&Ms (held-out) 0.717mm [95% CI 0.704, 0.731]. The gap
(-0.133mm) has a bootstrap 95% CI of [-0.172, -0.095] — entirely negative, unlikely
to be sampling noise. [Figure 6: `results/figures/cross_cohort_result.png`;
Table 3: `RESULTS_TABLES.md`]

**This is not presented as proof of superior generalization.** M&Ms's own
registration-derived ground truth is of measurably lower quality than ACDC's
(Demons mean Dice 0.641 vs. 0.752) — a harder-to-register cohort plausibly yields
a smoother, less-detailed displacement target, which could be easier for a smooth
CNN to match closely independent of true motion-recovery transfer. We checked and
ruled out two simpler explanations: a preprocessing/spacing mismatch (identical
canonical representation confirmed, Gate 2), and M&Ms simply having less motion to
recover (true motion magnitude is only ~7% smaller on M&Ms, not enough to explain
a ~16% smaller error).

The most informative single fact here is that a second metric disagrees with the
first on *direction*: warped-mask Dice is 0.589 on M&Ms vs. 0.664 on ACDC (95% CI
of the gap [-0.095, -0.055], also excluding zero) — worse on M&Ms, the *expected*
direction for a genuine cross-cohort drop. If the model were genuinely
generalizing better or worse on equal-quality ground truth, both metrics would be
expected to move the same way. This disagreement is best read as evidence that the
ground-truth-quality confound, not genuine superior generalization, is driving the
endpoint-error result.

**The cleanest positive result: consistency across vendors.** Within the held-out
cohort (not subject to the ACDC-vs-M&Ms ground-truth-quality confound, since it's
an entirely within-M&Ms comparison), endpoint error is tightly clustered across
all 4 vendors: 0.720, 0.736, 0.707, 0.674mm (range 0.062mm), with no outlier
vendor. [Figure 6b: `results/figures/vendor_breakdown.png`; Table 4:
`RESULTS_TABLES.md`]

No single p-value is headlined anywhere in this study (fragile/misleading at this
N); bootstrap confidence intervals and Cohen's d effect sizes are reported instead.

## 5. Limitations

*(Full detail in `LIMITATIONS.md`; summarized here.)*

1. **Ground-truth motion is registration-derived, not true motion.** Median
   boundary error ~1 voxel (~1.5mm), rising to ~2-3 voxels for large-contraction
   hearts. RV is the least accurate structure. Through-plane (base-apex) motion is
   under-resolved by 10mm slice thickness.
2. **The cross-cohort result's ground-truth-quality confound** (Section 4.2) is
   not ruled out and should be treated as the primary caveat on the headline
   number.
3. **Synthetic Doppler assumes full-field, shadow-free acoustic access** and
   additive Gaussian noise — real ultrasound has acoustic shadowing/dropout,
   multiplicative speckle (with decorrelation under large deformation), dynamic
   acoustic impedance, speed-of-sound distortion, and myocardial anisotropy, none
   of which are modeled. These results should be read as an optimistic upper
   bound on achievable motion recovery from real acoustic hardware, not a
   performance estimate of one.
4. **Single ED→ES transition, not full cardiac cycle** (Section 3.1).
5. Low N by deep-learning standards (150/342), appropriate to a pilot; no
   claims here should be read as more than a feasibility signal.

## 6. Future work

- Full-cycle (multi-frame) motion recovery, once the ED→ES feasibility signal is
  established (this study).
- A shadowing/dropout ablation (randomly zero angular sectors or myocardial
  segments during Doppler synthesis) to directly probe robustness to the single
  most important acoustic-realism gap.
- Extending to a genuine acoustic-simulation layer — the eventual target this
  pilot is de-risking, not attempted here.

## 7. Reproducibility

Seeds fixed throughout (`src/seed.py`); pinned `requirements.txt`; full run log
with every config/result/decision in `LOG.md`; a checksummed freeze manifest of
all Phase 1-4 outputs before Phase 5 began (`results/phase4_freeze_manifest.csv`).
`data/` is gitignored (never committed) — a reader would need their own ACDC/M&Ms
access to reproduce end-to-end, but every processing step is scripted in `src/`.

## References

*Dataset and guideline citations below are given with source identifiers
(PMID/URL) gathered during this project's research. Author lists, exact
journal/volume/page details, and formatting should be verified against the
original records before final submission — they are not fabricated, but full
bibliographic details were not independently re-verified beyond the source
title/identifier.*

1. Bernard O, et al. "Deep Learning Techniques for Automatic MRI Cardiac
   Multi-Structures Segmentation and Diagnosis: Is the Problem Solved?" IEEE
   Transactions on Medical Imaging, 2018. (ACDC dataset.) *[Verify exact
   citation before submission.]*
2. Campello VM, et al. "Multi-Centre, Multi-Vendor and Multi-Disease Cardiac
   Segmentation: The M&Ms Challenge." IEEE Transactions on Medical Imaging,
   2021. (M&Ms dataset.) *[Verify exact citation before submission.]*
3. "Guidance for accurate and consistent tissue Doppler velocity measurement:
   comparison of echocardiographic methods using a simple vendor-independent
   method for local validation." PMID: 24699322.
4. "Reproducibility of pulsed wave tissue Doppler echocardiography." PMID:
   10359921.
5. "Pulsed-wave tissue Doppler and color tissue Doppler echocardiography:
   calibration with M-mode, agreement, and reproducibility in a clinical
   setting." PMID: 19594813.
6. "Accuracy of spectral Doppler flow and tissue velocity measurements in
   ultrasound systems." PMID: 14962617.
7. "Grading of myocardial dysfunction by tissue Doppler echocardiography: a
   comparison between velocity, displacement, and strain imaging in acute
   ischemia." PMID: 16631008.
8. "Measurements of left ventricular myocardial longitudinal systolic
   displacement using spectral and colour tissue Doppler: time for a
   reassessment?" PMC2661320.
9. American Society of Echocardiography. Recommendations for chamber
   quantification / right ventricular size and function (ASE 2015 guideline
   family) — normal wall thickness ranges cited in Figure 5. *[Verify exact
   citation: "Recommendations for Cardiac Chamber Quantification by
   Echocardiography in Adults," J Am Soc Echocardiogr, 2015.]*
