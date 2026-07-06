# Pilot Study Protocol — Cross-Cohort Generalization of Cardiac Motion Recovery from Synthetic Doppler Projections

**A lab-notebook-style working protocol**
Version 0.1 · Working draft

---

## How to use this document

This is a working protocol, not a finished methods section. Treat it like a lab notebook: work top to bottom, and **do not proceed to the next phase until that phase's validation gate passes.** Each phase has three parts — what to *prepare*, what to *do*, and how to *validate* before moving on. The validation gates are the whole point; skipping them is how this kind of pilot quietly fails.

Record everything as you go: what you ran, what the numbers were, what surprised you. If a gate fails, log *why* and what you changed, before you move on.

---

## 0. Scope and hypothesis (do this before any code)

### The one-sentence hypothesis
Write this down and keep it visible. Everything downstream serves it:

> A model trained to recover cardiac tissue displacement from synthetic multi-angle Doppler-like velocity projections — using ground-truth motion derived from single-center cine MRI — retains accuracy when evaluated on a different center/vendor's MRI-derived motion.

### What this study is NOT
State this explicitly, to yourself and later in the writeup:
- Not a solution to motion-compensated USCT
- Not using acoustic wave physics or simulation at all
- Not a clinical validation
- Not a claim about real Doppler hardware

It is a **feasibility / proxy study** testing whether the underlying learning-and-generalization idea holds *before* anyone invests in the expensive acoustic-simulation layer.

### Success criteria (decide now, not after seeing results)
Define, in advance, what would count as:
- **Encouraging:** cross-cohort accuracy drop is modest and the model clearly beats a naïve baseline
- **Null-but-useful:** large cross-cohort drop — still publishable as "here's why this is harder than it looks"
- **Broken:** in-distribution model fails even on its own test fold → pipeline/design problem, not a real finding

Pre-registering your own interpretation guardrails keeps you honest when you're emotionally invested in a positive result.

---

## Phase 1 — Environment and data access

### 1.1 Prepare

**Compute:** A laptop or free Google Colab tier is sufficient for a first pass. No GPU cluster needed. Note: classical registration (Phase 3) is CPU-bound and can be slow — plan for overnight runs on the full cohort.

**Software stack (pin versions in a `requirements.txt`):**
- `python` (3.10+)
- `SimpleITK` — registration
- `nibabel` — reading MRI (NIfTI) files
- `numpy`, `scipy` — math, vector projection
- `scikit-learn` — baseline models, cross-validation utilities
- `torch` — the small CNN (only if you go beyond simple regression)
- `matplotlib` — every sanity check should be *looked at*, not just numbered

**Repository hygiene from day one:**
- Git repo, committed frequently
- Fixed random seeds everywhere (`numpy`, `torch`, any split) — log the seed
- A `data/` folder that is `.gitignore`d (never commit patient data)
- A running `LOG.md` where you paste each run's config + result

### 1.2 Do
1. Register for **ACDC** (Automated Cardiac Diagnosis Challenge) data access. Free, academic.
2. Register for **M&Ms** (Multi-Centre, Multi-Vendor & Multi-Disease Cardiac Segmentation) — and/or **M&Ms-2**. Free, academic.
3. Download both. Confirm you have, per patient: cine MRI volumes across the cardiac cycle, plus the provided segmentation masks (at least end-diastole (ED) and end-systole (ES)).

### 1.3 ⛔ Validation gate 1
- [ ] You can load one patient's MRI volume and its mask, overlay them with `matplotlib`, and the mask visibly lines up with the heart. **Look at it.** Do not trust that files are correct because they loaded without error.
- [ ] You can read voxel spacing / slice thickness from the headers for both ACDC and M&Ms, and you have **written down how they differ.** (You will need this in Phase 5 — mismatched spacing is a top cause of fake "generalization failure.")

> ⚠️ **Lead-time pitfall:** data access approval can take days. Start Phase 1 *first*, before writing any other code, so approval runs in the background.

---

## Phase 2 — Preprocessing and normalization

### 2.1 Prepare
Decide and document a single canonical representation that *both* datasets get mapped into:
- Common voxel spacing (resample both to the same mm/voxel)
- Common orientation convention
- Intensity normalization (e.g., per-volume z-score or min–max)
- A defined region of interest (crop around the heart) if you want to reduce dimensionality

### 2.2 Do
Write one preprocessing function. Run it on both datasets. The output of this phase is a clean, uniform set of volumes + masks with identical spacing and orientation.

### 2.3 ⛔ Validation gate 2
- [ ] A processed ACDC volume and a processed M&Ms volume have **identical spacing, orientation, and array conventions.** Print and compare them.
- [ ] Overlay a processed mask on its processed volume again — still aligned after resampling. (Resampling masks with the wrong interpolation — e.g., linear instead of nearest-neighbor — silently corrupts labels. Use nearest-neighbor for masks.)
- [ ] No patient was silently dropped or NaN'd during preprocessing. Count patients in vs. out.

---

## Phase 3 — Ground-truth motion extraction (the load-bearing phase)

**This is the step your entire result rests on.** If the "ground truth" motion is wrong, everything downstream is confidently wrong. Spend disproportionate care here.

### 3.1 Prepare
- Choose registration method: **start with classical SimpleITK deformable registration** (B-spline or Demons). It's transparent and debuggable. Only consider VoxelMorph later if speed becomes the bottleneck — and if you do, it becomes a second thing you must validate independently.
- Decide the registration target: consecutive-frame registration within each patient's cine sequence, producing a displacement field for each frame transition.

### 3.2 Do
1. For each patient, register frame *t* → frame *t+1* across the cardiac cycle.
2. Store the resulting per-voxel displacement field for every transition.
3. Keep the registration parameters fixed and logged.

### 3.3 ⛔ Validation gate 3 (the most important gate in the whole protocol)
For every patient (or a solid random subset if compute is tight):
- [ ] **Warp-and-check:** take the segmentation mask at a labeled frame (ED), apply your chain of displacement fields to warp it toward the other labeled frame (ES), and compute **Dice overlap against the actual ES mask.** ACDC gives you both — this is a real check-point.
- [ ] Establish a Dice threshold *in advance* (e.g., you decide ≥ 0.80 is acceptable ground truth). Log the distribution across patients.
- [ ] **Look at the worst 3 patients.** Visualize the warped vs. true mask. Understand *why* they failed (through-plane motion? large deformation? poor image quality?).

> ⚠️ **The silent killer:** most people skip this gate, train a model on unverified "ground truth," and only much later discover the model faithfully learned to reproduce registration artifacts. If this gate fails, STOP and fix registration — do not proceed.

> ⚠️ **ACDC label sparsity pitfall:** ACDC labels ED and ES only, not every frame. Your check-point Dice only directly validates the ED→ES chain. Frames in between are validated only indirectly. Note this limitation explicitly; consider it when deciding whether to use all frames or just the ED↔ES interval for the pilot.

---

## Phase 4 — Synthetic Doppler signal generation

### 4.1 Prepare
- Choose a small number of hypothetical probe positions around the torso. **Justify them physiologically** — mimic real clinical acoustic windows (e.g., apical, parasternal) rather than arbitrary points. Document the coordinates and the rationale.
- Find a real, citable number for clinical Doppler velocity measurement error to calibrate your noise model. Do not invent a noise level.

### 4.2 Do
1. For each probe position and each timepoint, compute the **radial component** of the ground-truth velocity field: for each tissue voxel, dot the displacement/velocity vector with the unit vector pointing from probe to voxel. This is your synthetic "Doppler measurement."
2. Assemble the multi-angle signal: stack the projections from all probe positions → this is your model **input**.
3. The corresponding ground-truth displacement field is your model **output/target.**
4. Produce two versions: **noiseless** (for the sanity check below) and **noisy** (calibrated additive noise, for the real experiment).

### 4.3 ⛔ Validation gate 4
- [ ] **Noiseless recovery test:** train/fit a model on the *noiseless* synthetic input. It should recover the displacement field nearly perfectly. If it can't, you have a bug or an ill-posed input geometry — fix that *before* blaming noise or generalization.
- [ ] Sanity-check dimensions and units: velocities in sensible physical ranges (mm/s), projection math sign-correct (motion toward probe vs. away has correct sign).
- [ ] Confirm input tensor shape is consistent and documented: `[n_probes × n_timepoints × spatial dims]` or whatever you settle on — write it down.

---

## Phase 5 — Model training and in-distribution evaluation

### 5.1 Prepare
- **Keep the model small.** With low N, a small 3D CNN or even per-region regression is correct. A large architecture guarantees overfitting and makes your generalization number uninterpretable (you won't know if a gap is generalization failure or variance from too few examples).
- Define at least one **naïve baseline** to beat (e.g., predict mean motion field / zero-motion / nearest-neighbor). A result only means something relative to a baseline.

### 5.2 Do
1. Run **k-fold cross-validation *within ACDC only*.** This gives your stable in-distribution performance estimate and your realistic best-case ceiling.
2. Fix the entire pipeline — architecture, hyperparameters, preprocessing — based on ACDC cross-validation **only.**

### 5.3 ⛔ Validation gate 5
- [ ] Model beats the naïve baseline on ACDC cross-validation (if not, there's no signal to test generalization of).
- [ ] Training/validation curves inspected for overfitting; if the model memorizes, shrink it.
- [ ] **You have NOT looked at M&Ms performance yet.** Confirm this. The moment you tune against M&Ms, it stops being a held-out test.

> ⚠️ **Leakage pitfall:** every decision must be made on ACDC. M&Ms is touched exactly once, in Phase 6. Iterating against M&Ms silently converts your generalization test into training data and invalidates the paper's whole point.

---

## Phase 6 — Cross-cohort evaluation (the actual result)

### 6.1 Prepare
- Freeze everything. No more changes after this point.
- Re-confirm ACDC and M&Ms are in the identical canonical representation from Phase 2 (spacing, orientation, normalization).

### 6.2 Do
1. Take the model trained on ACDC. Evaluate **once**, cleanly, on the held-out M&Ms cohort.
2. Compute the same metrics you used internally.

### 6.3 ⛔ Validation gate 6
- [ ] Before interpreting any drop as "generalization failure," rule out **preprocessing mismatch** as the cause — check that spacing/temporal-resolution differences between cohorts were actually normalized. A drop caused by mismatched slice thickness is not a generalization finding.
- [ ] Report ACDC-internal and M&Ms numbers **side by side**, and quantify the gap. **The gap is your result** — not either number alone.

---

## Phase 7 — Metrics, statistics, and honest reporting

### 7.1 Metrics (report both — they catch different failures)
- **Dice / IoU** of the warped segmentation using predicted vs. true displacement
- **Endpoint error** (mean displacement error, in mm)

### 7.2 Statistics appropriate for low N
- Do **not** headline a single p-value. With small N it's fragile and misleading.
- Report **effect sizes with bootstrap confidence intervals.**
- Frame explicitly as a **feasibility signal**, not a definitive generalization claim.

### 7.3 Reporting honesty (this protects your credibility with the labs you want to reach)
State clearly in the writeup:
- MRI-derived motion, not acoustic measurement
- Geometrically-projected synthetic Doppler, no wave physics
- Idealized probe geometry; noise model is an approximation
- Low N; single cross-cohort comparison
- Explicit sentence: *"a first step toward the harder acoustic USCT reconstruction problem, which we do not attempt here."*

> ⚠️ **Overclaiming pitfall:** framing this as "we solved motion-compensated USCT" damages you with exactly the expert audience you want to attract. Precise scoping reads as *more* credible, not less.

---

## Phase 8 — Writeup and preprint

### 8.1 Prepare
- Target venue framing: a cardiac-motion / computational-modeling workshop (e.g., STACOM at MICCAI) is a realistic, lower-barrier first venue.
- Assemble: hypothesis, data, method (all phases above), the ACDC-vs-M&Ms gap, limitations, and a clearly-scoped "future work" pointing at the acoustic-simulation layer.

### 8.2 Do
- Draft the paper. Make the limitations section genuinely thorough — it's a trust signal.
- Prepare a clean, reproducible repo (seeds, `requirements.txt`, run instructions).
- Post to arXiv only when the work is actually sound. A rushed, overclaiming preprint is a permanent record that costs more than not having one.

### 8.3 ⛔ Validation gate 8 (pre-submission checklist)
- [ ] Every claim in the abstract is supported by a number in the results.
- [ ] Limitations section names every idealization above.
- [ ] Someone else could clone the repo and reproduce the headline number.
- [ ] The word "solved" appears nowhere near "USCT."

---

## Appendix A — Recommended order of operations

1. **Start data access immediately** (has lead time) — Phase 1
2. Preprocessing + normalization — Phase 2
3. **Validate ground-truth registration in isolation** — Phase 3 (Gate 3 is critical)
4. Build + sanity-check Doppler synthesis, noiseless recovery test — Phase 4 (Gate 4)
5. Train + cross-validate **within ACDC only** — Phase 5
6. **One clean evaluation on M&Ms** — Phase 6
7. Metrics, stats, honest writeup — Phases 7–8

---

## Appendix B — Master pitfall table

| Phase | Pitfall | Consequence if missed |
|---|---|---|
| 1 | Assuming instant data access | Timeline slips before you start |
| 2 | Resampling masks with linear interpolation | Silently corrupted labels |
| 2 | Not recording spacing differences between cohorts | Can't distinguish real generalization gap from preprocessing artifact later |
| 3 | Treating registration output as truth without checking | Model learns registration artifacts, not real motion |
| 3 | Ignoring ACDC's ED/ES-only labeling | Over-trusting unvalidated inter-frame motion |
| 4 | Idealized, noiseless Doppler synthesis | Inflates apparent feasibility |
| 4 | Sign errors in radial projection | Physically wrong input, may still "train" and mislead |
| 5 | Large model, small N | Overfitting masquerades as success |
| 5 | Peeking at M&Ms during tuning | Held-out test becomes leaked validation data |
| 6 | Blaming a drop on generalization without ruling out spacing mismatch | Wrong conclusion in the paper |
| 7 | Headlining a single p-value at low N | Fragile, misleading claim |
| 8 | Overclaiming ("solved USCT") | Loses credibility with target audience |

---

## Appendix C — Running log template

Copy this block per run into `LOG.md`:

```
### Run YYYY-MM-DD-##
- Phase:
- Seed:
- Config / hyperparams:
- Dataset + split:
- Result (metric = value ± CI):
- Gate passed? (Y/N):
- Observations / surprises:
- Next action:
```

---

*End of protocol v0.1. Revise as reality pushes back — that's what the log is for.*
