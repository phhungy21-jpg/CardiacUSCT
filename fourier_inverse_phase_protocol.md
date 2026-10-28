# Fourier-analytic inverse problem — phase protocol

**A lab-notebook-style working protocol for `fourier_inverse/`.**
Version 0.1 · Working draft · Started 2026-10-24

## Why this is a NEW project, not a continuation

Every prior Phase II workspace (`jwave_test/`, `jwave_ring_tomography/`,
`jwave_cylindrical_tomography/`) asked the same question empirically:
*simulate full-wave physics, run a reconstruction, measure RMSE, diagnose
the failure*. That loop produced real findings, but most of them were
identifiability findings discovered the expensive way:

- `jwave_test` runs -74/-77 (injectivity probes): only 1 of 14 boundary
  locations was encoded in the sparse-probe data at all.
- `jwave/` scout + `jwave_ring_tomography` run -07: the blood/myocardium
  boundary is invisible to transmission, and run -08 found reflection
  sees it.
- `jwave_ring_tomography` run -06: SIRT cannot remove the sparse-angle
  star artifact.
- `jwave_cylindrical_tomography` runs -14/-18: accuracy is limited by
  element density and (probably) wavelength, not grid resolution.

Each of these is a statement about **what the measurement operator can
and cannot invert**, and each has an analytic counterpart in classical
inverse-scattering theory (Fourier slice theorem, Fourier diffraction
theorem, 1D layer-peeling / reciprocity). This project derives those
statements directly — with Fourier analysis and numerical linear algebra
on CPU in seconds — instead of rediscovering them via GPU simulation.

**Central question:** for a given acquisition design (source positions,
receiver positions, frequency band, transmission vs. reflection), *which
features of the medium are uniquely determined by ideal noise-free data,
which are determined but ill-conditioned, and which are fundamentally
non-identifiable?* Only after that is answered does reconstruction
accuracy become a meaningful question.

**Objective refined 2026-10-24 (user):** fully blind reconstruction.
The solver gets no tissue labels and no property values. The target is
an arbitrary object of finitely many constant regions whose count,
boundaries and values are all unknown. Priority: uniqueness, posed
within this finite-region class. Uniqueness for general compactly
supported objects is trivially true by analytic continuation and does
not discriminate designs (LOG run -04). Stability is deferred but
tracked, because uniqueness without it is not usable. Test objects are
randomly generated synthetic phantoms; tissue values are used only to
set physically plausible contrast ranges.

## Non-goals

- No full-wave simulation (no jWave/k-Wave, no GPU) in Phases 0–3. The
  forward models are closed-form or small linear operators.
  **Amended 2026-10-24 (user decision):** the nonlinear ("real physics")
  step may use jWave on a Colab GPU, driven via the colab-mcp connector.
  Solver: jWave's convergent Born series, validated against an exact
  solution (LOG run -10). Not GMRES with the default PML, which failed
  that check.
- No learned models. Same reasoning as `jwave_test`'s closure: a prior
  can only hallucinate into a null space; this project's job is to map
  that null space first.
- Not a claim about real tissue classification. These models recover
  acoustic parameters (impedance, sound speed, attenuation, contrast);
  naming them as tissues is a separate step.

## Reused vs. not reused

- **Reused unchanged:** cited tissue properties (Mast 2000 / ICRU 61 /
  Duck 1990: WATER, BLOOD, MYOCARDIUM, CHEST_WALL_PROXY), cloned into
  `fourier_inverse/src/tissue.py`. No new tissue values without a cited
  source (e.g. fat and bone are *not* in the cited set; do not add them
  from the external consultation that motivated this project).
- **Reused as test cases, not code:** prior empirical findings listed
  above become *predictions to verify*, checked against the analytic
  theory in Phase 4.
- **Not reused:** all jWave simulation and reconstruction code.

## Phase structure

**Phase dependencies (amended 2026-10-24, user chose to run Phase 3
before Phases 1–2):** Phases 1, 2 and 3 are independent analytic tracks
(1D layered / straight-ray / diffraction). Each depends only on Gate 0,
the numerical-correctness checks. Gate 1 gates Phase 1's own conclusions;
it is not a prerequisite for entering Phase 3. Phase 4 needs whichever
tracks it cross-checks, and Phase 5 needs Gate 2.

### Phase 0 — Scoping and sanity checks (cheap, CPU, seconds)
- 0.1 State each forward model mathematically, with its approximation
  and validity regime written down (this file + `LOG.md`).
- 0.2 Independently verify the external consultation's headline claim
  rather than accept it: **reversing a 1D layered stack (same
  surrounding medium both sides) leaves the complex transmission
  coefficient T(f) unchanged at every frequency, including multiple
  reflections and loss; reflection R(f) is NOT invariant.** Exact
  transfer-matrix model, random stacks, fixed seed.
- ⛔ **Gate 0:** sanity checks reproduce known closed-form results to
  numerical precision.

### Phase 1 — 1D layered medium: what is uniquely recoverable?
Forward model: exact acoustic transfer matrix (normal incidence, all
multiples). Unknowns per layer: thickness d, sound speed c, density ρ
(→ impedance Z = ρc), attenuation α.
- 1.1 Transmission-only: Jacobian rank / singular spectrum of T(f) over
  a band, versus number of layers. Characterize the null space
  (reversal symmetry is one known discrete ambiguity; find the others).
- 1.2 Add reflection R(f): test the classical result that the
  reflection impulse response determines the impedance profile *as a
  function of travel time* uniquely (layer-peeling / Goupillaud).
  Expected residual non-uniqueness: Z vs. (ρ, c) split, and travel time
  vs. depth. Verify numerically.
- 1.3 Conditioning: how bandwidth (low-frequency cutoff, high-frequency
  attenuation limit) turns "identifiable" into "identifiable but
  unstable".
- ⛔ **Gate 1:** numerical rank/null-space results agree with the
  analytic predictions, or the disagreement is explained.

### Phase 2 — 2D straight-ray transmission: Fourier slice theorem
Forward model: travel-time (and log-amplitude) line integrals = Radon
transform of slowness (and attenuation). Fourier slice theorem: each
projection angle gives one radial line of the object's 2D spectrum.
- 2.1 Angular sampling requirement for a band-limited object of a given
  diameter; predict the star-artifact angular spacing and compare with
  `jwave_ring_tomography` run -06.
- 2.2 Limited-angle coverage (e.g. anterior arc) → missing wedge in
  k-space → explicit null space (which boundary orientations vanish).
- 2.3 Straight-ray validity: bent-ray error scale vs. contrast and size
  (cross-check `jwave_ring_tomography` run -28/-29's bent-ray gains).

### Phase 3 — 2D diffraction tomography: Fourier diffraction theorem
Forward model: first Born and first Rytov approximations of the
Helmholtz equation. Each (incident direction, frequency) pair samples
the object's spectrum on an Ewald circle; transmission samples the
low-|K| region, reflection the high-|K| region up to |K| = 2k₀.
- 3.1 k-space coverage maps for: single source + 180° receiver arc (the
  external consultation's design), full ring, ring with reflection,
  multi-frequency. Coverage holes = null space.
- 3.2 Explain the blood/myocardium finding in k-space: the boundary's
  information lives at high |K|, reached only by reflection; separate
  *identifiability* (coverage) from *detectability* (contrast ~0.5%
  in c, i.e. an SNR question).
- 3.3 Validity of Born/Rytov at cardiac scale: accumulated phase
  k₀·D·Δc/c for a ~100 mm heart at 0.8–2.5 MHz. State honestly where
  the linear theory breaks and what replaces it (distorted-Born /
  iterative linearization).
- 3.2b Density vs. compressibility: Born far-field amplitude is
  k²[γ̂(K) − cosθ·β̂(K)] (γ compressibility contrast, β = ρ₀/ρ − 1), with
  cosθ = 1 − |K|²/2k². At one frequency, each K carries a single
  combination, so the two cannot be separated. Multiple frequencies
  reach the same K at different θ, which can separate them. Map where
  this is possible and how well conditioned it is.
- 3.4 Discretize the linearized operator, compute SVD; visualize the
  null space and the singular-value decay for each acquisition design.
- ⛔ **Gate 2 (acoustic-physics signoff):** a collaborator with
  acoustic-physics expertise reviews the forward-model approximations
  and their stated validity regimes. Not passable by Claude or the user
  alone (CLAUDE.md). Does not inherit any prior project's Gate 2.
  **Waived 2026-10-24 by user decision** ("this is another approach so
  don't need sign off"): this project proceeds as an exploratory track
  without Gate 2. All of its results are NOT collaborator-reviewed and
  must not be presented as established. This is a deviation from
  CLAUDE.md's general jWave-phase Gate 2 rule, recorded here
  deliberately.

### Phase 4 — Cross-check against prior empirical results
Use the theory to *predict* the outcomes already measured in
`jwave_test`, `jwave_ring_tomography` and `jwave_cylindrical_tomography`,
then compare. Agreement validates the theory as a design tool;
disagreement is itself a finding. Real-anatomy arrays derived here stay
gitignored.

### Phase 5 — Design recommendation (only after Gate 2)
Minimum acquisition (angles × receivers × band × reflection) for which
the target features (outer boundary, inner boundary, bulk contraction)
are identifiable and adequately conditioned — to decide whether any
future simulation compute is worth spending.

## Rules (carried over)
- Fixed seeds (`fourier_inverse/src/seed.py`, SEED=42); log seed per run.
- Never invent tissue properties; every value cited.
- Append a LOG.md entry (Appendix-C style: phase, seed, config,
  dataset, result, gate, observations, next action) after every run;
  update MANIFEST.md when on-disk state changes.
- Never commit patient-derived data.
- Commit/push only when explicitly asked.
