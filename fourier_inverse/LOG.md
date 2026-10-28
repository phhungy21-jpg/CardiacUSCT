# LOG — fourier_inverse/

Running lab notebook. Protocol: `../fourier_inverse_phase_protocol.md`.
Entries follow the Appendix-C style used across this repo.

### Run 2026-10-24-00 — Project created: analytic/Fourier identifiability instead of full-wave simulation
- Phase: 0.1 (scoping).
- Motivation: user decision to restart Phase II on a math model with
  Fourier analysis for unique inverse solving, rather than further
  compute-heavy jWave work. Prompted by an external consultation
  (pasted transcript) on what a source + 180° receiver arc can and
  cannot recover: travel time / attenuation give only aggregate
  constraints, 1D transmission cannot distinguish a stack from its
  reversal (reciprocity), and reflection + more illumination angles are
  what break ambiguities. Those claims are treated as hypotheses to
  verify, not results.
- Context read before starting: `jwave_test` PROJECT CLOSURE,
  `jwave_ring_tomography` MANIFEST + runs -06/-07/-08/-28/-31/-43/-44,
  `jwave_cylindrical_tomography` MANIFEST + runs -14 to -19 (that
  project is still open, awaiting a Colab GPU memory test from run -19;
  not touched here).
- Reused: cited tissue properties cloned unchanged into `src/tissue.py`.
  The consultation's fat/bone/air examples are NOT added (no cited
  source in this repo).
- Housekeeping: found `jwave_cylindrical_tomography/single_ring_36_
  multistatic_scan_patient023.npz` (12.7 MB, patient-derived) at that
  project's root, NOT gitignored; added `jwave_cylindrical_tomography/
  *.npz` to `.gitignore`. Also pre-emptively ignored
  `fourier_inverse/results/*patient*.npz`.
- Gate passed? N/A.
- Next action: Phase 0.2 sanity check.

### Run 2026-10-24-01 — Phase 0.2: 1D layer-reversal transmission ambiguity CONFIRMED (after catching an attenuation-sign bug the first gate missed)
- Phase: 0.2. Seed: 42 (`src/seed.py`). Script:
  `src/phase0_layer_reversal_check.py`. CPU, ~seconds per run.
- Config: exact normal-incidence transfer matrix, all multiples;
  f = 0.05–5 MHz (4000 pts); 1000 random stacks of 2–8 layers drawn
  from {BLOOD, MYOCARDIUM, CHEST_WALL_PROXY, WATER}, thicknesses
  U(1, 30) mm, embedded in WATER. Attenuation = cited 1 MHz value ×
  (f / 1 MHz); the power-law exponent 1 is an ASSUMPTION for this check
  only.
- Run history (all kept, nothing discarded):
  1. First run: gate printed FAIL, because min rel |R_A − R_B| = 0. Cause:
     the stack generator, not the physics. Adjacent identical tissues
     form one layer, and some stacks are palindromes after merging, so
     they are their own reversal. Fix: merge adjacent identical layers
     and exclude palindromes (86/1000, reported in the output). The pass
     criterion did not change.
  2. Second run: gate printed PASS, but the figure showed |T| rising
     from 1 to 4.5 with frequency. A passive lossy stack cannot do that.
     Cause: the attenuation sign did not match the transfer-matrix time
     convention (k = w/c + iα with an exp(+iwt) matrix), so loss became
     gain. The gate missed it because checks (a)–(c) were either
     lossless, where the sign has no effect, or invariant to the sign
     (the reversal identity is algebraic). Fix: k = w/c − iα. Added
     check (d), passivity |R|²+|T|² ≤ 1, and check (e), a slab embedded
     in its own medium must give |T| = exp(−αd) exactly. Either check
     would have caught the bug.
  3. Third run, final: all checks pass.
- Result (final run):
  - (a) single-layer |T|² vs closed form: max err 8.9e-16
  - (b) lossless energy conservation: max err 1.7e-15
  - (c) reversal, 914 non-palindromic LOSSY stacks: max rel
    |T_A − T_B| = 7.0e-16 (machine precision); rel |R_A − R_B| min
    0.077, median 0.220, max 0.474
  - (d) lossy passivity: max |R|²+|T|² = 0.997
  - (e) self-embedded slab vs exp(−αd): max err 4.4e-16
  - Cardiac-like example (wall 12 | myo 10 | blood 40 | myo 6 mm):
    T identical to 6e-16, R differs by 19%.
  - Figure: `results/figures/phase0_layer_reversal.png`.
- Physical sanity checked? by whom?: Claude. Inspected the figure
  instead of trusting the PASS line, which is how the sign bug was
  found. Gate 0 is a numerical-correctness gate only. It is not Gate 2
  (physics signoff), which still needs a collaborator.
- Gate passed? Gate 0 (phase 0.2): PASS.
- Observations: the consultation's headline claim holds exactly. With
  the same medium on both sides, through-transmission cannot
  distinguish a lossy layered stack from its mirror image, even with
  infinite bandwidth and all multiples. This is non-identifiability,
  not noise. One reflection channel breaks the symmetry in every
  non-palindromic case tested. At high frequency the R difference is
  dominated by the first interface, because deeper echoes are
  attenuated. This is a first hint that bandwidth helps reflection
  identifiability only down to an attenuation-limited depth (Phase 1.3).
- Next action: Phase 1.1. Compute the Jacobian rank and singular
  spectrum of T(f) and of [T(f), R(f)] with respect to per-layer
  (d, c, ρ, α). Find the continuous null directions beyond the discrete
  reversal symmetry. Expected from theory: normal-incidence reflection
  fixes Z(τ) (impedance vs. one-way travel time) but not c and ρ
  separately, and not depth.

### Run 2026-10-24-02 — Phase 3.3: Born is unusable for the whole heart; Rytov holds for the inner wall everywhere tested
- Phase: 3.3. User chose to run Phase 3 before Phases 1–2. The protocol
  was amended so that Phases 1/2/3 are independent tracks gated only by
  Gate 0. Gate 1 is not skipped; it gates Phase 1's own conclusions.
  Seed 42. Script: `src/phase3_born_rytov_validity.py`. CPU ~10 s.
- Model: uniform fluid cylinder (radius a) in a uniform background,
  plane-wave incidence, 2D. Exact solution = Bessel series (no
  simulation). Born far field derived for variable density by
  linearizing ∇·(ρ⁻¹∇p) + ω²κp = 0. With γ = κ/κ₀ − 1, β = ρ₀/ρ − 1 and
  integration by parts in the far field:
  F_B(θ) = (i/4) k² [γ − β cosθ] Ŝ(K), K = 2k sin(θ/2). Sanity of the
  form: forward (θ=0) weight ≈ −2δc/c, so density drops out, as travel
  time requires. Backscatter (θ=π) ≈ −2δZ/Z, the reflection
  coefficient. Rytov evaluated in the near field (r = 1.5a),
  density-matched case.
- Checks: (a) Bessel-series Born vs closed-form Born: 2.6e-15.
  (b) exact → Born as contrast → 0: errors 3.7e-1 / 3.7e-2 / 3.7e-3 for
  δc/c = 1e-2 / 1e-3 / 1e-4, ratios 9.95 / 9.99 per decade, so first
  order as theory says. The first version used an absolute bound
  (err < 1e-3 at 1e-4) that was arbitrary and printed FAIL. Replaced it
  with the convergence-order test; no physics changed. PASS.
- Result (relative scattered-field error; Born ≈ 0.45 × accumulated
  phase Δφ = 2aω|1/c₀ − 1/c₁|):
  - Myocardium in water, a = 30–45 mm (whole heart): Δφ = 7–11 rad at
    0.8 MHz and 22–33 rad at 2.5 MHz. Born error 5–19 (500–1900%),
    Rytov 0.7–1.9. **Both linearizations fail for the whole heart.**
    Born is usable (< 0.3) only for Δφ ≲ 0.6 rad. Rytov is usable to
    a ≈ 20 mm at 0.8 MHz and a ≈ 5 mm at 2.5 MHz.
  - Blood in myocardium (inner wall): Born usable to a ≈ 20 mm at
    0.8 MHz and ≈ 5 mm at 2.5 MHz (LV-cavity scale ~20 mm is borderline
    to failing at 2.5 MHz). **Rytov error ≤ 0.08 everywhere tested**,
    up to a = 45 mm at 2.5 MHz.
  - Density makes little difference to the Born error (true densities
    vs. c-only far-field columns nearly identical).
  - Figure: `results/figures/phase3_born_rytov_validity.png`.
- Gate passed? N/A (Phase 3 is gated by Gate 2 at its end).
- Observations: any Fourier/linear picture of USCT must linearize
  about a background that already contains the outer heart wall
  (distorted Born / iterative linearization), not about water. Inside
  that, the inner-wall problem is genuinely close to linear (Rytov). So
  the k-space maps of run -03 are exact statements about the Fréchet
  derivative around a known background, and an idealization for the
  whole-object problem. A consequence for the earlier jWave work: the
  outer wall's refraction distorts the illumination of the inner wall,
  so plane-wave coverage maps overstate how cleanly the inner wall is
  illuminated.
- Next action: run -03, the k-space coverage maps.

### Run 2026-10-24-03 — Phase 3.1/3.2/3.2b: k-space coverage maps; time-of-flight sees ~11% of a wall's information, full-wave transmission ~97%, the pasted single-source design ~22%
- Phase: 3.1, 3.2, 3.2b. Seed 42. Script: `src/phase3_kspace_coverage.py`.
  CPU ~2 min.
- Config: Fourier diffraction theorem about water (c₀ = 1520), plane
  waves, far-field receivers. K = k(s − s₀). Single frequency 2.5 MHz,
  or broadband 0.5–2.5 MHz (120 samples). FFT grid 100 mm / 1024.
  Synthetic LV-scale annulus (outer radius 30 mm, inner 20 mm; not
  patient data). Local wall bulges 0.5 mm thick, ±10° wide, at 8
  normal directions (90° = anterior). Designs: (1) pasted single source
  + 180° far arc; (2) same + a ±5° source-side receiver; (3) anterior 60°
  arc pulse-echo (jwave_test-like); (4) full ring, time-of-flight only
  (forward cone |θ| ≤ √(λ/L), L = 120 mm, the Fresnel limit, an
  approximation); (5) full ring, transmission |θ| ≤ 90°; (6) full ring,
  all pairs.
- Bugs caught before reporting (all kept in the history):
  1. The single-source designs first showed 0.00% coverage even with
     broadband. A single incident direction is measure-zero, so the
     inverse map from grid points never landed on it exactly. Fixed by
     rasterizing forward from (k, receiver angle) → K.
  2. Hermitian symmetry was missing. A lossless object (c, ρ contrast
     only) is real, so measuring K also gives −K. This was spotted
     because the anterior arc showed identical 0.20 at 90° and 270°.
     Fixed with mask ∪ (−mask), and the flip indexing on the even FFT
     grid was verified directly. Attenuation would make the object
     complex and break this; it is not modeled.
- Checks: inverse map K → (α, β) → K error 6e-16. Full ring all-pairs
  coverage 100.00%. Full-ring transmission 50.00% (analytic: disc
  |K| ≤ √2·k, i.e. half the area). PASS.
- Result, coverage of the |K| ≤ 2k_max disc (single freq / broadband):
  pasted design 0.3% (a curve, area ≈ 0) / 9.0%; + source-side receiver
  0.3% / 11.7%; anterior arc 2.9% / 26.1%; time-of-flight 0.13% /
  0.13%; transmission 50% / 50%; all pairs 100% / 100%.
- Result, fraction of a wall bulge's spectral energy measured
  (broadband, mean over normals; min in brackets):
  - pasted design 0.22 (0.02, for normals along the beam axis)
  - + source-side receiver 0.25 (0.16)
  - anterior arc 0.11 (0.006, for lateral normals)
  - time-of-flight 0.11 (0.105)
  - full-wave transmission 0.97 (0.97)
  - all pairs 1.00
  Inner and outer walls behave almost identically here. Geometrically,
  coverage does not distinguish them.
- Result, contrast (3.2): weight |γ − β cosθ| is, for the outer wall
  (water → myocardium), 0.066 forward and 0.190 backscatter; for the
  inner wall (myocardium → blood), 0.0088 forward and 0.0088
  backscatter. The equality is because the cited densities are
  identical: Mast 2000 gives 1060 for both. Inner/outer amplitude
  ratio: 0.134 forward, 0.046 backscatter.
- Result, 3.2b (density vs. compressibility, all pairs, 0.5–2.5 MHz,
  noise-free conditioning): not separable at a single frequency (each K
  carries one combination). With the band: condition number 73 at
  ~6 mm feature scale, 3.0 at ~1.2 mm, 2.4 at ~0.6 mm. Not separable
  for |K| > 2k_min, where only one frequency reaches K at a given θ.
  Coarse structure only constrains the forward-scatter combination,
  i.e. sound speed.
- Discrete angular sampling (3.1b, analytic, not a run): unique
  recovery of an arbitrary object in a disc of radius a needs about
  2ka + 1 sources and receivers. Ring project (2.5 MHz, a ≈ 8 mm):
  about 166, used 36, so 4.6× undersampled. Cylindrical project
  (0.8 MHz, a ≈ 45 mm): about 299; 12/ring is 25× under and 36/ring is
  8.3× under. Direction consistent with cylindrical run -18 (12 → 36
  elements: 15.3 → 5.8 mm). Recorded as a Phase 4 prediction, not an
  explanation: localizing a boundary of known topology needs less than
  recovering an arbitrary object.
- Figures: `results/figures/phase3_kspace_coverage.png`,
  `phase3_wall_information.png`, `phase3_density_separation.png`.
- Physical sanity checked? by whom?: Claude, via analytic
  cross-checks and figure inspection; both bugs were found by looking
  at outputs that disagreed with physical expectation. Gate 2 still
  needs a collaborator.
- Observations / interpretation:
  1. **The inner wall's transmission invisibility (`jwave/` scout,
     `jwave_ring_tomography` run -07) is a time-of-flight limitation,
     not a transmission-geometry limitation.** Time of flight keeps
     only a Fresnel-sized neighbourhood of K = 0 (~11% of a wall
     bulge's information). Full-wave forward scatter (phase and
     amplitude at all forward receivers) covers ~97%. On top of that,
     the inner wall's forward contrast is 0.13× the outer wall's. Both
     effects compound for time of flight.
  2. Reflection's advantage for the inner wall is not contrast: the
     inner/outer backscatter ratio, 0.046, is worse than forward,
     0.134. Its advantage is the high-|K| band, √2k < |K| ≤ 2k, that
     only reflection reaches, together with time separation of echoes
     (not captured in this frequency-domain picture).
  3. The pasted design (one source + 180° arc), even with broadband,
     measures ~9% of k-space and is blind (2%) to boundaries whose
     normal lies along the beam. Adding a source-side receiver helps
     modestly (min 2% → 16%). Multiple illumination angles are what
     matter, which agrees with the consultation's own conclusion.
  4. Anterior-arc pulse-echo (the jwave_test geometry) measures only
     boundaries whose normal points along the arc axis (0.6% for
     lateral normals). This is a k-space restatement of jwave_test's
     closure: lateral wall segments sit in the null space and cannot be
     recovered by any algorithm or learned prior without hallucination.
- Next action: 3.4, an SVD of the discretized linearized operator for
  the discrete-element designs (36-element ring etc.): singular-value
  decay and explicit null-space images, adding conditioning and
  aliasing on top of the continuous coverage picture. Then Gate 2
  review of runs -02/-03's approximations.

### Run 2026-10-24-04 — Objective refined to BLIND uniqueness; visual of what each design measures; deterministic "unique but unstable" demo; CORRECTION to run -03's null-space wording
- Phase: 3. Seed 42. Scripts: `src/phase3_blind_object_views.py`,
  `src/phase3_continuation_instability.py`. CPU ~2 min total.
- User direction (2026-10-24): the object must be fully blind. No tissue
  labels and no assumed property values; recover regions (unknown
  count, positions, values) of an arbitrary layered object. Priority is
  uniqueness. Stability is deferred, and the user proposes a later
  gradient-descent or ML stabilizer. The user asked why instability
  exists in a deterministic Fourier problem, and asked for the hidden
  object's outline to be traced on plots.
- Blind-object views: random object with 5 painted regions (overlap and
  nesting), synthetic contrasts in ±0.08, not tissue values. Each design
  shows the linear minimum-norm image (inverse FFT of the measured
  spectrum only), with the true outlines in black. Relative part not
  directly measured: pasted design 0.78; + source-side receiver 0.78;
  anterior arc 1.00 (only top/bottom edge fragments); time-of-flight
  0.25 (blurred blobs, no edges); full-wave transmission 0.06; all
  pairs 0.04 (the remainder is edge content beyond |K| = 2k_max, the
  resolution limit). Figure:
  `results/figures/phase3_blind_object_views.png`.
- Instability demo (1D, deterministic): blind layered object on a
  known 20 mm support, 200 unknowns. Exact Fourier data on |K| in
  [0.03, 0.30] × Nyquist only. The singular values are all positive in
  exact arithmetic, because compact support makes Ô entire, so any
  interval determines it: UNIQUE. They hold flat for the ~54 Slepian
  directions, then fall exponentially and reach the float64 SVD floor
  at index ~89. (The flat shelf at ~1e-15 is rounding in the SVD, not
  the true values; the first printout counted it as "195 visible" and
  was corrected.) Full inversion error: 1.8 with float64-exact data,
  1.1e3 with a 1e-12 perturbation, 1.1e9 with 1e-6. Keeping only stable
  directions (σ > 1e-3 σ_max) at 1e-6 error: 0.42, and the **layer
  edges land at the correct positions**; the error is the smooth
  offset from the missing low-K band. Figure:
  `results/figures/phase3_continuation_instability.png`.
- **CORRECTION to run -03, interpretations 3/4 and the MANIFEST
  wording.** I called unmeasured k-space regions "the null space"
  ("cannot be recovered by any algorithm"). For a compactly supported
  object in the linear model that is wrong in exact mathematics. Ô is
  analytic, so any covered set with interior determines it everywhere.
  Broadband anterior-arc, time-of-flight and the pasted design are
  therefore all UNIQUE in principle. The correct statement: the
  unmeasured part can only be reached by analytic continuation, which
  is exponentially unstable (this run's demo). It is unrecoverable at
  any finite data precision, not non-identifiable. Genuine exact
  non-uniqueness remains only where coverage has no interior, e.g. a
  single source at a single frequency (an Ewald curve: there are
  nonradiating objects whose spectrum vanishes on it), and the 1D
  transmission-only reversal (run -01). The practical conclusion for
  jwave_test (lateral walls unrecoverable from anterior-arc data
  without a prior supplying them) stands; the reason is extreme
  instability, not a strict null space.
- Observations:
  1. For a blind compactly supported object, uniqueness is cheap. It
     holds for almost every design, so on its own it cannot
     distinguish good designs from hopeless ones. What separates them
     is how many directions are stable. This matters for the plan to
     "defer stability to a post-solve stabilizer": a stabilizer cannot
     restore information carried at 1e-20. It can only replace it with
     its own prior, which is the hallucination risk jwave_test's
     closure warned about.
  2. The user's region framing is what makes uniqueness meaningful. An
     object made of finitely many constant regions has finitely many
     parameters (boundaries + values). The magenta curve shows that
     the stable directions already place the jumps correctly; a
     3-layer object has ~6 parameters against ~54 stable directions.
     Uniqueness within this finite-parameter class is the right
     target, and it is plausibly stable without any learned prior.
- Gate passed? N/A.
- Next action: proposed to the user. Redefine Phase 3+ as blind
  finite-region identifiability. Test (a) whether two different region
  configurations can produce identical stable-band data, via an
  adversarial search over region parameters, and (b) whether the
  region count is identifiable.

### Run 2026-10-24-05 — Blind finite-region uniqueness: adversarial search STOPPED (optimizer-limited); Jacobian test shows every design is LOCALLY UNIQUE for a 4-region, 32-parameter object class
- Phase: 3. Seed 42. Scripts: `src/phase3_adversarial_uniqueness.py`
  (stopped), `src/phase3_local_uniqueness_jacobian.py`.
- Object class: n star-shaped regions painted in order (overlap
  allowed). 8 unknowns each: center, r0, 2 radial harmonics, value.
  Sigmoid edges (0.25 mm) for differentiability. Hidden A = 4 random
  regions, all visibly present. Grid 128², 64 mm; band 0.2–0.7 MHz.
  Data = exact Fourier values on each design's coverage set
  (linearized model). The same renderer generates the data, i.e. the
  inverse crime is ON PURPOSE, to test the map itself.
- Adversarial search (maximize the image difference subject to a
  small data misfit, μ sweep, L-BFGS-B with torch gradients): timing
  test ~38 s per fit, and a random-start blind fit stuck at 35% misfit
  (a local minimum: an optimization difficulty, not evidence about
  uniqueness). Launched with a ~35–40 min estimate; stopped by the
  user's decision after ~10 min with 2/7 designs done. Partial output
  (`results/phase3_adversarial_run.txt`): no fit got below misfit 1e-3,
  so ω(ε) for ε → 0 was unmeasurable. The test was optimizer-limited,
  and it could not have answered the uniqueness question as configured.
  My error: optimizer reachability should have been checked before
  launching the sweep.
- Replacement: local uniqueness via the Jacobian. Forward-mode autodiff
  of the renderer gives J_img (16384 × 32). The data are linear in the
  image, so J_d = FFT of J_img columns. (Reverse-mode over ~22k outputs
  ran out of memory on the first attempt.) Parameter directions that
  change no pixel are removed. The singular values of
  G = J_d V S⁻¹ = relative data change per unit relative image change,
  which is independent of parameterization. Check: autograd vs central
  finite difference, rel err 1.4e-10, PASS. Runtime 7 s.
- Result (σ_min / σ_max of G, 32 image-changing directions; trivial
  directions 0 for this A):
  - pasted design, broadband: 0.18 / 1.33
  - pasted design, single frequency: 0.042 / 1.37
  - + source-side receiver: 0.32 / 1.33
  - anterior 60° arc: 0.17 / 9.9
  - full ring, time-of-flight: 0.0018 / 1.09
  - full ring, transmission: 0.97 / 1.00
  - full ring, all pairs: 0.996 / 1.00
  No direction below 1e-3 in any design. **Every design is locally
  unique for this object class.** That includes the single-frequency
  single-source design, which is exactly non-unique for general objects
  (run -04). The least-visible image change is, in every design, a
  boundary shift where that design's coverage is weakest: beam-axis
  normal for the pasted design, lateral edge for the anterior arc, a
  fine overlap edge for time-of-flight. This matches run -03.
  Figures: `results/figures/phase3_local_jacobian_spectra.png`,
  `phase3_local_jacobian_worst_change.png`.
  Note: σ is normalized by each design's OWN measured data norm, so it
  measures sensitivity relative to what that design records. A design
  measuring little total energy can still score well.
- Observations:
  1. This is the first positive result for the user's blind-region
     goal. Restricting to finitely many regions turns
     infinite-dimensional non-uniqueness and instability into a
     finite, well-conditioned map, even for weak designs. The reason:
     32 unknowns against 90 (time-of-flight) to 21,818 (all pairs)
     real data values.
  2. Not yet shown: (a) GLOBAL uniqueness, since a different
     configuration far from A could still fit; (b) dependence on
     object complexity. With only 2 harmonics these shapes are smooth.
     As harmonics and regions grow toward "arbitrary", the unknowns
     approach the stable data count, and σ_min must collapse toward
     run -04's continuum behavior. Where that happens, per design, is
     the real answer to how complex a blind object each design can
     uniquely resolve. (c) Region-count identifiability, which the
     stopped run never reached. (d) Validity outside the linearized
     model (run -02).
- Gate passed? N/A.
- Next action (proposed): complexity sweep. σ_min vs. harmonics
  (2 → 16) and region count (2 → 8) per design, averaged over several
  random hidden objects, to find each design's complexity limit.
  Region count via the Jacobian of an (n+1)-region model at the
  n-region truth.

### Run 2026-10-24-06 — Complexity sweep with SHARP convex + concave corners: local uniqueness follows a counting rule (unknowns vs. independent data); the full ring stays well-conditioned up to 280 unknowns
- Phase: 3. Seed 42. Script: `src/phase3_complexity_sweep.py`. Log:
  `results/phase3_complexity_run.txt`. Runtime 124 s.
- User asked for more complex geometry with "convex angles", read as
  sharp corners. Object class: n star-shaped POLYGONS with straight
  edges and V vertices. Per vertex, an unknown radius and angular
  jitter; per region, unknown center and value, so 2V + 3 unknowns per
  region. Hidden objects use vertex radii r0·U(0.5, 1), giving both
  convex and concave (re-entrant) corners: 2 to 18 concave corners in
  the gallery objects. Sweep: n ∈ {2, 4, 6, 8} × V ∈ {5, 8, 12, 16},
  i.e. 26–280 unknowns, 3 random hidden objects each, median σ_min.
  Same parameterization-free measure as run -05.
- Check: polygon forward-mode Jacobian vs finite difference 1.6e-9, PASS.
- Result (median σ_min, from 26 → 280 unknowns):
  - full ring all pairs: 0.94 → 0.55
  - full ring transmission: 0.81 → 0.17–0.25
  - anterior arc: 0.55 → 0.14
  - pasted + source-side receiver: 0.41 → 0.07
  - pasted design broadband: 0.35 → 0.017–0.03
  - pasted design single frequency: 0.11 → 0.0013–0.0023 up to 162
    unknowns, then **0** (1e-16) at 210+
  - time-of-flight: 1.4e-4 / 1.3e-5 at 26 / 38 unknowns, then **0** at
    52+
- **Counting rule confirmed.** Independent real data = number of
  Hermitian-closed K points (K and −K are redundant for a real object):
  time-of-flight 45, pasted single-frequency 183, pasted broadband
  1059, + receiver 1409, anterior 2568, transmission 5457, all pairs
  10909. Local uniqueness breaks exactly when unknowns exceed this
  count: time-of-flight between 38 and 52 (limit 45), single frequency
  between 162 and 210 (limit 183). Below the count, every design stays
  locally unique; the decline in σ_min with complexity is a stability
  trend, not a uniqueness loss. Collapsed cases show the exact null
  change in the gallery: small scattered edge patches. The full ring
  barely degrades (0.94 → 0.55) at 280 unknowns, about 2.6% of its data
  count.
- Figures: `results/figures/phase3_complexity_sigma_min.png` (σ_min vs
  unknowns with data-count lines), `phase3_complexity_gallery.png`
  (hidden objects with corners and traced outlines, and the
  least-visible change per design).
- Efficiency / process errors (mine, kept on record):
  1. Plotted the time-of-flight data count as 90 (raw reals) instead
     of 45 (independent, Hermitian). Caught when the collapse landed at
     45; corrected.
  2. The gallery left collapsed panels blank, because the null vector
     was dropped when unknowns > data. Fixed with `full_matrices`, but
     applied unconditionally. On the all-pairs design that builds a
     21818 × 21818 U (3.8 GB) and made the sweep ~200× slower (216 s for
     the first row). Fixed: full_matrices only when rows < columns.
  3. A run piped through `tail` hid progress. When the user
     backgrounded it, I checked for the process by the wrong name, so a
     stale slow run kept going in parallel and contended for CPU. Final
     figures verified to come from the corrected run (timestamps match
     its log). From now on: unbuffered output straight to a log file,
     and time one case before any sweep.
- Gate passed? N/A.
- Observations: for blind piecewise-constant objects with sharp
  corners, local uniqueness is essentially a counting question, and
  the full ring has orders of magnitude of headroom (10,909 independent
  data). Still open: global uniqueness (a configuration far from the
  truth), region-count identifiability, and validity of the
  linearization for large high-contrast objects (run -02).
- Next action: user decision. Options: region-count test, a cheap
  global check from many random starts on the full-ring design, or
  stress tests (nested thin layers, near-equal contrasts).

### Run 2026-10-24-07 — Blind solver step 1: regions FORM correctly from full-ring data alone (visual check); the pasted single-source and anterior-arc designs give artifacts or nothing
- Phase: 3. Seed 42. Script: `src/phase3_blind_region_formation.py`.
  CPU ~8 s.
- Decisions (user): further "find a look-alike" searches stopped as an
  endless chase. Global uniqueness rests on theory instead: Paley–Wiener
  analyticity for the linearized model with full-disc coverage, and
  2D fixed-frequency results (Nachman/Bukhgeim) for the nonlinear one;
  those citations are still to be verified. Step 1 is judged by visual
  inspection only, with no quantification and no labels.
- Pipeline (sees only data: noise-free linearized spectrum on the
  design's measured set):
  1. direct image = inverse FFT of the measured spectrum, no starting
     guess;
  2. 2× 5×5 median filter against ringing;
  3. edges where the Sobel gradient > 8% of max;
  4. regions = connected areas enclosed by edges; fragments < 40 px and
     edge pixels go to the nearest region.
  No labels, no region count, no values given. Hidden objects: 3/5/8
  polygon regions with 8/12 vertices and 5–11 concave corners.
- Method history (kept): value-histogram clustering was tried first and
  failed twice. First, water (~85% of pixels) swamped the region peaks
  and merged touching regions. Then, with water treated as a known
  level 0, blurred edge pixels created spurious peaks (12–15 "levels").
  Replaced by the edge-based definition above, which needs neither value
  levels nor the water assumption.
- Result (visual; regions found include water):
  - Full ring, all pairs and transmission: identical. Green boundaries
    form along every true outline, including internal boundaries
    between touching regions and concave corners. Found 4/5/8 vs. true
    4/6/9 (incl. water). Each miss is a thin sliver left where one
    painted region is mostly occluded by another, at or below
    resolution / MIN_PIX.
  - Pasted design (1 source + 180° arc, broadband): boundaries follow
    the horizontal streak artifacts, not the object (5/16/8 "regions").
  - Anterior 60° arc: almost nothing forms (1/4/2).
  - Figure: `results/figures/phase3_blind_region_formation.png`.
- Physical sanity checked? by whom?: Claude, visually; thresholds
  (8%, 40 px) were set once and not tuned per object.
- Gate passed? N/A.
- Observations: the user's blind-region goal works end to end in the
  linearized, noise-free setting with a full ring. Data alone yields the
  right regions, with no prior and no starting guess, and the direct
  image is good enough that a later refinement stage (local fit,
  well-posed per runs -05/-06) has a correct start. Untested: noise,
  thin layers below resolution, near-equal neighbouring contrasts, and
  the nonlinear regime (run -02), which is the real obstacle for
  whole-heart scale.
- Next action: user decision. Candidates: add noise; refinement stage;
  move to nonlinear (distorted-Born) physics; Gate 2 review packet.

### Run 2026-10-24-08 — Compute scoping for nonlinear ("real physics") inversion: measured 2D Helmholtz factorization cost on this laptop
- Phase: 3 → nonlinear scoping. No seed (deterministic timing). Script:
  scratchpad `helmholtz_timing.py` (throwaway, not in repo): 2D 5-point
  Helmholtz, 10 points/wavelength, 20-cell damping layer, SciPy SuperLU
  (COLAMD ordering).
- Machine: i7-1260P, 16 logical cores, 15.6 GB RAM, CPU only.
- Measured:
  - 340² (0.12M unknowns): factor 19.6 s, per-source solve 0.06 s, LU 0.19 GB
  - 640² (0.41M): factor 35.2 s, solve 0.17 s, LU 0.84 GB
  - 940² (0.88M): factor 74.2 s, solve 0.46 s, LU 1.99 GB
- Size mapping (grid ≈ 10 × domain/λ): our scaled test (64 mm,
  0.7 MHz) ≈ 300²; 2D heart slice (130 mm) at 0.8 MHz ≈ 680², at
  1.2 MHz ≈ 1030², at 2.5 MHz ≈ 2150² (~4.6M unknowns; extrapolated LU
  ~12–15 GB, 10–15 min per factorization, so RAM-bound on this laptop).
- Estimate for one blind nonlinear reconstruction. Assumptions: 128
  sources; per step = 1 factorization + forward and adjoint solves for
  every source; ~10 frequencies × ~10 iterations × 1.5 for line search
  ≈ 150 steps. Results:
  - scaled test ~1.5 h
  - 2D heart ≤ 0.8 MHz ~3–4 h
  - 2D heart ≤ 1.2 MHz ~8 h
  - 2D heart 2.5 MHz: not feasible here (RAM)
  - 3D: not feasible on CPU. The time-domain GPU route
    (jwave_cylindrical run -08: ~19 s per forward sim) gives 96 sources
    × (forward + adjoint) ≈ 1 h per iteration, i.e. tens of GPU-hours.
- Caveats: estimates, not runs. SuperLU with COLAMD is not the best
  solver for 2D Helmholtz; a nested-dissection solver (e.g. MKL PARDISO)
  would likely be several times faster, which is untested and would add
  a dependency. Iteration counts are assumed, not measured.
- Next action: user decision on whether and at what scale to start.

### Run 2026-10-24-09 — Colab GPU (T4) timing of jWave Helmholtz solvers: GMRES with autodiff is the WRONG route (slower than the laptop); Born series plus a manual adjoint looks like ~1–2 GPU-hours for a 2D heart, but solver agreement at large grids is unverified
- Phase: 3 → nonlinear scoping. User chose the GPU route. Ran via the
  official `googlecolab/colab-mcp` connector, registered in a project
  `.mcp.json` with uv installed via pip. The `claude` CLI is not on
  PATH in the VS Code extension, so `claude mcp add` could not be used.
  Notebook cells driven directly by Claude.
- Protocol change (user decision): jWave/GPU now used for the
  nonlinear step, contrary to the original "no full-wave in Phases 0–3"
  non-goal. To be amended in the protocol.
- Environment: Tesla T4 15 GB, Python 3.13, jwave 0.2.1, jax/jaxlib
  0.4.38 + cuda12 (pins from jwave_cylindrical). Colab's preinstalled
  jax-cuda13 plugin clashed (PJRT ALREADY_EXISTS errors) and was
  uninstalled before a kernel restart.
- Script: `fourier_inverse/src/phase3_gpu_helmholtz_timing.py` (copied
  into the notebook). Heart-like density-matched phantom from cited
  speeds (water 1520 / myo 1576 / blood 1584). Synthetic only. 4
  points/wavelength, 130 mm domain, PML 16.
- Solver-agreement diagnostic (96² grid, 0.6 MHz), run before timing:
  both solvers converged (GMRES tol 1e-3 → 1e-6 changes the result by
  2.5e-4; Born series by 6e-7). Fields correlation 0.9992; best scalar
  −1.068. So the earlier "2× disagreement" (CPU check, run in the
  previous session) is a SIGN convention. The residual after scaling,
  4%, is attributed to the two solvers' different PML placement
  (inside vs outside the domain).
- Timing (T4):
  - 128²: GMRES 0.18 s, Born series 0.15 s, GMRES autodiff gradient
    (1 src) 3.3 s, 8-src vmap 0.08 s/src
  - 288² (heart 0.8 MHz): GMRES 0.59 s, Born 0.21 s, gradient 6.7 s,
    vmap 0.90 s/src
  - 416² (1.2 MHz): GMRES 1.47 s, Born 0.12 s, gradient 12.2 s, vmap
    2.44 s/src
  - 864² (2.5 MHz): GMRES 6.48 s, Born 0.99 s, gradient 53.5 s, vmap
    10.1 s/src
  - No out-of-memory failure up to 864². One harmless XLA
    constant-folding slowness warning at 864² during compilation.
- Interpretation:
  1. Autodiff through GMRES costs 8–11× a forward solve, and vmap
     batching is slower per source than single solves (vmapped GMRES
     runs until the slowest source converges). The script's printed
     summary (7.6 / 54 / 108 / 446 GPU-hours) is for this route and is
     WORSE than the laptop direct-LU estimate (run -08), which reuses
     one factorization across all sources.
  2. Born series is fastest (up to 12× faster than GMRES at 416²). It is
     not reverse-differentiable (lax.while_loop), but the adjoint-state
     gradient needs only one extra solve with the receivers as sources
     (reciprocity). Estimate at ~2 Born solves per source per step, 128
     sources, 150 steps: 0.8 MHz ≈ 2.2 GPU-h, 1.2 MHz ≈ 1.3 GPU-h,
     2.5 MHz ≈ 10.6 GPU-h. Estimates, not runs. Born-series iteration
     counts depend on contrast; line search is not included.
  3. **Unresolved:** GMRES vs Born-series relative difference grows
     with grid size: 2.11 → 2.34 → 2.78 → 4.62. At 96² it was almost
     entirely the sign (true mismatch ~4%). At larger grids one solver
     is likely not converging, probably GMRES hitting maxiter=1000 with
     restart=10. This must be resolved before trusting either solver at
     heart scale.
  4. Born series ignores density and attenuation (jWave docstring), so
     a Born-series inversion is sound-speed-only.
- Gate passed? N/A.
- Next action: (a) at 416², compare the Born series against tightly
  converged GMRES (larger restart/maxiter) and against itself at
  tighter tolerance, after removing the sign convention; (b) if the
  Born series is confirmed accurate, implement and verify the
  reciprocity adjoint gradient against a finite difference on a small
  grid.

### Run 2026-10-24-10 — Solver accuracy settled against an EXACT solution: the Born series is accurate (2.4–2.7%), jWave GMRES with default PML is NOT (37–39%)
- Phase: 3 → nonlinear scoping. Colab T4 via colab-mcp, same session
  and environment as run -09.
- Step 1 (416², 1.2 MHz, heart phantom; Born series at tight tol as
  reference): GMRES default and GMRES tight (tol 1e-6, restart 50,
  maxiter 5000; took 418 s) give the SAME field. Born series 1e-3 and
  1e-6 also agree with each other (mismatch 0.001). Both are converged,
  yet GMRES vs Born differ by 41% after the best complex scaling
  (correlation 0.91). So the growing gap in run -09 was not
  non-convergence: the two solve different discrete problems.
- Step 2 (ground truth): homogeneous water, point source, 416²,
  1.2 MHz. Exact 2D field = H0⁽¹⁾(kr) up to one complex scalar.
  Interior only (40 cells from the edge, r > 8 cells from the source).
  Shape mismatch after the best scalar:
  - source 4 cells from the PML: GMRES 0.387, Born series 0.027
  - source 60 cells inside: GMRES 0.365, Born series 0.024
- Verdict: the convergent Born series reproduces the exact solution to
  ~2.5%. GMRES `helmholtz_solver` with its default internal 16-cell PML
  at 4 points/wavelength is ~37% wrong (likely PML reflections; the
  Born series uses a large weak PML outside the domain), independent
  of source placement. Run -09's GMRES timings and the
  autodiff-through-GMRES route are therefore moot, not just slow. The
  "sign convention" reading of run -09's 96² diagnostic was only
  partly right: the two solvers also differ in accuracy.
- Consequence: Born series + reciprocity adjoint (run -09, item 2) is
  both the fastest AND the accurate route. Estimated ~1–2 T4-hours for
  a 2D heart-slice reconstruction (sound speed only, since the Born
  series ignores density and attenuation). Still to verify: the adjoint
  gradient against a finite difference.
- Protocol amended (user decision): jWave/GPU is permitted for the
  nonlinear step.
- Gate passed? N/A.
- Next action: implement the Born-series forward + reciprocity-adjoint
  gradient, verify against a finite difference on a small grid, then
  a small-object blind nonlinear reconstruction on Colab.

### Run 2026-10-24-11 — First BLIND NONLINEAR reconstruction (real full-wave physics): the object's shape, corners and a slow region are recovered from plain water; the weak internal boundaries appear in the image but not in the automatic segmentation
- Phase: 3 → nonlinear. Colab T4 via colab-mcp. Seed 42 (numpy
  default_rng for the hidden object and for the test directions).
  Code: `src/phase3_fwi_born.py` (copied into the notebook as
  `fwi_born.py`), plus notebook cells.
- Step A, adjoint gradient check: J = ½Σ|P_s(x_r) − D|². Gradient by
  reciprocity: 1 forward + 1 adjoint Born-series solve per source,
  dk²/dc = −2ω²/c³. The solver normalization K is MEASURED by applying
  L = −(∇² + k²) spectrally to a solved field at the source pixel, not
  fitted: K = −4812 − 8.27e6i, spread across sources 1.2e-3. Verified
  against exact forward-mode (jvp) directional derivatives through the
  Born series: rel err 1.4e-2, 1.8e-2, 9.2e-3 for 3 random
  blob directions (128², 0.5 MHz, 4 sources, 64 receivers, tol 1e-6).
  PASS (the ~1–2% is likely the spectral-Laplacian K measurement).
- Step B config: 128² grid, 64 mm, ring radius 28 mm, 32 sources, 128
  receivers. Hidden object: 4 random polygons (6–10 vertices, random
  radii, so convex and concave corners), painted in order. Synthetic
  speeds 1479, 1558, 1564, 1568 m/s vs water 1520, chosen within ±60 m/s
  and NOT tissue claims. Strongly nonlinear: ~3.5 rad of extra phase
  across the object at 0.7 MHz. Data from the same solver and grid
  (INVERSE CRIME, an optimistic first test), noise-free. Start =
  homogeneous water; no prior, no regularization. L-BFGS-B, 25
  iterations per frequency, frequency continuation 0.15 → 0.25 →
  0.35 → 0.5 → 0.7 MHz. Unknown scaled as (c − 1520)/100 m/s.
- Process errors (mine):
  1. First launch stopped at 0 iterations: the raw-units gradient was
     below L-BFGS's default gtol. Fixed by rescaling.
  2. Time estimate badly wrong (said ~5 min, based on a gradient at the
     water model). As the model fills with structure the Born series
     needs many more iterations, and the line search adds evaluations:
     stages took 29 s, 135 s, 413 s, 744 s. The run was interrupted
     during the 0.7 MHz stage at ~22 min. The 0.15–0.5 MHz results were
     kept.
- Result (model error = ‖c − c_true‖ / ‖c_true − 1520‖):
  - water start: 1.00
  - 0.15 MHz: 0.417 (misfit 2.1e-3 → 1.3e-5)
  - 0.25 MHz: 0.323
  - 0.35 MHz: 0.297
  - 0.50 MHz: 0.274 (misfit 7.4e-6)
  Mean |speed error| at 0.5 MHz: 4.0 m/s inside objects, 1.9 m/s in
  water. Visual (`results/figures/phase3_fwi_blind_stages.png`, decoded
  from the notebook output): the outline sharpens stage by stage and
  fills the true polygons, corners included. The slow (1479) region is
  recovered in place. The three similar regions (1558/1564/1568, i.e.
  0.3–0.6% mutual contrast) are visibly different shades. Minor ring
  artifacts at the transducer positions.
- Segmentation (same edge rule as run -07, 8% threshold): 2 regions
  found (object + water) vs 5 true. The weak internal edges fall below
  8% of the object/water jump. This is a limitation of the single
  global threshold, not of the reconstruction.
- Gate passed? N/A. Gate 2 still pending.
- Observations: blind region formation survives real, strongly
  nonlinear wave physics for a full ring at modest compute (the first
  4 stages took ~22 min on a T4). Caveats: inverse crime, no noise,
  sound speed only (Born series ignores density and attenuation),
  small 64 mm scaled domain.
- Next action (proposed): (a) a segmentation that handles multi-scale
  contrast (e.g. a relative or hierarchical threshold) so weak internal
  edges are found; (b) remove the inverse crime (data on a finer grid
  or from a different solver) and add noise; (c) cap per-stage time
  (fewer iterations / looser Born tolerance at early stages) before
  any heart-scale run, and re-estimate time from measured
  later-iteration costs, not the first gradient.
- Addendum, "distinct regions or one blob?" (user question): per TRUE
  region (visible part, eroded 2 px), reconstructed mean ± pixel std at
  0.5 MHz vs truth:
  - region 1: 1564.7 ± 1.8 (true 1564.4)
  - region 2: 1558.2 ± 2.1 (true 1558.0)
  - region 3: 1478.3 ± 5.1 (true 1478.9)
  - region 4: 1568.5 ± 3.7 (true 1568.2)
  - water: 1520.3 ± 2.3
  A one-blob reading would be ~1542 for all regions. True pairwise gaps
  are reproduced (6.4/10.2/3.8 m/s → 6.5/10.3/3.9). Separation as gap /
  pooled std: slow region vs others ~20; 6–10 m/s pairs ~3.3;
  the 3.8 m/s pair 1.3 (pixel distributions overlap heavily).
  Correlation of region means with truth 1.000. Verdict: genuinely
  distinct regions IN THIS TEST. But the sub-m/s accuracy of the means
  is consistent with the inverse crime (same solver and grid,
  noise-free) and should not be expected under model error and noise.
  The 4–10 m/s internal contrasts are the most at risk. The fair test
  (different-grid data + noise) decides.

### Run 2026-10-24-12 — FAIR test (no inverse crime + 30 dB noise): the object outline and strong contrasts survive; weak internal contrasts (≤10 m/s) are LOST in noise; the water background fills with noise speckle
- Phase: 3 → nonlinear. Colab T4. Seeds: 42 (object, replayed exactly,
  verified identical to run -11's), 7 (noise).
- User framing: improve segmentation only if clinically suboptimal; run
  the fair test if the result is not yet comparable to clinical
  margins. My answer: no honest CT comparison is possible from an
  idealized test, so the segmentation is deferred and the fair test
  run. Moving to Phase 4 needs Gate 2 (collaborator) regardless.
- Data: the same 4 polygons rasterized (4× supersampled) on a 256²
  grid (2× finer than the 128² inversion grid). Born series tol 1e-4,
  transducers on the same physical ring with different pixel rounding.
  Per-pair complex calibration with a water-only reference shot on each
  grid (as a real scanner does). Complex Gaussian noise at 30 dB SNR
  (ASSUMED level, ~3%). Fair vs inverse-crime data differ by
  7.4% / 4.7% / 4.4% at 0.15 / 0.3 / 0.5 MHz.
- Inversion: identical method (start = water, no prior, no
  regularization). L-BFGS-B, 15 iterations × 3 stages (0.15, 0.3,
  0.5 MHz). Runtime 436 s (estimate was 12–15 min).
- Result:
  - misfit 4.1e-3 → 2.1e-3, 2.9e-3 → 7.2e-4, 2.0e-3 → 4.5e-4
  - model error 1.12 / 0.87 / 0.92 (vs 0.27 inverse crime), dominated
    by noise speckle over the large water area
  - per region at 0.5 MHz, fair mean ± std (inverse crime in brackets):
    - region 1: 1565.6 ± 6.9 (1564.7 ± 1.8), true 1564.4
    - region 2: 1557.5 ± 7.8 (1558.2 ± 2.1), true 1558.0
    - region 3: 1476.0 ± 7.5 (1478.3 ± 5.1), true 1478.9
    - region 4: 1569.7 ± 8.0 (1568.5 ± 3.7), true 1568.2
    - water: 1521.6 ± 10.6 (1520.3 ± 2.3)
  - separation, gap / pooled std:
    - slow region vs others 10.7–12.4 (CLEAR)
    - 6.4 m/s pair 1.1
    - 10.2 m/s pair 1.5
    - 3.8 m/s pair 0.5 (LOST)
  - Figure: `results/figures/phase3_fwi_fair_test.png` (truth / crime /
    fair, outlines traced). Object outline and the slow region are
    clearly recovered; the interior and the water are speckled; the
    internal shading from run -11 is no longer visible.
- Gate passed? N/A.
- Observations: (1) run -11's internal-region shading was largely a
  product of the inverse crime; the user's suspicion was correct.
  (2) Under ~3% data error and no prior, pixel noise is ~8 m/s, so
  contrasts below ~10 m/s are unrecoverable pixelwise. Region means
  stay within 1–3 m/s, so averaging over a known region would still
  work, but blind segmentation would not. (3) Blood/myocardium differs
  by 8 m/s (cited 1584 vs 1576) and sits exactly in the failing range.
  This test predicts the inner wall is NOT recoverable without a
  stabilizing prior, more data, or higher SNR.
- Next action (proposed; the user suggested patient anatomy): patient023
  mid-ventricular slice (ACDC: LV/RV cavity + myocardium only, no atria)
  with cited speeds, same fair-test conditions, comparing no
  regularization vs total-variation (piecewise-constant) regularization.
  Derived anatomy arrays must stay gitignored.

### Run 2026-10-24-13 — Real-anatomy slice (patient023) under fair-test conditions: without a prior the inner wall is invisible; with TV (λ = 1e-6) the blood/myocardium wall (8 m/s) becomes VISIBLE at ~2σ
- Phase: 3 → nonlinear. Colab T4. Seeds: noise 11.
- Object: `src/phase3_patient_slice.py` builds patient023 ED,
  mid-ventricular slice 3/9 (of slices 1–5 containing RV+myo+LV).
  ACDC has no atria. In-plane spacing 1.562 mm, heart bbox 85.9 × 75.0
  mm, scaled ×0.386 to fit the 64 mm domain (wall ~3–4 mm ≈ 1.1 λ at
  0.5 MHz: scaled toy of real anatomy, not real size). Nearest-
  neighbour resampling only. Spacing axis order handled per
  jwave_cylindrical run -04. Labels saved to
  `results/phase3_patient023_slice_labels.npz` (gitignored, verified
  with git check-ignore). Moved to Colab as a 2,019-character RLE
  string (round trip verified exact). Speeds: water 1520 (Duck 1990),
  blood 1584 and myocardium 1576 (Mast 2000). Born series ignores
  density.
- Data: fair-test conditions as run -12 (256² data grid, water-shot
  calibration, 30 dB noise). Inversion 128², 3 stages (0.15/0.3/0.5 MHz)
  × 15 L-BFGS-B iterations, start = water. TV term λ·Σ√(|∇m|² + 1e-6)
  on m = (c − 1520)/100, gradient via jax.
- λ selection: planned as the discrepancy principle against the
  noise-only level (~5e-4 normalized). It FAILED to decide: even λ = 0
  ends above it (7.55e-4), because the coarse-vs-fine model error adds
  to the floor. Truth-free selection needs a total-error (noise + model
  error) estimate, which we do not have. λ = 1e-6 was the single TV
  value tested, chosen a priori from a TV-scale estimate, not tuned on
  truth. Runs for λ = 3e-7 and 3e-6 were not done.
- Result (per structure, eroded 1 px; mean ± pixel std, m/s):
  - λ = 0 (454 s, misfit 7.55e-4): LV 1583.1 ± 11.9, RV 1583.9 ± 11.1,
    myo 1575.9 ± 11.7, water 1516.8 ± 22.3. LV–myo gap +7.1 (true +8),
    separation 0.61σ; RV–myo 0.70σ. Interior is pure speckle.
  - λ = 1e-6 (713 s, misfit 1.08e-3): LV 1583.0 ± 3.4, RV 1583.2 ± 3.6,
    myo 1575.9 ± 3.8, water 1518.2 ± 12.0. LV–myo gap +7.2, separation
    1.98σ; RV–myo 2.00σ. The LV cavity is visible as a distinct darker
    disc inside the myocardial ring, close to the true inner outline;
    the RV cavity is distinct from the wall.
  - Figure: `results/figures/phase3_fwi_patient023_tv_partial.png`
    (colour scale 1500–1600 m/s, narrower than earlier figures).
    Speckle and transducer-position artifacts persist in the water.
- Gate passed? N/A. Gate 2 pending.
- Observations: (1) the inner wall at the real cited contrast (8 m/s,
  0.5%) is recoverable blind, from noisy, model-mismatched data,
  once the minimal piecewise-constant prior (TV) is used. It is
  visible but fuzzy (2σ), not crisp. (2) Region means are unbiased
  with or without TV; TV mainly cuts pixel noise ~3×. (3) Open: λ
  choice without truth; whether stronger TV sharpens or erases the
  8 m/s wall; scaled (not real-size) anatomy; 2D; sound speed only;
  idealized ring.
- Next action (user decision): optionally λ = 3e-6 (~12 min) to probe
  the trade-off; a truth-free λ rule (e.g. an L-curve); real-size
  anatomy (heart-scale domain, more compute); Gate 2 packet.

### Run 2026-10-24-14 — Gate 2 WAIVED by user (exploratory approach); scale-up timing measured on the heart model
- User decision: "this is another approach so don't need sign off".
  Gate 2 is waived for fourier_inverse and recorded in the protocol as
  a deliberate deviation from CLAUDE.md's general jWave-phase rule.
  Results stay labelled NOT collaborator-reviewed.
- Timing (T4, one 32-source misfit+gradient, Born tol 1e-3, timed on
  the HEART model, not water, after the run -11 lesson):
  - scaled heart, 64 mm, 128², 0.5 MHz: 6.9 s
  - REAL-size heart (patient023 slice unscaled, ~87 × 75 mm, NN
    zoom of the cropped label map), 130 mm, 260², ring radius 56 mm,
    0.5 MHz: 27.1 s (3.9×)
  - real size at 0.8 MHz: interrupted before finishing (not
    measured); extrapolated ~45–70 s from CBS iterations scaling with
    size in wavelengths
  First attempt zoomed the whole label map including water (332² into
  260²) and crashed; fixed by cropping the heart before zooming.
- Estimates derived (see the reply to the user): real size,
  same schedule ≈ 30 min (no prior) / ≈ 45 min (TV); + a 0.8 MHz stage
  ≈ +20 min (±2×). Unmeasured optimizations are listed as estimates
  only.

### Run 2026-10-24-15 — Cheap speed-ups: the thin default PML was the bottleneck; tuned PML + warm start gives 713 s → 18 s (~40×) with BETTER accuracy and image quality
- Phase: 3 → nonlinear. Colab T4. Same data (run -13) and λ = 1e-6.
- Warm start: `src/cbs_warm.py`, a modified copy of jWave 0.2.1
  `born_series` (LGPL-3.0, header and attribution kept) taking an
  initial guess and returning the padded field and iteration count.
  Bug caught: the first version scaled the output by the PHYSICAL ω
  instead of the normalized ω = 1; the error was exactly ω
  (3,141,592). Fixed; zero guess now reproduces jWave exactly
  (rel diff 0.0). Warm start after a small model change: 4,236 → 1,113
  mean iterations, same answer (2.3e-4), but the vmapped batch waits
  for its slowest source (max 4,493), so wall time barely improved.
- Discovery: on the heart model the stock solver (PML 16, α = 1) hit
  max_iter = 5000 for some sources. All earlier runs were slow AND
  partly unconverged.
- PML sweep (4 sources, heart model, 0.5 MHz, tol 1e-3; accuracy =
  shape error vs exact H0 in water):

  | PML | α | its (mean/max) | time | error |
  |---|---|---|---|---|
  | 16 | 1.0 | 1952/2467 | 0.48 s | 2.9% |
  | 16 | 0.5 | 150/152 | 0.06 s | **64.9%** (fast but wrong) |
  | 16 | 0.25 | 951/956 | 0.18 s | 95.5% |
  | 32 | 1.0 | 183/185 | 0.05 s | 0.2% |
  | 32 | 0.5 | 70/70 | 0.04 s | 2.3% |
  | 32 | 0.25 | 129/131 | 0.07 s | 59.7% |
  | 64 | 1.0 | 231/234 | 0.11 s | 0.2% |
  | **64** | **0.5** | **92/93** | **0.06 s** | **<0.05%** |
  | 64 | 0.25 | 48/48 | 0.03 s | 2.0% |

  Chosen: PML 64, α 0.5 (most accurate, ~8× faster than the default).
  Lesson: every solver-speed change needs the exact-solution accuracy
  check; several fast settings converge to wrong fields.
- Fast reconstruction (PML 64/α 0.5 + warm-started forward and
  adjoint), same schedule: **18 s total vs 713 s**. Per structure:
  LV 1582.6 ± 2.8, RV 1582.2 ± 3.7, myo 1575.2 ± 3.6, water
  1520.0 ± 4.5 (was ± 12.0). Inner wall LV–myo gap +7.4 (true +8),
  separation **2.28σ** (was 1.98σ); RV–myo 1.89σ. Quality improved
  because the old ~3% solver error no longer has to be fitted.
- Code: `src/phase3_fwi_born.py` gains `make_fast_solver`,
  `zero_guess`, `misfit_and_grad_warm` (PML_FAST = 64, ALPHA_FAST =
  0.5).
- Gate passed? N/A (Gate 2 waived by user, run -14).
- Observations: runs -11 to -14 timings and the run -14 scale-up
  estimates are now obsolete (~40× too pessimistic). The run -13 data
  were generated with the old PML on the fine grid, so they carry ~3%
  solver error. That is fine as "model mismatch", but data for
  real-size tests should use the accurate solver. The TPU question is
  less pressing now.
- Next action: re-time real size with the fast solver; real-size
  reconstruction.

### Run 2026-10-24-16 — Real-size attempt FAILED to move; root cause: the warm start returns STALE fields. Run -15's 18 s result is now in doubt.
- Phase: 3 → nonlinear. Colab T4. Seed 13 (noise).
- Real-size setup: patient023 slice at true size (86 × 75 mm, NN zoom
  ×1/0.386 of the scaled fine labels). Data on 520² (0.25 mm, 130 mm
  domain, ring radius 56 mm, 32 src / 128 rec) with the accurate fast
  solver (PML 64, α 0.5, tol 1e-4), water-shot calibration, 30 dB
  noise, at 0.15/0.3/0.5/0.8 MHz. Data generation 146 s (incl.
  compile). Inversion grid 260² (0.5 mm).
- TV (λ = 1e-6) reconstruction with warm starts: every stage ended with
  the model unchanged (1520.0 ± 0.0 everywhere).
- Diagnosis by reasoning plus three cheap checks (the user asked to
  stop trial-and-error and reason through the code):
  1. Data, forward field and gradient are finite. Starting relative
     residual 0.38.
  2. The gradient is correct: steps along −g reduce J as predicted at
     small steps (t = 100: 7.2551e-2 vs predicted 7.2545e-2).
  3. L-BFGS-B backtracked 20 times to a step of ~1e-14 ("relative
     reduction ≤ factr·epsmch"), although its first trial step
     (unit length along −g) should decrease J (cold test: t = 1000
     → 0.0729 → 0.0697).
  4. Decisive check: at L-BFGS's first trial point, cold solve
     J = 7.111e-2 (152 its, decrease 1.76e-3); warm start from the
     previous field J = 7.283e-2 after only **2 iterations** (decrease
     4e-5). The Born series stops on residual/‖source‖ < tol, so after a
     small model update the old field already passes and is returned
     nearly unchanged. To the line search this looks like no decrease
     or noise.
- Consequences: warm start marked BROKEN in `src/phase3_fwi_born.py`
  (`misfit_and_grad_warm`). Run -15's "713 s → 18 s" used warm starts,
  so some of its steps may have been frozen. Its image quality looked
  good (2.28σ), but the speed and quality claims need re-verification
  with cold solves. The tuned PML (64, α 0.5) finding from run -15
  stands: it is a solver-accuracy result checked against the exact H0.
- A cold-solve real-size run was started and then interrupted by the
  user; no result.
- Also noted: a tol 1e-5 attempt was likely below what float32 CBS can
  reach (it would run to max_iter) and was interrupted. Do not set the
  Born tolerance below ~1e-4 in float32.
- Estimate for the real-size cold-solve TV reconstruction (reasoned
  from measured costs, not run): ~1 / 2–3 / 5 / 10–15 s per evaluation
  at 0.15 / 0.3 / 0.5 / 0.8 MHz, ~20 evaluations per stage → ~6–8 min,
  plus ~20–30 s compile per stage.
- Next action (user decision): run the cold-solve real-size
  reconstruction (~7 min) uninterrupted, and re-verify the scaled run
  -15 with cold solves (~1–2 min).

### Run 2026-10-24-17 — REAL-SIZE patient023 slice, blind, TV: the inner wall is clearly recovered (LV–myocardium 4.5σ, gap 7.8 vs true 8 m/s) in 536 s on a T4
- Phase: 3 → nonlinear. Colab T4. Data from run -16 (520², accurate
  fast solver, water-shot calibration, 30 dB noise, seed 13).
- Inversion: 260² (0.5 mm), 130 mm domain, ring radius 56 mm, 32
  sources / 128 receivers. Stages 0.15 / 0.3 / 0.5 / 0.8 MHz × 15
  L-BFGS-B iterations. TV λ = 1e-6 (same a-priori value as run -13,
  not re-tuned). COLD Born-series solves (PML 64, α 0.5, tol 1e-3),
  start = water, no labels, no region count.
- Progress: every stage ran its full 15 iterations (no freezing).
  J 3.67e-3 → 1.80e-3 → 1.48e-3 → 1.42e-3. Cumulative time 144 / 265 /
  390 / 536 s (estimate was 6–8 min plus compile: consistent).
- Result (per structure, eroded 1 px): LV 1583.7 ± 0.9, RV 1583.4 ±
  2.7, myo 1575.9 ± 2.3, water 1519.1 ± 4.2. LV–myo gap +7.8 m/s (true
  +8), separation **4.51σ**; RV–myo 2.98σ. Figure:
  `results/figures/phase3_fwi_patient023_realsize_tv.png` (truth vs
  reconstruction, true outlines, colour 1500–1600 m/s): the LV cavity is
  a distinct disc following the true inner outline, the RV is distinct,
  and the outer boundary is sharp. Mild artifacts at transducer
  positions.
- Interpretation: real size is EASIER than the ×0.386 scaled toy,
  because the wall (~8–10 mm) is ~4 λ at 0.8 MHz instead of ~1 λ. This
  is the first result recovering the blood/myocardium boundary at real
  cited contrast and real anatomical size, blind, from noisy
  finer-grid data.
- Caveats (all still apply): λ not chosen truth-free; data generated by
  the same solver TYPE (Born series) on a finer grid, which is not
  fully independent physics; sound speed only (density and attenuation
  ignored in both data and inversion, although blood/myo densities are
  equal in the cited table, water/tissue densities are not); 2D,
  static (no cardiac motion), idealized full ring with point
  transducers; 30 dB noise assumed. NOT collaborator-reviewed (Gate 2
  waived).
- Still pending: re-verify scaled run -15 with cold solves (its warm
  start was broken).
- Next action: user decision.

### Run 2026-10-24-18 — Attenuation: in the data it barely hurts a speed-only reconstruction (inner wall 4.5σ → 4.0σ); as an extra UNKNOWN (naive setup) it was NOT recovered and slightly degraded speed (3.5σ)
- Phase: 3 → nonlinear. Colab T4 via colab-mcp. Seed 17 (noise).
- Solver: `src/cbs_ext.py` (Born series with attenuation, a modified
  copy of jWave 0.2.1, LGPL-3.0): k = ω/c + iα (Np/m). Checks: α = 0
  matches the fast solver exactly (0.0); homogeneous α = 20 Np/m shape
  error vs exact H0(complex k) 0.86% (26% if attenuation is ignored);
  adjoint gradients vs jvp: speed 0.25%, attenuation 0.1%. One bug
  fixed: the zero-guess helper was hard-coded to 32 sources.
- Data: real-size patient023 (as run -16/-17) with cited attenuation
  (myo 0.52, blood 0.20 dB/cm/MHz [Mast 2000], water 0.0022 [Duck
  1990]; α ∝ f is an ASSUMED exponent). 520² grid, water-shot
  calibration (attenuating water), 30 dB noise. Attenuation effect:
  median amplitude change over all pairs −0.11 / −0.19 / −0.33 /
  −0.51 dB at 0.15 / 0.3 / 0.5 / 0.8 MHz (a first printout of "−6 dB
  everywhere" compared uncalibrated vs calibrated data, i.e. 2×
  grid scaling; caught and corrected).
- (a) Speed-only reconstruction on attenuating data (TV 1e-6, cold,
  586 s): LV 1583.3 ± 1.0, RV 1583.2 ± 2.8, myo 1575.5 ± 2.6, water
  1518.9 ± 5.2. LV–myo gap +7.8, separation **4.02σ** (vs 4.51σ on
  non-attenuating data, run -17); RV–myo 2.85σ. Ignoring attenuation
  costs little at ≤0.8 MHz.
- (b) Speed + attenuation as unknowns (attenuation in dB/cm/MHz,
  scaled u = a/0.5, bounds [0, 2], same TV λ on both maps, 574 s):
  speed LV 1583.1 ± 1.0, RV 1582.8 ± 3.6, myo 1575.3 ± 3.0, water
  1519.4 ± 5.1, LV–myo separation **3.46σ**, RV–myo 2.27σ.
  Attenuation map ≈ 0.009–0.010 dB/cm/MHz UNIFORM everywhere (true
  0.52 myo / 0.20 blood): NOT recovered. Final misfit 2.29e-3 vs
  2.13e-3 for speed-only, i.e. worse fit, so the optimizer did not
  converge on the extra unknowns.
- Likely causes (not yet tested): the attenuation gradient is tiny
  relative to speed (attenuation changes the data by only ~0.1–0.5 dB),
  so with the arbitrary scaling (SA = 0.5) L-BFGS barely moves it; the
  shared TV λ may dominate the weakly constrained attenuation map
  (pushing it to a constant); 15 iterations per stage is short. A fair
  attempt needs gradient-balanced scaling, a separate λ for
  attenuation, and more iterations, or a staged scheme (speed first,
  then attenuation).
- Figure: `results/figures/phase3_fwi_patient023_realsize_attenuation.png`
  (truth / speed-only / speed+atten speed / speed+atten attenuation).
- Checkpoint on the Colab runtime disk: `/content/checkpoint_session.npz`
  (1.14 MB, all reconstructions and datasets of runs -13 to -18). Not
  transferred locally (user decision); lost if the free runtime is
  recycled.
- Colab-mcp note: the browser link drops after an interrupted tool call.
  Pressing Enter in the old tab's address bar (same #mcpProxyToken/
  port) reattaches it without reloading.
- Gate passed? N/A (Gate 2 waived).
- Next action: user decision (fix attenuation scaling, density, or
  stop here and write up).

### Run 2026-10-24-19 — INDEPENDENT time-domain simulator data (removes the "same solver" caveat) + DENSITY in the data: inner wall survives (4.45σ control, 5.35σ with density); density effects confined to the outer boundary
- Phase: 3 → nonlinear. Colab T4. Seeds 21 (control), 22 (density), 23
  (all three; data generated, reconstruction NOT run per user).
- Simulator: jWave time-domain (PSTD) via the ring project's
  attenuating extension (copied into the notebook as `atten_td.py`;
  α = 0 → stock jWave). Mask sources vmapped (8 per batch), 128
  pressure sensors, 520² grid (0.25 mm), dt 47.3 ns, 2746 steps,
  derivative-of-Gaussian pulse (σ 0.25 µs). ~60 s per 32-source
  dataset on a T4. Datasets: water reference; control (speed only,
  water density everywhere); speed + cited density (water 994, blood
  and myocardium 1060 kg/m³); all three with attenuation set per
  frequency (this solver's damping is frequency-independent, so one
  run per frequency). Total 415 s.
- DFT convention check against the Born-series solver (control,
  0.5 MHz, object/water ratios): sign +1 corr 1.0000, rel diff
  **0.008**; sign −1 corr 0.50. Independent confirmation that the
  inversion's wave model matches a different simulator to <1%.
- Calibration: D = (TD object/TD water) × Born-series water field on the
  inversion grid, then 30 dB noise. Data differences vs control:
  density 4.6–4.7%; all three 4.7 / 5.1 / 5.5 / 6.3% at 0.15–0.8 MHz.
- Reconstructions (speed-only model, TV 1e-6, cold solves, 260²,
  4 stages × 15 iterations, ~490 s each):
  - control: LV 1583.7 ± 0.9, RV 1583.4 ± 2.7, myo 1575.9 ± 2.3, water
    1519.5 ± 2.0. Gap +7.9, **4.45σ**; RV–myo 2.97σ; final J 1.12e-3.
    Matches run -17 (Born-series data, 4.51σ): the result does not
    depend on the inverse crime.
  - density: LV 1583.7 ± 1.0, RV 1583.6 ± 2.1, myo 1576.1 ± 1.7, water
    1519.4 ± 2.0. Gap +7.7, **5.35σ**; RV–myo 3.88σ; final J 1.25e-3
    (higher: the speed-only model cannot explain density). The
    difference map (density − control, figure run by the user in the
    notebook) shows the change confined to a thin band along the
    OUTER heart boundary (water/tissue density step 994→1060); the
    heart interior and inner wall are essentially unchanged, as
    predicted (equal cited blood/myocardium densities).
- All-three reconstruction: not run (user asked for an estimate).
  Estimate: inner wall ~4–5σ, i.e. the density result (~5σ) minus the
  attenuation penalty seen in run -18 (~10%), plus the
  outer-boundary density band.
- Checkpoints on the Colab disk: `/content/checkpoint_td_ctrl.npz`,
  `/content/checkpoint_td_den.npz`, plus `/content/checkpoint_session.npz`.
- Caveat that remains crucial: the inner-wall result relies on the
  cited EQUAL blood/myocardium densities; a small real density
  difference would change the inner-wall reflection (untested).
- Gate passed? N/A (Gate 2 waived).

### Run 2026-10-24-20 — Data-subset sweep ("which data carry the information?"): NO free redundancy at 32 sources; receivers matter; the lowest frequency is ESSENTIAL (without it the inversion fails outright)
- Phase: 3 → nonlinear. Colab T4. Same independent-simulator control
  data as run -19 (real size, 30 dB, seed 21), same method (TV 1e-6,
  cold solves, 15 L-BFGS-B iterations per stage). Subsets are taken
  from the existing data, so no new simulation. Results saved to
  `/content/sweep_results.json` and `/content/sweep_*.npy`.
- Metrics: inner-wall separation (LV–myo, σ units), gap, RV–myo
  separation, outer-boundary band RMSE (±2 px), water std, time.

  | config | src | rec | freqs | time | LV–myo σ | gap | RV–myo σ | outer RMSE | water std |
  |---|---|---|---|---|---|---|---|---|---|
  | A | 32 | 128 | 4 | 486 s | **4.45** | 7.87 | 2.97 | 18.8 | 2.0 |
  | B | 16 | 128 | 4 | 256 s | 3.34 | 7.57 | 2.38 | 19.6 | 2.7 |
  | C | 8 | 128 | 4 | 167 s | 1.76 | 7.27 | 1.40 | 21.6 | 4.9 |
  | D | 16 | 64 | 4 | 259 s | 2.57 | 7.52 | 1.68 | 21.1 | 3.7 |
  | E | 16 | 128 | 0.3, 0.8 | 208 s | **−0.85** | **−35.2** | 0.01 | 33.7 | 34.7 |
  | F | 8 | 128 | 0.3, 0.8 | 122 s | **−1.13** | **−41.6** | 0.02 | 43.6 | 36.8 |

- Findings:
  1. Time ∝ number of sources (16 → 256 s, 32 → 486 s). Halving
     sources costs ~25% of the inner-wall separation; 8 sources lose
     the wall. Quality has NOT saturated at 32 sources, consistent with
     run -03's angular-Nyquist estimate (far more angles needed for
     arbitrary objects). The data are not wastefully redundant here.
  2. Receivers carry real information (16 src: 128 rec 3.34σ vs 64 rec
     2.57σ) while costing almost no compute, so keep or add receivers.
  3. Dropping the 0.15 MHz start (and 0.5) makes the inversion FAIL
     (gaps of −35 to −42 m/s, misfit ~10× higher, water std ~35 m/s):
     classic cycle-skipping. The low-frequency stage is cheap (its
     cost is in the smaller solves) but essential for the nonlinear
     inversion to start in the right basin. The middle stages were
     not tested individually.
- Implication: speed must come from algorithms that keep all sources'
  information per unit of compute (source encoding with redrawn
  random phases, fewer iterations per stage), not from discarding
  sources or the low frequency. Untested: dropping the 0.3/0.5 MHz
  middle stages while keeping 0.15 MHz; >32 sources.
- Gate passed? N/A (Gate 2 waived).

### Run 2026-10-25-21 — Multi-resolution schedule (grid matched to each frequency): 486 s → 199 s (2.4×) at the same quality
- Phase: 3 → nonlinear. Colab T4. Motivation (user): avoid recomputing
  the same thing at needless detail. Clarified first that wave solves
  are global (cost is per source, not per pixel); the real waste was
  ~20 points/wavelength at 0.15 MHz on the fixed 260² grid.
- Schedule: 0.15 MHz on 64², 0.3 on 128², 0.5 on 216², 0.8 on 260²
  (~5 points/wavelength; 0.8 MHz at ~3.8 as before). Model carried up
  with jax.image.resize (linear). Same independent-simulator control
  data and SAME noise realization, mapped onto each grid as (noisy
  calibrated data ÷ 260² water field) × this grid's water field. TV
  scaled λ_n = 1e-6 × 260/n to keep the same physical regularization.
  32 sources, 128 receivers, 15 L-BFGS-B iterations per stage, cold
  solves.
- Result: stage times cumulative 12 / 28 / 80 / 199 s (the final 0.8 MHz
  stage on 260² is now ~60% of the total). Final J 1.05e-3 (baseline
  1.12e-3). Metrics vs baseline A (run -20): LV–myo separation 4.32σ
  (4.45), gap 7.94 (7.87), RV–myo 2.87σ (2.97), outer-band RMSE 19.6
  (18.8), water std 1.64 (1.98). **2.4× faster, same quality.**
- Saved: `/content/sweep_G_multires.npy`, `/content/sweep_results.json`
  (Colab disk).
- Next: (a) fewer iterations in the final stage; (b) region-of-interest
  unknowns after the early stages; (c) a correct warm start (stopping
  on the update, not the field). All aimed at the now-dominant final
  stage.
- Gate passed? N/A (Gate 2 waived).

### Run 2026-10-25-22 — Final-stage speed-ups: blind heart-region-only unknowns IMPROVE quality; with 8 final iterations the total is 143 s at baseline quality (3.4× faster than the original 486 s)
- Phase: 3 → nonlinear. Colab T4. Same data and settings as run -21
  (multi-resolution G).
- H: final 0.8 MHz stage 8 iterations instead of 15.
- I: from the 0.5 MHz stage on, only pixels in a BLIND region of
  interest are unknowns: |c − 1520| > 10 m/s in the current model,
  dilated ~8 mm (44% of pixels at 216², 42% at 260²). The rest are
  frozen via equal L-BFGS-B bounds.
- J: H + I.
- Results (LV–myo σ / gap / RV–myo σ / outer-band RMSE / water std /
  time):
  - G: 4.32 / 7.94 / 2.87 / 19.6 / 1.64 / 199 s
  - H: 3.84 / 7.98 / 2.44 / 20.3 / 1.76 / 146 s
  - I: **4.66** / 7.96 / **3.42** / 18.9 / 1.56 / 192 s
  - J: **4.25** / 7.98 / 2.96 / 19.3 / 1.64 / **143 s**
- Findings: (1) restricting unknowns to the blind ROI improves quality
  (fewer unknowns → less noise fitted outside the heart, better
  conditioning) while barely changing time, because wave solves still
  cover the whole domain. (2) Cutting final iterations 15 → 8 saves ~50 s
  but costs ~11% clarity on its own; combined with the ROI the loss is
  recovered. (3) Overall speed-up vs the original baseline: 486 → 143 s
  (3.4×) at equal quality (4.25σ vs 4.45σ), or 486 → 192 s (2.5×) at
  better quality (4.66σ).
- Not done (reasoned, user agreed to proceed without it): the correct
  warm start would need ~2/3 of cold iterations to be trustworthy, so
  ~1.5× gain for the engineering risk.

### Run 2026-10-25-23 — Session results migrated off Colab; CT-based chest phantom (lungs, ribs, chest wall + inserted heart) built for the lung/rib test
- Storage: Colab results saved to the user's Google Drive by mounting
  Drive in the notebook (a base64 printout of ~4.8M chars bogged down
  the tab and was abandoned; chunked transfer was stopped as too slow).
  The user downloaded the Drive folder as a zip. It was extracted in a
  scratch folder (no unsafe paths, 27 files), the packed
  `all_results_patient023.npz` md5 was verified (b574de2d…, matches
  Colab), and the folder was copied to
  `fourier_inverse/results/colab_session_2026-10-25/` (new gitignore rule
  `fourier_inverse/results/colab_session_*/`; verified that no array
  file is committable).
- Phantom: `src/phase3_ct_chest_phantom.py` → labels in
  `results/phase3_ct_chest_phantom_patient.npz` (gitignored). Source:
  user-supplied published axial chest CT (AJR figure, COVID-19 case),
  lung window, carina level.
  - TRACED: body outline and lungs. The first body threshold (25) was
    below the outside-air gray (~35) and took the whole frame; it was
    fixed by thresholding the bright chest-wall ring (>120) and filling
    holes. Lungs = dark regions inside the body, closed and filled;
    airways and vessels are lumped into homogeneous lung. Dense COVID
    consolidation partly remains soft tissue, which is acoustically
    reasonable (fluid-filled).
  - Outside air → water bath (label 0), per user.
  - INSERTED: patient023 ventricles at true size (86 × 75 mm) at the
    deepest mediastinal point. This CT level is anatomically ABOVE the
    ventricles, so it is a composite.
  - SCHEMATIC (not traced; the lung window does not separate bone): 12
    rib cross-sections (12 × 20 mm ellipses) on an ellipse in the chest
    wall, and a vertebral body (34 × 36 mm).
  - ASSUMED: body width 320 mm (no ruler) → 0.436 mm/px. Grid 0.5 mm,
    400 mm domain (800²).
  - Areas: soft tissue 437.5 cm², lung 137.7, bone 34.1, myocardium
    20.9, blood 25.5.
  - Figure `results/figures/phase3_ct_chest_phantom_patient.png`
    (gitignored: it reproduces the copyrighted CT as its left panel).
- UPDATE (same day): cited lung and bone values added to `src/tissue.py`
  (via PubMed). Lung from Miller 2016 UMB review
  (doi:10.1016/j.ultrasmedbio.2016.08.006) citing Dunn 1986
  (doi:10.1121/1.393818): 658 m/s and 37.3 dB/cm at 1 MHz, density
  200–400 kg/m³ (300 used). Measured 1–5 MHz, so 0.15–0.8 MHz is an
  EXTRAPOLATION. Cortical bone from Pichardo, Sin & Hynynen 2011 PMB
  (doi:10.1088/0031-9155/56/1/014): 2384 / 2471 m/s and 33 / 240 Np/m
  at 270 / 836 kHz, density 1904 ± 117 kg/m³. These are SKULL values
  used as a stand-in for the schematic ribs. The IT'IS database pages
  could not be read (JavaScript-rendered tables).
- Original note: NOT yet possible to simulate: lung and bone acoustic properties need
  cited sources (the cited table has water, blood, myocardium and a
  chest-wall muscle proxy only). Compute: a 400 mm domain is ~2.4× the
  linear size of the 130 mm runs (~9× the pixels at 0.5 mm), and
  lung/bone contrast will raise Born-series iteration counts
  substantially (to be measured).
- Gate passed? N/A (Gate 2 waived).
- Saved on Colab: `/content/sweep_H*.npy`, `sweep_I*.npy`, `sweep_J*.npy`,
  `sweep_results.json`.
- Gate passed? N/A (Gate 2 waived).

### Run 2026-10-25-24 — CHEST reachability test (lungs + ribs + chest wall, time domain with density and attenuation): only ~12–14% of the inner-wall signal survives; side views blocked; front window best
- Phase: 3 → nonlinear. NEW Colab notebook (T4), set up from Drive (solver
  files from the session folder). CT chest phantom from run -23 (800² at
  0.5 mm, 400 mm domain), transferred as zlib+base64 RLE (6,512 chars,
  round trip verified).
- Physics: jWave time domain + the attenuating extension (density and
  attenuation). Cited values from `src/tissue.py`:
  - water 1520 / 994
  - soft tissue = chest-wall proxy 1580 / 1050 / 0.74 dB/cm/MHz
  - lung 658 / 300, 37.3 dB/cm at 1 MHz scaled ∝ f (EXTRAPOLATED below
    1 MHz)
  - bone 2384 / 1904 / 33 Np/m (Pichardo, 270 kHz)
  - myocardium 1576 / 1060 / 0.52; blood 1584 / 1060 / 0.20
  The damping is frequency-independent, so attenuation was set at
  0.3 MHz (slightly pessimistic at 0.15 MHz).
- Setup: ring radius 185 mm, 32 sources, 128 receivers,
  derivative-of-Gaussian pulse (σ 0.5 µs), dt 63 ns, 4,768 steps,
  analysed at 0.15 and 0.3 MHz.
- Test (no reconstruction): inner-wall signal = ‖data(true heart) −
  data(LV/RV blood replaced by myocardium)‖ / ‖data(true)‖, in the
  CHEST vs the same heart in WATER (same grid and ring). 4 datasets,
  1,144 s total (estimate 17–20 min).
- Result:
  - 0.15 MHz: chest 1.56e-3 vs water 1.11e-2 → **14% survives**
  - 0.30 MHz: chest 2.27e-3 vs water 1.89e-2 → **12% survives**
  (≈ −17 dB). Per-sample vs 30 dB noise: chest 0.05–0.07×, water
  0.35–0.60×. Averaging over 32 × 128 = 4,096 pairs gives roughly 64×,
  i.e. ~3–4.5 (chest) vs ~22–38 (water) noise units: a rough
  heuristic, not a reconstruction.
- Acoustic windows (per-pair map, 0.3 MHz,
  `results/figures/phase3_chest_windows_patient.png`, gitignored
  because CT-derived): the straight-through transmission band seen in
  water is ABSENT in the chest (blocked by lungs). The signal survives
  in three source/receiver blocks. Per-source chest/water share: front
  (270°) 0.28, lower-front oblique (45°) 0.16, back/near spine (90°)
  0.12, lateral (0°/180°) 0.00–0.01. This is consistent with clinical
  echo windows.
- Interpretation: through a realistic chest, ~86–88% of the inner-wall
  information is lost and the lateral directions carry essentially
  none. Given that water-bath reconstructions reached ~4.5σ, an
  equivalent chest reconstruction at 30 dB would likely fall to around
  or below detection (~0.5σ if signal scales directly; NOT run).
  Recovery would need much higher SNR (averaging: ~64× more
  acquisitions for +18 dB), data concentrated in the windows, and/or
  frequency/geometry optimized for window paths.
- Caveats: lung values extrapolated below the measured 1–5 MHz;
  schematic ribs with skull-bone values; 2D (real ribs and lungs are 3D,
  and intercostal gaps differ per slice); composite anatomy (heart
  inserted at a carina-level slice); assumed 320 mm body width.
- Saved: `/content/chest_reachability_spectra_patient.npz` and the
  windows figure, copied to the Drive session folder.
- Gate passed? N/A (Gate 2 waived).

### Planning note 2026-10-25 (no run) — external consultation on repeated firing, learned priors and hallucination; plan for the next session
- Context (user): every result so far is a SINGLE-fire acquisition with one
  noise realization at 30 dB. Not yet implemented: repeated firing /
  coherent averaging, or ML (learned priors, shape fitting, quality
  enhancement, filling sparse regions), each of which brings its own
  uncertainty, compute and hallucination risk. The user consulted an
  external assistant; its points are recorded here as HYPOTHESES and
  plans, not results.
- Points from the consultation, with my assessment:
  1. Coherent averaging of N repeated, aligned, stationary firings with
     independent noise improves SNR ~√N (+6 dB per 4×). To offset an
     amplitude-sensitivity loss to 14%, N ≈ 1/0.14² ≈ 51 (+17 dB). This
     is ILLUSTRATIVE: run -24's 14% is a ratio of normalized
     differential data, not a calibrated detector sensitivity. The
     arithmetic is correct; the mapping to reconstruction quality is
     untested.
  2. Repetition raises Fisher information (N·JᵀΣ⁻¹J) but NOT the rank of
     J: anything invisible to every recorded path stays invisible.
     Correct, and consistent with runs -03/-04/-06. Fixed reverberation,
     coherent speckle and model error do not average away; cardiac and
     respiratory motion break alignment (ECG/respiratory gating needed).
  3. Classify improvements by whether they add MEASURED information:
     repeated firing reduces noise only; more angles, frequencies and
     receivers add information; learned priors and AI region filling
     add none and carry hallucination risk, highest where the data are
     insensitive (e.g. run -24's blocked lateral views). Agree. A
     cited 2024 learned-FWI hallucination study was NOT verified here.
  4. Suggested order: (A) repeated acquisition and noise modelling
     (1/4/16/64/256 repeats; then correlated noise, phase jitter,
     motion); (B) acquisition geometry (place transducers in the
     windows; does it raise the smallest useful singular values of the
     noise-whitened Jacobian?); (C) a physics-consistent learned prior
     (initializer or proximal operator, not free image generation) vs TV;
     (D) uncertainty and failure analysis (false boundaries, missed
     structures, out-of-distribution anatomy).
  5. Key experiment before any ML: a noise-vs-information curve.
     Reconstruct the SAME chest phantom at increasing SNR. If quality
     keeps improving, repetition is a viable engineering fix; if it
     plateaus even near noise-free, the bottleneck is missing angles,
     model mismatch or optimization, and more repeats will not help.
- My additions:
  - The averaging is cheap computationally: average the raw waveforms
    BEFORE inversion, so it costs one reconstruction, not N.
  - BLOCKER for the chest SNR curve: our fast inversion solver (Born
    series) has NO DENSITY term, while the chest has very strong density
    contrasts (lung 300, bone 1904 vs tissue ~1050). Inverting chest data
    with it would mix model-mismatch error into the SNR curve, which is
    exactly the confound point 5 is meant to separate. Needed first:
    either (a) a density-capable frequency-domain solver (the √ρ transform
    of the Helmholtz equation inside the Born series, validated against
    the time-domain simulator like run -19's 0.8% check) or (b)
    time-domain full-waveform inversion with jWave autodiff (more compute).
  - A cheaper first version of point 5 that avoids the blocker: run the
    SNR sweep on the WATER-bath real-size heart (Born-series solver is
    valid there; run -19 control data), at 20/30/40/50 dB plus noise-free,
    to see whether inner-wall quality is noise-limited or plateaus. GPU,
    ~5 × 2.5 min with the fast multires+ROI setting.
- Not testable meaningfully on CPU today: CPU reconstructions at these
  grids would take far longer than the GPU's ~2.5 min each.
- Plan for the next session (GPU): (1) water-bath SNR sweep (cheap, now);
  (2) density-capable Born-series solver + validation vs time domain;
  (3) chest SNR sweep with averaging and window-placed transducers;
  (4) only then learned priors, with out-of-distribution hallucination
  tests.

### Planning note 2026-10-25 (no run) — optimization roadmap (external consultation, user-supplied), with status against what is already done
- Baseline to beat: 143 s on a T4 at 4.25σ inner-wall separation
  (multires 64→128→216→260² + blind ROI + 8 final iterations, 32 src,
  128 rec, 0.15/0.3/0.5/0.8 MHz; run -22). Optimize for TIME TO A FIXED
  ACCURACY, rejecting any speed-up that relies on stale fields,
  suppressed weak structures or premature stopping. Targets: 90–120 s
  (engineering), 30–60 s (ambitious).
- Roadmap (priority order), with status:
  1. PROFILE first: forward vs adjoint time, FFT time, Born-series
     iterations per source, L-BFGS function evaluations, error per
     iteration. NOT DONE; do first.
  2. Adaptive per-stage stopping (residual reduction, gradient norm,
     update size) instead of fixed 15/15/15/8. NOT DONE.
  3. Correct warm start: reuse fields but validate against the updated
     operator and require objective/gradient agreement with cold
     solves. Run -16 found the stale-field failure. Fix idea: stop on a
     residual relative to the residual at the warm guess. My estimate:
     ~1.5×.
  4. Per-source Born-series cost: the vmap batch waits for the slowest
     source (seen in run -15). Measure the per-source iteration
     distribution with PML 64/α 0.5; group or terminate independently.
     NOT DONE.
  5. Simultaneous-source encoding (random phase/sign, redrawn per
     update, finish with full sources). Largest potential gain;
     discussed, NOT DONE. Needs a stochastic optimizer (L-BFGS line
     search breaks with changing encodings).
  6. Frequency schedule / iteration allocation: keep 0.15 MHz (dropping
     it failed in run -20); test fewer iterations at 0.15/0.3 and
     omitting 0.3 or 0.5. PARTLY (run -22: final 15→8 OK with ROI).
  7. Noise-aware stopping: stop when the residual reaches the noise
     level. Run -13 showed model error raises the floor above pure
     noise, so a total-error estimate is needed. NOT DONE.
  8. Better Helmholtz preconditioning / alternative solvers. PARTLY: the
     PML retune (run -15) gave ~8× fewer Born-series iterations at
     higher accuracy.
  9. Refine multires + ROI (smaller intermediate grid; re-test the 8 mm /
     10 m/s ROI rule on held-out phantoms). DONE in base form (runs
     -21/-22); held-out check NOT DONE.
  10. Joint multi-frequency refinement at the end (accuracy, not speed).
      NOT DONE.
  11. Fast direct initializer (diffraction tomography / backpropagation).
      Low priority: the low-frequency stages are already cheap (run -21:
      ~12 s at 0.15 MHz).
  12. Sparse/multiscale parameterization. NOT DONE.
  13. Repeated-firing averaging (see the previous planning note). NOT
      DONE.
  14. Geometry/frequency selection via noise-weighted singular values;
      do not drop sources just for speed (run -20). NOT DONE.
  15. Density + attenuation in the inversion (accuracy; costs more).
      Attenuation solver exists (`cbs_ext.py`, run -18; naive joint
      inversion failed); density NOT DONE and REQUIRED for any chest
      inversion.
  16. Learned numerical acceleration (initial fields, preconditioners)
      only after conventional optimization, always physics-verified.
  17. Validate every change on harder independent phantoms (RMSE, Dice,
      boundary displacement, runtime) vs the unmodified baseline.
      MANDATORY.
- Explicitly NOT to do: hard-threshold weak data; raise resolution
  without an accuracy goal; unrestricted image-enhancement networks;
  reuse L-BFGS history across grids blindly; learn replacement physics.
- Order for the next GPU session: profile (1) → adaptive stopping and
  iteration allocation (2, 6) → per-source batching and a correct warm
  start (4, 3) → source encoding (5) → independent benchmark (17). The
  water-bath SNR sweep from the previous note can run alongside (cheap).

### Run 2026-10-26-25 — Roadmap step 1 (PROFILE) + water-bath SNR sweep: wave solves are 87% of runtime; batch waste only 1–2%; quality PLATEAUS at ~4.7σ above 40 dB (noise-limited only below ~30 dB)
- Phase: 3 → nonlinear. NEW Google account (fresh Colab, T4), so the old
  Drive was not reachable; the session was rebuilt from the REPO (solver
  files written into the notebook, patient023 slice from the 2,019-char
  RLE). The time-domain control data were regenerated (120 s) and the
  water fields recomputed. Reproduction check: the 30 dB baseline gives
  EXACTLY run -22's metrics (4.25σ / gap 7.98 / RV 2.96 / outer 19.34 /
  water 1.64), so the rebuild is faithful.
- Profile of the run -22 "J" configuration (multires 64/128/216/260,
  blind ROI from 0.5 MHz, 15/15/15/8 iterations, cold Born-series solves
  with PML 64/α 0.5, TV 1e-6), 154 s total (143 s in run -22; the
  difference is the profiling sync points):
  - forward solves 68.9 s (45%); adjoint 64.4 s (42%);
    setup/compile 13.7 s (9%); gradient/TV/misc 5.9 s; scipy 0.9 s.
  - per stage, time / Born iterations per solve / batch waste:
    - 0.15 MHz @64²: 9.8 s / ~51 / 1%
    - 0.3 MHz @128²: 17.1 s / ~76 / 2%
    - 0.5 MHz @216²: 54.3 s / ~112 / 2%
    - 0.8 MHz @260²: 72.6 s / ~131 / 2%
  - 57 function evaluations for 53 iterations (almost no line-search
    overhead).
- Roadmap consequences: item 4 (per-source batching) DROPPED, since
  the vmapped batch wastes only 1–2% waiting for the slowest source.
  Wave solves are 87% of the time, so the levers are FEWER solves
  (adaptive stopping on the 0.5/0.8 stages, which are 82% of the time;
  source encoding) or CHEAPER solves (looser adjoint tolerance, better
  preconditioning). Compilation (14 s) is paid per reconstruction and
  could be amortized over a slice stack.
- SNR sweep (same phantom, geometry, method and seed; noise added to the
  calibrated time-domain data), LV–myo σ / water std / time:
  - 20 dB: 1.98 / 4.57 / 179 s
  - 30 dB: 4.25 / 1.64 / 154 s
  - 40 dB: 4.65 / 1.56 / 145 s
  - 50 dB: 4.69 / 1.60 / 145 s
  - noise-free: 4.68 / 1.63 / 145 s
  Gap stays 7.97–8.28 (true 8).
- Interpretation: in the WATER BATH the reconstruction is noise-limited
  only below ~30 dB. Above 40 dB it PLATEAUS at ~4.7σ, limited by
  resolution, regularization, the iteration budget or model (not
  noise). So in water, repeated firing beyond +10 dB from 30 dB buys
  only ~10%. In the CHEST (run -24: inner-wall signal −17 dB relative to
  water), 30 dB acquisition behaves like ~13 dB effective, i.e. deep in
  the noise-limited regime (worse than the 20 dB row), so averaging
  (~+17 dB, ~50 repeats) would be the right lever there. That
  prediction still needs the density-capable solver to test.
- Not saved off-Colab (new account, no Drive mounted); all numbers are
  recorded here.
- Gate passed? N/A (Gate 2 waived).
- Next (roadmap order): adaptive stopping on the 0.5/0.8 MHz stages;
  looser adjoint tolerance tested against gradient accuracy; then source
  encoding.

### Run 2026-10-26-26 — Roadmap steps 2/6 (iteration allocation, adaptive stopping) + looser adjoint tolerance: NO free speed; quality tracks iterations; the Born series has a ~100-iteration propagation floor
- Phase: 3 → nonlinear. Same session/data as run -25. Fixed: 30 dB,
  noise seed 21, run -22 "J" pipeline. Only 0.5/0.8 MHz stage settings
  or the adjoint tolerance vary. Colab cells 8–9; results in
  /content/session3_alloc.json (Colab disk only).
- Results (LV–myo σ / RV σ / outer RMSE / time / evals):
  - A baseline 15/15/15/8: 4.25 / 2.96 / 19.34 / 146 s / 57
  - B 0.5/0.8 at 10/5: 3.76 / 2.48 / 20.07 / 110 s / 49
  - C 0.5/0.8 at 6/4: 3.24 / 1.98 / 21.33 / 90 s / 44
  - D adaptive (stop at <2% decrease over 2 its): 4.04 / 2.78 / 19.57 /
    133 s / 53 (stopped 0.5 MHz at 11)
  - E adaptive 5%: 3.59 / 2.40 / 20.12 / 112 s / 47
  - F adjoint tol 3e-3: identical to A / 147 s
  - G adjoint tol 1e-2: identical to A (all metrics to 2 dp) / 141 s
- Check (cell 9; 0.8 MHz, 260², true model, receiver-only adjoint
  source), Born iterations / field error vs tol 1e-4:
  - tol 1e-4: 173
  - tol 1e-3: 141 / 0.15%
  - tol 3e-3: 124 / 0.44%
  - tol 1e-2: 114 / 0.70%
  - tol 3e-2: 100 / 2.45%
  So the tolerance works, but iterations fall only logarithmically:
  ~100 iterations are a PROPAGATION FLOOR (the series has to carry the
  wave across the domain). Gradient error at 1e-2 does not change
  the result.
- Interpretation: on this problem L-BFGS has no slack. Every iteration
  removed from the 0.5/0.8 stages costs inner-wall separation roughly
  proportionally (about −0.1σ per second saved). Adaptive stopping only
  moves along that same trade-off curve. The adjoint tolerance can be
  loosened to 1e-2 for free (≈4% of time). Iteration-count and tolerance
  tuning are therefore EXHAUSTED as speed levers; the remaining
  levers are structural:
  (a) source encoding (fewer solves per iteration; needs a stochastic
      optimizer, since crosstalk changes every iteration);
  (b) a faster per-solve solver (preconditioned Krylov or a better
      CBS shift; the ~100-iteration floor is the CBS's, not the
      problem's);
  (c) amortized compilation over a slice stack (−14 s/slice).
  Keep adj_tol = 1e-2 as the new default.
- Gate passed? N/A (Gate 2 waived).
- Next: source-encoding test (random-phase supershots, Adam/SGD with
  re-randomized codes) versus the A baseline at matched wall-clock time.

### Run 2026-10-26-27 — RECEIVER COUNT 128 → 256 → 512 (32 sources fixed): +30% inner-wall separation at ZERO extra reconstruction cost; the "~4.7σ plateau" was partly receiver under-sampling
- Phase: 3 → nonlinear. Question from the user: if receivers cost
  nothing in compute, why not use more?
- Theory: a receiver only samples the field, and the adjoint injects all
  receiver residuals into one solve per source, so the solve count is
  set by the 32 sources only. The scattered field's angular bandwidth
  is ~2ka ≈ 300 samples (heart a ≈ 45 mm, k = 3.3/mm at 0.8 MHz), so
  128 receivers under-sample above ~0.3 MHz.
- Config: same patient023 phantom, ring r = 56 mm, 32 sources. The
  independent time-domain data were regenerated for each receiver count
  (≈140 s each, water + speed-only control). Noise seed 21 at the SAME
  per-channel SNR, so more receivers means more measurements, as in
  real hardware. Run -22 J pipeline with adj_tol 1e-2 (run -26 default).
  Distinct receiver pixels per grid:
  - 256: 176/256/256/256 at 64/128/216/260
  - 512: 192/360/504/512 (coarse grids merge duplicates)
  Colab cell 10; /content/session3_nrec.json (Colab disk only).
- Results (LV–myo σ / RV–myo σ / outer RMSE / water std / recon time):
  - 128 @30 dB: 4.25 / 2.96 / 19.34 / 1.64 / 141 s
  - 128 noise-free: 4.68 / 2.99 / 19.38 / 1.63 / 145 s
  - 256 @30 dB: 5.00 / 3.30 / 19.06 / 1.83 / 142 s
  - 256 noise-free: 5.29 / 3.38 / 19.05 / 1.45 / 140 s
  - 512 @30 dB: 5.52 / 3.65 / 18.76 / 1.30 / 146 s
  - 512 noise-free: 5.71 / 3.65 / 18.74 / 1.22 / 145 s
  Gap stays 7.97–8.03 (true 8).
- Interpretation: the NOISE-FREE quality rises (4.68 → 5.71σ), so the
  gain is genuine angular-sampling information, not only √N averaging.
  The run -25 "plateau" was partly under-sampling at 128. At 512,
  30 dB is within 0.2σ of noise-free (averaging plus sampling both
  saturating). Reconstruction time is unchanged. The cost is
  hardware: 512 channels, data rate; full-ring USCT systems already
  use ~1–2k elements. For the chest (−17 dB inner-wall signal),
  receiver count is also a free √N SNR lever (512/128 = +6 dB).
- Gate passed? N/A (Gate 2 waived).
- Next: adopt 512 receivers as the default going forward; test 1024
  later (diminishing, ~2ka ≈ 300). Source-encoding test (approved) next.

### Run 2026-10-26-28 — SOURCE ENCODING (random-phase supershots): NO speedup at 32 sources; lands on the same quality-vs-compute curve as plain iteration cuts
- Phase: 3 → nonlinear. Same 30 dB / 128-receiver / seed-21 data as
  runs -25/-26.
- Method: K supershots S_k = Σ_s e^{iφ_ks} S_s, with the data encoded
  with the same codes. This is valid because every receiver records
  every source. Expected misfit is normalized by K. Codes are drawn
  from rng seed 7 and redrawn every R = 3 L-BFGS-B iterations (the
  optimizer is restarted on each draw). The solver normalization is
  measured per encoded source. Iterations per stage are multiplied
  by "x" to roughly match compute. Colab cell 11;
  /content/session3_enc.json (Colab disk only).
- Solve cost scales linearly with batch: at 260², batch 32/8/4 =
  3.84/1.13/0.56 s. Encoding does cut per-iteration cost proportionally.
- Results (LV–myo σ / RV σ / outer RMSE / water std / time /
  full-source-equivalent evals):
  - baseline (run -26 G): 4.25 / 2.96 / 19.34 / 1.64 / 141 s / 57
  - K8 R3 x4: 4.29 / 3.29 / 18.91 / 2.16 / 221 s / 71
  - K4 R3 x8: 4.72 / 3.47 / 18.99 / 3.76 / 245 s / 71
  - K8 R3 x2 (half): 3.68 / 2.72 / 19.75 / 2.34 / 122 s / 36
  - K8 x4 + 2 full its: 4.19 / 3.20 / 18.99 / 2.18 / 207 s / 109
- Interpretation: NEGATIVE as a speed lever here.
  - "Time-matched" runs cost 1.5–1.7× MORE wall-clock. Each code
    redraw restarts L-BFGS, which needs extra line-search evals,
    and small batches use the GPU less efficiently on the coarse
    grids.
  - Quality per unit compute equals plain iteration cuts: K8 x2 gives
    3.68σ at 122 s versus run -26 B 3.76σ at 110 s.
  - Crosstalk shows up as water-background noise (std 2.2–3.8 vs 1.6).
  - K4 x8 raises LV–myo separation but doubles background noise, so
    it is not a clean win.
  - Consistent with run -20 (no source redundancy at 32): with only 32
    sources, each iteration needs all of their information.
  - Encoding is a lever for HUNDREDS of transmitters (where solves per
    iteration dominate), not for 32.
- Optimization status after runs -25..-28: the 32-source, 143 s
  pipeline is near the floor of THIS algorithm on a T4. Iterations,
  tolerances, batching and encoding are all exhausted. The
  remaining speed levers are solver-level (break the ~100-iteration
  Born-series floor: preconditioned Krylov / better shift) and hardware
  (A100/H100). The only free QUALITY lever found is receiver count
  (run -27: 512 receivers 5.52σ, same time).
- Gate passed? N/A (Gate 2 waived).
- Next: adopt 512 receivers as default; solver-level speedup
  (preconditioned GMRES/BiCGSTAB with a correct PML, checked against
  exact H0 as in run -15) OR proceed to the density-capable solver
  needed for the chest.

### Run 2026-10-26-29 — DENSITY-capable Born series (`src/cbs_full.py`, Liouville transform): validated on the heart-in-water case; captures most of the density effect at 0.3–0.8 MHz with no smoothing best
- Phase: 3 → nonlinear (option 2 chosen by the user: density-capable
  solver before any chest inversion).
- Method: p = √ρ ψ turns ρ∇·(ρ⁻¹∇p) + k²p = −s into
  ∇²ψ + (k² + q)ψ = −s/√ρ, with q = ½∇²lnρ − ¼|∇lnρ|². This is a scalar
  Helmholtz equation, so the convergent Born series is unchanged with
  k² → k² + q (plus attenuation, as in cbs_ext). Sources and receivers
  are in uniform water, so √ρ is a constant that cancels in the
  water-shot calibration. ln ρ is optionally Gaussian-smoothed (σ cells)
  before spectral differentiation. The gradient with respect to c is
  unchanged (the operator stays complex-symmetric).
- V0: ρ = None reproduces the fast solver exactly (rel diff 0.0, 141
  iterations, 0.8 MHz 260²).
- V1 setup: patient023 heart in water, cited densities (blood/myo
  1060, water 994; Mast 2000, Duck 1990). Compared with the independent
  jWave time-domain solver on the same 520² grid (time-domain data
  59 s). Error = ‖R_CBS − R_TD‖ / ‖R_TD − 1‖ on water-calibrated ratios,
  i.e. relative to the SCATTERED signal (stringent). The density effect
  itself in the time domain is 2.0–3.0% of the scattered signal.
- V1 results (error / max Born iterations, 32 sources):
  - 0.15 MHz: no density 12.6% / 508; ρ σ=0 12.5% / 727; σ=1 12.4%
    (pre-existing low-frequency CBS–TD mismatch independent of
    density; weak pulse energy at 0.15 MHz)
  - 0.3 MHz: no density 2.6%; ρ σ=0 1.5%; σ=0.5 1.5%; σ=1 1.6%;
    σ=2 1.8%
  - 0.5 MHz: no density 3.4%; ρ σ=0 1.0%; σ=0.5 1.2%; σ=1 1.6%;
    σ=2 2.5%
  - 0.8 MHz: no density 4.2%; ρ σ=0 3.0%; σ=0.5 3.1%; σ=1 3.5%;
    σ=2 3.9%
  Iteration cost with density: +5–18% (more at 0.15 MHz).
- Interpretation: the density term removes about 40–70% of the
  speed-only model error at 0.3–0.8 MHz; no smoothing (σ=0) is best.
  The remaining 1–3% residual is at the level of the known CBS–TD
  grid-discretization differences, i.e. below 30 dB noise. Weak-contrast
  density (6%) is VALIDATED. The chest (lung 300 vs 1050, bone 1904,
  i.e. large ln ρ jumps) is NOT yet validated and is the next check
  (V2).
- Gate passed? N/A (Gate 2 waived).
- Next: V2 chest validation (density + attenuation), interleaved with
  the user's new MRI-traced heart phantom request.

### Run 2026-10-26-30 — REALISTIC MRI-traced heart (trabeculae, papillary muscles, valve/aortic-root walls, dim pericardial/fat lining, myocardial texture): blind speed-only reconstruction recovers region means to 2–4 m/s, the texture (r = 0.84 in thick myocardium) and the fat lining partially; boundary error higher than the simple slice
- Phase: 3 → nonlinear. User request: a more intricate, realistic
  phantom traced from a user-supplied ex-vivo cardiac MRI (long-axis,
  four-chamber-like view), with a less-bright pericardium and tissue
  irregularities. User asked to use "Hounsfield for density". NOT done:
  MRI intensity is not HU, and no cited calibration maps intensity to
  density or speed. Intensity was used for GEOMETRY only.
- Phantom (`src/phase3_mri_heart_phantom.py`; outputs gitignored:
  `results/phase3_mri_heart_phantom_patient.npz`,
  `results/figures/phase3_mri_heart_phantom_patient.png`):
  - median 3×3; bright tissue > 95 → myocardium (specks < 300 px
    dropped)
  - dim tissue 66–95, ≥ 2 px thick, touching bright tissue → label 3
    (pericardium / epicardial-fat proxy)
  - thin faint walls (≥ 60 px long) → myocardium
  - chambers = dark regions enclosed after a radius-12 px closing →
    blood; the rest → water
  - ASSUMED scale: tissue bounding box 110 mm (0.199 mm per source
    pixel); 386×440 at 0.25 mm; max radius 64 mm
  - KNOWN LIMITATION (accepted by the user): the upper atrial roof is
    a 1–2 px faint wall not fully traced, so that chamber is partly
    water
- Cited values:
  - water 1520 / 994 (Duck 1990)
  - myocardium 1576 / 1060 / 0.52 dB/cm/MHz, blood 1584 / 1060 / 0.20
    (Mast 2000)
  - NEW `tissue.FAT_PERICARDIUM_PROXY` 1464.5 m/s / 943.5 kg/m³ /
    0.07 cm⁻¹ at 1 MHz, from a CRC Press textbook chapter
    "Acoustical Properties of Biological Tissue", Table 5.1 (porcine
    fat: 1479/1450 m/s, 0.937/0.95 g/cm³; book title not confirmed from
    the publisher's sample file). Mast 2000 itself could not be fetched
    (author page down).
  - Myocardial speed TEXTURE: Gaussian random field, std 4.7 m/s (the
    same table's bovine heart muscle 1546 ± 4.7 m/s spread),
    correlation length 2 mm ASSUMED, rng seed 31.
  - Attenuation ∝ f (power law 1 ASSUMED), set per frequency in the
    time-domain runs.
- Geometry: ring radius 75 mm, 32 sources, 512 receivers; domain
  170 mm; time-domain data on a 680² grid (0.25 mm), Nt 2,957,
  speed + DENSITY + ATTENUATION + texture. Water reference plus one
  object run per frequency took 705 s. Reconstruction: blind,
  SPEED-ONLY Born series (density and attenuation not modelled),
  multires 84/168/280/340 (0.5 mm final), ROI, 15/15/15/8 iterations,
  TV 1e-6, adj_tol 1e-2, 30 dB, noise seed 21. Colab cells 14–15
  (cell 15 run by the user in the browser after an MCP drop).
- Result (441 s on T4), reconstructed vs true:
  - region means: myocardium 1573.3 (true 1576.5); blood 1580.1
    (1584.0); water 1519.2 (1520.0)
  - blood–myocardium gap on eroded cores 7.41 (true 7.46)
  - fat lining 1499.1 (true 1464.5): contrast vs myocardium −76
    (true −112), i.e. ~68% recovered for a thin (~1–3 mm) layer
  - separation blood vs myocardium 1.35σ; the TRUE map itself only
    reaches 2.24σ, because the cited texture spreads the myocardium
  - thick myocardial core (eroded 3 px): 1575.7 ± 4.0, and the
    reconstruction CORRELATES with the true texture at r = 0.844
  - outer-band RMSE 24.5 (patient023 simple slice: 18.8–19.3);
    whole-heart RMSE 14.1; water std 3.4 (vs 1.3–1.6)
- Interpretation:
  - On a realistic, intricate heart, with density, attenuation and
    texture present in the data but NOT in the inversion model, the
    blind reconstruction gets the tissue classes' speeds right to 2–4
    m/s and resolves ~mm-scale texture in thick myocardium. Thin
    structures (fat lining, trabeculae, thin atrial and valve walls) are
    blurred: contrast drops and the boundary error rises.
  - The higher water noise and edge error are consistent with the
    unmodelled density jumps (run -21 saw density artifacts at outer
    boundaries) plus the many more edges. The density-capable solver
    (run -29) now exists and is the natural fix to try: invert speed
    with density in the model (oracle test), then blind
    speed + density.
  - Caveats: assumed scale and tissue identity of the dim layer; a 2D
    slice; water bath (no chest); one noise realization.
- Gate passed? N/A (Gate 2 waived).
- Figure: `results/figures/phase3_mri_heart_recon_30db_512rec.png` (hidden /
  reconstruction / error; gitignored via *mri_heart*).
- Next: (a) same data, inversion WITH the density-capable forward model
  (true density = oracle bound) to measure how much of the edge error
  is density mis-modelling; (b) V2 chest validation of the density
  solver; (c) then the chest inversion.

### Run 2026-10-26-31 — BLIND JOINT speed + density on the MRI heart: density did NOT move (stayed at the water value everywhere); speed result unchanged; inconclusive about density identifiability (optimizer scaling)
- Phase: 3 → nonlinear. User request: try the blind density
  reconstruction; if it beats the speed-only baseline, go to the chest
  (V2 validation + chest inversion); if it costs disproportionate
  time, assess clinically allowed error margins.
- Method (Colab cells 17, 20; cell 20 run by the user in the browser):
  - forward = Liouville Born series (`cbs_full`, σ = 0)
  - unknowns: m = (c − 1520)/100 and d = ln(ρ/994)/0.05
  - d is free only inside r < ring_r − 8 mm and in the ROI from
    0.5 MHz
  - dJ/dk² = Re(ΣΛP)/K as before; dJ/d ln ρ = vjp of q(ln ρ) applied
    to dJ/dk²
  - TV on both maps (same λ); same multires / iterations / 30 dB data
    / seed as run -30
- Gradient check (cells 18–19, 0.3 MHz @168², central differences):
  - speed adjoint/FD ratio 1.034, 0.993, 0.969
  - density adjoint/FD ratio 0.919, 0.894, 0.939 over 3 directions:
    identical across step sizes 0.02/0.005/0.00125 and at solver tol
    1e-4/1e-5, so neither finite-difference nonlinearity nor solve
    truncation
  - jWave's unit normalization was checked (k_norm = k·dx, so
    q_norm = q·dx² is consistent)
  - Cause of the ~8% under-scaling still UNRESOLVED. The direction
    is right, so it did not block the run.
- Result (231 s; baseline 441 s, but the timings are not comparable
  because of compile caching within the session):
  - JOINT: blood–myocardium separation 1.33σ, gap 7.51, fat 1500.6
    (contrast −74), outer RMSE 24.83, heart RMSE 14.25, water std
    3.41, texture r = 0.825
  - SPEED-ONLY: 1.35σ, gap 7.41, fat 1499.1 (−76), outer RMSE 24.54,
    heart RMSE 14.05, water std 3.43, texture r = 0.844
  - Reconstructed DENSITY: 994.0 ± 0.1 in EVERY region (true 1060
    myocardium/blood, 943.5 fat); density RMSE in the heart 63 kg/m³
    = no information recovered
- Interpretation: NOT an improvement over speed-only. The density
  unknowns effectively never left their starting value. In this
  scaling the density gradient (an edge-only effect via ∇² ln ρ) is
  orders of magnitude smaller than the speed gradient, so L-BFGS
  never steps in d. This shows the CURRENT setup recovers no density.
  It does NOT show density is unidentifiable: a fair test needs
  gradient balancing / preconditioning between the two maps (or a
  density-only pass with speed fixed), plus the 8% gradient
  discrepancy resolved.
- Figure: `results/figures/phase3_mri_heart_joint_speed_density_30db.png`
  (gitignored via *mri_heart*).
- Gate passed? N/A (Gate 2 waived).

### Run 2026-10-26-32 — CLINICAL TRANSLATION of the speed-only MRI-heart reconstruction (run -30): outer heart boundary within CMR-level reproducibility; blood-pool area at echo level
- Phase: 3 → evaluation. Branch of the user's rule: density-accounting
  gave no gain, so assess the speed-only error against clinically
  allowed margins. Colab summary cell. 0.5 mm pixels; 2D slice areas as
  proxies for volumes.
- Pipeline (applied identically to the TRUE map as a ceiling):
  - outer heart = largest component of (c > 1548, opened, filled)
  - blood pool = Gaussian(σ 2 px)-smoothed c > 1580, inside the heart
  - metrics: Dice, symmetric mean / 95th-percentile contour distance,
    area error
- Results, reconstruction vs true map through the same pipeline:
  - OUTER boundary (epicardium/heart surface): Dice 0.988; contour
    distance mean 0.41 mm, 95th percentile 1.12 mm; area −0.97%
    (true-map ceiling: 1.0 / 0 / 0)
  - BLOOD POOL: Dice 0.841 (ceiling 0.853); contour distance 3.1 /
    12.7 mm (ceiling 2.6 / 10.5); area −6.0% (ceiling +2.5%)
- Clinical reference: interstudy reproducibility coefficient of
  variation, Grothues et al. 2002 Am J Cardiol 90:29–34
  (doi:10.1016/s0002-9149(02)02381-0, via PubMed):
  - CMR: EDV 2.9–4.9%, ESV 4.4–9.2%, EF 2.4–7.3%, mass 2.8–4.8%
  - 2D echo: EDV 5.5–10.5%, ESV 13.7–20.3%, EF 8.6–19.4%, mass
    11.6–15.7%
- Assessment (water bath, static 2D slice, one noise realization):
  - Outer geometry: a −1% area error with sub-mm boundary placement
    is inside CMR's own test–retest spread for mass/EDV.
  - Blood pool: −6% area (≈ −8.5% relative to the pipeline's own
    ceiling) is outside CMR's EDV range and comparable to 2D echo's
    EDV variability.
  - The blood-pool limit is largely INTRINSIC to speed-only contrast:
    blood vs myocardium differ by only 8 m/s, and the cited myocardial
    texture (±4.7 m/s) is comparable. Even the TRUE speed map
    thresholds to Dice 0.85 and a 10.5 mm 95th-percentile contour
    distance. Better endocardial delineation needs another contrast
    channel (attenuation, backscatter/reflection), not a better speed
    inversion.
- Caveats: 2D area is not 3D volume; EF needs two phases (ED/ES) and
  this is one static frame; water bath (the chest, run -24, removes
  ~86–88% of the inner-wall signal); assumed phantom scale; simple
  threshold segmentation.
- Gate passed? N/A (Gate 2 waived).
- Next (user decision): (a) a balanced-scaling density retry
  (~4–5 min) before concluding on density; or (b) the chest: V2
  validation of the density solver with lung/bone, then a chest
  inversion. The chest will need density/attenuation in the FORWARD
  model at least for lung/bone, since their contrasts (ρ 300/1904,
  c 658/2471) are far beyond what speed-only modelling can absorb.

### Run 2026-10-26-33 — STEP 3 (chest V2) FAILED → traced to two jWave Born-series defaults that break for SLOW inclusions (lung): source scaled by the grid's MINIMUM speed, absorbing layer tuned to the MAXIMUM wavenumber. Exact-cylinder (Mie) test established; time-domain solver validated for lung/bone
- Phase: 3 → forward-model validation (user: "move to 3 and 4").
- V2 setup (Colab cells 21a, 21):
  - CT chest phantom (run -23), transferred as lzma+base64 in 6
    md5-checked chunks (an earlier hand-copied zlib string was
    corrupted)
  - 800² at 0.5 mm, ring 185 mm, 8 of 32 sources, 512 receivers
  - time-domain data (pulse σ 0.5 µs, Nt 4,768, dt 63 ns): water
    73 s, chest 72 s per frequency
  - cited values as in run -24; bone attenuation as a power law
    through Pichardo's two points (33 Np/m at 270 kHz, 240 at 836 kHz,
    n = 1.756)
- V2 result (Born series vs time domain, error relative to the
  scattered signal / iterations):
  - 0.15 MHz: density+attenuation 1.95 / 8000 (cap); attenuation only
    2.04 / 2176; density only 2.16 / 8000 (cap)
  - 0.3 MHz: 2.12 / 8000 (cap); 2.10 / 3413; 2.70 / 8000 (cap)
  - water: 364–404 iterations
  - → >100% disagreement, and density solves not converging.
- Ground truth (cell 22b): exact series for a fluid cylinder
  (radius 25 mm, line source; p and (1/ρ)∂p/∂r continuous;
  n = ±100, unchanged vs ±140), 320² at 0.5 mm, ring 70 mm, 256
  receivers. Receivers within 2 mm of the source are excluded (an
  earlier pass gave NaN from the singular incident field at
  coincident pixels). Error vs exact (relative to the scattered
  signal), time domain / Born series:
  - blood-like 1584/1060: 0.15 MHz 0.008 / 0.014; 0.3 MHz 0.014 /
    0.010 (validates the exact-solution code)
  - lung speed only 658/994: 0.186 / 3.22; 0.207 / 3.95
  - lung-like 658/300 + 37.3 dB/cm/MHz: 0.023 / 3.59 (12,057 its);
    0.051 / 4.04
  - bone-like 2384/1904 + Pichardo attenuation: 0.042 / 0.052; 0.189 /
    0.078
  → The TIME-DOMAIN solver is right for lung and bone (2–5% lung,
  4–19% bone), so run -24's chest data stand. The BORN SERIES fails
  only for SLOW inclusions.
- Bug 1 (absorbing layer): jWave's `_cbs_pml` uses
  k = max(ω/c) over the grid. With lung present that is 2.3× the
  water wavenumber at the boundary, so the layer is mismatched. Patch:
  use the boundary (corner) speed. Lung error 3.6–4.0 → 1.9–2.1.
  Not sufficient.
- Bug 2 (source scaling, the main one): jWave's
  `scale_source_helmholtz` multiplies the source by 2/(dx · min c)
  over the WHOLE grid. Any medium slower than water rescales the
  field relative to the water reference by c_water/c_min (lung: 2.31).
  Measured best global factor 0.431+0.009j vs predicted 658/1520 =
  0.433. After removing it, lung error 2.12 → 0.092 (time domain
  0.051). Per-source error identical (2.12 for all 8), i.e. purely
  global. Tolerance 1e-5 (20,000 iterations) gave the same result, so
  not convergence.
  - Patch (local `src/cbs_full.py`, `cbs_warm.py`, `cbs_ext.py`):
    scale by the boundary (water-bath) speed. Identical to jWave when
    nothing is slower than water.
- IMPACT ON EARLIER RESULTS:
  - patient023 heart runs (-13…-28): tissues ≥ 1576 m/s, so only
    affected when the reconstructed model dipped below 1520 m/s
    (noise / edge undershoot; ~1% amplitude scaling at most).
    Minor, but the forward model was non-smooth there (a hidden
    dependence on the single slowest pixel that the gradient ignores).
  - MRI heart (runs -30/-31): the hidden fat proxy is 1464.5 m/s
    and reconstructed fat pixels go below 1520, so the forward model
    was rescaled by up to ~3–4%. This likely contributed to the
    higher water noise (3.4 vs 1.6) and boundary error. Those runs
    should be REPEATED with the patched solver.
  - The ~8% density-gradient mismatch (run -31) was measured on the
    MRI phantom WITH fat (min speed 1464.5), but with c held fixed
    during the FD, so this bug should not explain it. Still open.
- Colab runtime was RESET right after these tests (all /content state
  lost; fresh T4). The patches had not yet been re-tested in Colab;
  the PML-matched vs jWave-PML comparison with the source fix is
  pending.
- Gate passed? N/A (Gate 2 waived). Forward model for the chest: NOT
  yet validated (pending the patched rerun).
- Next: rebuild the session from repo files; rerun the cylinder test
  with both patches (and the source fix alone); rerun V2 on the chest;
  then step 4.

### Run 2026-10-26-34 — Patched Born series re-validated: BOTH fixes needed; chest error vs time domain 195–212% → 8.5–16%, but 20–40× the water iteration count makes a chest inversion with this solver infeasible on a T4
- Phase: 3 → forward-model validation (continuation of run -33 after
  the runtime reset). Session rebuilt from repo files (cells 1–4, 12)
  with both patches applied by checked string replacement (cell 25);
  Colab copies match `src/cbs_full.py` / `cbs_warm.py`.
- Exact-cylinder test, error vs exact relative to the scattered signal
  (0.15 / 0.3 MHz), all with the source-scaling fix:
  - PML matched to water:
    - blood-like 0.014 / 0.010
    - lung speed only 0.131 / 0.334 (time domain 0.186 / 0.207)
    - lung-like 0.108 / 0.097 (time domain 0.023 / 0.051)
    - bone-like 0.052 / 0.078 (time domain 0.042 / 0.189)
  - jWave PML (max k):
    - blood-like 0.014 / 0.010
    - lung speed only 1.29 / 1.55
    - lung-like 1.40 / 1.57
    - bone-like 0.052 / 0.078
  → Both bugs are real and independent. Lung iterations:
  1,756–7,849 (vs 160–190 blood-like).
- Chest V2 rerun (cell 26; 8 sources, 512 receivers,
  density + attenuation; time-domain data regenerated in 217 s and saved
  to /content/chest_td_8src_patient.npz):
  - 0.15 MHz: error vs scattered 0.085, vs total 0.102; 15,000
    iterations (CAP, not fully converged; water 404); 284 s per 8 sources
  - 0.3 MHz: 0.162 / 0.195; 7,792 iterations (water 364); 149 s
- Assessment:
  - Accuracy: 8.5–16% of the chest-scattered field. Run -24 measured
    the inner-wall signal through the chest at ~0.16–0.23% of the
    data, so this model error is ~50–100× the inner-wall signal: a
    speed inversion through the chest with this forward model would
    fit lung/rib model error, not the inner wall.
  - Cost: one 32-source solve ≈ 10–19 min, one gradient = 2 solves,
    a reconstruction ~50 gradients → ~17–30 h on a T4. INFEASIBLE in
    this setting.
  - The convergent Born series converges at a rate that falls with the
    contrast, and lung (c 658, ~5× k²) is the worst case. This is a
    property of the method, not a bug.
- Gate passed? N/A (Gate 2 waived). Step 3 outcome: forward model
  FIXED and validated to 8–16% in the chest, but not accurate or fast
  enough for step 4 as originally designed.
- Next (user decision): step 4 needs a different strategy (see report):
  oracle background + distorted-Born (linearized) inversion for the
  heart using background Green's functions from the accurate
  time-domain solver, OR time-domain FWI. Also: repeat the MRI-heart
  runs (-30/-31) with the patched solver.

### Run 2026-10-26-35 — MRI heart RERUN with the fixed Born series (source scaling + matched PML in cbs_warm): every metric improves; water noise −37%; blood-pool area −6.0% → −3.6%
- Phase: 3 → nonlinear. Same as run -30 (MRI phantom, time-domain data
  with speed + density + attenuation + texture, 512 receivers, 30 dB,
  seed 21, blind speed-only multires 84/168/280/340), but with BOTH
  fixes from run -33 in the fast solver (`cbs_warm`; PML fix also
  applied there now, locally in `src/cbs_warm.py` and
  `src/cbs_ext.py` too). Time-domain data rebuilt after the runtime
  reset (701 s); reconstruction cell 15 run by the user in the browser
  (MCP link dropped).
- Result (190 s; run -30 441 s, not comparable because of
  compile/caching), fixed vs before:
  - blood–myocardium separation 1.47σ vs 1.35 (true-map ceiling
    2.24); gap 7.33 vs 7.41 (true 7.46)
  - fat 1494.5 vs 1499.1 (true 1464.5): contrast −80.8 vs −75.8
    (true −112, now 72%)
  - myocardium 1575.4 (1576.5), blood 1582.7 (1584.0)
  - outer-band RMSE 23.52 vs 24.54; heart RMSE 13.36 vs 14.05
  - WATER std 2.16 vs 3.43 (−37%)
  - texture r 0.875 vs 0.844
  - clinical pipeline (run -32):
    - outer: Dice 0.989, contour 0.38 / 1.0 mm, area −0.8%
    - blood pool: Dice 0.842, 2.94 / 11.31 mm, area −3.6% (was −6.0%;
      pipeline ceiling on the true map +2.5%)
- Interpretation: the min-speed source-scaling bug (triggered by fat
  pixels below water speed) was a real source of the excess
  background noise and edge error in run -30. With it fixed, the
  blood-pool area error (−3.6%) falls inside CMR's EDV interstudy
  range (2.9–4.9%, Grothues 2002). Measured against the pipeline's
  own ceiling it is still ~−6%, so treat this as at best borderline
  CMR-level. Remaining errors: a thin rim at the outer surface and
  blurred thin fat/trabecular structures.
- Figure: `results/figures/phase3_mri_heart_recon_30db_512rec_FIXED.png`
  (gitignored).
- Run -31 (joint density) was NOT repeated; its density null result
  is likely independent of this bug but should be redone with the
  fixed solver before drawing conclusions.
- Lung frequency question (user: "will another frequency bypass
  this?"), analysis only, no run:
  - The lung loses signal by two mechanisms that move in OPPOSITE
    directions with frequency.
  - (i) Attenuation: 37.3 dB/cm at 1 MHz ∝ f (Dunn 1986 via Miller
    2016; ∝ f below 1 MHz is an extrapolation). That is ~3.7 dB/cm at
    0.1 MHz and ~19 dB/cm at 0.5 MHz, so a few cm of lung blocks
    ≥ 0.5 MHz.
  - (ii) Impedance mismatch (frequency-independent in the bulk model):
    soft tissue 1.66 MRayl vs lung 0.197 MRayl gives
    R = (1.66−0.197)/1.857 = 0.79, i.e. 38% intensity transmitted per
    interface and ~14% (−8.4 dB) in and out. This is close to run
    -24's measured 12–14% surviving inner-wall signal.
  - At LOWER frequencies the lung behaves as a soft effective medium
    with c ≈ 25–70 m/s (excised horse lung, audible range; Rice 1983,
    doi:10.1152/jappl.1983.54.1.304) and ~30 m/s in vivo at 125–500 Hz
    (Kraman 1983, doi:10.1152/jappl.1983.55.6.1862). Impedance then
    drops ~20× further and the mismatch gets WORSE (R → ~0.99), while
    resolution is lost (λ ≫ wall thickness).
  - Conclusion: no frequency bypasses the lung. Higher frequency is
    absorbed; lower frequency is reflected and unresolved; the
    0.1–0.3 MHz band is a weak compromise (attenuation lower, but
    inner-wall resolution only a few mm). The bypass is GEOMETRIC:
    acoustic windows where the heart touches the chest wall without
    lung (parasternal intercostal / cardiac notch, apical, subcostal
    through the liver), expiratory breath-hold, or transesophageal
    access. For the ring concept this means concentrating elements on
    windows, which turns it into limited-view tomography (the k-space
    coverage problem of runs -02/-24).
- Gate passed? N/A (Gate 2 waived).

### Run 2026-10-26-36 — Full metric set on the run -35 reconstruction (no rerun): DSC / IoU / ASSD / HD95 / HD / NSD and speed RMSE / MAE / bias / P95 / PSNR / SSIM
- Phase: 3 → evaluation. CPU-only on the existing reconstruction
  (MRI heart, fixed solver, 30 dB, 512 receivers). 0.5 mm pixels.
- Segmentation by threshold:
  - heart (myocardium + blood) = largest component of c > 1548,
    opened and filled
  - blood = Gaussian(2 px) > 1580 inside the heart
  - myocardium = heart − blood
  - fat = Gaussian(1 px) < 1500 within 6 px outside the heart
  - Scored vs the TRUE LABELS. "Ceiling" = the true speed map through
    the same segmentation, which isolates segmentation-method limits
    from reconstruction error.
- Results, recon [ceiling]: DSC / ASSD mm / HD95 mm / NSD@1mm /
  NSD@2mm / area %
  - heart: 0.985 [0.996] / 0.70 [0.27] / 3.15 [1.80] / 0.882 [0.933]
    / 0.919 [0.955] / −1.7 [−0.9]
  - blood pool: 0.842 [0.853] / 2.94 [2.57] / 11.31 [10.52] / 0.447
    [0.497] / 0.620 [0.664] / −3.6 [+2.5]
  - myocardium: 0.778 [0.792] / 1.68 [1.48] / 6.32 [6.26] / 0.654
    [0.677] / 0.799 [0.811] / −2.8 [−8.6]
  - fat lining: 0.413 [0.492] / 12.6 [7.1] / 40.3 [32.7] / 0.392
    [0.497] / 0.468 [0.558] / −70 [−66]
  - fat segmentation fails even on the true map: the threshold method
    is unsuitable for this thin layer
- Speed error (m/s): RMSE / MAE / bias / P95|err|
  - whole heart (dilated 2 px): 13.36 / 5.71 / −1.04 / 31.0
  - myocardium: 9.84 / 4.60 / −2.64 / 20.4
  - blood: 10.69 / 3.73 / −3.36 / 14.6
  - fat: 39.5 / 30.5 / +30.1 / 81.7 (thin layer under-resolved)
  - water: 3.13 / 1.24 / −0.05 / 4.13
  - thick myocardial core: 2.32 / 1.81 / −0.56 / 4.46
  - ±2 px boundary band: 23.5 / 16.4 / −0.63 / 50.4
  - PSNR (heart, range 125 m/s) 19.4 dB; SSIM (heart crop) 0.821;
    texture r 0.875
- Interpretation:
  - Heart boundary and myocardium track the segmentation ceiling
    closely: reconstruction adds ~0.4 mm ASSD and ~1.3 mm HD95 at
    the outer boundary, and ~0.2 mm ASSD for the myocardium.
  - Blood-pool and myocardium DSC (0.84 / 0.78) are limited mainly by
    speed-only contrast plus naive thresholding (ceiling 0.85 / 0.79),
    not by the reconstruction.
  - Speed error is concentrated at boundaries and thin fat: interior
    tissue is accurate to ~2 m/s (0.15%).
  - These are 2D, single-slice, water-bath, threshold-segmentation
    numbers, not directly comparable to published 3D CMR/echo
    segmentation benchmarks.
- Gate passed? N/A (Gate 2 waived).

### Planning note 2026-10-26 (no run) — Project AIM recorded: patient throughput via rapid acquisition + offline reconstruction
- User-supplied research note saved verbatim as
  `fourier_inverse/RESEARCH_NOTE_throughput_2026-10-26.md`, with annotations against the logged
  evidence.
- Aim: short patient-facing acquisition (idealized ~10–25 ms for 32 broadband firings, ~0.1–0.25 s
  with 10× averaging), patient released, FWI offline (~3 min/slice on a T4). The user sees this as
  the approach's main strength over MRI/CT.
- Annotation highlights:
  - the throughput advantage is clear vs MRI; vs CT the advantage is no radiation plus quantitative
    contrast, not speed
  - 325 receivers was not a tested configuration
  - motion tolerance (~λ/10–λ/20) is untested
  - favourable first targets: breast and extremities
- Implication for priorities: acquisition-time-sensitive experiments become first-class:
  - motion-tolerance sweep
  - fewer transmitters / repeats vs quality
  - reconstruction throughput per GPU-hour
  - a breast or limb phantom

### Run 2026-10-27-37 — Multi-frequency design A1: frequency-ladder cycle-skip margins (CPU, straight-ray proxy, MRI heart): start ≤ ~0.18 MHz from water; after a stage the safe jump is ~×3 at 0.3 MHz and ~×2 at 0.5 MHz; run -20's "0.3→0.8 failure" was the 0.3 MHz WATER START, not the jump
- Phase: 3 → acquisition/inversion design (user goal 2026-10-27: multi-frequency Fourier extraction for patient throughput;
  (a) frequency ladder for feed-forward FWI convergence/compute, (b) acquisition-time saving; tank time is the priority;
  record data up to 2.5 MHz, reconstruct up to 0.8 MHz for now; MRI heart phantom). Seed 42. CPU, 3.7 min.
- Script `src/phase3_multifreq_ladder.py`; figure `results/figures/phase3_multifreq_ladder_mri_heart.png` (gitignored).
- Method: straight-ray travel-time residuals over all 32×512 ring pairs on the MRI heart (cC2, 340², R 75 mm). "Truth seen
  at f" = slowness smoothed to FWHM λ/2; "model after stage f_k" = truth smoothed to FWHM β·λ_k, β CALIBRATED on the real
  run -35 reconstruction (recM residual at 0.8 MHz: |φ|/π p95 0.10, max 0.51 → β = 1.74). Criterion: max |φ| < π.
- Water start (anomaly p95 2.17 µs, max 2.74 µs): max |φ|/π = 0.53 @0.10, 0.81 @0.15, 1.08 @0.20, 1.62 @0.30, 4.4 @0.8 MHz.
  Consistent with run -20 (0.3 MHz start failed, 0.15 worked; that was patient023, so a cross-phantom check, not a
  fit). Run -20 E/F dropped BOTH 0.15 and 0.5, so the 0.3→0.8 jump itself was never tested.
- Safe jump from a stage (max|φ|<π; β 1.74 / pessimistic β 2.61): from 0.15 → ×4.3/×3.8; 0.3 → ×2.95/×2.4;
  0.5 → ×2.2/×1.85; 0.6 → ×2.0/×1.7. Rule of thumb: ratio shrinks as f rises (~×3 low, ~×2 near 0.5 MHz).
- Ladders (worst transition max|φ|/π; est. T4 time at 15 its/stage + 8 final, interpolated from the run -25 profile):
  - 0.15>0.8: 1.24 → SKIP; 0.3>0.8: 1.62 (start) → SKIP
  - 0.15>0.3>0.8: 0.87 on the last jump (tight under pessimistic β) ~123 s
  - **0.15>0.4>0.8: 0.81 (start) / 0.52 / 0.73, ~142 s** — most balanced 3-stage ladder
  - 0.15>0.3>0.5>0.8 (current): 0.81/0.36/0.49/0.63, ~190 s
  - 0.1>0.3>0.8: 0.53/0.45/0.87 ~123 s (better start margin, needs 0.1 MHz energy)
- Caveats: straight rays (no diffraction healing), β calibrated on the final stage only, cost model ignores that a
  bigger last jump may need more 0.8 MHz iterations — the time saving (190 → ~125–140 s) must be confirmed by FWI.
- Gate passed? N/A (Gate 2 waived). Next: run -38 (acquisition), then Colab ladder test.

### Run 2026-10-27-38 — Multi-frequency design A2: frequency isolation and tank time (CPU, exact cylinder transfer functions): frequency-division multiplexed tones isolate every frequency to ≤ −76 dB once the settle time exceeds one ring crossing; tank time 9.6 ms → 0.75–1.5 ms with ~35–40 dB MORE per-frequency SNR than the current pulse
- Phase: 3 → acquisition design. Seed 42. CPU, ~7 min. Script `src/phase3_multifreq_fdm.py`; figure
  `results/figures/phase3_multifreq_fdm_separability.png` (synthetic, committable).
- Forward model: exact 2D line-source + attenuating fluid cylinder (radius 45 mm, cited myocardium c/ρ, 0.52 dB/cm/MHz,
  α ∝ f ASSUMED, no dispersion), ring R 75 mm, 32 sources, 64 receivers; Bessel series (run -02 `exact_coeffs`, Graf
  addition) every 0.5 kHz to 3 MHz → 2 ms impulse responses. Acquisition simulated in time domain, demultiplexed by DFT,
  compared with the exact H. Equal drive peak per element; receiver noise fixed so the CURRENT pulse gives 30 dB at 0.8 MHz.
- Bug found and fixed in this run: band-limited near-receiver responses spread before t=0 and wrapped to the end of the
  periodic impulse response → a constant ~10% error on the receivers next to the source (looked like −20 dB "leakage" in
  every design). Fixed by a 50 µs built-in latency (pre-arrival energy 8.6% → 0.19%); single-tone steady state then
  matches exact H to ~1e-5.
- Results (leakage = crosstalk + transient + previous firing, noise-free; SNR = leakage + noise; dB):
  | design | tank/slice | SNR 0.15 / 0.3 / 0.5 / 0.8 | 1.2 / 1.7 / 2.5 MHz |
  |---|---|---|---|
  | current pulse σ 0.25 µs, 0.14 ms/firing | 4.5 ms | 29.9 / 32.6 / 33.6 / 32.7 (leak −33 at 0.15) | 26 / 12 / −22 |
  | current pulse, 0.30 ms/firing | 9.6 ms | 28.6 / 31.6 / 32.0 / 30.0 | 23 / 9 / −25 |
  | wide pulse σ 0.10 µs, 0.30 ms | 9.6 ms | 13.6 / 16.3 / 18.1 / 19.6 | 20 / 19 / 15 |
  | sequential tones K=1, Tw 0.1 ms | 11.2 ms | 67 / 65 / 63 / 61 (7 tones) | 59 / 57 / 55 |
  | FDM K=8, Tw 0.25 ms | 2.0 ms | 69 / 67 / 65 / 63 | 61 / 60 / 58 |
  | FDM K=16, Tw 0.5 ms | 1.5 ms | 72 / 70 / 68 / 66 | 64 / 63 / 61 |
  | FDM K=32, Tw 1.0 ms | 1.25 ms | 74 / 73 / 71 / 69 | 68 / 66 / 64 |
  | **FDM K=32, Tw 0.5 ms** | **0.75 ms** | 72 / 70 / 68 / 66 | 64 / 63 / 61 |
  | FDM K=32, Tw 1.0 ms, settle only 0.05 ms | 1.15 ms | 32 / 33 / 34 / 35 (LEAK-limited) | 36 / 38 / 40 |
- Findings:
  1. Isolation: with tones on the 1/Tw bin grid and settle ≥ 0.15 ms (> one ring crossing 2R/c = 99 µs), crosstalk
     between simultaneous sources/frequencies is ≤ −76 dB — far below noise. Settle 0.05 ms (< one crossing) → −31 dB.
     The settle time, not K, sets the isolation.
  2. Pulse firing interval 0.14 ms is too short at 0.15 MHz (leak −33 dB, previous firing's tail); 0.3 ms is clean.
  3. Tones concentrate drive energy in the bins actually used: ~+35–40 dB per frequency at equal peak drive. That budget
     can buy shorter windows, more simultaneous sources, lower drive, or skipped averaging.
  4. Adding the 1.2/1.7/2.5 MHz 'store' tones costs only ~2–3 dB on the recon tones (drive split 7 ways). The current
     pulse cannot deliver 1.7–2.5 MHz (SNR 12 / −22 dB); a wider pulse can but loses ~15 dB at low f.
  5. Price of FDM: each source has its own frequency, so a stage is a narrow band, spread (K−1)/Tw: K=32, Tw 0.5 ms →
     62 kHz (41% of 0.15 MHz). Combined with run -37 (water start max ≈ 0.18 MHz), the lowest band must sit at ~0.10–0.17
     MHz, or use K=16/Tw 0.5 ms (30 kHz).
- Caveats: 2D (the 2D wake is pessimistic vs 3D), no tank-wall reverberation (settle must also cover tank echoes),
  ideal 0.15–2.5 MHz elements (real elements are ~1–2 octaves: likely separate low/high-band elements), long tones raise
  time-averaged intensity/duty cycle (ISPTA, element heating) — the SNR surplus should be spent partly on lower drive.
  Equal-peak comparison only; no transducer response.
- Gate passed? N/A (Gate 2 waived).
- Next (Colab, needs colab-mcp reconnected): (B1) FWI ladder test on the MRI heart: 0.15>0.4>0.8, 0.15>0.3>0.8,
  0.1>0.3>0.8 vs current — quality and time incl. final-stage iterations; (B2) FDM-band inversion: per-source ω in the
  Born-series solver (vmap over ω), K=32 / Tw 0.5 ms bands, vs single-frequency stages at equal quality.

### Run 2026-10-27-39 — Per-source-frequency (FDM) Born-series solver validated; ADJOINT BUG found and fixed: duplicate receiver pixels on coarse grids halved the adjoint source (`.at[].set` → `.add`) — affects every 512-receiver reconstruction's early stages (runs -27…-36)
- Phase: 3 → nonlinear. Colab T4. Connected via a local MCP client driver (scratchpad `colab_driver.py`) because the
  session's colab-mcp tool binding had timed out at startup; the server itself is the stock googlecolab/colab-mcp.
  Notebook `notebooks/colab_session_2026-10-27_multiplex.{ipynb,py}` (gitignored; embeds the MRI-heart labels).
- New solver `make_solver_w`: the convergent Born series (repo `cbs_warm.py`, run -33 fixes) vmapped over a PER-SOURCE ω,
  so each source of a frequency-multiplexed stage is solved at its own tone frequency. Gradient
  dJ/dc = −2/c³ Σ_s ω_s² Re(Λ_s P_s / K_s), K_s measured per source.
- Checks (92² grid, MRI heart resampled):
  - vmap independence: a source's field with different batch-mates' ω: rel diff 0.0; traced-ω vs constant-ω solver: 0.0.
  - Adjoint vs FD: FAILED first (ratio 0.49–0.75), also with constant ω (0.53–0.56), i.e. not an FDM problem.
    Diagnosis (tol 1e-4, converged in 71 its): FD/jvp = 1.0003 (forward model smooth), adjoint/jvp = 0.56; with
    fat removed (min c 1530 > water) adjoint/jvp = 0.58 → NOT the slow-medium (fat < water) jWave issue of runs
    -33/-34. Cause: 512 receivers on coarse grids share pixels (distinct receiver pixels: 256/512 at 84², 472 at 168²,
    512 at 280²/340²) and the adjoint source was built with `.at[:, ri, rj].set(conj(res))`, which keeps ONE of the
    duplicate receivers' residuals instead of summing them. Run -11's check used 64 receivers (no duplicates), so it
    passed.
  - Fixed (`.add`): adjoint/jvp 0.996; adjoint/FD: single 0.15 MHz 0.994/0.996, FDM band (8 src, 0.09–0.16 MHz)
    0.978/0.982 (those solves hit the 5000-iteration cap at tol 1e-6).
- IMPACT: every reconstruction with 512 receivers (runs -27, -30, -31, -35, -36) had a wrong (≈ half-weight, distorted)
  gradient in the stages on grids < ~280² (0.15/0.3 MHz stages; at 128 receivers most stages had no duplicates). The
  0.8 MHz stage was correct. Their metrics stand as measured, but they were produced with a partly wrong gradient;
  this session's E0 is the corrected baseline, not a reproduction of run -35. The repo `src/phase3_fwi_born.py` had the same
  pattern (2 places); fixed to `.add`.
- Gate passed? N/A (Gate 2 waived).

### Run 2026-10-27-40 — FWI test of frequency ladders and FREQUENCY-MULTIPLEXED acquisition (MRI heart): FDM with 32 simultaneous sources (tank 0.75 ms) matches its single-frequency ladder exactly; vs the current pipeline it is 13× shorter in the tank and 35% faster to reconstruct for a small quality cost
- Phase: 3 → nonlinear. Colab T4 (driver as run -39). Notebook `notebooks/colab_session_2026-10-27_multiplex.*`.
- Data: independent jWave time-domain simulator, MRI heart (speed + density + attenuation + texture; run -30 phantom),
  680² / 0.25 mm, 32 sources, 512 receivers, broadband pulse; one run per attenuation frequency
  (0.1/0.15/0.3/0.4/0.5/0.65/0.8 MHz, α ∝ f ASSUMED) and DFT of every trace at 150 exact frequencies (each source at its
  own tone frequency), ratio to a water run; nearest-attenuation mismatch ≤ 0.07 MHz (≲0.3 dB). Equivalence of
  per-source DFT data to an FDM tone acquisition rests on run -38 (isolation ≤ −76 dB with settle ≥ 0.15 ms).
  Data generation 16 min (+ water 2 min).
- Method: blind speed-only FWI as run -35 (multires ~5 ppw grids, TV 1e-6 scaled, blind ROI on the last two stages,
  L-BFGS-B 15 its/stage + 8 final, cold Born-series solves, adjoint tol 1e-2), corrected adjoint (run -39), per-source ω.
  30 dB per stage (rms convention as before; run -38 shows tones give far more SNR, so this is conservative). Seed 21.
- Results (blood–myo separation σ / gap / outer-band RMSE / heart RMSE / water std / recon s / tank ms):
  | config | sep | gap | outer | heart | water | recon | tank |
  |---|---|---|---|---|---|---|---|
  | E0 current 0.15/0.3/0.5/0.8 | **1.47** | 7.33 | 23.55 | 13.38 | 2.06 | 198 | 9.6 |
  | L1 0.15/0.4/0.8 | 1.40 | 7.34 | 23.94 | 13.62 | 2.06 | 130 | 9.6 |
  | L2 0.15/0.3/0.8 | 1.32 | 7.46 | 24.40 | 14.01 | 2.07 | 118 | 9.6 |
  | L3 0.1/0.3/0.8 | 1.32 | 7.40 | 24.31 | 13.92 | 1.88 | 107 | 9.6 |
  | **F1 FDM K32 Tw 0.5 ms** (0.100–0.162 / 0.370–0.432 / 0.770–0.832) | **1.40** | 7.32 | 23.96 | 13.63 | 2.14 | **130** | **0.75** |
  | F2 FDM Tw 0.1 ms, low band 8 src (0.09–0.16 / 0.17–0.48 / 0.49–0.80) | 1.29 | 7.36 | 25.00 | 14.36 | 3.13 | 136 | 0.35 |
  (true-map ceiling sep 2.24, gap 7.46)
- Findings:
  1. Frequency multiplexing is free in quality: F1 = L1 (1.40 vs 1.40σ, heart RMSE 13.63 vs 13.62) although every source
     sits at a different frequency within each stage. Patient-facing acquisition 9.6 → 0.75 ms.
  2. Shorter ladders: no cycle-skipping in any (run -37's prediction holds); each dropped stage costs a little quality
     because the 8 final iterations start further from convergence (final-stage start misfit 0.0052 → 0.0060–0.0069).
     L1 (0.4 MHz middle) is the best 3-stage ladder, as run -37 predicted.
  3. F2 (0.35 ms) works but is measurably worse (water noise +50%, sep 1.29) — likely the 8-source lowest band and/or
     wide bands; not separated here.
  4. Corrected E0 ≈ run -35 (1.47 / 7.33 / 23.55 vs 1.47 / 7.33 / 23.52): the run -39 adjoint bug had negligible effect
     on final quality (early stages converge regardless; the 0.8 MHz stage was unaffected). Earlier results stand.
- Recommendation for the throughput aim: F1-type acquisition (32 simultaneous tone sources, Tw 0.5 ms, ~0.75 ms/slice)
  with the 3-stage 0.15(0.10–0.16)/0.4/0.8 ladder, ~130 s/slice on a T4.
- Caveats: 2D, synthetic, single phantom, 30 dB convention, attenuation α ∝ f without dispersion, ideal broadband
  elements, no tank-wall echoes, not collaborator-reviewed (Gate 2 waived).
- Gate passed? N/A (Gate 2 waived).
- Next: F1 with 10–12 final iterations (recover E0 quality at ~150 s?); F2 variants (16 low-band sources; narrower
  upper bands) to locate the 0.35–0.75 ms trade-off; clinical metric set (run -36) on F1.

### Run 2026-10-27-41 — Follow-ups: F1 with 12 final iterations (G1) recovers the current pipeline's quality at 0.75 ms tank / 154 s; 16 low-band sources (F3, 0.40 ms) only partly fix the short-window designs
- Phase: 3 → nonlinear. Colab T4, same session/data/method as run -40 (seed 21, 30 dB).
- G1 = F1 (32 simultaneous tones, Tw 0.5 ms; 0.100–0.162 / 0.370–0.432 / 0.770–0.832 MHz) with 12 instead of 8 final its.
- F3 = Tw 0.15 ms (6.67 kHz bins), tank ~0.40 ms: 16 sources (every 2nd) 0.073–0.173 MHz / 32 at 0.180–0.387 / 32 at
  0.593–0.800 (80 distinct tones). New data: water + 3 attenuation runs (0.12/0.3/0.7 MHz; max mismatch 0.107 MHz ≈
  ≤0.4 dB), 9 min.
- Results (sep σ / gap / outer / heart RMSE / water std / recon s / tank ms):
  - E0 current (run -40): 1.47 / 7.33 / 23.55 / 13.38 / 2.06 / 198 / 9.6
  - F1 (run -40): 1.40 / 7.32 / 23.96 / 13.63 / 2.14 / 130 / 0.75
  - **G1: 1.46 / 7.39 / 23.57 / 13.35 / 2.17 / 154 / 0.75**
  - F2 (run -40): 1.29 / 7.36 / 25.00 / 14.36 / 3.13 / 136 / 0.35
  - F3: 1.32 / 7.35 / 25.02 / 14.31 / 2.88 / 124 / 0.40
- Findings:
  1. G1 matches the current pipeline (heart RMSE 13.35 vs 13.38, sep 1.46 vs 1.47, outer 23.57 vs 23.55) with 13×
     shorter acquisition and 22% less reconstruction time. NEW DEFAULT: F1 bands + 12 final iterations.
  2. Doubling the low-band sources (F2 → F3) helps only slightly (water std 3.13 → 2.88, sep 1.29 → 1.32); the
     short-window designs' remaining gap is NOT mainly the low band. Likely cause: the wide upper bands (final band
     0.59–0.80 MHz, so half the sources sit well below 0.8 MHz → less high-frequency information). Not separated.
- Trade-off curve so far: 0.75 ms → E0-level quality (154 s); 0.35–0.40 ms → ~10% lower separation, ~40% more water noise.
- Gate passed? N/A (Gate 2 waived).
- Next (optional): short window with narrow final band (e.g. K=16 per group, 2 groups) to test the wide-band hypothesis;
  clinical metric set (run -36) on G1.

### Run 2026-10-27-42 — Full clinical metric set (run -36 set) on all 8 reconstructions: frequency-multiplexed G1 (0.75 ms tank) is at least as good as the current 9.6 ms pipeline on every segmentation and speed metric; short-window designs lose mainly on blood/myo boundaries and water noise
- Phase: 3 → evaluation. Colab (arrays in memory), CPU-type computation. New repo script `src/phase3_clinical_metrics.py`
  (run -36's code was never saved; re-implemented from its LOG description). Unrecorded conventions fixed by matching run
  -36's CEILING numbers on the true map: opening = 1 iteration, 8-connected 3×3; true heart = largest myo+blood
  component, holes filled; HD95 = 95th percentile of pooled surface distances. Ceiling now: heart 0.995 (run -36 0.996),
  blood identical (0.853 / 2.57 / 10.52), myo 0.791 (0.792), fat 0.521 (0.492). Thick-myocardium core = 3-px erosion
  (chosen; not recorded in run -36). Full table: `results/colab_session_2026-10-27/mux_clinical_metrics_table.txt`
  (gitignored) and `/content/mux_clinical_metrics.json`.
- Key results (heart DSC / ASSD / HD95 mm; blood DSC / area %; myo DSC; speed RMSE heart / myo core / water m/s; SSIM; texture r):
  | config (tank) | heart | blood | myo | RMSE heart / core / water | SSIM | tex r |
  |---|---|---|---|---|---|---|
  | ceiling | 0.995 / 0.24 / 1.58 | 0.853 / +2.5 | 0.791 | — | — | — |
  | E0 current (9.6 ms, 198 s) | 0.983 / 0.79 / 3.91 | 0.843 / −3.4 | 0.778 | 13.38 / 2.30 / 2.06 | 0.815 | 0.877 |
  | L1 0.15/0.4/0.8 (9.6 ms) | 0.983 / 0.80 / 3.75 | 0.841 / −3.3 | 0.775 | 13.62 / 2.41 / 2.06 | 0.790 | 0.861 |
  | L2 0.15/0.3/0.8 | 0.982 / 0.85 / 4.02 | 0.840 / −2.4 | 0.771 | 14.01 / 2.81 / 2.07 | 0.750 | 0.815 |
  | L3 0.1/0.3/0.8 | 0.982 / 0.84 / 4.02 | 0.839 / −0.9 | 0.767 | 13.92 / 2.76 / 1.88 | 0.752 | 0.819 |
  | F1 FDM (0.75 ms, 130 s) | 0.983 / 0.80 / 3.91 | 0.840 / −2.9 | 0.774 | 13.63 / 2.43 / 2.14 | 0.786 | 0.860 |
  | **G1 F1 + 12 its (0.75 ms, 154 s)** | **0.986 / 0.67 / 3.00** | 0.841 / −2.3 | 0.774 | **13.35** / 2.46 / 2.17 | 0.788 | 0.860 |
  | F2 FDM (0.35 ms) | 0.982 / 0.81 / 4.00 | 0.831 / −3.5 | 0.763 | 14.36 / 3.02 / 3.13 | 0.747 | 0.762 |
  | F3 FDM (0.40 ms) | 0.982 / 0.84 / 4.03 | 0.837 / −4.2 | 0.772 | 14.31 / 2.87 / 2.88 | 0.765 | 0.790 |
- Findings:
  1. Outer heart boundary: all designs within 0.982–0.986 DSC, ASSD 0.67–0.85 mm (ceiling 0.24): acquisition design
     barely matters for the outer contour. G1 is the best of all (ASSD 0.67 mm, HD95 3.0 mm, NSD@1mm 0.892).
  2. Blood pool: DSC 0.839–0.843 for every 3–4-stage design at ≥0.75 ms (ceiling 0.853); area error −0.9…−3.4%, all
     within CMR EDV interstudy range (2.9–4.9%, Grothues 2002, per run -35). Short windows (F2/F3) drop to 0.831–0.837.
  3. Texture/SSIM are the most sensitive: E0 0.877/0.815 > G1/F1/L1 ~0.860/0.79 > L2/L3 ~0.82/0.75 > F2 0.76/0.75. Fine
     intramyocardial texture benefits from the extra 0.5 MHz stage; G1's extra final iterations recover RMSE but not SSIM.
  4. Fat lining: unreliable by thresholding (ceiling 0.52); designs track it (0.375–0.422).
- Conclusion: for patient throughput, G1 (0.75 ms tank, 154 s recon) is equivalent to the current pipeline on clinical
  contour/volume metrics, slightly worse only on fine texture (r 0.860 vs 0.877). 2D synthetic, single phantom, not
  collaborator-reviewed.
- Gate passed? N/A (Gate 2 waived).

### Run 2026-10-27-43 (part A) — FULL-CHAIN FDM validation: all 32 elements transmitting G1's 3-tone waveform simultaneously through the MRI heart, demultiplexed by DFT, reconstructed with G1 — PASSES the proposed engineering gate vs sequential broadband acquisition under matched peak AND matched energy, 3 noise seeds
- Phase: 3 → acquisition validation (user-supplied external plan, 2026-10-27; its "Run 42" = this run -43, since -42 is
  the metrics run; its Dice-ceiling concern was already resolved in run -42: 0.995 vs run -36's 0.996).
- Simulation: same independent jWave time-domain simulator, MRI heart (speed + density + attenuation + texture), 680² /
  0.25 mm, 512 receivers. All 32 elements driven AT ONCE (multi-mask source) with their own 3 tones (G1 bands
  0.100–0.162 / 0.370–0.432 / 0.770–0.832 MHz on 2 kHz bins), cos² ramps 0.05 ms, settle 0.15 ms, window 0.5 ms
  (tank 0.75 ms); dt set so the window holds an integer number of samples and of every tone's cycles (17,952 steps).
  Per-element tone phases chosen for low crest factor (seed 7). One simulation per band (attenuation α ∝ f at the band
  centre, ASSUMED power law 1), received signals summed by linearity; water run with the same drive as reference.
  6 simulations, 181 s total. No tank walls (PML), ideal element bandwidth — part B.
- Demultiplexing vs the sequential per-source data (run -40 data at the same per-source frequencies):
  - ||R_fdm − R_seq|| / ||R_seq − 1|| = 3.3% / 1.9% / 1.6% (low / mid / high band); per-channel median 2.8 / 1.8 / 1.4%,
    p99 16 / 7.7 / 6.5% of the mean scattered signal. Includes the arms' different attenuation-frequency assignment and
    the sequential arm's 140 µs record truncation, so it is an upper bound on demultiplexing error.
  - Leakage into the empty bins just outside each band: max −64 dB, median −77/−78 dB re the median tone.
- Matched constraints: identical white receiver noise σ per time sample in both arms, σ set so the SEQUENTIAL arm has 30 dB
  (rms rule) in the 0.8 MHz band. Sequential noise per bin = σ√N_rec·dt over its 140 µs record; FDM = σ√N_w·dt over the
  0.5 ms window ÷ tone amplitude. Drive matched either by PEAK (each element's 3-tone waveform peaks at the pulse peak) or
  by ENERGY (each element radiates the same energy as one pulse: 0.0715× the peak-matched amplitude).
  Resulting SNR per band (dB) SEQ / FDM-peak / FDM-energy: 0.13 MHz 12.1 / 59.0 / 36.1; 0.40 MHz 25.8 / 64.3 / 41.4;
  0.80 MHz 30.0 / 67.6 / 44.7. (The sequential pulse is weak at low f: physically consistent noise makes its 0.13 MHz band
  ~18 dB worse than the per-stage-30-dB convention of runs -25…-42.)
- G1 reconstructions (mean ± sd over seeds 1–3; FDM-peak 1 seed):
  | arm | heart DSC | blood DSC | myo DSC | heart RMSE | blood–myo sep | texture r | water std |
  |---|---|---|---|---|---|---|---|
  | SEQ (9.6 ms) | 0.9828 ± 0.0001 | 0.8388 ± 0.0008 | 0.7785 ± 0.0022 | 13.65 ± 0.02 | 1.40 ± 0.02 | 0.843 | 3.36 |
  | FDM energy-matched (0.75 ms) | 0.9856 ± 0.0001 | 0.8410 ± 0.0004 | 0.7735 ± 0.0006 | 13.31 ± 0.02 | 1.46 ± 0.01 | 0.866 | 2.08 |
  | FDM peak-matched (0.75 ms) | 0.9856 | 0.8412 | 0.7738 | 13.32 | 1.47 | 0.866 | 2.07 |
- GATE (user-proposed engineering gate: |ΔDSC| heart & blood ≤ 0.01, heart RMSE within 5%): FDM-energy vs SEQ ΔDSC heart
  +0.0028, blood +0.0022, RMSE −2.5% → PASS; FDM-peak +0.0028 / +0.0025 / −2.4% → PASS. FDM is slightly BETTER than
  sequential under a physically consistent noise model (higher SNR at low f), not merely equal.
- Caveats: 2D; no tank-wall reverberation; ideal 0.1–0.83 MHz elements; within-band attenuation constant; noise-free water
  references; myo DSC 0.774 vs 0.779 (−0.005, within gate); not collaborator-reviewed (Gate 2 waived).
- Gate passed? Proposed engineering gate PASS (part A). Part B pending: tank-wall reflections (−20/−10 dB) × settle time,
  finite transducer bandwidth.

### Run 2026-10-27-44 — Two-thigh phantom (new anatomy, femurs): full-chain FDM acquisition works but leaks more (−34…−40 dB max); the blind G1 reconstruction FAILS — femoral cortex not recovered, fat/skin biased; G1 does NOT transfer to a bone-containing, 2.2× larger field of view as is
- Phase: 3 → cross-anatomy validation (user plan, 2026-10-27). Colab T4 (driver).
- Phantom `src/phase3_thigh_phantom.py` (outputs gitignored, `*thigh*`): traced from a user-supplied T1 axial MRI of one thigh
  (`thigh_ax_22.jpg`); mirrored for the contralateral thigh (medial sides facing, 12 mm water gap). Labels: skin (outer
  1.5 mm, ASSUMED — not separable on T1), subcutaneous fat, muscle (incl. fascia lines), intermuscular fat, femoral cortex
  (incl. linea aspera), fatty marrow, superficial vessels (dark blobs in subcutaneous fat), deep vessels/neurovascular
  structures (dark blobs between muscles + intermediate blobs embedded in fat; may include nerves — modelled as blood).
  Scale ASSUMED: thigh medial-lateral width 150 mm (0.396 mm/px; one thigh 150 × 126 mm). Bug fixed during build:
  `np.unique(...)[1:]` dropped a real label (cortex empty) → filter ids > 0. Tissue radius 158 mm → ring 178 mm
  (20 mm water clearance), domain 380 mm (1520² at 0.25 mm), 32 sources, 1024 receivers.
- New cited tissues (tissue.py): skin = dermis, Moran/Bush/Bamber 1995 UMB (doi:10.1016/0301-5629(95)00049-6): 1595 m/s,
  0.264 f^1.69 dB/cm (20–30 MHz, 24 °C; extrapolated), density PROXY = skeletal muscle 1050; marrow = Kawasaki et al. 2015
  JASA (doi:10.1121/1.4922764): 1410 m/s, 4.4 dB/cm at 10 MHz ∝ f² → 0.044 f² dB/cm (extrapolated), density PROXY = fat
  943.5. Others as before (muscle Mast 2000 1580/1050/0.74 f; fat CRC 1464.5/943.5/0.608 f; cortical bone Pichardo 2011
  2471/1904/240 Np/m at 836 kHz ∝ f^1.756 (skull, stand-in for femur; single speed, dispersion ignored); blood Mast 2000).
- Feasibility probe (Born series, 4 sources timed): 32-source solve 0.13 MHz @216² 2.4 s (water 2.1), 0.4 MHz @520² 12.4 s
  (8.3), 0.8 MHz @760² 55 s (31; femurs +75% iterations). Time domain 1520²: dt 30.4 ns (CFL set by bone), 6.5 ms/step/source.
- Full-chain FDM (G1 tones/phases, energy-matched, settle extended to 0.3 ms > one crossing 234 µs, window 0.5 ms; tank
  0.90 ms; 32,948 steps; 3 bands × water/object = 26 min): leakage into empty bins outside each band max −33.7 / −35.5 /
  −40.5 dB, median −48 / −53 / −60 dB (heart: −64 / −77 dB) — settle 0.3 ms is MARGINAL for this larger, bony field
  (longer reverberation). SNR with run -43's noise PSD: 31.9 / 37.2 / 40.5 dB.
- G1 reconstruction (stages 0.100–0.162 @216², 0.370–0.432 @544², 0.770–0.832 @760²; 15/15/12 its; speed bounds widened to
  1370–2520 m/s): 1401 s (23 min). Misfit after stage 1 0.0138 (heart 0.0013 — 10× worse already at the lowest band),
  stage 3 end 0.0091.
  - Dice: body 0.883, femur 0.000, cortex 0.000, fat 0.564, muscle 0.882; body outline ASSD 25.9 mm, HD95 101 mm.
  - Speed (true → rec): cortical bone 2471 → 1560 (−911), marrow 1410 → 1536, skin 1595 → 1518, subcut fat 1464.5 → 1498.8,
    intermusc fat 1464.5 → 1549.8, muscle 1580 → 1560.9, vessels 1584 → 1520 / 1548; water bias −1.3, RMSE 7.6.
  - Vessels: median CNR 0.12 (superficial, n=51) / 0.28 (deep, n=98); none detected (CNR > 2).
- Visual check of the reconstruction (`thigh_recon_FDM_patient.png`): NO wrong-sign/ghost structure (not the signature
  of cycle-skipping): both thigh outlines, the subcutaneous-fat band and faint rings at the true femur positions are in
  the right places, but ALL contrasts are strongly under-recovered and blurred (muscle −19, fat +34, bone −911 m/s).
  This looks like heavy under-convergence / over-smoothing of a 2.2× larger, high-contrast field with the heart's
  iteration budget and TV weight, rather than convergence to a wrong basin.
- Interpretation (diagnosis NOT yet done). Candidate causes, to separate: (1) cycle-skipping — two 150 mm thighs give a
  larger water-start travel-time anomaly than the heart (muscle alone ~3 µs over 130 mm; the femur adds ~10 µs·(30 mm
  path)… ), so 0.10–0.16 MHz likely exceeds run -37's half-period rule; (2) the femur's high contrast (k² ratio 0.38) makes
  the problem strongly nonlinear and shadows the medial/posterior compartments; (3) too few iterations for a 2.2× larger
  field; (4) 32 sources for a 356 mm ring (angular sampling ~2.4× sparser than the heart's).
- Saved (gitignored): Downloads `session_2026-10-27_part2.zip` → results/colab_session_2026-10-27/ (thigh FDM ratios +
  meta, thigh reconstruction, figure; run -43 arrays; heart RATIO incl. run -41 freqs; SPW; clinical metrics).
- Gate passed? NO (cross-anatomy). Sequential arm NOT run (runtime budget). Next: run -37-style cycle-skip analysis on the
  thigh (CPU, minutes) to pick a start frequency/ladder; a lower first band (≤ 0.05–0.08 MHz) and/or a bone-aware start
  (e.g. lower-frequency stage with more iterations), more sources; only then the sequential comparison.

### Run 2026-10-27-45 — Thigh failure diagnosis at the lowest band (0.100–0.162 MHz @216², ~1–3 min per test): iterations and a TV weight mis-scaled for bone are the main causes; receivers do not matter, sources modestly; the outer-margin error is TV/resolution, NOT the jWave slow-medium bug and NOT density mismatch
- Phase: 3 → diagnosis (user hypotheses: (1) iterations/regularization, (3) source/receiver sampling — not (2) bone
  shadowing, because the medial region between the femurs reconstructs better than anterior/posterior). Same run -44
  FDM data and noise realization; stage 1 only; error vs the truth blurred to this band's resolution (FWHM λ/2 at
  0.162 MHz). Regions: anterior y < −25 mm (NOTE: clips the top of the femur, so its RMSE is bone-dominated), posterior
  y > 25, medial |x| < 50, lateral |x| > 110 mm.
  | test | misfit | RMSE ant / post / med / lat (m/s) | muscle / fat mean (blurred truth 1577 / 1501) |
  |---|---|---|---|
  | T1 15 its, TV (run -44 setting) | 0.0138 | 58.6 / 17.6 / 22.2 / 17.0 | 1562 / 1518 |
  | T2 40 its, TV | 0.0111 | 51.3 / 12.8 / 14.8 / 18.1 | 1569 / 1505 |
  | T3 40 its, no TV | 0.0021 | 65.4 / 39.2 / 40.7 / 45.4 | 1571 / 1488 |
  | T4 40 its, TV, 512 receivers | 0.0109 | 51.7 / 12.8 / 14.9 / 18.1 | 1569 / 1505 |
  | T5 40 its, TV, 16 sources | 0.0079 | 55.8 / 18.6 / 18.8 / 22.9 | 1564 / 1509 |
  | T6 40 its, TV/10 | 0.0026 | 59.2 / 31.0 / 31.3 / 37.7 | 1572 / 1486 |
  | T7 40 its, TV, SPEED-ONLY data from the recon solver (no density/attenuation/time-domain mismatch) | 0.0073 | 44.4 / 11.9 / 14.9 / 19.9 | 1573 / 1506 |
  Outer-margin band (±2 px of the skin surface) RMSE / mean bias: T2 16.0 / −8.4; T6 32.1 / +1.4; T7 17.0 / −6.2.
  Femur (±1 px) RMSE / max speed (blurred truth max 2389): T2 455 / 1686; T6 420 / 1824; T7 367 / 1824.
- Findings:
  1. Iterations matter (15 → 40: posterior/medial error −27…−33%). Even with perfect physics (T7) 40 its leave the misfit
     ~20× above the noise floor (~3e-4): convergence-limited with the femurs present.
  2. The TV weight tuned on the heart is mis-scaled here: bone is a 9.5-unit jump in the scaled model (heart ≤ 1.2), so
     TV crushes the femur (max 1686 vs 2389), while TV/10 recovers more bone (1824) but doubles soft-tissue noise. One
     global TV weight cannot serve bone and soft tissue → needs bone-aware regularization / parameterization.
  3. Receivers: 1024 → 512 no change. Sources: 32 → 16 worse by ~25–45% (posterior 12.8 → 18.6) → more sources (64) may
     help moderately; receivers are not the bottleneck.
  4. Outer-margin errors: same with speed-only data (T7 17.0 vs T2 16.0) → at this band they are TV smoothing +
     resolution, NOT model mismatch, and NOT the run -33 jWave min-speed / max-k bug (patched solver; marrow 1410 m/s
     slower than water is handled by the same fix). Density/attenuation/time-domain mismatch accounts for ~20% of the femur
     error (455 → 367) and part of the misfit (0.011 → 0.0073). At 0.8 MHz the density mismatch may matter more at the
     margins (run -19: speed-only inversion of density data put artefacts at the outer boundary) — untested.
- Next: bone-aware regularization (TV on a compressed parameter, or TV weight per region/edge), more low-band iterations,
  64 sources; then the full 3-stage thigh reconstruction and the sequential comparison.
- Gate passed? N/A (diagnosis).

### Planning note 2026-10-27 (no run) — Thigh: computation problem or dead end? Assessment + next-session plan
- User question: is the thigh failure (runs -44/-45) "just a calc problem and not a dead end"? Assessment: PROBABLY a
  computation/regularization problem, NOT yet proven.
  - For fixable: no ghost/wrong-sign structure (anatomy in the right places, washed out = under-converged, not
    cycle-skipped); every lever moved the right way (iterations −30% error, weaker TV → more bone, more sources better);
    without TV the data fit 5× better (information is in the data; regularization/convergence hold it back); receivers
    already saturated (geometry not starved); transcranial FWI in the literature reconstructs through skull given low
    frequencies + iterations (general knowledge, no specific paper checked here).
  - Risks: bone is still the hard part (best low-band femur max 1824 vs blurred truth 2389 even with perfect physics +
    weak TV); the fixes all cost compute (thigh 23 min/slice vs heart 2.5; fixed version could be ~1 h/slice — tank time
    stays < 1 ms, so the throughput argument survives, but "minutes per slice" holds only for heart-sized bone-free
    fields); density mismatch at 0.8 MHz near bone/skin untested.
- DECIDING EXPERIMENT (first job next GPU session, ~1–1.5 h T4): full 3-stage thigh reconstruction with
  (a) bone-aware regularization (TV on a compressed speed parameter, or edge/region-weighted TV),
  (b) more low-band iterations (≥ 40), (c) 64 sources (needs new FDM data: 64 elements, bins 1 kHz apart if one group,
      i.e. Tw 1 ms, or 2 groups; recheck settle ≥ 0.3 ms and leakage, run -44 was −34…−40 dB max).
  Pass = femur outline recovered + soft tissue near heart-level errors. If the femur stays smeared → bone is a real
  limit; fallback = bone as a known/segmented prior (bone-aware but no longer fully blind).
- Then: sequential comparison arm for the thigh (~2.5 h data); run -43 part B (tank walls, element bandwidth).
- Resume notes: all saved arrays in results/colab_session_2026-10-27/ (thigh FDM ratios R0..R2 + YW + meta incl. WfT,
  run -44 recon; heart data/recons). Colab access when colab-mcp's tool binding is down: local MCP client driver in the
  session scratchpad (colab_driver.py + mcpcall.py + runcell.py; NOT in the repo) spawning the stock googlecolab/colab-mcp
  server; never re-call open_colab_browser_connection on a live link (user feedback) — probe with a dummy cell. Cell
  sources for runs -39…-45 + the driver scripts copied to notebooks/colab_session_2026-10-27_cells/ (gitignored; run
  in order after the multiplex notebook's cells 0–9: followup → metcell → run43a → run43b → probe/run44a → run44b → run45 →
  run45b → save_all).

### Run 2026-10-28-46 — Bone-cylinder test (external-evaluation suggestion): an isolated femur IS recovered (femur Dice 0.95–0.98) even with the density mismatch; density mismatch costs ~15–20%; the thigh failure is scale + iteration/source budget, not bone per se
- Phase: 3 → diagnosis. Colab T4 (new session, local MCP driver; link flapped repeatedly — retry logic added to the
  helper). Cell source: notebooks/colab_session_2026-10-27_cells/ (run46.py, copied there with this entry).
- External evaluation (user-pasted, 2026-10-28) checked: muscle/bone impedance 1.66 vs 4.70 MRayl → R² 22.9% (true
  densities) vs 4.8% (constant-density model) ✓; critical angle asin(1580/2471) = 39.7° ✓ (fluid model); 10 mm bone
  path −2.28 µs = 0.34 cycles @0.15 MHz, 1.83 @0.8 MHz ✓. Its linked references were not verifiable (favicon only).
- Setup: heart-scale ring (170 mm domain, ring 75 mm, 32 src, 512 rec, 340²); muscle cylinder r 45 mm (1580/1050) with a
  femur-like target 6 mm off-centre: cortex annulus r 8–14 mm (2471/1904), marrow core (1410/943.5). Data from the
  speed-only Born series (matched physics) or the density Born series (`cbs_full`, rho_smooth 0; normalized by its own
  water field), same grid (inverse crime on purpose), 40 dB. Density effect on data ||Dd−Ds||/||Ds−W|| = 0.33 / 0.20 /
  0.23 at 0.13 / 0.4 / 0.8 MHz. Born iterations with bone: speed-only 343 / 1575 / 1495, density 8813 / 9285 / 2996.
  Blind speed-only recon, single-frequency ladder 0.13 / 0.4 / 0.8 MHz @76 / 224 / 340², 40 / 15 / 12 its, bounds
  1370–2520 m/s, ~3 min each.
- Results (femur Dice / cortex Dice(c>2000) / cortex mean / max / marrow / muscle RMSE / final misfit):
  - B1 speed data, TV: 0.984 / 0.777 / 2019 / 2174 / 1640 / 11.0 / 0.0037
  - B2 density data, TV: 0.976 / 0.714 / 1961 / 2237 / 1615 / 13.3 / 0.0051
  - B3 speed data, TV/10: 0.962 / 0.822 / 2052 / 2259 / 1602 / 13.8 / 0.0024
  - B4 density data, TV/10: 0.948 / 0.687 / 1969 / 2408 / 1605 / 21.6 / 0.0042
  (true cortex 2471, marrow 1410, muscle 1580)
- Findings:
  1. Bone is NOT a hard block: the femur outline is recovered (Dice 0.95–0.98) and the cortex reaches ~2000–2400 m/s
     (6 mm cortex partial-volume smeared; the 16 mm marrow core is not resolved: ~1600 vs 1410).
  2. Density mismatch (speed-only model, density data) costs ~15–20% (cortex RMSE 462 → 543, cortex Dice 0.78 → 0.71,
     misfit +37%), consistent with run -45's ~20%. Joint speed+density inversion (run -31 inconclusive) is the fix.
  3. TV/10 helps bone slightly with matched physics but doubles muscle noise with density data → TV is not the main
     limiter at this scale.
  4. Therefore the run -44 thigh failure is mainly SCALE + BUDGET: 2.2× field of view, 15 (not ≥40) low-band
     iterations, 32 sources over a 356 mm ring (run -45: sources matter, receivers do not).
- Run -47 (frequency-to-angle assignment) ran after the link came back — see its own entry.
- Gate passed? N/A (diagnosis).

### Run 2026-10-28-47 — Frequency-to-angle assignment test (external eval point C): NO effect — monotonic G1, bit-reversed scramble and a single frequency give the same errors (≤ ~2 m/s differences); the anterior > posterior error persists with the femur excluded and does not follow the frequency assignment → anatomical, not an FDM acquisition artefact
- Phase: 3 → diagnosis. Colab T4, same session as run -46 (cell `run47.py` in notebooks/colab_session_2026-10-27_cells/).
- Setup: two-thigh phantom, lowest band only, MATCHED physics (speed-only data from the recon solver at 760²) + noise at
  run -44's band-0 SNR (31.9 dB), 40 its, TV as run -44 stage 1, recon @216². Error vs truth blurred to this band's
  resolution, in NON-overlapping per-thigh angular regions (anterior / posterior / medial / lateral, medial = facing the
  other thigh) with the femur EXCLUDED (dilated 3 px), plus 8 sectors of ring angle (0° = +x, counter-clockwise in image
  coordinates; 180–360° = upper/anterior half).
  | assignment | misfit | RMSE ant / post / med / lat | sectors 0–45 … 315–360 |
  |---|---|---|---|
  | monotonic (G1: f rises with source angle) | 0.00728 | 18.6 / 12.2 / 16.1 / 20.7 | 13.6 14.3 14.2 13.7 20.2 18.6 15.5 20.9 |
  | scrambled (bit-reversed) | 0.00691 | 17.8 / 13.7 / 15.9 / 18.5 | 15.1 14.6 15.8 13.3 17.3 20.4 16.3 20.0 |
  | single 0.131 MHz | 0.00608 | 21.2 / 12.3 / 17.2 / 21.5 | 14.2 14.2 16.3 13.1 21.9 19.6 17.3 23.3 |
- Findings:
  1. The frequency-to-angle coupling of G1 does NOT cause directional error: the three assignments differ by ≤ ~2 m/s
     per region with no consistent winner (scrambled slightly better anterior/lateral, worse posterior). Keeping G1's
     assignment is fine; scrambling is free and can be the default to remove the confound.
  2. With the femur excluded and regions non-overlapping, the ANTERIOR soft tissue is still worse than posterior
     (~18–21 vs ~12–14 m/s), and the upper-half ring sectors (180–360°) are worse than the lower half in ALL three
     assignments → the asymmetry is tied to the ANATOMY (thick anterior subcutaneous fat + skin, whose opposite
     contrasts vs water cancel when blurred; quadriceps/femur geometry), not to the acquisition.
  3. This partly revises run -45's anterior number (51 m/s, bone-dominated) — the femur-free anterior error is ~19 m/s.
- Gate passed? N/A (diagnosis).
- Next: the decider — full 3-stage thigh with ≥40 low-band its and 64 sources (+ scrambled assignment), then joint
  speed+density if the femur remains density-limited (run -46: ~15–20%).

### Run 2026-10-28-48 — Why the thigh error is spatially non-uniform: illumination hypothesis REJECTED; measured error profiles show three overlapping effects (surface skin/fat blur, sparse-source near-field at the lateral ends, path-complexity at depth)
- Phase: 3 → diagnosis (user question: degradation is not a uniform inward decline). CPU + Colab.
- (a) Illumination (Colab, 0.131 MHz @216², true model): pseudo-Hessian diagonal H = Σ_s|P_s|²·Σ_r|G_r|² varies only
  ~±70% over the soft tissue (log10 −0.22…+0.28); tissue does not change the forward energy at this band (true/water ≈
  1.00 in all regions). Spearman(log H, |error| of the run -47 single-frequency recon) = +0.10 (wrong sign for an
  illumination effect); best-illuminated tissue (lateral ends, 48 mm from the ring) has the HIGHEST error. → Uneven
  gradient illumination is NOT the explanation (at least at the low band); illumination preconditioning not a priority.
- (b) Error profiles of run -44's FINAL reconstruction (CPU, soft tissue, femur ±3 px excluded):
  - depth below skin (median |err| / bias): 0–2 mm 74 / −58 (skin), 2–5 mm 33 / +31 (fat), 5–10 33, 10–15 23,
    15–20 14, 20–30 15, 30–40 17, 40–50 19, 50–70 24 m/s → U-shaped: surface worst, 15–30 mm best, deeper worse.
  - distance to ring: 15–30 mm 25 (lateral ends), 30–50 13, 50–70 16, 70–90 14, 90–110 22, 110–130 21, 130–150 27,
    150–180 35 m/s (domain centre / medial bulk) → non-monotonic.
  - distance from femur: 0–5 mm 12 rising to 38 at 70–100 mm (tissue near the femur is mid-depth, not near surface).
  - muscle only, by depth: bias −36 (0–10 mm, partial volume with fat), −19, −15, −16, −20 (40–70 mm) → global
    contrast compression of ~−15…−20 m/s everywhere (under-convergence/TV), worse at the surface and slightly at depth.
- Interpretation (three overlapping effects):
  1. Outer ~10 mm: skin (+75 vs water) and fat (−55) are thin with opposite contrasts → blurred into ~water speed by
     resolution + TV → largest errors, independent of acquisition (consistent with run -47's anatomical asymmetry).
  2. Tissue within ~30 mm of the ring (lateral ends of both thighs): 32 sources are 35 mm apart, so a point 15–30 mm from
     the ring sees large angular gaps between nearby sources → near-field sampling artefacts (run -45: sources matter).
  3. Deepest tissue (>90 mm from the ring, medial bulk near the domain centre): every path crosses the most tissue
     (horizontal paths cross both thighs) → most accumulated phase / nonlinearity, slowest convergence (run -45: more
     iterations helped the deep regions most). Best band = 15–30 mm deep, 30–90 mm from the ring.
- Implications: more sources (64) for effect 2; more iterations / better nonlinear convergence for effect 3; effect 1 is
  a resolution/regularization limit for thin layers of opposite contrast (higher frequency or edge-preserving
  regularization), not an acquisition problem.
- Gate passed? N/A (diagnosis).

### Run 2026-10-28-49 — CHEST illumination (user question: does the lung starve every signal, or would a chest failure be a reconstruction problem?): the heart IS illuminated (−7 dB at 0.15 MHz, −9 dB at 0.3 MHz vs water), almost entirely through the ANTERIOR window; lateral views are blocked (−26…−34 dB) → a limited-view problem, not a dark heart
- Phase: 3 → access analysis. Colab T4. Cell `run49.py` (notebooks/colab_session_2026-10-27_cells/).
- Method: CT-traced chest phantom (run -23; labels water/soft/lung/bone/myo/blood, cited values as the 2026-10-26
  notebook: lung 658/300/37.3 dB/cm/MHz ∝ f, bone 2384/1904 Pichardo power law, soft 1580/1050/0.74, myo/blood Mast),
  800² @0.5 mm, ring 185 mm. Independent jWave time-domain solver (validated for lung/bone, runs -33/-34) with a running
  DFT of the WHOLE pressure field at 0.15 and 0.3 MHz (custom scan carrying the DFT accumulator), broadband pulse,
  8 sources (every 4th of 32), 300 µs. Media: full chest / lung replaced by soft tissue (ribs kept) / water.
  Illumination E(x) = Σ_s |P_s(x)|², ratio to water. ~5 min.
- Results (median [10–90th pct] ratio to water):
  | region | 0.15 MHz chest | 0.15 no-lung | 0.3 MHz chest | 0.3 no-lung |
  |---|---|---|---|---|
  | heart (myo+blood) | 0.190 [0.081–0.422] (−7.2 dB) | 0.609 (−2.2 dB) | 0.117 [0.051–0.292] (−9.3 dB) | 0.424 (−3.7 dB) |
  | lung | 0.002 (−27.4 dB) | 0.662 | 0.000 (−35.3 dB) | 0.446 |
  | chest wall/soft | 0.602 (−2.2 dB) | 0.744 | 0.498 (−3.0 dB) | 0.552 |
  Heart illumination per source (chest/water; +90° = image top = anterior): 0.15 MHz: 0° −26.5, −45° −8.6, −90° −10.4,
  −135° −9.4, ±180° −33.2, +135° −8.7, **+90° −2.0**, +45° −11.7 dB; 0.3 MHz: −32.6 / −10.6 / −11.8 / −17.2 / −34.0 /
  −12.1 / **−3.4** / −15.0 dB.
- Findings:
  1. The lung does NOT darken the heart: one-way energy reaching the heart is −7…−9 dB (12–19% of water); the lung costs
     ~5 dB of that (no-lung −2…−4 dB). By reciprocity the receiver side is similar, so round-trip ≈ −14…−18 dB —
     consistent with run -24's measured 12–14% surviving inner-wall signal (−17 dB).
  2. Access is strongly DIRECTIONAL: the anterior window (+90°) is nearly open (−2…−3 dB), obliques −9…−17 dB, the two
     lateral views (0°, 180°) are blocked (−26…−34 dB). The heart is therefore reachable but only through a limited set of
     views → a LIMITED-VIEW inverse problem (k-space gaps, runs -02/-03 analysis), not a no-signal problem.
  3. The lung itself is acoustically dark (−27…−35 dB) — its own speed cannot be reconstructed from transmission; any
     chest inversion must treat the lungs as a strong, poorly-constrained scatterer (needs a good prior/segmentation or
     reflection-based boundary).
  4. Hence a chest reconstruction failure would NOT be simple calibration: it combines (a) limited-view geometry (needs
     elements concentrated on the anterior/oblique windows), (b) the extreme lung nonlinearity (c 658 m/s, Born series
     7.8k–15k iterations, run -34 → infeasible solver cost), and (c) the dark lung region. Signal level alone (−7…−9 dB
     one-way) is workable with averaging/tones (run -38/-43 tones give +35–40 dB SNR headroom).
- Operational note: a stale colab-mcp driver from 2026-10-27 (PIDs 9944 tree) still polls the same command queue as
  today's driver, which explains the "Unknown tool" flapping; stopping it was not permitted from this session — the user
  should close it (or reboot) before the next Colab session.
- Gate passed? N/A (analysis).
