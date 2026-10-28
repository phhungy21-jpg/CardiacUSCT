# MANIFEST — fourier_inverse/

Quick "what exists right now" reference. Full reasoning: `LOG.md`.
Protocol: `../fourier_inverse_phase_protocol.md`.

## Status (2026-10-27): FDM acquisition validated full-chain on the heart (run -43); thigh (run -44) FAILED, diagnosis next. Gate 0 PASSED. Phase 3 (diffraction tomography) IN PROGRESS: 3.1/3.2/3.2b/3.3 done (runs -02/-03), 3.4 (SVD) next. Phases 1–2 not started (independent tracks, per protocol amendment 2026-10-24).

Analytic and Fourier identifiability study for the inverse problem.
There is no full-wave simulation and no GPU: every forward model is
closed-form or a small linear operator, and runs on CPU in seconds.
Started 2026-10-24 as a deliberate change of approach from the jWave
workspaces.

Gate 2 (acoustic-physics collaborator signoff on the forward-model
approximations) is NOT done and does not inherit from any prior project.

## Established so far
- 1D layer reversal: the complex transmission T(f) is exactly invariant
  under reversal of a lossy layered stack with the same medium on both
  sides (verified to 1e-16 over 914 random stacks). Reflection R(f) is
  not invariant (median 22% difference). Run -01.
- Linearization validity (run -02): Born and Rytov both FAIL for the
  whole heart in water (Born error 5–19× at a = 30–45 mm). The inner
  wall (blood/myocardium) is near-linear: Rytov error ≤ 8% at all tested
  sizes and frequencies. Linearize about a background containing the
  outer wall, not about water.
- k-space coverage (run -03): time-of-flight ≈ 11% of a wall feature's
  information; full-wave transmission ≈ 97%; all pairs 100%. Pasted
  single-source design ≈ 22% (2% for beam-axis normals). Anterior-arc
  pulse-echo ≈ 11% (0.6% for lateral normals). Unmeasured parts are NOT
  a strict null space for compactly supported objects (run -04
  correction): unique by analytic continuation, exponentially unstable.
- Blind finite-region objects (run -05): 4 regions / 32 unknowns are
  LOCALLY unique and well-conditioned in every design (σ_min 0.0018 for
  time-of-flight up to 0.996 for all pairs).
- Complexity (run -06): with sharp polygon corners, local uniqueness
  holds while unknowns ≤ independent data (time-of-flight 45,
  single-frequency pasted 183, full ring all pairs 10,909). The full ring
  keeps σ_min ≥ 0.55 at 280 unknowns. Global uniqueness: rests on theory
  (analyticity; Nachman/Bukhgeim, to verify), not on search (user decision).
- Blind region formation (run -07): from full-ring data alone (no prior,
  no guess, no labels), regions form along the true outlines, internal
  boundaries and concave corners included. Misses are only thin occluded
  slivers. The pasted and anterior designs give artifacts or nothing.
  Noise-free, linearized model only.
- Nonlinear solver choice (runs -09/-10, Colab T4): jWave convergent Born
  series matches the exact point-source field to ~2.5% and is the
  fastest (0.1–1 s per solve up to 864²). jWave GMRES with the default
  PML is ~37% wrong; do not use it. Estimated 2D heart-slice
  reconstruction ~1–2 T4-hours with a reciprocity adjoint (to be
  verified).
- First blind NONLINEAR reconstruction (run -11): adjoint gradient
  verified vs jvp (1–2%). From plain water, 4 polygon regions under
  strongly nonlinear physics were recovered: model error 0.27, mean
  speed error 4 m/s inside objects, corners and a slow region correct.
  The weak internal edges are visible but not segmented (global 8%
  threshold). Inverse crime, noise-free, sound speed only. Later
  stages cost far more than the first gradient suggested (the 0.7 MHz
  stage was interrupted at 22 min).
- FAIR test (run -12): finer-grid data + 30 dB noise. The object
  outline and strong contrasts survive (slow region ~11σ); contrasts
  ≤10 m/s are lost (pixel noise ~8 m/s). Blood/myocardium (8 m/s) is in
  the failing range without a prior. Next: patient-anatomy slice, no
  regularization vs TV.
- Real-anatomy slice (run -13, patient023, scaled ×0.386, cited speeds,
  fair conditions): no prior → inner wall invisible (0.6σ); TV λ = 1e-6
  → LV/RV cavities visible inside the myocardium (2.0σ, gap 7.2 vs
  true 8 m/s). λ not chosen truth-free (the discrepancy rule could not
  decide).
- Gate 2 WAIVED by user (run -14): results are exploratory, not
  collaborator-reviewed.
- Speed (run -15): tuned Born-series PML (64 cells, α 0.5) + warm start
  → scaled-heart TV reconstruction 713 s → 18 s, with better quality
  (inner wall 2.28σ, water noise ±4.5 m/s). Use `make_fast_solver`;
  the default PML (16, α 1) is slow and ~3% inaccurate.
- **CORRECTION (run -16): the warm start is BROKEN** (it returns stale
  fields after small model updates; the tolerance is relative to the
  source). Use cold solves (`zero_guess` every call). Run -15's 18 s
  figure needs re-verification. Real-size data (520², 4 frequencies)
  are generated in the Colab session.
- **REAL-SIZE result (run -17):** patient023 slice at true size, blind,
  TV λ = 1e-6, cold solves, 0.15–0.8 MHz, 536 s on a T4. The inner wall is
  clearly recovered: LV–myo gap 7.8 m/s (true 8), 4.5σ; RV–myo 3.0σ.
  Figure `results/figures/phase3_fwi_patient023_realsize_tv.png`.
  Caveats: λ hand-set, same solver type for data, sound speed only,
  2D static, idealized ring, not collaborator-reviewed.
- Attenuation (run -18, `src/cbs_ext.py`, validated vs exact H0 with
  complex k): in the data it barely hurts speed-only (inner wall
  4.0σ); as an extra unknown (naive scaling and shared TV) it was NOT
  recovered (map ~0.01 uniform) and speed degraded slightly (3.5σ).
  Density: not yet implemented (the Born series has no density term).
  Colab checkpoint `/content/checkpoint_session.npz` (runtime disk only).
- Independent simulator + density (run -19): jWave time-domain data
  agree with the Born-series model to 0.8%. Control (speed only) inner
  wall 4.45σ; with cited density 5.35σ, with density artifacts
  confined to the outer boundary. All-three estimated ~4–5σ (not run).
- Data-subset sweep (run -20): time ∝ sources. 32 → 16 sources halves the
  time but costs ~25% of inner-wall separation; 8 sources lose the
  wall. Receivers matter (and are compute-free). Dropping the 0.15 MHz
  stage makes the inversion fail (cycle-skipping). No free redundancy;
  speed must come from source encoding or iteration count, not from
  fewer sources.
- Multi-resolution (run -21): grids 64/128/216/260 for 0.15/0.3/0.5/0.8 MHz
  → 486 s → 199 s at the same quality (4.32σ). The final 0.8 MHz stage is
  now ~60% of the time. The reconstruction code lives in the Colab
  notebook; it still needs porting into `src/` (resume script).
- **Off-repo storage (2026-10-25):** user's Google Drive
  `MyDrive/cardiac_usct_fourier_inverse/session_2026-10-25/`:
  `all_results_patient023.npz` (41 arrays: all reconstructions runs
  -13..-22, Born-series and time-domain datasets, label maps, sweep
  table; md5 b574de2d9797350e54552d82b0dea57b), the Colab checkpoints,
  sweep .npy files, figures, and the notebook's solver files
  (fwi_born.py, cbs_ext.py, cbs_warm.py, atten_td.py). PATIENT-DERIVED:
  never copy into the repo except under `results/*patient*` (gitignored).
  **Migrated locally** to `fourier_inverse/results/colab_session_2026-10-25/`
  (gitignored folder; md5 verified, run -23).
- **Full Colab notebook (57 cells, runs -09..-23)** saved to
  `fourier_inverse/notebooks/colab_session_2026-10-25_patient.ipynb` (+ a `.py`
  export). It contains everything needed to rebuild a session (installs, solvers,
  time-domain generator, calibration, multi-resolution/ROI reconstruction,
  metrics). GITIGNORED: it embeds the patient023 RLE label string. To resume:
  open it in Colab (GPU), run the setup cells, and load data from
  `results/colab_session_2026-10-25/` or the Drive folder.
- **Profile + SNR sweep (run -25):** wave solves are 87% of runtime (forward
  45%, adjoint 42%); the 0.5/0.8 MHz stages are 82%; batch waste only 1–2%
  (per-source batching dropped from the roadmap). The water-bath quality
  PLATEAUS at ~4.7σ above 40 dB (20 dB → 2.0σ, 30 → 4.25, 40 → 4.65,
  noise-free → 4.68). Averaging helps the chest (deeply noise-limited),
  not the water bath.
- **Iteration/tolerance tuning (run -26):** exhausted. Quality tracks the 0.5/0.8
  MHz iteration count roughly linearly; adaptive stopping only moves along the same
  curve. Adjoint tol 1e-2 is free (new default, ~4% faster). The Born series has a ~100-iteration
  propagation floor. Remaining levers: source encoding, a faster solver, amortized compile.
- **Receiver count (run -27):** 128 → 512 receivers (32 sources): 4.25σ → 5.52σ at
  30 dB, noise-free 4.68 → 5.71σ, same recon time. Genuine angular-sampling gain
  (2ka ≈ 300). 512 receivers is the new default.
- **Source encoding (run -28):** negative at 32 sources. Same quality/compute
  curve as cutting iterations; L-BFGS restarts make it 1.5× slower wall-clock;
  crosstalk adds background noise. The algorithm-level speed levers are exhausted (~143 s/slice on a T4).
  Remaining: a solver-level speedup or hardware.
- **Density solver (run -29):** `src/cbs_full.py` (Liouville transform, k²→k²+q, plus
  attenuation). Validated vs time-domain on heart-in-water densities: removes 40–70% of
  the speed-only model error at 0.3–0.8 MHz; σ=0 (no smoothing) best. Chest (V2) NOT yet validated.
- **MRI-traced realistic heart (run -30):** `src/phase3_mri_heart_phantom.py` (outputs
  gitignored, *patient* / *mri_heart*). Labels 0 water / 1 myo / 2 blood / 3 dim lining
  (fat proxy, `tissue.FAT_PERICARDIUM_PROXY`). Blind speed-only reconstruction (data with
  density+attenuation+texture, 512 rec, 30 dB, 441 s): region means within 2–4 m/s,
  texture r=0.84, fat lining ~68% contrast, outer-band RMSE 24.5.
- **Joint speed+density (run -31):** density never moved (stayed at 994 everywhere); speed result unchanged.
  Inconclusive: density gradient far smaller than speed gradient (needs balancing); unexplained ~8%
  under-scaling of the density gradient vs finite differences.
- **Clinical translation (run -32):** speed-only outer boundary 0.41 mm mean / 1.1 mm 95th pct, area −1%
  (within CMR reproducibility, Grothues 2002); blood pool area −6% (2D-echo level), limited by the intrinsic
  8 m/s blood/myo speed contrast.
- **Born-series bugs fixed (runs -33/-34):** jWave's `scale_source_helmholtz` scales by the grid's MIN
  speed and `_cbs_pml` by the MAX wavenumber; both break for slow tissue (lung). Patched in
  `src/cbs_full.py`, `cbs_warm.py`, `cbs_ext.py` (boundary/water speed). Exact fluid-cylinder (Mie)
  test: lung 360–400% → ~10% error. Chest vs time domain: 8.5–16%, but 7.8k–15k+ iterations
  (water ~400) → a chest inversion with this solver is ~17–30 h on a T4 (infeasible). Time-domain
  solver validated for lung/bone (2–5% / 4–19%). MRI-heart runs -30/-31 should be repeated with
  the patch (fat < water speed triggered the scaling bug).
- **MRI heart, fixed solver (run -35):** water noise 3.43→2.16, blood-pool area −6.0%→−3.6%,
  outer contour 0.38/1.0 mm, texture r 0.875, fat contrast 72%. Figure `*_FIXED.png` (gitignored).
  Lung/frequency analysis: no frequency bypasses lung (attenuation ∝ f up; impedance mismatch worse at
  low f, Rice 1983/Kraman 1983); bypass must be geometric (acoustic windows).
- **Session 2026-10-26 saved (all gitignored, patient-derived):**
  `results/colab_session_2026-10-26/` (from the Colab zip, also kept in Downloads):
  - `arrays_mri_heart_and_chest_patient.npz`: recM (run -35 reconstruction), cC2/labC2/labF2/cF2_np
    (MRI phantom maps incl. texture), labCH (chest labels), RATIO2_*/WC2_* (MRI time-domain data
    ratios + water fields, 4 freqs, 512 receivers), RTD_chest_8src_* (chest time-domain ratios,
    0.15/0.3 MHz)
  - `chest_td_8src_patient.npz`, `mri_rec_30db.npy`, `mri_recon_30db.png`, `metrics.json`
  - Colab copies of the solver files
  Notebook: `notebooks/colab_session_2026-10-26_patient.{ipynb,py}` (29 cells).
  RESUME ORDER after a runtime reset: cell 1 (install, kernel restart) → cells 2–4 + the cbs_full
  writefile cell → cell 25 (applies the source-scaling + PML patches) → the "prep" cell (PML patch in
  cbs_warm + helpers). The writefile cells hold the UNPATCHED text; the patch cells must run after them.
  The repo `src/cbs_*.py` are already patched.
- **Project aim (2026-10-26):** patient THROUGHPUT — rapid acquisition, offline FWI. See
  `RESEARCH_NOTE_throughput_2026-10-26.md` (user note + evidence annotations).
- **Runs -46/-47 (2026-10-28):** isolated femur in muscle IS recovered (femur Dice 0.95–0.98; density mismatch costs
  ~15–20%) → thigh failure = scale + iteration/source budget, not bone per se. FDM frequency-to-angle assignment has no
  effect; anterior > posterior thigh error is anatomical (persists with femur excluded, independent of assignment).
- **NEXT SESSION PLAN (2026-10-27, supersedes the list below):** (1) deciding thigh experiment — full 3-stage recon with
  bone-aware regularization + ≥40 low-band its + 64 sources (LOG planning note after run -45); (2) thigh sequential arm;
  (3) run -43 part B (tank walls, element bandwidth). Run -45: thigh failure = iterations + TV mis-scaled for bone
  (+ sources moderately); receivers irrelevant; margin errors = TV/resolution, not the jWave slow-medium bug.
- **NEXT SESSION PLAN** (updated 2026-10-26, session archived after run -36 + throughput note).
  START HERE: read `RESEARCH_NOTE_throughput_2026-10-26.md`, then LOG runs -25…-36.
  Done since run -24: SNR sweep (-25), iteration/tolerance/encoding (exhausted, -26/-28), 512 receivers
  (-27), density solver (-29), MRI heart (-30/-35/-36), clinical translation (-32), solver bug fixes
  (-33/-34), lung/frequency analysis (-35).
  Open items, in priority order for the THROUGHPUT aim:
  (1) motion-tolerance sweep: displace/rotate the phantom between firings and measure quality;
  (2) acquisition budget vs quality: fewer transmitters (16/24/32) and repeats at 512 receivers;
  (3) breast or limb phantom (cited fat/glandular/muscle/bone values) as the first realistic target;
  (4) rerun joint speed+density (run -31) with the fixed solver; balance the gradients; resolve the
      ~8% density-gradient mismatch;
  (5) chest: only via oracle-background distorted Born or time-domain FWI (Born series too slow
      through lung, run -34). Deprioritized for the throughput aim.
  Colab resume: see "Session 2026-10-26 saved" above (cell order matters; patch cells after the
  writefile cells). This Google account has no Drive: download the session zip before it ends.
- **Multi-frequency design (runs -37/-38, 2026-10-27, CPU):** goal = patient throughput (tank time first) + feed-forward
  FWI efficiency; data recorded to 2.5 MHz, recon to 0.8 MHz, MRI heart.
  - Ladder (`src/phase3_multifreq_ladder.py`): water start must be ≤ ~0.18 MHz (run -20's 0.3→0.8 failure was the 0.3 MHz
    start, not the jump). Safe jump ~×3 from 0.3, ~×2 from 0.5 MHz. Best 3-stage candidate 0.15>0.4>0.8 (~142 s vs 190 s,
    unverified by FWI).
  - Acquisition (`src/phase3_multifreq_fdm.py`): frequency-division multiplexed tones (K sources at once on interleaved
    1/Tw bins) isolate to ≤ −76 dB if settle ≥ 0.15 ms (> one ring crossing); tank 9.6 ms → 0.75 ms (K=32, Tw 0.5 ms)
    with ~35–40 dB more per-frequency SNR than the current pulse. Price: per-source frequency → each stage is a band
    (spread (K−1)/Tw). 2.5 MHz 'store' tones cost ~2–3 dB.
  - **FWI result (run -40):** F1 FDM (32 simultaneous tone sources, Tw 0.5 ms, **tank 0.75 ms**) bands 0.10–0.16 /
    0.37–0.43 / 0.77–0.83 MHz = same quality as its single-frequency ladder 0.15/0.4/0.8 (sep 1.40σ, heart RMSE 13.6);
    vs current 4-stage (1.47σ, 198 s, 9.6 ms): **130 s recon, 13× shorter tank**. F2 (0.35 ms, 8 low-band sources) works
    but noisier (sep 1.29, water std 3.1). Notebook `notebooks/colab_session_2026-10-27_multiplex.*` (gitignored).
  - **Follow-up (run -41):** G1 = F1 + 12 final its → 1.46σ / heart RMSE 13.35 at **0.75 ms tank, 154 s** (= current pipeline
    quality: 1.47σ / 13.38 at 9.6 ms, 198 s). NEW DEFAULT. 0.35–0.40 ms designs (F2/F3) ~10% lower separation, noisier water;
    16 vs 8 low-band sources barely helps (wide upper bands suspected). G1/F3 arrays only on the Colab runtime (not downloaded).
  - **Clinical metrics (run -42, `src/phase3_clinical_metrics.py`, re-implementation of run -36 matched on its ceiling):**
    G1 heart DSC 0.986 / ASSD 0.67 mm / HD95 3.0 mm, blood DSC 0.841 / area −2.3% — ≥ current pipeline on contours/volumes;
    texture r 0.860 vs 0.877 (fine texture is the one thing the 4th stage buys). Table in results/colab_session_2026-10-27/.
  - **Full-chain FDM (run -43):** all 32 elements transmit G1 tones simultaneously through the MRI heart (time domain),
    DFT demux (leakage ≤ −64 dB), G1 recon vs sequential under matched energy / peak and same receiver noise, 3 seeds:
    heart DSC +0.0028, blood +0.0022, RMSE −2.5% → user-proposed gate PASS (FDM slightly better: more SNR at low f).
    Part B (tank walls, element bandwidth) NOT run.
  - **Two-thigh phantom (run -44, `src/phase3_thigh_phantom.py`, outputs `*thigh*` gitignored):** traced from a T1 MRI,
    mirrored, ring 178 mm / domain 380 mm / 1520², new cited skin (Moran 1995) + marrow (Kawasaki 2015) in tissue.py.
    FDM leakage −34…−40 dB max (settle 0.3 ms marginal). G1 recon FAILED (23 min): femur not recovered, all contrasts
    under-recovered/blurred (no ghosts) → under-convergence/regularization at 2.2× FOV + bone; diagnosis pending.
    Sequential arm not run.
  - **Session 2026-10-27 saved:** results/colab_session_2026-10-27/ (both zips extracted: heart ratios incl. 204 freqs, SPW,
    run -43 FDM ratios/params, all recons, thigh FDM ratios/meta/recon, metrics JSONs).
  - **Adjoint bug fixed (run -39):** 512 receivers share pixels on coarse grids; adjoint source must use `.at[].add`
    (was `.set`). Fixed in the notebook and `src/phase3_fwi_born.py`; effect on earlier results negligible (run -40 E0).
  - Session saved (gitignored, patient-derived): `results/colab_session_2026-10-27/` — mux_results.json, mux_rec_{E0,L1,L2,
    L3,F1,F2}.npy (340² speed maps), mux_ratios_patient.npz (150-frequency time-domain data ratios), mux_recons_mri_heart.png,
    plus the original zip.
  - Colab access when the colab-mcp tool binding fails: local MCP client driver (scratchpad, not in repo) spawning the
    same server; see LOG run -39.
- **Chest reachability (run -24):** through lungs + ribs + chest wall only
  ~12–14% of the inner-wall signal survives vs water (≈ −17 dB). Lateral
  views are blocked; the front window is best (28%). A chest reconstruction
  at 30 dB would likely be near or below detection (not run). Spectra on
  Drive; figure `results/figures/phase3_chest_windows_patient.png`
  (gitignored).
- CT chest phantom (run -23): `src/phase3_ct_chest_phantom.py` →
  `results/phase3_ct_chest_phantom_patient.npz` (gitignored). Traced body
  and lungs from a published CT, schematic ribs and vertebra, patient023
  heart inserted, outside air → water. Lung/bone acoustic values still
  need citations before simulation.
- Final-stage speed-ups (run -22): blind heart-ROI unknowns from 0.5 MHz
  IMPROVE quality (4.66σ); ROI + 8 final iterations → **143 s at
  baseline quality (4.25σ)**, 3.4× faster than the original 486 s. Inner-wall contrast is
  0.13× (forward) / 0.046× (backscatter) the outer wall's. Density is
  separable from compressibility only at fine scales and only with
  multiple frequencies.

## Layout
```
fourier_inverse/
  LOG.md, MANIFEST.md
  src/
    seed.py                          -- SEED=42, numpy-only
    tissue.py                        -- cited tissue properties (cloned unchanged)
    phase0_layer_reversal_check.py   -- 1D transfer matrix + Gate 0 checks (run -01)
    phase3_born_rytov_validity.py    -- exact cylinder vs Born/Rytov (run -02)
    phase3_kspace_coverage.py        -- Fourier diffraction coverage, wall info, density split (run -03)
    phase3_blind_object_views.py     -- per-design view of a random blind object, outlines traced (run -04)
    phase3_continuation_instability.py -- 1D unique-but-unstable demo (run -04)
    phase3_adversarial_uniqueness.py -- region model/renderer + adversarial search (run -05; search STOPPED, optimizer-limited)
    phase3_local_uniqueness_jacobian.py -- local uniqueness via Jacobian SVD, 7 s (run -05)
    phase3_complexity_sweep.py       -- polygon regions w/ sharp convex+concave corners, 26-280 unknowns, ~2 min (run -06)
    phase3_blind_region_formation.py -- blind solver step 1: direct image + edge-based regions, visual, 8 s (run -07)
    phase3_gpu_helmholtz_timing.py   -- jWave GMRES vs Born-series timing; run on Colab T4 via colab-mcp (run -09)
    phase3_multifreq_ladder.py       -- frequency-ladder cycle-skip margins, straight rays, MRI heart (run -37)
    phase3_clinical_metrics.py       -- run -36 metric set (DSC/IoU/ASSD/HD95/NSD, speed errors, SSIM, texture r), conventions matched (run -42)
    phase3_multifreq_fdm.py          -- exact-cylinder FDM/tone/pulse acquisition: isolation, SNR, tank time (run -38)
    phase3_fwi_born.py               -- Born-series full-wave sound-speed inversion + reciprocity adjoint gradient (run -11);
                                        make_fast_solver/misfit_and_grad_warm = tuned PML 64/alpha 0.5 + warm start, ~40x faster (run -15)
    cbs_warm.py                      -- warm-startable Born series, modified copy of jWave 0.2.1 born_series (LGPL-3.0) (run -15; warm start BROKEN, run -16)
    cbs_ext.py                       -- Born series with an attenuation map, modified copy of jWave 0.2.1 (LGPL-3.0) (run -18)
  results/figures/phase3_fwi_patient023_realsize_attenuation.png -- truth / speed-only / speed+atten speed / atten map (run -18)
  results/figures/phase3_fwi_blind_stages.png -- first blind nonlinear reconstruction, stages + outlines (run -11)
  results/figures/phase3_fwi_fair_test.png    -- truth / inverse-crime / fair test (finer-grid data + 30 dB noise) (run -12)
  results/figures/phase3_fwi_patient023_tv_partial.png -- patient023 slice: truth / no prior / TV 1e-6 (run -13)
  results/figures/phase3_fwi_patient023_realsize_tv.png -- TRUE-size patient023 slice: truth vs TV reconstruction (run -17)
  results/phase3_patient023_slice_labels.npz  -- scaled patient023 label map (GITIGNORED, patient-derived)
  src/phase3_patient_slice.py                 -- builds that label map (NN resampling, axis-order safe) (run -13)
../.mcp.json                         -- registers googlecolab/colab-mcp (machine-specific uvx path; do not commit)
  results/
    figures/phase0_layer_reversal.png
    figures/phase3_born_rytov_validity.png
    figures/phase3_kspace_coverage.png
    figures/phase3_wall_information.png
    figures/phase3_density_separation.png
```
Gitignore: `fourier_inverse/results/*patient*.npz`, for any future
real-anatomy arrays (Phase 4). Synthetic results are committable.

## Not yet done
- Phase 3.4: SVD / null-space images for discrete-element designs.
- Gate 2 review of runs -02/-03 approximations.
- Phase 1 (1D Jacobian/SVD), Phase 2 (Fourier slice), Phase 4
  (cross-check incl. the 2ka+1 angular-sampling prediction), Phase 5.
