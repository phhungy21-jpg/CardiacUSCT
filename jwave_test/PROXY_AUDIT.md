# Proxy acoustic-physics audit (solo-dev stand-in — NOT Gate 2)

**This is Claude acting as a technical stand-in reviewer, for this solo
dev run only.** It is explicitly NOT a substitute for Gate 2's real
requirement — a collaborator with acoustic-physics expertise reviewing and
signing off on the setup (per `acoustic_simulation_phase_protocol.md` and
`CLAUDE.md`: *"it is not passable by Claude or the user alone"*). Nothing
here changes that. What this document does: goes through every item in
`PHASE2_TO_PHASE3_DIAGNOSTIC_HANDOFF.md` with actual numbers/fixes instead
of leaving them as open flags, so that (a) solo development can continue
further into Phase 4 with real technical grounding, and (b) the eventual
real collaborator review starts from a much better-prepared position.
**Every result below still carries `labels.PENDING_SIGNOFF_BANNER`.**

## Item 1 — Points-per-wavelength: CHECKED, margin adequate

Computed the toneburst's actual spectrum (not just the carrier): at
f0=2.5MHz, -6dB bandwidth is 1.67–3.33MHz, -20dB bandwidth is
0.83–4.17MHz. Points-per-wavelength (PPW) at dx=0.1mm, c_min=1576m/s
(myocardium, slowest of our three tissues):

| frequency | PPW |
|---|---|
| f0 (2.5MHz) | 6.30 |
| -6dB upper edge (3.33MHz) | 4.73 |
| -20dB upper edge (4.17MHz) | 3.78 |

Even at the -20dB bandwidth edge, PPW=3.78 — above the Nyquist floor (2)
and within the commonly-cited adequate margin for k-space/pseudospectral
methods (≥3 PPW; these methods don't suffer classical FD numerical
dispersion, unlike finite-difference schemes needing PPW~10+). **Verdict:
adequate for this config, not certain — a real reviewer should confirm
this holds for Phase 4's actual interface geometries**, which are more
complex than this project's idealized circle/ring phantoms.

## Item 2 — PML: CHECKED via full domain-crossing run, adequate

Previous checks (Phase 2/3) deliberately truncated `t_end` before the wave
reached the domain edge — meaning PML was literally never tested. Ran a
homogeneous 300×300 case to FULL domain-crossing time (1409 steps) and
measured residual pressure in the interior during the last 20% of steps
(after the wavefront should have crossed and been absorbed):

- Peak pressure over the full run: 0.996 (source amplitude)
- Max residual interior pressure, final 20% of steps: 5.86e-5
- That's **-75dB relative to peak** — i.e. any energy reflecting back off
  the PML boundary is ~75dB below the source, far below the weakest real
  signal this project measures (the blood/myocardium reflection was
  ~-44dB, `../jwave/LOG.md` run -05). **Verdict: PML is not contaminating
  results at this config.**

## Item 3 — Source-amplitude calibration: PROPOSED, not yet confirmed

Anchored the arbitrary source amplitude to a physically-grounded value
via the FDA's Mechanical Index (MI = derated peak rarefactional pressure
in MPa / sqrt(f_MHz), regulatory limit MI≤1.9). Typical MEASURED
transthoracic cardiac imaging MI is ~0.18 (fundamental) / ~0.25
(harmonic) — see `src/calibration.py` for the implementation and
citations. At f0=2.5MHz with a representative clinical MI=0.2: derated
peak pressure = 0.2×√2.5 = **0.316 MPa**, i.e. our arbitrary amplitude=1.0
now maps to ≈316,228 Pa. **This is a proposal, not a confirmed
calibration** — a real reviewer should confirm whether anchoring to a
"typical clinical MI" (vs. a specific target device's measured output) is
appropriate for this study's eventual claims.

## Item 4 — Attenuation: FIXED AND VALIDATED (the big one)

Implemented `src/attenuation_solver.py`: a reimplementation of jWave's own
time-domain scan loop (same `momentum_conservation_rhs`/
`mass_conservation_rhs`/`pressure_from_density` operators, imported
directly, not duplicated) with one addition — per-step exponential
damping of BOTH the density and velocity fields, converting cited dB/cm
attenuation values to Nepers/m (assuming linear-with-frequency scaling,
y=1, a documented simplification vs. real tissue's y≈1.1–1.5).

**Validated against the analytic law** (`src/validate_attenuation.py`):
ran identical geometry with/without attenuation, measured the ratio at
increasing receiver distances (canceling the common geometric-spreading
factor). First attempt was wrong by ~2x (bug: only density was damped,
not velocity — the undamped velocity field kept re-injecting energy each
step). After damping both fields:

| distance from ref | observed ratio | analytic exp(-α·Δr) |
|---|---|---|
| +3mm | 0.9373 | 0.9381 |
| +6mm | 0.8801 | 0.8800 |
| +9mm | 0.8262 | 0.8256 |

Match within ~0.1% at every distance. **This closes the load-bearing gap**
— jWave's transient solver can now model attenuation, not just accept the
parameter inertly. Still a simplification (frequency-independent power
law), still needs a real reviewer's confirmation that y=1 is adequate —
but it is no longer "not implemented at all."

## Item 5 — Motion injection: already resolved (Phase 3, run -07)

Null test passed cleanly (recovered-radius std = 0.0000mm at zero
motion). No new work needed here.

## Item 6 — Staircasing: QUANTIFIED on real anatomy

Discovered while checking this: the Phase 4.1 benchmark's N=150 and
N=250 crops of patient001's real anatomy were **entirely inside the LV
cavity** (single label, no tissue boundary at all) — only N=350 actually
contained a myocardium/blood boundary. This is now corrected in
`phase4_benchmark.py`'s documentation (timing/memory numbers are
unaffected — jWave doesn't optimize for homogeneity, so the benchmark
itself is still valid — but the "real anatomy" framing for those two
points was overstated).

Re-ran the staircasing check on N=350 (genuine boundary): compared the
raw nearest-neighbor tissue map against a lightly Gaussian-smoothed
version (sigma=2 grid cells, ~0.2mm) of the SAME real boundary. Relative
L2 difference in the resulting wavefield: **0.59%**. Small, but not zero
— staircasing does perturb the field measurably on real (not idealized
circular) anatomy. **Verdict: small effect at this resolution, worth a
real reviewer's judgment on whether 0.6% is acceptable for this study's
claims, especially compounded across many boundaries/patients.**

## Bottom line

All six items now have real numbers or real fixes behind them, not just
flags. Item 4 (attenuation) went from "not implemented" to "implemented
and validated to ~0.1%." None of this is Gate 2 — it's the best a solo
technical audit can do, and it's now a much stronger starting point for
the real collaborator conversation than the original diagnostic handoff
alone.
