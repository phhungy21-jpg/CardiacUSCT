# Phase II Protocol — Acoustic-Simulation Layer (k-Wave / jWave)

**A lab-notebook-style working protocol**
**Builds on:** Phase I pilot (`cardiac_motion_pilot_protocol.md`, completed)
**Setting:** Collaboration (Yale cardiac-AI lab) — not a solo sprint
Version 0.1 · Working draft

---

## How to use this document

Same discipline as the Phase I pilot: work top to bottom, don't pass a validation gate until it's genuinely passed, log everything. Two things are different this time, and they matter:

1. **The failure modes are now invisible.** In Phase I a bad registration showed up as a bad Dice — errors were visible. In acoustic simulation, a physically-wrong setup produces *plausible-looking but meaningless* results. Most gates below exist specifically to catch silent unphysical errors. Do not skip them because the output "looks fine."
2. **This is a collaboration.** Decisions get discussed before they're committed, code is read by others, and you are the domain/clinical half — not the acoustic-physics half. Several gates are about *alignment with collaborators*, not just correctness.

The north star: this phase does not try to prove the idea works on real USCT. It characterizes how the approach behaves under real wave physics — a necessary rung before real hardware, and a stronger signal than the Phase I MRI proxy. Success = an honest map of what survives and what breaks, tied to physical causes. Not a clean win.

---

## Phase 0 — Scope, hypothesis, and collaboration setup (do before any code)

### The one-sentence hypothesis
> A learned motion-recovery model, trained and evaluated on *acoustically simulated* ultrasound signals (k-Wave/jWave) of moving cardiac anatomy, retains usable accuracy for bulk myocardial motion — and its failure modes (shadowing, thin structures, large deformation) can be characterized against the physical conditions that cause them.

### What this phase is NOT
- Not a real-USCT study (no hardware, no real acoustic measurements)
- Not proof the approach works on real hardware — simulation is another, tighter, upper bound
- Not a clinical claim
- Not an attempt to "win" — a characterized partial result is the target

### Reframe from Phase I (state explicitly, it's the whole point)
Phase I used a geometric projection that was *exactly invertible* (Gate 4 recovered motion at machine precision). That crutch is now removed: real wave propagation is lossy, scattering and attenuation destroy information, and the motion→signal relationship is no longer clean-linear. **Expect degradation relative to Phase I's 0.85mm. Degradation is the physics working, not a bug.**

### Pre-registered success criteria (decide now, before results)
- **Encouraging:** bulk LV motion recovery survives simulation with degraded-but-usable accuracy; failure modes map cleanly to physical causes → escalate to real-USCT collaboration ask.
- **Informative-negative:** recovery fails, but *why* is clearly characterized (which physical conditions defeat it) → still publishable, and redirects rather than escalates. A cheap "no" that prevents an expensive one.
- **Broken:** the simulation itself is unphysical / the pipeline has a silent error → not a scientific result, a setup problem to fix before anything is interpreted.

### Collaboration alignment (the new Phase 0 work)
Before writing code, settle with the Yale team:
- **Roles.** You bring clinical/domain judgment, the Phase I pipeline, and honest-scoping discipline. They bring acoustic-physics expertise and compute. Write down who owns what — especially who is accountable for the *physical correctness* of the simulation setup (this is the highest-risk, most-specialist part).
- **Their workflow.** How does the lab use git, run experiments, review code, track runs? Adopt their conventions rather than importing your solo-sprint habits. Ask explicitly.
- **Compute budget.** Get a real number from them for the planned scope (see Phase 4). GPU-hours are their resource; scope to it deliberately.
- **Definition of done.** Agree what result would justify the real-USCT escalation vs. what would redirect. Pre-committing this keeps the eventual interpretation honest.

### ⛔ Gate 0
- [ ] Hypothesis and success criteria written down and agreed with collaborators.
- [ ] Roles assigned; physical-correctness ownership explicitly named.
- [ ] Lab's workflow/conventions understood and adopted.
- [ ] Rough compute budget known.

---

## Phase 1 — Environment, tooling, and reproducibility parity

### 1.1 Prepare
- Decide k-Wave vs jWave (or both). Rule of thumb: **jWave** (JAX, differentiable) if you'll want physics-consistency losses / end-to-end learned reconstruction later; **k-Wave** if you want the mature, well-validated standard for the forward simulation itself. Many groups use k-Wave to generate data and a separate learned model downstream — confirm the lab's preference.
- GPU access confirmed and tested (a trivial simulation runs on their hardware).
- Repo hygiene carried over from Phase I: seeds, pinned dependencies, a running `LOG.md`, a `MANIFEST.md`. Same standards that made Phase I collaboration-ready.

### 1.2 Do
- Install and run the toolkit's own example/benchmark simulation end to end on the lab's GPU. Confirm it completes and reproduces the toolkit's documented reference output.

### 1.3 ⛔ Gate 1
- [ ] A known reference simulation from the toolkit's docs reproduces the documented result on your hardware. (If you can't reproduce a *known-correct* example, nothing downstream is trustworthy.)
- [ ] Seeds/versions pinned; a collaborator could clone and run.
- [ ] You can time a single simulation — you'll need this to estimate the full run (Phase 4).

> ⚠️ **Silent-failure pitfall starts here.** Never trust a simulation you haven't validated against something with a known answer. Reproducing the toolkit's own reference case is the first such check.

---

## Phase 2 — Acoustic model definition (the highest-risk phase)

This is where unphysical setups hide. Spend disproportionate care and lean hardest on the collaborator who owns physical correctness.

### 2.1 Prepare
- **Tissue acoustic properties.** Assign sound speed and density (and attenuation, and scattering behavior) to each tissue class from your Phase I segmentations — myocardium, blood, fat, etc. Use *cited literature values*, not invented ones (same discipline as the Phase I noise calibration). Document every value and its source.
- **Transducer geometry.** Define the array: element positions, frequency, transmit scheme. Justify geometry against real transthoracic acoustic windows (carry over the anterior-arc rationale from Phase I — no posterior window).
- **Grid resolution + timestep.** Set grid spacing fine enough to resolve the wavelength; set timestep to satisfy the stability (CFL/Courant) condition. Getting this wrong silently diverges or produces garbage.
- **Dimensionality decision.** Start 2D. Full 3D is the eventual target but is far heavier (Phase 4) and much easier to get wrong. 2D first, always.

### 2.2 Do
- Build the forward model for a single slice / single transmit as a fully specified, documented configuration.

### 2.3 ⛔ Gate 2 (the load-bearing physical-correctness gate)
- [ ] **Stability satisfied:** the simulation does not diverge; the CFL condition is met and documented.
- [ ] **Sanity physics on a known case:** simulate a trivial known configuration (e.g. a point source in homogeneous medium) and confirm the wavefield behaves correctly — right propagation speed, right geometry, no artifacts. *This is the acoustic analogue of Phase I's Gate 4 noiseless-recovery check: verify the machinery on something with a known answer before trusting it on real anatomy.*
- [ ] **Every acoustic property has a cited source.** No invented tissue values.
- [ ] **A collaborator with acoustic-physics expertise has reviewed and signed off on the setup.** This gate is not passable by you alone — it's the specialist half of the collaboration. Do not proceed on your own judgment here.

> ⚠️ **The silent killer of this phase:** an unphysical simulation that runs cleanly and produces plausible images that are *meaningless*. Everything downstream (motion recovery, characterization, the escalation decision) is built on this being physically correct. If Gate 2 is shaky, stop — a beautiful result on a wrong simulation is worse than no result.

---

## Phase 3 — Toy proof-of-concept: simulate → recover on a simple moving phantom

Before touching real cardiac anatomy, prove the whole loop works on something simple and controllable. This mirrors the Phase I advice to smoke-test before the full run.

### 3.1 Prepare
- A simple 2D moving phantom (e.g. a pulsating disk with known, prescribed motion — not real anatomy). You control the ground-truth motion exactly.

### 3.2 Do
1. Simulate the acoustic signal the array would record as the phantom moves (Phase 2 forward model).
2. Attempt to recover the prescribed motion from that simulated signal (your Phase I model, retrained, or a suitable variant).
3. Compare recovered vs. prescribed (known) motion.

### 3.3 ⛔ Gate 3
- [ ] On the toy phantom with *mild* acoustic realism, motion recovery is meaningfully better than a naive baseline. (If it can't recover known motion on a simple controlled case, the approach won't survive real anatomy — stop and diagnose.)
- [ ] You've dialed acoustic realism from gentle → aggressive and watched recovery degrade — confirming the *dial* works and you understand its effect. (This dial is your main experimental instrument in Phase 5.)

> ⚠️ **Pitfall:** succeeding on a toy phantom that's secretly too easy (too little attenuation, no shadowing) tells you nothing. Make sure the toy includes at least *some* of the lossy physics, or you've just rebuilt Phase I's invertible projection with extra steps.

---

## Phase 4 — Scale to cardiac anatomy (compute-heavy; scope to budget)

### 4.1 Prepare
- **Benchmark-then-multiply.** Time one representative cardiac-anatomy simulation on the lab GPU. Multiply by (transmit events × patients × conditions) to get the real compute estimate. Bring this number to the collaborators *before* committing — don't discover the budget mid-run.
- Decide scope honestly against budget: how many patients, 2D-slices vs 3D, how many acoustic conditions. It's better to characterize a smaller set thoroughly than to run a huge set you can't validate.
- Reuse Phase I anatomy (ACDC/M&Ms segmentations + registration-derived motion) as the moving-tissue input — carry over the ground-truth-quality caveats and quality weights.

### 4.2 Do
- Generate the simulated acoustic dataset across the planned patients/conditions.
- **Resumable driver.** Long GPU runs *will* be interrupted — make the driver skip already-completed cases (the lesson from Phase I's sleep/interrupt concern, now with real stakes because runs are long and expensive).

### 4.3 ⛔ Gate 4
- [ ] Compute estimate agreed with collaborators before the full run; actual usage tracked against it.
- [ ] Spot-check simulated outputs on several patients for physical sanity (a collaborator reviews) — not just "it ran."
- [ ] No silent patient loss / no cases where the simulation diverged unnoticed.
- [ ] Ground-truth-quality caveats from Phase I carried forward, not forgotten.

> ⚠️ **Pitfall:** the compute multiplier is what surprises people — one simulation is cheap, the full study is heavy because of the count. Scope to what you can validate and afford, not the maximum.

---

## Phase 5 — The actual study: characterize degradation vs. physical conditions

This is the scientific core, and the framing matters more than the numbers. **The paper is a characterization, not a validation.** Don't ask "does it work" — ask "how does accuracy degrade as a function of acoustic realism, and where does it break."

### 5.1 Prepare
- Define the conditions to sweep: shadowing/dropout severity, attenuation level, noise/speckle level, deformation magnitude, structure (LV vs RV vs thin/deep walls).
- Pre-register (as in Phase I) what "survives" vs "breaks" means numerically, before seeing results.
- Reuse Phase I's honest-stats discipline: bootstrap CIs, effect sizes, no headline p-value at low N, per-condition and per-structure breakdowns.

### 5.2 Do
- Evaluate motion recovery across the swept conditions. Produce the *map*: accuracy as a function of physical condition and anatomical region.

### 5.3 ⛔ Gate 5
- [ ] Results are reported as a characterization (what survives, what breaks, tied to physical cause) — not a binary claim.
- [ ] Every degradation is attributed to a physical cause where possible (shadowing → these regions; large deformation → these patients), not left as unexplained.
- [ ] The sim-to-real gap is stated plainly: this is simulation, a tighter upper bound than Phase I, still not real hardware.
- [ ] No overclaim: "survives acoustic simulation" is never written as "will work on real USCT."

> ⚠️ **The framing pitfall:** if you go in wanting a clean "it works," a partial result feels like failure and you'll be tempted to spin it. Go in wanting the *map* — then degradation is the result, and it's the credible, publishable, escalation-justifying one.

---

## Phase 6 — Interpretation, writeup, and the escalation decision

### 6.1 Prepare
- Assemble the characterization into the same honest structure as Phase I: scope, methods, results-with-caveats-in-the-same-breath, limitations, future work.
- Target venue with collaborators (methods venue / imaging-physics journal — depends on result strength).

### 6.2 The escalation decision (pre-committed in Gate 0)
Apply the ladder honestly:
- **Survives / informative-partial** → escalate: propose the real-USCT validation to a hardware-holding group, now as a *de-risked, specific* ask ("here's simulation evidence; real hardware is the one remaining thing that can't be simulated").
- **Fails informatively** → do not escalate; the cheap "no" did its job. Redirect. Still write it up — a well-characterized negative in an under-explored area is a real contribution.

### 6.3 ⛔ Gate 6
- [ ] Writeup leads with honest framing, names the sim-to-real gap, doesn't overclaim.
- [ ] The escalate/redirect decision follows the pre-committed criteria, not the emotional pull of the result.
- [ ] Collaborators have reviewed and co-own the interpretation (this is joint work now).

---

## Appendix A — What's different from the Phase I pilot

| | Phase I (pilot) | Phase II (this) |
|---|---|---|
| Compute | Laptop CPU | GPU / cluster (collaborator's) |
| Failure visibility | Errors visible (bad Dice) | Errors *invisible* (unphysical but plausible) — gates exist to catch this |
| Physics | None (geometric projection, invertible) | Real wave propagation (lossy, the crutch removed) |
| Who validates correctness | You alone | You + acoustic-physics collaborator (Gate 2 needs their signoff) |
| Working mode | Solo evening sprint | Collaboration — discuss before committing, adopt lab conventions |
| Expected result | Near-perfect (0.85mm) | Degraded, characterized — that's the point |
| Purpose | Prove you can execute | Tighten the upper bound; decide whether real USCT is worth asking for |

## Appendix B — Master pitfall table

| Phase | Pitfall | Consequence |
|---|---|---|
| 0 | Skipping role/correctness-ownership alignment | Nobody accountable for physical correctness — the highest-risk gap |
| 1 | Trusting a simulation without reproducing a known reference | Building on an unvalidated tool |
| 2 | Invented tissue acoustic values | Physically-wrong sim that looks fine |
| 2 | CFL/stability violated | Silent divergence or garbage |
| 2 | Proceeding without acoustic-expert signoff | Beautiful result on a wrong simulation |
| 3 | Toy phantom secretly too easy | Rebuilds Phase I's invertible projection, learns nothing |
| 4 | Not benchmarking compute before the full run | Blows the budget mid-study |
| 4 | Non-resumable driver on long GPU runs | Lose expensive compute to interruptions |
| 5 | Framing as "does it work" not "how does it degrade" | Weak paper, tempted to spin partial results |
| 5 | Overclaiming sim → real | Loses credibility with the exact experts you want to collaborate with |
| 6 | Escalating on emotion not pre-committed criteria | Asking for scarce USCT resources on a weak basis |

## Appendix C — Running log template

```
### Run YYYY-MM-DD-##
- Phase:
- Seed / config / grid / timestep:
- Acoustic properties + sources:
- Compute used (GPU-hours) vs budget:
- Result (metric = value ± CI):
- Physical sanity checked? by whom?:
- Gate passed? (Y/N):
- Observations / surprises:
- Next action:
```

---

## The one thing to carry from Phase I

The pilot was credible because it refused to overclaim, characterized its own limits honestly, and caught its own errors before reviewers could. Phase II is heavier, collaborative, and physics-laden — but the same discipline is what will make it good. The acoustic physics is new; the intellectual honesty that made the last one work is not. Bring that, lean on the collaborators for the physics, and let the characterization — not a clean win — be the result.

*End of protocol v0.1. Revise as reality (and the collaborators) push back.*
