# Planned figures (for the eventual writeup)

Noted now so the plan survives across sessions; not generated until the
relevant phase, except where marked done.

## Tier 1 — carry the paper

1. **Pipeline schematic.** MRI -> registration (ground-truth motion) ->
   synthetic Doppler projection -> model -> recovered motion -> cross-cohort
   validation. Orients the reader, makes the real-vs-synthetic boundary
   explicit. Conceptual, not data-bound — can be drafted anytime.
2. **Dice-vs-surface-distance scatter.** Done — `results/gate3_diagnostic/dice_vs_surfdist_scatter.png`
   (see `src/gate3_dice_vs_distance_diagnostic.py`). Visual proof the Dice
   gate "failure" was metric steepness, not bad registration. Annotate the
   <=2-voxel band when finalized for the paper.
3. **Registration quality vs. contraction magnitude.** The pathology-group
   finding (DCM best, NOR/HCM worst — `LOG.md` run -06) plotted as quality
   vs. ED->ES contraction ratio. Shows the ground-truth error isn't
   independent of the signal, addressed proactively rather than found by a
   reviewer. Data exists (`results/phase3_dice.csv` + `Info.cfg` groups);
   not yet plotted as a contraction-ratio scatter (only as group means).

## Tier 2 — "show it works"

4. **Qualitative warp panel — best, median, worst case.** ED, ES, warped-ED
   overlaid on true ES, disagreement map, for 3 patients spanning the
   quality range (not cherry-picked good ones). Worst-3 version exists
   (`src/gate3_worst_case_review.py`, `results/gate3_worst_case/`); need to
   add a median and best case for the full panel.
5. **Clinical-units error translation.** The ~1-3 voxel (~1.5-4.5mm) surface
   distance rendered against real cardiac dimensions (LV/RV wall thickness,
   typical wall excursion) — speaks to a clinical/cardiologist audience
   rather than voxels. Not yet built.

## Tier 3 — after Phase 5-6

6. **Cross-cohort generalization plot.** ACDC (in-distribution) vs. M&Ms
   (held-out) performance side by side, gap quantified — the headline
   result. Cannot exist until Phase 6.
7. **Per-patient quality-weight distribution.** Shows the cohort was
   weighted, not filtered — defends that methodological choice visually.
   Data exists once `results/phase3_quality_weights.csv` is generated
   (Phase 3 completion); quick histogram, low effort whenever needed.
