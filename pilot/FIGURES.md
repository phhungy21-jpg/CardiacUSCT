# Writeup figures — status

All planned figures generated (2026-07-07). See `RESULTS_TABLES.md` for the
full file list and companion tables. Regenerate any figure by re-running its
`src/fig_*.py` script — all are deterministic given the frozen results CSVs.

## Tier 1 — carry the paper

1. **Pipeline schematic** — `results/figures/pipeline_schematic.png` (`src/fig_pipeline_schematic.py`). Fully programmatic, no manual redraw needed.
2. **Dice-vs-surface-distance scatter** — `results/gate3_diagnostic/dice_vs_surfdist_scatter.png` (`src/gate3_dice_vs_distance_diagnostic.py`).
3. **Registration quality vs. contraction magnitude** — `results/figures/quality_vs_contraction.png` (`src/fig_quality_vs_contraction.py`). r = -0.777.

## Tier 2 — "show it works"

4. **Qualitative warp panel, best/median/worst** — `results/figures/warp_panel_best_median_worst.png` (`src/fig_warp_panel_best_median_worst.py`). patient106 (0.92) / patient022 (0.75) / patient142 (0.52).
5. **Clinical-units error translation** — `results/figures/clinical_units_error.png` (`src/fig_clinical_units.py`). Cites ASE 2015 chamber quantification guideline (RV free wall <=5mm, LV wall 6-10mm normal, >15mm HCM threshold).

## Tier 3 — after Phase 5-6

6. **Cross-cohort generalization** — `results/figures/cross_cohort_result.png` (`src/fig_cross_cohort.py`) + `results/figures/vendor_breakdown.png` (`src/fig_vendor_and_weights.py`).
7. **Per-patient quality-weight distribution** — `results/figures/quality_weight_distribution.png` (`src/fig_vendor_and_weights.py`).

## Not generated / manual-drawing note

None of the above required manual drawing — all are matplotlib output driven
by logged numbers. If a more polished pipeline diagram is wanted for a final
submission (e.g. icons, camera-ready styling), that's a design upgrade to the
existing programmatic version, not something that needed hand-drawing from
scratch.
