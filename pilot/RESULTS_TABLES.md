# Results tables (for the preprint)

Numbers pulled directly from `LOG.md` / `results/*.csv` — cross-check against
those sources if a number here ever looks stale.

## Table 1 — Cohort summary

| Cohort | N (usable) | Role | Centers/vendors | Notes |
|---|---|---|---|---|
| ACDC | 150 | Training + in-distribution CV | 1 center | 100 training + 50 testing splits (both include ED/ES GT) |
| M&Ms | 342 / 345 | Held-out, evaluated once | 4 vendors (A, B, C, D) | 3 excluded: invalid image headers (non-orthonormal direction cosines) |

## Table 2 — Phase 5: in-distribution 5-fold CV (ACDC only)

| Method | Endpoint error (mm) | Warped-mask Dice | Beats model on n patients |
|---|---|---|---|
| **Model (3,058-param CNN)** | **0.849 ± 0.227** | **0.664 ± 0.104** | — |
| Zero-motion baseline | 3.773 ± 1.047 | 0.596 ± 0.115 | 0/150 (model wins 100%) |
| Mean-motion-field baseline | 3.151 ± 0.908 | — | 0/150 (model wins 100%) |

## Table 3 — Phase 6/7: cross-cohort comparison (the headline result)

| Metric | ACDC (n=150) | M&Ms (n=342) | Gap (M&Ms − ACDC) | 95% CI of gap | Cohen's d |
|---|---|---|---|---|---|
| Endpoint error (mm) | 0.849 [0.814, 0.887] | 0.717 [0.704, 0.731] | −0.133 | [−0.172, −0.095] (excludes 0) | −0.819 |
| Warped-mask Dice | 0.664 [0.647, 0.680] | 0.589 [0.578, 0.599] | −0.075 | [−0.095, −0.055] (excludes 0) | −0.740 |

**Read together, not separately:** endpoint error says M&Ms is robustly
*better*; Dice says M&Ms is robustly *worse* (the expected direction). This
disagreement is the key diagnostic — see `LIMITATIONS.md`'s named confound
(M&Ms's own registration ground truth is lower quality, Demons Dice 0.641 vs
ACDC's 0.752).

## Table 4 — Vendor breakdown (M&Ms only — the clean result)

| Vendor | N | Endpoint error (mm) | 95% CI |
|---|---|---|---|
| A | 95 | 0.720 | [0.696, 0.746] |
| B | 125 | 0.736 | [0.713, 0.765] |
| C | 72 | 0.707 | [0.686, 0.727] |
| D | 50 | 0.674 | [0.653, 0.696] |

Range across vendors: 0.062mm. No outlier vendor. Not subject to the
ACDC-vs-M&Ms ground-truth-quality confound (within-M&Ms comparison).

## Table 5 — Registration quality by pathology group (ACDC, Phase 3)

| Group | N | Mean Dice | % passing Dice≥0.80 |
|---|---|---|---|
| DCM | 30 | 0.847 | 77% |
| MINF | 30 | 0.780 | 37% |
| RV pathology | 30 | 0.759 | 23% |
| NOR (healthy) | 30 | 0.705 | 0% |
| HCM | 30 | 0.667 | 0% |

Pearson correlation, registration Dice vs. LV contraction magnitude: **r = −0.777**
(`results/figures/quality_vs_contraction.png`) — large deformation, not
disease label per se, drives registration difficulty.

## Table 6 — M&Ms data-quality fixes applied

| Issue | Patients affected | Fix |
|---|---|---|
| Label convention reversed (1=LV/2=myo/3=RV vs. ACDC's 1=RV/2=myo/3=LV) | All 345 | Explicit remap before any processing |
| CSV placeholder ED=ES=0 (real frames elsewhere) | 25 | Auto-detect true labelled frames from mask; assign ED/ES by LV cavity size |
| Invalid image header (non-orthonormal direction cosines) | 3 | Excluded (all one vendor/centre; documented, not silently dropped) |

## Figures generated

| File | Content |
|---|---|
| `results/figures/pipeline_schematic.png` | Method overview (Tier 1) |
| `results/gate3_diagnostic/dice_vs_surfdist_scatter.png` | Dice-vs-surface-distance, all 3 labels (Tier 1) |
| `results/figures/quality_vs_contraction.png` | Registration quality vs. contraction magnitude, by pathology group (Tier 1) |
| `results/figures/warp_panel_best_median_worst.png` | Qualitative best/median/worst registration (Tier 2) |
| `results/figures/clinical_units_error.png` | Boundary error vs. real cardiac wall dimensions (Tier 2) |
| `results/figures/cross_cohort_result.png` | ACDC vs M&Ms, both metrics side by side (Tier 3) |
| `results/figures/vendor_breakdown.png` | 4-vendor consistency (Tier 3) |
| `results/figures/quality_weight_distribution.png` | Per-patient quality weight, all 150 retained (Tier 3) |
| `results/phase5_train_val_curves.png` | Overfitting check, 5 folds (supporting) |
