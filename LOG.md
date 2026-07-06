# Lab Notebook — Cardiac Motion Pilot Study

**Reference Protocol:** cardiac_motion_pilot_protocol.md  
**Version:** 0.1  
**Start Date:** 2026-07-06

---

## Hypothesis

> A model trained to recover cardiac tissue displacement from synthetic multi-angle Doppler-like velocity projections — using ground-truth motion derived from single-center cine MRI — retains accuracy when evaluated on a different center/vendor's MRI-derived motion.

---

## Success Criteria (Pre-registered)

- **Encouraging:** cross-cohort accuracy drop is modest and the model clearly beats a naïve baseline
- **Null-but-useful:** large cross-cohort drop — still publishable as "here's why this is harder than it looks"
- **Broken:** in-distribution model fails even on its own test fold → pipeline/design problem, not a real finding

---

## Phase Progress

- [ ] Phase 1 — Environment and data access
- [ ] Phase 2 — Preprocessing and normalization
- [ ] Phase 3 — Ground-truth motion extraction
- [ ] Phase 4 — Synthetic Doppler signal generation
- [ ] Phase 5 — Model training and in-distribution evaluation
- [ ] Phase 6 — Cross-cohort evaluation
- [ ] Phase 7 — Metrics, statistics, and reporting
- [ ] Phase 8 — Writeup and preprint

---

## Run History

(Each run: date, phase, config, result, notes)

