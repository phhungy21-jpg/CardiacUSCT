# Manifest

Snapshot of what's actually on disk. Update this when data/results/structure
changes materially — it's the fastest way for a new session (human or Claude)
to get oriented without re-deriving state from scratch.

_Last updated: 2026-07-06 (Phase 3 complete, final resolution — see below)_

## Code (tracked in git)

| Path | Purpose |
|---|---|
| `cardiac_motion_pilot_protocol.md` | Source of truth protocol — phases + validation gates |
| `Cardiac_Motion_Pilot_Protocol.docx` | Original doc export of the protocol (reference only, prefer the .md) |
| `README.md` | Setup + orientation for humans and future sessions |
| `MANIFEST.md` | This file |
| `LOG.md` | Hypothesis, success criteria, phase checklist, per-run log — update every run |
| `LIMITATIONS.md` | Pre-written scope/limitations notes for the eventual paper |
| `requirements.txt` | Pinned dependencies |
| `.gitignore` | Excludes `data/`, `venv/`, caches |
| `src/__init__.py` | Package marker |
| `src/seed.py` | `set_all_seeds()` — call before any split/init/training |
| `src/gate1_check.py`, `src/phase1_scan.py` | Phase 1: single-patient overlay check, full-cohort integrity scan |
| `src/preprocessing.py`, `src/run_phase2_preprocess.py`, `src/gate2_check.py` | Phase 2: canonical resample/crop/normalize pipeline + validation |
| `src/registration.py` | Phase 3: Demons/B-spline registration, Dice/surface-distance/diffeomorphism metrics |
| `src/run_phase3_registration_maskguided.py` | Phase 3 driver (current/adopted method) — LV-only mask-guided Demons |
| `src/run_phase3_registration.py` | Phase 3 driver (superseded) — intensity-only Demons, kept for historical comparison (`results/phase3_dice.csv`) |
| `src/phase3_smoke_test.py`, `src/phase3_diagnostic.py`, `src/phase3_mask_guided_test.py` | Phase 3 method-tuning scripts (6-patient dev set, not full-cohort validation) |
| `src/gate3_worst_case_review.py`, `src/gate3_myocardium_band_check.py` | Phase 3 Gate 3 validation — worst-case visual review, endocardial-band Dice decomposition |

`notebooks/` exists but is empty — all Phase 1-3 work so far is in `src/`
scripts, not notebooks.

## Data (gitignored — never committed, regenerate/redownload per-machine)

| Path | Contents | Size |
|---|---|---|
| `data/ACDC/ACDC/database/training/` | 100 ACDC patients, each with 4D cine volume, ED/ES frames + GT masks, `Info.cfg` | — |
| `data/ACDC/ACDC/database/testing/` | 50 ACDC patients, same structure (includes GT masks, unlike the original challenge's held-out testing split) | — |
| `data/ACDC/` (total) | | ~2.3 GB |
| `data/processed/ACDC/patientXXX.npz` | Phase 2 output: ED/ES frames + masks, resampled to (1.5625,1.5625,10.0)mm, LPS, 128x128 crop, z-scored. 150/150 patients. | — |
| `data/processed/ACDC_reg/patientXXX.npz` | Phase 3 output (final): displacement_field (Z,Y,X,3), warped_ed_mask, per-label Dice/surf-dist — intensity-only diffeomorphic Demons, full heart (RV+myo+LV), all 150 patients. The earlier LV-only mask-guided variant was tried and superseded (see LOG.md runs -07 through -11) once Gate 3's metric was corrected. | — |
| `results/phase3_quality_weights.csv` | Per-patient quality weight (`1/(1+mean_surface_dist_voxels)`) for Phase 5 training — down-weights, doesn't filter. | — |
| `data/processed/ACDC_doppler/patientXXX.npz` | Phase 4 output: `proj_noiseless`/`proj_noisy` (3 probes, Z, 128, 128) mm, `target_xy` (Z, 128, 128, 2) mm in-plane displacement, `quality_weight`. 150/150 patients. | — |
| `data/M&Ms/` | **Not yet downloaded** — Phase 1.2 step 2, deferred to Phase 6 to avoid contamination | — |

## Environment

- Python 3.11.9, venv at `venv/` (gitignored, machine-local — recreate with
  `python -m venv venv && pip install -r requirements.txt`)
- Installed versions pinned in `requirements.txt`; exact resolved set is in
  `pip freeze` output, currently matches pins exactly except transitive deps
  (torch 2.0.1 is CPU-only build on this machine).

## Protocol phase status

See the checklist in `LOG.md` for the authoritative up-to-date state.
- Phase 1 (env + data access): done, ACDC only.
- Phase 2 (preprocessing): done, ACDC only.
- Phase 3 (ground-truth motion): done. Full heart, all 150 patients
  retained, no exclusion. Gate 3's primary metric was corrected from Dice
  to surface distance (98.7% of patients have <=2 voxel mean boundary error
  — the earlier 27% Dice pass rate was a metric-steepness artifact for
  thin structures, not evidence of bad motion fields; both readings are
  documented in `LOG.md` for transparency). Per-patient quality weight
  (surface-distance-based) carried into Phase 5 instead of filtering — see
  `results/phase3_quality_weights.csv` and `LIMITATIONS.md`.
- Phase 4 (synthetic Doppler): done. 3 in-plane probes, noise SD=1.0mm
  (cited from tissue Doppler displacement reproducibility literature, not
  invented). Gate 4 (noiseless recovery) passed at machine precision.
- Phase 5 (model training, in-distribution CV): done. 3,058-param CNN,
  5-fold CV within ACDC, beats zero-motion and mean-motion baselines on
  100% of patients (endpoint error 0.85mm vs 3.1-3.8mm). No overfitting.
  This is the frozen ACDC-only config — no further tuning against M&Ms.
- Phase 6 (cross-cohort evaluation): done. 342/345 M&Ms patients usable
  (label-convention swap fixed, ED/ES CSV placeholder fallback added, 3
  patients excluded for genuine invalid headers). Endpoint error 0.717mm on
  M&Ms vs. 0.849mm on ACDC in-distribution — no drop, but the ground-truth
  quality confound (M&Ms registration Dice 0.641 vs ACDC's 0.752) is not
  ruled out, so this is not presented as clean proof of generalization.
- M&Ms: loaded at `data/MandMs/` (345 patients, 4 vendors). Data pipeline:
  `data/processed/MandMs/` (Phase 2), `MandMs_reg/` (Phase 3), `MandMs_doppler/` (Phase 4).
