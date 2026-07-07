# Preprint package

- `manuscript.docx` — full draft manuscript (Times New Roman 11pt), converted from `WRITEUP.md` via pandoc.
- `results_tables.docx` — the 6 results tables, standalone, converted from `RESULTS_TABLES.md`.
- `figures/` — all 9 figures as standalone PNGs (see `FIGURES.md` in the repo root for what each shows).
- `phase3_quality_weights.csv`, `phase5_cv_results.csv`, `phase6_evaluation.csv` — the underlying per-patient result data behind the tables/figures.

## Regenerating manuscript.docx / results_tables.docx

```
pandoc --print-default-data-file reference.docx > preprint_package/_default_reference.docx
python src/build_docx_template.py         # sets Times New Roman 11pt on the template
pandoc WRITEUP.md -o preprint_package/manuscript.docx --reference-doc=preprint_package/_default_reference.docx --standalone
pandoc RESULTS_TABLES.md -o preprint_package/results_tables.docx --reference-doc=preprint_package/_default_reference.docx --standalone
rm preprint_package/_default_reference.docx
```

## Before actually submitting (not yet done — see LOG.md 2026-07-07 for full detail)

1. **Author names and affiliations** — placeholders in `manuscript.docx`'s title block; fill in.
2. **References** — full bibliographic details (exact author lists, journal volume/pages) for the tissue-Doppler and ASE-guideline citations were not independently re-verified beyond the source title/PMID; verify against the original records. The ACDC/M&Ms dataset citations should also be double-checked.
3. **Venue formatting** — this is currently a general-purpose Word draft, not laid out to STACOM/MICCAI's LNCS page/column format. If submitting there, the text will need reflowing into that template (content itself doesn't need to change).
4. Word/page count has not been checked against any specific venue's limit.
