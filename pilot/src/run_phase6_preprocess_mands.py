"""Preprocess all M&Ms patients into the same canonical representation as
ACDC (Gate 2 cross-cohort check + data prep for Phase 6)."""

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from preprocessing_mands import load_metadata, preprocess_mands_patient  # noqa: E402

OUT_DIR = Path("data/processed/MandMs")


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    rows = load_metadata()
    n_ok, failures = 0, []

    for i, row in enumerate(rows):
        code = row["External code"]
        try:
            result = preprocess_mands_patient(row)
            np.savez_compressed(
                OUT_DIR / f"{code}.npz",
                ed_frame=result["ed_frame"], es_frame=result["es_frame"],
                ed_mask=result["ed_mask"], es_mask=result["es_mask"],
                ed_idx=result["ed_idx"], es_idx=result["es_idx"],
                spacing=result["spacing"], split=result["split"],
                vendor=result["vendor"], centre=result["centre"], pathology=result["pathology"],
            )
            n_ok += 1
        except Exception as e:
            failures.append((code, str(e)))

        if (i + 1) % 50 == 0:
            print(f"  ... {i + 1}/{len(rows)} done", flush=True)

    print(f"\nPatients in: {len(rows)}, processed: {n_ok}, failed: {len(failures)}")
    for code, err in failures:
        print(f"  ! {code}: {err}")


if __name__ == "__main__":
    main()
