"""Phase 2 driver — preprocess every ACDC patient (ED/ES only) and save to
data/processed/ACDC/. M&Ms is intentionally not touched here (see protocol
Phase 5.3: no M&Ms until Phase 6, to avoid contaminating the held-out test).
"""

from pathlib import Path

import numpy as np

import sys

sys.path.insert(0, str(Path(__file__).resolve().parent))
from preprocessing import preprocess_patient  # noqa: E402

RAW_ROOT = Path("data/ACDC/ACDC/database")
OUT_ROOT = Path("data/processed/ACDC")
SPLITS = ["training", "testing"]


def main() -> None:
    OUT_ROOT.mkdir(parents=True, exist_ok=True)
    n_in, n_out = 0, 0
    failures = []

    for split in SPLITS:
        split_dir = RAW_ROOT / split
        patient_dirs = sorted(p for p in split_dir.iterdir() if p.is_dir() and p.name.startswith("patient"))

        for patient_dir in patient_dirs:
            n_in += 1
            try:
                result = preprocess_patient(patient_dir)
                out_path = OUT_ROOT / f"{result['patient_id']}.npz"
                np.savez_compressed(
                    out_path,
                    ed_frame=result["ed_frame"],
                    es_frame=result["es_frame"],
                    ed_mask=result["ed_mask"],
                    es_mask=result["es_mask"],
                    ed_idx=result["ed_idx"],
                    es_idx=result["es_idx"],
                    spacing=result["spacing"],
                    split=split,
                )
                n_out += 1
            except Exception as e:
                failures.append((patient_dir.name, split, str(e)))

    print(f"Patients in: {n_in}, processed successfully: {n_out}, failed: {len(failures)}")
    for pid, split, err in failures:
        print(f"  ! {pid} [{split}]: {err}")


if __name__ == "__main__":
    main()
