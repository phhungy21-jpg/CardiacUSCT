"""Freeze/verify checksums of all Phase 1-4 outputs before Phase 5 begins.

Phase 5.1 requires the pipeline (preprocessing, ground truth, Doppler
synthesis) to be fixed before any model/hyperparameter decisions are made
against ACDC CV — this makes that freeze point verifiable rather than just
asserted. Run with no args to create the manifest; run with --verify to
confirm nothing has drifted since.
"""

import hashlib
import sys
from pathlib import Path

FROZEN_DIRS = [
    Path("data/processed/ACDC"),
    Path("data/processed/ACDC_reg"),
    Path("data/processed/ACDC_doppler"),
]
FROZEN_FILES = [
    Path("results/phase3_dice.csv"),
    Path("results/phase3_quality_weights.csv"),
]
MANIFEST_PATH = Path("results/phase4_freeze_manifest.csv")


def sha256_of(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def collect_files() -> list:
    files = []
    for d in FROZEN_DIRS:
        files.extend(sorted(d.glob("*.npz")))
    files.extend(FROZEN_FILES)
    return files


def create_manifest() -> None:
    files = collect_files()
    MANIFEST_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(MANIFEST_PATH, "w") as fh:
        for f in files:
            fh.write(f"{f.as_posix()},{sha256_of(f)},{f.stat().st_size}\n")
    print(f"Froze {len(files)} files. Manifest written to {MANIFEST_PATH}")


def verify_manifest() -> None:
    if not MANIFEST_PATH.exists():
        print(f"No manifest at {MANIFEST_PATH} — run without --verify first.")
        sys.exit(1)

    recorded = {}
    for line in open(MANIFEST_PATH):
        path_str, digest, size = line.strip().split(",")
        recorded[path_str] = (digest, int(size))

    current_files = {f.as_posix(): f for f in collect_files()}
    mismatches = []
    missing = []
    new_files = [p for p in current_files if p not in recorded]

    for path_str, (digest, size) in recorded.items():
        f = current_files.get(path_str)
        if f is None:
            missing.append(path_str)
            continue
        current_digest = sha256_of(f)
        if current_digest != digest:
            mismatches.append(path_str)

    print(f"Checked {len(recorded)} frozen files.")
    print(f"Mismatched (content changed): {len(mismatches)}")
    for m in mismatches:
        print(f"  ! CHANGED: {m}")
    print(f"Missing (deleted since freeze): {len(missing)}")
    for m in missing:
        print(f"  ! MISSING: {m}")
    print(f"New files not in original freeze: {len(new_files)}")
    for n in new_files:
        print(f"  + NEW: {n}")

    if not mismatches and not missing:
        print("\nFREEZE INTACT — no drift detected.")
    else:
        print("\nFREEZE VIOLATED — investigate before trusting downstream results.")


if __name__ == "__main__":
    if "--verify" in sys.argv:
        verify_manifest()
    else:
        create_manifest()
