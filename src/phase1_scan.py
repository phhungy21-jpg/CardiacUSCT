"""Phase 1.2/1.3 — confirm every ACDC patient has usable ED/ES volumes + masks.

Iterates all patients (training + testing), parses Info.cfg for ED/ES frame
numbers, loads each volume/mask pair, and checks: files present, shapes
match, no NaNs, spacing recorded. Flags anomalies rather than crashing, so
one bad patient doesn't block the scan of the rest.
"""

import configparser
from pathlib import Path

import nibabel as nib
import numpy as np

ROOT = Path("data/ACDC/ACDC/database")
SPLITS = ["training", "testing"]


def load_info(patient_dir: Path) -> dict:
    cfg_text = "[info]\n" + (patient_dir / "Info.cfg").read_text()
    parser = configparser.ConfigParser()
    parser.read_string(cfg_text)
    return dict(parser["info"])


def main() -> None:
    flags = []
    spacings = []
    n_ok = 0
    n_total = 0

    for split in SPLITS:
        split_dir = ROOT / split
        patient_dirs = sorted(p for p in split_dir.iterdir() if p.is_dir() and p.name.startswith("patient"))

        for patient_dir in patient_dirs:
            n_total += 1
            pid = patient_dir.name
            try:
                info = load_info(patient_dir)
                ed, es = int(info["ed"]), int(info["es"])

                for label, frame in (("ED", ed), ("ES", es)):
                    vol_path = patient_dir / f"{pid}_frame{frame:02d}.nii.gz"
                    mask_path = patient_dir / f"{pid}_frame{frame:02d}_gt.nii.gz"
                    if not vol_path.exists() or not mask_path.exists():
                        flags.append(f"{pid} [{split}] {label}: missing file(s)")
                        continue

                    vol_img = nib.load(vol_path)
                    mask_img = nib.load(mask_path)
                    vol = vol_img.get_fdata()
                    mask = mask_img.get_fdata()

                    if vol.shape != mask.shape:
                        flags.append(f"{pid} [{split}] {label}: shape mismatch vol={vol.shape} mask={mask.shape}")
                    if not np.allclose(vol_img.affine, mask_img.affine):
                        flags.append(f"{pid} [{split}] {label}: affine mismatch")
                    if np.isnan(vol).any():
                        flags.append(f"{pid} [{split}] {label}: NaNs in volume")
                    if np.unique(mask).size < 2:
                        flags.append(f"{pid} [{split}] {label}: mask has < 2 labels (empty?)")

                    spacings.append((pid, split, label, vol_img.header.get_zooms()))

                n_ok += 1
            except Exception as e:
                flags.append(f"{pid} [{split}]: exception during load — {e}")

    print(f"Scanned {n_total} patients ({n_ok} loaded without exception)")

    unique_spacings = sorted(set(s[3] for s in spacings))
    print(f"\nDistinct (x,y,z) spacings across ED+ES volumes: {len(unique_spacings)}")
    for sp in unique_spacings:
        count = sum(1 for s in spacings if s[3] == sp)
        print(f"  {sp} — {count} volumes")

    print(f"\nFlags: {len(flags)}")
    for f in flags:
        print(f"  ! {f}")

    if not flags:
        print("  none")


if __name__ == "__main__":
    main()
