"""Phase 4 driver — generate synthetic Doppler projections (noiseless +
noisy) for every ACDC patient, using Phase 3's displacement fields. Saves
model-ready input/target pairs for Phase 5. Seeded for reproducibility."""

import csv
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from doppler import synthesize_doppler  # noqa: E402
from seed import set_all_seeds  # noqa: E402

REG_DIR = Path("data/processed/ACDC_reg")
OUT_DIR = Path("data/processed/ACDC_doppler")
WEIGHTS_CSV = Path("results/phase3_quality_weights.csv")


def main() -> None:
    set_all_seeds()
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    weights = {r["patient_id"]: float(r["quality_weight"]) for r in csv.DictReader(open(WEIGHTS_CSV))}

    files = sorted(REG_DIR.glob("patient*.npz"))
    rng = np.random.default_rng(42)
    n_done = 0

    for f in files:
        pid = f.stem
        d = np.load(f)
        disp_field = d["displacement_field"]
        spacing = tuple(d["spacing"])

        proj_noiseless = synthesize_doppler(disp_field, spacing, add_noise=False)
        proj_noisy = synthesize_doppler(disp_field, spacing, add_noise=True, rng=rng)
        target_xy = disp_field[..., [2, 1]].astype(np.float32)  # (Z, Y, X, 2) = (dx, dy), in-plane only

        np.savez_compressed(
            OUT_DIR / f"{pid}.npz",
            proj_noiseless=proj_noiseless,
            proj_noisy=proj_noisy,
            target_xy=target_xy,
            quality_weight=weights.get(pid, 1.0),
            spacing=spacing,
            split=str(d["split"]),
        )
        n_done += 1

    print(f"Generated Doppler projections for {n_done}/{len(files)} patients -> {OUT_DIR}")
    sample = np.load(OUT_DIR / f"{files[0].stem}.npz")
    print(f"Shapes: proj_noiseless={sample['proj_noiseless'].shape}, "
          f"proj_noisy={sample['proj_noisy'].shape}, target_xy={sample['target_xy'].shape}")


if __name__ == "__main__":
    main()
