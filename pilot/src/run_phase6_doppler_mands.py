"""Phase 4-equivalent for M&Ms — exact frozen probe geometry/noise config
from src/doppler.py, no changes. Generates noisy projections (the model
input) and in-plane targets (for evaluation) for all 342 M&Ms patients."""

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from doppler import synthesize_doppler  # noqa: E402
from seed import set_all_seeds  # noqa: E402

REG_DIR = Path("data/processed/MandMs_reg")
OUT_DIR = Path("data/processed/MandMs_doppler")


def main() -> None:
    set_all_seeds()
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(42)

    files = sorted(REG_DIR.glob("*.npz"))
    for f in files:
        pid = f.stem
        d = np.load(f)
        disp_field = d["displacement_field"]
        spacing = tuple(d["spacing"])

        proj_noisy = synthesize_doppler(disp_field, spacing, add_noise=True, rng=rng)
        target_xy = disp_field[..., [2, 1]].astype(np.float32)

        np.savez_compressed(
            OUT_DIR / f"{pid}.npz",
            proj_noisy=proj_noisy, target_xy=target_xy,
            spacing=spacing, vendor=str(d["vendor"]),
        )

    print(f"Generated Doppler projections for {len(files)} M&Ms patients -> {OUT_DIR}")


if __name__ == "__main__":
    main()
