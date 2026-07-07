"""Protocol Gate 2 — validate processed ACDC output before moving to Phase 3.

Checks: spacing/array-convention uniformity across processed patients,
mask still aligned after resample+crop (visual), patient in/out count,
no NaNs. M&Ms is intentionally out of scope here (see Phase 5.3 — no
peeking until Phase 6); the cross-cohort half of this gate is deferred.
"""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

PROC_DIR = Path("data/processed/ACDC")
OUT_DIR = Path("results/gate2")


def overlay(volume: np.ndarray, mask: np.ndarray, slice_idx: int, title: str, out_path: Path) -> None:
    fig, ax = plt.subplots(figsize=(5, 5))
    ax.imshow(volume[slice_idx], cmap="gray")
    ax.imshow(np.ma.masked_where(mask[slice_idx] == 0, mask[slice_idx]), cmap="jet", alpha=0.4)
    ax.set_title(title)
    ax.axis("off")
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    files = sorted(PROC_DIR.glob("patient*.npz"))
    print(f"Processed patients on disk: {len(files)}")

    spacings = set()
    shapes = set()
    nan_flags = []

    for f in files:
        d = np.load(f)
        spacings.add(tuple(d["spacing"]))
        shapes.add(d["ed_frame"].shape[1:])  # in-plane (Y, X); Z varies by patient FOV, expected
        if np.isnan(d["ed_frame"]).any() or np.isnan(d["es_frame"]).any():
            nan_flags.append(f.name)

    print(f"Distinct spacings across processed patients: {spacings} (should be exactly 1 tuple)")
    print(f"Distinct in-plane (Y, X) shapes: {shapes} (should be exactly 1 — the OUT_SIZE crop)")
    print(f"Patients with NaNs: {len(nan_flags)} {nan_flags}")

    for pid in ["patient001", "patient101"]:
        f = PROC_DIR / f"{pid}.npz"
        if not f.exists():
            continue
        d = np.load(f)
        z = d["ed_frame"].shape[0] // 2
        overlay(d["ed_frame"], d["ed_mask"], z, f"{pid} ED processed (slice {z})", OUT_DIR / f"{pid}_ED_processed.png")
        overlay(d["es_frame"], d["es_mask"], z, f"{pid} ES processed (slice {z})", OUT_DIR / f"{pid}_ES_processed.png")
        print(f"Saved overlays for {pid}")


if __name__ == "__main__":
    main()
