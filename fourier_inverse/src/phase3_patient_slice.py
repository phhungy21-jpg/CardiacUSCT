"""Build a blind test object from a REAL heart cross-section (patient023, ED).

ACDC labels: 0 background, 1 RV cavity, 2 myocardium, 3 LV cavity (no atria
in ACDC). Mid-ventricular slice = the middle slice among those containing
all three labels. The heart is cropped and scaled DOWN to fit inside the
64 mm test domain's ring (radius 28 mm), so it is a scaled toy of real
anatomy, not real-size (caveat logged). Nearest-neighbour resampling only
(CLAUDE.md: masks never resample linearly).

Axis convention (caught in jwave_cylindrical run -04): `spacing` is
(row, col, slice), while `ed_mask` axes are (slice, row, col).

Output: results/phase3_patient023_slice_labels.npz (gitignored: *patient*)
plus a compact base64 text for embedding in the Colab notebook (scratch).
"""

import base64
import sys
from pathlib import Path

import numpy as np
from scipy import ndimage

ROOT = Path(__file__).resolve().parents[2]
OUT = Path(__file__).resolve().parent.parent / "results"
L_MM, NF = 64.0, 256          # fine (data) grid of the fair test
FIT_DIAMETER_MM = 44.0        # inside the 28 mm-radius ring with margin


def main(b64_out):
    d = np.load(ROOT / "pilot/data/processed/ACDC/patient023.npz")
    mask = d["ed_mask"]
    row_mm, col_mm = float(d["spacing"][0]), float(d["spacing"][1])
    full = [k for k in range(mask.shape[0]) if set(np.unique(mask[k])) >= {1, 2, 3}]
    k = full[len(full) // 2]
    sl = mask[k]
    rows, cols = np.nonzero(sl)
    sl = sl[rows.min():rows.max() + 1, cols.min():cols.max() + 1]
    size_mm = np.array(sl.shape) * [row_mm, col_mm]
    scale = FIT_DIAMETER_MM / np.hypot(*size_mm)
    print(f"slice {k} of {mask.shape[0]} (slices with LV+RV+myo: {full}); "
          f"in-plane spacing {row_mm:.3f} x {col_mm:.3f} mm; heart bbox {size_mm.round(1)} mm; "
          f"scaled by {scale:.3f} to fit {FIT_DIAMETER_MM} mm")
    dx = L_MM / NF
    z = ndimage.zoom(sl, (row_mm * scale / dx, col_mm * scale / dx), order=0)   # nearest-neighbour
    lab = np.zeros((NF, NF), np.uint8)
    r0, c0 = (NF - z.shape[0]) // 2, (NF - z.shape[1]) // 2
    lab[r0:r0 + z.shape[0], c0:c0 + z.shape[1]] = z
    area = {name: (lab == v).sum() * dx ** 2 for v, name in [(1, "RV"), (2, "myo"), (3, "LV")]}
    print("areas (mm^2, scaled):", {kk: round(vv, 1) for kk, vv in area.items()},
          "| labels present:", np.unique(lab))
    OUT.mkdir(exist_ok=True)
    np.savez(OUT / "phase3_patient023_slice_labels.npz", labels=lab, slice=k, scale=scale)
    packed = np.packbits(np.unpackbits(lab[..., None], axis=-1)[..., -2:].ravel())   # 2 bits/pixel
    Path(b64_out).write_text(base64.b64encode(packed).decode())
    print(f"embedded size: {len(base64.b64encode(packed))} base64 chars")


if __name__ == "__main__":
    main(sys.argv[1])
