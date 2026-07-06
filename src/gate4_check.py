"""Protocol Gate 4 — noiseless recovery test.

Fit (a closed-form per-voxel least-squares solve, not a learned model — the
point here is to sanity-check the projection geometry itself, not to train
anything yet) on the NOISELESS synthetic Doppler signal. It should recover
the true in-plane displacement field almost exactly. Also sanity-checks
tensor shapes/units and sign correctness of the radial projection.
"""

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from doppler import (  # noqa: E402
    probe_unit_vectors_at_voxel,
    recover_displacement_least_squares,
    synthesize_doppler,
)

REG_DIR = Path("data/processed/ACDC_reg")
TEST_PATIENTS = ["patient001", "patient002", "patient101", "patient029"]


def main() -> None:
    print("=== Sign/unit sanity check on a synthetic translation ===")
    fake_field = np.zeros((1, 5, 5, 3), dtype=np.float32)
    fake_field[..., 2] = 3.0  # dx = +3mm (rightward), dy=dz=0
    spacing = (1.5625, 1.5625, 10.0)
    proj = synthesize_doppler(fake_field, spacing, add_noise=False)
    unit_vecs = probe_unit_vectors_at_voxel((5, 5), (spacing[0], spacing[1]))
    center_voxel_projs = proj[:, 0, 2, 2]
    center_unit_vecs = unit_vecs[:, 2, 2, :]
    expected = center_unit_vecs[:, 0] * 3.0  # dot((3,0), unit_vec)
    print(f"Pure +x=3mm translation: probe projections = {center_voxel_projs}")
    print(f"Expected (unit_x * 3mm) = {expected}")
    print(f"Match: {np.allclose(center_voxel_projs, expected, atol=1e-4)}\n")

    print("=== Noiseless recovery test (closed-form least-squares) ===")
    for pid in TEST_PATIENTS:
        d = np.load(REG_DIR / f"{pid}.npz")
        disp_field = d["displacement_field"]  # (Z, Y, X, 3)
        spacing = tuple(d["spacing"])

        proj_noiseless = synthesize_doppler(disp_field, spacing, add_noise=False)
        unit_vecs = probe_unit_vectors_at_voxel(disp_field.shape[1:3], (spacing[0], spacing[1]))
        recovered = recover_displacement_least_squares(proj_noiseless, unit_vecs)  # (Z, Y, X, 2)

        true_xy = disp_field[..., [2, 1]]  # (Z, Y, X, 2) matching (dx, dy) order
        error = np.sqrt(((recovered - true_xy) ** 2).sum(axis=-1))  # (Z, Y, X) mm

        print(f"{pid}: shape={disp_field.shape}, recovery error (mm): "
              f"mean={error.mean():.6f} max={error.max():.6f}")

    print("\nTensor shape convention: synthesize_doppler returns (n_probes=3, Z, Y, X) mm displacement.")
    print("Ground-truth target for Phase 5 is disp_field[..., [2,1]] = (dx, dy) in-plane, mm — dz (through-plane) is NOT targeted, per the documented through-plane resolution limitation.")


if __name__ == "__main__":
    main()
