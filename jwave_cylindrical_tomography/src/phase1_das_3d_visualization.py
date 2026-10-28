"""Slice-by-slice qualitative visualization of the first 3D DAS
reconstruction (run -10) against patient023's real ground-truth
anatomy -- a cheap, quick sanity-check companion to the quantitative
evaluation already done in `phase1_das_3d_evaluation.py`. Not required
to trust that result (already established independently), but makes
the outer-boundary match visible.

Pure post-processing on already-saved arrays -- no new computation.
"""

import numpy as np
import matplotlib.pyplot as plt

EVAL_NPZ = r"D:\Cardiac USCT\jwave_cylindrical_tomography\results\das_3d_evaluation_patient023.npz"
OUT_PNG = r"D:\Cardiac USCT\jwave_cylindrical_tomography\results\figures\phase1_das_3d_reconstruction_patient023.png"

if __name__ == "__main__":
    d = np.load(EVAL_NPZ)
    accumulator = d["accumulator"]
    gt_coarse = d["gt_coarse"]
    boundary = d["boundary"]
    n = accumulator.shape[0]

    # pick 4 z-slices through the heart's own extent (not the empty water margin)
    tissue_z = np.where(gt_coarse.any(axis=(0, 1)))[0]
    z_slices = np.linspace(tissue_z.min(), tissue_z.max(), 4).astype(int)

    fig, axes = plt.subplots(2, 4, figsize=(16, 8))
    vmax = np.percentile(accumulator, 99.5)

    for col, z in enumerate(z_slices):
        ax_gt = axes[0, col]
        ax_gt.imshow(gt_coarse[:, :, z].T, origin="lower", cmap="gray")
        ax_gt.contour(boundary[:, :, z].T, colors="red", linewidths=1)
        ax_gt.set_title(f"Ground truth, z={z}")
        ax_gt.axis("off")

        ax_acc = axes[1, col]
        ax_acc.imshow(accumulator[:, :, z].T, origin="lower", cmap="inferno", vmin=0, vmax=vmax)
        ax_acc.contour(boundary[:, :, z].T, colors="cyan", linewidths=1)
        ax_acc.set_title(f"DAS reconstruction, z={z}")
        ax_acc.axis("off")

    axes[0, 0].set_ylabel("Ground truth\n(red = true boundary shell)")
    axes[1, 0].set_ylabel("DAS energy\n(cyan = true boundary shell)")
    fig.suptitle("patient023 static ED, 3D cylindrical multistatic DAS reconstruction (run -10)\n"
                 "outer tissue/water boundary: energy visibly concentrates near the true surface", fontsize=12)
    fig.tight_layout()
    fig.savefig(OUT_PNG, dpi=150)
    print(f"Saved {OUT_PNG}")
