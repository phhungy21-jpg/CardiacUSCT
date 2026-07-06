"""Protocol Gate 3 requirement: visualize the worst 3 patients and understand
why registration failed for them, rather than just noting a low Dice number."""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

PROC_DIR = Path("data/processed/ACDC")
REG_DIR = Path("data/processed/ACDC_reg")
OUT_DIR = Path("results/gate3_worst_case")

WORST = ["patient142", "patient104", "patient029"]


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    for pid in WORST:
        proc = np.load(PROC_DIR / f"{pid}.npz")
        reg = np.load(REG_DIR / f"{pid}.npz")

        ed_frame, es_frame = proc["ed_frame"], proc["es_frame"]
        ed_mask, es_mask = proc["ed_mask"], proc["es_mask"]
        warped_mask = reg["warped_ed_mask"]

        z = ed_frame.shape[0] // 2
        fig, axes = plt.subplots(1, 4, figsize=(20, 5))

        axes[0].imshow(ed_frame[z], cmap="gray")
        axes[0].imshow(np.ma.masked_where(ed_mask[z] == 0, ed_mask[z]), cmap="jet", alpha=0.4)
        axes[0].set_title(f"{pid}: ED + true ED mask (source)")

        axes[1].imshow(es_frame[z], cmap="gray")
        axes[1].imshow(np.ma.masked_where(es_mask[z] == 0, es_mask[z]), cmap="jet", alpha=0.4)
        axes[1].set_title("ES + true ES mask (target)")

        axes[2].imshow(es_frame[z], cmap="gray")
        axes[2].imshow(np.ma.masked_where(warped_mask[z] == 0, warped_mask[z]), cmap="jet", alpha=0.4)
        axes[2].set_title(f"ES + warped ED->ES mask (mean Dice={float(reg['mean_dice']):.2f})")

        diff = (es_mask[z] > 0).astype(int) - (warped_mask[z] > 0).astype(int)
        axes[3].imshow(es_frame[z], cmap="gray")
        axes[3].imshow(diff, cmap="bwr", alpha=0.5, vmin=-1, vmax=1)
        axes[3].set_title("Disagreement (red=missed, blue=extra)")

        for ax in axes:
            ax.axis("off")

        out_path = OUT_DIR / f"{pid}_worst_case.png"
        fig.savefig(out_path, dpi=140, bbox_inches="tight")
        plt.close(fig)
        print(f"Saved {out_path}")

        # Also check: how much does the heart's z-extent / in-plane position shift ED->ES?
        ed_z_range = np.nonzero(ed_mask.sum(axis=(1, 2)))[0]
        es_z_range = np.nonzero(es_mask.sum(axis=(1, 2)))[0]
        print(f"  {pid}: ED mask present in slices {ed_z_range.min()}-{ed_z_range.max()}, "
              f"ES mask present in slices {es_z_range.min()}-{es_z_range.max()}")


if __name__ == "__main__":
    main()
