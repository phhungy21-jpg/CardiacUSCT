"""Figure: qualitative warp panel — best, median, worst registration
(Tier 2, FIGURES.md #4). Not cherry-picked good cases only — shows the
full quality range."""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

PROC_DIR = Path("data/processed/ACDC")
REG_DIR = Path("data/processed/ACDC_reg")

CASES = [("patient106", "Best", 0.918), ("patient022", "Median", 0.753), ("patient142", "Worst", 0.522)]


def main() -> None:
    fig, axes = plt.subplots(3, 4, figsize=(16, 12))
    col_titles = ["ED + true mask", "ES + true mask", "ES + warped-ED mask", "Disagreement (red=missed, blue=extra)"]

    for row, (pid, label, dice) in enumerate(CASES):
        proc = np.load(PROC_DIR / f"{pid}.npz")
        reg = np.load(REG_DIR / f"{pid}.npz")
        ed_frame, es_frame = proc["ed_frame"], proc["es_frame"]
        ed_mask, es_mask = proc["ed_mask"], proc["es_mask"]
        warped_mask = reg["warped_ed_mask"]
        z = ed_frame.shape[0] // 2

        axes[row, 0].imshow(ed_frame[z], cmap="gray")
        axes[row, 0].imshow(np.ma.masked_where(ed_mask[z] == 0, ed_mask[z]), cmap="jet", alpha=0.4)

        axes[row, 1].imshow(es_frame[z], cmap="gray")
        axes[row, 1].imshow(np.ma.masked_where(es_mask[z] == 0, es_mask[z]), cmap="jet", alpha=0.4)

        axes[row, 2].imshow(es_frame[z], cmap="gray")
        axes[row, 2].imshow(np.ma.masked_where(warped_mask[z] == 0, warped_mask[z]), cmap="jet", alpha=0.4)

        diff = (es_mask[z] > 0).astype(int) - (warped_mask[z] > 0).astype(int)
        axes[row, 3].imshow(es_frame[z], cmap="gray")
        axes[row, 3].imshow(diff, cmap="bwr", alpha=0.5, vmin=-1, vmax=1)

        axes[row, 0].set_ylabel(f"{label}\n(mean Dice={dice:.2f})", fontsize=12, rotation=0, labelpad=70, va="center")

        for col in range(4):
            axes[row, col].set_xticks([])
            axes[row, col].set_yticks([])
            if row == 0:
                axes[row, col].set_title(col_titles[col], fontsize=11)

    fig.suptitle("Registration quality across the full range: best, median, worst case (ACDC)", fontsize=13)
    fig.tight_layout()
    out = Path("results/figures/warp_panel_best_median_worst.png")
    fig.savefig(out, dpi=150, bbox_inches="tight")
    print(f"Saved {out}")


if __name__ == "__main__":
    main()
