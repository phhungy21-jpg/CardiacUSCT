"""Figure: cross-cohort generalization (Tier 3, FIGURES.md #6) — the
headline result. Leads with endpoint error (no drop), shows Dice
disagreeing in direction (the key diagnostic), per the binding reporting
guidance in LOG.md (2026-07-07)."""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

# From LOG.md run 2026-07-07-04 (Phase 7 bootstrap stats)
DATA = {
    "Endpoint error (mm)": {
        "ACDC": (0.849, 0.814, 0.887), "M&Ms": (0.717, 0.704, 0.731),
    },
    "Warped-mask Dice": {
        "ACDC": (0.664, 0.647, 0.680), "M&Ms": (0.589, 0.578, 0.599),
    },
}


def main() -> None:
    fig, axes = plt.subplots(1, 2, figsize=(10, 4.5))

    for ax, (metric, cohorts) in zip(axes, DATA.items()):
        labels = list(cohorts.keys())
        means = [cohorts[k][0] for k in labels]
        lo = [cohorts[k][0] - cohorts[k][1] for k in labels]
        hi = [cohorts[k][2] - cohorts[k][0] for k in labels]
        colors = ["#4c72b0", "#dd8452"]
        ax.bar(labels, means, yerr=[lo, hi], capsize=6, color=colors, width=0.55)
        ax.set_title(metric)
        ax.set_ylabel(metric.split(" (")[0])

    fig.suptitle("ACDC (in-distribution) vs. M&Ms (held-out): metrics disagree on direction of the gap", fontsize=11)
    axes[0].text(0.5, -0.22, "No degradation (CI excludes zero, favors M&Ms) --\nsee ground-truth-quality confound, LIMITATIONS.md",
                 transform=axes[0].transAxes, ha="center", fontsize=8, color="#555")
    axes[1].text(0.5, -0.22, "Modest drop, expected direction\n(CI excludes zero, favors ACDC)",
                 transform=axes[1].transAxes, ha="center", fontsize=8, color="#555")
    fig.tight_layout()
    out = Path("results/figures/cross_cohort_result.png")
    fig.savefig(out, dpi=160, bbox_inches="tight")
    print(f"Saved {out}")


if __name__ == "__main__":
    main()
