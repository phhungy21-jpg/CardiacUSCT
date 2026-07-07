"""Figure: registration error in clinical/anatomical context (Tier 2,
FIGURES.md #5). Cites ASE 2015 chamber quantification guideline ranges:
normal LV wall thickness 6-10mm, RV free wall <=5mm, HCM threshold >15mm."""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

# From LOG.md run 2026-07-06-11 (surface distance diagnostic, run -06 intensity-only Demons cohort)
SURF_DIST_MEDIAN_MM = {"RV": 1.67, "myocardium": 0.69, "LV": 0.71}
SURF_DIST_P90_MM = {"RV": 3.77, "myocardium": 1.51, "LV": 1.79}

ANATOMY_REFS = [
    ("RV free wall thickness\n(normal, ASE 2015)", 0, 5, "#8ecae6"),
    ("LV wall thickness\n(normal, ASE 2015)", 6, 10, "#8ecae6"),
    ("LV wall thickness\n(HCM threshold, ASE 2015)", 15, 15, "#f4a261"),
]


def main() -> None:
    fig, ax = plt.subplots(figsize=(9, 5))

    y_positions = {"RV": 3, "myocardium": 2, "LV": 1}
    for label, y in y_positions.items():
        ax.barh(y, SURF_DIST_MEDIAN_MM[label], height=0.4, color="#4c72b0", label="Median surface distance" if y == 3 else None)
        ax.barh(y, SURF_DIST_P90_MM[label] - SURF_DIST_MEDIAN_MM[label], left=SURF_DIST_MEDIAN_MM[label],
                height=0.4, color="#a3c4e8", label="p90 surface distance" if y == 3 else None)
        ax.text(SURF_DIST_P90_MM[label] + 0.15, y, label, va="center", fontsize=10)

    for name, lo, hi, color in ANATOMY_REFS:
        if lo == hi:
            ax.axvline(lo, color="#e76f51", linestyle="--", linewidth=1.3)
            ax.text(lo + 0.1, 4.3, name, fontsize=8, color="#e76f51", rotation=90, va="bottom")
        else:
            ax.axvspan(lo, hi, alpha=0.15, color="#2a9d8f")
            ax.text((lo + hi) / 2, 4.3, name, fontsize=8, color="#2a9d8f", ha="center")

    ax.set_yticks([])
    ax.set_xlabel("mm")
    ax.set_xlim(0, 17)
    ax.set_ylim(0, 5)
    ax.set_title("Registration boundary error vs. real cardiac wall dimensions")
    ax.legend(loc="lower right")
    fig.tight_layout()
    out = Path("results/figures/clinical_units_error.png")
    fig.savefig(out, dpi=160)
    print(f"Saved {out}")


if __name__ == "__main__":
    main()
