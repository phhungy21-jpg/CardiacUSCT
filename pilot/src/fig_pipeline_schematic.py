"""Figure: pipeline schematic (Tier 1, FIGURES.md #1). Fully programmatic —
no manual drawing needed."""

import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch

STAGES = [
    ("Cine MRI\n(ACDC / M&Ms)", "#c8d9ea"),
    ("Registration\n(ED→ES displacement\nfield = ground truth)", "#cfe8cf"),
    ("Synthetic Doppler\nprojection\n(3 probes + noise)", "#f3e0b8"),
    ("Small CNN\n(3,058 params)", "#e6cbe9"),
    ("Recovered\nmotion field", "#cfe8cf"),
    ("Cross-cohort\nvalidation\n(ACDC → M&Ms)", "#f0b8b8"),
]


def main() -> None:
    fig, ax = plt.subplots(figsize=(15, 3.2))
    box_w, box_h, gap = 2.0, 1.3, 0.6
    y = 0.5

    for i, (label, color) in enumerate(STAGES):
        x = i * (box_w + gap)
        box = FancyBboxPatch(
            (x, y), box_w, box_h, boxstyle="round,pad=0.05,rounding_size=0.08",
            linewidth=1.2, edgecolor="black", facecolor=color, zorder=2,
        )
        ax.add_patch(box)
        ax.text(x + box_w / 2, y + box_h / 2, label, ha="center", va="center", fontsize=10, zorder=3)
        if i < len(STAGES) - 1:
            arrow = FancyArrowPatch(
                (x + box_w, y + box_h / 2), (x + box_w + gap, y + box_h / 2),
                arrowstyle="-|>", mutation_scale=18, linewidth=1.4, color="black", zorder=1,
            )
            ax.add_patch(arrow)

    ax.text(0.5 * box_w, y - 0.35, "real (patient data)", ha="center", fontsize=8, style="italic", color="#555")
    ax.text(2.5 * (box_w + gap), y - 0.35, "synthetic / proxy (no acoustic physics)", ha="center", fontsize=8, style="italic", color="#555")
    ax.text(5 * (box_w + gap) + 0.5 * box_w, y - 0.35, "held-out, evaluated once", ha="center", fontsize=8, style="italic", color="#555")

    ax.set_xlim(-0.3, len(STAGES) * (box_w + gap))
    ax.set_ylim(-0.6, y + box_h + 0.3)
    ax.axis("off")
    ax.set_title("Pipeline: real MRI-derived ground truth → synthetic Doppler proxy → learned recovery → cross-cohort check", fontsize=11)
    fig.tight_layout()
    fig.savefig("results/figures/pipeline_schematic.png", dpi=180, bbox_inches="tight")
    print("Saved results/figures/pipeline_schematic.png")


if __name__ == "__main__":
    main()
