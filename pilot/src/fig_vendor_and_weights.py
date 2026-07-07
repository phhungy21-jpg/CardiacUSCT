"""Figures: vendor breakdown (the clean within-M&Ms result) and per-patient
quality-weight distribution (Tier 3, FIGURES.md #7 + the vendor half of #6)."""

import csv
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

VENDOR_STATS = {  # from LOG.md run 2026-07-07-04
    "A": (0.720, 0.696, 0.746), "B": (0.736, 0.713, 0.765),
    "C": (0.707, 0.686, 0.727), "D": (0.674, 0.653, 0.696),
}


def fig_vendor() -> None:
    fig, ax = plt.subplots(figsize=(6, 4.5))
    labels = list(VENDOR_STATS.keys())
    means = [VENDOR_STATS[k][0] for k in labels]
    lo = [VENDOR_STATS[k][0] - VENDOR_STATS[k][1] for k in labels]
    hi = [VENDOR_STATS[k][2] - VENDOR_STATS[k][0] for k in labels]
    ax.bar(labels, means, yerr=[lo, hi], capsize=6, color="#55a868", width=0.6)
    ax.set_xlabel("M&Ms vendor")
    ax.set_ylabel("Endpoint error (mm)")
    ax.set_title("Consistent accuracy across 4 vendors (within-M&Ms comparison)")
    ax.set_ylim(0, 0.85)
    fig.tight_layout()
    out = Path("results/figures/vendor_breakdown.png")
    fig.savefig(out, dpi=160)
    print(f"Saved {out}")


def fig_quality_weights() -> None:
    rows = list(csv.DictReader(open("results/phase3_quality_weights.csv")))
    weights = np.array([float(r["quality_weight"]) for r in rows])

    fig, ax = plt.subplots(figsize=(7, 4.8))
    ax.hist(weights, bins=20, color="#4c72b0", edgecolor="white")
    ax.axvline(weights.mean(), color="#c44e52", linestyle="--", label=f"mean = {weights.mean():.3f}")
    ax.set_xlabel("Per-patient quality weight (1 / (1 + mean surface distance in voxels))")
    ax.set_ylabel("Number of patients")
    ax.set_title(f"Quality-weight distribution — all {len(weights)} ACDC patients\nretained, not filtered", fontsize=12)
    ax.legend()
    fig.tight_layout()
    out = Path("results/figures/quality_weight_distribution.png")
    fig.savefig(out, dpi=160)
    print(f"Saved {out}")


if __name__ == "__main__":
    fig_vendor()
    fig_quality_weights()
