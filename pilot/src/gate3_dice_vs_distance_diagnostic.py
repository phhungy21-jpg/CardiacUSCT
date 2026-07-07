"""Diagnostic-only re-analysis of results/phase3_dice.csv (run 2026-07-06-06,
intensity-only Demons, full 150-patient cohort). Does NOT re-run registration.

Question: does the 27% Dice pass-rate reflect genuinely bad motion fields,
or a Dice-vs-thin-structure metric artifact (low Dice but small, ~1-2 voxel
surface distance)?
"""

import csv
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

CSV_PATH = Path("results/phase3_dice.csv")
OUT_DIR = Path("results/gate3_diagnostic")
IN_PLANE_SPACING_MM = 1.5625
LABELS = {"rv": "RV", "myo": "myocardium", "lv": "LV"}


def main() -> None:
    rows = list(csv.DictReader(open(CSV_PATH)))
    n = len(rows)
    data = {}
    for key in LABELS:
        data[f"dice_{key}"] = np.array([float(r[f"dice_{key}"]) for r in rows])
        data[f"dist_{key}"] = np.array([float(r[f"dist_{key}_mm"]) for r in rows])

    print(f"n = {n} patients (run 2026-07-06-06, intensity-only Demons)\n")

    # 1. Surface distance distribution per label, mm and voxels
    print("=== 1. Surface distance distribution per label ===")
    for key, name in LABELS.items():
        d_mm = data[f"dist_{key}"]
        d_vox = d_mm / IN_PLANE_SPACING_MM
        p25, p50, p75, p90 = np.percentile(d_mm, [25, 50, 75, 90])
        p25v, p50v, p75v, p90v = np.percentile(d_vox, [25, 50, 75, 90])
        print(f"{name}:")
        print(f"  mm:     median={p50:.2f}  IQR=[{p25:.2f}, {p75:.2f}]  p90={p90:.2f}")
        print(f"  voxels: median={p50v:.2f}  IQR=[{p25v:.2f}, {p75v:.2f}]  p90={p90v:.2f}")
    print()

    # 2. Scatter: Dice vs surface distance per label
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    for ax, (key, name) in zip(axes, LABELS.items()):
        dice = data[f"dice_{key}"]
        dist_vox = data[f"dist_{key}"] / IN_PLANE_SPACING_MM
        ax.scatter(dist_vox, dice, alpha=0.5, s=20)
        ax.axhline(0.80, color="red", linestyle="--", linewidth=1, label="Dice=0.80")
        ax.axvline(2.0, color="gray", linestyle="--", linewidth=1, label="2 voxels")
        ax.set_xlabel("Surface distance (voxels)")
        ax.set_ylabel("Dice")
        ax.set_title(f"{name}: Dice vs surface distance")
        ax.legend(fontsize=8)
    fig.tight_layout()
    out_path = OUT_DIR / "dice_vs_surfdist_scatter.png"
    fig.savefig(out_path, dpi=140)
    plt.close(fig)
    print(f"=== 2. Scatter plot saved to {out_path} ===\n")

    # Quantify: among low-Dice patients (<0.80), what's their surface distance in voxels?
    print("=== 2b. Among Dice < 0.80 patients per label: surface distance (voxels) stats ===")
    for key, name in LABELS.items():
        dice = data[f"dice_{key}"]
        dist_vox = data[f"dist_{key}"] / IN_PLANE_SPACING_MM
        low_dice_mask = dice < 0.80
        n_low = low_dice_mask.sum()
        if n_low > 0:
            low_dist = dist_vox[low_dice_mask]
            print(f"{name}: {n_low}/{n} patients have Dice<0.80. Of those, surface dist (voxels): "
                  f"median={np.median(low_dist):.2f}, mean={np.mean(low_dist):.2f}, "
                  f"frac with dist<=2vox={np.mean(low_dist <= 2.0):.2%}")
    print()

    # 3. Count: patients with mean surface distance <= 2 voxels across all 3 labels, despite Dice < 0.80
    mean_dist_vox = np.mean([data[f"dist_{key}"] / IN_PLANE_SPACING_MM for key in LABELS], axis=0)
    mean_dice = np.array([float(r["mean_dice"]) for r in rows])

    small_dist_mask = mean_dist_vox <= 2.0
    low_dice_mask = mean_dice < 0.80
    both_mask = small_dist_mask & low_dice_mask

    print("=== 3. Metric-artifact count ===")
    print(f"Patients with mean surface distance <= 2 voxels (across RV/myo/LV): {small_dist_mask.sum()}/{n} ({small_dist_mask.mean():.1%})")
    print(f"Patients with mean_dice < 0.80: {low_dice_mask.sum()}/{n} ({low_dice_mask.mean():.1%})")
    print(f"Patients with BOTH (small surf-dist AND failing Dice gate) = likely metric artifact: {both_mask.sum()}/{n} ({both_mask.mean():.1%})")
    print(f"  -> of the {low_dice_mask.sum()} patients failing the Dice gate, {both_mask.sum()} ({both_mask.sum()/low_dice_mask.sum():.1%}) have small (<=2vox) mean surface distance")


if __name__ == "__main__":
    main()
