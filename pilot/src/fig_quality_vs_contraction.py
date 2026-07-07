"""Figure: registration quality vs. contraction magnitude (Tier 1, FIGURES.md #3).
Explains the DCM-best/HCM-worst pathology-group finding mechanistically —
contraction magnitude, not disease label, drives registration difficulty."""

from pathlib import Path

import csv
import matplotlib.pyplot as plt
import numpy as np

PROC_DIR = Path("data/processed/ACDC")
DICE_CSV = Path("results/phase3_dice.csv")

GROUP_COLORS = {"DCM": "#4c72b0", "HCM": "#dd8452", "MINF": "#55a868", "NOR": "#c44e52", "RV": "#8172b2"}


def contraction_ratio(pid: str) -> float:
    d = np.load(PROC_DIR / f"{pid}.npz")
    ed_lv = (d["ed_mask"] == 3).sum()
    es_lv = (d["es_mask"] == 3).sum()
    if ed_lv == 0:
        return float("nan")
    return 1.0 - (es_lv / ed_lv)  # fraction of LV cavity volume lost ED->ES


def get_group(pid: str) -> str:
    import configparser
    for split in ("training", "testing"):
        p = Path("data/ACDC/ACDC/database") / split / pid / "Info.cfg"
        if p.exists():
            cfg_text = "[info]\n" + p.read_text()
            parser = configparser.ConfigParser()
            parser.read_string(cfg_text)
            return parser["info"].get("group", "?")
    return "?"


def main() -> None:
    rows = list(csv.DictReader(open(DICE_CSV)))
    ratios, dices, groups = [], [], []
    for r in rows:
        pid = r["patient_id"]
        ratio = contraction_ratio(pid)
        ratios.append(ratio)
        dices.append(float(r["mean_dice"]))
        groups.append(get_group(pid))

    fig, ax = plt.subplots(figsize=(7, 6))
    for group, color in GROUP_COLORS.items():
        idx = [i for i, g in enumerate(groups) if g == group]
        ax.scatter([ratios[i] for i in idx], [dices[i] for i in idx], label=group, color=color, alpha=0.7, s=35)

    ax.axhline(0.80, color="gray", linestyle="--", linewidth=1, alpha=0.7)
    ax.set_xlabel("LV contraction magnitude (fraction of ED cavity volume lost by ES)")
    ax.set_ylabel("Registration mean Dice (RV+myo+LV)")
    ax.set_title("Registration quality vs. contraction magnitude, by pathology group")
    ax.legend(title="Group")
    fig.tight_layout()
    out = Path("results/figures/quality_vs_contraction.png")
    fig.savefig(out, dpi=160)
    print(f"Saved {out}")

    # Correlation, for the caption
    r_valid = [(x, y) for x, y in zip(ratios, dices) if not np.isnan(x)]
    xs, ys = zip(*r_valid)
    corr = np.corrcoef(xs, ys)[0, 1]
    print(f"Pearson correlation (contraction ratio vs Dice): {corr:.3f}")


if __name__ == "__main__":
    main()
