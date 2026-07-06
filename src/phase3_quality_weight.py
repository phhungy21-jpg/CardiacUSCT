"""Per-patient quality weight from surface distance (not a pass/fail filter).

weight = 1 / (1 + mean_surface_distance_in_voxels)

Simple, monotonic, bounded in (0, 1], no tuning knob to justify beyond "more
boundary error -> less trust" — appropriate for a low-N pilot. All 150
patients are retained; low-quality patients get low weight, not exclusion.
Also checks for pathology-group skew in the resulting weights, since a
systematic weight bias by group would quietly reintroduce a filtering-like
effect that should be visible and reported.
"""

import configparser
import csv
from pathlib import Path

import numpy as np

CSV_PATH = Path("results/phase3_dice.csv")
OUT_CSV = Path("results/phase3_quality_weights.csv")
IN_PLANE_SPACING_MM = 1.5625


def get_group(pid: str, split: str) -> str:
    cfg_path = Path("data/ACDC/ACDC/database") / split / pid / "Info.cfg"
    cfg_text = "[info]\n" + cfg_path.read_text()
    parser = configparser.ConfigParser()
    parser.read_string(cfg_text)
    return parser["info"].get("group", "?")


def main() -> None:
    rows = list(csv.DictReader(open(CSV_PATH)))
    out_rows = []

    for r in rows:
        dists_vox = [float(r[f"dist_{k}_mm"]) / IN_PLANE_SPACING_MM for k in ("rv", "myo", "lv")]
        mean_dist_vox = float(np.mean(dists_vox))
        weight = 1.0 / (1.0 + mean_dist_vox)
        group = get_group(r["patient_id"], r["split"])
        out_rows.append({
            "patient_id": r["patient_id"], "split": r["split"], "group": group,
            "mean_dist_vox": round(mean_dist_vox, 4), "quality_weight": round(weight, 4),
        })

    OUT_CSV.parent.mkdir(parents=True, exist_ok=True)
    with open(OUT_CSV, "w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(out_rows[0].keys()))
        writer.writeheader()
        writer.writerows(out_rows)

    n = len(out_rows)
    weights = np.array([r["quality_weight"] for r in out_rows])
    dists = np.array([r["mean_dist_vox"] for r in out_rows])

    print(f"n = {n} patients, all retained (no filtering)")
    print(f"Quality weight: mean={weights.mean():.3f} std={weights.std():.3f} min={weights.min():.3f} max={weights.max():.3f}")
    print(f"Patients with mean_dist_vox > 2 (low-weight tail): {(dists > 2).sum()}/{n}")
    for r in out_rows:
        if r["mean_dist_vox"] > 2:
            print(f"  {r['patient_id']} ({r['group']}): mean_dist_vox={r['mean_dist_vox']:.2f}, weight={r['quality_weight']:.3f} — kept, not excluded")

    print("\nMean quality weight by pathology group (check for skew):")
    groups = {}
    for r in out_rows:
        groups.setdefault(r["group"], []).append(r["quality_weight"])
    for g, w in sorted(groups.items()):
        w = np.array(w)
        print(f"  {g}: n={len(w)} mean_weight={w.mean():.3f} std={w.std():.3f}")

    print(f"\nWritten to {OUT_CSV}")


if __name__ == "__main__":
    main()
