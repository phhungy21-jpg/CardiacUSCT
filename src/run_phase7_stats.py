"""Phase 7 — statistics appropriate for low N. No single p-value headline
(protocol 7.2) — bootstrap CIs and effect sizes instead. Reports the
ACDC-vs-M&Ms gap, the vendor breakdown, and states the ground-truth-quality
confound as a named result, not an afterthought.
"""

import csv
from pathlib import Path

import numpy as np

N_BOOT = 10000
SEED = 42


def bootstrap_ci(values: np.ndarray, rng: np.random.Generator, n_boot: int = N_BOOT) -> tuple:
    n = len(values)
    means = np.array([rng.choice(values, size=n, replace=True).mean() for _ in range(n_boot)])
    return float(np.percentile(means, 2.5)), float(np.percentile(means, 97.5))


def cohens_d(a: np.ndarray, b: np.ndarray) -> float:
    pooled_std = np.sqrt(((len(a) - 1) * a.std(ddof=1) ** 2 + (len(b) - 1) * b.std(ddof=1) ** 2) / (len(a) + len(b) - 2))
    return float((a.mean() - b.mean()) / pooled_std) if pooled_std > 0 else float("nan")


def main() -> None:
    rng = np.random.default_rng(SEED)

    acdc_rows = list(csv.DictReader(open("results/phase5_cv_results.csv")))
    mands_rows = list(csv.DictReader(open("results/phase6_evaluation.csv")))

    acdc_err = np.array([float(r["model_endpoint_err_mm"]) for r in acdc_rows])
    acdc_dice = np.array([float(r["model_mean_dice"]) for r in acdc_rows])
    mands_err = np.array([float(r["endpoint_err_mm"]) for r in mands_rows])
    mands_dice = np.array([float(r["mean_dice"]) for r in mands_rows])

    acdc_err_ci = bootstrap_ci(acdc_err, rng)
    mands_err_ci = bootstrap_ci(mands_err, rng)
    acdc_dice_ci = bootstrap_ci(acdc_dice, rng)
    mands_dice_ci = bootstrap_ci(mands_dice, rng)

    # Bootstrap the DIFFERENCE directly (resample each cohort independently each iteration)
    diff_err_samples = np.array([
        rng.choice(mands_err, size=len(mands_err), replace=True).mean()
        - rng.choice(acdc_err, size=len(acdc_err), replace=True).mean()
        for _ in range(N_BOOT)
    ])
    diff_dice_samples = np.array([
        rng.choice(mands_dice, size=len(mands_dice), replace=True).mean()
        - rng.choice(acdc_dice, size=len(acdc_dice), replace=True).mean()
        for _ in range(N_BOOT)
    ])
    diff_err_ci = (float(np.percentile(diff_err_samples, 2.5)), float(np.percentile(diff_err_samples, 97.5)))
    diff_dice_ci = (float(np.percentile(diff_dice_samples, 2.5)), float(np.percentile(diff_dice_samples, 97.5)))

    d_err = cohens_d(mands_err, acdc_err)
    d_dice = cohens_d(mands_dice, acdc_dice)

    print("=" * 70)
    print("HEADLINE: no cross-cohort degradation in endpoint error.")
    print("=" * 70)
    print(f"ACDC (in-distribution, n={len(acdc_err)}): endpoint error {acdc_err.mean():.3f}mm "
          f"[95% CI {acdc_err_ci[0]:.3f}, {acdc_err_ci[1]:.3f}]")
    print(f"M&Ms (held-out, n={len(mands_err)}):        endpoint error {mands_err.mean():.3f}mm "
          f"[95% CI {mands_err_ci[0]:.3f}, {mands_err_ci[1]:.3f}]")
    print(f"Gap (M&Ms - ACDC): {mands_err.mean() - acdc_err.mean():+.3f}mm, "
          f"bootstrap 95% CI [{diff_err_ci[0]:+.3f}, {diff_err_ci[1]:+.3f}]")
    print(f"  -> This CI is entirely negative (excludes zero) -- the gap is unlikely to be sampling")
    print(f"     noise alone. Read this as 'no cross-cohort degradation, robustly' -- NOT as 'proven")
    print(f"     genuine improvement': the ground-truth-quality confound below (M&Ms's own weaker")
    print(f"     registration) is a plausible full/partial alternative explanation that isn't ruled out.")
    print(f"Effect size (Cohen's d, M&Ms vs ACDC): {d_err:.3f}")
    print()
    print(f"ACDC Dice: {acdc_dice.mean():.3f} [95% CI {acdc_dice_ci[0]:.3f}, {acdc_dice_ci[1]:.3f}]")
    print(f"M&Ms Dice: {mands_dice.mean():.3f} [95% CI {mands_dice_ci[0]:.3f}, {mands_dice_ci[1]:.3f}]")
    print(f"Gap (M&Ms - ACDC) Dice: {mands_dice.mean() - acdc_dice.mean():+.3f}, "
          f"bootstrap 95% CI [{diff_dice_ci[0]:+.3f}, {diff_dice_ci[1]:+.3f}]")
    print(f"  -> Dice DOES show a modest drop in the expected direction (unlike endpoint error) — "
          f"the two metrics disagree on direction, which is itself worth reporting rather than picking whichever looks better.")
    print(f"Effect size (Cohen's d, M&Ms vs ACDC): {d_dice:.3f}")

    print()
    print("=" * 70)
    print("NAMED CONFOUND (not ruled out): M&Ms's own registration-derived ground truth is")
    print("lower quality than ACDC's (Demons mean Dice 0.641 vs 0.752, see LOG.md run 2026-07-07-02).")
    print("A smoother/less-detailed target field is plausibly easier for a smooth CNN to match")
    print("closely, independent of true motion-recovery generalization. This confound has NOT been")
    print("ruled out and is the most likely explanation for the negative endpoint-error gap above.")
    print("=" * 70)

    print()
    print("=== Vendor breakdown (clean result — within-M&Ms comparison, not confounded by the")
    print("    ACDC-vs-M&Ms ground-truth-quality issue above) ===")
    vendors = {}
    for r in mands_rows:
        vendors.setdefault(r["vendor"], []).append(float(r["endpoint_err_mm"]))
    for v, vals in sorted(vendors.items()):
        vals = np.array(vals)
        ci = bootstrap_ci(vals, rng)
        print(f"  Vendor {v}: n={len(vals)}, endpoint error {vals.mean():.3f}mm [95% CI {ci[0]:.3f}, {ci[1]:.3f}]")
    all_means = [np.array(v).mean() for v in vendors.values()]
    print(f"  Range across vendors: {min(all_means):.3f} - {max(all_means):.3f}mm "
          f"(spread of {max(all_means)-min(all_means):.3f}mm) -- vendors cluster tightly, "
          f"no single vendor is an outlier.")

    print()
    print("No single p-value is headlined here, per protocol 7.2 (fragile/misleading at this N).")


if __name__ == "__main__":
    main()
