"""Visualizes WHERE the 3D boundary extraction (run -12/-13) is accurate
vs. inaccurate, direction by direction -- the natural follow-up to the
slice figure, which showed the DAS energy tracing the true contour
clearly on only one side. Plots the 1000 Fibonacci-sphere directions in
azimuth/elevation with per-direction signed error, plus a predicted-vs-
true radius scatter, to confirm/quantify that visual impression rather
than leaving it as an eyeballed guess.
"""

import numpy as np
import matplotlib.pyplot as plt

BOUNDARY_NPZ = r"D:\Cardiac USCT\jwave_cylindrical_tomography\results\das_3d_boundary_extraction_patient023.npz"
OUT_PNG = r"D:\Cardiac USCT\jwave_cylindrical_tomography\results\figures\phase1_das_3d_boundary_error_map_patient023.png"

if __name__ == "__main__":
    d = np.load(BOUNDARY_NPZ)
    directions = d["directions"]
    r_pred = d["r_pred_mm"]
    r_true = d["r_true_mm"]
    valid = ~np.isnan(r_true)

    az = np.arctan2(directions[:, 1], directions[:, 0])   # azimuth
    el = np.arcsin(np.clip(directions[:, 2], -1, 1))       # elevation
    error = r_pred - r_true

    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    sc = axes[0].scatter(np.degrees(az[valid]), np.degrees(el[valid]), c=error[valid],
                          cmap="coolwarm", vmin=-15, vmax=15, s=14)
    axes[0].set_xlabel("azimuth (deg)")
    axes[0].set_ylabel("elevation (deg)")
    axes[0].set_title("Signed boundary error by direction\n(red=predicted too far out, blue=too far in)")
    fig.colorbar(sc, ax=axes[0], label="r_pred - r_true (mm)")

    axes[1].scatter(r_true[valid], r_pred[valid], s=8, alpha=0.4)
    lims = [min(r_true[valid].min(), r_pred[valid].min()), max(r_true[valid].max(), r_pred[valid].max())]
    axes[1].plot(lims, lims, "k--", linewidth=1, label="perfect")
    axes[1].set_xlabel("true boundary radius (mm)")
    axes[1].set_ylabel("predicted boundary radius (mm)")
    axes[1].set_title(f"Predicted vs. true (RMSE={d['rmse_mm']:.2f}mm)")
    axes[1].legend()
    axes[1].set_aspect("equal")

    fig.suptitle("patient023 3D boundary extraction: where it works vs. doesn't (run -12/-13)")
    fig.tight_layout()
    fig.savefig(OUT_PNG, dpi=150)
    print(f"Saved {OUT_PNG}")

    print(f"\nfraction of directions with |error| < 2mm: {(np.abs(error[valid]) < 2).mean():.1%}")
    print(f"fraction of directions with |error| < 5mm: {(np.abs(error[valid]) < 5).mean():.1%}")
    print(f"fraction of directions with |error| > 10mm: {(np.abs(error[valid]) > 10).mean():.1%}")
