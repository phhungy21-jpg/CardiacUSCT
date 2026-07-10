"""Turns run -33's raw energy-elevation finding (backscatter/speckle is
a real, 3.93x-detectable channel) into an actual IMAGE -- the same bar
this project holds every other channel to (a real, extractable spatial
map, not just a known-time-window energy number at a known angle).

Reuses the DAS (delay-and-sum) machinery from run -31 UNCHANGED: for
every candidate pixel and every one of the 36 pitch-catch firings,
predict the bistatic (src->pixel->rcv) delay and sample that shot's
matched-filter envelope there, accumulating across all firings. Here,
the "phantom" slot is the SPECKLE trace and the "background" slot is
the HOMOGENEOUS trace (both from run -33's `build_medium_
concentric_circles`/`build_medium_speckle_myocardium`, SAME R_outer=80/
R_inner=60 geometry) -- so das_straight_ray_image's existing background-
subtraction (phantom minus water-only, clipped at 0) isolates SPECIFICALLY
the extra scattering contribution speckle adds on top of the two shared
boundary reflections (which are identical in both phantoms and cancel
out), rather than the boundaries themselves.

PREDICTION stated before running: if backscatter is spatially localizable
(not just detectable in aggregate, run -33), the resulting image should
show ELEVATED intensity specifically filling the myocardium ANNULUS
(between R_inner=60 and R_outer=80), close to zero in the blood core
(R<60, homogeneous, nothing to scatter) and in the water background
(R>80).

No new jWave simulation -- reuses `results/speckle_channel_raw_traces.npz`
(run -33), pure post-processing.
"""

import numpy as np
from scipy.signal import correlate, hilbert

from phase1_circular_positive_control import R_OUTER, R_INNER
from phase1_das_reflectivity_imaging import das_straight_ray_image, extract_boundary_from_image
from phase1_matched_filter_echo_extraction import _template
from phase1_reflection_channel_scout import thetas, direction_vector
from phase1_rotating_transmission_scout import center
import phase2_config as cfg
import labels

from matplotlib import pyplot as plt
import os


def matched_filter_envelope(trace):
    correlated = correlate(trace, _template, mode="full")
    return np.abs(hilbert(correlated))


if __name__ == "__main__":
    print(f"*** {labels.PENDING_SIGNOFF_BANNER} ***")
    print(f"*** {labels.TOY_EXACT_GT_CAPTION} ***")
    print("BACKSCATTER/SPECKLE IMAGING: turns run -33's energy-elevation finding into an "
          "actual DAS image. Does the myocardium annulus (R_outer=80, R_inner=60) show up as "
          "a real, spatially-localized bright region, distinct from the homogeneous blood "
          "core and water background? No new jWave simulation -- reuses cached raw traces.")

    d = np.load("results/speckle_channel_raw_traces.npz")
    homog_traces, speckle_traces = d["homogeneous_traces"], d["speckle_traces"]
    homog_env = [matched_filter_envelope(tr) for tr in homog_traces]
    speckle_env = [matched_filter_envelope(tr) for tr in speckle_traces]

    print("\n=== Building DAS backscatter image (speckle minus homogeneous, 36-angle accumulation) ===")
    image, img_rows, img_cols = das_straight_ray_image(speckle_env, homog_env, img_size=150)

    # radial profile: mean intensity vs. radius from center, to check localization quantitatively
    RR, CC = np.meshgrid(img_rows, img_cols, indexing="ij")
    r_grid = np.hypot(RR - center[0], CC - center[1])
    r_bins = np.arange(0, 140, 2.0)
    radial_profile = np.array([image[(r_grid >= r) & (r_grid < r + 2.0)].mean()
                                if ((r_grid >= r) & (r_grid < r + 2.0)).any() else np.nan
                                for r in r_bins])

    mean_in_core = image[r_grid < R_INNER - 3].mean()
    mean_in_wall = image[(r_grid > R_INNER + 3) & (r_grid < R_OUTER - 3)].mean()
    mean_outside = image[r_grid > R_OUTER + 3].mean()
    print(f"  mean intensity: blood core (r<{R_INNER-3:.0f})={mean_in_core:.4g}, "
          f"myocardium wall ({R_INNER+3:.0f}<r<{R_OUTER-3:.0f})={mean_in_wall:.4g}, "
          f"outside water (r>{R_OUTER+3:.0f})={mean_outside:.4g}")
    print(f"  wall/core ratio: {mean_in_wall/mean_in_core:.2f}x, wall/outside ratio: {mean_in_wall/mean_outside:.2f}x")

    theta_plot = np.linspace(0, 2 * np.pi, 200)
    outer_row = center[0] + R_OUTER * np.cos(theta_plot)
    outer_col = center[1] + R_OUTER * np.sin(theta_plot)
    inner_row = center[0] + R_INNER * np.cos(theta_plot)
    inner_col = center[1] + R_INNER * np.sin(theta_plot)

    fig, axes = plt.subplots(1, 2, figsize=(13, 5.5))
    im = axes[0].imshow(image, cmap="hot", origin="upper",
                         extent=[img_cols.min(), img_cols.max(), img_rows.max(), img_rows.min()])
    axes[0].plot(outer_col, outer_row, "c--", linewidth=1.2, label="true outer (R=80)")
    axes[0].plot(inner_col, inner_row, "b--", linewidth=1.2, label="true inner (R=60)")
    axes[0].set_title("DAS backscatter image\n(speckle minus homogeneous, accumulated)")
    axes[0].legend(fontsize=7)
    plt.colorbar(im, ax=axes[0], shrink=0.7)

    axes[1].plot(r_bins, radial_profile, "o-")
    axes[1].axvline(R_INNER, color="b", linestyle="--", label="true inner (R=60)")
    axes[1].axvline(R_OUTER, color="c", linestyle="--", label="true outer (R=80)")
    axes[1].set_xlabel("radius from center (cells)")
    axes[1].set_ylabel("mean DAS backscatter intensity")
    axes[1].set_title("Radial profile: is the myocardium annulus localized?")
    axes[1].legend(fontsize=8)

    fig.suptitle("Backscatter/speckle DAS imaging: is myo-vs-blood distinction spatially extractable?")
    labels.add_banner(fig)
    plt.tight_layout()
    os.makedirs("results/figures", exist_ok=True)
    out_fig = "results/figures/phase1_backscatter_das_image.png"
    plt.savefig(out_fig, dpi=140)
    print(f"\nSaved {out_fig}")
