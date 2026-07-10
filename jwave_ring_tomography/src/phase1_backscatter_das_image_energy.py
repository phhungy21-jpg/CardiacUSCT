"""Fixes run -34's weak/blurry backscatter localization by switching from
COHERENT delay-and-sum (summing raw envelope AMPLITUDE, built for a single
specular reflecting point) to INCOHERENT energy accumulation (summing
squared envelope, i.e. ENERGY/intensity) -- the physically appropriate
accumulator for a spatially-extended, randomly-phased scattering source
like speckle, where different firing angles see genuinely different,
uncorrelated realizations of the same underlying microstructure and so
never coherently "agree" on a single delay the way a real boundary does.
Squaring before summing also disproportionately suppresses many small,
inconsistent contributions relative to genuinely elevated ones, which
should sharpen the wall-vs-background contrast.

Also addresses run -34's central DAS crossing-artifact directly: EXCLUDES
a generic central region (r<20 cells, not informed by the specific true
contour) from all bulk statistics, and compares the WALL annulus against
a "clean" mid-core annulus (r=25-50, deliberately avoiding both the
central artifact zone AND the wall itself) rather than the naive
"everything inside r<57" region that run -34 showed was contaminated.

No new jWave simulation -- reuses `results/speckle_channel_raw_traces.npz`
(run -33), pure post-processing, same data as run -34's amplitude-based
attempt for a direct, apples-to-apples comparison.
"""

import numpy as np
from scipy.signal import correlate, hilbert

from phase1_circular_positive_control import R_OUTER, R_INNER
from phase1_reflection_channel_scout import thetas, pitch_catch_positions
from phase1_matched_filter_echo_extraction import _lag_t_arr, _template
from phase1_rotating_transmission_scout import center, N, dx
import phase2_config as cfg
import labels

from matplotlib import pyplot as plt
import os

IMG_SIZE = 150
CENTRAL_EXCLUDE_R = 20.0  # generic exclusion radius for the known central crossing artifact


def matched_filter_envelope(trace):
    correlated = correlate(trace, _template, mode="full")
    return np.abs(hilbert(correlated))


def das_energy_image(phantom_envelopes, water_envelopes, img_size=IMG_SIZE):
    """Same accumulation as das_straight_ray_image, but sums SQUARED
    (energy) excess envelope instead of raw amplitude -- the physically
    appropriate accumulator for an incoherent, spatially-extended
    scattering source."""
    img_rows = np.linspace(0, N[0], img_size)
    img_cols = np.linspace(0, N[1], img_size)
    RR, CC = np.meshgrid(img_rows, img_cols, indexing="ij")
    accumulator = np.zeros((img_size, img_size))
    for i, theta in enumerate(thetas):
        src, rcv = pitch_catch_positions(theta)
        dist_src = np.hypot(RR - src[0], CC - src[1]) * dx[0]
        dist_rcv = np.hypot(RR - rcv[0], CC - rcv[1]) * dx[0]
        t_pred = (dist_src + dist_rcv) / cfg.WATER.sound_speed
        excess_env = np.clip(phantom_envelopes[i] - water_envelopes[i], 0, None)
        sampled = np.interp(t_pred.ravel(), _lag_t_arr, excess_env, left=0.0, right=0.0)
        accumulator += (sampled.reshape(img_size, img_size)) ** 2
    return accumulator, img_rows, img_cols


if __name__ == "__main__":
    print(f"*** {labels.PENDING_SIGNOFF_BANNER} ***")
    print(f"*** {labels.TOY_EXACT_GT_CAPTION} ***")
    print("BACKSCATTER IMAGING, INCOHERENT/ENERGY UPGRADE: sums squared (energy) excess "
          "envelope instead of raw amplitude -- the physically correct accumulator for an "
          "incoherent, spatially-extended scattering source (speckle), vs. run -34's coherent "
          "delay-and-sum (built for a single specular point). Also excludes the known central "
          f"crossing-artifact region (r<{CENTRAL_EXCLUDE_R:.0f} cells) from bulk statistics. "
          "No new jWave simulation -- reuses cached raw traces from run -33.")

    d = np.load("results/speckle_channel_raw_traces.npz")
    homog_traces, speckle_traces = d["homogeneous_traces"], d["speckle_traces"]
    homog_env = [matched_filter_envelope(tr) for tr in homog_traces]
    speckle_env = [matched_filter_envelope(tr) for tr in speckle_traces]

    print("\n=== Building INCOHERENT (energy) DAS backscatter image ===")
    image, img_rows, img_cols = das_energy_image(speckle_env, homog_env)

    RR, CC = np.meshgrid(img_rows, img_cols, indexing="ij")
    r_grid = np.hypot(RR - center[0], CC - center[1])
    r_bins = np.arange(0, 140, 2.0)
    radial_profile = np.array([image[(r_grid >= r) & (r_grid < r + 2.0)].mean()
                                if ((r_grid >= r) & (r_grid < r + 2.0)).any() else np.nan
                                for r in r_bins])

    clean_core_mask = (r_grid > CENTRAL_EXCLUDE_R) & (r_grid < R_INNER - 5)
    wall_mask = (r_grid > R_INNER + 5) & (r_grid < R_OUTER - 5)
    outside_mask = r_grid > R_OUTER + 5
    mean_clean_core = image[clean_core_mask].mean()
    mean_wall = image[wall_mask].mean()
    mean_outside = image[outside_mask].mean()
    print(f"  mean intensity: CLEAN core ({CENTRAL_EXCLUDE_R:.0f}<r<{R_INNER-5:.0f}, "
          f"central artifact excluded)={mean_clean_core:.4g}, "
          f"myocardium wall ({R_INNER+5:.0f}<r<{R_OUTER-5:.0f})={mean_wall:.4g}, "
          f"outside water (r>{R_OUTER+5:.0f})={mean_outside:.4g}")
    print(f"  wall/clean-core ratio: {mean_wall/mean_clean_core:.2f}x, "
          f"wall/outside ratio: {mean_wall/mean_outside:.2f}x")
    if mean_wall / mean_clean_core > 1.3:
        print("  -> IMPROVEMENT over run -34's coherent DAS (wall/core was 1.00x there): "
              "incoherent energy accumulation localizes the wall better.")
    else:
        print("  -> still no clear localization vs. the clean core region -- incoherent "
              "accumulation alone did not fix it either.")

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
    circle_excl = plt.Circle((center[1], center[0]), CENTRAL_EXCLUDE_R, color="white",
                              linestyle=":", fill=False, linewidth=1.2, label="central exclusion zone")
    axes[0].add_patch(circle_excl)
    axes[0].set_title("INCOHERENT (energy) DAS backscatter image\n(speckle minus homogeneous, squared+accumulated)")
    axes[0].legend(fontsize=7)
    plt.colorbar(im, ax=axes[0], shrink=0.7)

    axes[1].plot(r_bins, radial_profile, "o-")
    axes[1].axvline(CENTRAL_EXCLUDE_R, color="gray", linestyle=":", label=f"central exclusion (r={CENTRAL_EXCLUDE_R:.0f})")
    axes[1].axvline(R_INNER, color="b", linestyle="--", label="true inner (R=60)")
    axes[1].axvline(R_OUTER, color="c", linestyle="--", label="true outer (R=80)")
    axes[1].set_xlabel("radius from center (cells)")
    axes[1].set_ylabel("mean INCOHERENT backscatter intensity")
    axes[1].set_title("Radial profile: incoherent (energy) accumulation")
    axes[1].legend(fontsize=8)

    fig.suptitle("Backscatter imaging, INCOHERENT/energy upgrade: does this localize the myocardium annulus?")
    labels.add_banner(fig)
    plt.tight_layout()
    os.makedirs("results/figures", exist_ok=True)
    out_fig = "results/figures/phase1_backscatter_das_image_energy.png"
    plt.savefig(out_fig, dpi=140)
    print(f"\nSaved {out_fig}")
