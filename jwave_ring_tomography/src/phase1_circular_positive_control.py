"""Isolating experiment for run -09's unexplained +7.8 cell inner-
boundary bias: is it a peak-detection/group-delay timing artifact, or
off-radial specular-point geometry from the real anatomy's
irregularity?

Run -09 ruled out the sound-speed-substitution mechanism as the main
cause (predicts only ~0.7 cells, ~10x too small) and run -11 ruled out
attenuation (negligible at this scale). Two candidates remained: (1)
peak-detection/group-delay timing precision -- a property of the
DETECTION METHOD, independent of boundary shape; (2) the true specular
reflection point on an IRREGULAR real boundary not lying exactly on
the assumed straight radial line -- a property of the SHAPE.

This isolates them: build a PERFECTLY CIRCULAR, CENTERED two-tissue
phantom (blood core + myocardium ring, same tissue contrast as
patient001, matched R_outer/R_inner scale). For a centered circle, a
radially-placed monostatic pitch-catch pair's specular reflection point
MUST lie exactly on that radial line, by symmetry -- mechanism (2) is
structurally impossible here. If the bias vanishes for this case,
mechanism (2) explains the real-anatomy bias. If the bias persists at
a similar magnitude, mechanism (1) (a property of the detection method
itself) is the real cause, and would show up regardless of shape.
"""

import numpy as np

from jax import numpy as jnp
from jwave import FourierSeries
from jwave.geometry import Medium

from phase1_reflection_channel_scout import simulate_pitch_catch, thetas, predicted_reflection_times, peak_in_window
from phase1_fused_channel_reconstruction import blind_two_peak_distances
from phase1_rotating_transmission_scout import center, N, domain
import phase2_config as cfg
import labels

from matplotlib import pyplot as plt
import os

R_OUTER = 80.0  # matches patient001's real outer radius scale (run -07: max_radius=79.5)
R_INNER = 60.0  # matches patient001's real inner radius scale (run -07's inner boundary ~60 cells)


def build_medium_concentric_circles():
    yy, xx = np.mgrid[0:N[0], 0:N[1]]
    dist = np.sqrt((xx - center[1]) ** 2 + (yy - center[0]) ** 2)
    inside_myo = dist < R_OUTER
    inside_lv = dist < R_INNER
    sound_speed_map = np.where(inside_myo, cfg.MYOCARDIUM.sound_speed, cfg.WATER.sound_speed).astype(np.float32)
    density_map = np.where(inside_myo, cfg.MYOCARDIUM.density, cfg.WATER.density).astype(np.float32)
    sound_speed_map = np.where(inside_lv, cfg.BLOOD.sound_speed, sound_speed_map).astype(np.float32)
    density_map = np.where(inside_lv, cfg.BLOOD.density, density_map).astype(np.float32)
    ssm = jnp.expand_dims(jnp.array(sound_speed_map), -1)
    dm = jnp.expand_dims(jnp.array(density_map), -1)
    return Medium(domain=domain, sound_speed=FourierSeries(ssm, domain),
                  density=FourierSeries(dm, domain))


def build_medium_water_only():
    sound_speed_map = np.full(N, cfg.WATER.sound_speed, dtype=np.float32)
    density_map = np.full(N, cfg.WATER.density, dtype=np.float32)
    ssm = jnp.expand_dims(jnp.array(sound_speed_map), -1)
    dm = jnp.expand_dims(jnp.array(density_map), -1)
    return Medium(domain=domain, sound_speed=FourierSeries(ssm, domain),
                  density=FourierSeries(dm, domain))


if __name__ == "__main__":
    print(f"*** {labels.PENDING_SIGNOFF_BANNER} ***")
    print(f"*** {labels.TOY_EXACT_GT_CAPTION} ***")
    print("POSITIVE CONTROL: perfectly circular, centered two-tissue phantom "
          f"(R_outer={R_OUTER}, R_inner={R_INNER}, matching patient001's real scale). "
          "Isolates whether run -09's +7.8 cell inner-boundary bias is a detection-"
          "method artifact (would persist here) or off-radial specular geometry from "
          "real-anatomy irregularity (structurally impossible here, by symmetry).")
    print("  compute estimate: 36 angles x 2 media = 72 forward sims (same as prior runs) "
          "-- ~15-20 minutes based on that precedent")

    medium_water = build_medium_water_only()
    medium_phantom = build_medium_concentric_circles()

    print("\n=== Simulating water-only control, pitch-catch at 36 angles ===")
    water_envelopes = [simulate_pitch_catch(medium_water, th) for th in thetas]
    print("=== Simulating concentric-circle phantom, pitch-catch at 36 angles ===")
    phantom_envelopes = [simulate_pitch_catch(medium_phantom, th) for th in thetas]

    print("\n=== Blind per-angle reflection peak detection ===")
    refl_r_outer, refl_r_inner = [], []
    for i in range(len(thetas)):
        water_max = water_envelopes[i].max()
        r_out, r_in = blind_two_peak_distances(phantom_envelopes[i], water_max)
        refl_r_outer.append(r_out)
        refl_r_inner.append(r_in)

    n_outer_found = sum(r is not None for r in refl_r_outer)
    n_inner_found = sum(r is not None for r in refl_r_inner)
    outer_err = np.array([r - R_OUTER for r in refl_r_outer if r is not None])
    inner_err = np.array([r - R_INNER for r in refl_r_inner if r is not None])

    print(f"\n--- Result: circular/centered positive control ---")
    print(f"  outer boundary found {n_outer_found}/{len(thetas)}, mean error={outer_err.mean():+.2f} cells "
          f"(std={outer_err.std():.2f})")
    print(f"  inner boundary found {n_inner_found}/{len(thetas)}, mean error={inner_err.mean():+.2f} cells "
          f"(std={inner_err.std():.2f})")
    print(f"\n  COMPARE: real-anatomy patient001 (run -09) inner bias was +7.8 cells")
    if abs(inner_err.mean()) < 2.0:
        print("  -> bias VANISHES for the circular case: consistent with off-radial specular-point "
              "geometry (real anatomy's irregularity) being the dominant real-anatomy cause.")
    else:
        print("  -> bias PERSISTS at similar magnitude for the circular case: consistent with a "
              "detection-method/group-delay artifact independent of boundary shape.")

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.axhline(0, color="gray", linewidth=0.8)
    ax.plot(thetas, [r - R_OUTER if r is not None else np.nan for r in refl_r_outer], "o-", label="outer boundary error")
    ax.plot(thetas, [r - R_INNER if r is not None else np.nan for r in refl_r_inner], "s-", label="inner boundary error")
    ax.axhline(7.8, color="lime", linestyle=":", alpha=0.6, label="real-anatomy inner bias (run -09): +7.8 cells")
    ax.set_xlabel("angle (deg)")
    ax.set_ylabel("radius error (cells)")
    ax.set_title("Positive control: circular/centered phantom -- reflection radius error vs. angle")
    ax.legend(fontsize=8)

    fig.suptitle("Isolating run -09's inner-boundary bias: detection artifact vs. shape-irregularity geometry")
    labels.add_banner(fig)
    plt.tight_layout()
    os.makedirs("results/figures", exist_ok=True)
    out_fig = "results/figures/phase1_circular_positive_control.png"
    plt.savefig(out_fig, dpi=140)
    print(f"\nSaved {out_fig}")
