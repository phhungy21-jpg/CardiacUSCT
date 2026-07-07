"""Phase 3 — diagnostic: does the outer (myocardium/chest-wall-proxy)
interface have real, independently-recoverable signal at all, or is its
run -39 failure specifically caused by the inner (blood/myocardium)
boundary dominating/masking it?

Run -39 found the ring phantom's inner boundary recovers excellently
(0.27mm RMSE) but the outer boundary does not (2.44mm RMSE, landing on
fixed artifacts or the inner boundary's own fit). Per user's correction:
attributing this to "Thread 1's weak blood/myocardium interface problem"
is WRONG, because that would predict the INNER boundary should fail
too (it's the same interface) -- but it didn't. The real, narrower
question is whether this is a two-boundary SEPARATION/masking problem,
not a blanket weak-signal problem.

Most decisive, cheapest test (per user): build a myocardium DISK in a
chest-wall-proxy background -- NO inner blood cavity, NO competing
boundary at all -- at the same radius as the ring's true outer boundary
(R=90 cells, matching run -39's ED frame), and run the identical
validated naive + global shape-fit pipeline. If this recovers well
(comparable to every other single-boundary shape tried this thread:
circle 0.24mm, triangle 0.23-0.81mm, heart-cartoon 0.23mm), that PROVES
the myocardium/chest-wall-proxy interface has real, detectable signal
on its own -- meaning the ring's outer-boundary failure is specifically
caused by the inner boundary's presence (masking/domination/two-
boundary-search ambiguity), not a fundamentally undetectable interface.
If it does NOT recover well even in isolation, that points to the
interface itself being too weak at this probe/frequency setup,
independent of the two-boundary complication.
"""

import numpy as np

from jax import numpy as jnp
from jwave import FourierSeries
from jwave.geometry import Medium

from phase3_backprojection_shape_fit_triangle import (
    capture_all_pairs, build_medium_homogeneous, backproject,
    img_rows, img_cols, center, dx, N, domain, cfg, p3cfg, labels,
)
from phase3_ring_phantom_shapefit import fit_circle_radius, OUTER_R_GRID

from matplotlib import pyplot as plt
import os


def build_medium_myocardium_disk(R):
    """Myocardium disk in chest-wall-proxy background -- isolates the
    outer interface alone, no inner blood cavity, no competing boundary."""
    yy, xx = np.mgrid[0:N[0], 0:N[1]]
    dist = np.sqrt((xx - center[1]) ** 2 + (yy - center[0]) ** 2)
    inside = dist < R
    sound_speed_map = np.where(inside, cfg.MYOCARDIUM.sound_speed, cfg.CHEST_WALL_PROXY.sound_speed).astype(np.float32)
    density_map = np.where(inside, cfg.MYOCARDIUM.density, cfg.CHEST_WALL_PROXY.density).astype(np.float32)
    ssm = jnp.expand_dims(jnp.array(sound_speed_map), -1)
    dm = jnp.expand_dims(jnp.array(density_map), -1)
    return Medium(domain=domain, sound_speed=FourierSeries(ssm, domain),
                  density=FourierSeries(dm, domain))


if __name__ == "__main__":
    print(f"*** {labels.PENDING_SIGNOFF_BANNER} ***")
    print(f"*** {labels.TOY_EXACT_GT_CAPTION} ***")
    print("Outer-boundary isolation test: myocardium disk in chest-wall-"
          "proxy background, NO inner blood cavity -- tests whether the "
          "myocardium/chest-wall-proxy interface has real recoverable "
          "signal on its own, or whether run -39's outer-boundary "
          "failure is specifically caused by the inner boundary's "
          "presence.")

    R_TEST = 90.0  # matches run -39's ED-frame true outer radius exactly
    print(f"\n=== Capturing: myocardium disk R={R_TEST} cells + homogeneous reference ===")
    pairs_ref = capture_all_pairs(build_medium_homogeneous())
    pairs_disk = capture_all_pairs(build_medium_myocardium_disk(R_TEST))

    accumulator_ref = backproject(pairs_ref)
    accumulator_disk = backproject(pairs_disk)
    accumulator_clean = accumulator_disk - accumulator_ref

    fitted_R, scores = fit_circle_radius(accumulator_clean, OUTER_R_GRID)
    err_mm = abs(fitted_R - R_TEST) * dx[0] * 1e3
    print(f"\nTrue R={R_TEST:.1f} cells, fitted R={fitted_R:.1f} cells, error={err_mm:.2f}mm")

    print("\n--- Verdict ---")
    if err_mm < 1.0:
        print(f"  Recovers well ({err_mm:.2f}mm, comparable to every other single-boundary "
              f"shape this thread: circle 0.24mm, triangle 0.23-0.81mm, heart-cartoon 0.23mm).")
        print("  -> The myocardium/chest-wall-proxy interface DOES have real, detectable "
              "signal on its own. Run -39's outer-boundary failure is caused by the inner "
              "boundary's presence (masking/domination/two-boundary-search ambiguity), "
              "NOT a fundamentally undetectable interface. This is a two-boundary "
              "separation problem, confirmed.")
    else:
        print(f"  Does NOT recover well ({err_mm:.2f}mm) even in isolation.")
        print("  -> The myocardium/chest-wall-proxy interface itself may be too weak to "
              "detect at this probe/frequency setup, independent of the two-boundary "
              "complication. Reconsider tissue contrast/frequency, not just the fitting "
              "algorithm.")

    # Visual: reconstructed image with true and fitted circle overlaid.
    fig, ax = plt.subplots(figsize=(6, 6))
    vmax = np.abs(accumulator_clean).max()
    ax.imshow(np.abs(accumulator_clean), cmap="hot", vmin=0, vmax=vmax, origin="upper",
              extent=[img_cols.min(), img_cols.max(), img_rows.max(), img_rows.min()])
    theta_plot = np.linspace(0, 2 * np.pi, 100)
    ax.plot(center[1] + R_TEST * np.cos(theta_plot), center[0] + R_TEST * np.sin(theta_plot),
            "c--", linewidth=1.3, alpha=0.8, label=f"true R={R_TEST:.0f}")
    ax.plot(center[1] + fitted_R * np.cos(theta_plot), center[0] + fitted_R * np.sin(theta_plot),
            "g:", linewidth=1.8, alpha=0.9, label=f"fitted R={fitted_R:.1f}")
    ax.set_title(f"Myocardium disk alone (no inner boundary)\nerr={err_mm:.2f}mm", fontsize=11)
    ax.legend(fontsize=9)
    ax.axis("off")
    labels.add_banner(fig)
    plt.tight_layout()
    os.makedirs("results/figures", exist_ok=True)
    plt.savefig("results/figures/phase3_outer_boundary_diagnostic.png", dpi=140)
    print("\nSaved results/figures/phase3_outer_boundary_diagnostic.png")
