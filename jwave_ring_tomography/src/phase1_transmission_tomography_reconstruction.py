"""Phase 1 scout, extended: does the circular sequential scan carry
enough spatial information to reconstruct an image at all?

Per user: "why not visualise the blind reconstruction from the
circular scan and overlap it with the cross section so i can see if
circular sequential scan works?"

Before building this, checked run -03's actual geometry: every
diametrically-opposite tx/rx pair's straight-line path passes through
the EXACT SAME center point (150,150), for every single angle
(confirmed numerically). That means run -03's design gives only ONE
degree of freedom per transmit angle (a single line integral through
the center) -- mathematically insufficient to reconstruct a 2D image,
regardless of how the phantom is shaped. That's why its phantom curve
was flat: a necessary consequence of the geometry, not evidence the
scan "works."

FIX (no extra simulation cost): jWave computes the full pressure field
for each transmit event; instead of sampling only the single opposite
receiver, sample EVERY other probe angle's position from that same
field. This gives ~N_ANGLES*(N_ANGLES-1) ray paths at many different
offsets from center -- an actual multistatic transmission-tomography
dataset, suitable for a straight-ray backprojection reconstruction.
Still only ONE transmitter fires at a time (sequential, per run -02's
decision) -- multiple receivers per transmit is standard practice
(matches jwave_test's own `capture_all_pairs` pattern), not a
reversal of that decision.

Reconstruction method: simple UNFILTERED straight-ray backprojection of
each pair's EXCESS DELAY (phantom arrival - water-only arrival, same
ray, isolating the tissue's effect) smeared uniformly along that ray's
path in the image -- the classic first-pass tomography visualization
(same spirit as this whole project's very first naive-backprojection
sanity check, before any refinement). A real inversion (ART/SIRT,
filtered backprojection) is future work; this is a first "does
spatial information exist at all" visual check.
"""

import numpy as np
from scipy.signal import hilbert

from jax import jit
from jwave.acoustics import simulate_wave_propagation

from phase1_rotating_transmission_scout import (
    direction_vector, probe_position, build_medium_water_only, build_medium_static_phantom,
    dx, center, N, domain, time_axis, dt, t_arr, _signal_template,
    PROBE_RADIUS_CELLS, PHANTOM_RADIUS_CELLS, N_ANGLES,
)
import phase2_config as cfg
import labels

from matplotlib import pyplot as plt
import os

thetas = np.linspace(0, 360, N_ANGLES, endpoint=False)
_MIN_ANGLE_SEP_DEG = 20.0  # exclude near-self pairs (near-field/direct-coupling artifact, not a clean through-tissue transmission)


def angular_sep(a, b):
    d = abs(a - b) % 360
    return min(d, 360 - d)


def simulate_transmit_all_receivers(medium, theta_tx_deg):
    tx = probe_position(theta_tx_deg)
    from jwave.geometry import Sources
    sources = Sources(positions=([tx[0]], [tx[1]]), signals=_signal_template, dt=dt, domain=domain)

    @jit
    def run(m):
        return simulate_wave_propagation(m, time_axis, sources=sources)

    pressure = run(medium)
    field = pressure.on_grid[..., 0]  # (n_steps, N0, N1)
    arrivals = {}
    for theta_rx in thetas:
        if angular_sep(theta_tx_deg, theta_rx) < _MIN_ANGLE_SEP_DEG:
            continue
        rx = probe_position(theta_rx)
        trace = np.array(field[:, rx[0], rx[1]])
        envelope = np.abs(hilbert(trace))
        arrival_idx = int(np.argmax(envelope))
        arrivals[theta_rx] = t_arr[arrival_idx]
    return arrivals


def backproject_excess_delay(pairs_excess_delay, img_size=150):
    """Unfiltered straight-ray backprojection: smear each pair's excess
    delay uniformly along its ray path (perpendicular distance <
    RAY_HALF_WIDTH_CELLS from the tx-rx line, 0<=t<=1 along the
    segment)."""
    RAY_HALF_WIDTH_CELLS = 3.0
    img_rows = np.linspace(0, N[0], img_size)
    img_cols = np.linspace(0, N[1], img_size)
    RR, CC = np.meshgrid(img_rows, img_cols, indexing="ij")
    accumulator = np.zeros((img_size, img_size))
    weight = np.zeros((img_size, img_size))
    for (theta_tx, theta_rx), delay_ns in pairs_excess_delay.items():
        p1 = probe_position(theta_tx)
        p2 = probe_position(theta_rx)
        d_row, d_col = p2[0] - p1[0], p2[1] - p1[1]
        length = np.hypot(d_row, d_col)
        w_row, w_col = RR - p1[0], CC - p1[1]
        t = (w_row * d_row + w_col * d_col) / (length ** 2)
        perp_dist = np.abs(w_row * d_col - w_col * d_row) / length
        mask = (t >= 0) & (t <= 1) & (perp_dist < RAY_HALF_WIDTH_CELLS)
        accumulator[mask] += delay_ns
        weight[mask] += 1.0
    with np.errstate(invalid="ignore", divide="ignore"):
        image = np.where(weight > 0, accumulator / weight, 0.0)
    return image, img_rows, img_cols


if __name__ == "__main__":
    print(f"*** {labels.PENDING_SIGNOFF_BANNER} ***")
    n_pairs_expected = sum(1 for tt in thetas for tr in thetas if angular_sep(tt, tr) >= _MIN_ANGLE_SEP_DEG)
    print("PHASE 1 SCOUT (extended): full multistatic transmission tomography + straight-ray backprojection.")
    print(f"  {N_ANGLES} transmit angles, up to {N_ANGLES-1} receivers each "
          f"(excluding pairs within {_MIN_ANGLE_SEP_DEG} deg) -- ~{n_pairs_expected} total ray paths")
    print(f"  compute estimate: {N_ANGLES} simulations x 2 media = {2*N_ANGLES} forward sims "
          f"(SAME as run -03 -- extra receivers are free, sampled from the same computed field) "
          f"-- ~15-20 minutes based on run -03's precedent")

    medium_water = build_medium_water_only()
    medium_phantom = build_medium_static_phantom()

    print("\n=== Simulating water-only control, all transmit angles, all receivers ===")
    water_arrivals = {}
    for theta_tx in thetas:
        water_arrivals[theta_tx] = simulate_transmit_all_receivers(medium_water, theta_tx)

    print("=== Simulating static phantom, all transmit angles, all receivers ===")
    phantom_arrivals = {}
    for theta_tx in thetas:
        phantom_arrivals[theta_tx] = simulate_transmit_all_receivers(medium_phantom, theta_tx)

    pairs_excess_delay_ns = {}
    for theta_tx in thetas:
        for theta_rx, t_water in water_arrivals[theta_tx].items():
            if theta_rx not in phantom_arrivals[theta_tx]:
                continue
            t_phantom = phantom_arrivals[theta_tx][theta_rx]
            pairs_excess_delay_ns[(theta_tx, theta_rx)] = (t_phantom - t_water) * 1e9

    n_pairs = len(pairs_excess_delay_ns)
    delays = np.array(list(pairs_excess_delay_ns.values()))
    print(f"\n--- {n_pairs} ray paths captured ---")
    print(f"  excess delay range: {delays.min():.1f} to {delays.max():.1f} ns, mean={delays.mean():.1f}ns")
    print(f"  (rays that miss the phantom entirely should be near 0ns; rays crossing it should be negative "
          f"-- myocardium is faster than water -- with magnitude depending on how much of the chord crosses tissue)")

    print("\n=== Backprojecting excess delay (unfiltered straight-ray) ===")
    image, img_rows, img_cols = backproject_excess_delay(pairs_excess_delay_ns)

    fig, axes = plt.subplots(1, 2, figsize=(13, 6))
    axes[0].hist(delays, bins=40)
    axes[0].set_xlabel("excess delay (ns)")
    axes[0].set_ylabel("count")
    axes[0].set_title(f"Per-ray excess delay distribution ({n_pairs} rays)")

    im = axes[1].imshow(image, cmap="hot_r", origin="upper",
                         extent=[img_cols.min(), img_cols.max(), img_rows.max(), img_rows.min()])
    theta_plot = np.linspace(0, 2 * np.pi, 200)
    true_row = center[0] + PHANTOM_RADIUS_CELLS * np.cos(theta_plot)
    true_col = center[1] + PHANTOM_RADIUS_CELLS * np.sin(theta_plot)
    axes[1].plot(true_col, true_row, "c--", linewidth=1.5, label="true phantom boundary")
    axes[1].set_title("Unfiltered straight-ray backprojection\n(more negative = faster/tissue-like)")
    axes[1].legend(fontsize=8)
    plt.colorbar(im, ax=axes[1], label="mean excess delay (ns)", shrink=0.8)

    fig.suptitle("Phase 1 scout: does the circular sequential scan carry spatial information?\n"
                 "(STATIC centered phantom, full multistatic transmission tomography)")
    labels.add_banner(fig)
    plt.tight_layout()
    os.makedirs("results/figures", exist_ok=True)
    out_fig = "results/figures/phase1_transmission_tomography_reconstruction.png"
    plt.savefig(out_fig, dpi=140)
    print(f"\nSaved {out_fig}")
