"""First 3D blind reconstruction attempt, per the Phase 1 plan: DAS
(delay-and-sum) reflectivity imaging, the most directly portable method
from `jwave_ring_tomography` (run -31's coherent version for the outer
boundary; run -35's INCOHERENT/energy version, which worked better for
real/speckle-like anatomy -- used here as the default given patient023
is real anatomy, not a clean synthetic boundary).

3D extension of the established 2D approach: for every candidate VOXEL
and every (tx, rx) pair (excluding near/same-position pairs, direct-
coupling artifacts -- same convention as 2D's near-angle exclusion),
predict the straight-ray bistatic delay (tx->voxel->rx, homogeneous
water speed) and sample that pair's matched-filter, background-
subtracted envelope at that delay; accumulate (squared, incoherent)
across all valid pairs.

Uses a COARSER reconstruction grid (IMG_SIZE=60, ~2.17mm/voxel) than
the N=150 simulation grid, for compute tractability on CPU -- 96x95
pairs x 60^3 voxels is already a substantial vectorized computation;
looping over pairs (not voxels) and vectorizing per-pair over all
voxels at once, matching the established 2D DAS code pattern exactly.

No new jWave simulation -- pure post-processing on the already-
completed `cylindrical_multistatic_scan_patient023.npz`.
"""

import numpy as np
from scipy.signal import correlate, hilbert

SCAN_NPZ = r"D:\Cardiac USCT\jwave_cylindrical_tomography\cylindrical_scan_output-20260710T100320Z-2-001\cylindrical_scan_output\cylindrical_multistatic_scan_patient023.npz"

IMG_SIZE = 150
N_GRID = 150
MIN_ELEMENT_SEP_CELLS = 15.0  # exclude near-position pairs (direct-coupling artifact, same
                               # convention as jwave_ring_tomography's near-angle exclusion)


def build_template(f0_hz, dt, n_cycles=3):
    """The known transmitted toneburst, for matched filtering -- must match
    the simulation script's own signal construction exactly."""
    duration_s = n_cycles / f0_hz * 3  # generous window covering the gaussian envelope
    n = int(duration_s / dt)
    t = np.arange(n) * dt
    center_t = n_cycles / f0_hz
    signal = np.sin(2 * np.pi * f0_hz * t) * np.exp(-((t - center_t) ** 2) / (2 * (n_cycles / 3 / f0_hz) ** 2))
    return signal.astype(np.float32)


def matched_filter_envelope(trace, template):
    correlated = correlate(trace, template, mode="full")
    return np.abs(hilbert(correlated))


if __name__ == "__main__":
    print("DAS 3D RECONSTRUCTION: incoherent (energy) delay-and-sum imaging of patient023's "
          "static ED volume, from the real completed 96-element cylindrical multistatic scan.")

    d = np.load(SCAN_NPZ)
    water_traces = d["water_traces"]      # (96 tx, n_steps, 96 rx)
    phantom_traces = d["phantom_traces"]
    positions = d["positions"]            # (96, 3) grid-cell coords
    dt = float(d["dt"])
    n_steps = int(d["n_steps"])
    n_grid = int(d["n_grid"])
    dx_m = float(d["dx_m"])
    f0_hz = float(d["f0_hz"])
    n_elements = positions.shape[0]
    print(f"  loaded scan: {n_elements} elements, n_steps={n_steps}, dx={dx_m*1e3:.3f}mm, f0={f0_hz/1e6}MHz")

    WATER_SOUND_SPEED = 1520.0
    template = build_template(f0_hz, dt)
    lag_t = (np.arange(len(template) + n_steps - 1) - (len(template) - 1)) * dt
    print(f"  matched-filter template length: {len(template)} samples")

    # precompute matched-filter, background-subtracted, SQUARED (incoherent) envelopes
    # for every valid (tx, rx) pair
    print("\n=== Matched-filtering all valid (tx, rx) pairs ===")
    pairs = []
    excess_energy = {}
    for tx in range(n_elements):
        for rx in range(n_elements):
            if tx == rx:
                continue
            sep = np.linalg.norm(positions[tx] - positions[rx])
            if sep < MIN_ELEMENT_SEP_CELLS:
                continue
            env_p = matched_filter_envelope(phantom_traces[tx, :, rx], template)
            env_w = matched_filter_envelope(water_traces[tx, :, rx], template)
            excess = np.clip(env_p - env_w, 0, None)
            excess_energy[(tx, rx)] = (excess ** 2).astype(np.float32)
            pairs.append((tx, rx))
    print(f"  {len(pairs)} valid pairs (excluded near-position pairs, sep<{MIN_ELEMENT_SEP_CELLS} cells)")

    # image grid (coarser than the simulation grid, for compute tractability)
    img_coords = np.linspace(0, n_grid, IMG_SIZE)
    RR, CC, ZZ = np.meshgrid(img_coords, img_coords, img_coords, indexing="ij")
    accumulator = np.zeros((IMG_SIZE, IMG_SIZE, IMG_SIZE), dtype=np.float32)

    print(f"\n=== Accumulating DAS image ({IMG_SIZE}^3 = {IMG_SIZE**3:,} voxels x {len(pairs)} pairs) ===")
    for i, (tx, rx) in enumerate(pairs):
        tx_pos, rx_pos = positions[tx], positions[rx]
        dist_tx = np.sqrt((RR - tx_pos[0])**2 + (CC - tx_pos[1])**2 + (ZZ - tx_pos[2])**2) * dx_m
        dist_rx = np.sqrt((RR - rx_pos[0])**2 + (CC - rx_pos[1])**2 + (ZZ - rx_pos[2])**2) * dx_m
        t_pred = (dist_tx + dist_rx) / WATER_SOUND_SPEED
        sampled = np.interp(t_pred.ravel(), lag_t, excess_energy[(tx, rx)], left=0.0, right=0.0)
        accumulator += sampled.reshape(IMG_SIZE, IMG_SIZE, IMG_SIZE)
        if (i + 1) % 500 == 0:
            print(f"  {i+1}/{len(pairs)} pairs accumulated")

    np.savez_compressed("results/das_3d_reconstruction_patient023.npz",
                         accumulator=accumulator, img_coords=img_coords,
                         positions=positions, dx_m=dx_m)
    print(f"\nSaved results/das_3d_reconstruction_patient023.npz")
    print(f"accumulator stats: min={accumulator.min():.4g}, max={accumulator.max():.4g}, "
          f"mean={accumulator.mean():.4g}")
