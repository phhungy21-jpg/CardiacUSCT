"""Phase 1: cylindrical scanner element layout -- the 3D analog of
`jwave_ring_tomography`'s single-ring/36-angle convention. Per the
design point decided in runs -05/-07/-08 (N=150, domain=130mm,
dx=0.867mm, f0=0.8MHz) and patient023's real heart bounding box
(~90mm, run -05): 8 rings (Z positions) x 12 elements/ring = 96 total
element positions on a cylinder, probe radius chosen to sit in the
water margin between the heart surface and the domain edge (leaving
room for a PML absorbing boundary at the domain edge too).

SEQUENTIAL acquisition (unchanged from `jwave_ring_tomography`'s own
Phase 0.1 decision, run -02 there): one element fires at a time. Unlike
2D, receivers must be specified via jWave's `sensors` API BEFORE running
each simulation (run -07's fix) -- but this is not a real limitation in
practice, since ALL other 95 elements can be recorded simultaneously in
ONE simulation call, exactly mirroring 2D's "multiple receivers per
transmit is free" finding (`jwave_ring_tomography` run -04).
"""

import numpy as np

N_RINGS = 8
N_PER_RING = 12
N_ELEMENTS = N_RINGS * N_PER_RING  # 96

N_GRID = 150
DOMAIN_MM = 130.0
DX_MM = DOMAIN_MM / N_GRID
CENTER_CELL = N_GRID // 2

PROBE_RADIUS_MM = 56.0  # fits between the ~90mm heart bbox and the 130mm domain edge
PROBE_RADIUS_CELLS = PROBE_RADIUS_MM / DX_MM

# ring z-positions: spread across the domain's central region (avoiding the very top/bottom
# where the resampled heart volume may not extend, and leaving PML standoff at the edges)
Z_SPAN_MM = 80.0  # comfortably within the ~90mm heart bbox's own through-plane extent
Z_SPAN_CELLS = Z_SPAN_MM / DX_MM


def element_positions():
    """Returns a list of (row, col, z) grid-cell coordinates for all 96
    cylindrical scanner elements."""
    positions = []
    ring_thetas = np.linspace(0, 360, N_PER_RING, endpoint=False)
    ring_zs = np.linspace(CENTER_CELL - Z_SPAN_CELLS / 2, CENTER_CELL + Z_SPAN_CELLS / 2, N_RINGS)
    for z in ring_zs:
        for theta_deg in ring_thetas:
            theta = np.deg2rad(theta_deg)
            row = CENTER_CELL - PROBE_RADIUS_CELLS * np.cos(theta)
            col = CENTER_CELL + PROBE_RADIUS_CELLS * np.sin(theta)
            positions.append((round(row), round(col), round(z)))
    return positions


if __name__ == "__main__":
    positions = element_positions()
    print(f"N_ELEMENTS={len(positions)} ({N_RINGS} rings x {N_PER_RING}/ring)")
    print(f"PROBE_RADIUS_CELLS={PROBE_RADIUS_CELLS:.1f} ({PROBE_RADIUS_MM}mm)")
    print(f"first 3 positions: {positions[:3]}")
    print(f"last 3 positions: {positions[-3:]}")
    rows, cols, zs = zip(*positions)
    print(f"row range: [{min(rows)}, {max(rows)}] (grid is 0-{N_GRID-1})")
    print(f"col range: [{min(cols)}, {max(cols)}] (grid is 0-{N_GRID-1})")
    print(f"z range: [{min(zs)}, {max(zs)}] (grid is 0-{N_GRID-1})")
    assert min(rows) >= 0 and max(rows) < N_GRID, "row out of bounds"
    assert min(cols) >= 0 and max(cols) < N_GRID, "col out of bounds"
    assert min(zs) >= 0 and max(zs) < N_GRID, "z out of bounds"
    print("All element positions within grid bounds.")
