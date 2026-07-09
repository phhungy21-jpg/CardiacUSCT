"""Shared tomography reconstruction algorithms for the ring/water-bath
project.

`unfiltered_backprojection` matches run -04/-05's original method
(smear each ray's excess delay uniformly along its path, normalize by
overlap count) -- kept here for direct side-by-side comparison, not
because it's the recommended method; it produces the known 36-point
star artifact from a small number of discrete view angles.

`sirt_reconstruct` is new: a Simultaneous Iterative Reconstructive
Technique (SIRT)-style solver, addressing that artifact by iteratively
minimizing the mismatch between the image's predicted per-ray average
and the actually-measured excess delay, rather than a single raw sum.
Standard, well-established fix for exactly this class of streak
artifact in sparse-angle tomography.
"""

import numpy as np

RAY_HALF_WIDTH_CELLS = 3.0


def _build_ray_masks(pairs_excess_delay_ns, probe_position_fn, img_size, N):
    img_rows = np.linspace(0, N[0], img_size)
    img_cols = np.linspace(0, N[1], img_size)
    RR, CC = np.meshgrid(img_rows, img_cols, indexing="ij")
    keys = list(pairs_excess_delay_ns.keys())
    n_rays = len(keys)
    masks = np.zeros((n_rays, img_size, img_size), dtype=np.float32)
    y = np.zeros(n_rays, dtype=np.float32)
    for i, (theta_tx, theta_rx) in enumerate(keys):
        p1 = probe_position_fn(theta_tx)
        p2 = probe_position_fn(theta_rx)
        d_row, d_col = p2[0] - p1[0], p2[1] - p1[1]
        length = np.hypot(d_row, d_col)
        w_row, w_col = RR - p1[0], CC - p1[1]
        t = (w_row * d_row + w_col * d_col) / (length ** 2)
        perp = np.abs(w_row * d_col - w_col * d_row) / length
        masks[i] = ((t >= 0) & (t <= 1) & (perp < RAY_HALF_WIDTH_CELLS)).astype(np.float32)
        y[i] = pairs_excess_delay_ns[(theta_tx, theta_rx)]
    return masks, y, img_rows, img_cols


def unfiltered_backprojection(pairs_excess_delay_ns, probe_position_fn, img_size, N):
    masks, y, img_rows, img_cols = _build_ray_masks(pairs_excess_delay_ns, probe_position_fn, img_size, N)
    accumulator = (masks * y[:, None, None]).sum(axis=0)
    weight = masks.sum(axis=0)
    image = np.divide(accumulator, weight, out=np.zeros_like(accumulator), where=weight > 0)
    return image, img_rows, img_cols


def sirt_reconstruct(pairs_excess_delay_ns, probe_position_fn, img_size, N, n_iters=30, relax=0.15):
    masks, y, img_rows, img_cols = _build_ray_masks(pairs_excess_delay_ns, probe_position_fn, img_size, N)
    ray_lengths = masks.sum(axis=(1, 2))
    ray_lengths = np.where(ray_lengths > 0, ray_lengths, 1.0)
    pixel_hits = masks.sum(axis=0)
    pixel_hits_safe = np.where(pixel_hits > 0, pixel_hits, 1.0)

    x = np.zeros((img_size, img_size), dtype=np.float32)
    residual_history = []
    for it in range(n_iters):
        pred = (masks * x[None, :, :]).sum(axis=(1, 2)) / ray_lengths
        residual = y - pred
        residual_history.append(float(np.sqrt(np.mean(residual ** 2))))
        update = (masks * residual[:, None, None]).sum(axis=0) / pixel_hits_safe
        x = x + relax * update
    return x, img_rows, img_cols, residual_history
