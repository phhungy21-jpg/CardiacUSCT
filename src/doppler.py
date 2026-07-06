"""Phase 4 — synthetic multi-angle Doppler-like projection of the Phase 3
ground-truth displacement field.

Probe geometry: 3 in-plane probe positions at angles [-45, 0, 45] degrees
(anterior-left arc only), justified physiologically — real transthoracic
acoustic windows (parasternal, apical, subcostal) are confined to the
anterior/left chest and cannot access the posterior chest wall (blocked by
spine/lungs). Probes are treated as lying in the same short-axis slice
plane as the imaging data (a documented simplification — see LIMITATIONS.md
on through-plane motion being unresolved by this dataset already).

Signal: for each probe and each voxel, the "Doppler-like measurement" is the
component of the ED->ES displacement vector along the probe-to-voxel unit
vector (radial projection). This is a projected DISPLACEMENT (mm), not a
true instantaneous velocity, because Phase 3 produces one displacement field
per patient (ED->ES only, not a continuous cine) — see LOG.md's ED->ES
scoping decision. Framed honestly as a displacement proxy, not velocity.

Noise calibration (not invented — see LOG.md for citations): additive
Gaussian, SD = 1.0mm, based on tissue Doppler myocardial DISPLACEMENT
reproducibility literature reporting SDs of ~0.6-1.3mm across comparison
methods (spectral/colour TDI vs. anatomic M-mode) in normal and ischemic
myocardium.
"""

import numpy as np

PROBE_ANGLES_DEG = (-45.0, 0.0, 45.0)
PROBE_RADIUS_MM = 150.0
NOISE_SD_MM = 1.0


def probe_positions_xy(image_center_xy: tuple, angles_deg: tuple = PROBE_ANGLES_DEG,
                        radius_mm: float = PROBE_RADIUS_MM) -> np.ndarray:
    """Returns (n_probes, 2) array of probe (x, y) positions in mm, at the
    given angles measured from the +y (anterior) axis, at fixed radius from
    the image center."""
    cx, cy = image_center_xy
    positions = []
    for angle_deg in angles_deg:
        theta = np.deg2rad(angle_deg)
        x = cx + radius_mm * np.sin(theta)
        y = cy + radius_mm * np.cos(theta)
        positions.append((x, y))
    return np.array(positions)


def voxel_grid_xy_mm(shape_yx: tuple, spacing_xy: tuple, center_xy: tuple) -> tuple:
    """Returns (X, Y) meshgrids of physical in-plane coordinates (mm) for an
    array of shape (Y, X), centered at center_xy."""
    ny, nx = shape_yx
    sx, sy = spacing_xy
    xs = (np.arange(nx) - nx / 2) * sx + center_xy[0]
    ys = (np.arange(ny) - ny / 2) * sy + center_xy[1]
    X, Y = np.meshgrid(xs, ys)
    return X, Y


def synthesize_doppler(displacement_field: np.ndarray, spacing: tuple, add_noise: bool,
                        rng: np.random.Generator = None) -> np.ndarray:
    """displacement_field: (Z, Y, X, 3) mm, in (dz, dy, dx) order matching
    sitk.GetArrayFromImage convention used in Phase 3.
    Returns (n_probes, Z, Y, X) projected radial displacement, mm."""
    z_dim, y_dim, x_dim, _ = displacement_field.shape
    sx, sy, _ = spacing
    center_xy = (0.0, 0.0)  # crop is already centered on the heart (Phase 2)
    X, Y = voxel_grid_xy_mm((y_dim, x_dim), (sx, sy), center_xy)
    probes = probe_positions_xy(center_xy)

    dx = displacement_field[..., 2]  # (Z, Y, X)
    dy = displacement_field[..., 1]

    n_probes = len(probes)
    output = np.zeros((n_probes, z_dim, y_dim, x_dim), dtype=np.float32)

    for i, (px, py) in enumerate(probes):
        vec_x = X - px  # (Y, X), probe -> voxel
        vec_y = Y - py
        norm = np.sqrt(vec_x ** 2 + vec_y ** 2)
        unit_x = vec_x / norm
        unit_y = vec_y / norm
        # broadcast over Z
        projected = dx * unit_x[None, :, :] + dy * unit_y[None, :, :]
        output[i] = projected

    if add_noise:
        rng = rng or np.random.default_rng()
        output = output + rng.normal(0.0, NOISE_SD_MM, size=output.shape).astype(np.float32)

    return output


def probe_unit_vectors_at_voxel(shape_yx: tuple, spacing_xy: tuple) -> np.ndarray:
    """Returns (n_probes, Y, X, 2) unit vectors (probe -> voxel, in-plane)
    for use in the closed-form least-squares recovery (Gate 4 check)."""
    center_xy = (0.0, 0.0)
    X, Y = voxel_grid_xy_mm(shape_yx, spacing_xy, center_xy)
    probes = probe_positions_xy(center_xy)
    n_probes = len(probes)
    ny, nx = shape_yx
    unit_vecs = np.zeros((n_probes, ny, nx, 2), dtype=np.float64)
    for i, (px, py) in enumerate(probes):
        vec_x = X - px
        vec_y = Y - py
        norm = np.sqrt(vec_x ** 2 + vec_y ** 2)
        unit_vecs[i, :, :, 0] = vec_x / norm
        unit_vecs[i, :, :, 1] = vec_y / norm
    return unit_vecs


def recover_displacement_least_squares(projections: np.ndarray, unit_vecs: np.ndarray) -> np.ndarray:
    """projections: (n_probes, Z, Y, X). unit_vecs: (n_probes, Y, X, 2).
    Solves the per-voxel 2D least-squares system p_i = d . u_i for d=(dx,dy).
    Returns (Z, Y, X, 2) recovered (dx, dy)."""
    n_probes, z_dim, y_dim, x_dim = projections.shape
    recovered = np.zeros((z_dim, y_dim, x_dim, 2), dtype=np.float64)

    for y in range(y_dim):
        for x in range(x_dim):
            U = unit_vecs[:, y, x, :]  # (n_probes, 2)
            p = projections[:, :, y, x]  # (n_probes, Z)
            # (U^T U)^-1 U^T p, solved for all Z at once
            sol, *_ = np.linalg.lstsq(U, p, rcond=None)  # (2, Z)
            recovered[:, y, x, :] = sol.T

    return recovered
