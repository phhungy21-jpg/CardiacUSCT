"""Phase 3 pivot #2 — delay-and-sum (DAS) beamformed image reconstruction.

Per run -21's finding: single-point multi-angle triangulation cannot
recover a 2D vector (specular sensitivity is always normal-only). The
correct fix, per user direction: reconstruct a full spatial image via
standard delay-and-sum beamforming (one broad transmit + many receive
channels + per-pixel coherent summation), then track the boundary/
surface WITHIN that reconstructed image across frames -- genuine 2D
spatial localization, not a single time-of-flight number. This is
literally how clinical B-mode / synthetic-aperture / plane-wave
ultrasound imaging works.

Design (first cut, deliberately scoped):
- ONE plane-wave transmit (all array elements fire synchronously) --
  the "1 source of echo" broadly illuminating the field of view.
- Many VIRTUAL receive channels extracted from the SAME simulation's
  full spatial field (jWave gives the full field for free -- no extra
  simulation cost for more "receivers").
- Per-pixel DAS: for each image pixel, sum all receive channels' signal
  values at the geometrically-correct time-of-flight (plane-wave
  transmit arrival time + pixel-to-channel travel time), assuming a
  constant reference sound speed (standard DAS simplification -- no
  refraction/heterogeneous-speed correction, flagged as such).
- Coarser image grid (150x150) than the underlying 300x300 simulation
  grid, to keep per-frame reconstruction cost reasonable.
- Reuses the clean (non-speckle, non-attenuating) ring phantom -- this
  tests image-formation validity, a separable question from attenuation/
  speckle (both already explored elsewhere this session).
"""

import numpy as np
from jax import numpy as jnp
from jax import jit

from jwave import FourierSeries
from jwave.geometry import Domain, Medium, TimeAxis, Sources
from jwave.acoustics import simulate_wave_propagation

import phase2_config as cfg
import phase3_config as p3cfg
import labels

from matplotlib import pyplot as plt
import os

c_ref = cfg.CHEST_WALL_PROXY.sound_speed

N = (300, 300)
dx = (cfg.DX_M, cfg.DX_M)
domain = Domain(N, dx)
center = (N[0] // 2, N[1] // 2)
array_y = 30

N_CHANNELS = 32
CHANNEL_SPAN_CELLS = 200  # +/-10mm receive aperture
IMAGE_SIZE = 150  # coarser than the 300x300 sim grid, for reconstruction speed


def build_medium(lv_radius_cells, wall_thickness_cells):
    yy, xx = np.mgrid[0:N[0], 0:N[1]]
    dist = np.sqrt((xx - center[1]) ** 2 + (yy - center[0]) ** 2)
    label_map = np.zeros(N, dtype=int)
    label_map[dist < lv_radius_cells + wall_thickness_cells] = 2
    label_map[dist < lv_radius_cells] = 3
    sound_speed_map = np.zeros(N, dtype=np.float32)
    density_map = np.zeros(N, dtype=np.float32)
    for label, tissue in cfg.ACDC_LABEL_TO_TISSUE.items():
        m = label_map == label
        sound_speed_map[m] = tissue.sound_speed
        density_map[m] = tissue.density
    ssm = jnp.expand_dims(jnp.array(sound_speed_map), -1)
    dm = jnp.expand_dims(jnp.array(density_map), -1)
    return (Medium(domain=domain, sound_speed=FourierSeries(ssm, domain),
                   density=FourierSeries(dm, domain)), label_map)


_dummy_medium, _ = build_medium(p3cfg.LV_RADIUS_ED_CELLS, p3cfg.WALL_THICKNESS_CELLS)
_base_time_axis = TimeAxis.from_medium(_dummy_medium, cfl=cfg.CFL)
dt = _base_time_axis.dt
t_end = 0.5 * _base_time_axis.t_end  # covers transmit -> whole ring -> back
time_axis = TimeAxis(dt=dt, t_end=t_end)
n_steps = int(time_axis.Nt)  # BUG CAUGHT: TimeAxis.Nt uses ceil(), not round() --
                              # a manual round()-based n_steps caused an off-by-one
                              # length mismatch against the actual simulated field.
t_arr = np.arange(n_steps) * dt


def toneburst(t):
    duration = cfg.N_CYCLES / cfg.F0_HZ
    sigma = duration / 6
    window = np.exp(-(t - duration / 2) ** 2 / (2 * sigma ** 2))
    return np.sin(2 * np.pi * cfg.F0_HZ * t) * window


# FOCUSED transmit (see chat log): an unfocused point source put the
# boundary echo ~1700x weaker than the direct wave (0.00013 vs 0.22
# amplitude) -- a genuine signal-to-clutter problem, not a bug. Fixed by
# reusing the validated delay-focusing law (phase2_forward_model.py /
# toy_2d_array_source.py) to concentrate transmit energy at the ring's
# center, matching the successful single-echo tests earlier this session.
FOCUS_ROW, FOCUS_COL = float(center[0]), float(center[1])
_tx_xs = np.linspace(center[1] - 140, center[1] + 140, cfg.N_ELEMENTS).astype(int)
_tx_ys = np.full(cfg.N_ELEMENTS, array_y, dtype=int)
_dist_to_focus = np.sqrt((_tx_xs - FOCUS_COL) ** 2 + (_tx_ys - FOCUS_ROW) ** 2) * dx[0]
_focus_delays = (_dist_to_focus.max() - _dist_to_focus) / c_ref
_signal = jnp.array(np.stack([toneburst(t_arr - d) for d in _focus_delays]))
plane_wave_sources = Sources(positions=(list(_tx_xs), list(_tx_ys)), signals=_signal,
                             dt=dt, domain=domain)
# Time all element wavefronts converge at the focus (same physics as
# phase3_vector_triangulation.py's expected_round_trip_time derivation).
T_FOCUS_ARRIVAL = _dist_to_focus.max() / c_ref


@jit
def run(medium):
    return simulate_wave_propagation(medium, time_axis, sources=plane_wave_sources)


def simulate_and_extract_channels(lv_radius_cells):
    medium, label_map = build_medium(lv_radius_cells, p3cfg.WALL_THICKNESS_CELLS)
    pressure = run(medium)
    field = pressure.on_grid[..., 0]  # (n_steps, N0, N1)
    channel_xs = np.linspace(center[1] - CHANNEL_SPAN_CELLS // 2,
                             center[1] + CHANNEL_SPAN_CELLS // 2, N_CHANNELS).astype(int)
    # BUG CAUGHT DURING VALIDATION (see chat log): every other script this
    # session indexes field as [:, x_value, y_value] (column-like index
    # FIRST, row-like SECOND) -- this line originally had it backwards
    # ([:, array_y, channel_xs]), causing an asymmetric reconstruction
    # artifact. Fixed to match the established convention.
    channels = np.array(field[:, channel_xs, array_y])  # (n_steps, N_CHANNELS)
    return channels, channel_xs, label_map


def das_reconstruct(channels, channel_xs):
    """Standard delay-and-sum: for each image pixel, sum every channel's
    signal at the geometrically-correct time-of-flight (plane-wave
    transmit arrival + pixel-to-channel travel), constant-speed
    assumption (c_ref), linear interpolation in time."""
    img_rows = np.linspace(array_y, N[0] - 1, IMAGE_SIZE)
    img_cols = np.linspace(0, N[1] - 1, IMAGE_SIZE)
    RR, CC = np.meshgrid(img_rows, img_cols, indexing="ij")

    # Focused-transmit arrival time, CORRECTED (see chat log): the
    # "virtual point source at the focus" approximation only holds
    # BEYOND the focus -- our targets (myocardial boundary) sit BEFORE
    # the focus (ring center), in the still-converging region where that
    # shortcut is wrong. Correct approach: the wave genuinely reaches a
    # pixel as soon as the EARLIEST element's (delayed) wavefront gets
    # there -- compute min over all elements of (delay_i +
    # dist(element_i, pixel)/c_ref) directly, valid everywhere (before,
    # at, or after the focus), not just in the far field.
    t_tx = np.full(RR.shape, np.inf)
    for elem_x, elem_delay in zip(_tx_xs, _focus_delays):
        dist_elem = np.sqrt((RR - array_y) ** 2 + (CC - elem_x) ** 2) * dx[0]
        t_tx = np.minimum(t_tx, elem_delay + dist_elem / c_ref)

    image = np.zeros((IMAGE_SIZE, IMAGE_SIZE))
    for ch_idx, ch_x in enumerate(channel_xs):
        dist_rx = np.sqrt((RR - array_y) ** 2 + (CC - ch_x) ** 2) * dx[0]
        t_rx = dist_rx / c_ref
        t_total = t_tx + t_rx
        # Linear interpolation of this channel's trace at t_total (per pixel).
        image += np.interp(t_total, t_arr, channels[:, ch_idx], left=0, right=0)
    return image, img_rows, img_cols


if __name__ == "__main__":
    print(f"*** {labels.PENDING_SIGNOFF_BANNER} ***")
    print(f"*** {labels.TOY_EXACT_GT_CAPTION} ***")
    print("DAS beamformed image reconstruction: 1 plane-wave transmit + "
          f"{N_CHANNELS} virtual receive channels -> {IMAGE_SIZE}x{IMAGE_SIZE} image.")

    for label, lv_radius in [("ED", p3cfg.LV_RADIUS_ED_CELLS),
                             ("ES", p3cfg.LV_RADIUS_ES_CELLS)]:
        print(f"\n=== Frame: {label} (lv_radius={lv_radius} cells) ===")
        channels, channel_xs, label_map = simulate_and_extract_channels(lv_radius)
        image, img_rows, img_cols = das_reconstruct(channels, channel_xs)

        # Validate: does the reconstructed image show a bright ring at the
        # correct radius? Extract a radial brightness profile along the
        # vertical (on-axis) direction for a quick, honest check.
        col_center_idx = np.argmin(np.abs(img_cols - center[1]))
        profile = np.abs(image[:, col_center_idx])

        # NEAR-FIELD/DIRECT-WAVE EXCLUSION (see chat log): pixels very
        # close to the transmit source have near-zero round-trip time
        # regardless of any real reflector, so the direct (non-reflected)
        # wave's own energy dominates the reconstruction there -- a known
        # DAS artifact (real systems need TGC/near-field blanking for the
        # same reason). Exclude rows within ~25 cells of the array before
        # looking for the true boundary peak.
        near_field_exclude_rows = array_y + 25
        excl_mask = img_rows > near_field_exclude_rows
        profile_excl = np.where(excl_mask, profile, 0)
        expected_outer_r_cells = lv_radius + p3cfg.WALL_THICKNESS_CELLS
        expected_row = center[0] - expected_outer_r_cells if False else array_y + expected_outer_r_cells
        # (outer boundary is BELOW the array by expected_outer_r_cells,
        # i.e. at absolute row = array_y + (center[0]-array_y) - expected_outer_r_cells... )
        expected_row = center[0] - expected_outer_r_cells
        peak_row = img_rows[np.argmax(profile)]
        peak_row_excl = img_rows[np.argmax(profile_excl)]
        print(f"On-axis profile: expected outer-boundary row~{expected_row:.1f}")
        print(f"  raw peak (incl. near-field artifact) at row~{peak_row:.1f}")
        print(f"  peak AFTER near-field exclusion at row~{peak_row_excl:.1f} "
              f"(diff={abs(peak_row_excl-expected_row):.1f} cells = "
              f"{abs(peak_row_excl-expected_row)*dx[0]*1e3:.2f}mm)")

        fig, axes = plt.subplots(1, 2, figsize=(13, 5.5))
        axes[0].imshow(np.abs(image), cmap="hot", origin="upper",
                       extent=[img_cols.min(), img_cols.max(), img_rows.max(), img_rows.min()])
        axes[0].set_title(f"DAS-reconstructed |image|, frame={label}")
        axes[0].set_xlabel("col"); axes[0].set_ylabel("row")
        axes[1].plot(img_rows, profile)
        axes[1].axvline(expected_row, color="k", linestyle="--", label="expected boundary")
        axes[1].axvline(peak_row, color="r", linestyle=":", label="reconstructed peak")
        axes[1].set_xlabel("row"); axes[1].set_ylabel("|image| on-axis")
        axes[1].set_title("On-axis profile validation")
        axes[1].legend(fontsize=8)
        fig.suptitle(f"DAS beamforming validation, frame={label}\n(TOY: exact prescribed ground truth)")
        plt.tight_layout(rect=[0, 0.06, 1, 0.92])
        labels.add_banner(fig)
        os.makedirs("results/figures", exist_ok=True)
        plt.savefig(f"results/figures/phase3_das_beamforming_{label}.png", dpi=150)
        print(f"Saved results/figures/phase3_das_beamforming_{label}.png")
