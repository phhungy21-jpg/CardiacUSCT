"""Phase 3 — simplified sanity test: reconstruct a beating CIRCLE (not the
full myocardial ring) from acoustic reflections, visualized as a
filmstrip of frames across the cardiac cycle.

Per user request: simplify to the minimal meaningful test --
- one filled disk (single boundary, not a ring with inner+outer walls)
- one probe array, one focused transmit per frame, many virtual receivers
- per-pixel DAS reconstruction (reusing the validated, bug-fixed formula
  from phase3_das_beamforming.py -- corrected per-element earliest-
  arrival transmit-time model)
- ARTIFACT CANCELLATION: the recurring fixed-depth coherent-summation
  artifact (confirmed in phase3_das_beamforming.py and
  phase3_four_probe_focused.py to be medium-independent -- identical in
  a homogeneous medium with NO reflector at all) is removed by
  subtracting a one-time reference reconstruction of the homogeneous
  background (no circle) from every frame. This is principled, not a
  blind hack: the control test already proved the artifact doesn't
  depend on what's in the medium, so subtracting it should cancel it
  while preserving the true (differential) reflector signal.
- Goal: visual assessment (a movie/filmstrip), not a numeric RMSE claim
  -- the point is to SEE the circle beat, as a sanity check before
  returning to the harder myocardial-wall case.
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
center = (150, 150)
array_y = 30

N_CHANNELS = 32
CHANNEL_SPAN_CELLS = 200
IMAGE_SIZE = 150
FOCUS_ROW, FOCUS_COL = float(center[0]), float(center[1])


def build_medium_circle(radius_cells):
    """Single filled disk (blood) in a chest-wall-proxy background --
    ONE boundary, not a ring with inner+outer walls."""
    yy, xx = np.mgrid[0:N[0], 0:N[1]]
    dist = np.sqrt((xx - center[1]) ** 2 + (yy - center[0]) ** 2)
    inside = dist < radius_cells
    sound_speed_map = np.where(inside, cfg.BLOOD.sound_speed, cfg.CHEST_WALL_PROXY.sound_speed).astype(np.float32)
    density_map = np.where(inside, cfg.BLOOD.density, cfg.CHEST_WALL_PROXY.density).astype(np.float32)
    ssm = jnp.expand_dims(jnp.array(sound_speed_map), -1)
    dm = jnp.expand_dims(jnp.array(density_map), -1)
    return Medium(domain=domain, sound_speed=FourierSeries(ssm, domain),
                  density=FourierSeries(dm, domain))


def build_medium_homogeneous():
    """The artifact-only reference: chest-wall-proxy everywhere, no
    circle/reflector at all."""
    sound_speed_map = np.full(N, cfg.CHEST_WALL_PROXY.sound_speed, dtype=np.float32)
    density_map = np.full(N, cfg.CHEST_WALL_PROXY.density, dtype=np.float32)
    ssm = jnp.expand_dims(jnp.array(sound_speed_map), -1)
    dm = jnp.expand_dims(jnp.array(density_map), -1)
    return Medium(domain=domain, sound_speed=FourierSeries(ssm, domain),
                  density=FourierSeries(dm, domain))


_dummy_medium = build_medium_circle(p3cfg.LV_RADIUS_ED_CELLS)
_base_time_axis = TimeAxis.from_medium(_dummy_medium, cfl=cfg.CFL)
dt = _base_time_axis.dt
t_end = 0.5 * _base_time_axis.t_end
time_axis = TimeAxis(dt=dt, t_end=t_end)
n_steps = int(time_axis.Nt)
t_arr = np.arange(n_steps) * dt


def toneburst(t, t_delay=0.0):
    tau = t - t_delay
    duration = cfg.N_CYCLES / cfg.F0_HZ
    sigma = duration / 6
    window = np.exp(-(tau - duration / 2) ** 2 / (2 * sigma ** 2))
    return np.sin(2 * np.pi * cfg.F0_HZ * tau) * window


# Focused transmit (validated in phase3_das_beamforming.py, bug #3 fix).
_tx_xs = np.linspace(center[1] - 140, center[1] + 140, cfg.N_ELEMENTS).astype(int)
_tx_ys = np.full(cfg.N_ELEMENTS, array_y, dtype=int)
_dist_to_focus = np.sqrt((_tx_xs - FOCUS_COL) ** 2 + (_tx_ys - FOCUS_ROW) ** 2) * dx[0]
_focus_delays = (_dist_to_focus.max() - _dist_to_focus) / c_ref
_signal = jnp.array(np.stack([toneburst(t_arr, d) for d in _focus_delays]))
sources = Sources(positions=(list(_tx_xs), list(_tx_ys)), signals=_signal, dt=dt, domain=domain)


@jit
def run(medium):
    return simulate_wave_propagation(medium, time_axis, sources=sources)


def extract_channels(medium):
    pressure = run(medium)
    field = pressure.on_grid[..., 0]
    channel_xs = np.linspace(center[1] - CHANNEL_SPAN_CELLS // 2,
                             center[1] + CHANNEL_SPAN_CELLS // 2, N_CHANNELS).astype(int)
    channels = np.array(field[:, channel_xs, array_y])  # established field[:, x, y] convention
    return channels, channel_xs


def das_reconstruct(channels, channel_xs):
    """Corrected DAS: true per-element earliest-arrival transmit time
    (fix #4 from phase3_das_beamforming.py -- valid before AND after
    focus, unlike the 'virtual point source at focus' shortcut)."""
    img_rows = np.linspace(array_y, N[0] - 1, IMAGE_SIZE)
    img_cols = np.linspace(0, N[1] - 1, IMAGE_SIZE)
    RR, CC = np.meshgrid(img_rows, img_cols, indexing="ij")

    t_tx = np.full(RR.shape, np.inf)
    for elem_x, elem_delay in zip(_tx_xs, _focus_delays):
        dist_elem = np.sqrt((RR - array_y) ** 2 + (CC - elem_x) ** 2) * dx[0]
        t_tx = np.minimum(t_tx, elem_delay + dist_elem / c_ref)

    image = np.zeros((IMAGE_SIZE, IMAGE_SIZE))
    for ch_idx, ch_x in enumerate(channel_xs):
        dist_rx = np.sqrt((RR - array_y) ** 2 + (CC - ch_x) ** 2) * dx[0]
        t_total = t_tx + dist_rx / c_ref
        image += np.interp(t_total, t_arr, channels[:, ch_idx], left=0, right=0)
    return image, img_rows, img_cols


if __name__ == "__main__":
    print(f"*** {labels.PENDING_SIGNOFF_BANNER} ***")
    print(f"*** {labels.TOY_EXACT_GT_CAPTION} ***")
    print("Beating-circle reconstruction movie: one filled disk, focused "
          "transmit, DAS reconstruction, artifact cancellation via "
          "homogeneous-medium subtraction.")

    print("\n=== Reference (homogeneous, no circle) -- for artifact cancellation ===")
    channels_ref, channel_xs = extract_channels(build_medium_homogeneous())
    image_ref, img_rows, img_cols = das_reconstruct(channels_ref, channel_xs)

    N_FRAMES_MOVIE = 12
    phases = np.linspace(0, 1, N_FRAMES_MOVIE)
    radii_cells = [p3cfg.lv_radius_at_phase(p) for p in phases]

    frames = []
    for i, r in enumerate(radii_cells):
        print(f"=== Frame {i+1}/{N_FRAMES_MOVIE} (radius={r:.1f} cells = {r*dx[0]*1e3:.2f}mm) ===")
        channels, channel_xs = extract_channels(build_medium_circle(r))
        image, _, _ = das_reconstruct(channels, channel_xs)
        image_clean = image - image_ref  # artifact cancellation
        frames.append(image_clean)

    # BIDIRECTIONAL SEQUENTIAL TRACKING (see chat log): a pure forward-only
    # sequential search correctly tracks frames 0-3, then drifts once the
    # true signal weakens near mid-contraction, and gets PERMANENTLY stuck
    # on the fixed residual artifact for the rest of the cycle -- once
    # drifted, a forward-only tracker has no way back. Fix: exploit the
    # cycle's known symmetry (ED at both frame 0 and the last frame, ES in
    # the middle) -- track FORWARD from frame 0 and BACKWARD from the last
    # frame simultaneously, meeting in the middle. Drift then only has
    # ~half the frames to accumulate before reaching the weak-signal
    # region from EITHER direction, rather than all 11 steps from one end.
    col_idx = np.argmin(np.abs(img_cols - center[1]))
    SEARCH_MARGIN_CELLS = 15
    search_mask_wide = (img_rows > array_y + 15) & (img_rows < center[0])

    def peak_at(frame_idx, mask):
        profile = np.abs(frames[frame_idx][:, col_idx])
        row = img_rows[mask][np.argmax(profile[mask])]
        val = profile[mask].max()
        return row, val

    n = len(frames)
    mid = n // 2  # split point: frames [0..mid-1] from forward pass, [mid..n-1] from backward pass
    tracked_rows = [None] * n
    amp_conf = [None] * n

    tracked_rows[0], amp_conf[0] = peak_at(0, search_mask_wide)
    for i in range(1, mid):
        mask = ((img_rows > tracked_rows[i - 1] - SEARCH_MARGIN_CELLS) &
                (img_rows < tracked_rows[i - 1] + SEARCH_MARGIN_CELLS))
        tracked_rows[i], amp_conf[i] = peak_at(i, mask)

    tracked_rows[n - 1], amp_conf[n - 1] = peak_at(n - 1, search_mask_wide)
    for i in range(n - 2, mid - 1, -1):
        mask = ((img_rows > tracked_rows[i + 1] - SEARCH_MARGIN_CELLS) &
                (img_rows < tracked_rows[i + 1] + SEARCH_MARGIN_CELLS))
        tracked_rows[i], amp_conf[i] = peak_at(i, mask)

    print("--- Amplitude-based (bidirectional range tracking) ---")
    for i in range(n):
        expected_row = center[0] - radii_cells[i]
        pass_label = "forward" if i < mid else "backward"
        print(f"  frame {i} ({pass_label}): expected_row={expected_row:.1f}, "
              f"tracked_row={tracked_rows[i]:.1f}, confidence={amp_conf[i]:.4f}")

    # DOPPLER/MTI-STYLE FUSION (per user direction): frame-differencing
    # (frame[i]-frame[i-1]) cancels the stationary artifact more cleanly
    # than the homogeneous-reference subtraction (both frames share the
    # same real array geometry), giving a STRONGER signal near ED where
    # wall velocity is high -- but it goes to ~zero exactly at ES, where
    # velocity crosses zero (a real, clinically-documented Tissue Doppler
    # limitation, not a bug). Fuse: at each frame, use whichever cue
    # (amplitude-based range, or differencing-based velocity) has higher
    # confidence (peak signal strength) at that specific frame.
    diff_row = [None] * n
    diff_conf = [None] * n
    for i in range(1, n):
        diff_frame = frames[i] - frames[i - 1]
        profile = np.abs(diff_frame[:, col_idx])
        mask = search_mask_wide
        diff_row[i] = img_rows[mask][np.argmax(profile[mask])]
        diff_conf[i] = profile[mask].max()

    print("\n--- Fused (amplitude vs. differencing, whichever is more confident) ---")
    fused_rows = [tracked_rows[0]]  # frame 0 has no differencing available
    fused_method = ["amplitude (anchor)"]
    for i in range(1, n):
        if diff_conf[i] > amp_conf[i]:
            fused_rows.append(diff_row[i])
            fused_method.append("differencing")
        else:
            fused_rows.append(tracked_rows[i])
            fused_method.append("amplitude")
    for i in range(n):
        expected_row = center[0] - radii_cells[i]
        err_mm = abs(fused_rows[i] - expected_row) * dx[0] * 1e3
        print(f"  frame {i}: expected_row={expected_row:.1f}, fused_row={fused_rows[i]:.1f} "
              f"({fused_method[i]}), error={err_mm:.2f}mm")
    tracked_rows = fused_rows  # use the fused result for the figure below

    # Filmstrip: grid of frames across the cycle, |image| for visibility.
    n_cols = 6
    n_rows = int(np.ceil(N_FRAMES_MOVIE / n_cols))
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(3 * n_cols, 3.3 * n_rows))
    axes = np.array(axes).reshape(-1)
    vmax = max(np.abs(f).max() for f in frames)
    for i, (ax, frame, r) in enumerate(zip(axes, frames, radii_cells)):
        ax.imshow(np.abs(frame), cmap="hot", vmin=0, vmax=vmax, origin="upper",
                  extent=[img_cols.min(), img_cols.max(), img_rows.max(), img_rows.min()])
        # overlay the TRUE circle boundary for visual reference
        theta = np.linspace(0, 2 * np.pi, 100)
        true_row = center[0] + r * np.sin(theta)
        true_col = center[1] + r * np.cos(theta)
        ax.plot(true_col, true_row, "c--", linewidth=1, alpha=0.7)
        # mark the bidirectional-sequential-tracked position
        ax.plot(center[1], tracked_rows[i], "g+", markersize=14, markeredgewidth=2)
        err_mm = abs(tracked_rows[i] - (center[0] - r)) * dx[0] * 1e3
        ax.set_title(f"phase={phases[i]:.2f}, r={r*dx[0]*1e3:.2f}mm\ntracked err={err_mm:.2f}mm", fontsize=8)
        ax.axis("off")
    for ax in axes[len(frames):]:
        ax.axis("off")
    fig.suptitle("Beating circle: DAS-reconstructed frames (artifact-cancelled)\n"
                "cyan dashed = true circle boundary\n(TOY: exact prescribed ground truth)",
                fontsize=11)
    plt.tight_layout(rect=[0, 0.02, 1, 0.90])
    labels.add_banner(fig)
    os.makedirs("results/figures", exist_ok=True)
    plt.savefig("results/figures/phase3_beating_circle_movie.png", dpi=130)
    print("\nSaved results/figures/phase3_beating_circle_movie.png")
