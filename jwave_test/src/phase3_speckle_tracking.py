"""Phase 3 pivot — distributed speckle tracking (first cut).

Per run -16's finding: single-echo boundary detection (Levels 0-4) is
capped by a structural ambiguity between two comparably-weak reflecting
interfaces, not by detector sophistication -- and this ceiling sits right
in the realistic clinical SNR range (20-34dB) for cardiac ultrasound.

This pivots to tracking a point WITHIN the myocardial wall via its
scattering texture (distributed sub-resolution scatterers), rather than
range-detecting a single boundary echo -- the same principle real 2D
speckle-tracking echocardiography uses. Key difference from Levels 0-4:
there, we detected "an echo" (ambiguous WHICH echo); here, we track how a
whole local RF waveform PATTERN shifts between frames, using many
scatterers' aggregate signature rather than betting on one reflection.

SIMPLIFICATIONS (flagged, not hidden -- this is a first proof-of-concept,
not a publication-grade speckle simulation):
- Scatterers are placed via a thin-wall approximation (uniform in
  normalized radial fraction rho, not area-corrected for the annulus).
- Motion model: pure radial scaling (each scatterer's rho, theta fixed;
  actual radius = lv_radius(t) + rho*wall_thickness) -- matches the ring
  phantom's existing motion model, not full myocardial mechanics.
- Only ~400 scatterers on this domain -- sparser than true fully-developed
  speckle would need, sufficient to test the TRACKING PRINCIPLE, not to
  validate realistic speckle statistics.
- Tracks ONE material point (mid-wall, rho=0.5), not a full 2D
  displacement field -- proof of concept, not full 2D speckle tracking.
"""

import numpy as np
from scipy.signal import correlate

import phase3_motion_recovery as pmr
import phase2_config as cfg
import phase3_config as p3cfg
import labels

from matplotlib import pyplot as plt
import os

c_ref = cfg.CHEST_WALL_PROXY.sound_speed

# --- Fixed "material" scatterer field, generated once ---------------------
N_SCATTERERS = 400
_scatterer_rng = np.random.default_rng(42)  # fixed seed, per CLAUDE.md
SCATTERER_RHO = _scatterer_rng.uniform(0, 1, N_SCATTERERS)      # normalized radial position within wall
SCATTERER_THETA = _scatterer_rng.uniform(0, 2 * np.pi, N_SCATTERERS)
SCATTERER_DC = _scatterer_rng.normal(0, 15.0, N_SCATTERERS)     # m/s perturbation, fixed per scatterer
SCATTERER_DRHO_DENSITY = _scatterer_rng.normal(0, 10.0, N_SCATTERERS)  # kg/m^3 perturbation


def build_medium_with_speckle(lv_radius_cells, wall_thickness_cells):
    """Same base tissue map as phase3_motion_recovery.build_medium, PLUS
    scatterer perturbations mapped to this frame's actual geometry (the
    scatterers move WITH the material -- same rho/theta, different
    absolute position as lv_radius changes)."""
    N = pmr.N
    center = pmr.center
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

    # Map each scatterer's (rho, theta) to this frame's actual grid position.
    r_actual = lv_radius_cells + SCATTERER_RHO * wall_thickness_cells
    sx = np.clip((center[1] + r_actual * np.cos(SCATTERER_THETA)).astype(int), 0, N[1] - 1)
    sy = np.clip((center[0] + r_actual * np.sin(SCATTERER_THETA)).astype(int), 0, N[0] - 1)
    sound_speed_map[sy, sx] += SCATTERER_DC
    density_map[sy, sx] += SCATTERER_DRHO_DENSITY

    from jwave import FourierSeries
    from jwave.geometry import Medium
    sound_speed_map = np.expand_dims(sound_speed_map, -1)
    density_map = np.expand_dims(density_map, -1)
    import jax.numpy as jnp
    return Medium(domain=pmr.domain,
                  sound_speed=FourierSeries(jnp.array(sound_speed_map), pmr.domain),
                  density=FourierSeries(jnp.array(density_map), pmr.domain))


def simulate_frame_speckle(lv_radius_cells):
    medium = build_medium_with_speckle(lv_radius_cells, p3cfg.WALL_THICKNESS_CELLS)
    pressure = pmr.run(medium)
    field = pressure.on_grid[..., 0]
    return np.array(field[:, pmr.rcv_x, pmr.array_y])


def expected_mid_wall_range_mm(lv_radius_cells, wall_thickness_cells):
    r_mid = lv_radius_cells + 0.5 * wall_thickness_cells
    return pmr.vertical_dist_mm - r_mid * cfg.DX_M * 1e3


_duration = cfg.N_CYCLES / cfg.F0_HZ
_template_t = np.arange(0, _duration, pmr.dt)
_template = pmr.toneburst(_template_t)

REF_WINDOW_HALFWIDTH_S = _duration  # reference window width around mid-wall echo
SEARCH_MARGIN_S = 3.0e-6  # same margin as Level 2, sized to plausible motion range


def extract_reference_window(reference_trace, t_mid_ref):
    idx_center = int(round(t_mid_ref / pmr.dt))
    half_w = int(round(REF_WINDOW_HALFWIDTH_S / pmr.dt))
    return reference_trace[max(0, idx_center - half_w):idx_center + half_w]


def track_mid_wall_range_mm(trace, ref_window, t_mid_ref, c_ref):
    window_mask = ((pmr.t_arr > max(t_mid_ref - SEARCH_MARGIN_S, pmr.DIRECT_EXCLUDE_S)) &
                   (pmr.t_arr < t_mid_ref + SEARCH_MARGIN_S))
    segment = trace[window_mask]
    if len(segment) < len(ref_window):
        return np.nan
    corr = np.abs(correlate(segment, ref_window, mode="valid"))
    idx_local = np.argmax(corr)
    t_window_start = pmr.t_arr[window_mask][0]
    t_echo = t_window_start + idx_local * pmr.dt + REF_WINDOW_HALFWIDTH_S
    return t_echo * c_ref / 2 * 1e3


if __name__ == "__main__":
    print(f"*** {labels.PENDING_SIGNOFF_BANNER} ***")
    print(f"*** {labels.TOY_EXACT_GT_CAPTION} ***")
    print("Distributed speckle tracking: tracking a mid-wall material point "
          "via scattering texture, not a single boundary echo.")

    phases = np.linspace(0, 1, p3cfg.N_FRAMES)
    lv_radii_motion = [p3cfg.lv_radius_at_phase(p) for p in phases]
    expected_mid_wall_mm = np.array([
        expected_mid_wall_range_mm(r, p3cfg.WALL_THICKNESS_CELLS) for r in lv_radii_motion
    ])
    # For comparability with earlier results, also compute as "recovered
    # outer radius"-equivalent: vertical_dist - range = a material-point
    # "radius" analogous to outer_radius_mm used in runs -07/-14/-15/-16.
    ground_truth_mid_wall_radius_mm = pmr.vertical_dist_mm - expected_mid_wall_mm

    print("=== Simulating speckle-textured frames ===")
    traces_speckle = [simulate_frame_speckle(r) for r in lv_radii_motion]

    t_mid_ref = expected_mid_wall_mm[0] * 2 * 1e-3 / c_ref  # round-trip time at frame 0
    ref_window = extract_reference_window(traces_speckle[0], t_mid_ref)
    print(f"Reference mid-wall time: {t_mid_ref:.3e}s, window length: {len(ref_window)} samples")

    # Noiseless per-frame sanity check BEFORE trusting the noisy sweep.
    print("\nNoiseless per-frame check (tracked vs expected mid-wall radius):")
    for i, trace in enumerate(traces_speckle):
        tracked_range = track_mid_wall_range_mm(trace, ref_window, t_mid_ref, c_ref)
        tracked_radius = pmr.vertical_dist_mm - tracked_range
        print(f"  frame {i}: expected={ground_truth_mid_wall_radius_mm[i]:.3f}mm "
              f"tracked={tracked_radius:.3f}mm")

    # FAIRNESS FIX (caught by comparing signal amplitudes before trusting
    # the result -- see chat log): the mid-wall speckle window's peak is
    # only ~0.15% of the overall trace's peak (dominated by the strong
    # boundary echoes elsewhere in the trace). Scaling noise to "X% of
    # THIS trace's own peak" -- the convention used in runs -07/-14/-15/-16
    # for boundary detection -- makes noise ~13x LARGER than the speckle
    # signal at noise=0.02, an apples-to-oranges comparison (real receiver
    # noise is a fixed absolute floor, not something that scales with
    # whatever the strongest reflector in the scene happens to be). Fixed:
    # noise is scaled to a FIXED reference peak (the overall trace peak,
    # same for every frame/noise-level, computed once) instead of each
    # call's own local peak.
    reference_peak = np.max(np.abs(traces_speckle[0]))

    N_REALIZATIONS = 20
    print(f"\n(RMSE averaged over {N_REALIZATIONS} independent noise realizations, "
          f"noise scaled to a FIXED reference peak={reference_peak:.4f}, not each trace's own peak)")
    print(f"\n{'noise':>8} {'speckle-tracking RMSE (mm)':>28}")
    speckle_rmses = []
    baseline_pred = np.mean(ground_truth_mid_wall_radius_mm)
    baseline_rmse = np.sqrt(np.mean((ground_truth_mid_wall_radius_mm - baseline_pred) ** 2))
    for noise_level in p3cfg.NOISE_LEVELS:
        rmses = []
        for trial in range(N_REALIZATIONS):
            rng = np.random.default_rng(1000 * trial + 7)
            recovered = []
            for trace in traces_speckle:
                noisy = trace + rng.normal(0, noise_level * reference_peak, size=trace.shape)
                r = track_mid_wall_range_mm(noisy, ref_window, t_mid_ref, c_ref)
                recovered.append(pmr.vertical_dist_mm - r)
            recovered = np.array(recovered)
            rmses.append(np.sqrt(np.mean((recovered - ground_truth_mid_wall_radius_mm) ** 2)))
        speckle_rmses.append(np.mean(rmses))
        print(f"{noise_level:>8} {np.mean(rmses):>28.4f}")

    print(f"\nnaive constant-baseline RMSE: {baseline_rmse:.4f}mm")
    print("\nFor comparison, Level 2 (boundary ref-tracking, run -15/-16): "
          "0.40 / 1.01 / 1.01 / 1.01 mm at noise 0.0/0.02/0.05/0.10")

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot([0.0] + p3cfg.NOISE_LEVELS[1:] if 0.0 in p3cfg.NOISE_LEVELS else p3cfg.NOISE_LEVELS,
            speckle_rmses, "o-", color="purple", label="distributed speckle tracking")
    ax.axhline(baseline_rmse, color="gray", linestyle=":", label="naive constant baseline")
    ax.set_xlabel("noise level (arbitrary, fraction of trace peak)")
    ax.set_ylabel("RMSE vs. expected mid-wall position (mm)")
    ax.set_title("Distributed speckle tracking\n(TOY: exact prescribed ground truth)")
    ax.legend(fontsize=9)
    plt.tight_layout(rect=[0, 0.06, 1, 1])
    labels.add_banner(fig)
    os.makedirs("results/figures", exist_ok=True)
    plt.savefig("results/figures/phase3_speckle_tracking.png", dpi=150)
    print("Saved results/figures/phase3_speckle_tracking.png")
