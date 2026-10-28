"""Multi-frequency design A1 -- how big a frequency jump can feed-forward FWI take?

Frequency continuation: stage k is started from stage k-1's model. The jump is
safe while the travel-time residual of that starting model, seen at the next
frequency, stays under half a period on (nearly) all source-receiver pairs:
    phi = 2 pi f_next |dt| < pi   (classic cycle-skipping criterion).

Proxy (straight rays, CPU, MRI heart phantom of runs -30/-35, 340^2 @ 0.5 mm,
ring R = 75 mm, 32 sources x 512 receivers):
  - "truth seen at f" = slowness perturbation Gaussian-smoothed to FWHM lambda_f/2
    (lambda in water); the wave cannot resolve finer detail.
  - "model after stage f_k" = truth smoothed to FWHM beta * lambda_k. beta is
    CALIBRATED on the real run -35 reconstruction (recM): beta such that the
    smoothed-truth residual at 0.8 MHz matches recM's residual at 0.8 MHz.
  - water start: model = water (dt = whole-object travel-time anomaly).
Validation: the same rule must say "0.3 MHz from water fails, 0.15 works"
(run -20 on patient023: configs E/F failed, A worked).

Straight rays ignore diffraction (wavefront healing) -> phases are an upper-ish
estimate for thin structures; it is a design tool, not a convergence proof.
Inputs are patient-derived (gitignored); the figure name contains "mri_heart".
"""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from scipy import ndimage

from seed import SEED, set_all_seeds
from tissue import WATER

ROOT = Path(__file__).resolve().parent.parent / "results"
ARR = ROOT / "colab_session_2026-10-26" / "arrays_mri_heart_and_chest_patient.npz"
FIG = ROOT / "figures" / "phase3_multifreq_ladder_mri_heart.png"
N, L, R_RING, N_SRC, N_REC = 340, 170e-3, 75e-3, 32, 512
DX = L / N
C0 = WATER.sound_speed
F_FINAL = 0.8e6

# per-L-BFGS-iteration cost on a T4 (run -25 profile, patient023 multires grids), s/iteration
PROF_F = np.array([0.15e6, 0.3e6, 0.5e6, 0.8e6])
PROF_S_PER_IT = np.array([9.8 / 15, 17.1 / 15, 54.3 / 15, 72.6 / 8])
MRI_SCALE = 190.0 / 154.0          # MRI heart run -35 total vs patient023 profiled total

LADDERS = {
    "0.8 (water start)": [0.8],
    "0.3 > 0.8 (run -20 E-like)": [0.3, 0.8],
    "0.15 > 0.8": [0.15, 0.8],
    "0.15 > 0.3 > 0.8": [0.15, 0.3, 0.8],
    "0.15 > 0.4 > 0.8": [0.15, 0.4, 0.8],
    "0.15 > 0.5 > 0.8": [0.15, 0.5, 0.8],
    "0.15 > 0.3 > 0.5 > 0.8 (current)": [0.15, 0.3, 0.5, 0.8],
    "0.15 > 0.25 > 0.4 > 0.8": [0.15, 0.25, 0.4, 0.8],
    "geometric x1.52 (5 st.)": [0.15, 0.228, 0.346, 0.526, 0.8],
    "0.1 > 0.3 > 0.8": [0.1, 0.3, 0.8],
}


def ring_xy(n, offset=0.0):
    a = 2 * np.pi * np.arange(n) / n + offset
    return R_RING * np.sin(a), R_RING * np.cos(a)       # (row, col) offsets as in the notebook's Ring


def travel_time_operator():
    """All source-receiver chord integrals of a slowness map (s/m) -> travel time (s)."""
    sr, sc = ring_xy(N_SRC)
    rr, rc = ring_xy(N_REC)
    n_samp = 900                                        # ~0.17 mm steps on the longest chord
    t = (np.arange(n_samp) + 0.5) / n_samp
    pr = sr[:, None, None] + (rr[None, :, None] - sr[:, None, None]) * t
    pc = sc[:, None, None] + (rc[None, :, None] - sc[:, None, None]) * t
    seg = np.hypot(rr[None, :] - sr[:, None], rc[None, :] - sc[:, None]) / n_samp
    coords = np.stack([pr / DX + N / 2, pc / DX + N / 2]).reshape(2, -1)

    def apply(ds):
        v = ndimage.map_coordinates(ds, coords, order=1, mode="constant").reshape(N_SRC, N_REC, n_samp)
        return v.sum(-1) * seg
    return apply


def smooth(ds, fwhm_m):
    return ds if fwhm_m <= 0 else ndimage.gaussian_filter(ds, fwhm_m / 2.3548 / DX, mode="constant")


def lam(f):
    return C0 / f


def seen(ds, f):
    return smooth(ds, lam(f) / 2)


def phase_stats(dt, f):
    p = 2 * np.abs(dt).ravel() * f                      # |phi| / pi
    return dict(p95=np.percentile(p, 95), p99=np.percentile(p, 99), max=p.max(), frac_skip=np.mean(p > 1))


def main():
    set_all_seeds(SEED)
    d = np.load(ARR)
    c_true, c_rec = d["cC2"].astype(float), d["recM"].astype(float)
    ds_true = 1 / c_true - 1 / C0
    ds_rec = 1 / c_rec - 1 / C0
    T = travel_time_operator()

    print("== water-start (whole-object anomaly) ==")
    dt_full = T(ds_true)
    print(f"travel-time anomaly: p95 {np.percentile(np.abs(dt_full), 95)*1e6:.2f} us, "
          f"max {np.abs(dt_full).max()*1e6:.2f} us")
    for f in [0.1e6, 0.15e6, 0.2e6, 0.25e6, 0.3e6, 0.5e6, 0.8e6]:
        s = phase_stats(T(seen(ds_true, f)), f)
        print(f"  start {f/1e6:.2f} MHz: |phi|/pi p95 {s['p95']:.2f}  p99 {s['p99']:.2f}  max {s['max']:.2f}"
              f"  rays skipped {100*s['frac_skip']:.1f}%")

    # calibrate beta on the real reconstruction: residual of recM vs truth seen at 0.8 MHz
    target_08 = T(seen(ds_true, F_FINAL))
    res_rec = phase_stats(target_08 - T(ds_rec), F_FINAL)
    betas = np.linspace(0.25, 3.0, 56)
    p95_beta = np.array([phase_stats(target_08 - T(smooth(ds_true, b * lam(F_FINAL))), F_FINAL)["p95"] for b in betas])
    beta_cal = float(np.interp(res_rec["p95"], p95_beta, betas))
    print(f"\nrecM residual at 0.8 MHz: |phi|/pi p95 {res_rec['p95']:.3f} max {res_rec['max']:.3f}"
          f" -> calibrated beta = {beta_cal:.2f} (FWHM of a stage's model = beta * lambda)")

    cache = {}                                           # T is linear: cache travel times per smoothing width

    def T_smooth(fwhm):
        key = round(fwhm * 1e7)
        if key not in cache:
            cache[key] = T(smooth(ds_true, fwhm))
        return cache[key]

    def step_stats(f_from, f_to, beta):
        model = 0.0 if f_from is None else T_smooth(beta * lam(f_from))
        return phase_stats(T_smooth(lam(f_to) / 2) - model, f_to)

    # maximum safe next frequency from each stage (max |phi| < pi; and p99 < 0.5 pi as a conservative rule)
    f_grid = np.array([0.1, 0.15, 0.2, 0.25, 0.3, 0.4, 0.5, 0.6]) * 1e6
    f_next_grid = np.linspace(0.12e6, 2.5e6, 60)
    curves = {}
    for beta in (beta_cal, 1.5 * beta_cal):
        rmax, rcons = [], []
        for fk in f_grid:
            ok_max, ok_cons = fk, fk
            for fn in f_next_grid[f_next_grid > fk]:
                s = step_stats(fk, fn, beta)
                if s["max"] < 1.0: ok_max = fn
                if s["p99"] < 0.5: ok_cons = fn
            rmax.append(ok_max / fk); rcons.append(ok_cons / fk)
        curves[beta] = (np.array(rmax), np.array(rcons))
        print(f"\nbeta {beta:.2f}: max safe jump ratio f_next/f_k (max|phi|<pi | p99<pi/2):")
        for fk, a, b in zip(f_grid, rmax, rcons):
            print(f"  from {fk/1e6:.2f} MHz: x{a:.2f} (to {a*fk/1e6:.2f}) | x{b:.2f} (to {b*fk/1e6:.2f})")

    def stage_cost(f, its):
        return MRI_SCALE * its * np.exp(np.interp(np.log(f), np.log(PROF_F), np.log(PROF_S_PER_IT)))

    print(f"\n== ladders (beta {beta_cal:.2f}); worst transition |phi|/pi max / p99; est. T4 time at 15 its/stage, 8 final ==")
    rows = []
    for name, lad in LADDERS.items():
        fs = [f * 1e6 for f in lad]
        worst_max, worst_p99, trans = 0.0, 0.0, []
        for i, f in enumerate(fs):
            s = step_stats(None if i == 0 else fs[i - 1], f, beta_cal)
            trans.append(f"{s['max']:.2f}")
            worst_max, worst_p99 = max(worst_max, s["max"]), max(worst_p99, s["p99"])
        cost = sum(stage_cost(f, 8 if i == len(fs) - 1 else 15) for i, f in enumerate(fs))
        verdict = "SKIP" if worst_max >= 1 else ("marginal" if worst_p99 >= 0.5 else "safe")
        rows.append((name, worst_max, worst_p99, cost, verdict))
        print(f"  {name:34s} per-transition max {'/'.join(trans):22s} worst p99 {worst_p99:.2f}"
              f"  ~{cost:4.0f} s  {verdict}")

    # figure
    fig, ax = plt.subplots(1, 3, figsize=(16, 4.6))
    ax[0].imshow(c_true, cmap="gray", extent=[-L / 2 * 1e3, L / 2 * 1e3] * 2)
    ax[0].add_patch(plt.Circle((0, 0), R_RING * 1e3, fill=False, color="c"))
    ax[0].set_title("MRI heart truth (c), ring 75 mm")
    fstart = np.linspace(0.08e6, 0.5e6, 30)
    st = [phase_stats(T(seen(ds_true, f)), f) for f in fstart]
    ax[1].plot(fstart / 1e6, [s["max"] for s in st], label="max")
    ax[1].plot(fstart / 1e6, [s["p99"] for s in st], label="p99")
    ax[1].axhline(1, color="r", ls="--", lw=1)
    ax[1].set_xlabel("start frequency from water (MHz)"); ax[1].set_ylabel("|phase error| / pi")
    ax[1].set_title("Water start: cycle-skip margin"); ax[1].legend()
    for beta, (rmax, rcons) in curves.items():
        ax[2].plot(f_grid / 1e6, rmax, "o-", label=f"max|phi|<pi, beta {beta:.2f}")
        ax[2].plot(f_grid / 1e6, rcons, "s--", label=f"p99<pi/2, beta {beta:.2f}")
    ax[2].set_xlabel("current stage f_k (MHz)"); ax[2].set_ylabel("max safe f_next / f_k")
    ax[2].set_title("Safe frequency jump (straight-ray proxy)"); ax[2].legend(fontsize=7)
    fig.tight_layout(); fig.savefig(FIG, dpi=130)
    print(f"\nfigure: {FIG}")


if __name__ == "__main__":
    main()
