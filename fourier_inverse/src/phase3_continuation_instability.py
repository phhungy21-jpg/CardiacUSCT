"""Why a deterministic, noise-free Fourier problem can be unique yet unstable.

A blind 1D layered object (unknown number of layers, positions, values) on a
known support [-a, a]. Data: its exact Fourier transform on a PARTIAL band of
K only. Because the support is compact, O_hat(K) is an entire analytic
function, so its values on ANY interval determine it everywhere (analytic
continuation) => the object is UNIQUELY determined. But continuation is
exponentially ill-conditioned: the singular values of the restricted Fourier
operator fall off a cliff past ~ (band width x support) / pi of them
(Slepian/prolate count). Any data error of size eps -- including float64
rounding, 1e-16 -- is multiplied by 1/sigma_min.

Shown: singular values, and recovery from exact (float64) data vs data with
tiny deterministic perturbations.
"""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from seed import SEED, set_all_seeds

RESULTS = Path(__file__).resolve().parent.parent / "results"
A_SUPPORT = 10e-3          # object lives in [-a, a]
N_X = 200                  # unknowns (dx = 0.1 mm)
BAND = (0.03, 0.30)        # measured |K| as fraction of the grid Nyquist pi/dx
N_K = 600


def layered_object(rng, x):
    n = rng.integers(3, 8)
    edges = np.sort(rng.uniform(-A_SUPPORT, A_SUPPORT, n - 1))
    vals = rng.uniform(-1, 1, n)
    return vals[np.searchsorted(edges, x)], n


def main():
    rng = set_all_seeds(SEED)
    (RESULTS / "figures").mkdir(parents=True, exist_ok=True)
    dx = 2 * A_SUPPORT / N_X
    x = -A_SUPPORT + (np.arange(N_X) + 0.5) * dx
    obj, n = layered_object(rng, x)
    kny = np.pi / dx

    Kpos = np.linspace(BAND[0] * kny, BAND[1] * kny, N_K // 2)
    K = np.concatenate([-Kpos[::-1], Kpos])
    A = np.exp(-1j * np.outer(K, x)) * dx
    s = np.linalg.svd(A, compute_uv=False)
    slepian = (2 * (Kpos[-1] - Kpos[0])) * (2 * A_SUPPORT) / (2 * np.pi)
    print(f"blind 1D object: {n} layers; unknowns {N_X}; measured band "
          f"{BAND[0]:.2f}-{BAND[1]:.2f} of Nyquist")
    print(f"sigma_max {s[0]:.2e}, sigma_min {s[-1]:.2e}, ratio {s[0]/s[-1]:.1e}")
    print(f"Slepian count (well-determined directions) ~ {slepian:.0f} of {N_X}")
    # Past ~1e-15 the computed sigmas hit the float64 SVD floor (a flat shelf);
    # the TRUE values keep falling exponentially, so count only the clean decay.
    floor = 1e-14
    print(f"directions above the float64 SVD floor (~{floor:.0e}): {np.sum(s / s[0] > floor)}"
          " -- the shelf below is rounding, true sigmas keep falling")

    y = A @ obj
    recs = {}
    for eps in [0.0, 1e-12, 1e-6]:
        pert = eps * np.linalg.norm(y) / np.sqrt(y.size) * np.cos(37.0 * np.arange(y.size))
        r = np.linalg.lstsq(A, y + pert, rcond=1e-17)[0].real
        recs[eps] = r
        print(f"  data perturbation {eps:.0e} (deterministic) -> recovery rel err "
              f"{np.linalg.norm(r - obj) / np.linalg.norm(obj):.2e}")
    stable = np.linalg.lstsq(A, y + 1e-6 * np.linalg.norm(y) / np.sqrt(y.size)
                             * np.cos(37.0 * np.arange(y.size)), rcond=1e-3)[0].real
    print(f"  stable part only (sigma > 1e-3 sigma_max), data error 1e-6 -> rel err "
          f"{np.linalg.norm(stable - obj) / np.linalg.norm(obj):.2e}")

    fig, ax = plt.subplots(1, 2, figsize=(12, 4.2))
    ax[0].semilogy(s / s[0], ".", ms=3)
    ax[0].axhline(1e-16, color="r", lw=0.8, ls="--", label="float64 rounding")
    ax[0].annotate("flat shelf = SVD rounding floor;\ntrue σ keep falling", (120, 2e-15),
                   (110, 1e-9), fontsize=8, arrowprops=dict(arrowstyle="->", lw=0.6))
    ax[0].axvline(slepian, color="gray", lw=0.8, ls=":", label="Slepian count")
    ax[0].set(title="singular values: all > 0 in exact math (UNIQUE)\n"
                    "but most are below rounding (UNSTABLE)",
              xlabel="index", ylabel="σ / σ_max")
    ax[0].legend()
    ax[1].step(x * 1e3, obj, "k", where="mid", lw=2, label=f"HIDDEN object ({n} layers)")
    for eps, r in recs.items():
        ax[1].plot(x * 1e3, r, lw=0.6, alpha=0.6, label=f"full inverse, data error {eps:.0e}")
    ax[1].plot(x * 1e3, stable, "m", lw=2, label="stable directions only, data error 1e-6")
    ax[1].set_ylim(-3, 3)
    ax[1].set(title="exact inversion of the same unique problem", xlabel="x (mm)")
    ax[1].legend(fontsize=8)
    fig.tight_layout()
    out = RESULTS / "figures" / "phase3_continuation_instability.png"
    fig.savefig(out, dpi=130)
    print(f"figure: {out}")


if __name__ == "__main__":
    main()
