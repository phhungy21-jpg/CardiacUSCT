"""Phase 3 visual -- what each acquisition design directly measures of a BLIND
object, with the hidden object's true region outlines traced on every panel.

The object is random (unknown number of regions, irregular shapes, nesting,
synthetic contrast values -- NOT tissue values); the solver side never uses
its generator parameters. For each design the panel shows the linear
minimum-norm image: inverse FFT of the object spectrum restricted to the
measured set C (Hermitian-closed). Nothing is assumed about the object, so
this is exactly "what the data say directly". What is missing from it is
NOT necessarily lost -- see phase3_continuation_instability.py and LOG run -04.
"""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from phase3_kspace_coverage import (DESIGNS, F_BAND, L_DOM, N_BAND, N_GRID,
                                    coverage, k_grid)
from seed import SEED, set_all_seeds

RESULTS = Path(__file__).resolve().parent.parent / "results"


def blind_object(rng, n_regions=None):
    """Random star-shaped regions painted in order (later ones overwrite, which
    produces nesting/overlap = multiple unknown layers). Returns (contrast, labels)."""
    x = (np.arange(N_GRID) - N_GRID / 2 + 0.5) * (L_DOM / N_GRID)
    X, Y = np.meshgrid(x, x, indexing="xy")
    obj = np.zeros((N_GRID, N_GRID))
    lab = np.zeros((N_GRID, N_GRID), dtype=int)
    n = n_regions or rng.integers(5, 9)
    for i in range(1, n + 1):
        cx, cy = rng.uniform(-18e-3, 18e-3, 2)
        r0 = rng.uniform(6e-3, 22e-3)
        harm = rng.integers(2, 6)
        amps = rng.uniform(0, 0.35, harm) / np.arange(1, harm + 1)
        phases = rng.uniform(0, 2 * np.pi, harm)
        th = np.arctan2(Y - cy, X - cx)
        rb = r0 * (1 + sum(a * np.cos((j + 1) * th + p)
                           for j, (a, p) in enumerate(zip(amps, phases))))
        m = np.hypot(X - cx, Y - cy) < rb
        obj[m] = rng.uniform(-0.08, 0.08)
        lab[m] = i
    return obj, lab, n


def main():
    rng = set_all_seeds(SEED)
    (RESULTS / "figures").mkdir(parents=True, exist_ok=True)
    obj, lab, n = blind_object(rng)
    print(f"blind object: {n} painted regions (hidden from the solver side)")

    KX, KY = k_grid()
    F = np.fft.fftshift(np.fft.fft2(obj))
    freqs = np.linspace(*F_BAND, N_BAND)
    ext = np.array([-1, 1, -1, 1]) * L_DOM / 2 * 1e3
    vmax = np.abs(obj).max()

    fig, axes = plt.subplots(2, 4, figsize=(17, 8.8))
    axes = axes.ravel()

    def show(a, img, title):
        a.imshow(img, origin="lower", extent=ext, cmap="RdBu_r", vmin=-vmax, vmax=vmax)
        a.contour(np.linspace(ext[0], ext[1], N_GRID), np.linspace(ext[2], ext[3], N_GRID),
                  lab, levels=np.arange(lab.max() + 1) + 0.5, colors="k", linewidths=0.7)
        a.set_title(title, fontsize=9)
        a.set_xlim(-45, 45), a.set_ylim(-45, 45)
        a.set_xticks([]), a.set_yticks([])

    show(axes[0], obj, f"HIDDEN object ({n} regions)\nblack = true outlines")
    for a, name in zip(axes[1:], DESIGNS):
        mask = coverage(DESIGNS[name], KX, KY, freqs)
        rec = np.real(np.fft.ifft2(np.fft.ifftshift(F * mask)))
        err = np.linalg.norm(rec - obj) / np.linalg.norm(obj)
        show(a, rec, f"{name.replace(chr(10), ' ')}\nmeasured directly; rel. miss {err:.2f}")
        print(f"  {name.replace(chr(10), ' '):42s} relative missing part {err:.3f}")

    a = axes[7]
    a.imshow(np.log10(np.abs(F) + 1e-12), origin="lower", cmap="magma",
             extent=np.array([KX.min(), KX.max(), KY.min(), KY.max()]) / 1e3)
    kmax = 2 * np.pi * F_BAND[1] / 1520.0
    a.add_patch(plt.Circle((0, 0), 2 * kmax / 1e3, fill=False, color="c", ls="--"))
    a.set_title("object spectrum log|Ô(K)|\ncyan: |K| = 2k_max (full-ring limit)", fontsize=9)
    a.set_xticks([]), a.set_yticks([])

    fig.suptitle("0.5–2.5 MHz. Each panel: the part of the blind object the design measures "
                 "directly (no assumptions, no continuation).", fontsize=11)
    fig.tight_layout()
    out = RESULTS / "figures" / "phase3_blind_object_views.png"
    fig.savefig(out, dpi=120)
    print(f"figure: {out}")


if __name__ == "__main__":
    main()
