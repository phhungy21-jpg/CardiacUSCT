"""Phase 3 -- LOCAL uniqueness of blind finite-region objects via the Jacobian.

Same object class, hidden object A, renderer and design masks as
phase3_adversarial_uniqueness.py (which was stopped: its optimizer could not
reach data misfit < 1e-3, so it could not probe the eps -> 0 limit -- LOG
run -05).

For parameter perturbation dp around A:
    image change  dO = J_img dp     data change  dd = J_d dp
Parameter directions with J_img dp = 0 are trivial (they change no visible
pixel, e.g. an occluded part of a boundary) and are removed. On the rest,
with J_img = U S V^T (non-null part), the matrix
    G = J_d V S^-1        (normalized: data rel. to |d_A|, image rel. to |O_A|)
maps a UNIT image change to the data change it causes. Its singular values
are therefore parametrization-independent:
    sigma_min(G) > 0      => no nearby object gives the same data (locally unique)
    sigma_min(G) ~ 0      => a real local ambiguity; its singular vector,
                             mapped back to dO, is the INVISIBLE image change.
Local only: says nothing about a different object far from A.
"""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch

from phase3_adversarial_uniqueness import (HI, LO, L_MM, N, N_TRUE, data,
                                           design_masks, hidden_object, render, x)
from seed import SEED, set_all_seeds

RESULTS = Path(__file__).resolve().parent.parent / "results"
SCALE = torch.from_numpy(np.tile(HI - LO, N_TRUE))
NULL_TOL = 1e-8


def image_jacobian(pA, imgA):
    """Forward mode: 32 inputs << 16384 outputs (reverse mode ran out of memory)."""
    p0 = torch.from_numpy(pA.ravel() / np.tile(HI - LO, N_TRUE))

    def img_of(u):
        return render((u * SCALE).reshape(N_TRUE, 8)).reshape(-1) / torch.linalg.norm(imgA)

    Ji = torch.autograd.functional.jacobian(img_of, p0, strategy="forward-mode",
                                            vectorize=True)
    return p0, Ji.numpy()


def data_jacobian(Ji, mask, imgA, dA):
    """Data are LINEAR in the image (FFT then select), so J_d = FFT of J_img columns."""
    cols = torch.from_numpy(Ji.T.reshape(-1, N, N)) * torch.linalg.norm(imgA)
    D = torch.fft.fft2(cols)[:, mask] / torch.linalg.norm(dA)
    return torch.cat([D.real, D.imag], dim=1).numpy().T


def dat_of_factory(mask, dA):
    def dat_of(u):
        d = data(render((u * SCALE).reshape(N_TRUE, 8)), mask) / torch.linalg.norm(dA)
        return torch.cat([d.real, d.imag])
    return dat_of


def main():
    rng = set_all_seeds(SEED)
    (RESULTS / "figures").mkdir(parents=True, exist_ok=True)
    pA = hidden_object(rng)          # same draw as the adversarial script
    imgA, labA = render(torch.from_numpy(pA), return_labels=True)
    masks = design_masks()

    # check: autograd Jacobian column vs central finite difference
    p0, Ji = image_jacobian(pA, imgA)
    name0 = [k for k in masks if "all pairs" in k][0]
    dA0 = data(imgA, masks[name0])
    Jd0 = data_jacobian(Ji, masks[name0], imgA, dA0)
    dat_of = dat_of_factory(masks[name0], dA0)
    h, j = 1e-6, 2
    e = torch.zeros_like(p0)
    e[j] = h
    fd = ((dat_of(p0 + e) - dat_of(p0 - e)) / (2 * h)).numpy()
    fd_err = np.linalg.norm(fd - Jd0[:, j]) / np.linalg.norm(fd)
    print(f"(a) Jacobian vs finite difference (column {j}): rel err {fd_err:.1e}")

    U, S, Vt = np.linalg.svd(Ji, full_matrices=False)
    keep = S > NULL_TOL * S[0]
    V, Sk = Vt[keep].T, S[keep]
    out = {}
    for name, mask in masks.items():
        dA = data(imgA, mask)
        Jd = data_jacobian(Ji, mask, imgA, dA)
        G = Jd @ (V / Sk)
        _, sg, Wt = np.linalg.svd(G, full_matrices=False)
        worst_dp = (V / Sk) @ Wt[-1]
        dO = (Ji @ worst_dp).reshape(N, N)
        out[name] = (sg, dO, int((~keep).sum()))
        print(f"{name:55s} trivial dirs {int((~keep).sum()):2d} | "
              f"sigma max {sg[0]:.2e} min {sg[-1]:.2e} | "
              f"#<1e-3: {np.sum(sg < 1e-3):2d}  #<1e-6: {np.sum(sg < 1e-6):2d}  "
              f"#<1e-10: {np.sum(sg < 1e-10):2d}  of {sg.size}")

    names = list(masks)
    fig, ax = plt.subplots(figsize=(8, 4.8))
    for name in names:
        ax.semilogy(out[name][0], ".-", label=name, ms=4)
    ax.axhline(1e-14, color="r", ls="--", lw=0.8, label="float64 floor (~1e-14)")
    ax.set(xlabel="direction index (image-changing directions only)",
           ylabel="data change per unit image change",
           title="Local uniqueness: smallest value > 0 ⇒ no nearby object fits the same data")
    ax.legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(RESULTS / "figures" / "phase3_local_jacobian_spectra.png", dpi=130)

    ext = [-L_MM / 2, L_MM / 2, -L_MM / 2, L_MM / 2]
    xs = x.numpy()
    levels = np.arange(N_TRUE + 1) + 0.5
    ncols = len(names) + 1
    fig, axes = plt.subplots(1, ncols, figsize=(3.1 * ncols, 3.9))
    vmax = np.abs(imgA.numpy()).max()
    axes[0].imshow(imgA.numpy(), origin="lower", extent=ext, cmap="RdBu_r", vmin=-vmax, vmax=vmax)
    axes[0].set_title(f"HIDDEN object A ({N_TRUE} regions)", fontsize=8)
    for a, name in zip(axes[1:], names):
        sg, dO, _ = out[name]
        m = np.abs(dO).max()
        a.imshow(dO, origin="lower", extent=ext, cmap="PuOr", vmin=-m, vmax=m)
        a.set_title(f"{name}\nleast-visible change: σ={sg[-1]:.1e}", fontsize=7)
    for a in axes:
        a.contour(xs, xs, labA.numpy(), levels=levels, colors="k", linewidths=0.7)
        a.set_xticks([]), a.set_yticks([])
    fig.suptitle("Image change the data are LEAST sensitive to (black = true outlines). "
                 "σ = data change per unit image change.", fontsize=10)
    fig.tight_layout()
    fig.savefig(RESULTS / "figures" / "phase3_local_jacobian_worst_change.png", dpi=120)
    print("figures: phase3_local_jacobian_spectra.png, phase3_local_jacobian_worst_change.png")
    print("CHECK:", "PASS" if fd_err < 1e-5 else "FAIL")


if __name__ == "__main__":
    main()
