"""Phase 3 -- adversarial uniqueness test for BLIND finite-region objects.

Object class: n star-shaped regions painted in order (overlap/nesting
allowed), each with 8 unknowns: center (cx, cy), base radius r0, two radial
harmonics (a1, b1, a2, b2), value v. Edges are smoothed by a sigmoid of
width W so the model is differentiable; the hidden object uses the same
class and renderer ("inverse crime" ON PURPOSE: this tests uniqueness of the
mathematical map, not robustness to model error).

Data for a design: the object's exact 2D Fourier transform on that design's
measured set C (from phase3_kspace_coverage, Hermitian-closed), linearized
Fourier-diffraction model (validity: LOG runs -02/-03).

Adversary: for each mu, minimize over region parameters B
    J(B) = |d_B - d_A|^2 / |d_A|^2  -  mu |O_B - O_A|^2 / |O_A|^2
and record (data misfit eps, object difference delta). The upper envelope
    omega(eps) = max { delta : misfit <= eps }
is an empirical modulus of continuity of the inverse map:
    omega -> 0 as eps -> 0   <=> unique (within the class)
    slope of omega            <=> stability.
A local optimizer can MISS the worst B, so omega is a LOWER bound: finding a
far-away B with tiny misfit proves practical non-uniqueness; failing to find
one is evidence, not proof.

Region count: fit n = 1..n_A+1 regions (misfit only). Random starts get
stuck in local minima (~35% misfit, an OPTIMIZATION problem, not a
uniqueness one), so seeds give fewer regions their best chance: n < n_A
starts from every subset of the true regions, n = n_A from the perturbed
truth, n_A + 1 from the truth plus one random region.
"""

from itertools import combinations

import sys
import time
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
from scipy.optimize import minimize

from phase3_kspace_coverage import DESIGNS, coverage
from seed import SEED, set_all_seeds

torch.set_default_dtype(torch.float64)
RESULTS = Path(__file__).resolve().parent.parent / "results"

N, L_MM = 128, 64.0
DX_MM = L_MM / N
W_MM = 0.25
F_BAND = (0.2e6, 0.7e6)
N_FREQ = 60
N_TRUE = 4
MUS = np.logspace(-4, 1, 7)
MAXITER = 1500

# per-region bounds: cx, cy, r0, a1, b1, a2, b2, v
LO = np.array([-18, -18, 3, -0.25, -0.25, -0.12, -0.12, -0.1])
HI = np.array([18, 18, 13, 0.25, 0.25, 0.12, 0.12, 0.1])

x = (torch.arange(N) - N / 2 + 0.5) * DX_MM
Y, X = torch.meshgrid(x, x, indexing="ij")


def render(p, return_labels=False):
    """p: (n, 8) tensor -> image; optionally the visible-region label map."""
    img = torch.zeros(N, N)
    lab = torch.zeros(N, N, dtype=torch.long)
    for i in range(p.shape[0]):
        cx, cy, r0, a1, b1, a2, b2, v = p[i]
        dx, dy = X - cx, Y - cy
        r = torch.sqrt(dx ** 2 + dy ** 2 + 1e-12)
        th = torch.atan2(dy, dx)
        rb = r0 * (1 + a1 * torch.cos(th) + b1 * torch.sin(th)
                   + a2 * torch.cos(2 * th) + b2 * torch.sin(2 * th))
        m = torch.sigmoid((rb - r) / W_MM)
        img = img * (1 - m) + v * m
        if return_labels:
            lab[m > 0.5] = i + 1
    return (img, lab) if return_labels else img


def design_masks():
    k1d = 2 * np.pi * np.fft.fftshift(np.fft.fftfreq(N, d=DX_MM * 1e-3))
    KX, KY = np.meshgrid(k1d, k1d, indexing="xy")
    freqs = np.linspace(*F_BAND, N_FREQ)
    masks = {}
    for name, d in DESIGNS.items():
        label = name.replace("\n", " ")
        masks[label + " [band]"] = coverage(d, KX, KY, freqs)
        if "pasted" in label:
            masks[label + " [1 freq]"] = coverage(d, KX, KY, [F_BAND[1]])
    # unshifted boolean masks for direct use with fft2 output
    return {k: torch.from_numpy(np.fft.ifftshift(m)) for k, m in masks.items()}


def data(img, mask):
    return torch.fft.fft2(img)[mask]


def random_params(rng, n):
    return LO + (HI - LO) * rng.uniform(size=(n, 8))


def hidden_object(rng):
    """Random object whose N_TRUE regions are all visibly present."""
    while True:
        p = random_params(rng, N_TRUE)
        p[:, 2] = rng.uniform(6, 13, N_TRUE)            # not too small
        p[:, 7] = rng.choice([-1, 1], N_TRUE) * rng.uniform(0.03, 0.1, N_TRUE)
        _, lab = render(torch.from_numpy(p), return_labels=True)
        if all((lab == i + 1).sum() > 150 for i in range(N_TRUE)):
            return p


def fit(p0, objective, n):
    lo, hi = np.tile(LO, n), np.tile(HI, n)

    def fg(flat):
        t = torch.tensor(flat.reshape(n, 8), requires_grad=True)
        J = objective(t)
        J.backward()
        return J.item(), t.grad.numpy().ravel()

    res = minimize(fg, np.clip(p0.ravel(), lo, hi), jac=True, method="L-BFGS-B",
                   bounds=list(zip(lo, hi)),
                   options=dict(maxiter=MAXITER, ftol=1e-16, gtol=1e-12))
    return res.x.reshape(n, 8)


def rel(a, b):
    return (torch.linalg.norm(a - b) / torch.linalg.norm(b)).item()


def main(quick=False):
    rng = set_all_seeds(SEED)
    (RESULTS / "figures").mkdir(parents=True, exist_ok=True)
    pA = hidden_object(rng)
    imgA, labA = render(torch.from_numpy(pA), return_labels=True)
    masks = design_masks()

    t0 = time.time()
    _ = fit(pA + 0.01, lambda t: torch.sum(render(t) ** 2), N_TRUE)
    print(f"timing: one short fit {time.time() - t0:.1f}s")
    if quick:
        return

    names = list(masks)
    results, worst = {}, {}
    for name in names:
        mask = masks[name]
        dA = data(imgA, mask)
        nA2, oA2 = torch.sum(torch.abs(dA) ** 2), torch.sum(imgA ** 2)
        pts = []
        for mu in MUS:
            obj = lambda t, mu=mu: (torch.sum(torch.abs(data(render(t), mask) - dA) ** 2) / nA2
                                    - mu * torch.sum((render(t) - imgA) ** 2) / oA2)
            starts = [pA + 0.05 * (HI - LO) * rng.standard_normal(pA.shape),
                      random_params(rng, N_TRUE)]
            for s in starts:
                pB = fit(s, obj, N_TRUE)
                imgB = render(torch.from_numpy(pB))
                eps = rel(data(imgB, mask), dA)
                delta = rel(imgB, imgA)
                pts.append((eps, delta, pB))
        results[name] = pts
        ok = [p for p in pts if p[0] <= 1e-2]
        worst[name] = max(ok, key=lambda p: p[1]) if ok else min(pts, key=lambda p: p[0])
        env = lambda e: max([p[1] for p in pts if p[0] <= e], default=float("nan"))
        print(f"{name:55s} omega(1e-4)={env(1e-4):.3f}  omega(1e-3)={env(1e-3):.3f}  "
              f"omega(1e-2)={env(1e-2):.3f}  ({time.time() - t0:.0f}s)")
        sys.stdout.flush()

    # region-count identifiability on two contrasting designs
    count = {}
    for name in [n for n in names if "all pairs" in n or "anterior" in n]:
        mask = masks[name]
        dA = data(imgA, mask)
        nA2 = torch.sum(torch.abs(dA) ** 2)
        obj = lambda t: torch.sum(torch.abs(data(render(t), mask) - dA) ** 2) / nA2
        for n in range(1, N_TRUE + 2):
            if n < N_TRUE:
                seeds = [pA[list(c)] for c in combinations(range(N_TRUE), n)]
            elif n == N_TRUE:
                seeds = [pA + 0.05 * (HI - LO) * rng.standard_normal(pA.shape)]
            else:
                seeds = [np.vstack([pA, random_params(rng, 1)])]
            best = min(rel(data(render(torch.from_numpy(fit(s, obj, n))), mask), dA)
                       for s in seeds)
            count[(name, n)] = best
            print(f"  count test {name[:40]:40s} n={n}: best misfit {best:.2e}")
            sys.stdout.flush()

    # --- figures ---
    fig, ax = plt.subplots(figsize=(7.5, 5))
    for name in names:
        e = np.array([p[0] for p in results[name]])
        d = np.array([p[1] for p in results[name]])
        o = np.argsort(e)
        env = np.maximum.accumulate(d[o])
        ax.loglog(e[o], np.maximum(env, 1e-4), "-", label=name)
    ax.set(xlabel="data misfit ε (relative)", ylabel="max object difference found ω(ε)",
           title="Adversarial uniqueness: does ω(ε) → 0 as ε → 0?")
    ax.legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(RESULTS / "figures" / "phase3_adversarial_omega.png", dpi=130)

    ext = [-L_MM / 2, L_MM / 2, -L_MM / 2, L_MM / 2]
    vmax = np.abs(imgA.numpy()).max()
    ncols = len(names) + 1
    fig, axes = plt.subplots(1, ncols, figsize=(3.2 * ncols, 3.8))
    levels = np.arange(N_TRUE + 1) + 0.5
    xs = x.numpy()

    def panel(a, img, title, labB=None):
        a.imshow(img, origin="lower", extent=ext, cmap="RdBu_r", vmin=-vmax, vmax=vmax)
        a.contour(xs, xs, labA.numpy(), levels=levels, colors="k", linewidths=0.8)
        if labB is not None:
            a.contour(xs, xs, labB, levels=levels, colors="lime", linewidths=0.9,
                      linestyles="--")
        a.set_title(title, fontsize=8)
        a.set_xticks([]), a.set_yticks([])

    panel(axes[0], imgA.numpy(), f"HIDDEN object A ({N_TRUE} regions)\nblack = true outlines")
    for a, name in zip(axes[1:], names):
        eps, delta, pB = worst[name]
        imgB, labB = render(torch.from_numpy(pB), return_labels=True)
        panel(a, imgB.numpy(), f"{name}\nmost different B with misfit {eps:.0e}:\n"
                               f"differs by {delta:.2f}  (green = B outlines)", labB.numpy())
    fig.tight_layout()
    fig.savefig(RESULTS / "figures" / "phase3_adversarial_worst_B.png", dpi=120)
    print("figures: phase3_adversarial_omega.png, phase3_adversarial_worst_B.png")
    print(f"total {time.time() - t0:.0f}s")


if __name__ == "__main__":
    main(quick="--quick" in sys.argv)
