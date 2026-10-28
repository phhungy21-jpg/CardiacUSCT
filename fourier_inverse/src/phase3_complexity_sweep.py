"""Phase 3 -- local uniqueness vs. object complexity, with SHARP corners.

Object class: n star-shaped POLYGONS (straight edges, V vertices each),
painted in order (overlap/nesting allowed). Per region unknowns:
center (cx, cy), value v, and per vertex a radius r_j and an angular
jitter d_j (vertex angle theta_j = 2 pi (j + d_j) / V, |d_j| < 0.3 keeps
the order). 2V + 3 unknowns per region. Vertices with a smaller radius
than their neighbours give CONCAVE (re-entrant) corners; hidden objects
draw radii from r0 * U(0.5, 1) so both convex and concave corners occur.

Ray-segment boundary: for a pixel at angle th from the center, with u =
(cos th, sin th) and the edge P_j -> P_{j+1},
    r_b(th) = cross(P_j, P_{j+1} - P_j) / cross(u, P_{j+1} - P_j).
Edges are smoothed by a sigmoid of width W (differentiability).

Same local-uniqueness measure as phase3_local_uniqueness_jacobian.py:
singular values of G = (data change) per unit (image change), restricted
to image-changing directions. sigma_min ~ 0 => a nearby different object
fits the same data (local NON-uniqueness).
"""

import time
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch

from phase3_adversarial_uniqueness import L_MM, N, W_MM, X, Y, data, design_masks, x
from seed import SEED, set_all_seeds

torch.set_default_dtype(torch.float64)
RESULTS = Path(__file__).resolve().parent.parent / "results"
REGIONS = [2, 4, 6, 8]
VERTICES = [5, 8, 12, 16]
N_SEEDS = 3
NULL_TOL = 1e-8


def cross(ax, ay, bx, by):
    return ax * by - ay * bx


def render_polys(flat, n, V, return_labels=False):
    p = flat.reshape(n, 2 * V + 3)
    img = torch.zeros(N, N)
    lab = torch.zeros(N, N, dtype=torch.long)
    j = torch.arange(V, dtype=torch.float64)
    for i in range(n):
        cx, cy, v = p[i, 0], p[i, 1], p[i, 2]
        r, d = p[i, 3:3 + V], p[i, 3 + V:]
        th = 2 * np.pi * (j + d) / V
        Px, Py = r * torch.cos(th), r * torch.sin(th)
        dx, dy = X - cx, Y - cy
        rr = torch.sqrt(dx ** 2 + dy ** 2 + 1e-12)
        ang = torch.atan2(dy, dx)
        rel = torch.remainder(ang - th[0].detach(), 2 * np.pi)
        th_rel = torch.remainder(th.detach() - th[0].detach(), 2 * np.pi)
        k = (torch.searchsorted(th_rel.contiguous(), rel.reshape(-1).contiguous(),
                                right=True) - 1).clamp(0, V - 1).reshape(N, N)
        kn = (k + 1) % V
        ax, ay = Px[k], Py[k]
        ex, ey = Px[kn] - ax, Py[kn] - ay
        rb = cross(ax, ay, ex, ey) / cross(torch.cos(ang), torch.sin(ang), ex, ey)
        m = torch.sigmoid((rb - rr) / W_MM)
        img = img * (1 - m) + v * m
        if return_labels:
            lab[m > 0.5] = i + 1
    return (img, lab) if return_labels else img


def hidden_polys(rng, n, V):
    rmax = 13.0 if n <= 4 else 10.0
    for _ in range(2000):
        rows, concave = [], 0
        for _ in range(n):
            r0 = rng.uniform(0.6 * rmax, rmax)
            r = r0 * rng.uniform(0.5, 1.0, V)
            nb = np.minimum(np.roll(r, 1), np.roll(r, -1))
            concave += int(np.sum(r < 0.85 * nb))
            rows.append(np.concatenate([rng.uniform(-16, 16, 2),
                                        [rng.choice([-1, 1]) * rng.uniform(0.03, 0.1)],
                                        r, rng.uniform(-0.25, 0.25, V)]))
        flat = torch.from_numpy(np.concatenate(rows))
        _, lab = render_polys(flat, n, V, return_labels=True)
        if all((lab == i + 1).sum() > 60 for i in range(n)):
            return flat, concave
    raise RuntimeError("could not place visible regions")


def local_sigmas(flat, n, V, masks):
    img0 = render_polys(flat, n, V)
    f = lambda u: render_polys(u, n, V).reshape(-1) / torch.linalg.norm(img0)
    Ji = torch.autograd.functional.jacobian(f, flat, strategy="forward-mode",
                                            vectorize=True).numpy()
    U, S, Vt = np.linalg.svd(Ji, full_matrices=False)
    keep = S > NULL_TOL * S[0]
    Vk, Sk = Vt[keep].T, S[keep]
    cols = torch.from_numpy((Ji @ (Vk / Sk)).T.reshape(-1, N, N)) * torch.linalg.norm(img0)
    out = {}
    for name, mask in masks.items():
        dA = data(img0, mask)
        D = torch.fft.fft2(cols)[:, mask] / torch.linalg.norm(dA)
        G = torch.cat([D.real, D.imag], dim=1).numpy().T
        # When unknowns > data rows, Wt[-1] must be an exact NULL vector, which
        # needs full_matrices. Only then: full_matrices on a 21818-row G builds a
        # 3.8 GB U and made the sweep ~200x slower (LOG run -06).
        _, sg, Wt = np.linalg.svd(G, full_matrices=G.shape[0] < G.shape[1])
        sg = np.concatenate([sg, np.zeros(max(0, Vk.shape[1] - sg.size))])
        worst = (Ji @ ((Vk / Sk) @ Wt[-1])).reshape(N, N)
        # Hermitian-closed mask: K and -K carry the same information for a
        # real object, so independent real data ~= number of masked K points
        out[name] = (sg, worst, int(mask.sum()))
    return out, int(keep.sum()), img0


def main():
    rng = set_all_seeds(SEED)
    (RESULTS / "figures").mkdir(parents=True, exist_ok=True)
    masks = design_masks()
    names = list(masks)
    t0 = time.time()

    # check: forward-mode column vs finite difference on one polygon object
    flat, _ = hidden_polys(rng, 2, 8)
    img0 = render_polys(flat, 2, 8)
    Jcol = torch.autograd.functional.jacobian(
        lambda u: render_polys(u, 2, 8).reshape(-1), flat,
        strategy="forward-mode", vectorize=True)[:, 5]
    e = torch.zeros_like(flat)
    e[5] = 1e-6
    fd = (render_polys(flat + e, 2, 8) - render_polys(flat - e, 2, 8)).reshape(-1) / 2e-6
    fd_err = (torch.linalg.norm(fd - Jcol) / torch.linalg.norm(fd)).item()
    print(f"(a) polygon Jacobian vs finite difference: rel err {fd_err:.1e}")

    table, gallery = {}, {}
    for n in REGIONS:
        for V in VERTICES:
            mins = {nm: [] for nm in names}
            for s in range(N_SEEDS):
                flat, concave = hidden_polys(rng, n, V)
                out, n_eff, img0 = local_sigmas(flat, n, V, masks)
                for nm in names:
                    mins[nm].append(out[nm][0][-1])
                if s == 0 and (n, V) in [(2, 5), (4, 12), (8, 16)]:
                    gallery[(n, V)] = (flat, out, concave)
            table[(n, V)] = (n * (2 * V + 3), n_eff,
                             {nm: float(np.median(mins[nm])) for nm in names})
            row = "  ".join(f"{np.median(mins[nm]):8.1e}" for nm in names)
            print(f"n={n} V={V:2d} unknowns={n*(2*V+3):3d} (image-changing {n_eff:3d}) | {row}"
                  f"  ({time.time()-t0:.0f}s)", flush=True)
    print("columns: " + " | ".join(names))
    print("independent real data per design: " + ", ".join(f"{out[nm][2]}" for nm in names))

    # --- figure 1: sigma_min vs unknowns ---
    fig, ax = plt.subplots(figsize=(8.5, 5))
    for nm in names:
        u = np.array([table[k][0] for k in table])
        s = np.array([table[k][2][nm] for k in table])
        o = np.argsort(u)
        ax.semilogy(u[o], np.maximum(s[o], 1e-17), "o-", ms=3, label=nm)
    ax.axhline(1e-14, color="r", ls="--", lw=0.8, label="float64 floor")
    for nm, col in [(next(n_ for n_ in names if "time-of-flight" in n_), "tab:purple"),
                    (next(n_ for n_ in names if "1 freq" in n_), "tab:orange")]:
        cnt = int(masks[nm].sum())
        ax.axvline(cnt, color=col, ls=":", lw=1,
                   label=f"independent data, {nm.split('[')[0].strip()[:22]}… = {cnt}")
    ax.set(xlabel="unknowns (regions × (2·vertices + 3))",
           ylabel="σ_min (median of 3 hidden objects)",
           title="Local uniqueness vs. complexity, sharp convex + concave corners")
    ax.legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(RESULTS / "figures" / "phase3_complexity_sigma_min.png", dpi=130)

    # --- figure 2: gallery, outlines traced ---
    show = [nm for nm in names if "all pairs" in nm or "anterior" in nm
            or ("pasted" in nm and "[band]" in nm) or "time-of-flight" in nm]
    ext = [-L_MM / 2, L_MM / 2, -L_MM / 2, L_MM / 2]
    xs = x.numpy()
    fig, axes = plt.subplots(len(gallery), len(show) + 1,
                             figsize=(3.0 * (len(show) + 1), 3.1 * len(gallery)))
    for r, ((n, V), (flat, out, concave)) in enumerate(gallery.items()):
        img, lab = render_polys(flat, n, V, return_labels=True)
        levels = np.arange(n + 1) + 0.5
        vm = np.abs(img.numpy()).max()
        a = axes[r, 0]
        a.imshow(img.numpy(), origin="lower", extent=ext, cmap="RdBu_r", vmin=-vm, vmax=vm)
        a.set_title(f"HIDDEN: {n} regions × {V} vertices\n"
                    f"{n*(2*V+3)} unknowns, {concave} concave corners", fontsize=8)
        for c, nm in enumerate(show, start=1):
            sg, worst, _ = out[nm]
            a = axes[r, c]
            if worst is not None:
                m = np.abs(worst).max()
                a.imshow(worst, origin="lower", extent=ext, cmap="PuOr", vmin=-m, vmax=m)
            a.set_title(f"{nm}\nleast-visible change σ={sg[-1]:.1e}", fontsize=7)
        for a in axes[r]:
            a.contour(xs, xs, lab.numpy(), levels=levels, colors="k", linewidths=0.6)
            a.set_xticks([]), a.set_yticks([])
    fig.suptitle("Black = true outlines of the hidden object. Colour = the image change "
                 "each design is LEAST able to see.", fontsize=10)
    fig.tight_layout()
    fig.savefig(RESULTS / "figures" / "phase3_complexity_gallery.png", dpi=110)
    print("figures: phase3_complexity_sigma_min.png, phase3_complexity_gallery.png")
    print("CHECK:", "PASS" if fd_err < 1e-5 else "FAIL", f"total {time.time()-t0:.0f}s")


if __name__ == "__main__":
    main()
