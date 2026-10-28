"""Phase 3 step 1 -- blind region formation from scanner data (visual check).

Hidden objects: random sharp-cornered polygon regions (phase3_complexity_sweep
class). The pipeline sees ONLY the data (object spectrum on the design's
measured set, linearized model, noise-free):
  1. direct image  = inverse FFT of the measured spectrum (no starting guess)
  2. smoothing     = median filter, to suppress ringing while keeping edges
  3. edges         = pixels where the image jumps (gradient > EDGE_FRAC of max)
  4. regions       = connected areas enclosed by edges (count found blind);
                     edge pixels and tiny fragments go to the nearest region
No labels, no region count, no property values are given to the pipeline.
Output is for visual inspection only: green = found boundaries,
black = true outlines.
"""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
from scipy import ndimage

from phase3_adversarial_uniqueness import L_MM, N, design_masks, x
from phase3_complexity_sweep import hidden_polys, render_polys
from seed import SEED, set_all_seeds

RESULTS = Path(__file__).resolve().parent.parent / "results"
OBJECTS = [(3, 8), (5, 12), (8, 12)]       # (regions, vertices)
DESIGN_KEYS = ["all pairs", "transmission", "pasted design) [band]", "anterior"]
EDGE_FRAC = 0.08   # a jump counts as an edge above 8% of the strongest jump
MIN_PIX = 40       # regions smaller than this are fragments


def direct_image(img, mask):
    F = torch.fft.fft2(img)
    return torch.fft.ifft2(F * mask).real.numpy()


def blind_regions(rec):
    sm = ndimage.median_filter(ndimage.median_filter(rec, 5), 5)
    # Boundaries = where the image jumps. (Value-histogram clustering was tried
    # first and failed: water swamped the region peaks, then blurred edge
    # pixels made spurious peaks -- LOG run -07.)
    grad = np.hypot(ndimage.sobel(sm, 0), ndimage.sobel(sm, 1))
    edge = grad > EDGE_FRAC * grad.max()
    comp, k = ndimage.label(~edge)
    sizes = ndimage.sum(np.ones_like(comp), comp, range(1, k + 1))
    keep = np.flatnonzero(sizes >= MIN_PIX) + 1
    region = np.where(np.isin(comp, keep), comp, 0)
    # hand edge pixels and dropped fragments to the nearest kept region
    _, idx = ndimage.distance_transform_edt(region == 0, return_indices=True)
    region = region[idx[0], idx[1]]
    boundary = (np.diff(region, axis=0, prepend=region[:1]) != 0) | \
               (np.diff(region, axis=1, prepend=region[:, :1]) != 0)
    return sm, boundary, len(keep)


def main():
    rng = set_all_seeds(SEED)
    (RESULTS / "figures").mkdir(parents=True, exist_ok=True)
    masks = design_masks()
    names = [next(k for k in masks if key in k) for key in DESIGN_KEYS]
    ext = [-L_MM / 2, L_MM / 2, -L_MM / 2, L_MM / 2]
    xs = x.numpy()

    fig, axes = plt.subplots(len(OBJECTS), 1 + 2 * len(names),
                             figsize=(2.6 * (1 + 2 * len(names)), 2.9 * len(OBJECTS)))
    for r, (n, V) in enumerate(OBJECTS):
        flat, concave = hidden_polys(rng, n, V)
        img, lab = render_polys(flat, n, V, return_labels=True)
        true = img.numpy()
        vm = np.abs(true).max()
        axes[r, 0].imshow(true, origin="lower", extent=ext, cmap="RdBu_r", vmin=-vm, vmax=vm)
        axes[r, 0].set_title(f"HIDDEN: {n} regions, {V}-gons\n{concave} concave corners", fontsize=8)
        print(f"object {r}: {n} regions x {V} vertices")
        for c, nm in enumerate(names):
            rec = direct_image(img, masks[nm])
            sm, boundary, n_reg = blind_regions(rec)
            short = nm.split("[")[0].replace("(pasted design)", "(pasted)").strip()
            a = axes[r, 1 + 2 * c]
            a.imshow(rec, origin="lower", extent=ext, cmap="RdBu_r", vmin=-vm, vmax=vm)
            a.set_title(f"{short}\n1. direct image", fontsize=7)
            a = axes[r, 2 + 2 * c]
            a.imshow(sm, origin="lower", extent=ext, cmap="RdBu_r", vmin=-vm, vmax=vm)
            ov = np.zeros((N, N, 4))
            ov[boundary] = [0, 0.85, 0, 1]
            a.imshow(ov, origin="lower", extent=ext, interpolation="nearest")
            a.set_title(f"2. regions found: {n_reg}\n(incl. water)", fontsize=7)
            print(f"   {short:40s} regions found {n_reg:2d} (incl. water)")
        for a in axes[r]:
            a.contour(xs, xs, lab.numpy(), levels=np.arange(n + 1) + 0.5,
                      colors="k", linewidths=0.6)
            a.set_xticks([]), a.set_yticks([])
    fig.suptitle("Blind region formation from scanner data only. "
                 "Black = TRUE outlines (hidden from the pipeline); green = boundaries it found.",
                 fontsize=10)
    fig.tight_layout()
    out = RESULTS / "figures" / "phase3_blind_region_formation.png"
    fig.savefig(out, dpi=110)
    print(f"figure: {out}")


if __name__ == "__main__":
    main()
