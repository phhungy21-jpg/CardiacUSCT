"""Two-thigh phantom traced from a user-supplied T1 axial MRI of one thigh (thigh_ax_22.jpg), mirrored to make the
contralateral thigh, both placed in the water-bath ring with a water gap to the transducers.

T1 appearance used for tracing (intensity, 0-255): fat (subcutaneous, intermuscular, fatty marrow) saturated ~255;
muscle ~140; cortical bone, flow voids and fascia < 60; slowly-flowing vessels / nerves ~160-180 inside fat.
What is TRACED vs ASSUMED:
  - TRACED: body outline; subcutaneous fat (bright, connected to the surface layer); intermuscular fat (other bright
    regions); muscle (intermediate); femoral cortex (dark ring around the marrow, incl. linea aspera); fatty marrow
    (bright, enclosed by cortex); superficial vessels (compact dark blobs in the subcutaneous fat); deep neurovascular
    structures (compact dark blobs between muscles + compact intermediate blobs embedded in fat, e.g. the femoral and
    profunda bundles).
  - ASSUMED: skin = outer SKIN_MM of the body (not separable from fat on T1); fascia lines (thin dark) -> muscle;
    deep intermediate blobs in fat may be nerves (e.g. sciatic) as well as vessels -- all modelled as BLOOD;
    scale: thigh medial-lateral width = THIGH_WIDTH_MM (no ruler in the image); the second thigh is the mirror image
    (patients are not symmetric).
Labels: 0 water, 1 skin, 2 subcutaneous fat, 3 muscle, 4 intermuscular fat, 5 cortical bone, 6 marrow,
        7 superficial vessel, 8 deep vessel / neurovascular.
Outputs (gitignored, derived from a user-supplied medical image): results/phase3_thigh_phantom_patient.npz,
results/figures/phase3_thigh_phantom_patient.png
"""

import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
from scipy import ndimage

OUT = Path(__file__).resolve().parent.parent / "results"
THIGH_WIDTH_MM = 150.0     # ASSUMPTION: medial-lateral width of the traced thigh
SKIN_MM = 1.5              # ASSUMPTION: skin thickness (dermis+epidermis) on the thigh
GAP_MM = 12.0              # water between the two thighs (medial surfaces)
CLEAR_MM = 20.0            # minimum water clearance between tissue and the transducer ring
DX_MM = 0.25               # output grid (time-domain data grid, as the MRI heart)
FAT_T, DARK_T = 200, 60    # intensity thresholds (histogram: background 0, muscle ~140, fat 255)
NAMES = ["water", "skin", "subcut fat", "muscle", "intermusc fat", "cortical bone", "marrow", "superficial vessel", "deep vessel/NV"]


def disk(r):
    y, x = np.ogrid[-r:r + 1, -r:r + 1]
    return x * x + y * y <= r * r


def compact_blobs(mask, amin, amax, max_len, min_fill):
    """components of `mask` that are small and compact (not thin fascia lines)."""
    lab, n = ndimage.label(mask)
    keep = np.zeros_like(mask)
    for i, sl in enumerate(ndimage.find_objects(lab), 1):
        comp = lab[sl] == i; a = comp.sum(); h, w = comp.shape
        if amin <= a <= amax and max(h, w) <= max_len and a / (h * w) >= min_fill:
            keep[sl] |= comp
    return keep


def trace(img, px_mm):
    im = ndimage.median_filter(img, size=3).astype(int)
    lab, _ = ndimage.label(im > 30)
    body = ndimage.binary_fill_holes(ndimage.binary_closing(lab == lab[250, 250], structure=disk(2)))
    skin_px = max(1, int(round(SKIN_MM / px_mm)))
    skin = body & ~ndimage.binary_erosion(body, structure=disk(1), iterations=skin_px)
    inner = body & ~skin
    bright, dark = (im > FAT_T) & inner, (im < DARK_T) & inner
    # femur: marrow = bright component at the femoral centre; cortex = dark pixels connected to the marrow's rim
    blab, _ = ndimage.label(bright)
    marrow = ndimage.binary_fill_holes(blab == blab[238, 217])
    dlab, _ = ndimage.label(dark | ndimage.binary_dilation(marrow, iterations=2) & dark)
    ring = ndimage.binary_dilation(marrow, iterations=3) & dark
    ids = np.unique(dlab[ring]); cortex = np.isin(dlab, ids[ids > 0])
    bone = ndimage.binary_fill_holes(ndimage.binary_closing(cortex | marrow, structure=disk(2)))
    cortex = bone & ~marrow
    # subcutaneous fat = bright components touching the skin layer; the rest of the bright = intermuscular fat
    touch = ndimage.binary_dilation(skin, iterations=2)
    ids = np.unique(blab[touch & bright]); subq = np.isin(blab, ids[ids > 0]) & bright & ~bone
    imf = bright & ~subq & ~bone
    # vessels: compact dark blobs (flow voids) and compact intermediate blobs fully embedded in fat
    vdark = compact_blobs(dark & ~bone, 3, 250, 22, 0.35)
    mid = (im >= DARK_T) & (im <= FAT_T) & inner & ~bone
    mlab, nm = ndimage.label(mid)
    embedded = np.zeros_like(mid)
    for i, sl in enumerate(ndimage.find_objects(mlab), 1):
        comp = np.zeros_like(mid); comp[sl] = mlab[sl] == i
        a = comp.sum()
        if 15 <= a <= 600:
            rim = ndimage.binary_dilation(comp, iterations=2) & ~comp
            if (bright[rim]).mean() > 0.75:
                embedded |= comp
    vessels = ndimage.binary_dilation(vdark | embedded, iterations=1) & inner & ~bone
    sup_v = vessels & ndimage.binary_dilation(subq, iterations=3)
    deep_v = vessels & ~sup_v
    out = np.zeros(img.shape, np.uint8)
    out[body] = 3                                # muscle (incl. fascia lines) by default inside the body
    out[imf] = 4; out[subq] = 2
    out[deep_v] = 8; out[sup_v] = 7
    out[cortex] = 5; out[marrow] = 6
    out[skin] = 1
    return out


def main(img_path):
    img = np.asarray(Image.open(img_path).convert("L")).astype(float)
    # thigh extent at native resolution -> pixel size from the assumed width
    rows, cols = np.nonzero(ndimage.binary_fill_holes((lambda l: l == l[250, 250])(ndimage.label(img > 30)[0])))
    px_mm = THIGH_WIDTH_MM / (cols.max() - cols.min() + 1)
    lab = trace(img, px_mm)
    rr, cc = np.nonzero(lab); lab = lab[rr.min():rr.max() + 1, cc.min():cc.max() + 1]
    one = ndimage.zoom(lab, px_mm / DX_MM, order=0)                    # nearest neighbour (labels)
    two = one[:, ::-1]                                                 # mirror = contralateral thigh
    h, w = one.shape; gap = int(round(GAP_MM / DX_MM))
    pair = np.zeros((h, 2 * w + gap), np.uint8); pair[:, :w] = one; pair[:, w + gap:] = two
    # The image's right side is MEDIAL (femoral vessels), so the traced thigh is the patient's RIGHT thigh shown on the
    # image left; mirrored copy = left thigh, placed so the medial sides face each other across the gap.
    yy, xx = np.nonzero(pair); cy, cx = (yy.min() + yy.max()) / 2, (xx.min() + xx.max()) / 2
    r_tissue = np.sqrt(((yy - cy) ** 2 + (xx - cx) ** 2).max()) * DX_MM
    ring_r = r_tissue + CLEAR_MM
    L = 2 * (ring_r + 12.0)                                            # 12 mm beyond the ring for the absorbing layer
    n = int(np.ceil(L / DX_MM / 8) * 8)
    grid = np.zeros((n, n), np.uint8)
    o0, o1 = int(round(n / 2 - cy)), int(round(n / 2 - cx))
    grid[o0:o0 + pair.shape[0], o1:o1 + pair.shape[1]] = pair
    np.savez_compressed(OUT / "phase3_thigh_phantom_patient.npz", labels=grid, dx_mm=DX_MM, ring_r_mm=ring_r,
                        domain_mm=n * DX_MM, px_mm_source=px_mm, names=np.array(NAMES))
    print(f"source pixel {px_mm:.3f} mm (assumed width {THIGH_WIDTH_MM} mm); one thigh {one.shape[1]*DX_MM:.0f} x {one.shape[0]*DX_MM:.0f} mm")
    print(f"tissue radius {r_tissue:.1f} mm -> ring radius {ring_r:.1f} mm (clearance {CLEAR_MM} mm); domain {n} px = {n*DX_MM:.0f} mm")
    for k, nm in enumerate(NAMES):
        print(f"  {k} {nm:20s} {100*np.mean(grid == k):6.2f}% of grid, {np.sum(grid == k)*DX_MM**2/100:8.1f} cm^2")
    # figure: source with traced outlines, label map, tub layout
    fig, ax = plt.subplots(1, 3, figsize=(21, 7))
    src_lab = trace(img, px_mm)
    ax[0].imshow(img, cmap="gray")
    cols_ = ["y", "orange", "lime", "cyan", "r", "m", "deepskyblue", "blue"]
    for k, col in zip(range(1, 9), cols_):
        ax[0].contour(src_lab == k, [0.5], colors=col, linewidths=0.6)
    ax[0].set_title("traced on source (contours: " + ", ".join(f"{k}={NAMES[k]}" for k in range(1, 9)) + ")", fontsize=7)
    cmap = plt.get_cmap("tab10", 9)
    im1 = ax[1].imshow(src_lab, cmap=cmap, vmin=-0.5, vmax=8.5); ax[1].set_title("label map (one thigh)")
    cb = plt.colorbar(im1, ax=ax[1], ticks=range(9), fraction=0.046); cb.ax.set_yticklabels(NAMES, fontsize=7)
    ext = [-n * DX_MM / 2, n * DX_MM / 2, n * DX_MM / 2, -n * DX_MM / 2]
    ax[2].imshow(grid, cmap=cmap, vmin=-0.5, vmax=8.5, extent=ext)
    ax[2].add_patch(plt.Circle((0, 0), ring_r, fill=False, color="k", lw=1.5))
    ax[2].set_title(f"two thighs in the tub: ring r = {ring_r:.0f} mm, {CLEAR_MM:.0f} mm clearance, gap {GAP_MM:.0f} mm")
    for a in ax[:2]: a.axis("off")
    fig.tight_layout(); fig.savefig(OUT / "figures" / "phase3_thigh_phantom_patient.png", dpi=110)
    print("figure saved")


if __name__ == "__main__":
    main(sys.argv[1])
