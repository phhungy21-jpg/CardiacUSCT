"""Chest phantom for the lung/rib test: traced body + lungs from a published
axial chest CT (lung window, carina level), with the patient023 ventricle slice
inserted in the mediastinum and SCHEMATIC ribs/vertebra.

Source image: user-supplied AJR figure (74-year-old woman, COVID-19), lung
window. Treated as patient data: derived arrays are gitignored (*patient*).

What is traced vs assumed:
  - TRACED: body outline (intensity > background, holes filled, CT table line
    removed) and lungs (dark regions inside the body; airways included and
    lumped with lung as one homogeneous aerated region, per user request).
  - INSERTED: patient023 ED mid-ventricular slice (LV/RV blood, myocardium) at
    true size, centred in the mediastinum. This slice level is ABOVE the
    ventricles anatomically; the heart is a composite placement.
  - SCHEMATIC (not traced; the lung window does not separate bone): rib
    cross-sections on an ellipse inside the chest wall, and a vertebral body
    at the posterior midline.
  - ASSUMED scale: body width 320 mm (no ruler in the image).

Labels: 0 water bath, 1 soft tissue (chest wall + mediastinum), 2 lung (aerated),
3 bone (schematic), 4 myocardium, 5 blood (LV + RV cavities).
Acoustic values for lung and bone are NOT assigned here: they need cited sources.
"""

import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
from scipy import ndimage

ROOT = Path(__file__).resolve().parents[2]
OUT = Path(__file__).resolve().parent.parent / "results"
BODY_WIDTH_MM = 320.0          # ASSUMPTION
DX_MM = 0.5                    # phantom pixel
DOMAIN_MM = 400.0              # water margin around the body


def trace(img):
    # threshold on the bright chest-wall ring, then fill: the lungs are holes inside it
    # (first version used > 25, below the outside-air gray ~35, and took the whole frame)
    body = img > 120
    body[img.shape[0] - 25:, :] = False                        # CT table line at the bottom
    body = ndimage.binary_opening(body, iterations=2)
    lab, n = ndimage.label(body)
    body = lab == (np.argmax(ndimage.sum(body, lab, range(1, n + 1))) + 1)
    body = ndimage.binary_fill_holes(body)
    dark = (img < 140) & ndimage.binary_erosion(body, iterations=8)   # lungs/airways, away from the skin edge
    dark = ndimage.binary_opening(dark, iterations=1)
    lab, n = ndimage.label(dark)
    sizes = ndimage.sum(dark, lab, range(1, n + 1))
    keep = [i + 1 for i, s in enumerate(sizes) if s > 300]     # the two lungs + main airways
    # vessels and the COVID opacities inside/at the edge of the lungs -> lung (homogeneous, per user)
    lung = ndimage.binary_closing(np.isin(lab, keep), iterations=12)
    lung = ndimage.binary_fill_holes(lung) & body
    return body, lung


def main(img_path):
    img = np.asarray(Image.open(img_path).convert("L")).astype(float)
    body, lung = trace(img)
    cols = np.nonzero(body.any(axis=0))[0]
    px_mm = BODY_WIDTH_MM / (cols.max() - cols.min() + 1)
    print(f"image {img.shape}, body width {cols.max()-cols.min()+1} px -> {px_mm:.3f} mm/px (assumed {BODY_WIDTH_MM} mm)")

    zoom = px_mm / DX_MM                                       # nearest-neighbour onto the phantom grid
    body_g = ndimage.zoom(body.astype(np.uint8), zoom, order=0).astype(bool)
    lung_g = ndimage.zoom(lung.astype(np.uint8), zoom, order=0).astype(bool)
    n = int(DOMAIN_MM / DX_MM)
    lab = np.zeros((n, n), np.uint8)
    r0, c0 = (n - body_g.shape[0]) // 2, (n - body_g.shape[1]) // 2
    sl = np.s_[r0:r0 + body_g.shape[0], c0:c0 + body_g.shape[1]]
    lab[sl][body_g] = 1
    lab[sl][lung_g & body_g] = 2

    # schematic bone: ribs on an ellipse inside the chest wall + vertebral body (posterior midline)
    rr, cc = np.nonzero(lab > 0)
    cy, cx = rr.mean(), cc.mean()
    ay, ax = (rr.max() - rr.min()) / 2, (cc.max() - cc.min()) / 2
    Y, X = np.mgrid[0:n, 0:n]
    for ang in np.deg2rad(np.r_[np.arange(15, 166, 25), np.arange(200, 341, 28)]):
        ry, rx = cy - 0.86 * ay * np.sin(ang), cx + 0.86 * ax * np.cos(ang)
        rib = ((Y - ry) / (6 / DX_MM)) ** 2 + ((X - rx) / (10 / DX_MM)) ** 2 < 1
        lab[rib & (lab == 1)] = 3
    vert = ((Y - (cy + 0.62 * ay)) / (17 / DX_MM)) ** 2 + ((X - cx) / (18 / DX_MM)) ** 2 < 1
    lab[vert & (lab == 1)] = 3

    # insert patient023 ventricles (true size) in the mediastinum: soft tissue between the lungs
    p = np.load(OUT / "phase3_patient023_slice_labels.npz")["labels"]      # scaled x0.386 @0.25 mm
    rows, cols2 = np.nonzero(p)
    heart = ndimage.zoom(p[rows.min():rows.max() + 1, cols2.min():cols2.max() + 1],
                         (0.25 / 0.386) / DX_MM, order=0)                   # back to true size @ DX_MM
    soft = lab == 1
    dist = ndimage.distance_transform_edt(soft)
    mid = np.zeros_like(soft); mid[:, n // 2 - int(60 / DX_MM): n // 2 + int(60 / DX_MM)] = True
    hy, hx = np.unravel_index(np.argmax(dist * mid), dist.shape)           # deepest mediastinal point
    h0, w0 = heart.shape
    tgt = np.s_[hy - h0 // 2: hy - h0 // 2 + h0, hx - w0 // 2: hx - w0 // 2 + w0]
    region = lab[tgt]
    region[heart == 2] = 4
    region[(heart == 1) | (heart == 3)] = 5
    over_lung = int(((heart > 0) & (lab[tgt] == 2)).sum())
    print(f"heart {h0*DX_MM:.0f} x {w0*DX_MM:.0f} mm placed at ({hy*DX_MM:.0f}, {hx*DX_MM:.0f}) mm")

    names = ["water", "soft tissue", "lung", "bone (schematic)", "myocardium", "blood"]
    for k, nm in enumerate(names):
        print(f"  {k} {nm:18s} {(lab == k).sum() * DX_MM**2 / 100:8.1f} cm^2")
    OUT.mkdir(exist_ok=True)
    np.savez_compressed(OUT / "phase3_ct_chest_phantom_patient.npz", labels=lab, dx_mm=DX_MM,
                        assumed_body_width_mm=BODY_WIDTH_MM)

    fig, ax = plt.subplots(1, 2, figsize=(12, 5.2))
    ax[0].imshow(img, cmap="gray"); ax[0].contour(lung, [0.5], colors="c", linewidths=0.8)
    ax[0].contour(body, [0.5], colors="y", linewidths=0.8)
    ax[0].set_title("source CT (lung window): traced body (yellow), lungs (cyan)", fontsize=9)
    cmap = plt.matplotlib.colors.ListedColormap(["#9ecae1", "#fdd0a2", "#f7f7f7", "#636363", "#e6550d", "#a50f15"])
    ext = [0, n * DX_MM, n * DX_MM, 0]
    ax[1].imshow(lab, cmap=cmap, vmin=-0.5, vmax=5.5, extent=ext, interpolation="nearest")
    ax[1].set_title("phantom: water / soft tissue / lung / bone (schematic) /\nmyocardium / blood (patient023 heart inserted)", fontsize=9)
    ax[1].set_xlabel("mm"); ax[1].set_ylabel("mm")
    plt.tight_layout()
    out = OUT / "figures" / "phase3_ct_chest_phantom_patient.png"
    fig.savefig(out, dpi=110)
    print("figure:", out)


if __name__ == "__main__":
    main(sys.argv[1])
