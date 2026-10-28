"""Intricate heart phantom traced from a user-supplied ex-vivo cardiac MRI
(long-axis, four-chamber-like view, fine trabeculae and valve/aortic-root walls).

What is traced vs assumed:
  - TRACED: tissue = bright pixels (MRI intensity > TISSUE_THR, small specks
    removed); chambers = dark regions ENCLOSED by tissue (after a morphological
    closing of radius CLOSE_PX that bridges the gaps between wall fragments).
    Everything else (dark regions open to the outside, the black corners outside
    the MRI field of view, the noisy band) -> water bath.
  - Intensity is NOT used as a property map: MRI signal is not Hounsfield units,
    and no cited calibration maps it to density or sound speed. Tissue gets the
    cited myocardium values, chambers the cited blood values (tissue.py; Mast 2000,
    Duck 1990 for water). Valve/vessel walls are lumped with myocardium.
  - ASSUMED scale: the tissue bounding box's larger side = HEART_BOX_MM (no ruler).

  - DIM tissue (MRI intensity DIM_THR..TISSUE_THR, at least 2 px thick, touching the
    bright tissue): the less-bright outer lining (pericardium / epicardial fat) and
    dim wall segments. Modelled with cited FAT values (tissue.py, FAT_PERICARDIUM_PROXY):
    the tissue identity is an ASSUMPTION (MRI sequence unknown).
  - Irregularities: the traced anatomy itself (trabeculae, papillary muscles, ragged
    endocardium). Optional myocardial speed TEXTURE is added later, at solve time
    (see the LOG), not stored here.

Labels: 0 water, 1 myocardium (bright tissue), 2 blood (enclosed chambers),
3 dim tissue (pericardium / epicardial-fat proxy).
Outputs are gitignored (derived from a user-supplied medical image).
"""

import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
from scipy import ndimage

OUT = Path(__file__).resolve().parent.parent / "results"
TISSUE_THR = 95          # tissue ~100-250, chambers/background ~40-55 (histogram)
DIM_THR = 66             # less-bright lining/connective tissue, above background noise
MIN_SPECK_PX = 300       # drop isolated bright specks (noise band, debris)
CLOSE_PX = 12            # gap-bridging radius for deciding what is "enclosed"
THIN_MIN_PX = 60         # thin faint wall segments kept if at least this long
HEART_BOX_MM = 110.0     # ASSUMPTION: larger side of the tissue bounding box
DX_MM = 0.25             # phantom pixel (fine grid for the time-domain data)


def disk(r):
    y, x = np.ogrid[-r:r + 1, -r:r + 1]
    return x * x + y * y <= r * r


def trace(img):
    img = ndimage.median_filter(img, size=3)                    # speckle noise in the background
    tissue = img > TISSUE_THR
    tissue = ndimage.binary_closing(tissue, structure=disk(1))
    lab, n = ndimage.label(tissue)
    sizes = ndimage.sum(tissue, lab, range(1, n + 1))
    tissue = np.isin(lab, [i + 1 for i, s in enumerate(sizes) if s >= MIN_SPECK_PX])
    dim = (img > DIM_THR) & ~tissue
    dim = ndimage.binary_opening(dim, structure=disk(1))        # >= ~2 px thick: not the 1-px partial-volume rim
    lab, n = ndimage.label(dim)
    touch = np.unique(lab[ndimage.binary_dilation(tissue, iterations=2) & dim])
    sizes = ndimage.sum(dim, lab, range(1, n + 1))
    dim = np.isin(lab, [i for i in touch if i > 0 and sizes[i - 1] >= 40])
    # thin faint walls (1-2 px, e.g. the atrial roof) that the opening above removes: long
    # components above DIM_THR away from the bright tissue's 1-px partial-volume rim -> myocardium
    thin = (img > DIM_THR) & ~ndimage.binary_dilation(tissue, iterations=1) & ~dim
    lab, n = ndimage.label(thin, structure=np.ones((3, 3)))
    sizes = ndimage.sum(thin, lab, range(1, n + 1))
    tissue = tissue | np.isin(lab, [i + 1 for i, s in enumerate(sizes) if s >= THIN_MIN_PX])
    wall = tissue | dim
    pad = CLOSE_PX + 2                                          # closing must not touch the image edge
    closed = ndimage.binary_closing(np.pad(wall, pad), structure=disk(CLOSE_PX))[pad:-pad, pad:-pad]
    enclosed = ndimage.binary_fill_holes(closed)
    blood = enclosed & ~wall
    blood = ndimage.binary_opening(blood, structure=disk(1))   # drop 1-px slivers between trabeculae
    lab = np.zeros(img.shape, np.uint8)
    lab[tissue] = 1
    lab[blood] = 2
    lab[dim] = 3
    return lab


def to_physical(lab):
    rr, cc = np.nonzero(lab)
    crop = lab[rr.min():rr.max() + 1, cc.min():cc.max() + 1]
    px_mm = HEART_BOX_MM / max(crop.shape)
    return ndimage.zoom(crop, px_mm / DX_MM, order=0), px_mm  # nearest-neighbour (labels)


def main(img_path):
    img = np.asarray(Image.open(img_path).convert("L")).astype(float)
    lab = trace(img)
    phys, px_mm = to_physical(lab)
    rr, cc = np.nonzero(phys)
    cy, cx = (np.array(phys.shape) - 1) / 2
    r_max = np.sqrt((rr - cy) ** 2 + (cc - cx) ** 2).max() * DX_MM
    print(f"image {img.shape}, source pixel {px_mm:.3f} mm (assumed {HEART_BOX_MM} mm box), "
          f"phantom {phys.shape} at {DX_MM} mm, max radius from box centre {r_max:.1f} mm")
    print("label fractions (myocardium, blood, dim):", [(phys == k).mean().round(3) for k in (1, 2, 3)])
    np.savez_compressed(OUT / "phase3_mri_heart_phantom_patient.npz", labels=phys, dx_mm=DX_MM,
                        assumed_box_mm=HEART_BOX_MM, r_max_mm=r_max)

    fig, ax = plt.subplots(1, 3, figsize=(18, 6.5))
    ax[0].imshow(img, cmap="gray"); ax[0].set_title("MRI (user-supplied)")
    ax[1].imshow(img, cmap="gray")
    ax[1].contour(lab == 1, [0.5], colors="yellow", linewidths=0.6)
    ax[1].contour(lab == 2, [0.5], colors="red", linewidths=0.8)
    ax[1].contour(lab == 3, [0.5], colors="cyan", linewidths=0.6)
    ax[1].set_title("traced: myocardium (yellow), blood (red), dim lining (cyan)")
    ax[2].imshow(phys, cmap="viridis", interpolation="nearest")
    ax[2].set_title(f"labels 0 water / 1 myocardium / 2 blood / 3 dim (fat proxy)\n{phys.shape[0]}x{phys.shape[1]} at {DX_MM} mm")
    for a in ax:
        a.axis("off")
    plt.tight_layout()
    fig.savefig(OUT / "figures" / "phase3_mri_heart_phantom_patient.png", dpi=110)


if __name__ == "__main__":
    main(sys.argv[1])
