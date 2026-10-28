"""Full clinical metric set for a 2D speed reconstruction of the MRI heart (re-implementation of LOG run -36,
which was computed ad hoc and not saved; validated by reproducing run -36's CEILING numbers on the true map).

Segmentation by threshold (run -36 rules), scored against the TRUE labels (0 water / 1 myo / 2 blood / 3 fat proxy):
  heart = largest component of c > 1548, opened and hole-filled
  blood = Gaussian(2 px) > 1580 inside the heart;  myocardium = heart - blood
  fat   = Gaussian(1 px) < 1500 within 6 px outside the heart
"Ceiling" = the true speed map through the same segmentation (segmentation-method limit, no reconstruction error).
Overlap/surface: DSC, IoU, ASSD, HD95, HD (mm), NSD at 1 and 2 mm, area error %.
Speed error (m/s): RMSE / MAE / bias / P95|err| per region; PSNR (heart, range 125 m/s); SSIM (heart crop);
texture r (myocardial core, mean removed).
Pure numpy/scipy (+ skimage for SSIM if available); CPU, seconds.

Unrecorded run -36 details, fixed here by matching its CEILING numbers (LOG run -42): opening = ONE iteration with an
8-connected 3x3 element; true heart = largest connected myo+blood component, holes filled; HD95 = 95th percentile of
the pooled (both-direction) surface distances. Residual vs run -36 ceiling: heart DSC 0.995 vs 0.996, blood exact,
myo 0.791 vs 0.792, fat 0.521 vs 0.492 (threshold fat segmentation is unreliable, as run -36 noted).
"""

import numpy as np
from scipy import ndimage

OPEN_ITERS = 1       # with an 8-connected element: best match to run -36's ceiling
CORE_ERODE = 3       # "thick myocardial core": erosion depth not recorded -- chosen here


def segment(c):
    m = c > 1548
    lab, n = ndimage.label(m)
    if n == 0:
        z = np.zeros_like(m)
        return dict(heart=z, blood=z, myo=z, fat=z)
    heart = lab == (np.argmax(ndimage.sum(m, lab, range(1, n + 1))) + 1)
    heart = ndimage.binary_fill_holes(ndimage.binary_opening(heart, structure=np.ones((3, 3), bool), iterations=OPEN_ITERS))
    blood = (ndimage.gaussian_filter(c, 2) > 1580) & heart
    fat = (ndimage.gaussian_filter(c, 1) < 1500) & ndimage.binary_dilation(heart, iterations=6) & ~heart
    return dict(heart=heart, blood=blood, myo=heart & ~blood, fat=fat)


def _surface(m):
    return m & ~ndimage.binary_erosion(m)


def overlap_surface(pred, true, dx):
    inter = np.logical_and(pred, true).sum(); ps, ts = pred.sum(), true.sum()
    out = dict(DSC=2 * inter / (ps + ts) if ps + ts else np.nan, IoU=inter / np.logical_or(pred, true).sum() if ps + ts else np.nan,
               area_pct=100 * (ps - ts) / ts if ts else np.nan)
    sp, st = _surface(pred), _surface(true)
    if sp.sum() == 0 or st.sum() == 0:
        return dict(out, ASSD=np.nan, HD95=np.nan, HD=np.nan, NSD1=np.nan, NSD2=np.nan)
    d_to_t = ndimage.distance_transform_edt(~st) * dx
    d_to_p = ndimage.distance_transform_edt(~sp) * dx
    a, b = d_to_t[sp], d_to_p[st]
    both = np.concatenate([a, b])
    out.update(ASSD=both.mean(), HD95=np.percentile(both, 95), HD=both.max(),
               NSD1=np.mean(both <= 1.0), NSD2=np.mean(both <= 2.0))
    return out


def speed_errors(c, ctrue, lab):
    heart = ndimage.binary_dilation(lab > 0, iterations=2)
    band = ndimage.binary_dilation(lab > 0, iterations=2) & ~ndimage.binary_erosion(lab > 0, iterations=2)
    regions = dict(heart=heart, myo=lab == 1, blood=lab == 2, fat=lab == 3,
                   water=ndimage.binary_erosion(lab == 0, iterations=1),
                   myo_core=ndimage.binary_erosion(lab == 1, iterations=CORE_ERODE), band=band)
    out = {}
    for k, m in regions.items():
        e = (c - ctrue)[m]
        out[k] = dict(RMSE=np.sqrt(np.mean(e ** 2)), MAE=np.mean(np.abs(e)), bias=np.mean(e), P95=np.percentile(np.abs(e), 95))
    mse = np.mean((c - ctrue)[heart] ** 2)
    out["PSNR_heart"] = 10 * np.log10(125.0 ** 2 / mse)
    rr, cc = np.nonzero(heart); sl = (slice(rr.min(), rr.max() + 1), slice(cc.min(), cc.max() + 1))
    try:
        from skimage.metrics import structural_similarity
        out["SSIM_heart"] = structural_similarity(ctrue[sl].astype(float), c[sl].astype(float), data_range=125.0)
    except ImportError:
        out["SSIM_heart"] = np.nan
    core = regions["myo_core"]
    out["texture_r"] = np.corrcoef(c[core] - c[core].mean(), ctrue[core] - ctrue[core].mean())[0, 1]
    return out


def full_metrics(c, lab, ctrue, dx_mm=0.5):
    seg = segment(c)
    lc, n = ndimage.label((lab == 1) | (lab == 2))
    heart_true = ndimage.binary_fill_holes(lc == (np.argmax(ndimage.sum(lc > 0, lc, range(1, n + 1))) + 1))
    truth = dict(heart=heart_true, blood=lab == 2, myo=lab == 1, fat=lab == 3)
    return dict(seg={k: overlap_surface(seg[k], truth[k], dx_mm) for k in truth}, speed=speed_errors(c, ctrue, lab))


def table(results):
    """results: {name: full_metrics(...)} -> printable text."""
    lines = []
    for region in ["heart", "blood", "myo", "fat"]:
        lines.append(f"\n[{region}] DSC / IoU / ASSD mm / HD95 mm / HD mm / NSD@1 / NSD@2 / area %")
        for name, r in results.items():
            s = r["seg"][region]
            lines.append(f"  {name:28s} {s['DSC']:.3f} / {s['IoU']:.3f} / {s['ASSD']:.2f} / {s['HD95']:.2f} / {s['HD']:.1f}"
                         f" / {s['NSD1']:.3f} / {s['NSD2']:.3f} / {s['area_pct']:+.1f}")
    for region in ["heart", "myo", "blood", "fat", "water", "myo_core", "band"]:
        lines.append(f"\n[speed {region}] RMSE / MAE / bias / P95 (m/s)")
        for name, r in results.items():
            if name.startswith("CEILING"):
                continue
            e = r["speed"][region]
            lines.append(f"  {name:28s} {e['RMSE']:.2f} / {e['MAE']:.2f} / {e['bias']:+.2f} / {e['P95']:.1f}")
    lines.append("\n[image] PSNR heart dB / SSIM heart / texture r")
    for name, r in results.items():
        if name.startswith("CEILING"):
            continue
        sp = r["speed"]
        lines.append(f"  {name:28s} {sp['PSNR_heart']:.1f} / {sp['SSIM_heart']:.3f} / {sp['texture_r']:.3f}")
    return "\n".join(lines)
