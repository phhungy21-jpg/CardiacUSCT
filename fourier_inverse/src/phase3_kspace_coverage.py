"""Phase 3.1 / 3.2 / 3.2b -- Fourier diffraction theorem: k-space coverage,
inner- vs outer-wall information content, and density/compressibility split.

Linearized (Born / Fourier diffraction) model about a homogeneous water
background, plane-wave incidence, far-field receivers (an idealization; see
LOG run -02/-03 for validity -- run -02 showed this holds for inner-wall
perturbations but NOT for the whole heart, so read these maps as "the
Frechet derivative around a known background", not as a whole-object
inversion).

A transmit direction s0 (angle alpha), receive direction s (angle beta) and
wavenumber k measure the object spectrum at K = k (s - s0), with weight
    k^2 [gamma_hat(K) - cos(theta) beta_hat(K)],  theta = beta - alpha,
gamma = compressibility contrast, beta = rho0/rho - 1 (derivation: LOG run -02).

Inverse map used to rasterize coverage: for |K| <= 2k, with psi = angle(K)
and t = 2 arcsin(|K| / 2k), the two (alpha, beta) solutions are
    alpha = psi -/+ pi/2 -/+ t/2,  beta = psi -/+ pi/2 +/- t/2.
Checked numerically below (check (a)).

Position conventions: anterior = +y = 90 deg. A source at position angle p
launches alpha = p + pi; a receiver at position angle q records beta = q.
"""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from seed import SEED, set_all_seeds
from tissue import BLOOD, MYOCARDIUM, WATER

C0 = WATER.sound_speed
F_SINGLE = 2.5e6
F_BAND = (0.5e6, 2.5e6)
N_BAND = 120
RING_DIAMETER = 0.12          # m, for the time-of-flight Fresnel limit
L_DOM, N_GRID = 0.10, 1024    # FFT grid for perturbation spectra
R_OUT, R_IN = 30e-3, 20e-3    # synthetic LV-scale annulus (not patient data)
BULGE_T, BULGE_HALF_DEG = 0.5e-3, 10.0
PSI_DEG = np.arange(0, 360, 45)
RESULTS = Path(__file__).resolve().parent.parent / "results"


def wrap(a):
    return (a + np.pi) % (2 * np.pi) - np.pi


def in_arc(angle, center_deg, half_deg):
    return np.abs(wrap(angle - np.deg2rad(center_deg))) <= np.deg2rad(half_deg) + 1e-12


def fresnel_theta(f):
    return np.sqrt((C0 / f) / RING_DIAMETER)


class SingleSource:
    """A single incident direction is measure-zero in alpha, so the inverse map
    never lands on it exactly; these designs are rasterized forward instead
    (first version did the inverse test and wrongly reported 0% -- LOG run -03)."""

    def __init__(self, alpha_deg, receivers):
        self.alpha = np.deg2rad(alpha_deg)
        self.receivers = receivers  # beta -> bool


# Each design: allowed(alpha, beta, f) -> bool, or a SingleSource
DESIGNS = {
    "1 src + 180° far arc\n(pasted design)":
        SingleSource(0, lambda b: in_arc(b, 0, 90)),
    "1 src + far arc\n+ src-side receiver":
        SingleSource(0, lambda b: in_arc(b, 0, 90) | in_arc(b, 180, 5)),
    "anterior 60° arc\npulse-echo":
        lambda a, b, f: in_arc(a, 90 + 180, 30) & in_arc(b, 90, 30),
    "full ring,\ntime-of-flight only":
        lambda a, b, f: np.abs(wrap(b - a)) <= fresnel_theta(f),
    "full ring,\ntransmission (±90°)":
        lambda a, b, f: np.abs(wrap(b - a)) <= np.pi / 2,
    "full ring,\nall pairs (T+R)":
        lambda a, b, f: np.ones_like(a, dtype=bool),
}


def k_grid():
    k1d = 2 * np.pi * np.fft.fftshift(np.fft.fftfreq(N_GRID, d=L_DOM / N_GRID))
    KX, KY = np.meshgrid(k1d, k1d, indexing="xy")
    return KX, KY


def solutions(KX, KY, k):
    Kmag = np.hypot(KX, KY)
    ok = Kmag <= 2 * k
    psi = np.arctan2(KY, KX)
    t = 2 * np.arcsin(np.clip(Kmag / (2 * k), 0, 1))
    a1, b1 = psi - np.pi / 2 - t / 2, psi - np.pi / 2 + t / 2
    a2, b2 = psi + np.pi / 2 + t / 2, psi + np.pi / 2 - t / 2
    return ok, (a1, b1), (a2, b2)


def coverage_forward(design, KX, KY, freqs):
    dk = KX[0, 1] - KX[0, 0]
    k0 = KX[0, 0]
    n = KX.shape[0]
    mask = np.zeros(KX.shape, dtype=bool)
    ks = 2 * np.pi * np.asarray(freqs) / C0
    if ks.size > 1:  # densify so consecutive K samples are < dk apart
        ks = np.linspace(ks.min(), ks.max(), int((ks.max() - ks.min()) / (dk / 2)) + 2)
    for k in ks:
        betas = np.linspace(-np.pi, np.pi, int(2 * np.pi * k / (dk / 2)) + 1)
        betas = betas[design.receivers(betas)]
        Kx = k * (np.cos(betas) - np.cos(design.alpha))
        Ky = k * (np.sin(betas) - np.sin(design.alpha))
        ix = np.rint((Kx - k0) / dk).astype(int)
        iy = np.rint((Ky - k0) / dk).astype(int)
        ok = (ix >= 0) & (ix < n) & (iy >= 0) & (iy < n)
        mask[iy[ok], ix[ok]] = True
    return mask


def hermitian(mask):
    """A lossless (real) object has O(-K) = conj O(K), so measuring K also
    gives -K. The FFT grid is even-sized: -K of index i is index (n - i) % n
    in unshifted order, i.e. a flip plus a one-cell roll in shifted order.
    Attenuation makes O complex and breaks this -- not modeled here."""
    return mask | np.roll(mask[::-1, ::-1], 1, axis=(0, 1))


def coverage(design, KX, KY, freqs):
    if isinstance(design, SingleSource):
        return hermitian(coverage_forward(design, KX, KY, freqs))
    mask = np.zeros(KX.shape, dtype=bool)
    for f in freqs:
        k = 2 * np.pi * f / C0
        ok, (a1, b1), (a2, b2) = solutions(KX, KY, k)
        mask |= ok & (design(a1, b1, f) | design(a2, b2, f))
    return hermitian(mask)


def check_inverse_map(rng):
    k = 2 * np.pi * F_SINGLE / C0
    K = rng.uniform(-2 * k, 2 * k, size=(2, 2000)) / np.sqrt(2)
    _, (a1, b1), (a2, b2) = solutions(K[0], K[1], k)
    err = 0.0
    for a, b in [(a1, b1), (a2, b2)]:
        Kr = k * np.array([np.cos(b) - np.cos(a), np.sin(b) - np.sin(a)])
        err = max(err, np.max(np.abs(Kr - K)) / (2 * k))
    return err


def bulge_spectrum(r_wall, psi_deg, supersample=4):
    """|FT|^2 of a thin outward bulge on a circular wall, normal angle psi."""
    n = N_GRID * supersample
    x = (np.arange(n) - n / 2 + 0.5) * (L_DOM / n)
    X, Y = np.meshgrid(x, x, indexing="xy")
    R, TH = np.hypot(X, Y), np.arctan2(Y, X)
    m = (R >= r_wall) & (R < r_wall + BULGE_T) & in_arc(TH, psi_deg, BULGE_HALF_DEG)
    img = m.reshape(N_GRID, supersample, N_GRID, supersample).mean(axis=(1, 3))
    F = np.fft.fftshift(np.fft.fft2(img)) * (L_DOM / N_GRID) ** 2
    return np.abs(F) ** 2


def contrasts(bg_rho, old, new):
    """(delta gamma, delta beta) of replacing tissue `old` by `new`, water ref."""
    kap = lambda t: 1.0 / (t.density * t.sound_speed ** 2)
    kap0 = kap(WATER)
    return (kap(new) - kap(old)) / kap0, bg_rho / new.density - bg_rho / old.density


def separation_condition(Kmag, kmin, kmax, n=N_BAND):
    """cond of [1, -cos theta_k] over band samples reaching |K| (3.2b)."""
    ks = np.linspace(kmin, kmax, n)
    out = np.full(Kmag.shape, np.inf)
    for i, K in enumerate(Kmag):
        kk = ks[ks >= K / 2]
        if kk.size < 2:
            continue
        w = 1 - K ** 2 / (2 * kk ** 2)
        s = np.linalg.svd(np.column_stack([np.ones_like(w), -w]), compute_uv=False)
        out[i] = s[0] / s[-1] if s[-1] > 0 else np.inf
    return out


def main():
    rng = set_all_seeds(SEED)
    (RESULTS / "figures").mkdir(parents=True, exist_ok=True)

    err = check_inverse_map(rng)
    print(f"(a) inverse map K -> (alpha, beta) -> K, max rel err: {err:.2e}")

    KX, KY = k_grid()
    Kmag = np.hypot(KX, KY)
    kmax = 2 * np.pi * F_BAND[1] / C0
    band = Kmag <= 2 * kmax
    freqs_b = np.linspace(*F_BAND, N_BAND)

    masks = {}
    for name, d in DESIGNS.items():
        masks[(name, "single")] = coverage(d, KX, KY, [F_SINGLE])
        masks[(name, "band")] = coverage(d, KX, KY, freqs_b)
    full = masks[(list(DESIGNS)[-1], "single")]
    print(f"(b) full ring all pairs @2.5MHz covers {full[band].mean()*100:.2f}% of |K|<=2k disc")
    trans = masks[(list(DESIGNS)[4], "single")][band].mean()
    print(f"(c) full ring transmission @2.5MHz: {trans*100:.2f}% (analytic: disc |K|<=sqrt2 k = 50%)")

    print("\nk-space coverage of the |K| <= 2k_max disc:")
    for name in DESIGNS:
        s, b = masks[(name, "single")][band].mean(), masks[(name, "band")][band].mean()
        print(f"  {name.replace(chr(10), ' '):42s} single {s*100:6.2f}%   band {b*100:6.2f}%")

    # 3.2: information content of local wall bulges, per design and normal angle
    print("\nFraction of bulge spectral energy (|K|<=2k_max) inside coverage, broadband:")
    frac = {}
    for wall, r in [("outer", R_OUT), ("inner", R_IN)]:
        for psi in PSI_DEG:
            P = bulge_spectrum(r, psi)
            tot = P[band].sum()
            for name in DESIGNS:
                frac[(wall, psi, name)] = P[masks[(name, "band")] & band].sum() / tot
    for name in DESIGNS:
        lab = name.replace("\n", " ")
        o = [frac[("outer", p, name)] for p in PSI_DEG]
        i = [frac[("inner", p, name)] for p in PSI_DEG]
        print(f"  {lab:42s} outer min/mean {min(o):.3f}/{np.mean(o):.3f}  "
              f"inner min/mean {min(i):.3f}/{np.mean(i):.3f}")

    # 3.2: contrast -- forward (theta=0) vs backscatter (theta=pi) weight
    dg_o, db_o = contrasts(WATER.density, WATER, MYOCARDIUM)
    dg_i, db_i = contrasts(WATER.density, MYOCARDIUM, BLOOD)
    print("\nScattering weight gamma - cos(theta) beta:")
    for lab, dg, db in [("outer (water->myo)", dg_o, db_o), ("inner (myo->blood)", dg_i, db_i)]:
        print(f"  {lab:20s} forward {abs(dg - db):.4f}   backscatter {abs(dg + db):.4f}")
    print(f"  inner/outer amplitude ratio: forward {abs(dg_i-db_i)/abs(dg_o-db_o):.3f}, "
          f"backscatter {abs(dg_i+db_i)/abs(dg_o+db_o):.3f}")

    # 3.2b density vs compressibility separability (broadband, all pairs)
    kmin = 2 * np.pi * F_BAND[0] / C0
    Ks = np.linspace(50, 2 * kmax * 0.999, 400)
    cond_b = separation_condition(Ks, kmin, kmax)
    for target in [0.05, 0.25, 0.5, 1.0, 1.5]:
        K = target * 2 * kmax
        c = separation_condition(np.array([K]), kmin, kmax)[0]
        print(f"  3.2b |K|={target:.2f}x2k_max (feature ~{2*np.pi/K*1e3:.2f} mm): cond={c:.1f}"
              if np.isfinite(c) else f"  3.2b |K|={target:.2f}x2k_max: not separable")

    # --- figures ---
    ext = np.array([KX.min(), KX.max(), KY.min(), KY.max()]) / kmax
    fig, ax = plt.subplots(2, len(DESIGNS), figsize=(3.1 * len(DESIGNS), 6.6))
    for j, name in enumerate(DESIGNS):
        for i, mode in enumerate(["single", "band"]):
            a = ax[i, j]
            a.imshow(masks[(name, mode)], origin="lower", extent=ext, cmap="Blues", vmin=0, vmax=1.4)
            a.add_patch(plt.Circle((0, 0), 2, fill=False, ls="--", color="gray", lw=0.8))
            a.set_xlim(-2.1, 2.1), a.set_ylim(-2.1, 2.1)
            a.set_xticks([]), a.set_yticks([])
            if i == 0:
                a.set_title(name, fontsize=9)
        ax[0, 0].set_ylabel("2.5 MHz only")
        ax[1, 0].set_ylabel("0.5–2.5 MHz")
    fig.suptitle("Measured region of object spectrum (dashed: |K| = 2k_max, the resolution limit)")
    fig.tight_layout()
    fig.savefig(RESULTS / "figures" / "phase3_kspace_coverage.png", dpi=130)

    fig, ax = plt.subplots(1, 2, figsize=(11, 4), sharey=True)
    names = list(DESIGNS)
    for a, wall in zip(ax, ["outer", "inner"]):
        M = np.array([[frac[(wall, p, n)] for p in PSI_DEG] for n in names])
        im = a.imshow(M, vmin=0, vmax=1, cmap="viridis", aspect="auto")
        a.set_xticks(range(len(PSI_DEG)), [f"{p}°" for p in PSI_DEG])
        a.set_yticks(range(len(names)), [n.replace("\n", " ") for n in names], fontsize=8)
        a.set(title=f"{wall} wall bulge: fraction measured",
              xlabel="bulge normal direction (90° = anterior)")
        for (r, c), v in np.ndenumerate(M):
            a.text(c, r, f"{v:.2f}", ha="center", va="center", fontsize=7,
                   color="white" if v < 0.5 else "black")
    fig.colorbar(im, ax=ax, shrink=0.8)
    fig.savefig(RESULTS / "figures" / "phase3_wall_information.png", dpi=130, bbox_inches="tight")

    fig, a = plt.subplots(figsize=(6, 3.8))
    a.semilogy(Ks / (2 * kmax), cond_b)
    a.set(xlabel="|K| / 2k_max", ylabel="condition number",
          title="3.2b: separating density from compressibility (0.5–2.5 MHz)")
    a.axvline(kmin / kmax, color="gray", ls="--", lw=0.8)
    fig.tight_layout()
    fig.savefig(RESULTS / "figures" / "phase3_density_separation.png", dpi=130)
    print("\nfigures: phase3_kspace_coverage.png, phase3_wall_information.png, "
          "phase3_density_separation.png")
    ok = err < 1e-10 and full[band].mean() > 0.99 and abs(trans - 0.5) < 0.01
    print("CHECKS:", "PASS" if ok else "FAIL")


if __name__ == "__main__":
    main()
