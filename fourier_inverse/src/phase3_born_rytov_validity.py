"""Phase 3.3 -- where do the Born / Rytov linearizations hold at cardiac scale?

Test object: a uniform fluid cylinder (radius a, speed c1, density rho1) in
a uniform background (c0, rho0), plane-wave incidence. 2D, exp(-i w t),
H = H^(1) outgoing. Everything below is closed-form, with no simulation:

  Exact (Bessel series): u_s = sum_n i^n A_n H_n(k0 r) e^{i n th},
    A_n = (q J_n(x0) J_n'(x1) - J_n'(x0) J_n(x1))
          / (H_n'(x0) J_n(x1) - q H_n(x0) J_n'(x1)),
    x0 = k0 a, x1 = k1 a, q = k1 rho0 / (k0 rho1).

  Born far field (variable density, derived in LOG run -02):
    F_B(th) = (i/4) k0^2 [gamma - beta cos th] S(K),
    gamma = kappa1/kappa0 - 1, beta = rho0/rho1 - 1,
    S(K) = 2 pi a J1(K a) / K,  K = 2 k0 sin(th/2).
  Exact far field: F_ex(th) = sqrt(1/(8 pi k0))... common factor dropped on
  both sides: F_ex(th) = sum_n A_n e^{i n th}, F_B uses the same scaling
  (u_s ~ sqrt(2/(pi k0 r)) e^{i(k0 r - pi/4)} F, and F_B carries the i/4).

  Born near field, density-matched case (Bessel series, Lommel integral):
    A_n^B = (i pi/2) (k1^2 - k0^2) (a^2/2) [J_n(x0)^2 - J_{n-1}(x0) J_{n+1}(x0)]
  Rytov near field: u_tot ~ u_inc exp(u_s^B / u_inc).

Checks built in:
  (a) density-matched: Bessel-series Born far field == closed-form F_B
  (b) contrast -> 0: exact -> Born (relative error -> 0 linearly)
Then the validity sweep: relative scattered-field error vs. the accumulated
phase across the diameter, dphi = 2 a w |1/c0 - 1/c1|.
"""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from scipy.special import h1vp, hankel1, j1, jv, jvp

from seed import SEED, set_all_seeds
from tissue import BLOOD, MYOCARDIUM, WATER

THETA = np.linspace(-np.pi, np.pi, 1440, endpoint=False)
RADII_MM = [2.5, 5.0, 10.0, 20.0, 30.0, 45.0]
FREQS_MHZ = [0.8, 2.5]
RESULTS = Path(__file__).resolve().parent.parent / "results"


def n_terms(x):
    return int(x + 4.0 * x ** (1 / 3) + 25)


def exact_coeffs(k0, k1, rho0, rho1, a):
    nmax = n_terms(max(k0, k1) * a)
    n = np.arange(-nmax, nmax + 1)
    x0, x1 = k0 * a, k1 * a
    q = (k1 * rho0) / (k0 * rho1)
    J0, J0p, J1, J1p = jv(n, x0), jvp(n, x0), jv(n, x1), jvp(n, x1)
    H0, H0p = hankel1(n, x0), h1vp(n, x0)
    return n, (q * J0 * J1p - J0p * J1) / (H0p * J1 - q * H0 * J1p)


def born_coeffs_matched(k0, k1, a):
    nmax = n_terms(max(k0, k1) * a)
    n = np.arange(-nmax, nmax + 1)
    x0 = k0 * a
    lommel = 0.5 * a ** 2 * (jv(n, x0) ** 2 - jv(n - 1, x0) * jv(n + 1, x0))
    return n, 0.5j * np.pi * (k1 ** 2 - k0 ** 2) * lommel


def far_field(n, A, theta=THETA):
    return np.exp(1j * np.outer(theta, n)) @ A


def born_far_field(k0, c0, rho0, c1, rho1, a, theta=THETA):
    gamma = (rho0 * c0 ** 2) / (rho1 * c1 ** 2) - 1.0
    beta = rho0 / rho1 - 1.0
    K = 2 * k0 * np.abs(np.sin(theta / 2))
    Ka = np.maximum(K * a, 1e-12)
    S = 2 * np.pi * a ** 2 * j1(Ka) / Ka
    return 0.25j * k0 ** 2 * (gamma - beta * np.cos(theta)) * S


def near_field(n, A, k0, R, theta=THETA):
    return np.exp(1j * np.outer(theta, n)) @ (1j ** n * A * hankel1(n, k0 * R))


def rel_err(approx, exact):
    return np.linalg.norm(approx - exact) / np.linalg.norm(exact)


def run_case(bg, obj, f, a, density_matched=False):
    w = 2 * np.pi * f
    c0, c1 = bg.sound_speed, obj.sound_speed
    rho0 = bg.density
    rho1 = bg.density if density_matched else obj.density
    k0, k1 = w / c0, w / c1
    n, A = exact_coeffs(k0, k1, rho0, rho1, a)
    F_ex = far_field(n, A)
    F_B = born_far_field(k0, c0, rho0, c1, rho1, a)
    out = {"dphi": 2 * a * w * abs(1 / c0 - 1 / c1), "born_ff": rel_err(F_B, F_ex)}
    if density_matched:
        R = 1.5 * a
        u_inc = np.exp(1j * k0 * R * np.cos(THETA))
        nB, AB = born_coeffs_matched(k0, k1, a)
        us_ex = near_field(n, A, k0, R)
        us_B = near_field(nB, AB, k0, R)
        us_R = u_inc * (np.exp(us_B / u_inc) - 1.0)
        out.update(born_nf=rel_err(us_B, us_ex), rytov_nf=rel_err(us_R, us_ex),
                   born_ff_series=rel_err(far_field(nB, AB), F_B))
    return out


def main():
    set_all_seeds(SEED)  # deterministic anyway; logged for convention
    (RESULTS / "figures").mkdir(parents=True, exist_ok=True)

    # (a) Bessel-series Born == closed-form Born (density matched)
    chk = run_case(WATER, MYOCARDIUM, 1e6, 10e-3, density_matched=True)
    print(f"(a) Born series vs closed-form far field: rel err {chk['born_ff_series']:.2e}")

    # (b) contrast -> 0: exact -> Born
    print("(b) exact vs Born as contrast -> 0 (a=10mm, 1MHz, density matched):")
    small = []
    for eps in [1e-2, 1e-3, 1e-4]:
        from tissue import TissueProperties
        obj = TissueProperties("tiny", WATER.sound_speed * (1 + eps), WATER.density, 0.0, "synthetic")
        e = run_case(WATER, obj, 1e6, 10e-3, density_matched=True)["born_ff"]
        small.append(e)
        print(f"    dc/c={eps:.0e}: rel err {e:.2e}")
    # Born error is first order in the contrast: each 10x smaller contrast
    # must give a ~10x smaller error. (An absolute bound was used at first
    # and was arbitrary -- LOG run -02.)
    ratios = [small[0] / small[1], small[1] / small[2]]
    print(f"    convergence ratios per decade: {ratios[0]:.2f}, {ratios[1]:.2f} (expect ~10)")
    ok_b = all(9.0 < r < 11.0 for r in ratios)

    cases = [("myocardium in water", WATER, MYOCARDIUM),
             ("blood in myocardium", MYOCARDIUM, BLOOD)]
    rows = []
    print("\nValidity sweep (rel. scattered-field error; >~0.3 = linearization unusable)")
    print(f"{'case':22s} {'f MHz':>5s} {'a mm':>5s} {'dphi rad':>8s} "
          f"{'Born FF':>8s} {'Born NF*':>8s} {'Rytov NF*':>9s}")
    for label, bg, obj in cases:
        for fm in FREQS_MHZ:
            for amm in RADII_MM:
                full = run_case(bg, obj, fm * 1e6, amm * 1e-3)
                matched = run_case(bg, obj, fm * 1e6, amm * 1e-3, density_matched=True)
                rows.append((label, fm, amm, full["dphi"], full["born_ff"],
                             matched["born_nf"], matched["rytov_nf"]))
                print(f"{label:22s} {fm:5.1f} {amm:5.1f} {full['dphi']:8.2f} "
                      f"{full['born_ff']:8.3f} {matched['born_nf']:8.3f} {matched['rytov_nf']:9.3f}")
    print("* NF columns: density-matched object (sound-speed contrast only), receivers at r=1.5a")

    fig, ax = plt.subplots(1, 2, figsize=(11, 4.2), sharey=True)
    for i, (label, _, _) in enumerate(cases):
        r = [x for x in rows if x[0] == label]
        dphi = np.array([x[3] for x in r])
        o = np.argsort(dphi)
        ax[i].loglog(dphi[o], np.array([x[4] for x in r])[o], "o-", label="Born, far field (true densities)")
        ax[i].loglog(dphi[o], np.array([x[5] for x in r])[o], "s--", label="Born, near field (c only)")
        ax[i].loglog(dphi[o], np.array([x[6] for x in r])[o], "^:", label="Rytov, near field (c only)")
        ax[i].axvline(np.pi, color="gray", lw=0.8)
        ax[i].axhline(0.3, color="gray", lw=0.8, ls="--")
        ax[i].set(title=label, xlabel="accumulated phase across diameter (rad)")
    ax[0].set_ylabel("relative error of scattered field")
    ax[0].legend(fontsize=8)
    fig.tight_layout()
    out = RESULTS / "figures" / "phase3_born_rytov_validity.png"
    fig.savefig(out, dpi=130)
    print(f"\nfigure: {out}")
    print("CHECKS:", "PASS" if chk["born_ff_series"] < 1e-6 and ok_b else "FAIL")


if __name__ == "__main__":
    main()
