"""Phase 0.2 -- verify the 1D layer-reversal transmission ambiguity.

Claim under test (from the external consultation that motivated this
project, NOT assumed true): for a 1D layered stack embedded in the same
medium on both sides, reversing the layer order leaves the complex
transmission coefficient T(f) unchanged at every frequency -- multiples
and loss included -- while reflection R(f) changes.

Forward model: exact normal-incidence acoustic transfer matrix. The layer
matrix below goes with time convention exp(+i w t), so the decaying
wavenumber is k = w/c - i*alpha (the first version used +i*alpha, which made
the lossy stack AMPLIFY; checks (d)/(e) were added to catch exactly that).
Layer matrix (state [p, u]):
    M_j = [[cos(k d), i Z sin(k d)], [i sin(k d) / Z, cos(k d)]],  Z = rho w / k
Stack embedded in medium Z0:
    T = 2 / (M11 + M12/Z0 + Z0 M21 + M22)
    R = (M11 + M12/Z0 - Z0 M21 - M22) / (same denominator)

Attenuation: alpha(f) = a_1MHz * (f / 1 MHz), i.e. power-law exponent 1.
The exponent is an ASSUMPTION for this check (only the 1 MHz values are
cited); the reversal claim should not depend on it.

Gate 0 checks:
  (a) single layer |T|^2 matches the closed form 1/(cos^2 + (m+1/m)^2/4 sin^2)
  (b) lossless stack conserves energy |R|^2 + |T|^2 = 1
  (c) reversal: max |T_A - T_B| at machine precision; |R_A - R_B| is not
  (d) lossy stacks are passive: |R|^2 + |T|^2 <= 1
  (e) a lossy slab embedded in its own medium gives |T| = exp(-alpha d) exactly
"""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from seed import SEED, set_all_seeds
from tissue import ALL_TISSUES, BLOOD, CHEST_WALL_PROXY, MYOCARDIUM, WATER

DB_PER_NP = 20.0 / np.log(10.0)  # 8.686
FREQS = np.linspace(0.05e6, 5.0e6, 4000)
N_RANDOM_STACKS = 1000
RESULTS = Path(__file__).resolve().parent.parent / "results"


def alpha_np_per_m(tissue, f, lossless=False):
    if lossless:
        return np.zeros_like(f)
    return tissue.attenuation * (f / 1e6) / DB_PER_NP * 100.0


def stack_matrix(layers, f, lossless=False):
    """layers: list of (TissueProperties, thickness_m). Returns M[..., 2, 2]."""
    w = 2 * np.pi * f
    M = np.broadcast_to(np.eye(2, dtype=complex), f.shape + (2, 2)).copy()
    for tissue, d in layers:
        k = w / tissue.sound_speed - 1j * alpha_np_per_m(tissue, f, lossless)
        Z = tissue.density * w / k
        phi = k * d
        Mj = np.empty(f.shape + (2, 2), dtype=complex)
        Mj[..., 0, 0] = np.cos(phi)
        Mj[..., 0, 1] = 1j * Z * np.sin(phi)
        Mj[..., 1, 0] = 1j * np.sin(phi) / Z
        Mj[..., 1, 1] = np.cos(phi)
        M = M @ Mj
    return M


def transmission_reflection(layers, f, z0=WATER.impedance, lossless=False):
    M = stack_matrix(layers, f, lossless)
    a = M[..., 0, 0] + M[..., 0, 1] / z0
    b = z0 * M[..., 1, 0] + M[..., 1, 1]
    den = a + b
    T = 2.0 / den
    R = (M[..., 0, 0] + M[..., 0, 1] / z0 - z0 * M[..., 1, 0] - M[..., 1, 1]) / den
    return T, R


def check_single_layer():
    d = 0.01
    T, _ = transmission_reflection([(CHEST_WALL_PROXY, d)], FREQS, lossless=True)
    m = CHEST_WALL_PROXY.impedance / WATER.impedance
    phi = 2 * np.pi * FREQS / CHEST_WALL_PROXY.sound_speed * d
    closed = 1.0 / (np.cos(phi) ** 2 + 0.25 * (m + 1 / m) ** 2 * np.sin(phi) ** 2)
    return np.max(np.abs(np.abs(T) ** 2 - closed))


def check_self_embedded_attenuation():
    d = 0.02
    w = 2 * np.pi * FREQS
    a = alpha_np_per_m(MYOCARDIUM, FREQS)
    z_own = MYOCARDIUM.density * w / (w / MYOCARDIUM.sound_speed - 1j * a)
    T, _ = transmission_reflection([(MYOCARDIUM, d)], FREQS, z0=z_own)
    return np.max(np.abs(np.abs(T) - np.exp(-a * d)))


def random_stack(rng):
    """Random stack with adjacent identical tissues merged (they are one layer)."""
    n = rng.integers(2, 9)
    tissues = [ALL_TISSUES[i] for i in rng.integers(1, len(ALL_TISSUES), size=n)]
    thick = rng.uniform(1e-3, 30e-3, size=n)
    merged = []
    for t, d in zip(tissues, thick):
        if merged and merged[-1][0] is t:
            merged[-1] = (t, merged[-1][1] + d)
        else:
            merged.append((t, d))
    return merged


def is_palindrome(stack, tol=1e-12):
    """A stack equal to its own reverse is trivially reversal-invariant in R too."""
    return all(a[0] is b[0] and abs(a[1] - b[1]) < tol
               for a, b in zip(stack, stack[::-1]))


def main():
    rng = set_all_seeds(SEED)
    RESULTS.mkdir(exist_ok=True)
    (RESULTS / "figures").mkdir(exist_ok=True)

    err_single = check_single_layer()
    print(f"(a) single-layer |T|^2 vs closed form, max abs err: {err_single:.2e}")

    err_atten = check_self_embedded_attenuation()
    print(f"(e) self-embedded slab |T| vs exp(-alpha d), max abs err: {err_atten:.2e}")

    energy_err, t_diff, r_diff_rel, n_palindromes = 0.0, [], [], 0
    max_lossy_energy = 0.0
    for _ in range(N_RANDOM_STACKS):
        stack = random_stack(rng)
        if is_palindrome(stack):
            n_palindromes += 1
            continue
        T, R = transmission_reflection(stack, FREQS, lossless=True)
        energy_err = max(energy_err, np.max(np.abs(np.abs(T) ** 2 + np.abs(R) ** 2 - 1)))

        TA, RA = transmission_reflection(stack, FREQS)
        TB, RB = transmission_reflection(stack[::-1], FREQS)
        max_lossy_energy = max(max_lossy_energy, np.max(np.abs(TA) ** 2 + np.abs(RA) ** 2),
                               np.max(np.abs(TB) ** 2 + np.abs(RB) ** 2))
        t_diff.append(np.max(np.abs(TA - TB)) / np.max(np.abs(TA)))
        r_diff_rel.append(np.max(np.abs(RA - RB)) / np.max(np.abs(RA)))
    t_diff, r_diff_rel = np.array(t_diff), np.array(r_diff_rel)

    print(f"(b) lossless energy conservation, max |R|^2+|T|^2-1: {energy_err:.2e}")
    print(f"(d) lossy passivity, max |R|^2+|T|^2: {max_lossy_energy:.6f} (must be <= 1)")
    print(f"(c) reversal over {len(t_diff)} random lossy stacks "
          f"({n_palindromes} palindromic stacks excluded after merging):")
    print(f"    max rel |T_A - T_B|: {t_diff.max():.2e}")
    print(f"    rel |R_A - R_B|: min {r_diff_rel.min():.3f}, "
          f"median {np.median(r_diff_rel):.3f}, max {r_diff_rel.max():.3f}")

    # Illustrative cardiac-like stack: wall -> myocardium -> blood -> myocardium
    example = [(CHEST_WALL_PROXY, 12e-3), (MYOCARDIUM, 10e-3),
               (BLOOD, 40e-3), (MYOCARDIUM, 6e-3)]
    TA, RA = transmission_reflection(example, FREQS)
    TB, RB = transmission_reflection(example[::-1], FREQS)
    ex_t = np.max(np.abs(TA - TB))
    ex_r = np.max(np.abs(RA - RB)) / np.max(np.abs(RA))
    print(f"example cardiac-like stack: max |T_A-T_B| = {ex_t:.2e}, "
          f"rel |R_A-R_B| = {ex_r:.3f}")

    fig, ax = plt.subplots(1, 2, figsize=(11, 4))
    fm = FREQS / 1e6
    ax[0].plot(fm, np.abs(TA), label="A: wall|myo|blood|myo")
    ax[0].plot(fm, np.abs(TB), "--", label="B: reversed")
    ax[0].set(title="|T(f)| -- identical", xlabel="f (MHz)")
    ax[1].plot(fm, np.abs(RA), label="A")
    ax[1].plot(fm, np.abs(RB), "--", label="B reversed")
    ax[1].set(title="|R(f)| -- differs", xlabel="f (MHz)")
    for a in ax:
        a.legend()
    fig.tight_layout()
    out = RESULTS / "figures" / "phase0_layer_reversal.png"
    fig.savefig(out, dpi=130)
    print(f"figure: {out}")

    passed = err_single < 1e-10 and energy_err < 1e-10 and t_diff.max() < 1e-10 \
        and r_diff_rel.min() > 1e-6 and max_lossy_energy <= 1 + 1e-12 \
        and err_atten < 1e-10
    print("GATE 0 (phase 0.2):", "PASS" if passed else "FAIL")


if __name__ == "__main__":
    main()
