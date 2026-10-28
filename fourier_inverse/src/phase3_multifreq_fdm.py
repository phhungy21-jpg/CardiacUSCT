"""Multi-frequency design A2 -- can the frequencies be isolated cleanly, and how
short can the patient-facing acquisition be?

Exact forward model (no simulation): 2D line source on a ring (R = 75 mm, 32
sources, 64 receivers half-step offset), centred attenuating fluid cylinder
(radius 45 mm, cited myocardium c/rho, 0.52 dB/cm/MHz with alpha ~ f ASSUMED,
no dispersion) in water. Transfer function H(f) = incident + Bessel-series
scattered field (Graf addition theorem, coefficients from run -02's
`exact_coeffs`), sampled every 0.5 kHz to 3 MHz -> impulse responses (2 ms).

Acquisition designs (same peak drive amplitude 1 per element, same white
receiver noise; noise level set so the CURRENT design gives 30 dB at 0.8 MHz,
the SNR used in runs -25..-36):
  PULSE   sequential broadband pulses (derivative of Gaussian, sigma 0.25 us
          = current; 0.10 us = wider band for the 2.5 MHz 'store' tones),
          one source per firing interval, DFT at exact f over that interval.
  TONE    sequential multi-tone bursts (K = 1), cosine ramps, settle, then a
          rectangular window of T_w holding an integer number of cycles of
          every tone (tones on the 1/T_w bin grid -> orthogonal).
  FDM     K sources fire at once; source j of a group uses f_k + j/T_w
          (interleaved bins), groups fire back to back. Demultiplexed by the
          same DFT. Spread (K-1)/T_w is reported vs f_k (the recon must use
          each source's own frequency; the stage becomes a narrow band).
Errors are measured against the exact H, per frequency:
  crosstalk/leakage (noise-free) and noise (noise-only through the same
  extraction), as dB relative to ||H||.
2D caveat: the 2D Green's function has a slowly decaying wake (absent in 3D),
which makes the transient/tail figures pessimistic. No tank-wall reverberation
(the real tank's echoes must be added once its geometry is known).
Synthetic only (no patient data).
"""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from scipy.special import h1vp, hankel1, jv, jvp

from phase3_born_rytov_validity import n_terms
from seed import SEED, set_all_seeds
from tissue import MYOCARDIUM, WATER

FIG = Path(__file__).resolve().parent.parent / "results" / "figures" / "phase3_multifreq_fdm_separability.png"
FS = 10e6                       # sampling rate
DF_H = 500.0                    # transfer-function spacing -> 2 ms impulse responses
F_MAX = 3.0e6
R_RING, A_CYL = 75e-3, 45e-3
N_SRC, N_REC = 32, 64
C0, RHO0 = WATER.sound_speed, WATER.density
C1, RHO1 = MYOCARDIUM.sound_speed, MYOCARDIUM.density
ALPHA1 = MYOCARDIUM.attenuation * 100 / 8.686 / 1e6       # Np/m per Hz (alpha ~ f, ASSUMED power law 1)
RECON = [0.15e6, 0.3e6, 0.5e6, 0.8e6]
STORE = [1.2e6, 1.7e6, 2.5e6]
SNR_REF_DB = 30.0
DELAY = 50e-6                   # fixed latency so the band-limited responses are causal inside the 2 ms period
ND = int(DELAY * FS)


def coeffs_pos(k0, k1, a):
    """n >= 0 half of run -02's exact coefficients (A_-n = A_n), complex k1 allowed."""
    n = np.arange(0, n_terms(max(k0, abs(k1)) * a) + 1)
    x0, x1 = k0 * a, k1 * a
    q = (k1 * RHO0) / (k0 * RHO1)
    J0, J0p, J1, J1p = jv(n, x0), jvp(n, x0), jv(n, x1), jvp(n, x1)
    H0, H0p = hankel1(n, x0), h1vp(n, x0)
    return n, (q * J0 * J1p - J0p * J1) / (H0p * J1 - q * H0 * J1p)


def transfer_functions():
    """H_np(dtheta, f) on the rfft grid (numpy e^{-i 2 pi f t} convention = conj of e^{-i w t} physics)."""
    nh = int(FS / DF_H)
    f = np.fft.rfftfreq(nh, 1 / FS)
    dth = 2 * np.pi * (np.arange(N_REC) + 0.5) / N_REC              # receiver angle minus source angle
    d = 2 * R_RING * np.sin(dth / 2)
    H = np.zeros((N_REC, f.size), complex)
    band = (f > 15e3) & (f < F_MAX)
    for i in np.nonzero(band)[0]:
        k0 = 2 * np.pi * f[i] / C0
        k1 = 2 * np.pi * f[i] / C1 + 1j * ALPHA1 * f[i]
        n, A = coeffs_pos(k0, k1, A_CYL)
        with np.errstate(all="ignore"):
            t = np.nan_to_num(A * hankel1(n, k0 * R_RING) ** 2)
        w = np.where(n == 0, 1.0, 2.0)
        scat = np.cos(np.outer(dth, n)) @ (w * t)
        H[:, i] = np.conj(0.25j * (hankel1(0, k0 * d) + scat))
    taper = np.clip((f - 15e3) / 45e3, 0, 1) * np.clip((F_MAX - f) / 0.4e6, 0, 1)
    H *= (np.sin(np.pi / 2 * taper) ** 2)[None, :] * np.exp(-2j * np.pi * f * DELAY)[None, :]
    return f, H


def exact_at(f_grid, H, f):
    i = np.round(f / DF_H).astype(int)
    assert np.allclose(f_grid[i], f, atol=1e-6 * DF_H * 0 + 1.0), "tone not on the transfer-function grid"
    return H[:, i]


def receive(h_t, xs, t_len):
    """y_r(t) = sum_s h(theta_r - theta_s) * x_s; all on a common FFT grid."""
    n = int(2 ** np.ceil(np.log2(t_len + h_t.shape[1])))
    Hl = np.fft.rfft(h_t, n)                                        # (N_REC, n/2+1)
    Xs = np.fft.rfft(xs, n)                                         # (N_SRC, n/2+1)
    Y = np.zeros((N_REC, Hl.shape[1]), complex)
    for s in range(N_SRC):
        idx = (np.arange(N_REC) - 2 * s) % N_REC
        Y += Hl[idx] * Xs[s]
    return np.fft.irfft(Y, n)[:, :t_len] / FS                       # / FS: discrete convolution -> integral


def dft(y, t0, n, f):
    t = (t0 + np.arange(n)) / FS
    return y[:, t0:t0 + n] @ np.exp(-2j * np.pi * np.outer(t, f)) / FS


def dog(t, sigma, t0):
    return -(t - t0) / sigma ** 2 * np.exp(-(t - t0) ** 2 / (2 * sigma ** 2))


def pulse_design(sigma, t_fire, freqs):
    """Sequential pulses: returns per-source drive, analysis windows, and the drive spectrum at freqs."""
    nf = int(round(t_fire * FS)); t_len = N_SRC * nf
    xs = np.zeros((N_SRC, t_len)); tt = np.arange(nf) / FS
    p = dog(tt, sigma, 1.5e-6); p /= np.abs(p).max()
    for s in range(N_SRC):
        xs[s, s * nf:(s + 1) * nf] = p
    Xf = (p @ np.exp(-2j * np.pi * np.outer(tt, freqs))) / FS
    wins = [(s * nf, nf, Xf * np.exp(-2j * np.pi * freqs * s * nf / FS)) for s in range(N_SRC)]
    return xs, t_len, [[w] for w in wins], [[s] for s in range(N_SRC)], freqs, N_SRC * t_fire


def tone_design(K, t_w, t_settle, t_ramp, freqs, rng):
    """K simultaneous sources per group, interleaved bins. Returns windows and per-source tone frequencies."""
    G = N_SRC // K
    nw, ns, nr = int(round(t_w * FS)), int(round(t_settle * FS)), int(round(t_ramp * FS))
    nb = 2 * nr + ns + nw; t_len = G * nb
    base = np.round(np.asarray(freqs) * t_w) / t_w                  # nominal tones on the 1/t_w grid
    env = np.ones(nb); r = np.sin(np.pi / 2 * np.arange(nr) / nr) ** 2
    env[:nr] = r; env[-nr:] = r[::-1]
    tt = np.arange(nb) / FS
    xs = np.zeros((N_SRC, t_len)); src_f, amps = {}, {}
    for s in range(N_SRC):
        g, j = divmod(s, K)
        fs_ = base + j / t_w
        best = None
        for _ in range(200):                                        # low-crest-factor random phases
            ph = rng.uniform(0, 2 * np.pi, fs_.size)
            w = np.cos(2 * np.pi * np.outer(tt + g * nb / FS, fs_) + ph).sum(1)
            pk = np.abs(w * env).max()
            if best is None or pk < best[0]:
                best = (pk, ph)
        pk, ph = best
        xs[s, g * nb:(g + 1) * nb] = env * np.cos(2 * np.pi * np.outer(tt + g * nb / FS, fs_) + ph).sum(1) / pk
        src_f[s], amps[s] = fs_, np.exp(1j * ph) / pk
    groups = [list(range(g * K, (g + 1) * K)) for g in range(G)]
    starts = [g * nb + nr + ns for g in range(G)]
    return xs, t_len, starts, nw, groups, src_f, amps, G * nb / FS, (K - 1) / t_w


def evaluate_pulse(fH, H, h_t, sigma, t_fire, freqs, noise_sd, rng):
    xs, t_len, wins, groups, f, t_tank = pulse_design(sigma, t_fire, np.asarray(freqs))
    y = receive(h_t, xs, t_len + ND)
    nz = noise_sd * rng.standard_normal(y.shape)
    err, noi, ref = np.zeros(len(f)), np.zeros(len(f)), np.zeros(len(f))
    med = []
    for s, ((t0, n, Xf),) in enumerate(wins):
        t0 += ND
        idx = (np.arange(N_REC) - 2 * s) % N_REC
        Htrue = exact_at(fH, H, f)[idx]
        est = dft(y, t0, n, f) / Xf
        med.append(np.abs(est - Htrue) / np.abs(Htrue))
        err += np.sum(np.abs(est - Htrue) ** 2, 0)
        noi += np.sum(np.abs(dft(nz, t0, n, f) / Xf) ** 2, 0)
        ref += np.sum(np.abs(Htrue) ** 2, 0)
    return f, 10 * np.log10(err / ref), 10 * np.log10(noi / ref), t_tank, 20 * np.log10(np.median(np.concatenate(med), 0))


def evaluate_tone(fH, H, h_t, K, t_w, t_settle, t_ramp, freqs, noise_sd, rng):
    xs, t_len, starts, nw, groups, src_f, amps, t_tank, spread = tone_design(K, t_w, t_settle, t_ramp, freqs, rng)
    y = receive(h_t, xs, t_len + ND)
    nz = noise_sd * rng.standard_normal(y.shape)
    nf = len(freqs)
    err, noi, ref = np.zeros(nf), np.zeros(nf), np.zeros(nf)
    med = []
    starts = [st + ND for st in starts]
    for g, members in enumerate(groups):
        for s in members:
            f = src_f[s]
            idx = (np.arange(N_REC) - 2 * s) % N_REC
            Htrue = exact_at(fH, H, f)[idx]
            scale = 2 / (amps[s] * nw / FS)
            est = dft(y, starts[g], nw, f) * scale
            med.append(np.abs(est - Htrue) / np.abs(Htrue))
            err += np.sum(np.abs(est - Htrue) ** 2, 0)
            noi += np.sum(np.abs(dft(nz, starts[g], nw, f) * scale) ** 2, 0)
            ref += np.sum(np.abs(Htrue) ** 2, 0)
    return np.asarray(freqs), 10 * np.log10(err / ref), 10 * np.log10(noi / ref), t_tank, spread, 20 * np.log10(np.median(np.concatenate(med), 0))


def main():
    rng = set_all_seeds(SEED)
    fH, H = transfer_functions()
    h_t = np.fft.irfft(H, int(FS / DF_H)) * FS                      # impulse responses, 2 ms
    # sanity: causality -- energy before the direct arrival (shortest path) is ~0
    t = np.arange(h_t.shape[1]) / FS
    d = 2 * R_RING * np.sin(np.pi * (np.arange(N_REC) + 0.5) / N_REC)
    pre = np.array([np.sum(h_t[r, t < DELAY + 0.9 * d[r] / C0 - 10e-6] ** 2) / np.sum(h_t[r] ** 2) for r in range(N_REC)])
    tail = np.sum(h_t[:, t > DELAY + 1.0e-3] ** 2) / np.sum(h_t ** 2)
    print(f"causality: max pre-arrival energy fraction {pre.max():.1e}; energy after 1 ms {tail:.1e}")

    freqs_all = RECON + STORE
    # noise calibrated on the CURRENT design: 30 dB at 0.8 MHz
    _, _, n_unit, _, _ = evaluate_pulse(fH, H, h_t, 0.25e-6, 0.3e-3, [0.8e6], 1.0, np.random.default_rng(1))
    noise_sd = 10 ** ((-SNR_REF_DB - n_unit[0]) / 20)
    print(f"noise sd (drive-peak units) for 30 dB @0.8 MHz with the current pulse: {noise_sd:.3e}\n")

    rows = []
    def show(name, f, e, n, t_tank, spread=None, md=None):
        tot = 10 * np.log10(10 ** (e / 10) + 10 ** (n / 10))
        rows.append((name, f, e, n, tot, t_tank, spread))
        sp = "" if spread is None else f" | spread {spread/1e3:.0f} kHz ({100*spread/f[0]:.0f}% of {f[0]/1e6:.2f} MHz)"
        print(f"{name:40s} tank {t_tank*1e3:6.2f} ms{sp}")
        print("   f (MHz)    " + " ".join(f"{x/1e6:6.2f}" for x in f))
        print("   leak dB    " + " ".join(f"{x:6.1f}" for x in e))
        print("   leak med dB" + " ".join(f"{x:6.1f}" for x in md))
        print("   noise dB   " + " ".join(f"{x:6.1f}" for x in n))
        print("   SNR dB     " + " ".join(f"{-x:6.1f}" for x in tot))

    for sig, tf in [(0.25e-6, 0.14e-3), (0.25e-6, 0.3e-3), (0.10e-6, 0.3e-3)]:
        f, e, n, tt_, md = evaluate_pulse(fH, H, h_t, sig, tf, freqs_all, noise_sd, rng)
        show(f"PULSE sigma {sig*1e6:.2f}us, {tf*1e3:.2f} ms/firing", f, e, n, tt_, None, md)
    for K, tw, ts in [(1, 0.1e-3, 0.15e-3), (8, 0.25e-3, 0.15e-3), (16, 0.5e-3, 0.15e-3),
                      (32, 1.0e-3, 0.15e-3), (32, 1.0e-3, 0.05e-3), (32, 0.5e-3, 0.15e-3)]:
        for fl, tag in [(RECON, "recon 4"), (freqs_all, "recon+store 7")]:
            f, e, n, tt_, sp, md = evaluate_tone(fH, H, h_t, K, tw, ts, 0.05e-3, fl, noise_sd, rng)
            show(f"{'TONE' if K == 1 else 'FDM'} K={K} Tw={tw*1e3:.2f} Ts={ts*1e3:.2f} ms [{tag}]", f, e, n, tt_, sp, md)

    fig, ax = plt.subplots(1, 2, figsize=(14, 5))
    for name, f, e, n, tot, t_tank, sp in rows:
        if "recon 4" in name: continue
        ax[0].plot(f / 1e6, -tot, "o-", label=f"{name} ({t_tank*1e3:.1f} ms)")
    ax[0].axhline(30, color="k", ls=":", lw=1)
    ax[0].set_xlabel("frequency (MHz)"); ax[0].set_ylabel("effective SNR (dB): leakage + noise")
    ax[0].set_title("Per-frequency SNR at equal drive peak and receiver noise"); ax[0].legend(fontsize=6)
    for name, f, e, n, tot, t_tank, sp in rows:
        if "recon 4" in name: continue
        ax[1].scatter(t_tank * 1e3, -tot[f == 0.8e6], s=30)
        ax[1].annotate(name.split(" [")[0], (t_tank * 1e3, -tot[f == 0.8e6][0]), fontsize=6)
    ax[1].set_xscale("log"); ax[1].set_xlabel("patient-facing acquisition per slice (ms)")
    ax[1].set_ylabel("SNR at 0.8 MHz (dB)"); ax[1].set_title("Tank time vs data quality")
    fig.tight_layout(); fig.savefig(FIG, dpi=130)
    print(f"\nfigure: {FIG}")


if __name__ == "__main__":
    main()
