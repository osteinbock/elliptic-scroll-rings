"""Fourier spectra of a filament ring snapshot in the polar angle (used by buckle_save.py).

spectra(lines) bins the radial distance rho and the height z of the filament points in NB polar-
angle bins about the centroid (points farther than 6 from the median radius are ignored) and
returns the median radius R, the complex Fourier coefficients fr[m], fz[m] of rho and z, and the
fraction of occupied bins.
"""
import numpy as np

NB = 1024


def spectra(lines):
    lines = [l for l in lines if len(l) > 20]
    p = np.concatenate(lines)
    c = p.mean(0)
    th = np.arctan2(p[:, 1] - c[1], p[:, 0] - c[0])
    rho = np.hypot(p[:, 0] - c[0], p[:, 1] - c[1])
    R = np.median(rho)
    keep = np.abs(rho - R) < 6
    th, rho, z = th[keep], rho[keep], p[keep, 2]
    b = ((th + np.pi) / (2 * np.pi) * NB).astype(int) % NB
    cnt = np.bincount(b, minlength=NB)
    rb = np.bincount(b, rho, NB)
    zb = np.bincount(b, z, NB)
    ok = cnt > 0
    ang = (np.arange(NB) + 0.5) / NB * 2 * np.pi - np.pi
    rb = np.interp(ang, ang[ok], rb[ok] / cnt[ok], period=2 * np.pi)
    zb = np.interp(ang, ang[ok], zb[ok] / cnt[ok], period=2 * np.pi)
    fr = np.fft.rfft(rb - rb.mean()) * 2 / NB
    fz = np.fft.rfft(zb - zb.mean()) * 2 / NB
    return R, fr, fz, ok.mean()
