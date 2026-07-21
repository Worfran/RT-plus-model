"""Physics helper functions.

No data loading, plotting, optimization, or command-line code belongs here.
"""
from __future__ import annotations

import numpy as np

kB = 8.617333262145e-5  # eV/K


def diffusion_coeff(D0: float, Em: float, T_K: float) -> float:
    return float(D0 * np.exp(-Em / (kB * T_K)))


def coalescence_rate(P0: float, Ea: float, T_K: float) -> float:
    return float(P0 * np.exp(-Ea / (kB * T_K)))


def loop_content_from_radius(R: float, C: float, b: float, Omega0: float) -> float:
    """N = (pi*b/Omega0)*<R^2>*C for content-equivalent radius R."""
    return float((np.pi * b / Omega0) * (R ** 2) * C)


def lognormal_mean_radius_from_rms(R_rms: float, k: float) -> float:
    """Convert sqrt(<R^2>) to a lognormal arithmetic mean.

    Here ``k = std(R)/mean(R)``, so ``<R^2> = mean(R)^2*(1+k^2)``.
    """
    return float(R_rms / np.sqrt(1.0 + float(k) ** 2))


def lognormal_rms_radius_from_mean(R_mean: float, k: float) -> float:
    """Convert a lognormal arithmetic mean to sqrt(<R^2>)."""
    return float(R_mean * np.sqrt(1.0 + float(k) ** 2))


def compute_radius(N: float, C: float, b: float, Omega0: float, eps: float = 1e-300) -> float:
    """Content-equivalent radius sqrt(<R^2>), safe for ODE/debugging."""
    N_eff = max(float(N), eps)
    C_eff = max(float(C), eps)
    return float(np.sqrt(Omega0 * N_eff / (np.pi * b * C_eff)))


def logterminv(R: float, r0: float) -> float:
    """Safe 1/log(8R/r0)."""
    R_min = 1.01 * r0 / 8.0
    R_eff = max(float(R), R_min)
    return float(1.0 / np.log(8.0 * R_eff / r0))


def loop_flux(R: float, Di: float, Ci: float, r0: float) -> float:
    """Bawane et al. Eq. S6 interstitial flux to a toroidal loop.

    The source ODEs S3-S5 use this quantity as ``R_L*C_L*j_i^L``.
    ``rhs`` deliberately preserves that published convention in both the
    mobile-interstitial sink and stored-loop-content source.
    """
    Ci_eff = max(float(Ci), 0.0)
    return float(2.0 * np.pi**2 * R * Di * Ci_eff * logterminv(R, r0))
