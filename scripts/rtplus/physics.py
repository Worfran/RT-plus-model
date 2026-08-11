"""Physics helper functions.

No data loading, plotting, optimization, or command-line code belongs here.
"""
from __future__ import annotations

import numpy as np

kB = 8.617333262145e-5  # eV/K

# Interaction-driven coalescence scenario tau_CI from Table 2 of the loop
# lifetime paper.  beta applies to 1/tau; multiplying by loop density in the
# number balance gives a total density exponent beta + 1 = 8/3.
COALESCENCE_RADIUS_EXPONENT = 1.0
COALESCENCE_LIFETIME_DENSITY_EXPONENT = 5.0 / 3.0


def diffusion_coeff(D0: float, Em: float, T_K: float) -> float:
    return float(D0 * np.exp(-Em / (kB * T_K)))


def coalescence_rate(lambda0: float, Ea: float, T_K: float) -> float:
    """Temperature-dependent lifetime amplitude lambda(T)."""

    return float(lambda0 * np.exp(-Ea / (kB * T_K)))


def coalescence_inverse_lifetime(
    coefficient: float,
    loop_radius: float,
    loop_number_density: float,
    radius_exponent: float = COALESCENCE_RADIUS_EXPONENT,
    density_exponent: float = COALESCENCE_LIFETIME_DENSITY_EXPONENT,
) -> float:
    """Return 1/tau for the interaction-driven coalescence scenario.

    The publication defines

        1/tau = lambda(T) * R**alpha * C**beta,

    with alpha=1 and beta=5/3 for interaction-driven coalescence.  With R in
    cm and C in cm^-3, the default coefficient has units cm^4/s.
    """

    coefficient = float(coefficient)
    loop_radius = float(loop_radius)
    loop_number_density = float(loop_number_density)
    if coefficient < 0.0:
        raise ValueError("Coalescence coefficient must be nonnegative.")
    if loop_radius < 0.0:
        raise ValueError("Loop radius must be nonnegative.")
    if loop_number_density < 0.0:
        raise ValueError("Loop number density must be nonnegative.")
    return float(
        coefficient
        * loop_radius**float(radius_exponent)
        * loop_number_density**float(density_exponent)
    )


def coalescence_number_loss(
    coefficient: float,
    loop_radius: float,
    loop_number_density: float,
    model: str = "interaction_driven",
) -> float:
    """Return the interaction-driven coalescence loss of loop density.

    This is ``C/tau``.  With the fixed publication exponents it is proportional
    to ``R*C**(8/3)`` and has units cm^-3/s.  Coalescence removes loop objects
    but does not remove their stored point-defect inventory.  Consequently,
    this loss belongs only in the loop-number balance; the content-equivalent
    radius then increases automatically as inventory divided by number density
    increases.

    The factor sometimes written as 1/2 in binary-collision equations is
    included in the definition of ``coefficient`` used by this model.
    """

    model = str(model).strip().lower()
    if model == "interaction_driven":
        inverse_lifetime = coalescence_inverse_lifetime(
            coefficient,
            loop_radius,
            loop_number_density,
        )
        return float(inverse_lifetime * float(loop_number_density))
    if model == "legacy_quadratic":
        coefficient = float(coefficient)
        loop_number_density = float(loop_number_density)
        if coefficient < 0.0 or loop_number_density < 0.0:
            raise ValueError(
                "Coalescence coefficient and loop density must be nonnegative."
            )
        return float(coefficient * loop_number_density**2)
    raise ValueError(
        "Coalescence model must be 'interaction_driven' or "
        "'legacy_quadratic'."
    )


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
