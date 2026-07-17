"""ODE system for the RT+ model."""
from __future__ import annotations

import numpy as np

from .physics import compute_radius, loop_flux


def rhs(t: float, y: np.ndarray, params: dict) -> np.ndarray:
    """Right-hand side for y = [Ci, Cv, Nf, Np, Cf, Cp].

    The Frank-loop factor 1/3 is included in b, so the content relation is:
        N = (pi*b/Omega0)*R^2*C
    """

    if not np.all(np.isfinite(y)):
        return np.zeros(6)

    Ci, Cv, Nf, Np, Cf, Cp = y
    a = params["a"]
    Dv = params.get("Dv", 0.0)
    Ziv_iK = params.get("Ziv_iK", 0.0)
    Ziv_vK = params.get("Ziv_vK", 0.0)
    Di = params["Di"]
    Rii = params["Rii"]
    Zii = params["Zii"]
    Puf = params["Puf"]
    Pcs = params["Pcs"]
    b = params["b"]
    Omega0 = params["Omega0"]
    r0 = params["r0"]

    G0i = params.get("G0i", 0.0)
    G0v = params.get("G0v", 0.0)

    Ci_eff = max(float(Ci), 0.0)
    Cv_eff = max(float(Cv), 0.0)
    Nf_eff = max(float(Nf), 1e-300)
    Np_eff = max(float(Np), 1e-300)
    Cf_eff = max(float(Cf), 1e-300)
    Cp_eff = max(float(Cp), 1e-300)

    Rf = compute_radius(Nf_eff, Cf_eff, b, Omega0)
    Rp = compute_radius(Np_eff, Cp_eff, b, Omega0)

    jf = loop_flux(Rf, Di, Ci_eff, r0)
    jp = loop_flux(Rp, Di, Ci_eff, r0)

    geometry_factor = np.pi * b / Omega0
    kiv = (Omega0 / a**2) * (Ziv_iK * Di + Ziv_vK * Dv)
    # Lamella thickness: 100 nm = 1e-5 cm.
    # For two absorbing planar surfaces, the surface sink strength is 12/L^2.
    L = 1e-5
    surface_sink_strength = 12.0 / L**2

    df = np.zeros(6)

    df[0] = G0i - Rf * Cf_eff * jf - Rp * Cp_eff * jp - 2.0 * Zii * Di * Ci_eff**2 - kiv * Ci_eff * Cv_eff - surface_sink_strength * Di * Ci_eff
    df[1] = G0v - kiv * Ci_eff * Cv_eff - surface_sink_strength * Dv * Cv_eff

    # Stored interstitial content in faulted/perfect loops.
    # geometry_factor*Rf^2*Cf = Nf by definition.
    df[2] = Rf * Cf_eff * jf - Puf * geometry_factor * Rf**2 * Cf_eff
    df[3] = Rp * Cp_eff * jp + Puf * geometry_factor * Rf**2 * Cf_eff

    # Faulted and perfect loop number densities.
    df[4] = Rii * Di * Ci_eff**2 - Puf * Cf_eff
    df[5] = Puf * Cf_eff - Pcs * Cp_eff**2

    if not np.all(np.isfinite(df)):
        return np.zeros(6)

    return df
